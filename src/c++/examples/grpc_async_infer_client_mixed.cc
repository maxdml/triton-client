#include <iostream>
#include <fstream>
#include <string>
#include "grpc_client.h"
#include <random>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <tuple>

namespace tc = triton::client;

#define RUNTIME 5 //  in seconds

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }


/** TIME STUFF */
/***************/
static inline uint64_t rdtscp(uint32_t *auxp) {
    uint32_t a, d, c;
    asm volatile("rdtscp" : "=a" (a), "=d" (d), "=c" (c));
    if (auxp)
        *auxp = c;
    return ((uint64_t)a) | (((uint64_t)d) << 32);
}

static inline void cpu_serialize(void) {
        asm volatile("xorl %%eax, %%eax\n\t"
             "cpuid" : : : "%rax", "%rbx", "%rcx", "%rdx");
}

uint64_t since_epoch(const std::chrono::steady_clock::time_point &time) {
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(time).time_since_epoch().count();
}

uint64_t ns_diff(const std::chrono::steady_clock::time_point &start,
                 const std::chrono::steady_clock::time_point &end) {
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
    if (ns < 0) {
        ns = -1;
    }
    return ns;
}

float cycles_per_ns;
uint32_t cycles_per_us;
// From Adam's base OS
inline int time_calibrate_tsc(void) {
    struct timespec sleeptime;
    sleeptime.tv_sec = 0;
    sleeptime.tv_nsec = 5E8; /* 1/2 second */
    struct timespec t_start, t_end;

    cpu_serialize();
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t_start) == 0) {
        uint64_t ns, end, start;
        double secs;

        start = rdtscp(NULL);
        nanosleep(&sleeptime, NULL);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);
        end = rdtscp(NULL);
        ns = ((t_end.tv_sec - t_start.tv_sec) * 1E9);
        ns += (t_end.tv_nsec - t_start.tv_nsec);

        cycles_per_ns = ((end - start) * 1.0) / ns;
        printf("Time calibration: detected %.03f ticks / ns\n", cycles_per_ns);
        cycles_per_us = cycles_per_ns * 1e3;

        return 0;
    }

    return -1;
}

typedef struct Histogram_t {
    uint64_t min, max, total, count;
    std::map<uint32_t, uint64_t> buckets;
} Histogram_t;

/** WORKLOAD GENERATION STUFF */
/******************************/

class Model {
    public: tc::InferOptions options;
    public: std::vector<tc::InferInput *> inputs;
    public: std::vector<const tc::InferRequestedOutput *> outputs;
    public: const std::string name;
    public: Model(std::string &n) : name(n), options(n) {}

    private: std::vector<int32_t> input_data_;
    private: tc::InferInput *input_;
    private: tc::InferRequestedOutput *output_;

    public: int create_model_io(const char *input_name, const char *output_name,
                                const char *type, std::vector<int64_t> &shape) {

    assert(shape.size() == 4);

    /* Init the underlying memory */
    size_t n_entries = shape[0] * shape[1] * shape[2] * shape[3];
    input_data_.reserve(n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
        input_data_[i] = i;
    }
    FAIL_IF_ERR(tc::InferInput::Create(&input_, input_name, shape, type), "unable to get input");
    FAIL_IF_ERR(
        input_->AppendRaw(reinterpret_cast<uint8_t*>(&input_data_[0]), n_entries * sizeof(int32_t)),
        "unable to set data for INPUT"
    );
    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&output_, output_name), "unable to get output");

    inputs = {input_};
    outputs = {output_};

    return 0;
}

};

class Request {
    public: Model *model;
    public: uint64_t send_time;
    public: uint64_t receive_time;
    public: uint64_t latency;
    public: uint32_t id;
};

class Schedule {
    public: std::chrono::seconds duration;
    public: std::chrono::steady_clock::time_point start_time;
    public: std::chrono::steady_clock::time_point end_time;
    public: std::chrono::steady_clock::time_point last_send_time;
    public: double rate = 0;
    public: bool uniform = false;
    public: std::vector<double> ratios;
    public: std::vector<std::chrono::steady_clock::time_point> send_times;
    public: uint32_t send_index = 0;
    public: uint32_t recv_requests = 0;
    public: uint32_t attempts = 0;
    public: uint32_t ev_count = 0;
    public: uint32_t n_skipped = 0;
    public: uint32_t n_requests = 0;
    public: std::vector<Request *> requests;
    public: std::vector<std::string> models;

    public: Schedule () {}

    // Goodput
    public: double getRequestsPerSecond() {
        return ((double)recv_requests) / (ns_diff(start_time, end_time)/ 1e9);
    }

    // Throughput
    public: double getOfferedLoad() {
        return ((double)send_index) / (ns_diff(start_time, last_send_time)/ 1e9);
    }

    public: int gen_schedule(std::mt19937 &seed, std::vector<Model *> model_list) {
        std::exponential_distribution<double> exp_dist(rate / 1e9);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        uint32_t base_rid = 0;
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        uint64_t time = 0;
        uint64_t end_time = duration.count() * 1e9; // in ns
        std::vector<uint16_t> type_counts(models.size(), 0);
        while (time <= end_time) {
            // Select request type (0: short, 1:long)
            double r = uniform_dist(seed);
            int cmd_idx = -1;
            for (size_t i = 0; i < ratios.size(); ++i) {
                if (r < ratios[i]) {
                    cmd_idx = i;
                    break;
                } else {
                    r -= ratios[i];
                }
            }
            cmd_idx >= 0 ? (void)0 : abort();

            type_counts[cmd_idx]++;

            //Pick interval
            if (uniform) {
                double interval_ns = 1e9*1.0 / rate;
                time += interval_ns;
            } else {
                uint64_t next_ns = exp_dist(seed);
                time += next_ns;
            }
            std::chrono::nanoseconds send_time(time);
            send_times.push_back(send_time + start_time); // Fill send time
            // Generate the request itself
            uint32_t rid = base_rid + n_requests++;
            Request *cr = new Request();
            requests.push_back(cr);
            assert(requests[rid] == cr);
            cr->id = rid;
            cr->model = model_list[cmd_idx];
        }
        /*
        std::cout << "Created " << n_requests << " requests spanning " << duration << ":" << std::endl;
        for (int i = 0; i < models.size(); ++i) {
            if (type_counts[i] > 0) {
                std::cout << req_type_str[i] << ": " << type_counts[i] << std::endl;
            }
        }
        */
        return 0;
    }
};
Schedule *sched;

int log_latency(Histogram_t &hist,
                std::ostream &output, std::ostream &hist_output,
                std::vector<Request *> requests, bool prune);


/** CONFIG STUFF */
/*****************/
namespace bpo = boost::program_options;
static int parse_args(int argc, char **argv, bpo::options_description &opts) {
    opts.add_options()
        ("help", "produce help message");

    bpo::variables_map vm;
    try {
        bpo::parsed_options parsed =
            bpo::command_line_parser(argc, argv).options(opts).run();
        bpo::store(parsed, vm);
        if (vm.count("help")) {
            std::cerr << opts << std::endl;
            return -1;
        }
        notify(vm);
    } catch (const bpo::error &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << opts << std::endl;
        return -1;
    }

    return 0;
}

int parse_schedule(std::string &schedule_file, Schedule *sched) {
    if (not sched)
        return EINVAL;

    try {
        YAML::Node config = YAML::LoadFile(schedule_file);
        YAML::Node rate = config["rate"];
        YAML::Node duration = config["duration"];
        YAML::Node uniform = config["uniform"];
        YAML::Node ratios = config["ratios"];
        YAML::Node models = config["models"];
        YAML::Node versions = config["versions"];

        sched->rate = rate.as<double>();
        sched->duration = std::chrono::seconds(duration.as<uint64_t>());
        sched->uniform = uniform.as<bool>();
        sched->models = models.as<std::vector<std::string>>();
        versions = versions.as<std::vector<uint16_t>>();
        sched->ratios = ratios.as<std::vector<double>>();

        std::vector<Model *> model_list;

        std::random_device rd;
        std::mt19937 seed(rd());
        for (size_t i = 0; i < sched->models.size(); ++i) {
            Model *model = new Model(sched->models[i]);
            model->options.model_version_ = versions[i].as<char>();
            model->options.client_timeout_ = 0; // FIXME unlimited timeout?
            if (model->name == "googlenet") {
                std::vector<int64_t> shape{1, 3, 224, 224};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            } else if (model->name == "mnist-8") {
                std::vector<int64_t> shape{1, 1rror: ‘const class inference::InferTensorContents’ has no member named ‘byte_contents’; did you mean ‘bytes_contents, 28, 28};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            } else if (model->name == "arcfaceresnet100-8") {
                std::vector<int64_t> shape{1, 3, 112, 112};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            } else if (model->name == "densenet121") {
                std::vector<int64_t> shape{1, 3, 224, 224};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            } else if (model->name == "inception_v3") {
                std::vector<int64_t> shape{1, 3, 224, 224};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            } else if (model->name == "mobilenet_v2") {
                std::vector<int64_t> shape{1, 3, 224, 224};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            } else if (model->name == "resnet50") {
                std::vector<int64_t> shape{1, 3, 224, 224};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            } else if (model->name == "ultraface320") {
                std::vector<int64_t> shape{1, 3, 240, 320};
                model->create_model_io("input_1", "output_0", "FP32", shape);
            }

            model_list.push_back(model);
        }

        sched->gen_schedule(seed, model_list);
    } catch (YAML::ParserException& e) {
        std::cout << "Failed to parse schedule: " << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

//std::vector<double> latencies;
//std::vector<uint64_t> send_times;
uint32_t recv_requests = 0;
std::mutex cbtex;
int receive_callback(tc::InferResult *result) {
    if (result->RequestStatus().IsOk()) {
        std::string rid;
        result->Id(&rid);
        std::lock_guard<std::mutex> lock(cbtex);
        auto req = sched->requests[std::stoi(rid)];
        req->receive_time = rdtscp(NULL);
        {
            //std::cout << "received result for query " << rid << std::endl;
            //uint64_t end_time = rdtscp(NULL) - send_times[std::stoi(rid)];
            //latencies.push_back(end_time);
            recv_requests++;
        }
    } else {
        std::cout << "Error processing request " << result->RequestStatus() << std::endl;
    }

    return 0;
}

int main(int argc, char** argv) {
    time_calibrate_tsc();
    bool verbose = false;
    tc::Headers http_headers;

    /*
    send_times.reserve(1e6);
    latencies.reserve(1e6);
    */
    /* Parse options */
    int max_concurrency;
    std::string label, schedule_file;
    std::string remote_host, remote_port;
    std::string output_filename;
    bpo::options_description desc{"Rate loop client options"};
    desc.add_options()
        ("ip,I", bpo::value<std::string>(&remote_host)->required(), "server IP")
        ("port,P", bpo::value<std::string>(&remote_port)->required(), "server's port")
        ("label,l", bpo::value<std::string>(&label)->default_value("rateclient"), "experiment label")
        ("max-concurrency,m", bpo::value<int>(&max_concurrency)->default_value(1e9), "maximum number of in-flight requests")
        ("schedule-file,s", bpo::value<std::string>(&schedule_file)->required(), "path to experiment schedule")
        ("out,o", bpo::value<std::string>(&output_filename), "path to output file (defaults to log directory)");

    if (parse_args(argc, argv, desc)) {
        std::cerr << "Error parsing arguments" << std::endl;
        exit(-1);
    }

    std::string url(remote_host);
    url.append(":");
    url.append(remote_port);

    // Parse schedule
    sched = new Schedule();
    parse_schedule(schedule_file, sched);

    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create grpc client"
    );

    // Set up a bunch of timing related variables
    bool terminate = false;
    sched->start_time = std::chrono::steady_clock::now();
    std::chrono::nanoseconds start_offset = std::chrono::duration_cast<std::chrono::nanoseconds>(
            sched->start_time - sched->send_times[0]
    );
    std::vector<std::chrono::steady_clock::time_point>::iterator send_time_it = sched->send_times.begin();
    std::chrono::steady_clock::time_point next_send_time = (*send_time_it + start_offset);
    sched->end_time = sched->send_times.back() + start_offset;
    std::chrono::steady_clock::time_point max_end_time = sched->end_time + std::chrono::seconds(5);
    std::chrono::steady_clock::time_point iter_time = sched->start_time;
    std::cout << "Starting experiment " << schedule_file << std::endl;
    while (recv_requests < sched->n_requests and !terminate) {
        iter_time = std::chrono::steady_clock::now();
        int32_t inflight = sched->send_index - recv_requests;
        // While we are due some requests and not hitting max outstanding cap, send new requests
        while (iter_time > next_send_time and iter_time < sched->end_time and inflight < max_concurrency) {
            auto req = sched->requests[sched->send_index];
            req->model->options.request_id_ = std::to_string(sched->send_index);
            FAIL_IF_ERR(
                client->AsyncInfer(receive_callback, req->model->options, req->model->inputs, req->model->outputs, http_headers),
                "Unable to run model"
            );
            {
                std::lock_guard<std::mutex> lock(cbtex);
                //send_times[sched->send_index++] = rdtscp(NULL);
                sched->send_index++;
                req->send_time = rdtscp(NULL);
            }
            next_send_time = (*send_time_it + start_offset);
            send_time_it++;
        }
        // Check whether we need to exit the send_time loop
        if (next_send_time >= sched->end_time or iter_time > sched->end_time) {
            while (std::chrono::steady_clock::now() - sched->start_time > max_end_time.time_since_epoch()); // run grace period
            terminate = true;
            //std::cout << "send_time complete or iter_time > end_time" << std::endl;
        }
    }

    std::cout << "Sent: " << sched->send_index << ". Received: " << recv_requests << std::endl;
    if (recv_requests == 0) {
        std::cerr << "Received 0 requests?" << std::endl;
        exit(1);
    }
    sched->recv_requests = recv_requests;

    // Process latencies
    Histogram_t hist;
    std::ofstream lat_file(output_filename);
    std::ofstream hist_file(output_filename + "_hist");
    log_latency(hist, lat_file, hist_file, sched->requests, false);
    /*
    int32_t cycles_per_us = cycles_per_ns * 1e3;
    std::sort(latencies.begin(), latencies.end());
    std::cout << std::fixed
              << "median: " << int(latencies[latencies.size() / 2] / cycles_per_us) << " us"
              << " 75th: " << int(latencies[latencies.size() * .75] / cycles_per_us) << " us"
              << " 90th: " << int(latencies[latencies.size() * .90] / cycles_per_us) << " us"
              << " 99th: " << int(latencies[latencies.size() * .99] / cycles_per_us) << " us"
              << " 99.9th: " << int(latencies[latencies.size() * .999] / cycles_per_us) << " us"
              << " average: " << int(std::accumulate(latencies.begin(), latencies.end(), 0) / latencies.size()) << " us"
              << std::endl;
    */
     return 0;
}

/** RESULTS STUFF */
/******************/
int check_hist(Histogram_t &hist, std::vector<Request *> &reqs, uint64_t nr) {
    std::cout << "======================================" << std::endl;
    std::cout << "Hist min: " << hist.min
              << ". Min: " << (reqs[0]->receive_time - reqs[0]->send_time) / cycles_per_us << std::endl;
    uint32_t c = 0;
    for (auto bucket: hist.buckets) {
        uint64_t i = bucket.first;
        if (c + bucket.second >= hist.count * .25 and c < hist.count * .25) {
            std::cout << std::fixed << "Hist 25th: " <<  (i + (i+1)) / 2.0
                      << ". 25th: "
                      << (reqs[nr*25/100]->receive_time - reqs[nr*25/100]->send_time) / (cycles_per_us)
                      << std::endl;
        }
        if (c + bucket.second >= hist.count * .5 and c < hist.count * .5) {
            std::cout << std::fixed << "Hist 50th: " <<  (i + (i+1)) / 2.0
                      << ". 50th: "
                      << (reqs[nr/2]->receive_time - reqs[nr/2]->send_time) / (cycles_per_us)
                      << std::endl;
        }
        if (c + bucket.second >= hist.count * .75 and c < hist.count * .75) {
            std::cout << std::fixed << "Hist 75th: " <<  (i + (i+1)) / 2.0
            //std::cout << "Hist 75th: " << ((1UL << i) + (1UL << (i-1))) / 2.0
                      << ". 75th: "
                      << (reqs[nr*75/100]->receive_time - reqs[nr*75/100]->send_time) / (cycles_per_us)
                      << std::endl;
        }
        if (c + bucket.second >= hist.count * .9 and c < hist.count * .9) {
            std::cout << std::fixed << "Hist 90th: " <<  (i + (i+1)) / 2.0
                      << ". 90th: "
                      << (reqs[nr*90/100]->receive_time - reqs[nr*90/100]->send_time) / (cycles_per_us)
                      << std::endl;
        }
        if (c + bucket.second >= hist.count * .99 and c < hist.count * .99) {
            std::cout << std::fixed << "Hist 99th: " <<  (i + (i+1)) / 2.0
                      << ". 99th: "
                      << (reqs[nr*99/100]->receive_time - reqs[nr*99/100]->send_time) / (cycles_per_us)
                      << std::endl;
        }
        if (c + bucket.second >= hist.count * .999 and c < hist.count * .999) {
            std::cout << std::fixed << "Hist 99.9th: " <<  (i + (i+1)) / 2.0
                      << ". 99.9th: "
                      << (reqs[nr*999/1000]->receive_time - reqs[nr*999/1000]->send_time) / (cycles_per_us)
                      << std::endl;
        }
        if (c + bucket.second >= hist.count * .9999 and c < hist.count * .9999) {
            std::cout << std::fixed << "Hist 99.99th: " <<  (i + (i+1)) / 2.0
                      << ". 99.99th: "
                      << (reqs[nr*9999/10000]->receive_time - reqs[nr*9999/10000]->send_time) / (cycles_per_us)
                      << std::endl;
        }
        c += bucket.second;
    }
    assert(c == hist.count);

    std::cout << "Hist max: " << hist.max
              << ". Max: " << (reqs[nr]->receive_time - reqs[nr]->send_time) / (cycles_per_us) << std::endl;
    std::cout << "Hist count: " << c << ". Num requests: " << nr+1 << std::endl;
    std::cout << "======================================" << std::endl;
    return 0;
}
#define BUCKET_SIZE 1 // in us

static inline void insert_value(Histogram_t *hist, uint64_t val) {
    if (val > hist->max) {
        hist->max = val;
    }
    if (val < hist->min or hist->min == 0) {
        hist->min = val;
    }
    hist->total += val;

    int bucket = 0;
    bucket = val / BUCKET_SIZE;
    hist->buckets[bucket]++;
    hist->count++;
};

static uint64_t req_latency(const Request &req) {
    if (req.receive_time == 0) {
        return LLONG_MAX;
    } else {
        return (req.receive_time - req.send_time) / cycles_per_us;
    }
}

int log_latency(Histogram_t &hist,
                std::ostream &output, std::ostream &hist_output,
                std::vector<Request *> requests, bool prune=false) {
    if (requests.size() == 0) {
        return ENOENT;
    }
    float prune_factor = 0.0;
    if (prune) {
        prune_factor = 0.1;
    }
    std::cout << "Initial sample size: " << requests.size() << std::endl;
    std::vector<Request *> pruned_requests(requests.begin() + requests.size() * prune_factor, requests.end());

    // Sort all requests by response time
    auto lat_cmp = [](Request *a, Request *b) { return req_latency(*a) < req_latency(*b); };
    std::sort(pruned_requests.begin(), pruned_requests.end(), lat_cmp);

    // Histograms
    std::cout << "Processing " << pruned_requests.size() << " samples" << std::endl;
    hist.buckets.clear();
    hist.min = 0; hist.max = 0; hist.count = 0; hist.total = 0;
    for (uint64_t i = 0; i < pruned_requests.size(); ++i) {
        if (pruned_requests[i]->receive_time == 0)
            continue;
        // Store values in microseconds
        insert_value(&hist, (pruned_requests[i]->receive_time - pruned_requests[i]->send_time) / cycles_per_us);
    }
    if (hist.count == 0) {
        std::cout << "no values inserted in histogram" << std::endl;
        return -1;
    }
    std::cout << "Valid sample size: " << hist.count << std::endl;
    // First line is histo for all samples
    hist_output << "MODEL\tMIN\tMAX\tCOUNT\tTOTAL";
    for (auto bucket: hist.buckets) {
        hist_output << "\t" << bucket.first;
    }
    hist_output << std::endl;
    hist_output << "ALL\t"
                << hist.min << "\t"
                << hist.max << "\t"
                << hist.count << "\t"
                << hist.total;
    for (auto bucket: hist.buckets) {
        hist_output << "\t" << bucket.second;
    }
    hist_output << std::endl;
    check_hist(hist, pruned_requests, hist.count - 1);
    // Then one line per request type
    std::unordered_map<Model *, std::vector<Request *>> requests_by_type{};
    for (uint64_t i = 0; i < pruned_requests.size(); ++i) {
        if (pruned_requests[i]->receive_time == 0)
            continue;

        auto &req_ptr = pruned_requests[i];
        if (requests_by_type.find(req_ptr->model) == requests_by_type.end())
            requests_by_type[req_ptr->model] = std::vector<Request *>{};

        requests_by_type[req_ptr->model].push_back(req_ptr);
    }
    for (auto &rtype: requests_by_type) {
        hist.buckets.clear();
        hist.min = 0; hist.max = 0; hist.count = 0; hist.total = 0;
        auto reqs = rtype.second;
        for (uint64_t i = 0; i < reqs.size(); ++i) {
            // Store values in nanoseconds
            insert_value(&hist, (reqs[i]->receive_time - reqs[i]->send_time) / cycles_per_us);
        }
        if (hist.count == 0) {
            std::cout
                << "No values inserted in histogram for model "
                << rtype.first->name
                << std::endl;
            continue;
        }
        hist_output << "MODEL\tMIN\tMAX\tCOUNT\tTOTAL";
        for (auto bucket: hist.buckets) {
            hist_output << "\t" << bucket.first;
        }
        hist_output << std::endl;
        hist_output << rtype.first->name << "\t"
                    << hist.min << "\t"
                    << hist.max << "\t"
                    << hist.count << "\t"
                    << hist.total;
        for (auto bucket: hist.buckets) {
            hist_output << "\t" << bucket.second;
        }
        hist_output << std::endl;
        assert(hist.count == reqs.size());

        check_hist(hist, reqs, reqs.size() - 1);
    }
    return 0;
}
