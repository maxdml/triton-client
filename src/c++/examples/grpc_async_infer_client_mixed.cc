#include <iostream>
#include <string>
#include "grpc_client.h"
#include <random>

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

float cycles_per_ns;
// From Adam's base OS
inline int time_calibrate_tsc(void) {
    struct timespec sleeptime;
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

        secs = (double)ns / 1000;
        cycles_per_ns = ((uint64_t)((end - start) / secs)) / 1000.0;
        printf("time: detected %.03f ticks / us\n", cycles_per_ns);

        return 0;
    }

    return -1;
}

/** WORKLOAD GENERATION STUFF */
/******************************/

std::unordered_map<std::string, std::tuple<tc::InferOptions, std::vector<tc::InferInput *>, std::vector<tc::InferRequestedOutput *>> options;

class Request {
    public: uint32_t model_id;
    public: uint64_t send_time;
    public: uint64_t receive_time;
    public: uint64_t latency;
    public: uint32_t id;
}

class Schedule {
    public: std::chrono::seconds duration;
    public: std::chrono::seconds max_duration;
    public: tp start_time;
    public: tp end_time;
    public: tp last_send_time;
    public: uint32_t n_requests = 0;
    public: double rate = 0;
    public: bool uniform = false;
    public: std::vector<double> ratios;
    public: std::vector<tp> send_times;
    public: uint32_t send_index = 0;
    public: uint32_t recv_requests = 0;
    public: uint32_t attempts = 0;
    public: uint32_t ev_count = 0;
    public: uint32_t n_skipped = 0;
    public: std::vector<Request *> requests;
    public: std::vector<std::string> models;

    public: Schedule (uint32_t i) : schedule_id(i) {}

    // Goodput
    public: double getRequestsPerSecond() {
        return ((double)recv_requests) / (ns_diff(start_time, end_time)/ 1e9);
    }

    // Throughput
    public: double getOfferedLoad() {
        return ((double)send_index) / (ns_diff(start_time, last_send_time)/ 1e9);
    }

    public: int gen_schedule(std::mt19937 &seed) {
        std::exponential_distribution<double> exp_dist(rate / 1e9);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        uint32_t base_rid = 0;
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        uint64_t time = 0;
        uint64_t end_time = duration.count() * 1e9; // in ns
        std::vector<uin16_t> type_counts(models.size(), 0);
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

            type_counts[req_offset + cmd_idx]++;

            //Pick interval
            if (uniform) {
                double interval_ns = 1e9*1.0 / rate;
                time += interval_ns;
            } else {
                uint64_t next_ns = exp_dist(seed);
                time += next_ns;
            }
            boost::chrono::nanoseconds send_time(time);
            send_times.push_back(send_time + start_time); // Fill send time
            // Generate the request itself
            uint32_t rid = base_rid + n_requests++;
            Request *cr = new Request();
            requests.push_back(cr);
            assert(requests[rid] == cr);
            cr->id = rid;
            cr->model_id = cmd_idx;
        }
        std::cout << "Created " << n_requests << " requests spanning " << duration << ":" << std::endl;
        for (int i = 0; i < models.size(); ++i) {
            if (type_counts[i] > 0) {
                std::cout << req_type_str[i] << ": " << type_counts[i] << std::endl;
            }
        }
        return 0;
    }
};

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

int create_model_io(std::share<tc::InferInput> &input_ptr, std::shared<tc::InferRequestedOutput> &output_ptr,
                    const char *input_name, const char *output_name,
                    const char *type, std::vector<int64_t> &shape) {

    std::vector<std::vector<std::vector<int32_t>>> input_data;
    for (size_t i = 0; i < shape[1]; ++i) {
        for (size_t j = 0; j < shape[2]; ++j) {
            for (size_t k = 0; k < shape[3]; ++k) {
                input_data[i][j][k] = k;
            }
        }
    }
    tc::InferInput *input;
    FAIL_IF_ERR(tc::InferInput::Create(&input0, input_name.c_str(), shape, type.c_str()), "unable to get input");
    input_ptr.reset(input);
    FAIL_IF_ERR(
        input_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input_data[0]),
            input_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT"
    );

    tc::InferRequestedOutput *output;
    FAIL_IF_ERR(tc::InferRequestedOutput::Create(&output0, output_name.c_str()), "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    output_ptr.reset(output);

    return 0;
}

int parse_schedule(std::string &schedule_file, Schedule *sched) {
    if (not sched)
        return EINVAL;

    try {
        YAML::Node config = YAML::LoadFile(schedule_file);
        YAML::Node rate = config["rate"];
        YAML::Node duration = config["duration"];
        YAML::NODE uniform = config["uniform"];
        YAML::NODE ratios = config["ratios"];
        YAML::NODE models = config["models"];
        YAML::NODE versions = config["versions"];

        sched->rate = rate.as<double>();
        sched->duration = std::chrono::seconds(duration.as<uint64_t>());
        sched->max_duration = std::chrono::seconds(duration.as<uint64_t>() + 2);
        sched->uniform = uniform.as<bool>();
        sched->models = models.as<std::vector<std::string>>();
        versions = versions.as<std::vector<uint16_t>>();
        sched->ratios = ratios.as<std::vector<double>>();

        for (size_t i = 0; i < sched->models.size(); ++i) {
            tc::InferOptions opts(model);
            opts.model_version_ = versions[i];
            opts.client_timeout_ = 0; // FIXME unlimited timeout?
            if (model == "googlenet") {
                //std::vector<std::vector<std::vector<int32_t>>> input0_data(3, std::vector<std::vector<int32_t>>(224, std::vector<int32_t>(224)));
                std::vector<int64_t> shape{1, 3, 224, 224};
                // Initialize the inputs with the data.
                std::shared_ptr<tc::InferInput> input_ptr;
                std::shared_ptr<tc::InferRequestedOutput> output_ptr;

                create_model_io(&intput, &output, "input_1", "output_0", "FP32", shape);

                // The inference settings.
                std::vector<tc::InferInput*> inputs = {input_ptr.get()};
                std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};

                options[model] = std::tuple<tc::InferOptions, std::vector<tc::InferInput *>, std::vector<tc::InferRequestedOutput *>(
                    opts, inputs, outputs,
                ):
            }
            //TODO the rest of the models
        }

        sched->gen_schedule();
    } catch (YAML::ParserException& e) {
        std::cout << "Failed to parse schedule: " << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

std::vector<double> latencies;
std::vector<uint64_t> send_times;
uint32_t recv_requests = 0;
std::mutex cbtex;
int receive_callback(tc::InferResult *result) {
    if (result->RequestStatus().IsOk()) {
        {
            // lock latency vector
            std::lock_guard<std::mutex> lock(cbtex);
            std::string rid;
            result->Id(&rid);
            //std::cout << "received result for query " << rid << std::endl;
            uint64_t end_time = rdtscp(NULL) - send_times[std::stoi(rid)];
            latencies[recv_requests++] = end_time;
        }
    } else {
        // TODO log error
    }

    return 0;
}

int main(int argc, char** argv) {
    bool verbose = false;
    tc::Headers http_headers;
    uint32_t client_timeout = 0;

    send_times.reserve(1e6);
    latencies.reserve(1e6);


    /* Parse options */
    int max_concurrency;
    std::string label, schedule_file;
    uint16_t remote_port;
    std::string remote_host;
    std::string output_filename;
    std::string output_dirname;
    int downsample = 0;
    unsigned int collect_logs = 0;
    bpo::options_description desc{"Rate loop client options"};
    desc.add_options()
        ("ip,I", bpo::value<std::vector<std::string>>(&remote_hosts)->multitoken()->required(), "server IPs")
        ("port,P", bpo::value<std::vector<uint16_t>>(&remote_ports)->multitoken()->required(), "server's port")
        ("label,l", bpo::value<std::string>(&label)->default_value("rateclient"), "experiment label")
        ("max-concurrency,m", bpo::value<int>(&max_concurrency)->default_value(-1), "maximum number of in-flight requests")
        ("schedule-file,s", bpo::value<std::string>(&schedule_file)->required(), "path to experiment schedule")
        ("out,o", bpo::value<std::string>(&output_filename), "path to output file (defaults to log directory)")
        ("outdir,o", bpo::value<std::string>(&output_dirname), "name of output dir")
        ("collect-logs", bpo::value<unsigned int>(&collect_logs), "Activate log collection")
        ("sample,S", bpo::value<int>(&downsample), "-1: histogram, 0: full timeseries, >0: timeseries downsampled to this number");

    if (parse_args(argc, argv, desc)) {
        PSP_ERROR("Error parsing arguments");
        exit(-1);
    }

    std::string url(remote_host);
    url.append(":");
    url.append(remote_port);

    // Parse schedule
    Schedule *sched = new Schedule();
    parse_schedule(sched_file, sched);

    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create grpc client"
    );

    // TODO : update to use the schedule
    bool terminate = false;
    int32_t send_index = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point next_send_time = start_time;
    //auto duration = std::chrono::seconds(RUNTIME);
    std::chrono::steady_clock::time_point end_time = start_time + std::chrono::seconds(RUNTIME);
    std::chrono::steady_clock::time_point max_duration = end_time + std::chrono::seconds(2);
    //auto max_duration =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::microseconds(end_time) + std::chrono::seconds(2));
    std::chrono::steady_clock::time_point iter_time;
    while (!terminate) {
        iter_time = std::chrono::steady_clock::now();
        int32_t inflight = send_index - recv_requests;
        // While we are due some requests and not hitting max outstanding cap, send new requests
        while (iter_time > next_send_time and iter_time < end_time and inflight < max_concurrency) {
            options.request_id_ = std::to_string(send_index);
            FAIL_IF_ERR(
                client->AsyncInfer(receive_callback, options, inputs, outputs, http_headers),
                "Unable to run model");
            {
                std::lock_guard<std::mutex> lock(cbtex);
                send_times[send_index++] = rdtscp(NULL);
            }
            auto gen_us = std::chrono::microseconds(static_cast<uint64_t>(exp_dist(seed)));
            next_send_time = iter_time + gen_us;
        }
        // Check whether we need to exit the sending loop
        if (next_send_time >= end_time or iter_time > end_time) {
            while (std::chrono::steady_clock::now() - start_time > max_duration.time_since_epoch()); // run grace period
            terminate = true;
            //std::cout << "Sending complete or iter_time > end_time" << std::endl;
        }
    }

    std::cout << "Sent: " << send_index << ". Received: " << recv_requests << std::endl;

    // Process latencies
    int32_t cycles_per_us = cycles_per_ns * 1e3;
    std::sort(latencies.begin(), latencies.end());
    std::cout << "median: " << latencies[latencies.size() / 2] / cycles_per_us
              << " 75th: " << latencies[latencies.size() * .75] / cycles_per_us
              << " 90th: " << latencies[latencies.size() * .90] / cycles_per_us
              << " 99th: " << latencies[latencies.size() * .99] / cycles_per_us
              << " 99.9th: " << latencies[latencies.size() * .999] / cycles_per_us
              //<< " average: " << average
              << std::endl;
     return 0;
}
