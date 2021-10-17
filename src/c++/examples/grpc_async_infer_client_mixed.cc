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

    const char *server_ip_str = argv[1];
    const char *server_port = argv[2];
    std::string url(server_ip_str);
    url.append(":");
    url.append(server_port);

    // Likewy max num jobs should be <= n cores
    uint32_t max_num_jobs = atoi(argv[3]);
    double mean_inter_time = atof(argv[4]); // in microseconds

    std::exponential_distribution<double> exp_dist(mean_inter_time);
    std::random_device rd;
    std::mt19937 seed(rd());

    //TODO give a list of models and ratios
    const char *model_name = argv[5];
    const char *model_version = argv[6];

    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create grpc client"
    );

    /* Setup input and output */
    std::vector<std::vector<std::vector<int32_t>>> input0_data(3, std::vector<std::vector<int32_t>>(224, std::vector<int32_t>(224)));
    //std::vector<std::vector<int32_t>> input0_data(3, std::vector<int32_t>(224));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 224; ++j) {
            for (size_t k = 0; k < 224; ++k) {
                input0_data[i][j][k] = k;
            }
        }
    }
    std::vector<int64_t> shape{1, 3, 224, 224};
    // Initialize the inputs with the data.
    tc::InferInput* input0;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input0, "input_1", shape, "FP32"), "unable to get INPUT0"
     );
    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    FAIL_IF_ERR(
        input0_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input0_data[0]),
            input0_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT0"
    );
    // Generate the outputs to be requested.
    tc::InferRequestedOutput* output0;
    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output0, "output_0"),
        "unable to get 'OUTPUT0'");
    std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
    output0_ptr.reset(output0);

    // The inference settings.
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = 0; // FIXME unlimited timeout?
    std::vector<tc::InferInput*> inputs = {input0_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output0_ptr.get()};

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
        while (iter_time > next_send_time and iter_time < end_time and inflight < max_num_jobs) {
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
