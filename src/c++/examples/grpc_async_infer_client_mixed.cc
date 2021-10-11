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

std::vector<double> latencies;
uint32_t recv_requests = 0;
std::mutex cbtex;
int receive_callback(tc::InferResult *result) {
    if (result->RequestStatus().IsOk()) {
        {
            // lock latency vector
            std::lock_guard<std::mutex> lock(cbtex);
            /*
            uint64_t end_time = std::chrono::steady_clock::now() - std::chrono::duration_cast<std::chrono::microseconds>(
            ).count();
            */
            uint64_t end_time = reinterpret_cast<InferResultGrpc>(result)->Timer()::Duration(SEND_START, SEND_END);
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
    std::vector<int32_t> input0_data(16);
    std::vector<int32_t> input1_data(16);
    for (size_t i = 0; i < 16; ++i) {
        input0_data[i] = i;
        input1_data[i] = 1;
    }
    std::vector<int64_t> shape{1, 16};
    // Initialize the inputs with the data.
    tc::InferInput* input0;
    tc::InferInput* input1;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input0, "INPUT0", shape, "INT32"), "unable to get INPUT0"
     );
    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    FAIL_IF_ERR(
        tc::InferInput::Create(&input1, "INPUT1", shape, "INT32"), "unable to get INPUT1"
    );
    std::shared_ptr<tc::InferInput> input1_ptr;
    input1_ptr.reset(input1);
    FAIL_IF_ERR(
        input0_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input0_data[0]),
            input0_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT0");
    FAIL_IF_ERR(
        input1_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input1_data[0]),
            input1_data.size() * sizeof(int32_t)),
        "unable to set data for INPUT1");
     // Generate the outputs to be requested.
     tc::InferRequestedOutput* output0;
     tc::InferRequestedOutput* output1;
     FAIL_IF_ERR(
         tc::InferRequestedOutput::Create(&output0, "OUTPUT0"),
         "unable to get 'OUTPUT0'");
     std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
     output0_ptr.reset(output0);
     FAIL_IF_ERR(
         tc::InferRequestedOutput::Create(&output1, "OUTPUT1"),
         "unable to get 'OUTPUT1'");
     std::shared_ptr<tc::InferRequestedOutput> output1_ptr;
     output1_ptr.reset(output1);

    // The inference settings.
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = 0; // FIXME unlimited timeout?
    std::vector<tc::InferInput*> inputs = {input0_ptr.get(), input1_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output0_ptr.get(), output1_ptr.get()};

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
            FAIL_IF_ERR(
                client->AsyncInfer(receive_callback, options, inputs, outputs, http_headers),
                "Unable to run model");
            send_index++;
            auto gen_us = std::chrono::microseconds(static_cast<uint64_t>(exp_dist(seed)));
            next_send_time = iter_time + gen_us;
        }
        // Check whether we need to exit the sending loop
        if (next_send_time >= end_time or iter_time > end_time) {
            while (std::chrono::steady_clock::now() - start_time > max_duration.time_since_epoch()); // run grace period
            terminate = true;
            std::cout << "Sending complete or iter_time > end_time" << std::endl;
        }
    }

    // Process latencies
    std::sort(latencies.begin(), latencies.end());
    std::cout << "median: " << latencies[latencies.size() / 2]
              << " 75th: " << latencies[latencies.size() * .75]
              << " 90th: " << latencies[latencies.size() * .90]
              << " 99th: " << latencies[latencies.size() * .99]
              << " 99.9th: " << latencies[latencies.size() * .999]
              //<< " average: " << average
              << std::endl;
     return 0;
}
