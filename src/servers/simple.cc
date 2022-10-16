// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <map>
#include "src/servers/common.h"
#include "triton/core/tritonserver.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

#include "chan.h"

namespace ni = nvidia::inferenceserver;

namespace {

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

/**** WORKLOAD & CLIENT ***/
class Model {
    public:
        std::string name;
        std::vector<int64_t> input_shape;
        std::vector<char> inputs;
        float ratio;

        Model(std::string name, std::vector<int64_t> input_shape, float ratio) :
            name(name), input_shape(input_shape), ratio(ratio) {}
}

class Request {
    public: Model *model;
    public: uint64_t send_time;
    public: uint64_t receive_time;
    public: uint64_t latency;
    public: uint32_t id;
};

#define TOTAL_REQUESTS 3000

std::vector<float> model_ratios{
    .111, .111, .111, .111, .111, .111, .111, .111, .111
};

std::vector<Model> models{
    Model("densenet-9", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("googlenet-9", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("inceptionv3", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("mobilenetv2-7", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("resnet18-v2-7", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("resnet34-v2-7", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("resnet50-v2-7", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("squeezenet1.1-7", std::vector<int64_t> {1, 3, 224, 224}, .111)
    Model("mnist-8", std::vector<int64_t> {1, 3, 224, 224}, .111)
};

void setupModels() {
  for (auto &model: models) {
      size_t num_elements =
          model.input_shape[0] * model.input_shape[1] * model.input_shape[2] * model.input_shape[3];
      model.inputs->reserve(num_elements);
      for (size_t i = 0; i < num_elements; ++i) {
        ((int*)model.input->data())[i] = i;
      }
  }
}

ipc::ShmChannelCpuWriter submit_channel;
ipc::ShmChannelCpuReader results_channel;

void client_setup() {
    setup_models();
    submit_channel.connect("triton-client-submit-chan", 65536);
    results_channel.connect("triton-client-results-chan", 65536);
}


const max_inflight = 256;
void client(float sigma, float rate) {
    std::random_device rd;
    std::mt19937 seed(rd());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::lognormal_distribution<double> lognormal_dist(std::log(1000000000.0 / rate) - (sigma * sigma/2), sigma);
    // Start a sending loop
    int nsent = 0;
    const int *msg_buffer[max_inflight];
    while (nsent < TOTAL_REQUESTS) {
        // Pick a model
        double r = uniform_dist(seed);
        int cmd_idx = -1;
        for (size_t i = 0; i < models.size(); ++i) {
            if (r < model_ratios[i]) {
                cmd_idx = i;
                break;
            } else {
                r -= model_ratios[i];
            }
        }
        cmd_idx >= 0 ? (void)0 : abort();

        // Pick an interval
        uint64_t next_ns = lognormal_dist(seed);

        auto send_time = std::chrono::steady_clock::now() + std::chrono::nanoseconds(next_ns);
        while (std::chrono::steady_clock::now() < send_time) {}

        msg_buffer[nrequests % max_inflight] = cmd_idx;
        submit_channel.write(&msg_buffer[nrequests % max_inflight]);
    }
}


/********************/

// bool enforce_memory_type = false;
bool enforce_memory_type = true;
TRITONSERVER_MemoryType requested_memory_type = TRITONSERVER_MEMORY_GPU;

#ifdef TRITON_ENABLE_GPU
static auto cuda_data_deleter = [](void* data) {
  if (data != nullptr) {
    cudaPointerAttributes attr;
    auto cuerr = cudaPointerGetAttributes(&attr, data);
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to get CUDA pointer attribute of " << data
                << ": " << cudaGetErrorString(cuerr) << std::endl;
    }
    if (attr.type == cudaMemoryTypeDevice) {
      cuerr = cudaFree(data);
    } else if (attr.type == cudaMemoryTypeHost) {
      cuerr = cudaFreeHost(data);
    }
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to release CUDA pointer " << data << ": "
                << cudaGetErrorString(cuerr) << std::endl;
    }
  }
};
#endif  // TRITON_ENABLE_GPU

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    if (enforce_memory_type) {
      *actual_memory_type = requested_memory_type;
    }

    switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_CPU_PINNED: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "unable to recover current CUDA device: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }

        err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "cudaHostAlloc failed: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }
        break;
      }

      case TRITONSERVER_MEMORY_GPU: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "unable to recover current CUDA device: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }

        err = cudaMalloc(&allocated_ptr, byte_size);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "cudaMalloc failed: " + std::string(cudaGetErrorString(err)))
                  .c_str());
        }
        break;
      }
#endif  // TRITON_ENABLE_GPU

      // Use CPU memory if the requested memory type is unknown
      // (default case).
      case TRITONSERVER_MEMORY_CPU:
      default: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      std::cout << "allocated " << byte_size << " bytes in "
                << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                << " for result tensor " << tensor_name << std::endl;
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  std::cout << "Releasing buffer " << buffer << " of size " << byte_size
            << " in " << TRITONSERVER_MemoryTypeString(memory_type)
            << " for result '" << *name << "'" << std::endl;
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
#ifdef TRITON_ENABLE_GPU
    case TRITONSERVER_MEMORY_CPU_PINNED: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFreeHost(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
    case TRITONSERVER_MEMORY_GPU: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFree(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
#endif  // TRITON_ENABLE_GPU
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }

  delete name;

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  // We reuse the request so we don't delete it here.
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

TRITONSERVER_Error*
ParseModelMetadata(
    const rapidjson::Document& model_metadata, bool* is_int,
    bool* is_torch_model)
{
  std::string seen_data_type;
  for (const auto& input : model_metadata["inputs"].GetArray()) {
    if (strcmp(input["datatype"].GetString(), "INT32") &&
        strcmp(input["datatype"].GetString(), "FP32")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    }
    if (seen_data_type.empty()) {
      seen_data_type = input["datatype"].GetString();
    } else if (strcmp(seen_data_type.c_str(), input["datatype"].GetString())) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }
  for (const auto& output : model_metadata["outputs"].GetArray()) {
    if (strcmp(output["datatype"].GetString(), "INT32") &&
        strcmp(output["datatype"].GetString(), "FP32")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "simple lib example only supports model with data type INT32 or "
          "FP32");
    } else if (strcmp(seen_data_type.c_str(), output["datatype"].GetString())) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of 'simple' model must have the data type");
    }
  }

  *is_int = (strcmp(seen_data_type.c_str(), "INT32") == 0);
  *is_torch_model =
      (strcmp(model_metadata["platform"].GetString(), "pytorch_libtorch") == 0);
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {

  std::string model_repository_path = "/models";
  int verbose_level = 0;
  requested_memory_type = TRITONSERVER_MEMORY_GPU;

  float rate = atof(argv[0]);
  float sigma = atof(argv[1]);
  const char experiment_name = argv[2];

  // setup client
  client_setup();

  // Create the server...
  TRITONSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsNew(&server_options),
      "creating server options");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(
          server_options, model_repository_path.c_str()),
      "setting model repository path");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
      "setting verbose logging level");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          server_options, "/opt/tritonserver/backends"),
      "setting backend directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
      "setting strict model configuration");
  double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          server_options, min_compute_capability),
      "setting minimum supported CUDA compute capability");

  TRITONSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsDelete(server_options),
      "deleting server options");

  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Wait until the server is both live and ready.
  size_t health_iters = 0;
  while (true) {
    bool live, ready;
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsLive(server.get(), &live),
        "unable to get server liveness");
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsReady(server.get(), &ready),
        "unable to get server readiness");
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      break;
    }

    if (++health_iters >= 10) {
      FAIL("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Print status of the server.
  {
    TRITONSERVER_Message* server_metadata_message;
    FAIL_IF_ERR(
        TRITONSERVER_ServerMetadata(server.get(), &server_metadata_message),
        "unable to get server metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(
            server_metadata_message, &buffer, &byte_size),
        "unable to serialize server metadata message");

    std::cout << "Server Status:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;

    FAIL_IF_ERR(
        TRITONSERVER_MessageDelete(server_metadata_message),
        "deleting status metadata");
  }

  // Wait for the model to become available.
  bool is_torch_model = false;
  bool is_int = false;
  bool is_ready = false;
  health_iters = 0;
  while (!is_ready) {
    for (auto model: models) {
        FAIL(
            TRITONSERVER_ServerModelIsReady(
                server.get(), model.name.c_str(), 1, &is_ready),
            "unable to get model readiness");
        if (!is_ready) {
          if (++health_iters >= 10 * models.size()) {
            FAIL("model failed to be ready in 10 iterations");
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
          continue;
        }

        TRITONSERVER_Message* model_metadata_message;
        FAIL_IF_ERR(
            TRITONSERVER_ServerModelMetadata(
                server.get(), model.name.c_str(), 1, &model_metadata_message),
            "unable to get model metadata message");
        const char* buffer;
        size_t byte_size;
        FAIL_IF_ERR(
            TRITONSERVER_MessageSerializeToJson(
                model_metadata_message, &buffer, &byte_size),
            "unable to serialize model status protobuf");

        rapidjson::Document model_metadata;
        model_metadata.Parse(buffer, byte_size);
        if (model_metadata.HasParseError()) {
          FAIL(
              "error: failed to parse model metadata from JSON: " +
              std::string(GetParseError_En(model_metadata.GetParseError())) +
              " at " + std::to_string(model_metadata.GetErrorOffset()));
        }

        FAIL_IF_ERR(
            TRITONSERVER_MessageDelete(model_metadata_message),
            "deleting status protobuf");

        if (strcmp(model_metadata["name"].GetString(), model.name.c_str())) {
          FAIL("unable to find metadata for model");
        }

        bool found_version = false;
        if (model_metadata.HasMember("versions")) {
          for (const auto& version : model_metadata["versions"].GetArray()) {
            if (strcmp(version.GetString(), "1") == 0) {
              found_version = true;
              break;
            }
          }
        }
        if (!found_version) {
          FAIL("unable to find version 1 status for model");
        }

        FAIL_IF_ERR(
            ParseModelMetadata(model_metadata, &is_int, &is_torch_model),
            "parsing model metadata");
    }
  }


  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
      "creating response allocator");


  std::thread(&client, sigma, rate);

  const TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_FP32;
  auto input0 = "INPUT0";
  auto output0 = "OUTPUT0";
  int nrequests = 0;
  std::vector<*Request> requests;
  requests.reserve(TOTAL_REQUESTS);
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  while (nrequests < TOTAL_REQUESTS) {
      // TODO attempt to dequeue a "request" (model to process) from channel
      int n = 1;
      Model model = models[n];

      Request req = new Request();
      req->model = &model;
      req->id = nrequests;
      req->send_time = rdtscp(NULL);;

      // Inference
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestNew(
              &irequest, server.get(), model.name.c_str(), -1 /* model_version */),
          "creating inference request");
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestSetId(irequest, itoa(n)), // XXX does that need to be model ID or req ID?
          "setting ID for the request");
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestSetReleaseCallback(
              irequest, InferRequestComplete, nullptr /* request_release_userp */),
          "setting request release callback");

      // Inputs
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddInput(
              irequest, input0, datatype, &model.input_shape[0], model.input_shape.size()),
          "setting input 0 meta-data for the request");
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
          "requesting output 0 for the request");

      size_t input0_size = model.inputs.size();

      // Copy request inputs to GPU
      std::unique_ptr<void, decltype(cuda_data_deleter)> input0_gpu(nullptr, cuda_data_deleter);
      FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
      void* dst;
      FAIL_IF_CUDA_ERR(cudaMalloc(&dst, input0_size), "allocating GPU memory for INPUT0 data");
      input0_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
        cudaMemcpy(dst, &model_inputs_0[model_name][0], input0_size, cudaMemcpyHostToDevice),
        "setting INPUT0 data in GPU memory");
      const void* input0_base = input0_gpu.get()

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input0, input0_base, input0_size, requested_memory_type,
              0 /* memory_type_id */),
          "assigning INPUT0 data");

      // Perform inference...
      {
        auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
        std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestSetResponseCallback(
                irequest, allocator, nullptr /* response_allocator_userp */,
                InferResponseComplete, reinterpret_cast<void*>(p)),
            "setting response callback");

        FAIL_IF_ERR(
            TRITONSERVER_ServerInferAsync(
                server.get(), irequest, nullptr /* trace */),
            "running inference");

        // Wait for the inference to complete.
        TRITONSERVER_InferenceResponse* completed_response = completed.get();

        req->receive_time = rdtscp(NULL);
        req->latency = req->receive_time - req->send_time;
        requests.push_back(&req);

        FAIL_IF_ERR(
            TRITONSERVER_InferenceResponseError(completed_response),
            "response status");

        FAIL_IF_ERR(
            TRITONSERVER_InferenceResponseDelete(completed_response),
            "deleting inference response");
      }

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestRemoveAllInputData(irequest, input0),
          "removing INPUT0 data");

      nrequests += 1;
  }

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestDelete(irequest),
      "deleting inference request");
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorDelete(allocator),
      "deleting response allocator");

    std::string output_filename = experiment_label + "-results.csv";
    std::ofstream output(output_filename);
    output << "REQ_ID\tMODEL\tSENT\tRECEIVED\tLATENCY\t" << std::endl;
    for (auto req: requests) {
        output <<
            req->id << "\t"
            req->model.name << "\t"
            req->send_time / cycles_per_ns << "\t"
            req->receive_time / cycles_per_ns << "\t"
            req->latency / cycles_per_ns << std::endl;
    }
    output.close();

  return 0;
}
