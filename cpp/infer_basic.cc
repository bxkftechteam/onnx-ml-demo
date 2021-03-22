// A slightly modified version of
// https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h
// * Env and session were initialized to share thread pool and memory pool
// * Handled multiple output values and various types of output values
// * Added stopwatch for measuring inference cost
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "stopwatch.h"

OrtCustomOp *CreateViterbiOp();

struct CustomOpDomainDeleter {
  void operator()(OrtCustomOpDomain *domain) { Ort::OrtRelease(domain); }
};
using OrtCustomOpDomainUniquePtr =
    std::unique_ptr<OrtCustomOpDomain, CustomOpDomainDeleter>;

OrtCustomOpDomainUniquePtr custom_op_domain_;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage %s <onnx_model_path> <N>\n", argv[0]);
    return 1;
  }
  const char *model_path = argv[1];
  size_t N = atoi(argv[2]);
  if (N <= 0) {
    fprintf(stderr, "N <= 0\n");
    return 1;
  }
  bool no_output = false;
  if (argc == 4 && strcmp(argv[3], "q") == 0) {
    no_output = true;
  }

  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info

  // XXX: We initialize ONNX runtime with global (shared) thread pool and
  // shared allocator between sessions.
  OrtThreadingOptions *ort_tp_options;
  Ort::ThrowOnError(Ort::GetApi().CreateThreadingOptions(&ort_tp_options));
  Ort::Base<OrtThreadingOptions> auto_release_threading_options(ort_tp_options);
  Ort::ThrowOnError(
      Ort::GetApi().SetGlobalIntraOpNumThreads(ort_tp_options, 0));
  Ort::ThrowOnError(
      Ort::GetApi().SetGlobalInterOpNumThreads(ort_tp_options, 0));
  Ort::ThrowOnError(Ort::GetApi().SetGlobalSpinControl(ort_tp_options, 0));
  Ort::Env env(ort_tp_options, ORT_LOGGING_LEVEL_WARNING, "test");
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  env.CreateAndRegisterAllocator(memory_info, nullptr);

  // XXX: Register custom ops for inferencing Multinomial HMM
  OrtCustomOpDomain *custom_op_domain = nullptr;
  Ort::ThrowOnError(
      Ort::GetApi().CreateCustomOpDomain("ml.hmm", &custom_op_domain));
  custom_op_domain_.reset(custom_op_domain);
  OrtCustomOp *op = CreateViterbiOp();
  Ort::ThrowOnError(Ort::GetApi().CustomOpDomain_Add(custom_op_domain, op));

  // XXX: enable shared thread pool and shared memory allocators to reduce
  // memory usage when you have multiple sessions in one process.
  Ort::SessionOptions session_options;
  // session_options.SetIntraOpNumThreads(1);
  session_options.DisablePerSessionThreads();
  session_options.AddConfigEntry("session.use_env_allocators", "1");
  // Sets graph optimization level
  session_options.Add(custom_op_domain);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  // create session and load model into memory
  fprintf(stderr, "Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char *> input_node_names(num_input_nodes);
  fprintf(stderr, "Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char *input_name = session.GetInputName(i, allocator);
    fprintf(stderr, "Input %zu : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    fprintf(stderr, "Input %zu : type=%d\n", i, type);

    // print input shapes/dims
    auto input_node_dims = tensor_info.GetShape();
    fprintf(stderr, "Input %zu : num_dims=%zu\n", i, input_node_dims.size());
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      fprintf(stderr, "Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);
    }
  }

  // iterate over all output nodes
  std::vector<const char *> output_node_names;
  size_t num_output_nodes = session.GetOutputCount();
  for (size_t i = 0; i < num_output_nodes; i++) {
    // print output node names
    char *output_name = session.GetOutputName(i, allocator);
    fprintf(stderr, "Output %zu : name=%s\n", i, output_name);
    output_node_names.push_back(output_name);

    // print output node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    ONNXType onnx_type = type_info.GetONNXType();
    if (onnx_type == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      ONNXTensorElementDataType type = tensor_info.GetElementType();
      fprintf(stderr, "Output %zu : onnx_type=Tensor\n", i);
      fprintf(stderr, "Output %zu : type=%d\n", i, type);

      // print output shapes/dims
      auto output_node_dims = tensor_info.GetShape();
      fprintf(stderr, "Output %zu : num_dims=%zu\n", i,
              output_node_dims.size());
      for (size_t j = 0; j < output_node_dims.size(); j++) {
        fprintf(stderr, "Output %zu : dim %zu=%jd\n", i, j,
                output_node_dims[j]);
      }
    } else if (onnx_type == ONNX_TYPE_SEQUENCE) {
      fprintf(stderr, "Output %zu : onnx_type=Sequence\n", i);
    } else if (onnx_type == ONNX_TYPE_MAP) {
      fprintf(stderr, "Output %zu : onnx_type=Map\n", i);
    } else {
      fprintf(stderr, "Output %zu : onnx_type=unknown\n", i);
    }
  }

  //*************************************************************************
  // Score the model using sample data, and inspect values

  // Create fill input_tensor_values and fill with random numbers
  std::srand(unsigned(time(nullptr)));
  Ort::Value input_tensor(nullptr);
  const char *mode = getenv("MODE");
  if (mode == nullptr || strcmp(mode, "hmm") != 0) {
    const size_t n_features = 3;
    size_t input_tensor_size = N * n_features;
    std::vector<int64_t> input_node_dims = {static_cast<int64_t>(N), 3};
    static std::vector<float> input_tensor_values(input_tensor_size);
    std::generate(input_tensor_values.begin(), input_tensor_values.end(),
                  []() { return 2.0 * rand() / RAND_MAX; });
    input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size,
        input_node_dims.data(), input_node_dims.size());
  } else {
    size_t input_tensor_size = N;
    std::vector<int64_t> input_node_dims = {static_cast<int64_t>(N)};
    static std::vector<int32_t> input_tensor_values(input_tensor_size);
    std::generate(input_tensor_values.begin(), input_tensor_values.end(),
                  []() { return rand() % 6; });
    input_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info, input_tensor_values.data(), input_tensor_size,
        input_node_dims.data(), input_node_dims.size());
  }

  // score model & input tensor, get back output tensor
  Stopwatch timer;
  auto output_values = session.Run(
      Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1,
      output_node_names.data(), output_node_names.size());
  assert(output_values.size() == output_node_names.size());
  auto run_cost = timer.Lap().count();

  for (size_t i = 0; i < output_values.size(); i++) {
    printf("Output %s:\n", output_node_names[i]);
    Ort::TypeInfo type_info = output_values[i].GetTypeInfo();
    ONNXType onnx_type = type_info.GetONNXType();
    if (onnx_type == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      size_t tensor_size = tensor_info.GetElementCount();
      auto elem_type = tensor_info.GetElementType();
      if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float *outarr = output_values[i].GetTensorData<float>();
        if (!no_output) {
          for (size_t i = 0; i < tensor_size; i++) {
            printf("  %f\n", outarr[i]);
          }
        }
      } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t *outarr = output_values[i].GetTensorData<int64_t>();
        if (!no_output) {
          for (size_t i = 0; i < tensor_size; i++) {
            printf("  %ld\n", outarr[i]);
          }
        }
      } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t *outarr = output_values[i].GetTensorData<int32_t>();
        if (!no_output) {
          for (size_t i = 0; i < tensor_size; i++) {
            printf("  %d\n", outarr[i]);
          }
        }
      } else {
        fprintf(stderr,
                "handler for tensor element type %d is not implemented\n",
                elem_type);
      }
    } else if (onnx_type == ONNX_TYPE_SEQUENCE) {
      // For simplicity, we assume that it is a sequence of Map[Tensor<int64>,
      // Tensor<float32>].
      size_t seq_size = output_values[i].GetCount();
      for (size_t j = 0; j < seq_size; j++) {
        Ort::Value ort_element_value = output_values[i].GetValue(j, allocator);
        Ort::Value ort_key = ort_element_value.GetValue(0, allocator);
        Ort::Value ort_val = ort_element_value.GetValue(1, allocator);
        // assume that ort_key is Tensor<int64>, and ort_val is
        // Tensor<float32>, and ort_key and ort_val are 1-D tensor and have
        // same length
        auto key_type_shape = ort_key.GetTensorTypeAndShapeInfo();
        auto val_type_shape = ort_val.GetTensorTypeAndShapeInfo();
        assert(key_type_shape.GetShape().size() == 1);
        assert(key_type_shape.GetShape() == val_type_shape.GetShape());
        const int64_t *key_data = ort_key.GetTensorData<int64_t>();
        const float *val_data = ort_val.GetTensorData<float>();
        if (!no_output) {
          size_t num_entries = key_type_shape.GetShape()[0];
          printf("  ");
          for (size_t k = 0; k < num_entries; k++) {
            printf("%ld->%f ", key_data[k], val_data[k]);
          }
          printf("\n");
        }
      }
    } else {
      fprintf(stderr, "handler for type %d is not implemented\n", onnx_type);
    }
  }
  auto fetch_result_cost = timer.Lap().count();
  auto total_cost = timer.Reset().count();

  fprintf(stderr,
          "Done. total_cost: %ld, run_cost: %ld, fetch_result_cost: %ld\n",
          total_cost, run_cost, fetch_result_cost);
  for (const char *input_node_name : input_node_names) {
    allocator.Free((void *)input_node_name);
  }
  for (const char *output_node_name : output_node_names) {
    allocator.Free((void *)output_node_name);
  }
  return 0;
}
