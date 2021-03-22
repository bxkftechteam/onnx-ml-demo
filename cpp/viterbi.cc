// Implements viterbi as a custom ONNX operator, which powers the multinomial
// HMM model.

#include <onnxruntime_c_api.h>

#include <algorithm>

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>  // NOLINT [build/include_order]
#undef ORT_API_MANUAL_INIT

namespace {

class KernelViterbi {
 public:
  explicit KernelViterbi(OrtApi api) : api_(api), ort_(api_) {}

  void Compute(OrtKernelContext *context) {
    // Setup inputs
    const OrtValue *ort_log_startprob = ort_.KernelContext_GetInput(context, 0);
    const OrtValue *ort_log_transmat_t =
        ort_.KernelContext_GetInput(context, 1);
    const OrtValue *ort_log_frameprob = ort_.KernelContext_GetInput(context, 2);
    const float *log_startprob = ort_.GetTensorData<float>(ort_log_startprob);
    const float *log_transmat_t = ort_.GetTensorData<float>(ort_log_transmat_t);
    const float *log_frameprob = ort_.GetTensorData<float>(ort_log_frameprob);

    int64_t log_startprob_shape[1] = {0};
    int64_t log_transmat_t_shape[2] = {0, 0};
    int64_t log_frameprob_shape[2] = {0, 0};
    int64_t output_shape[1] = {0};

    OrtTensorTypeAndShapeInfo *ort_type_shape_info = nullptr;
    ort_type_shape_info = ort_.GetTensorTypeAndShape(ort_log_startprob);
    ort_.GetDimensions(ort_type_shape_info, log_startprob_shape, 1);
    ort_.ReleaseTensorTypeAndShapeInfo(ort_type_shape_info);
    ort_type_shape_info = ort_.GetTensorTypeAndShape(ort_log_transmat_t);
    ort_.GetDimensions(ort_type_shape_info, log_transmat_t_shape, 2);
    ort_.ReleaseTensorTypeAndShapeInfo(ort_type_shape_info);
    ort_type_shape_info = ort_.GetTensorTypeAndShape(ort_log_frameprob);
    ort_.GetDimensions(ort_type_shape_info, log_frameprob_shape, 2);
    ort_.ReleaseTensorTypeAndShapeInfo(ort_type_shape_info);

    const int64_t n_samples = log_frameprob_shape[0];
    const int64_t n_components = log_frameprob_shape[1];
    if (log_transmat_t_shape[0] != log_transmat_t_shape[1]) {
      throw Ort::Exception("log_transmat is not a square matrix",
                           ORT_INVALID_GRAPH);
    }
    if (log_transmat_t_shape[0] != log_startprob_shape[0]) {
      throw Ort::Exception(
          "length of log_startprob does not match with size of log_transmat",
          ORT_INVALID_GRAPH);
    }
    if (n_components != log_startprob_shape[0]) {
      throw Ort::Exception("input shape does not match with model shape",
                           ORT_INVALID_ARGUMENT);
    }

    // Setup output
    output_shape[0] = n_samples;
    OrtValue *output =
        ort_.KernelContext_GetOutput(context, 0, output_shape, 1);
    int32_t *out = ort_.GetTensorMutableData<int32_t>(output);

    // Skip computation when input log_frameprob is empty
    if (n_samples == 0) {
      return;
    }

    // Compute viterbi and copy result to kernel output
    Compute(log_startprob, log_transmat_t, log_frameprob, out, n_components,
            n_samples);
  }

 protected:
  static void Compute(const float *log_startprob, const float *log_transmat_t,
                      const float *log_frameprob, int32_t *state_sequence,
                      int64_t n_components, int64_t n_samples) {
    const int64_t viterbi_lattice_size = n_samples * n_components;
    const int64_t work_buffer_size = n_components;
    const int64_t workspace_size = viterbi_lattice_size + work_buffer_size;
    float *workspace = new float[workspace_size];
    float *viterbi_lattice = workspace;
    float *work_buffer = workspace + viterbi_lattice_size;
    DoCompute(log_startprob, log_transmat_t, log_frameprob, state_sequence,
              n_components, n_samples, viterbi_lattice, work_buffer);
    delete[] workspace;
  }

  // Compute estimated sequence using viterbi algorithm
  static void DoCompute(const float *log_startprob, const float *log_transmat_t,
                        const float *log_frameprob, int32_t *state_sequence,
                        int64_t n_components, int64_t n_samples,
                        float *viterbi_lattice, float *work_buffer) {
    // Initial lattice
    for (int64_t i = 0; i < n_components; i++) {
      viterbi_lattice[i] = log_startprob[i] + log_frameprob[i];
    }

    // Induction
    float *row_viterbi_lattice = viterbi_lattice;
    float *next_row_viterbi_lattice = row_viterbi_lattice + n_components;
    const float *row_log_frameprob = log_frameprob + n_components;
    for (int64_t t = 1; t < n_samples; t++) {
      const float *row_log_transmat_t = log_transmat_t;
      for (int64_t i = 0; i < n_components; i++) {
        for (int64_t j = 0; j < n_components; j++) {
          work_buffer[j] = row_log_transmat_t[j] + row_viterbi_lattice[j];
        }
        next_row_viterbi_lattice[i] =
            work_buffer[ArgMax(work_buffer, n_components)] +
            row_log_frameprob[i];
        row_log_transmat_t += n_components;
      }
      row_viterbi_lattice = next_row_viterbi_lattice;
      next_row_viterbi_lattice += n_components;
      row_log_frameprob += n_components;
    }

    // Observation traceback
    size_t where_from = ArgMax(row_viterbi_lattice, n_components);
    state_sequence[n_samples - 1] = where_from;
    row_viterbi_lattice -= n_components;
    for (int64_t t = n_samples - 2; t >= 0; t--) {
      const float *row_log_transmat_t =
          &log_transmat_t[where_from * n_components];
      for (int64_t i = 0; i < n_components; i++) {
        work_buffer[i] = row_viterbi_lattice[i] + row_log_transmat_t[i];
      }
      where_from = ArgMax(work_buffer, n_components);
      state_sequence[t] = where_from;
      row_viterbi_lattice -= n_components;
    }
  }

  static size_t ArgMax(const float *start, size_t n_elements) {
    return (std::max_element(start, start + n_elements) - start);
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

class CustomOpViterbi
    : public Ort::CustomOpBase<CustomOpViterbi, KernelViterbi> {
 public:
  void *CreateKernel(OrtApi api, const OrtKernelInfo * /* info */) const {
    return new KernelViterbi(api);
  }

  const char *GetName() const { return "Viterbi"; }

  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
} custom_op_viterbi;

}  // namespace

OrtCustomOp *CreateViterbiOp() { return &custom_op_viterbi; }

extern "C" OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                     const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain("ml.hmm", &domain)) {
    return status;
  }
  OrtCustomOp *op = CreateViterbiOp();
  if (auto status = ortApi->CustomOpDomain_Add(domain, op)) {
    return status;
  }
  return ortApi->AddCustomOpDomain(options, domain);
}
