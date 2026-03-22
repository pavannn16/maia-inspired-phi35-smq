#include <torch/extension.h>

// Declared in shared_scale_dequant_cuda.cu
torch::Tensor dequant_int4_shared_scale(
    torch::Tensor packed_w,
    torch::Tensor scales,
    int64_t group_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "dequant_int4_shared_scale",
    &dequant_int4_shared_scale,
    "Shared-scale symmetric int4 dequant kernel (CUDA).\n"
    "Args: packed_w uint8[rows, cols/2], scales fp16/bf16[rows, cols/group_size], group_size int.\n"
    "Returns: fp16/bf16[rows, cols] dequantized weights."
  );
}
