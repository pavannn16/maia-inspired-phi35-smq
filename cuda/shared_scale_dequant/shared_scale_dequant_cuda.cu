#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
 * Shared-Scale int4 Dequantization Kernel
 * =========================================
 * Mirrors quant/shared_scale_quant.py::dequantize_weights exactly.
 *
 * Packing convention (must match Python reference):
 *   Low  nibble (bits 0-3) -> even column index
 *   High nibble (bits 4-7) -> odd column index
 * Sign extension: unsigned [8,15] -> signed [-8,-1] via subtract-16.
 *
 * Layout:
 *   packed_w : uint8  [rows, cols/2]
 *   scales   : fp16 or bf16 [rows, cols/group_size]
 *   out      : fp16 or bf16 [rows, cols]
 */

template <typename scalar_t>
__global__ void dequant_int4_shared_scale_kernel(
    const uint8_t* __restrict__ packed_w,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    const int rows,
    const int cols,
    const int group_size
) {
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;

    // Unpack int4 from packed uint8
    const int byte_idx = r * (cols / 2) + c / 2;
    const uint8_t byte_val = packed_w[byte_idx];
    int nibble = (c % 2 == 0)
        ? (int)(byte_val & 0x0F)           // low nibble  -> even col
        : (int)((byte_val >> 4) & 0x0F);   // high nibble -> odd col

    // Sign extension: [0,7] positive, [8,15] -> [-8,-1]
    if (nibble >= 8) nibble -= 16;

    // Per-group scale lookup
    const int n_groups_per_row = cols / group_size;
    const int scale_idx = r * n_groups_per_row + c / group_size;
    const float scale = static_cast<float>(scales[scale_idx]);

    out[r * cols + c] = static_cast<scalar_t>(static_cast<float>(nibble) * scale);
}

torch::Tensor dequant_int4_shared_scale(
    torch::Tensor packed_w,
    torch::Tensor scales,
    int64_t group_size
) {
    TORCH_CHECK(packed_w.is_cuda(), "packed_w must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(),   "scales must be a CUDA tensor");
    TORCH_CHECK(packed_w.is_contiguous(), "packed_w must be contiguous");
    TORCH_CHECK(scales.is_contiguous(),   "scales must be contiguous");
    TORCH_CHECK(packed_w.dtype() == torch::kUInt8, "packed_w must be uint8");
    TORCH_CHECK(packed_w.dim() == 2, "packed_w must be 2-D [rows, cols/2]");

    const int rows = static_cast<int>(packed_w.size(0));
    const int cols = static_cast<int>(packed_w.size(1)) * 2;
    const int gs   = static_cast<int>(group_size);

    TORCH_CHECK(cols % gs == 0,
        "cols (", cols, ") must be divisible by group_size (", gs, ")");
    TORCH_CHECK(scales.size(0) == rows && scales.size(1) == cols / gs,
        "scales shape must be [rows, cols/group_size]");

    auto out = torch::empty({rows, cols}, scales.options());

    // 32x8 thread block: 32 columns, 8 rows per block
    const dim3 threads(32, 8);
    const dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scales.scalar_type(), "dequant_int4_shared_scale",
        [&] {
            dequant_int4_shared_scale_kernel<scalar_t><<<blocks, threads>>>(
                packed_w.data_ptr<uint8_t>(),
                scales.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows, cols, gs
            );
        }
    );

    // Propagate any kernel errors
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
