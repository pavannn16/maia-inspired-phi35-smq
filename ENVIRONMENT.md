# Environment + Reproducibility

This project targets **Google Colab A100**.

## Hardware Specifications (A100 80 GB SXM)

| Spec | Value |
|------|-------|
| GPU | NVIDIA A100 80 GB SXM |
| Compute capability | sm_80 |
| BF16 Tensor Core throughput | 312 TFLOPS |
| HBM2e bandwidth | 2 TB/s |
| Memory | 80 GB HBM2e |
| NVLink | 600 GB/s (bidirectional) |

Note: **sm_80 ≠ sm_90** (Hopper). FP8 via TransformerEngine (which requires Hopper/sm_90) is not available on A100. This project uses BF16 and W4 paths which are fully supported on sm_80.

## What to record for every run
- GPU: name + total memory
- Driver + CUDA version
- PyTorch version
- Transformers version
- vLLM version (if used)
- Model ID + revision hash
- Exact config hash (C0, C2–C7)

The runners emit an environment snapshot into each JSONL record.

## Version pinning workflow (recommended)
1. Start with `requirements.txt` minimum versions.
2. When a config is validated end-to-end, freeze exact versions into a `requirements.lock.txt` (or paste into the report appendix).

## Colab notes
- Colab images change; expect occasional wheel incompatibilities.
- Prefer pip wheels over source builds. Keep CUDA extension builds minimal.
- CUDA extension in `cuda/shared_scale_dequant/` targets `-arch=sm_80` for A100.
- Compile with: `cd cuda/shared_scale_dequant && pip install -e . --no-build-isolation`
