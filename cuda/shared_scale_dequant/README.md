# shared_scale_dequant (CUDA extension)

CUDA extension for shared-scale W4 dequantization, targeting **sm_80 (A100)**.

Exposes:
- `dequant_int4_shared_scale(packed_w, scales, group_size) -> fp16/bf16 weights`

On Colab A100, compile with:
```bash
cd cuda/shared_scale_dequant
pip install -e . --no-build-isolation
```

Keep this extension small to avoid long compile times on Colab.
