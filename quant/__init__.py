# quant package — Scale Metadata Quantization (SMQ) reference implementation
from quant.shared_scale_quant import (
    SharedScaleLinear,
    dequantize_weights,
    quant_error,
    quantize_scale,
    quantize_weights,
)

__all__ = [
    "quantize_scale",
    "quantize_weights",
    "dequantize_weights",
    "quant_error",
    "SharedScaleLinear",
]
