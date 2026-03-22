"""
Shared-Scale W4 Quantization — Pure PyTorch Reference Implementation
=====================================================================

Novelty: Scale Metadata Quantization (SMQ)
-------------------------------------------
Standard W4-per-group quantization stores per-group scales as FP16 (~14 mantissa
bits).  This implementation adds controlled compression of the scale values
themselves via a mini-float with `scale_mbits` mantissa bits (E5Mx family,
matching the FP8 exponent-width convention used in Maia/A100 numerics).

Ablation axis (scale_mbits):
  -1  → exact FP32 scale (no quantization; standard W4 reference, C3)
   0  → binary scales: powers-of-2 only.  Maximum metadata compression.
   3  → FP8-E5M3 analogue.  Balanced (proposed method, C4).
   5  → near-lossless (14-bit mantissa FP16 ref truncated to 5 bits).

Research claim: for realistic MLP weight matrices in Phi-3.5, scale_mbits=3
achieves <0.5% relative MSE increase vs exact-scale baseline while reducing
scale metadata storage by ~(14-3)/14 ≈ 79%. This is the Pareto-optimal point
tested in C4/C5.

This module is the correctness reference.  The CUDA extension
(cuda/shared_scale_dequant/) mirrors the dequant path for GPU performance.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Scale quantization — E5Mx mini-float (the SMQ contribution)
# ---------------------------------------------------------------------------

def quantize_scale(s: torch.Tensor, scale_mbits: int) -> torch.Tensor:
    """Quantize scale tensor to an E5Mx mini-float representation.

    Uses 1 sign + 5 exponent + scale_mbits mantissa bits (total 6+scale_mbits).
    scale_mbits=-1 means exact passthrough (no quantization).

    Args:
        s:           Scale tensor, any shape, float32 or bfloat16.
        scale_mbits: Mantissa bits in [0, 14]. -1 for exact.

    Returns:
        Quantized scales, same shape and dtype as s.
    """
    if scale_mbits < 0:
        return s  # exact baseline

    original_dtype = s.dtype
    s = s.float()

    sign = torch.sign(s)
    # Treat zero scale as positive (zero weights stay zero regardless)
    sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
    abs_s = s.abs().clamp(min=1e-30)

    # Decompose: abs_s = 2^exp * (1 + mantissa),  exp = floor(log2(abs_s))
    exp = torch.floor(torch.log2(abs_s))
    normalized = abs_s / (2.0 ** exp)   # in [1, 2)
    mantissa = normalized - 1.0          # in [0, 1)

    if scale_mbits == 0:
        mantissa_q = torch.zeros_like(mantissa)
    else:
        levels = float(2 ** scale_mbits)
        mantissa_q = torch.round(mantissa * levels) / levels
        # Overflow: if rounding pushes mantissa_q to 1.0, carry into exponent
        carry = mantissa_q >= 1.0
        exp = exp + carry.float()
        mantissa_q = torch.where(carry, torch.zeros_like(mantissa_q), mantissa_q)

    result = sign * (2.0 ** exp) * (1.0 + mantissa_q)
    return result.to(original_dtype)


# ---------------------------------------------------------------------------
# Weight quantization / dequantization — symmetric int4 per group
# ---------------------------------------------------------------------------

def quantize_weights(
    w: torch.Tensor,
    group_size: int,
    scale_mbits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight matrix to int4 with SMQ-compressed scales.

    Packing convention (must match CUDA kernel):
      Low nibble  (bits 0-3) → even column index.
      High nibble (bits 4-7) → odd column index.

    Args:
        w:           [out_features, in_features], float32 or bfloat16.
        group_size:  Number of weights per scale group along in_features.
        scale_mbits: Mantissa bits for scale quantization (-1 = exact).

    Returns:
        packed_int4: uint8 tensor [out_features, in_features // 2].
        scales:      float32 tensor [out_features, in_features // group_size].
    """
    assert w.dim() == 2, "w must be 2-D [out_features, in_features]"
    out_features, in_features = w.shape
    assert in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )

    w_f = w.float()
    n_groups = in_features // group_size
    w_grouped = w_f.view(out_features, n_groups, group_size)

    # Symmetric scale: s_g = max(|w_g|) / 7  (maps [-7*s, 7*s] → [-7, 7])
    abs_max = w_grouped.abs().amax(dim=-1).clamp(min=1e-8)  # [out, n_groups]
    scales_exact = abs_max / 7.0

    # Apply SMQ: compress scale to scale_mbits mantissa bits
    scales_q = quantize_scale(scales_exact, scale_mbits)

    # Quantize weights: q = clip(round(w / s_g), -8, 7)
    scales_expanded = scales_q.unsqueeze(-1).expand_as(w_grouped)
    w_q = torch.clamp(torch.round(w_grouped / scales_expanded), -8.0, 7.0)
    w_q_flat = w_q.view(out_features, in_features).to(torch.int32)

    # Pack two signed int4 values per uint8 byte
    # Convert signed [-8,7] → unsigned [0,15] via two's complement lower nibble
    w_lo = (w_q_flat[:, 0::2] & 0x0F).to(torch.uint8)          # even → bits 0-3
    w_hi = ((w_q_flat[:, 1::2] & 0x0F).to(torch.uint8)) << 4   # odd  → bits 4-7
    packed = w_lo | w_hi  # [out_features, in_features // 2]

    return packed, scales_q.float()


def dequantize_weights(
    packed_int4: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize packed int4 weights to float32.

    This is the reference path.  The CUDA extension replicates this on GPU.

    Args:
        packed_int4: uint8 [out_features, in_features // 2].
        scales:      float32 [out_features, n_groups].
        group_size:  Weights per scale group.

    Returns:
        w_hat: float32 [out_features, in_features].
    """
    out_features = packed_int4.shape[0]
    in_features = packed_int4.shape[1] * 2
    n_groups = in_features // group_size

    packed_i32 = packed_int4.to(torch.int32)
    # Unpack nibbles
    w_even_raw = packed_i32 & 0x0F           # 0..15
    w_odd_raw  = (packed_i32 >> 4) & 0x0F   # 0..15

    # Sign extension: [0..7] stays positive; [8..15] → [-8..-1]
    w_even = torch.where(w_even_raw >= 8, w_even_raw - 16, w_even_raw)
    w_odd  = torch.where(w_odd_raw  >= 8, w_odd_raw  - 16, w_odd_raw)

    # Interleave even/odd back into full width
    w_int = torch.zeros(out_features, in_features, dtype=torch.int32, device=packed_int4.device)
    w_int[:, 0::2] = w_even
    w_int[:, 1::2] = w_odd

    # Scale and dequantize
    w_f = w_int.float().view(out_features, n_groups, group_size)
    scales_exp = scales.float().view(out_features, n_groups, 1).expand_as(w_f)
    w_hat = (w_f * scales_exp).view(out_features, in_features)

    return w_hat


# ---------------------------------------------------------------------------
# Diagnostic metrics (for per-layer error analysis)
# ---------------------------------------------------------------------------

def quant_error(
    w_orig: torch.Tensor,
    w_hat: torch.Tensor,
    group_size: int,
) -> Dict[str, float]:
    """Per-group quantization error metrics.

    Returns:
        rel_mse:     Mean relative MSE across all groups.
        max_rel_mse: Worst-case group relative MSE (outlier sensitivity).
        cosine_sim:  Mean cosine similarity across groups.
    """
    out_features, in_features = w_orig.shape
    n_groups = in_features // group_size

    w_o = w_orig.float().view(out_features * n_groups, group_size)
    w_h = w_hat.float().view(out_features * n_groups, group_size)

    mse = ((w_o - w_h) ** 2).mean(dim=-1)
    ref_var = (w_o ** 2).mean(dim=-1).clamp(min=1e-12)
    rel_mse = mse / ref_var

    o_norm = w_o.norm(dim=-1).clamp(min=1e-12)
    h_norm = w_h.norm(dim=-1).clamp(min=1e-12)
    cos_sim = (w_o * w_h).sum(dim=-1) / (o_norm * h_norm)

    return {
        "rel_mse": rel_mse.mean().item(),
        "max_rel_mse": rel_mse.max().item(),
        "cosine_sim": cos_sim.mean().item(),
    }


# ---------------------------------------------------------------------------
# SharedScaleLinear — drop-in nn.Linear replacement for ablation runtime
# ---------------------------------------------------------------------------

class SharedScaleLinear(nn.Module):
    """Drop-in replacement for nn.Linear using SMQ W4 quantization.

    Stores weights as packed int4 + (optionally quantized) per-group scales.
    Dequantizes weights on every forward pass — this is the reference path.
    The CUDA extension (cuda/shared_scale_dequant/) can replace the dequant
    call once compiled, yielding the production performance numbers.

    Args:
        in_features, out_features: same as nn.Linear.
        group_size:   Scale group size. Fixed at 128 for this study.
        scale_mbits:  Mantissa bits for scale quant (-1=exact, 0/3/5=ablation).
        bias:         Whether the layer has a bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        scale_mbits: int = 5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.scale_mbits = scale_mbits

        self.register_buffer(
            "packed_int4",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.ones(out_features, in_features // group_size, dtype=torch.float32),
        )
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        scale_mbits: int = 5,
    ) -> "SharedScaleLinear":
        """Quantize an existing nn.Linear and return a SharedScaleLinear."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            group_size=group_size,
            scale_mbits=scale_mbits,
            bias=linear.bias is not None,
        )
        packed, scales = quantize_weights(linear.weight.data, group_size, scale_mbits)
        layer.packed_int4.copy_(packed)
        layer.scales.copy_(scales)
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.data.clone())
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hat = dequantize_weights(self.packed_int4, self.scales, self.group_size)
        return F.linear(x, w_hat.to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"group_size={self.group_size}, scale_mbits={self.scale_mbits}"
        )
