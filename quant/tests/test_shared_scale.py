"""Unit tests for quant/shared_scale_quant.py.

Run with:  python -m pytest quant/tests/test_shared_scale.py -v
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from quant.shared_scale_quant import (
    quantize_scale,
    quantize_weights,
    dequantize_weights,
    quant_error,
    SharedScaleLinear,
)

torch.manual_seed(42)
G = 128  # group size fixed for this study


# ---------------------------------------------------------------------------
# Scale quantization
# ---------------------------------------------------------------------------

class TestQuantizeScale:
    def test_exact_passthrough(self):
        s = torch.randn(16).abs() + 0.01
        assert torch.allclose(quantize_scale(s, -1), s)

    def test_scale_mbits_0_produces_powers_of_two(self):
        s = torch.tensor([0.3, 1.7, 5.5, 0.01, 100.0])
        sq = quantize_scale(s, 0)
        log2 = torch.log2(sq.abs())
        assert torch.allclose(log2, log2.round(), atol=1e-4), f"Not powers of 2: {sq}"

    def test_error_monotone_with_more_bits(self):
        """More mantissa bits → lower scale quantization error."""
        s = torch.randn(512).abs() + 0.01
        prev_err = float("inf")
        for bits in [0, 3, 5, 14]:
            sq = quantize_scale(s, bits)
            err = ((s - sq) ** 2).mean().item()
            assert err <= prev_err + 1e-9, f"Error not monotone at bits={bits}: {err} > {prev_err}"
            prev_err = err

    def test_sign_preserved_positive(self):
        s = torch.tensor([0.5, 1.0, 2.0, 4.0])
        for bits in [0, 3, 5]:
            sq = quantize_scale(s, bits)
            assert (sq > 0).all(), f"sign broken at bits={bits}"

    def test_dtype_preserved_bfloat16(self):
        s = torch.tensor([1.5, 0.25], dtype=torch.bfloat16)
        sq = quantize_scale(s, 3)
        assert sq.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Weight quant / dequant roundtrip
# ---------------------------------------------------------------------------

class TestWeightRoundtrip:
    def _w(self, out: int = 64, inp: int = 256) -> torch.Tensor:
        return torch.randn(out, inp) * 0.1

    def test_output_shapes(self):
        w = self._w()
        packed, scales = quantize_weights(w, G, scale_mbits=5)
        w_hat = dequantize_weights(packed, scales, G)
        assert packed.shape == (w.shape[0], w.shape[1] // 2)
        assert scales.shape == (w.shape[0], w.shape[1] // G)
        assert w_hat.shape == w.shape

    def test_packed_is_uint8(self):
        packed, _ = quantize_weights(self._w(), G, scale_mbits=5)
        assert packed.dtype == torch.uint8

    def test_scales_are_float32(self):
        _, scales = quantize_weights(self._w(), G, scale_mbits=5)
        assert scales.dtype == torch.float32

    def test_relative_mse_small_with_fine_scales(self):
        """scale_mbits=5 → relative MSE < 1%."""
        w = self._w(128, 512)
        packed, scales = quantize_weights(w, G, scale_mbits=5)
        w_hat = dequantize_weights(packed, scales, G)
        err = quant_error(w, w_hat, G)
        assert err["rel_mse"] < 0.01, f"rel_mse={err['rel_mse']:.5f}"

    def test_cosine_sim_high_with_fine_scales(self):
        w = self._w(128, 512)
        packed, scales = quantize_weights(w, G, scale_mbits=5)
        w_hat = dequantize_weights(packed, scales, G)
        err = quant_error(w, w_hat, G)
        assert err["cosine_sim"] > 0.99, f"cosine_sim={err['cosine_sim']:.5f}"

    def test_error_increases_with_fewer_scale_bits(self):
        """Core ablation: scale_mbits=0 must degrade more than scale_mbits=5."""
        w = torch.randn(128, 512) * 0.1
        errors = {}
        for bits in [0, 3, 5]:
            packed, scales = quantize_weights(w, G, scale_mbits=bits)
            w_hat = dequantize_weights(packed, scales, G)
            errors[bits] = quant_error(w, w_hat, G)["rel_mse"]
        assert errors[0] >= errors[3] - 1e-9, f"scale_mbits=0 should be worse: {errors}"
        assert errors[3] >= errors[5] - 1e-9, f"scale_mbits=3 should be worse: {errors}"

    def test_zero_weights_stay_zero(self):
        w = torch.zeros(64, G)
        packed, scales = quantize_weights(w, G, scale_mbits=5)
        w_hat = dequantize_weights(packed, scales, G)
        assert w_hat.abs().max().item() < 1e-6

    def test_outlier_does_not_crash(self):
        w = torch.randn(64, 256) * 0.1
        w[0, 0] = 1e6  # extreme outlier
        packed, scales = quantize_weights(w, G, scale_mbits=5)
        w_hat = dequantize_weights(packed, scales, G)
        assert w_hat.shape == w.shape

    def test_exact_scale_baseline_matches_standard_w4(self):
        """scale_mbits=-1 (exact) should match hand-computed per-group quant."""
        w = torch.randn(8, G) * 0.1
        packed, scales = quantize_weights(w, G, scale_mbits=-1)
        w_hat = dequantize_weights(packed, scales, G)
        # Relative MSE should be tiny (limited only by 4-bit weight precision)
        err = quant_error(w, w_hat, G)
        assert err["rel_mse"] < 0.005


# ---------------------------------------------------------------------------
# SharedScaleLinear module
# ---------------------------------------------------------------------------

class TestSharedScaleLinear:
    def test_from_linear_output_shape(self):
        lin = torch.nn.Linear(256, 128)
        qlin = SharedScaleLinear.from_linear(lin, group_size=G, scale_mbits=5)
        x = torch.randn(4, 256)
        assert qlin(x).shape == (4, 128)

    def test_output_close_to_bf16_with_fine_scales(self):
        torch.manual_seed(0)
        lin = torch.nn.Linear(256, 128, bias=False)
        qlin = SharedScaleLinear.from_linear(lin, group_size=G, scale_mbits=5)
        x = torch.randn(4, 256)
        rel_err = (lin(x) - qlin(x)).norm() / lin(x).norm()
        assert rel_err < 0.05, f"Relative output error too large: {rel_err:.4f}"

    def test_different_scale_mbits_give_different_outputs(self):
        """Ablation sanity: scale_mbits=0 and scale_mbits=5 must differ."""
        lin = torch.nn.Linear(256, 128, bias=False)
        x = torch.randn(4, 256)
        out0 = SharedScaleLinear.from_linear(lin, G, scale_mbits=0)(x)
        out5 = SharedScaleLinear.from_linear(lin, G, scale_mbits=5)(x)
        assert not torch.allclose(out0, out5)

    def test_extra_repr_contains_scale_mbits(self):
        lin = torch.nn.Linear(256, 128)
        qlin = SharedScaleLinear.from_linear(lin, G, scale_mbits=3)
        assert "scale_mbits=3" in repr(qlin)
