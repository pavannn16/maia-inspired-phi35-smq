"""
Per-Layer Quantization Sensitivity Analysis
============================================

Loads the model in BF16, iterates over all Linear layers (MLP + attention),
and for each layer runs quantize_weights + dequantize_weights + quant_error
across a configurable list of scale_mbits values.

Output CSV columns:
    layer_name, scale_mbits, rel_mse, max_rel_mse, cosine_sim

Usage:
    python analysis/per_layer_sensitivity.py \\
        --model microsoft/Phi-3.5-mini-instruct \\
        --scale_mbits_list -1 0 3 5 \\
        --out_csv results/aggregate/layer_sensitivity.csv \\
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quant.shared_scale_quant import quantize_weights, dequantize_weights, quant_error


def _iter_linear_layers(model: nn.Module):
    """Yield (full_name, module) for every nn.Linear in the model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def analyze_model(
    model_id: str,
    scale_mbits_list: List[int],
    group_size: int,
    device: str,
) -> List[dict]:
    """Load model and compute per-layer quantization error for each scale_mbits.

    Args:
        model_id:         HuggingFace model ID.
        scale_mbits_list: List of mantissa bit counts to test.
        group_size:       SMQ group size (default 128).
        device:           'cuda' or 'cpu'.

    Returns:
        List of dicts with keys: layer_name, scale_mbits, rel_mse, max_rel_mse, cosine_sim.
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_id} in BF16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    # Keep on CPU to avoid OOM on large models; move individual tensors to device as needed
    rows = []

    linear_layers = list(_iter_linear_layers(model))
    print(f"Found {len(linear_layers)} Linear layers. Running sensitivity analysis ...")

    for layer_idx, (layer_name, linear_module) in enumerate(linear_layers):
        w = linear_module.weight.data.float()

        # Skip layers that are too small for group_size (e.g. vocab embedding heads)
        out_f, in_f = w.shape
        if in_f < group_size or in_f % group_size != 0:
            print(f"  Skipping {layer_name} (in_features={in_f} not divisible by group_size={group_size})")
            continue

        if layer_idx % 20 == 0:
            print(f"  Layer {layer_idx}/{len(linear_layers)}: {layer_name}")

        for mbits in scale_mbits_list:
            packed, scales = quantize_weights(w, group_size, scale_mbits=mbits)
            w_hat = dequantize_weights(packed, scales, group_size)
            err = quant_error(w, w_hat, group_size)
            rows.append({
                "layer_name": layer_name,
                "scale_mbits": mbits,
                "rel_mse": err["rel_mse"],
                "max_rel_mse": err["max_rel_mse"],
                "cosine_sim": err["cosine_sim"],
            })

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-layer SMQ sensitivity: quant error vs scale_mbits for every Linear layer"
    )
    ap.add_argument("--model", required=True, help="HuggingFace model ID")
    ap.add_argument(
        "--scale_mbits_list",
        type=int,
        nargs="+",
        default=[-1, 0, 3, 5],
        help="Mantissa bit counts to test (default: -1 0 3 5)",
    )
    ap.add_argument("--group_size", type=int, default=128, help="SMQ group size")
    ap.add_argument(
        "--out_csv",
        default="results/aggregate/layer_sensitivity.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    args = ap.parse_args()

    rows = analyze_model(
        model_id=args.model,
        scale_mbits_list=args.scale_mbits_list,
        group_size=args.group_size,
        device=args.device,
    )

    if not rows:
        print("No rows collected — check that the model has compatible Linear layers.")
        return

    import csv

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["layer_name", "scale_mbits", "rel_mse", "max_rel_mse", "cosine_sim"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten {len(rows)} rows to {args.out_csv}")
    print(f"Layers analyzed: {len(set(r['layer_name'] for r in rows))}")
    print(f"scale_mbits tested: {sorted(set(r['scale_mbits'] for r in rows))}")


if __name__ == "__main__":
    main()
