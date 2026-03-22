"""
Memory Audit — Empirical GPU Memory vs Theoretical Savings
===========================================================

Loads the model in BF16, records torch.cuda.memory_allocated().
Then applies SMQ quantization for each scale_mbits in the configured list
and records memory after each quantization.
Also applies bitsandbytes NF4 quantization and records its memory.

Outputs a CSV and prints a formatted table showing theoretical vs empirical
memory savings for each configuration.

Output CSV columns:
    config, scale_mbits, model_mem_mb, theoretical_saving_pct

Usage:
    python analysis/memory_audit.py \\
        --model microsoft/Phi-3.5-mini-instruct \\
        --group_size 128 \\
        --out_csv results/aggregate/memory_audit.csv \\
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quant.shared_scale_quant import SharedScaleLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_mem_mb() -> Optional[float]:
    """Current GPU memory allocated in MB."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _is_target_linear(name: str) -> bool:
    """Target all nn.Linear layers (MLP + attention) for the audit."""
    return True  # audit all layers for comprehensive measurement


def _replace_all_linear(model: nn.Module, group_size: int, scale_mbits: int) -> int:
    """Replace compatible nn.Linear layers with SharedScaleLinear."""
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            in_f = child.in_features
            if in_f < group_size or in_f % group_size != 0:
                continue  # skip incompatible shapes
            setattr(
                module,
                child_name,
                SharedScaleLinear.from_linear(child, group_size, scale_mbits),
            )
            replaced += 1
    return replaced


def _theoretical_weight_mb(model: nn.Module) -> float:
    """Sum of all parameter storage in MB (BF16 = 2 bytes/param)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)


def _theoretical_smq_saving_pct(
    model: nn.Module,
    group_size: int,
    scale_mbits: int,
) -> float:
    """Estimate theoretical memory saving from SMQ vs BF16.

    SMQ replaces BF16 weights with int4 (2x compression on weights) plus
    compressed per-group scales.  Returns percentage reduction vs BF16.
    """
    bf16_bytes = 0
    smq_bytes = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        out_f = module.out_features
        in_f = module.in_features
        if in_f < group_size or in_f % group_size != 0:
            continue
        # BF16: 2 bytes per weight
        bf16_bytes += out_f * in_f * 2
        # SMQ: 0.5 bytes per weight (int4) + scale bytes
        n_groups = (out_f * in_f) // group_size
        scale_bits = (6 + max(scale_mbits, 0)) if scale_mbits >= 0 else 32
        smq_bytes += (out_f * in_f) // 2 + (n_groups * scale_bits) // 8

    if bf16_bytes == 0:
        return 0.0
    return (1.0 - smq_bytes / bf16_bytes) * 100.0


# ---------------------------------------------------------------------------
# Main audit routine
# ---------------------------------------------------------------------------

def run_audit(
    model_id: str,
    scale_mbits_list: List[int],
    group_size: int,
    device: str,
) -> List[dict]:
    """Run the full memory audit and return a list of measurement rows."""
    from transformers import AutoModelForCausalLM

    rows = []

    # ---- BF16 baseline ----
    print(f"\n[1/3] Loading {model_id} in BF16 ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_bf16.eval().to(device)
    bf16_mem = _gpu_mem_mb()
    print(f"  BF16 model memory: {bf16_mem:.1f} MB")
    rows.append({
        "config": "BF16 baseline",
        "scale_mbits": -999,
        "model_mem_mb": bf16_mem,
        "theoretical_saving_pct": 0.0,
    })

    # ---- SMQ variants ----
    print(f"\n[2/3] Running SMQ audit for scale_mbits={scale_mbits_list} ...")
    for scale_mbits in scale_mbits_list:
        # Load fresh BF16 model for each variant
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_smq = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model_smq.eval()

        # Compute theoretical saving before quantizing (use BF16 params)
        theoretical_saving = _theoretical_smq_saving_pct(model_smq, group_size, scale_mbits)

        # Apply SMQ quantization
        n_replaced = _replace_all_linear(model_smq, group_size, scale_mbits)
        model_smq.to(device)
        smq_mem = _gpu_mem_mb()

        label = f"SMQ scale_mbits={scale_mbits}" if scale_mbits >= 0 else "SMQ exact (FP32 scales)"
        print(f"  {label}: {smq_mem:.1f} MB  (replaced {n_replaced} layers)  theoretical saving: {theoretical_saving:.1f}%")

        rows.append({
            "config": label,
            "scale_mbits": scale_mbits,
            "model_mem_mb": smq_mem,
            "theoretical_saving_pct": round(theoretical_saving, 2),
        })

        del model_smq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- bitsandbytes NF4 ----
    print(f"\n[3/3] Running bitsandbytes NF4 audit ...")
    try:
        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_bnb = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        model_bnb.eval()
        bnb_mem = _gpu_mem_mb()
        print(f"  bitsandbytes NF4 (double-quant): {bnb_mem:.1f} MB")
        rows.append({
            "config": "bitsandbytes NF4 double-quant (C6)",
            "scale_mbits": -998,
            "model_mem_mb": bnb_mem,
            "theoretical_saving_pct": None,
        })
        del model_bnb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  bitsandbytes NF4 skipped: {e}")
        rows.append({
            "config": "bitsandbytes NF4 double-quant (C6)",
            "scale_mbits": -998,
            "model_mem_mb": None,
            "theoretical_saving_pct": None,
        })

    del model_bf16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def _print_table(rows: List[dict], bf16_mem: Optional[float]) -> None:
    """Print a formatted ASCII table of results."""
    print("\n" + "=" * 72)
    print(f"{'Config':<40} {'Mem (MB)':>10} {'Empirical Δ':>12} {'Theoretical Δ':>14}")
    print("-" * 72)
    for row in rows:
        mem = row["model_mem_mb"]
        mem_str = f"{mem:.1f}" if mem is not None else "N/A"
        if mem is not None and bf16_mem is not None and row["config"] != "BF16 baseline":
            empirical_pct = (1.0 - mem / bf16_mem) * 100.0
            empirical_str = f"{empirical_pct:+.1f}%"
        else:
            empirical_str = "—"
        theor = row["theoretical_saving_pct"]
        theor_str = f"{theor:.1f}%" if theor is not None else "N/A"
        print(f"{row['config']:<40} {mem_str:>10} {empirical_str:>12} {theor_str:>14}")
    print("=" * 72)
    print("Note: Negative Δ = memory reduction vs BF16 baseline.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Memory audit: empirical vs theoretical GPU memory for BF16, SMQ, NF4"
    )
    ap.add_argument("--model", required=True, help="HuggingFace model ID")
    ap.add_argument("--group_size", type=int, default=128, help="SMQ group size")
    ap.add_argument(
        "--scale_mbits_list",
        type=int,
        nargs="+",
        default=[-1, 0, 3, 5],
        help="SMQ scale_mbits values to audit (default: -1 0 3 5)",
    )
    ap.add_argument(
        "--out_csv",
        default="results/aggregate/memory_audit.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("Warning: CUDA not available. Memory audit will report None for GPU memory.")

    rows = run_audit(
        model_id=args.model,
        scale_mbits_list=args.scale_mbits_list,
        group_size=args.group_size,
        device=args.device,
    )

    # Print summary table
    bf16_row = next((r for r in rows if r["config"] == "BF16 baseline"), None)
    bf16_mem = bf16_row["model_mem_mb"] if bf16_row else None
    _print_table(rows, bf16_mem)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["config", "scale_mbits", "model_mem_mb", "theoretical_saving_pct"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten: {args.out_csv}")


if __name__ == "__main__":
    main()
