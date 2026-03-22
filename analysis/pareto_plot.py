"""
Pareto Plot — Scale Bits vs Perplexity (lambada_openai)
========================================================

Reads the aggregate results CSV (from scripts/aggregate_results.py) and
the lm-eval results JSON, and produces a 2-panel figure:

  Panel 1: scale_mbits vs scale storage bytes per layer (theoretical).
  Panel 2: scale_mbits vs perplexity on lambada_openai.

C4 (E5M5, scale_mbits=5) is marked as the Pareto-optimal point.

Saves to results/figures/pareto.png.
Uses matplotlib only (no seaborn dependency).

Usage:
    python analysis/pareto_plot.py \\
        --agg_csv results/aggregate/paper_table.csv \\
        --lmeval_json results/lm_eval/C0_completion.json \\
        --out_png results/figures/pareto.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Scale storage computation (theoretical)
# ---------------------------------------------------------------------------

def _scale_bits(scale_mbits: int) -> int:
    """Total bits per scale value for an E5Mx format.

    scale_mbits=-1 (exact FP32): 32 bits.
    scale_mbits=k >= 0: 1 sign + 5 exp + k mantissa = 6+k bits.
    """
    if scale_mbits < 0:
        return 32
    return 6 + scale_mbits


def _scale_bytes_per_layer(
    out_features: int,
    in_features: int,
    group_size: int,
    scale_mbits: int,
) -> float:
    """Theoretical scale storage in bytes for one weight matrix."""
    n_groups = (out_features * in_features) // group_size
    bits = _scale_bits(scale_mbits)
    return (n_groups * bits) / 8.0


# Phi-3.5 MLP layer dimensions (representative)
_PHI35_MLP_OUT = 3072
_PHI35_MLP_IN = 3072
_GROUP_SIZE = 128

# The ablation scale_mbits values used in this study
_ABLATION_MBITS = [-1, 0, 3, 5]
# Map mbits -> config label
_MBITS_TO_CONFIG = {-1: "C3", 0: "C5-a", 3: "C5-b", 5: "C4"}


# ---------------------------------------------------------------------------
# lm-eval JSON parsing
# ---------------------------------------------------------------------------

def _extract_lambada_perplexity(lmeval_json: str) -> Optional[float]:
    """Extract lambada_openai perplexity from an lm-eval output JSON.

    Returns None if the task/metric is not found.
    """
    try:
        with open(lmeval_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        # lm-eval 0.4+ uses 'lambada_openai' key; perplexity metric
        task = results.get("lambada_openai", {})
        # Try various metric names used by different lm-eval versions
        for key in ["word_perplexity,none", "perplexity,none", "perplexity"]:
            if key in task:
                return float(task[key])
    except Exception as e:
        print(f"[pareto_plot] Warning: could not parse {lmeval_json}: {e}")
    return None


# ---------------------------------------------------------------------------
# Aggregate CSV parsing
# ---------------------------------------------------------------------------

def _load_config_perplexities(agg_csv: str) -> Dict[str, float]:
    """Return a mapping config_id -> perplexity from aggregate CSV.

    This is a stub: the aggregate CSV contains latency/throughput metrics,
    not perplexity.  Perplexity must come from lm-eval JSON files.
    If multiple lm-eval JSONs are provided, call _extract_lambada_perplexity
    for each.
    """
    return {}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_pareto_plot(
    out_png: str,
    mbits_values: List[int],
    storage_bytes: List[float],
    perplexities: List[Optional[float]],
    pareto_mbits: int = 5,
) -> None:
    """Create and save the 2-panel Pareto figure.

    Args:
        out_png:        Output file path.
        mbits_values:   List of scale_mbits values (x-axis for both panels).
        storage_bytes:  Corresponding theoretical scale storage (bytes/layer).
        perplexities:   Corresponding lambada_openai perplexity (None = missing).
        pareto_mbits:   The Pareto-optimal point (default 5 = C4/E5M5).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "SMQ Scale Metadata Quantization — Pareto Analysis\n"
        "Model: Phi-3.5-mini-instruct | Hardware: A100 80 GB",
        fontsize=13,
        fontweight="bold",
    )

    x_labels = [str(m) if m >= 0 else "exact\n(FP32)" for m in mbits_values]
    colors = ["#e74c3c" if m == pareto_mbits else "#3498db" for m in mbits_values]

    # ---- Panel 1: scale_mbits vs storage bytes per layer ----
    ax1 = axes[0]
    bars = ax1.bar(range(len(mbits_values)), storage_bytes, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(range(len(mbits_values)))
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("scale_mbits", fontsize=11)
    ax1.set_ylabel("Scale storage per MLP layer (KB)", fontsize=11)
    ax1.set_title("Theoretical Scale Metadata Size", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1024:.0f}"))

    # Annotate bars with config labels
    for i, (mbits, val) in enumerate(zip(mbits_values, storage_bytes)):
        label = _MBITS_TO_CONFIG.get(mbits, f"m{mbits}")
        ax1.text(i, val + max(storage_bytes) * 0.01, label,
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ---- Panel 2: scale_mbits vs lambada perplexity ----
    ax2 = axes[1]
    valid = [(i, m, p) for i, (m, p) in enumerate(zip(mbits_values, perplexities)) if p is not None]

    if valid:
        xs, ms, ps = zip(*valid)
        point_colors = ["#e74c3c" if m == pareto_mbits else "#3498db" for m in ms]
        ax2.scatter(xs, ps, c=point_colors, s=120, zorder=5, edgecolors="black", linewidths=0.8)
        ax2.plot(xs, ps, color="#7f8c8d", linewidth=1.5, linestyle="--", zorder=4)

        # Annotate Pareto point
        for x, m, p in zip(xs, ms, ps):
            if m == pareto_mbits:
                ax2.annotate(
                    f"C4 / E5M5\n(Pareto-optimal)\nppl={p:.2f}",
                    xy=(x, p),
                    xytext=(x + 0.3, p - (max(ps) - min(ps)) * 0.15),
                    fontsize=9,
                    color="#e74c3c",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
                )
    else:
        ax2.text(
            0.5, 0.5,
            "No lm-eval perplexity data available.\nRun eval/lm_eval_runner.py first.",
            transform=ax2.transAxes,
            ha="center", va="center",
            fontsize=10, color="gray",
        )

    ax2.set_xticks(range(len(mbits_values)))
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("scale_mbits", fontsize=11)
    ax2.set_ylabel("lambada_openai perplexity (lower = better)", fontsize=11)
    ax2.set_title("Quality vs Scale Precision", fontsize=12)

    # Legend
    pareto_patch = mpatches.Patch(color="#e74c3c", label="Pareto-optimal (C4, E5M5)")
    other_patch = mpatches.Patch(color="#3498db", label="Other configs")
    fig.legend(handles=[pareto_patch, other_patch], loc="lower center",
               ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 2-panel Pareto plot: scale bits vs perplexity"
    )
    ap.add_argument(
        "--agg_csv",
        default="results/aggregate/paper_table.csv",
        help="Aggregate results CSV (from scripts/aggregate_results.py)",
    )
    ap.add_argument(
        "--lmeval_json",
        default=None,
        help="lm-eval results JSON for C0 (or whichever config has lambada_openai results)",
    )
    ap.add_argument(
        "--lmeval_jsons",
        nargs="*",
        default=None,
        help=(
            "Multiple lm-eval JSON paths, one per scale_mbits, in the same order as "
            "--mbits_list. Overrides --lmeval_json."
        ),
    )
    ap.add_argument(
        "--mbits_list",
        type=int,
        nargs="+",
        default=_ABLATION_MBITS,
        help="scale_mbits values for x-axis (default: -1 0 3 5)",
    )
    ap.add_argument(
        "--out_features",
        type=int,
        default=_PHI35_MLP_OUT,
        help="Representative MLP out_features for storage calculation",
    )
    ap.add_argument(
        "--in_features",
        type=int,
        default=_PHI35_MLP_IN,
        help="Representative MLP in_features for storage calculation",
    )
    ap.add_argument(
        "--group_size",
        type=int,
        default=_GROUP_SIZE,
        help="SMQ group size (default 128)",
    )
    ap.add_argument(
        "--out_png",
        default="results/figures/pareto.png",
        help="Output PNG path",
    )
    args = ap.parse_args()

    # Compute theoretical storage for each mbits value
    storage_bytes = [
        _scale_bytes_per_layer(args.out_features, args.in_features, args.group_size, m)
        for m in args.mbits_list
    ]

    # Collect perplexities
    perplexities: List[Optional[float]] = [None] * len(args.mbits_list)

    if args.lmeval_jsons and len(args.lmeval_jsons) == len(args.mbits_list):
        for i, json_path in enumerate(args.lmeval_jsons):
            perplexities[i] = _extract_lambada_perplexity(json_path)
    elif args.lmeval_json:
        # Use a single JSON for all points (typically C0 BF16 reference)
        ppl = _extract_lambada_perplexity(args.lmeval_json)
        if ppl is not None:
            # Place the BF16 reference at exact-scale position (-1)
            for i, m in enumerate(args.mbits_list):
                if m == -1:
                    perplexities[i] = ppl
                    break

    make_pareto_plot(
        out_png=args.out_png,
        mbits_values=args.mbits_list,
        storage_bytes=storage_bytes,
        perplexities=perplexities,
        pareto_mbits=5,
    )


if __name__ == "__main__":
    main()
