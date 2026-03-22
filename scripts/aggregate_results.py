"""
Aggregate Results — Mean / Std / 95% CI Across Repeats
=======================================================

Reads all summary CSVs from results/summary/ for given config IDs,
groups by (config_id, prompt_len, output_len), and emits a wide table
with mean ± CI for each metric.  This is the input for report tables
and Pareto plots.

Usage:
    python scripts/aggregate_results.py \\
        --summary_dir results/summary \\
        --config_ids C0 C2 C3 C4 \\
        --out_csv results/aggregate/table.csv

Output columns:
    config_id, prompt_len, output_len,
    ttft_ms_mean, ttft_ms_std, ttft_ms_ci95,
    tpot_ms_mean, tpot_ms_std, tpot_ms_ci95,
    throughput_output_tok_s_mean, ...
    model_mem_mb_mean, ...
    peak_gpu_mem_mb_mean, ...
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

METRICS = [
    "ttft_ms",
    "tpot_ms",
    "throughput_total_tok_s",
    "throughput_output_tok_s",
    "model_mem_mb",
    "peak_gpu_mem_mb",
    "joules_per_token",
]


def ci95(values: np.ndarray) -> float:
    """95% confidence interval half-width using t-distribution."""
    n = len(values)
    if n < 2:
        return float("nan")
    try:
        from scipy import stats
        return stats.t.ppf(0.975, df=n - 1) * values.std(ddof=1) / (n ** 0.5)
    except ImportError:
        # Fallback: 1.96 * SEM (large-sample approximation)
        return 1.96 * values.std(ddof=1) / (n ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", default="results/summary")
    ap.add_argument("--config_ids", nargs="+", default=None, help="Filter to these config IDs (default: all)")
    ap.add_argument("--out_csv", default="results/aggregate/table.csv")
    args = ap.parse_args()

    csv_files = glob.glob(os.path.join(args.summary_dir, "*.csv"))
    if not csv_files:
        print(f"No CSVs found in {args.summary_dir}")
        return

    frames: List[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    if args.config_ids:
        all_data = all_data[all_data["config_id"].isin(args.config_ids)]

    group_keys = ["config_id", "prompt_len", "output_len"]
    agg_rows = []

    for keys, group in all_data.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        for col in METRICS:
            if col not in group.columns:
                continue
            vals = group[col].dropna().values.astype(float)
            if len(vals) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                row[f"{col}_ci95"] = float("nan")
            else:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else float("nan")
                row[f"{col}_ci95"] = ci95(vals)
        row["n_repeats"] = len(group)
        agg_rows.append(row)

    result_df = pd.DataFrame(agg_rows)
    result_df = result_df.sort_values(["config_id", "prompt_len", "output_len"])

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    result_df.to_csv(args.out_csv, index=False)
    print(f"Written: {args.out_csv}  ({len(result_df)} rows)")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
