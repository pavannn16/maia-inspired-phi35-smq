"""
Offline Benchmark Harness
=========================

Drives runtime/torch_runner.py as a subprocess — one subprocess per repeat
so the model is reloaded fresh each repeat (independent samples for CI).
Within each repeat, torch_runner loads the model ONCE and sweeps all
(prompt_len, output_len) combinations in-process (no repeated model loads).

Quant mode and scale_mbits are read from the YAML quant section, so
experiment_matrix.yaml is the single source of truth.

For C5 (scale_mbits_sweep), the harness iterates over each scale_mbits value
and runs it as a sub-config C5_m0, C5_m3, C5_m5.

C6 (w4a16_bnb_dq): bitsandbytes double-quant — direct SMQ comparison baseline.
C7 (w4_shared_scale, quant_target=all): attention + MLP quantization.

Multi-model: if experiment_matrix.yaml has a `models` list, the --model
argument selects the primary model; pass --model2 to also run the second
model in the same subprocess for cross-model validation.

Security note: all subprocess arguments are passed as a list, never as a
shell string, to prevent any form of command injection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _quant_args(quant: Dict[str, Any], scale_mbits_override: Optional[int] = None) -> List[str]:
    """Build torch_runner quant CLI args from YAML quant block."""
    mode = quant.get("mode", "none")
    scale_mbits = scale_mbits_override if scale_mbits_override is not None else quant.get("scale_mbits", -1)
    group_size = quant.get("group_size", 128)
    quant_target = quant.get("quant_target", "mlp")
    return [
        "--quant_mode", mode,
        "--scale_mbits", str(scale_mbits),
        "--group_size", str(group_size),
        "--quant_target", quant_target,
    ]


def _run_one_repeat(
    model: str,
    prompt_lengths: List[int],
    output_lengths: List[int],
    quant_args: List[str],
    raw_path: str,
    run_id_prefix: str,
    warmup_runs: int,
    model2: Optional[str] = None,
) -> None:
    """Call torch_runner (subprocess list form, all combos in one shot)."""
    cmd = (
        [sys.executable, "-m", "runtime.torch_runner"]
        + ["--model", model]
        + (["--model2", model2] if model2 else [])
        + ["--prompt_lens"] + [str(p) for p in prompt_lengths]
        + ["--output_lens"] + [str(o) for o in output_lengths]
        + quant_args
        + ["--warmup_runs", str(warmup_runs)]
        + ["--out_jsonl", raw_path]
        + ["--run_id_prefix", run_id_prefix]
    )
    subprocess.run(cmd, check=True)


def _parse_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _to_summary_row(rec: Dict[str, Any], config_id: str, repeat_idx: int) -> Dict[str, Any]:
    res = rec["result"]
    return {
        "config_id": config_id,
        "repeat": repeat_idx,
        "subrun_id": rec["run_id"],
        "model": res.get("model", ""),
        "quant_mode": res.get("quant_mode", ""),
        "scale_mbits": res.get("scale_mbits", -1),
        "prompt_len": res["prompt_len"],
        "output_len": res["output_len"],
        "ttft_ms": res["ttft_ms"],
        "tpot_ms": res["tpot_ms"],
        "throughput_total_tok_s": res["throughput_total_tok_s"],
        "throughput_output_tok_s": res["throughput_output_tok_s"],
        "model_mem_mb": res.get("model_mem_mb"),
        "peak_gpu_mem_mb": res["peak_gpu_mem_mb"],
        "joules_per_token": res.get("joules_per_token"),
    }


def _aggregate_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (config_id, model, scale_mbits, prompt_len, output_len) → mean ± CI95."""
    from scipy import stats  # type: ignore

    group_keys = ["config_id", "model", "scale_mbits", "prompt_len", "output_len"]
    metrics = [
        "ttft_ms", "tpot_ms", "throughput_output_tok_s",
        "model_mem_mb", "peak_gpu_mem_mb", "joules_per_token",
    ]
    rows = []
    # Drop group columns that may be missing
    actual_keys = [k for k in group_keys if k in df.columns]
    for keys, g in df.groupby(actual_keys):
        row = dict(zip(actual_keys, keys if len(actual_keys) > 1 else [keys]))
        for m in metrics:
            vals = g[m].dropna().values if m in g.columns else []
            if len(vals) == 0:
                row[f"{m}_mean"] = None
                row[f"{m}_ci95"] = None
            elif len(vals) == 1:
                row[f"{m}_mean"] = float(vals[0])
                row[f"{m}_ci95"] = None
            else:
                mean = float(vals.mean())
                sem = float(stats.sem(vals))
                ci95 = float(stats.t.ppf(0.975, df=len(vals) - 1) * sem)
                row[f"{m}_mean"] = mean
                row[f"{m}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def run_config(
    cfg: Dict[str, Any],
    config_id: str,
    model: str,
    repeats: int,
    out_dir: str,
    model2: Optional[str] = None,
) -> None:
    conf = cfg["configs"][config_id]
    wl = cfg["workloads"]["offline_batch_decode"]
    quant = conf.get("quant", {})
    warmup_runs = cfg.get("fairness_constraints", {}).get("warmup_runs", 1)

    raw_dir = os.path.join(out_dir, "raw")
    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # C5 has a scale_mbits_sweep — run as separate sub-configs
    mbits_list: Optional[List[int]] = quant.get("scale_mbits_sweep")
    if mbits_list:
        for mb in mbits_list:
            sub_id = f"{config_id}_m{mb}"
            run_config_with_mbits(
                cfg, config_id, sub_id, mb, model, repeats, out_dir, wl, quant, warmup_runs,
                model2=model2,
            )
    else:
        run_config_with_mbits(
            cfg, config_id, config_id, None, model, repeats, out_dir, wl, quant, warmup_runs,
            model2=model2,
        )


def run_config_with_mbits(
    cfg: Dict[str, Any],
    config_id: str,
    sub_config_id: str,
    scale_mbits_override: Optional[int],
    model: str,
    repeats: int,
    out_dir: str,
    wl: Dict[str, Any],
    quant: Dict[str, Any],
    warmup_runs: int,
    model2: Optional[str] = None,
) -> None:
    raw_dir = os.path.join(out_dir, "raw")
    summary_dir = os.path.join(out_dir, "summary")
    q_args = _quant_args(quant, scale_mbits_override)
    token = uuid.uuid4().hex[:8]
    all_rows: List[Dict[str, Any]] = []

    for repeat_idx in range(repeats):
        run_id_prefix = f"{sub_config_id}-r{repeat_idx}-{token}"
        raw_path = os.path.join(raw_dir, f"{run_id_prefix}.jsonl")
        print(f"\n[{sub_config_id}] repeat {repeat_idx + 1}/{repeats} ...")
        _run_one_repeat(
            model=model,
            prompt_lengths=wl["prompt_lengths"],
            output_lengths=wl["output_lengths"],
            quant_args=q_args,
            raw_path=raw_path,
            run_id_prefix=run_id_prefix,
            warmup_runs=warmup_runs,
            model2=model2,
        )
        recs = _parse_jsonl(raw_path)
        for rec in recs:
            all_rows.append(_to_summary_row(rec, sub_config_id, repeat_idx))

    df = pd.DataFrame(all_rows)
    flat_csv = os.path.join(summary_dir, f"{sub_config_id}_{token}_flat.csv")
    df.to_csv(flat_csv, index=False)
    print(f"Flat summary: {flat_csv}")

    try:
        agg_df = _aggregate_ci(df)
        agg_csv = os.path.join(summary_dir, f"{sub_config_id}_{token}_agg.csv")
        agg_df.to_csv(agg_csv, index=False)
        print(f"Aggregated CI: {agg_csv}")
    except ImportError:
        print("scipy not installed — skipping CI aggregation. Run: pip install scipy")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline ablation benchmark harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--config_id", required=True,
                    help="e.g. C0, C4, C6 (C5 auto-sweeps mbits), C7")
    ap.add_argument("--model", required=True, help="Primary model HuggingFace ID")
    ap.add_argument("--model2", default=None,
                    help="Optional second model for cross-model validation")
    ap.add_argument("--repeats", type=int, default=None,
                    help="Override YAML repeats if specified")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    repeats = args.repeats or cfg["workloads"]["offline_batch_decode"].get("repeats", 3)
    run_config(cfg, args.config_id, args.model, repeats, args.out_dir, model2=args.model2)


if __name__ == "__main__":
    main()
