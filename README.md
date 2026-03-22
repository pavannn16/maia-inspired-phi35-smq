# Maia-Inspired Mixed Precision — Scale Metadata Quantization (SMQ)

This repo studies **Maia-inspired mixed precision** using an **A100 proxy**: **W4A8** (4-bit weights + 8-bit activations) plus a **custom shared-scale metadata path**.

Key honesty statement: **A100 ≠ Maia-200**. This project frames W4A8 + shared-scale as an **emulation/proxy** for the Maia-style idea, validated on A100 BF16 paths via software simulation.

## What you get
- Fixed experiment matrix **C0, C2–C7** (configs + workloads + metrics)
- Reproducible artifact logging: raw JSONL + parsed CSV
- Baselines: BF16, W4A16 (bitsandbytes NF4), bitsandbytes double-quant (C6)
- Novelty: W4 + custom shared-scale format (SMQ) — MLP-only (C3/C4/C5) and full attention+MLP (C7)
- Cross-model validation: Phi-3.5-mini-instruct + Llama-3.2-1B
- Analysis scripts: per-layer sensitivity, Pareto plot, memory audit

## Quick start (C0 BF16 offline benchmark)
1. Install deps:
   - `pip install -r requirements.txt`
2. Run offline sweep (prompt lengths {128,512,2048}, output lengths {64,256}):
   - `python -m bench.offline_bench --config configs/experiment_matrix.yaml --model microsoft/Phi-3.5-mini-instruct --config_id C0 --repeats 1`

## Analysis scripts
```bash
# Per-layer quantization sensitivity
python analysis/per_layer_sensitivity.py --model microsoft/Phi-3.5-mini-instruct --out_csv results/aggregate/layer_sensitivity.csv

# Memory audit (BF16 vs SMQ vs NF4)
python analysis/memory_audit.py --model microsoft/Phi-3.5-mini-instruct --out_csv results/aggregate/memory_audit.csv

# Pareto plot (after running benchmarks + lm-eval)
python analysis/pareto_plot.py \
    --agg_csv results/aggregate/paper_table.csv \
    --lmeval_json results/lm_eval/C0_completion.json \
    --out_png results/figures/pareto.png
```

Artifacts:
- `results/raw/*.jsonl` (per-request timing + env)
- `results/summary/*.csv` (one row per run)
- `results/aggregate/*.csv` (CI-aggregated tables)
- `results/figures/*.png` (plots)

## Repro contract
- **Do not edit** the experiment matrix after publishing results.
- Lock prompt formatting/chat template and decoding params.

See `ENVIRONMENT.md` and `configs/experiment_matrix.yaml`.
