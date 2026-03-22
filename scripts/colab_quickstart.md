# Colab Step-by-Step Workflow (A100)

All commands should be run from the repo root.
Expected runtime on a Colab A100: ~75–120 min for a full C0+C4+C6+C7 ablation run.

---

## Phase 0 — Environment setup (run once)

```bash
# Pull latest code
git pull

# Install deps
pip install -r requirements.txt

# Optional: compile CUDA extension (sm_80 = A100)
cd cuda/shared_scale_dequant
pip install -e . --no-build-isolation
cd ../..

# Sanity-check CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA A100-SXM4-80GB (or similar A100 variant)
```

---

## Phase 1 — Unit tests (run before any benchmarking)

```bash
python -m pytest quant/tests/test_shared_scale.py -v
```

All tests must pass before proceeding.  If any fail, open an issue.

---

## Phase 2 — Offline latency benchmarks

Run one `config_id` at a time.  Each command:
- Loads model ONCE per repeat
- Sweeps all (prompt_len, output_len) combos in-process
- Writes raw JSONL + summary CSV to `results/`

```bash
MODEL=microsoft/Phi-3.5-mini-instruct
MODEL2=meta-llama/Llama-3.2-1B
CFG=configs/experiment_matrix.yaml

# C0 — BF16 baseline (~5 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C0

# C2 — W4 NF4 via bitsandbytes (~6 min, requires bitsandbytes)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C2

# C3 — W4 SMQ exact-scale reference (~6 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C3

# C4 — W4 SMQ scale_mbits=5 (proposed method) (~6 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C4

# C5 — SMQ ablation sweep (runs scale_mbits=0,3,5 automatically) (~15 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C5

# C6 — bitsandbytes double-quant (direct SMQ comparison baseline) (~6 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C6

# C7 — W4 + SMQ all layers (attention + MLP) (~6 min)
python -m bench.offline_bench --config $CFG --model $MODEL --config_id C7
```

With `repeats: 3` (from YAML), each command runs the full sweep 3 times for CI.

For a quick smoke-run use `--repeats 1`.

### Cross-model validation (optional)

Add `--model2` to also run Llama-3.2-1B in the same subprocess:

```bash
python -m bench.offline_bench --config $CFG --model $MODEL --model2 $MODEL2 --config_id C4
```

---

## Phase 3 — Quality evaluation (lm-eval)

Run hellaswag, arc_challenge, gsm8k, lambada_openai.

```bash
# C0 BF16 baseline
python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C0

# C2 W4-NF4 quality check
python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C2

# C6 bitsandbytes double-quant quality check
python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C6

# C4 SMQ quality check (note: uses custom SharedScaleLinear; lm_eval invokes
# the HF model, which does not apply SMQ automatically. Only perf ablation is
# meaningful here until a custom lm_eval ModelWrapper is implemented.)
# python -m eval.lm_eval_runner --config $CFG --model $MODEL --config_id C4
```

---

## Phase 4 — Analysis scripts

```bash
# Per-layer quantization sensitivity (outputs layer_sensitivity.csv)
python analysis/per_layer_sensitivity.py \
    --model $MODEL \
    --scale_mbits_list -1 0 3 5 \
    --out_csv results/aggregate/layer_sensitivity.csv

# Memory audit (empirical vs theoretical memory savings)
python analysis/memory_audit.py \
    --model $MODEL \
    --out_csv results/aggregate/memory_audit.csv

# Pareto plot (requires lm-eval JSON + aggregate CSV)
python scripts/aggregate_results.py \
    --summary_dir results/summary \
    --config_ids C0 C2 C3 C4 C5_m0 C5_m3 C5_m5 C6 C7 \
    --out_csv results/aggregate/paper_table.csv

python analysis/pareto_plot.py \
    --agg_csv results/aggregate/paper_table.csv \
    --lmeval_json results/lm_eval/C0_completion.json \
    --out_png results/figures/pareto.png
```

---

## Phase 5 — Push results to GitHub

```bash
# Stage everything under results/
git add results/ --force
git commit -m "Add benchmark results: $(date +%Y-%m-%d)"
git push
```

Then on your local machine:
```bash
git pull
```

---

## Interpreting results

| Config | What it measures |
|--------|------------------|
| C0     | BF16 baseline (gold reference) |
| C2     | Industry-standard W4 (bitsandbytes NF4) |
| C3     | W4 + exact per-group scales (our arch, no scale compression) |
| C4     | W4 + E5M5 scales (proposed SMQ method) |
| C5     | Ablation sweep of scale precision (scale_mbits=0,3,5) |
| C6     | bitsandbytes double-quant (direct SMQ comparison baseline) |
| C7     | W4 + SMQ E5M5, all layers (attention + MLP) |

Key claims to verify:
1. C3 ≈ C4 in throughput (scale compression adds no latency)
2. C4 ≈ C3 in quality (E5M5 scale quantization is nearly lossless)
3. C5 shows quality degrades as scale_mbits decreases
4. C4 vs C6: SMQ E5Mx framework vs bitsandbytes block-scale compression
5. C7 vs C4: cost of quantizing attention layers in addition to MLP

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: runtime` | Run `python -m bench.offline_bench ...` not `python bench/offline_bench.py ...` |
| `RuntimeError: CUDA out of memory` | Reduce `--repeats 1`; close other processes |
| `bitsandbytes not installed` | Run `pip install bitsandbytes`; skip C2/C6 if wheel unavailable |
| `pynvml_error` in results | Power sampling unavailable; joules_per_token will be null (non-critical) |
| CUDA extension compile error | Ensure CUDA toolkit matches PyTorch; try without `--no-build-isolation` |
| `matplotlib not installed` | Run `pip install matplotlib>=3.7.0` for analysis/pareto_plot.py |
