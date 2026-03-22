# Maia-Inspired Mixed Precision (SMQ) — Research Report

**Project:** maia-inspired-phi35-smq
**Hardware:** Colab A100 80 GB (sm_80, 312 TFLOPS BF16, 2 TB/s HBM2e)
**Models:** microsoft/Phi-3.5-mini-instruct (3.8B), meta-llama/Llama-3.2-1B
**Date:** (fill after runs)

---

## Abstract

(1 paragraph — fill after runs complete)

Research question: how many mantissa bits do per-group quantization scales
actually need before model quality degrades, and what does the answer mean
for LLM inference efficiency on A100-class hardware?

---

## Method

### Config matrix C0, C2–C7

| Config | Name | Key change |
|--------|------|-----------|
| C0 | BF16 baseline | Full precision reference |
| C2 | W4A16 NF4 | bitsandbytes 4-bit NF4 |
| C3 | W4 exact-scale | SMQ arch, scale_mbits=-1 |
| C4 | W4 + E5M5 scales | **Proposed SMQ method** |
| C5 | SMQ sweep | scale_mbits ∈ {0,3,5} |
| C6 | bnb double-quant | 8-bit block scale compression |
| C7 | SMQ all layers | attention + MLP quantization |

### Shared-scale quant format definition

See `quant/shared_scale_spec.md` for the formal E5Mx specification.

### Kernel design

CUDA kernel in `cuda/shared_scale_dequant/` targets sm_80 (A100).
32×8 threadblock; AT_DISPATCH for fp16/bf16.

---

## Experimental Setup

- **Hardware:** Colab A100 80 GB  (record exact GPU SKU from env snapshot)
- **Software:** (fill from requirements.lock.txt after validation)
- **Measurement protocol:** 1 warmup pass (128-tok prefill, 32-tok decode),
  3 timed repeats, 95% CI via t-distribution
- **Prompt lengths:** 128, 512, 2048 tokens
- **Output lengths:** 64, 256 tokens
- **Decoding:** greedy (temperature=0, seed=42)

---

## Results

### Quality (lm-eval, zero-shot)

| Config | hellaswag | arc_challenge | gsm8k | lambada ppl | Avg drop vs C0 |
|--------|-----------|---------------|-------|-------------|----------------|
| C0 BF16 | — | — | — | — | — |
| C2 NF4 | | | | | |
| C4 SMQ E5M5 | | | | | |
| C6 bnb dq | | | | | |

Acceptance target: avg relative drop < 3% vs C0.

### Performance (Phi-3.5-mini, prompt=512, output=64)

| Config | TTFT (ms) | TPOT (ms) | Out tok/s | Model mem (MB) | Peak mem (MB) |
|--------|-----------|-----------|-----------|----------------|---------------|
| C0 BF16 | | | | | |
| C2 NF4 | | | | | |
| C3 exact | | | | | |
| C4 E5M5 | | | | | |
| C6 bnb dq | | | | | |
| C7 all | | | | | |

### Energy (joules/token)

| Config | J/token | vs C0 |
|--------|---------|-------|
| C0 | | — |
| C4 | | |

---

## Diagnostic Analysis

### Per-layer sensitivity

See `results/aggregate/layer_sensitivity.csv` for layer-level quant error
across scale_mbits ∈ {-1, 0, 3, 5}.

Key finding: (fill)

### Pareto plot

See `results/figures/pareto.png`.

Claim to verify: C4 (E5M5) is Pareto-optimal — smallest scale_mbits
that keeps perplexity within acceptance bounds.

### Memory audit

See `results/aggregate/memory_audit.csv`.

Empirical vs theoretical memory savings table.

---

## Cross-Model Validation

| Model | Config | TPOT (ms) | Ppl (lambada) |
|-------|--------|-----------|---------------|
| Phi-3.5-mini | C0 | | |
| Phi-3.5-mini | C4 | | |
| Llama-3.2-1B | C0 | | |
| Llama-3.2-1B | C4 | | |

---

## Threats to Validity

- **A100 proxy vs Maia:** A100 (sm_80) lacks Maia's hardware-native E5Mx scale
  tensors. SMQ is a software emulation — the bandwidth savings are theoretical.
- **No calibration data:** SMQ uses max-abs scaling without calibration sets
  (unlike AWQ/GPTQ). Quality may improve with calibration.
- **Serving scheduler confounds:** vLLM numbers are in a separate table and
  not compared directly to torch_runner ablation.
- **Colab variability:** GPU throttling and shared-tenant noise may inflate
  variance. All runs flag throttle_reasons from pynvml.

---

## Reproducibility

```bash
git clone https://github.com/pavannn16/maia-inspired-phi35-smq.git
cd maia-inspired-phi35-smq
pip install -r requirements.txt
python -m pytest quant/tests/ -v
# See scripts/colab_quickstart.md for full workflow
```

### Artifact table

| Artifact | Path | Description |
|----------|------|-------------|
| Raw timings | `results/raw/*.jsonl` | Per-request JSONL with env snapshot |
| Summary CSVs | `results/summary/*.csv` | Per-config flat tables |
| Aggregate table | `results/aggregate/paper_table.csv` | Mean ± CI95 |
| Layer sensitivity | `results/aggregate/layer_sensitivity.csv` | Per-layer quant error |
| Memory audit | `results/aggregate/memory_audit.csv` | Empirical vs theoretical |
| Pareto plot | `results/figures/pareto.png` | 2-panel figure |
| lm-eval results | `results/lm_eval/*.json` | Quality task outputs |
