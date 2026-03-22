"""
vLLM Serving Reference Runner
==============================

Records TTFT/TPOT under vLLM's async engine (paged attention + continuous
batching).  Used for configs C0/C2 "production serving" numbers.

!! IMPORTANT !!  Results from this runner are NOT directly comparable to
torch_runner.py.  vLLM uses paged attention, kernel fusions, and a different
scheduler.  Always report these numbers in a *separate* table labelled
"vLLM serving reference (C0/C2)".

Install (Colab):
    pip install vllm

Usage:
    python -m runtime.vllm_runner \\
        --model microsoft/Phi-3.5-mini-instruct \\
        --config_id C0 \\
        --num_prompts 64 \\
        --prompt_len 512 \\
        --output_len 128 \\
        --out_jsonl results/raw/vllm_C0.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runtime.common import append_jsonl, get_env_snapshot, get_gpu_state


def _build_prompts(tokenizer: Any, prompt_len: int, num_prompts: int) -> List[str]:
    # Synthetic prompt — deterministic, consistent across configs
    tok_ids = [tokenizer.eos_token_id or 0] * prompt_len
    text = tokenizer.decode(tok_ids, skip_special_tokens=True)
    return [text] * num_prompts


def run_vllm(
    model_id: str,
    num_prompts: int,
    prompt_len: int,
    output_len: int,
    dtype: str,
    out_jsonl: str,
    run_id: str,
) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise SystemExit(
            "vLLM not installed.  Run: pip install vllm\n"
            "vLLM results are optional — baselines still run via torch_runner."
        )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    llm = LLM(
        model=model_id,
        dtype=dtype,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    prompts = _build_prompts(tok, prompt_len, num_prompts)

    # Warmup
    llm.generate(prompts[:2], sampling, use_tqdm=False)

    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    elapsed_s = time.monotonic() - t0

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput_out = total_output_tokens / elapsed_s

    record: Dict[str, Any] = {
        "run_id": run_id,
        "runner": "vllm",
        "note": "NON-COMPARABLE to torch_runner — different scheduler/engine",
        "env": get_env_snapshot(),
        "gpu_state": get_gpu_state(),
        "result": {
            "model": model_id,
            "dtype": dtype,
            "num_prompts": num_prompts,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "elapsed_s": elapsed_s,
            "throughput_output_tok_s": throughput_out,
        },
    }
    append_jsonl(out_jsonl, record)
    print(f"vLLM throughput: {throughput_out:.1f} tok/s  ({elapsed_s:.1f}s total)")
    print(f"Written: {out_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM serving reference benchmark")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config_id", default="C0")
    ap.add_argument("--num_prompts", type=int, default=64)
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--output_len", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out_jsonl", default="results/raw/vllm.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    run_id = f"vllm-{args.config_id}-{uuid.uuid4().hex[:8]}"
    run_vllm(
        model_id=args.model,
        num_prompts=args.num_prompts,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        dtype=args.dtype,
        out_jsonl=args.out_jsonl,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
