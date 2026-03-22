"""
Online Serving Benchmark — Concurrency and QPS Sweep (vLLM)
============================================================

Measures TTFT/TPOT under concurrent load.  Requires vLLM.

Install:  pip install vllm

Usage:
    python -m bench.online_bench \\
        --config configs/experiment_matrix.yaml \\
        --model microsoft/Phi-3.5-mini-instruct \\
        --config_id C0 \\
        --out_dir results

!! NOTE !!  Results are vLLM serving context only — not comparable to
torch_runner.  Report in separate table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runtime.common import append_jsonl, get_env_snapshot, get_gpu_state


async def _send_request(engine: Any, prompt: str, sampling_params: Any, req_id: str) -> Dict[str, Any]:
    t0 = time.monotonic()
    first_token_t = None
    n_output = 0
    async for output in engine.generate(prompt, sampling_params, request_id=req_id):
        if first_token_t is None and output.outputs and output.outputs[0].token_ids:
            first_token_t = time.monotonic()
        n_output = len(output.outputs[0].token_ids) if output.outputs else 0
    t1 = time.monotonic()
    ttft = (first_token_t - t0) * 1000.0 if first_token_t else float("nan")
    decode_ms = (t1 - (first_token_t or t0)) * 1000.0
    tpot = decode_ms / max(n_output - 1, 1)
    return {"ttft_ms": ttft, "tpot_ms": tpot, "total_ms": (t1 - t0) * 1000.0, "n_output_tokens": n_output}


async def _run_concurrency(engine: Any, prompts: List[str], sampling_params: Any, concurrency: int) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(i, p):
        async with sem:
            return await _send_request(engine, p, sampling_params, str(i))

    return await asyncio.gather(*[bounded(i, p) for i, p in enumerate(prompts)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--config_id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--output_len", type=int, default=128)
    ap.add_argument("--num_prompts", type=int, default=64)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    except ImportError:
        raise SystemExit("vLLM not installed. Run: pip install vllm")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    concurrencies: List[int] = cfg["workloads"]["online_serving"]["concurrency"]

    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    run_id = f"online-{args.config_id}-{uuid.uuid4().hex[:8]}"
    out_path = os.path.join(raw_dir, f"{run_id}.jsonl")

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompt = tok.decode([tok.eos_token_id or 0] * args.prompt_len, skip_special_tokens=True)
    prompts = [prompt] * args.num_prompts

    for c in concurrencies:
        results = asyncio.run(_run_concurrency(engine, prompts, sampling, c))
        ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] == r["ttft_ms"]]
        tpots = [r["tpot_ms"] for r in results]

        import statistics
        record: Dict[str, Any] = {
            "run_id": f"{run_id}-c{c}",
            "runner": "vllm_async",
            "note": "NON-COMPARABLE to torch_runner",
            "env": get_env_snapshot(),
            "result": {
                "config_id": args.config_id,
                "concurrency": c,
                "prompt_len": args.prompt_len,
                "output_len": args.output_len,
                "ttft_p50_ms": statistics.median(ttfts) if ttfts else float("nan"),
                "ttft_p95_ms": sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) > 1 else float("nan"),
                "tpot_p50_ms": statistics.median(tpots),
            },
        }
        append_jsonl(out_path, record)
        print(f"concurrency={c}  ttft_p50={record['result']['ttft_p50_ms']:.1f}ms  tpot_p50={record['result']['tpot_p50_ms']:.1f}ms")

    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
