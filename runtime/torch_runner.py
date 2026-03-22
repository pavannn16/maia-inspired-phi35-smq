"""
PyTorch Ablation Runtime
========================

Loads the model ONCE, then sweeps all (prompt_len, output_len) combinations
in-process so the model is not reloaded for every cell.

Quant modes
-----------
  none             BF16 baseline (C0)
  w4a16_bnb        4-bit NF4 weights via bitsandbytes (C2)
  w4a16_bnb_dq     4-bit NF4 weights with explicit double-quant (C6 baseline)
  w4_shared_scale  Custom SMQ int4 + quantized per-group scales (C3/C4/C5/C7)

quant_target
------------
  mlp   Replace only MLP linear layers (default; isolates bandwidth effect)
  all   Replace MLP + attention Q/K/V/O projections (C7)

Timing model
------------
  prefill_ms       : Single forward pass with use_cache=False (pure prefill cost)
  total_gen_ms     : model.generate() end-to-end (prefill + all decode steps)
  ttft_ms          : Approximated as prefill_ms (dominates for typical prompt lengths)
  tpot_ms          : total_gen_ms / n_output_tokens (average per-token decode time)

Using model.generate() avoids manual KV-cache management and is compatible
with all transformers versions and custom model architectures.

Memory tracking
---------------
  model_mem_mb     : GPU memory after model load, before inference.
  peak_gpu_mem_mb  : Peak GPU memory during the timed generate run.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from runtime.common import (
    PowerSampler,
    append_jsonl,
    get_env_snapshot,
    get_gpu_state,
    maybe_cuda_sync,
    now_ms,
)

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------------------
# Timing dataclass
# ---------------------------------------------------------------------------

@dataclass
class Timings:
    prefill_ms: float       # forward pass only, use_cache=False
    total_gen_ms: float     # model.generate() end-to-end
    n_output: int

    @property
    def ttft_ms(self) -> float:
        """Approximated as prefill latency (dominates TTFT for typical prompts)."""
        return self.prefill_ms

    @property
    def tpot_ms(self) -> float:
        """Average per-output-token time across the full generate call."""
        if self.n_output <= 0:
            return float("nan")
        return self.total_gen_ms / self.n_output


# ---------------------------------------------------------------------------
# Core generate loop — uses model.generate() for full version compatibility
# ---------------------------------------------------------------------------

def generate_timed(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, Timings]:
    """Time prefill and full generation separately using model.generate().

    Prefill is measured via a standalone forward pass (use_cache=False) to
    isolate the prefill cost without triggering any KV-cache API calls.
    Full generation is measured via model.generate() which manages the cache
    internally — compatible with all custom model architectures.
    """
    # --- Prefill timing (no cache, no output tokens) ---
    maybe_cuda_sync()
    t_pre0 = now_ms()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    maybe_cuda_sync()
    prefill_ms = now_ms() - t_pre0

    # --- Full generation timing ---
    maybe_cuda_sync()
    t_gen0 = now_ms()
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    maybe_cuda_sync()
    total_gen_ms = now_ms() - t_gen0

    output_ids = gen_ids[:, input_ids.shape[1]:]
    n_output = int(output_ids.shape[1])

    return output_ids, Timings(
        prefill_ms=prefill_ms,
        total_gen_ms=total_gen_ms,
        n_output=n_output,
    )


# ---------------------------------------------------------------------------
# Layer targeting — supports mlp-only and all (mlp + attention) modes
# ---------------------------------------------------------------------------

def _is_target_layer(name: str, quant_target: str) -> bool:
    n = name.lower()
    is_mlp = ".mlp." in n or any(t in n for t in ["ffn", "feed_forward", "intermediate"])
    if is_mlp:
        return True
    if quant_target == "all":
        is_attn = any(t in n for t in [
            ".q_proj", ".k_proj", ".v_proj", ".o_proj",
            ".query", ".key", ".value", ".query_key_value", ".out_proj",
        ])
        if is_attn:
            return True
    return False


def _replace_layers_with_shared_scale(
    model: nn.Module,
    group_size: int,
    scale_mbits: int,
    quant_target: str,
) -> int:
    from quant.shared_scale_quant import SharedScaleLinear
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}"
            if isinstance(child, nn.Linear) and _is_target_layer(full, quant_target):
                setattr(module, child_name,
                        SharedScaleLinear.from_linear(child, group_size, scale_mbits))
                replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _current_gpu_mem_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 * 1024)


def load_model(
    model_id: str,
    dtype: str,
    quant_mode: str,
    scale_mbits: int,
    group_size: int,
    device: str,
    quant_target: str = "mlp",
) -> Tuple[nn.Module, Any, Optional[float]]:
    torch_dtype = DTYPE_MAP[dtype]
    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if quant_mode == "none":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype
        )
        model.eval().to(device)

    elif quant_mode in ("w4a16_bnb", "w4a16_bnb_dq"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise SystemExit("bitsandbytes not installed. Run: pip install bitsandbytes")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model.eval()

    elif quant_mode == "w4_shared_scale":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype
        )
        model.eval()
        n = _replace_layers_with_shared_scale(model, group_size, scale_mbits, quant_target)
        print(f"[SMQ] Replaced {n} linear layers "
              f"(quant_target={quant_target}, group_size={group_size}, scale_mbits={scale_mbits})")
        model.to(device)

    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    maybe_cuda_sync()
    model_mem_mb = _current_gpu_mem_mb()
    return model, tok, model_mem_mb


# ---------------------------------------------------------------------------
# Single (prompt_len, output_len) timed run
# ---------------------------------------------------------------------------

def _peak_gpu_mem_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def run_one(
    model: nn.Module,
    tok: Any,
    prompt_len: int,
    out_len: int,
    device: str,
    model_id: str,
    quant_mode: str,
    scale_mbits: int,
    model_mem_mb: Optional[float] = None,
) -> Dict[str, Any]:
    torch.manual_seed(42)

    text = "hello " * (prompt_len // 2 + 1)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=prompt_len)
    input_ids = enc["input_ids"].to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sampler = PowerSampler()
    sampler.start()

    output_ids, timings = generate_timed(model, tok, input_ids, max_new_tokens=out_len)

    joules = sampler.stop()

    n_prompt = int(input_ids.shape[1])
    n_output = timings.n_output
    total_tok = n_prompt + n_output
    jpt = (joules / n_output) if (joules is not None and n_output > 0) else None

    return {
        "model": model_id,
        "quant_mode": quant_mode,
        "scale_mbits": scale_mbits,
        "prompt_len": n_prompt,
        "output_len": n_output,
        "prefill_ms": round(timings.prefill_ms, 3),
        "ttft_ms": round(timings.ttft_ms, 3),
        "tpot_ms": round(timings.tpot_ms, 3),
        "total_gen_ms": round(timings.total_gen_ms, 3),
        "throughput_total_tok_s": round(total_tok / (timings.total_gen_ms / 1000), 2) if timings.total_gen_ms > 0 else None,
        "throughput_output_tok_s": round(n_output / (timings.total_gen_ms / 1000), 2) if timings.total_gen_ms > 0 else None,
        "model_mem_mb": round(model_mem_mb, 1) if model_mem_mb is not None else None,
        "peak_gpu_mem_mb": round(_peak_gpu_mem_mb() or 0, 1),
        "joules_per_token": round(jpt, 5) if jpt is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch ablation runner (model loads once)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model2", default=None)
    ap.add_argument("--prompt_lens", type=int, nargs="+", required=True)
    ap.add_argument("--output_lens", type=int, nargs="+", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--quant_mode",
                    choices=["none", "w4a16_bnb", "w4a16_bnb_dq", "w4_shared_scale"],
                    default="none")
    ap.add_argument("--scale_mbits", type=int, default=-1)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--quant_target", choices=["mlp", "all"], default="mlp")
    ap.add_argument("--warmup_runs", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--run_id_prefix", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    model_ids: List[str] = [args.model]
    if args.model2 and args.model2 != args.model:
        model_ids.append(args.model2)

    env = get_env_snapshot()

    for model_id in model_ids:
        model_tag = model_id.replace("/", "_").replace("-", "_")
        print(f"\n=== Model: {model_id} ===")
        print(f"Loading ({args.quant_mode}, scale_mbits={args.scale_mbits}, quant_target={args.quant_target}) ...")

        model, tok, model_mem_mb = load_model(
            model_id, args.dtype, args.quant_mode,
            args.scale_mbits, args.group_size, args.device,
            quant_target=args.quant_target,
        )

        gpu_state_before = get_gpu_state()

        print(f"Warming up ({args.warmup_runs} pass) ...")
        for _ in range(args.warmup_runs):
            run_one(model, tok, 128, 32, args.device, model_id,
                    args.quant_mode, args.scale_mbits, model_mem_mb)

        for prompt_len in args.prompt_lens:
            for output_len in args.output_lens:
                sub_id = f"{args.run_id_prefix}-{model_tag}-p{prompt_len}-o{output_len}"
                result = run_one(
                    model, tok, prompt_len, output_len,
                    args.device, model_id, args.quant_mode, args.scale_mbits,
                    model_mem_mb,
                )
                record: Dict[str, Any] = {
                    "run_id": sub_id,
                    "env": env,
                    "gpu_state_before": gpu_state_before,
                    "gpu_state_after": get_gpu_state(),
                    "result": result,
                }
                append_jsonl(args.out_jsonl, record)
                print(
                    f"  {sub_id}  "
                    f"prefill={result['prefill_ms']:.1f}ms  "
                    f"tpot={result['tpot_ms']:.2f}ms  "
                    f"out_tok/s={result['throughput_output_tok_s']}  "
                    f"model_mem={result['model_mem_mb']}MB  "
                    f"peak_mem={result['peak_gpu_mem_mb']:.0f}MB"
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
