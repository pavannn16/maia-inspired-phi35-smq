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

Energy
------
pynvml PowerSampler runs in a background thread (100 ms interval).
Joules/token is reported per run; if pynvml is unavailable the field is null.

Warmup
------
One warmup pass (prompt=128, output=32) is always performed before recording
any timings to ensure CUDA kernels and caches are hot.

Memory tracking
---------------
model_mem_mb  : GPU memory allocated immediately after model load (before inference).
peak_gpu_mem_mb: Peak GPU memory during the timed decode run.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
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
    prefill_ms: float
    first_token_ms: float
    decode_token_ms: List[float] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float:
        return self.prefill_ms + self.first_token_ms

    @property
    def tpot_ms(self) -> float:
        if not self.decode_token_ms:
            return float("nan")
        return sum(self.decode_token_ms) / len(self.decode_token_ms)


# ---------------------------------------------------------------------------
# Core decode loop with per-step timing
# ---------------------------------------------------------------------------

def greedy_decode_timed(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, Timings]:
    device = input_ids.device

    # Prefill
    maybe_cuda_sync()
    t0 = now_ms()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    maybe_cuda_sync()
    t1 = now_ms()

    past = out.past_key_values
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    # First decode step (= TTFT minus prefill)
    maybe_cuda_sync()
    t2s = now_ms()
    with torch.no_grad():
        out2 = model(input_ids=next_tok, past_key_values=past, use_cache=True)
    maybe_cuda_sync()
    t2e = now_ms()

    generated = [next_tok]
    past = out2.past_key_values
    next_tok = torch.argmax(out2.logits[:, -1, :], dim=-1, keepdim=True)

    decode_times: List[float] = []
    for _ in range(max_new_tokens - 1):
        maybe_cuda_sync()
        td0 = now_ms()
        with torch.no_grad():
            outn = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        maybe_cuda_sync()
        td1 = now_ms()
        decode_times.append(td1 - td0)
        past = outn.past_key_values
        next_tok = torch.argmax(outn.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_tok)

    gen_ids = torch.cat(generated, dim=1) if generated else torch.empty(
        (1, 0), device=device, dtype=input_ids.dtype
    )
    return gen_ids, Timings(
        prefill_ms=t1 - t0,
        first_token_ms=t2e - t2s,
        decode_token_ms=decode_times,
    )


# ---------------------------------------------------------------------------
# Layer targeting — supports mlp-only and all (mlp + attention) modes
# ---------------------------------------------------------------------------

def _is_target_layer(name: str, quant_target: str) -> bool:
    """Return True if this layer should be replaced by SharedScaleLinear.

    Args:
        name:         Full dotted module name (e.g. 'model.layers.0.mlp.gate_proj').
        quant_target: 'mlp' to target only MLP layers, 'all' to also target
                      attention Q/K/V/O projection layers.
    """
    n = name.lower()

    # MLP layers — always targeted when quant_target in ('mlp', 'all')
    is_mlp = ".mlp." in n or any(t in n for t in ["ffn", "feed_forward", "intermediate"])
    if is_mlp:
        return True

    # Attention layers — only targeted when quant_target == 'all'
    if quant_target == "all":
        is_attn_proj = any(t in n for t in [
            ".q_proj", ".k_proj", ".v_proj", ".o_proj",
            ".query", ".key", ".value",
            ".query_key_value",  # fused QKV (e.g. Falcon)
            ".out_proj",         # some attention output projections
        ])
        if is_attn_proj:
            return True

    return False


def _replace_layers_with_shared_scale(
    model: nn.Module,
    group_size: int,
    scale_mbits: int,
    quant_target: str,
) -> int:
    """Replace Linear layers matching quant_target with SharedScaleLinear.

    Args:
        model:        The loaded HuggingFace model.
        group_size:   SMQ group size (fixed at 128).
        scale_mbits:  Mantissa bits for scale quantization.
        quant_target: 'mlp' or 'all'.

    Returns:
        Number of layers replaced.
    """
    from quant.shared_scale_quant import SharedScaleLinear

    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}"
            if isinstance(child, nn.Linear) and _is_target_layer(full, quant_target):
                setattr(
                    module,
                    child_name,
                    SharedScaleLinear.from_linear(child, group_size, scale_mbits),
                )
                replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# Model loading (quant dispatch)
# ---------------------------------------------------------------------------

def _current_gpu_mem_mb() -> Optional[float]:
    """Current (not peak) GPU memory allocated, in MB."""
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
    """Load and optionally quantize the model.

    Returns:
        (model, tokenizer, model_mem_mb) where model_mem_mb is GPU memory
        allocated after loading (before any inference), or None if no GPU.
    """
    torch_dtype = DTYPE_MAP[dtype]
    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if quant_mode == "none":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, trust_remote_code=True
        )
        model.eval().to(device)

    elif quant_mode in ("w4a16_bnb", "w4a16_bnb_dq"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise SystemExit("bitsandbytes not installed. Run: pip install bitsandbytes")
        # Both C2 (w4a16_bnb) and C6 (w4a16_bnb_dq) use double-quant;
        # C6 is the explicit double-quant comparison baseline for SMQ.
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
            trust_remote_code=True,
        )
        model.eval()

    elif quant_mode == "w4_shared_scale":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, trust_remote_code=True
        )
        model.eval()
        n = _replace_layers_with_shared_scale(model, group_size, scale_mbits, quant_target)
        print(
            f"[SMQ] Replaced {n} linear layers "
            f"(quant_target={quant_target}, group_size={group_size}, scale_mbits={scale_mbits})"
        )
        model.to(device)

    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    # Record model footprint before any inference
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

    # Synthetic prompt — uniform, deterministic (perf benchmark only)
    text = "hello " * (prompt_len // 2 + 1)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=prompt_len)
    input_ids = enc["input_ids"].to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sampler = PowerSampler()
    sampler.start()

    gen_ids, timings = greedy_decode_timed(model, input_ids, max_new_tokens=out_len)

    joules = sampler.stop()

    n_prompt = int(input_ids.shape[1])
    n_output = int(gen_ids.shape[1])
    total_tok = n_prompt + n_output
    decode_ms = timings.first_token_ms + sum(timings.decode_token_ms)
    total_ms = timings.prefill_ms + decode_ms
    jpt = (joules / n_output) if (joules is not None and n_output > 0) else None

    return {
        "model": model_id,
        "quant_mode": quant_mode,
        "scale_mbits": scale_mbits,
        "prompt_len": n_prompt,
        "output_len": n_output,
        "prefill_ms": round(timings.prefill_ms, 3),
        "first_token_ms": round(timings.first_token_ms, 3),
        "ttft_ms": round(timings.ttft_ms, 3),
        "tpot_ms": round(timings.tpot_ms, 3),
        "total_ms": round(total_ms, 3),
        "throughput_total_tok_s": round(total_tok / (total_ms / 1000), 2) if total_ms > 0 else None,
        "throughput_output_tok_s": round(n_output / (decode_ms / 1000), 2) if decode_ms > 0 else None,
        "model_mem_mb": round(model_mem_mb, 1) if model_mem_mb is not None else None,
        "peak_gpu_mem_mb": round(_peak_gpu_mem_mb() or 0, 1),
        "joules_per_token": round(jpt, 5) if jpt is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch ablation runner (model loads once)")
    ap.add_argument("--model", required=True, help="Primary model ID (HuggingFace)")
    ap.add_argument("--model2", default=None,
                    help="Optional second model ID for cross-model validation")
    ap.add_argument("--prompt_lens", type=int, nargs="+", required=True)
    ap.add_argument("--output_lens", type=int, nargs="+", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--quant_mode",
                    choices=["none", "w4a16_bnb", "w4a16_bnb_dq", "w4_shared_scale"],
                    default="none")
    ap.add_argument("--scale_mbits", type=int, default=-1)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--quant_target", choices=["mlp", "all"], default="mlp",
                    help="Which layers to quantize: 'mlp' (default) or 'all' (mlp+attention)")
    ap.add_argument("--warmup_runs", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--run_id_prefix", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    # Build list of models to run
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

        # Warmup (results discarded)
        print(f"Warming up ({args.warmup_runs} pass) ...")
        for _ in range(args.warmup_runs):
            run_one(
                model, tok, 128, 32, args.device, model_id,
                args.quant_mode, args.scale_mbits, model_mem_mb,
            )

        # Timed sweep
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
                    f"  {sub_id}  ttft={result['ttft_ms']:.1f}ms  "
                    f"tpot={result['tpot_ms']:.2f}ms  "
                    f"out_tok/s={result['throughput_output_tok_s']}  "
                    f"model_mem={result['model_mem_mb']}MB  "
                    f"peak_mem={result['peak_gpu_mem_mb']:.0f}MB"
                )

        # Free model memory before loading the next one
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
