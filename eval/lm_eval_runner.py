"""
LM Evaluation Harness Runner
=============================

Runs lm-eval-harness (0.4+) for a given experiment config ID and writes
results to results/lm_eval_{config_id}.json.

Key decisions (frozen, must match experiment_matrix.yaml fairness_constraints):
  - trust_remote_code omitted  (Phi-3.5 natively supported since transformers 4.39)
  - dtype=bfloat16          (all ablation configs)
  - num_fewshot=0           (zero-shot for instruct model)
  - do_sample=False, temperature=0  (deterministic greedy)
  - chat template NOT applied for standard completion tasks
    (hellaswag, arc_challenge, lambada_openai are multiple-choice/perplexity)
  - For gsm8k: add --apply_chat_template since it is open-ended generation

Usage:
    python -m eval.lm_eval_runner \\
        --config configs/experiment_matrix.yaml \\
        --model microsoft/Phi-3.5-mini-instruct \\
        --config_id C0

    # For a quantized config:
    python -m eval.lm_eval_runner \\
        --config configs/experiment_matrix.yaml \\
        --model microsoft/Phi-3.5-mini-instruct \\
        --config_id C4 \\
        --scale_mbits 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model_args(model_id: str, quant: Dict[str, Any], scale_mbits: int) -> str:
    """Build the lm_eval --model_args string for the given quant config."""
    mode = quant.get("mode", "none")
    dtype = "bfloat16"  # frozen per fairness_constraints

    parts = [f"pretrained={model_id}", f"dtype={dtype}"]

    if mode in ("w4a16_bnb", "w4a16_bnb_dq"):
        parts += ["load_in_4bit=True", "bnb_4bit_quant_type=nf4",
                  "bnb_4bit_use_double_quant=True"]
    # w4_shared_scale cannot be applied directly via lm_eval model_args;
    # for quality eval of C3/C4/C5/C7 the model must be pre-quantized or
    # evaluated via a custom lm_eval model class (future work).
    # For now this runner supports C0 (BF16), C2 and C6 (bnb NF4/dq).

    return ",".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="lm-eval harness runner")
    ap.add_argument("--config", required=True, help="Path to experiment_matrix.yaml")
    ap.add_argument("--model", required=True, help="HuggingFace model ID")
    ap.add_argument("--config_id", required=True,
                    help="Experiment config ID (e.g. C0, C2). Used for output naming.")
    ap.add_argument("--scale_mbits", type=int, default=None,
                    help="Override scale_mbits (for C5 sub-runs)")
    ap.add_argument("--num_fewshot", type=int, default=None,
                    help="Override few-shot count (default from YAML)")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--out_dir", default="results/lm_eval")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    conf = cfg["configs"][args.config_id]
    quant = conf.get("quant", {})
    quality = cfg["metrics"]["quality"]
    tasks: List[str] = quality["tasks"]
    num_fewshot = args.num_fewshot if args.num_fewshot is not None else quality.get("num_fewshot", 0)

    scale_mbits = args.scale_mbits if args.scale_mbits is not None else quant.get("scale_mbits", -1)
    model_args = _build_model_args(args.model, quant, scale_mbits)

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"{args.config_id}.json")

    # Separate gsm8k (requires chat template) from completion tasks
    completion_tasks = [t for t in tasks if t != "gsm8k"]
    chat_tasks = [t for t in tasks if t == "gsm8k"]

    def _run_tasks(task_list: List[str], extra_flags: List[str], suffix: str) -> None:
        if not task_list:
            return
        out = out_json.replace(".json", f"{suffix}.json")
        cmd = (
            [sys.executable, "-m", "lm_eval"]
            + ["--model", "hf"]
            + ["--model_args", model_args]
            + ["--tasks", ",".join(task_list)]
            + ["--device", "cuda"]
            + ["--batch_size", args.batch_size]
            + ["--num_fewshot", str(num_fewshot)]
            + ["--gen_kwargs", "temperature=0,do_sample=False"]
            + ["--output_path", out]
            + ["--log_samples"]
            + extra_flags
        )
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Results: {out}")

    _run_tasks(completion_tasks, [], "_completion")
    _run_tasks(chat_tasks, ["--apply_chat_template"], "_chat")


if __name__ == "__main__":
    main()
