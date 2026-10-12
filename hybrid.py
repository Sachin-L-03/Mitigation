"""Integrated execution for the 4 core experimental conditions:
  Condition A: Baseline Model
  Condition B: Representation Steering Only
  Condition C: Parameter Editing Only
  Condition D: Hybrid (Parameter Editing + Representation Steering)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from representation_steering import SubspaceSuppressionHook

logger = logging.getLogger("prompt_sensitivity.hybrid")


def run_condition_inference(
    condition_name: str,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    records: List[Any],
    output_file: str | Path,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    steering_hook: Optional[SubspaceSuppressionHook] = None,
    max_new_tokens: int = 64,
    resume: bool = True,
) -> Dict[str, Any]:
    """Execute inference across a list of prompt records under a specified intervention condition.

    Args:
        condition_name: 'baseline', 'steering_only', 'editing_only', or 'hybrid'
        model: Base or PEFT-adapted model
        tokenizer: Model tokenizer
        records: List of PromptRecord objects
        output_file: Destination path for .jsonl predictions
        model_name: Hugging Face model identifier
        steering_hook: Optional SubspaceSuppressionHook for conditions B and D
        max_new_tokens: Maximum tokens to generate
        resume: Skip prompts already recorded in output_file

    Returns:
        Summary dict containing condition, total records, average latency, and peak memory.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    if resume and out_path.is_file():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        completed_ids.add(data["prompt_id"])
            logger.info(f"Resuming {condition_name}: {len(completed_ids)}/{len(records)} already completed.")
        except Exception as e:
            logger.warning(f"Failed to read existing file for resume: {e}")

    model.eval()
    device = next(model.parameters()).device

    total_time = 0.0
    generated_count = 0
    errors_count = 0

    if steering_hook:
        steering_hook.attach()

    try:
        with open(out_path, "a", encoding="utf-8") as f_out:
            for idx, rec in enumerate(records, start=1):
                pid = getattr(rec, "prompt_id")
                if pid in completed_ids:
                    continue

                cid = getattr(rec, "cluster_id")
                vtype = getattr(rec, "variant_type", "original")
                prompt_text = getattr(rec, "prompt_text")

                start_t = time.perf_counter()
                error_msg: Optional[str] = None
                gen_text = ""

                try:
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                    prompt_len = inputs["input_ids"].shape[1]

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    gen_tokens = output_ids[0][prompt_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                except Exception as err:
                    error_msg = str(err)
                    errors_count += 1
                    logger.error(f"Inference error on {pid}: {err}")

                elapsed = round(time.perf_counter() - start_t, 4)
                total_time += elapsed
                generated_count += 1

                peak_vram = 0.0
                if torch.cuda.is_available():
                    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

                entry = {
                    "cluster_id": cid,
                    "prompt_id": pid,
                    "variant_type": vtype,
                    "prompt_text": prompt_text,
                    "model_name": model_name,
                    "condition": condition_name,
                    "generated_text": gen_text,
                    "runtime_sec": elapsed,
                    "peak_vram_gb": peak_vram,
                    "error": error_msg,
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_out.flush()

                if idx % 50 == 0 or idx == len(records):
                    logger.info(f"[{condition_name}] Processed {idx}/{len(records)} prompts...")

    finally:
        if steering_hook:
            steering_hook.remove()

    avg_latency = round(total_time / max(1, generated_count), 4)
    summary = {
        "condition": condition_name,
        "total_evaluated": len(records),
        "newly_generated": generated_count,
        "errors": errors_count,
        "average_latency_sec": avg_latency,
        "output_file": str(out_path),
    }
    logger.info(f"Finished {condition_name}: {summary}")
    return summary
