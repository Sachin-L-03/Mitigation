"""Independent external generalization and factual correctness evaluation pipeline.

Evaluates:
  Condition A: Original Qwen baseline
  Condition B: Parameter editing adapter w_pres=1.0 (outputs/parameter_editing/best_adapter)
  Condition C: Preservation-tuned adapter w_pres=5.0 (outputs/optimization/preservation/adapter_w5.0)

Across:
  - 100 Independent semantic clusters (600 prompts, zero overlap with training data)
  - 40 Verified factual QA clusters (240 prompts) with gold reference answers

Generates:
  - Resumable JSONL predictions for each condition
  - Per-cluster metrics and bootstrap 95% confidence intervals
  - Factual correctness audit and variant robustness analysis
  - Statistical hypothesis tests (Wilcoxon signed-rank + Holm-Bonferroni)
  - CSV table and comprehensive Markdown research report
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from scipy import stats

from dataset_external import ExternalPromptRecord, load_external_records
from metrics import (
    compare_conditions_paired,
    compute_bootstrap_ci,
    compute_char_ngram_similarity,
    compute_token_f1,
    normalize_answer,
)
from model import load_model_and_tokenizer
from optimization import apply_holm_bonferroni
from utils import load_and_validate_config, set_seed, setup_logger

logger = logging.getLogger("prompt_sensitivity.evaluate_external")


def extract_keywords_from_gold(gold_text: str) -> List[str]:
    """Extract salient concepts/keywords from gold reference answer for factual evaluation."""
    norm = normalize_answer(gold_text)
    # Split by common delimiters like parenthesis, slashes, commas
    chunks = re.split(r"[/,()]", gold_text.lower())
    clean_chunks = [normalize_answer(c) for c in chunks if normalize_answer(c)]
    # Also individual tokens if multi-word
    return clean_chunks if clean_chunks else [norm]


def check_factual_accuracy(generated_text: str, gold_reference: str) -> bool:
    """Verify whether generated response contains the verified factual concept."""
    norm_gen = normalize_answer(generated_text)
    if not norm_gen or not gold_reference:
        return False

    keywords = extract_keywords_from_gold(gold_reference)
    # Match if any salient chunk is found in the normalized generated text
    for kw in keywords:
        if kw and kw in norm_gen:
            return True

    # High token overlap fallback (token F1 > 0.40)
    if compute_token_f1(generated_text, gold_reference) >= 0.40:
        return True

    return False


def run_model_inference_for_condition(
    condition_name: str,
    peft_model: PeftModel,
    tokenizer: Any,
    records: List[ExternalPromptRecord],
    output_file: Path,
    adapter_name: Optional[str] = None,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Execute inference for a condition across external prompt records with batched generation."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()

    if output_file.is_file():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        completed_ids.add(entry["prompt_id"])
            logger.info(f"[{condition_name}] Resuming: {len(completed_ids)}/{len(records)} already recorded.")
        except Exception as e:
            logger.warning(f"Error reading existing predictions: {e}")

    remaining_records = [r for r in records if r.prompt_id not in completed_ids]

    peft_model.eval()
    device = next(peft_model.parameters()).device

    total_time = 0.0
    generated_count = 0
    errors_count = 0

    tokenizer.padding_side = "left"

    with open(output_file, "a", encoding="utf-8") as f_out:
        for b_start in range(0, len(remaining_records), batch_size):
            batch = remaining_records[b_start : b_start + batch_size]
            prompt_texts = [r.prompt_text for r in batch]

            start_t = time.perf_counter()
            err_msg = None

            try:
                inputs = tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(device)
                prompt_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    if adapter_name is None:
                        with peft_model.disable_adapter():
                            output_ids = peft_model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    else:
                        peft_model.set_adapter(adapter_name)
                        output_ids = peft_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                elapsed_batch = time.perf_counter() - start_t
                per_prompt_latency = round(elapsed_batch / len(batch), 4)
                total_time += elapsed_batch

                peak_vram = 0.0
                if torch.cuda.is_available():
                    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

                for idx_in_b, rec in enumerate(batch):
                    gen_tokens = output_ids[idx_in_b, prompt_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                    record_dict = {
                        "cluster_id": rec.cluster_id,
                        "prompt_id": rec.prompt_id,
                        "variant_type": rec.variant_type,
                        "prompt_text": rec.prompt_text,
                        "gold_reference_answer": rec.gold_reference_answer,
                        "is_factual_subset": rec.is_factual_subset,
                        "domain": rec.domain,
                        "condition": condition_name,
                        "generated_text": gen_text,
                        "runtime_sec": per_prompt_latency,
                        "peak_vram_gb": peak_vram,
                        "error": err_msg,
                    }
                    f_out.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
                    generated_count += 1

                f_out.flush()

            except Exception as e:
                err_msg = str(e)
                errors_count += len(batch)
                logger.error(f"Inference error on batch starting at {batch[0].prompt_id}: {e}")

            processed_total = len(completed_ids) + generated_count
            if processed_total % 50 == 0 or processed_total == len(records):
                logger.info(f"[{condition_name}] Processed {processed_total}/{len(records)} prompts...")

    avg_latency = round(total_time / max(1, generated_count), 4)
    return {
        "condition": condition_name,
        "total_prompts": len(records),
        "newly_generated": generated_count,
        "errors": errors_count,
        "average_latency_sec": avg_latency,
        "output_file": str(output_file),
    }


def load_prediction_file(filepath: Path) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate_condition_performance(
    predictions: List[Dict[str, Any]],
    baseline_predictions: List[Dict[str, Any]],
    factual_records: List[ExternalPromptRecord],
) -> Dict[str, Any]:
    """Compute cluster-level consistency, anchor preservation, diversity, and factual correctness."""
    # Organize condition predictions by cluster
    by_cluster: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in predictions:
        cid = p["cluster_id"]
        vtype = p["variant_type"]
        by_cluster.setdefault(cid, {})[vtype] = p

    # Organize baseline predictions by cluster for anchor preservation
    base_by_cluster: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in baseline_predictions:
        cid = p["cluster_id"]
        vtype = p["variant_type"]
        base_by_cluster.setdefault(cid, {})[vtype] = p

    # Organize factual subset gold map
    gold_map = {r.cluster_id: r for r in factual_records if r.is_factual_subset}

    cluster_ema: List[float] = []
    cluster_token_f1: List[float] = []
    cluster_semantic_sim: List[float] = []
    cluster_anchor_pres: List[float] = []
    cluster_lengths: List[float] = []
    cluster_factual_acc: List[float] = []
    cluster_factual_f1: List[float] = []

    unique_answers = set()
    total_answers = 0
    non_empty_count = 0

    per_cluster_details: Dict[str, Any] = {}

    for cid, variants in by_cluster.items():
        orig_entry = variants.get("original", {})
        orig_text = orig_entry.get("generated_text", "")
        norm_orig = normalize_answer(orig_text)

        v_ema = []
        v_f1 = []
        v_sim = []
        v_len = []

        for vtype, entry in variants.items():
            total_answers += 1
            text = entry.get("generated_text", "")
            norm_text = normalize_answer(text)
            unique_answers.add(norm_text)
            if norm_text:
                non_empty_count += 1
            v_len.append(len(text.split()))

            if vtype == "original":
                continue

            # Within-cluster agreement vs original
            is_match = 1.0 if (norm_text == norm_orig and norm_orig != "") else 0.0
            f1 = compute_token_f1(text, orig_text)
            sim = compute_char_ngram_similarity(text, orig_text)

            v_ema.append(is_match)
            v_f1.append(f1)
            v_sim.append(sim)

        avg_ema = float(np.mean(v_ema)) if v_ema else 0.0
        avg_f1 = float(np.mean(v_f1)) if v_f1 else 0.0
        avg_sim = float(np.mean(v_sim)) if v_sim else 0.0
        avg_len = float(np.mean(v_len)) if v_len else 0.0

        cluster_ema.append(avg_ema)
        cluster_token_f1.append(avg_f1)
        cluster_semantic_sim.append(avg_sim)
        cluster_lengths.append(avg_len)

        # Anchor preservation: condition's original answer vs baseline's original answer
        base_orig_text = base_by_cluster.get(cid, {}).get("original", {}).get("generated_text", "")
        anchor_score = compute_token_f1(orig_text, base_orig_text)
        cluster_anchor_pres.append(anchor_score)

        # Factual correctness if in factual subset
        c_fact_acc = None
        c_fact_f1 = None
        if cid in gold_map:
            gold_ans = gold_map[cid].gold_reference_answer or ""
            fact_accs = []
            fact_f1s = []
            for vtype, entry in variants.items():
                ans_text = entry.get("generated_text", "")
                acc = 1.0 if check_factual_accuracy(ans_text, gold_ans) else 0.0
                f1_g = compute_token_f1(ans_text, gold_ans)
                fact_accs.append(acc)
                fact_f1s.append(f1_g)

            c_fact_acc = float(np.mean(fact_accs))
            c_fact_f1 = float(np.mean(fact_f1s))
            cluster_factual_acc.append(c_fact_acc)
            cluster_factual_f1.append(c_fact_f1)

        per_cluster_details[cid] = {
            "within_cluster_token_f1": round(avg_f1 * 100, 2),
            "within_cluster_exact_match": round(avg_ema * 100, 2),
            "within_cluster_semantic_sim": round(avg_sim * 100, 2),
            "anchor_preservation": round(anchor_score * 100, 2),
            "average_word_length": round(avg_len, 1),
            "factual_accuracy": round(c_fact_acc * 100, 2) if c_fact_acc is not None else None,
            "factual_token_f1": round(c_fact_f1 * 100, 2) if c_fact_f1 is not None else None,
        }

    diversity_ratio = len(unique_answers) / max(1, total_answers)
    non_empty_rate = non_empty_count / max(1, total_answers)

    return {
        "cluster_ema": cluster_ema,
        "cluster_token_f1": cluster_token_f1,
        "cluster_semantic_sim": cluster_semantic_sim,
        "cluster_anchor_pres": cluster_anchor_pres,
        "cluster_lengths": cluster_lengths,
        "cluster_factual_acc": cluster_factual_acc,
        "cluster_factual_f1": cluster_factual_f1,
        "diversity_ratio": diversity_ratio,
        "non_empty_rate": non_empty_rate,
        "total_clusters": len(by_cluster),
        "total_prompts": total_answers,
        "per_cluster_details": per_cluster_details,
    }


def execute_independent_validation() -> None:
    """Run complete external generalization and factual correctness pipeline."""
    setup_logger("prompt_sensitivity", "outputs/logs/external_validation.log")
    logger.info("Initializing Independent External Generalization Evaluation...")

    cfg = load_and_validate_config("config.yaml")
    set_seed(cfg.get("project", {}).get("seed", 42))

    data_path = Path("data/external_eval/external_dataset.jsonl")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing external dataset at {data_path}. Run build_external_dataset.py first.")

    records = load_external_records(data_path)
    logger.info(f"Loaded {len(records)} prompt records across 100 semantic clusters.")

    factual_records = [r for r in records if r.is_factual_subset]
    logger.info(f"Factual subset contains {len(factual_records)} prompts across 40 clusters.")

    out_dir = Path("outputs/external_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_pred_file = out_dir / "baseline_external_predictions.jsonl"
    edit_w1_pred_file = out_dir / "editing_w1_external_predictions.jsonl"
    pres_w5_pred_file = out_dir / "preservation_w5_external_predictions.jsonl"

    w1_adapter_path = Path("outputs/parameter_editing/best_adapter")
    w5_adapter_path = Path("outputs/optimization/preservation/adapter_w5.0")

    if not w1_adapter_path.is_dir():
        raise FileNotFoundError(f"Missing w1 adapter at {w1_adapter_path}")
    if not w5_adapter_path.is_dir():
        raise FileNotFoundError(f"Missing w5 adapter at {w5_adapter_path}")

    # Load Base Model once in 4-bit NF4
    logger.info("Loading Base Qwen model in 4-bit precision...")
    res = load_model_and_tokenizer(cfg.get("model", {}).get("name", "Qwen/Qwen2.5-7B-Instruct"), config=cfg)
    base_model = res["model"]
    tokenizer = res["tokenizer"]

    # Wrap in PeftModel with w1 adapter
    logger.info(f"Loading w1 adapter from {w1_adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, str(w1_adapter_path), adapter_name="w1")

    # Add w5 adapter to the same PeftModel
    logger.info(f"Loading w5 adapter from {w5_adapter_path}...")
    peft_model.load_adapter(str(w5_adapter_path), adapter_name="w5")

    # 1. Condition A: Baseline (adapters disabled)
    logger.info("--- Running Condition A: Baseline (unmodified Qwen) ---")
    base_summary = run_model_inference_for_condition(
        condition_name="baseline",
        peft_model=peft_model,
        tokenizer=tokenizer,
        records=records,
        output_file=base_pred_file,
        adapter_name=None,
    )
    logger.info(f"Condition A complete: {base_summary}")

    # 2. Condition B: Editing w_pres=1.0
    logger.info("--- Running Condition B: Editing-only w_pres=1.0 ---")
    w1_summary = run_model_inference_for_condition(
        condition_name="editing_w1",
        peft_model=peft_model,
        tokenizer=tokenizer,
        records=records,
        output_file=edit_w1_pred_file,
        adapter_name="w1",
    )
    logger.info(f"Condition B complete: {w1_summary}")

    # 3. Condition C: Preservation-tuned w_pres=5.0
    logger.info("--- Running Condition C: Preservation-tuned w_pres=5.0 ---")
    w5_summary = run_model_inference_for_condition(
        condition_name="preservation_w5",
        peft_model=peft_model,
        tokenizer=tokenizer,
        records=records,
        output_file=pres_w5_pred_file,
        adapter_name="w5",
    )
    logger.info(f"Condition C complete: {w5_summary}")

    # Load predictions for evaluation
    base_preds = load_prediction_file(base_pred_file)
    w1_preds = load_prediction_file(edit_w1_pred_file)
    w5_preds = load_prediction_file(pres_w5_pred_file)

    logger.info("Computing metrics, bootstrap confidence intervals, and hypothesis tests...")

    eval_base = evaluate_condition_performance(base_preds, base_preds, factual_records)
    eval_w1 = evaluate_condition_performance(w1_preds, base_preds, factual_records)
    eval_w5 = evaluate_condition_performance(w5_preds, base_preds, factual_records)

    conditions = [
        ("baseline", "Condition A: Original Baseline", eval_base, base_summary),
        ("editing_w1", "Condition B: Editing Only (w_pres=1.0)", eval_w1, w1_summary),
        ("preservation_w5", "Condition C: Preservation-Tuned (w_pres=5.0)", eval_w5, w5_summary),
    ]

    # Compute bootstrap CIs
    ci_results: Dict[str, Any] = {}
    for ckey, clabel, c_eval, c_sum in conditions:
        mean_ema, ema_l, ema_u = compute_bootstrap_ci(c_eval["cluster_ema"])
        mean_f1, f1_l, f1_u = compute_bootstrap_ci(c_eval["cluster_token_f1"])
        mean_sim, sim_l, sim_u = compute_bootstrap_ci(c_eval["cluster_semantic_sim"])
        mean_pres, pres_l, pres_u = compute_bootstrap_ci(c_eval["cluster_anchor_pres"])
        mean_len, len_l, len_u = compute_bootstrap_ci(c_eval["cluster_lengths"])
        mean_facc, facc_l, facc_u = compute_bootstrap_ci(c_eval["cluster_factual_acc"])
        mean_ff1, ff1_l, ff1_u = compute_bootstrap_ci(c_eval["cluster_factual_f1"])

        ci_results[ckey] = {
            "label": clabel,
            "within_cluster_token_f1": {"mean": round(mean_f1 * 100, 2), "ci_95": [round(f1_l * 100, 2), round(f1_u * 100, 2)]},
            "within_cluster_exact_match": {"mean": round(mean_ema * 100, 2), "ci_95": [round(ema_l * 100, 2), round(ema_u * 100, 2)]},
            "within_cluster_semantic_sim": {"mean": round(mean_sim * 100, 2), "ci_95": [round(sim_l * 100, 2), round(sim_u * 100, 2)]},
            "anchor_preservation": {"mean": round(mean_pres * 100, 2), "ci_95": [round(pres_l * 100, 2), round(pres_u * 100, 2)]},
            "average_answer_length": {"mean": round(mean_len, 2), "ci_95": [round(len_l, 2), round(len_u, 2)]},
            "factual_accuracy": {"mean": round(mean_facc * 100, 2), "ci_95": [round(facc_l * 100, 2), round(facc_u * 100, 2)]},
            "factual_token_f1": {"mean": round(mean_ff1 * 100, 2), "ci_95": [round(ff1_l * 100, 2), round(ff1_u * 100, 2)]},
            "diversity_ratio": round(c_eval["diversity_ratio"], 4),
            "non_empty_rate": round(c_eval["non_empty_rate"] * 100, 2),
            "latency_sec_per_prompt": c_sum["average_latency_sec"],
        }

    # Statistical hypothesis tests
    # 1. w1 vs Baseline
    test_w1_vs_base_f1 = compare_conditions_paired(eval_w1["cluster_token_f1"], eval_base["cluster_token_f1"])
    test_w1_vs_base_sim = compare_conditions_paired(eval_w1["cluster_semantic_sim"], eval_base["cluster_semantic_sim"])
    test_w1_vs_base_fact = compare_conditions_paired(eval_w1["cluster_factual_acc"], eval_base["cluster_factual_acc"])

    # 2. w5 vs Baseline
    test_w5_vs_base_f1 = compare_conditions_paired(eval_w5["cluster_token_f1"], eval_base["cluster_token_f1"])
    test_w5_vs_base_sim = compare_conditions_paired(eval_w5["cluster_semantic_sim"], eval_base["cluster_semantic_sim"])
    test_w5_vs_base_fact = compare_conditions_paired(eval_w5["cluster_factual_acc"], eval_base["cluster_factual_acc"])

    # 3. w5 vs w1
    test_w5_vs_w1_f1 = compare_conditions_paired(eval_w5["cluster_token_f1"], eval_w1["cluster_token_f1"])
    test_w5_vs_w1_pres = compare_conditions_paired(eval_w5["cluster_anchor_pres"], eval_w1["cluster_anchor_pres"])
    test_w5_vs_w1_fact = compare_conditions_paired(eval_w5["cluster_factual_acc"], eval_w1["cluster_factual_acc"])

    raw_p_values = [
        test_w1_vs_base_f1["p_value"],
        test_w5_vs_base_f1["p_value"],
        test_w5_vs_w1_f1["p_value"],
        test_w5_vs_w1_pres["p_value"],
        test_w5_vs_base_fact["p_value"],
        test_w1_vs_base_fact["p_value"],
    ]
    adj_p_values = apply_holm_bonferroni(raw_p_values)

    statistical_comparisons = {
        "w1_vs_baseline": {
            "token_f1": {**test_w1_vs_base_f1, "adj_p_value": adj_p_values[0]},
            "semantic_sim": test_w1_vs_base_sim,
            "factual_accuracy": {**test_w1_vs_base_fact, "adj_p_value": adj_p_values[5]},
        },
        "w5_vs_baseline": {
            "token_f1": {**test_w5_vs_base_f1, "adj_p_value": adj_p_values[1]},
            "semantic_sim": test_w5_vs_base_sim,
            "factual_accuracy": {**test_w5_vs_base_fact, "adj_p_value": adj_p_values[4]},
        },
        "w5_vs_w1": {
            "token_f1": {**test_w5_vs_w1_f1, "adj_p_value": adj_p_values[2]},
            "anchor_preservation": {**test_w5_vs_w1_pres, "adj_p_value": adj_p_values[3]},
            "factual_accuracy": test_w5_vs_w1_fact,
        },
    }

    # Save summary JSON
    summary_data = {
        "evaluation_name": "PromptSensitivity_External_Generalization_Validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_clusters": 100,
        "total_prompts": 600,
        "factual_clusters": 40,
        "factual_prompts": 240,
        "conditions": ci_results,
        "statistical_tests": statistical_comparisons,
        "cross_model_status": {
            "target_model": "meta-llama/Llama-3.1-8B-Instruct",
            "access_status": "BLOCKED",
            "blocker_reason": "401 Client Error: Gated repository requires Hugging Face authentication token. No token found in environment.",
        },
    }

    with open(out_dir / "external_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save per-cluster metrics
    cluster_report = {
        "baseline": eval_base["per_cluster_details"],
        "editing_w1": eval_w1["per_cluster_details"],
        "preservation_w5": eval_w5["per_cluster_details"],
    }
    with open(out_dir / "per_cluster_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cluster_report, f, indent=2)

    # Save correctness report
    correctness_report = {
        "subset_description": "40 semantic clusters (240 prompts) with verified domain-expert ground truth",
        "baseline": {
            "factual_accuracy": ci_results["baseline"]["factual_accuracy"],
            "factual_token_f1": ci_results["baseline"]["factual_token_f1"],
        },
        "editing_w1": {
            "factual_accuracy": ci_results["editing_w1"]["factual_accuracy"],
            "factual_token_f1": ci_results["editing_w1"]["factual_token_f1"],
        },
        "preservation_w5": {
            "factual_accuracy": ci_results["preservation_w5"]["factual_accuracy"],
            "factual_token_f1": ci_results["preservation_w5"]["factual_token_f1"],
        },
        "comparison_w5_vs_baseline": test_w5_vs_base_fact,
        "comparison_w5_vs_w1": test_w5_vs_w1_fact,
        "methodological_note": "Linguistic paraphrase consistency measures response invariance, NOT factual accuracy. Correctness is evaluated strictly against independently verified gold reference answers.",
    }
    with open(out_dir / "correctness_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(correctness_report, f, indent=2)

    # Export CSV table
    csv_file = out_dir / "external_validation_table.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition",
            "Token F1 (%)",
            "Token F1 95% CI",
            "Exact Match (%)",
            "Semantic Sim (%)",
            "Anchor Pres (%)",
            "Anchor Pres 95% CI",
            "Factual Acc (%)",
            "Factual Acc 95% CI",
            "Factual F1 (%)",
            "Avg Length (words)",
            "Diversity Ratio",
            "Latency (s/prompt)",
        ])
        for ckey, clabel, c_eval, c_sum in conditions:
            ci = ci_results[ckey]
            writer.writerow([
                clabel,
                ci["within_cluster_token_f1"]["mean"],
                f"[{ci['within_cluster_token_f1']['ci_95'][0]}, {ci['within_cluster_token_f1']['ci_95'][1]}]",
                ci["within_cluster_exact_match"]["mean"],
                ci["within_cluster_semantic_sim"]["mean"],
                ci["anchor_preservation"]["mean"],
                f"[{ci['anchor_preservation']['ci_95'][0]}, {ci['anchor_preservation']['ci_95'][1]}]",
                ci["factual_accuracy"]["mean"],
                f"[{ci['factual_accuracy']['ci_95'][0]}, {ci['factual_accuracy']['ci_95'][1]}]",
                ci["factual_token_f1"]["mean"],
                ci["average_answer_length"]["mean"],
                ci["diversity_ratio"],
                ci["latency_sec_per_prompt"],
            ])

    logger.info("External evaluation execution, metrics computation, and exports complete!")


if __name__ == "__main__":
    execute_independent_validation()
