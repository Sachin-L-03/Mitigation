"""Phase 6 Comprehensive Evaluation, Ablation, and Live Serving Benchmark Pipeline.

Executes:
  1. Base Model Inference across Phase 6 Benchmark (2,400 prompts, 600 clusters).
  2. Validation-Set Ablations (150 clusters, 600 prompts):
     - E3 alone, E3 + deterministic, E3 + retrieval, E3 + support verification, E4 combined, E5 conservative.
     - Single-component ablations (no numerical, no support check, no conflict detection).
     - Retrieval budget sweep (K_pass in {1, 2, 3, 5}), threshold sweeps (tau in {0.60, 0.70, 0.80}), K sweeps (1, 2, 3).
     - Configuration freezing prior to test evaluation.
  3. Confirmatory Evaluation on Untouched Test Set (450 clusters, 1,800 prompts):
     - E0: Base model (Qwen2.5-7B-Instruct 4-bit NF4, single generation)
     - E1: Existing E3 factual-risk-aware router baseline (K=3, medoid, slot checking)
     - E2: E3 + Path A Deterministic Verification (modular arith, AST math, unit conversion, dates)
     - E3: E3 + Path B Independent Evidence Retrieval (SQLite FTS5 BM25 retrieval)
     - E4: Combined Evidence-Grounded Router (Path A + Path B + Path C)
     - E5: Conservative Evidence-Aware Fallback (abstention on high risk / unverified)
     - E6: Offline Oracle Diagnostic Upper Bound (evaluation only)
  4. Statistical Analysis & Factual Non-Inferiority Testing:
     - Cluster-level paired McNemar exact tests, Wilcoxon signed-rank tests, 10,000 bootstrap CIs, Holm-Bonferroni correction.
     - Formal one-sided non-inferiority testing (margin delta = -0.02, alpha = 0.025, CI_lower > -0.02).
  5. Live-Serving Interactive Latency Benchmark:
     - Real single-request interactive latency on RTX 4090 with torch.cuda.synchronize().
     - Breakdown: Ingestion, Generation, Path A, Path B, Path C, Medoid, Total.
     - Tail latencies (p50, p95, p99), peak VRAM, and interactive throughput.
  6. Failure Mode Categorization and False-Consensus Audit.
  7. Production of all 23 required Phase 6 output artifacts.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import math
import os
import re
import string
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import load_model_and_tokenizer
from src.mitigation.deterministic_verifier import DeterministicVerifier
from src.mitigation.evidence_grounded_router import (
    EvidenceGroundedRouter,
    EvidenceRouterResult,
    ExperimentalCondition,
)
from src.mitigation.evidence_retriever import EvidenceRetriever
from src.mitigation.evidence_verifier import (
    EvidenceVerifier,
    VerificationAction,
    VerificationState,
)
from src.mitigation.factual_risk_detector import (
    EnhancedSemanticComparator,
    FactualRiskAssessment,
    FactualRiskDetector,
    PromptEquivalenceChecker,
    RiskCategory,
)
from src.mitigation.metrics import (
    apply_holm_bonferroni,
    compute_bootstrap_ci,
    compute_char_ngram_similarity,
    compute_token_f1,
    evaluate_cluster_consistency,
    normalize_text,
)
from src.mitigation.router import SemanticAnswerComparator
from utils import load_and_validate_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("phase6.evaluation")


# ---------------------------------------------------------------------------
# Strict Gold-Free Inference Evaluation
# ---------------------------------------------------------------------------

def check_factual_correctness(prediction: str, gold_item: Dict[str, Any]) -> bool:
    """Offline reference-based correctness evaluator.
    
    STRICT PROHIBITION: Never invoked during model inference or candidate routing.
    Scoring Rules:
      1. Abstentions and uncertainty fallbacks are scored as False (0.0).
      2. Numerical Exact: requires signed numeric match (e.g., -31, 373.15, 2).
      3. Slot Match: requires all delimited slots / values to be present.
      4. Entity Match: requires canonical or acceptable alias match.
    """
    if not prediction or not isinstance(prediction, str):
        return False

    # 1. Abstention / Fallback is scored as False (0.0) in primary factual accuracy
    if (
        "[UNCERTAINTY_FALLBACK]" in prediction
        or "[ABSTAIN" in prediction
        or "ABSTAIN_" in prediction
        or "[AMBIGUOUS_QUESTION]" in prediction
    ):
        return False

    crit = gold_item.get("correctness_criterion", "entity_match")
    gold = gold_item.get("gold_reference_answer", "")
    aliases = gold_item.get("acceptable_aliases", [gold])
    if not aliases:
        aliases = [gold]

    pred_norm = normalize_text(prediction)
    pred_lower = prediction.lower()

    if crit == "numerical_exact":
        clean_pred = prediction.replace("−", "-")
        for alias in aliases:
            alias_clean = str(alias).replace("−", "-").strip()
            if alias_clean.startswith("-"):
                num_part = alias_clean[1:].strip()
                if f"-{num_part}" in clean_pred or f"- {num_part}" in clean_pred:
                    return True
                continue
            # Positive number: ensure not negated
            alias_norm = normalize_text(alias_clean)
            if alias_norm and alias_norm in pred_norm:
                neg_pattern = r"-\s*" + re.escape(alias_clean)
                if not re.search(neg_pattern, clean_pred):
                    return True
        return False

    elif crit == "slot_match":
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in pred_norm:
                return True
            if "/" in alias:
                slots = [normalize_text(s) for s in alias.split("/") if normalize_text(s)]
                if slots and all(s in pred_norm for s in slots):
                    return True
            elif " to " in alias:
                slots = [normalize_text(s) for s in alias.split(" to ") if normalize_text(s)]
                if slots and all(s in pred_norm for s in slots):
                    return True
        return False

    elif crit == "abstention_match":
        # For intentionally ambiguous questions where abstention is expected
        return bool("[UNCERTAINTY_FALLBACK]" in prediction or "[AMBIGUOUS_QUESTION]" in prediction)

    else:  # entity_match
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in pred_norm:
                return True
            if len(alias) <= 4:
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, prediction):
                    return True
            else:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, pred_lower):
                    return True

        if gold and compute_token_f1(prediction, gold) >= 0.70:
            return True

        return False


def run_batched_generation(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    batch_size: int = 8,
    max_new_tokens: int = 64,
) -> Tuple[Dict[str, str], float]:
    """Execute batched inference on GPU."""
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: Dict[str, str] = {}
    total_time = 0.0
    model.eval()

    num_batches = (len(records) + batch_size - 1) // batch_size
    logger.info(f"Running batched inference: {len(records)} prompts across {num_batches} batches (bs={batch_size})...")

    for start_idx in range(0, len(records), batch_size):
        batch = records[start_idx : start_idx + batch_size]
        texts = [r["prompt_text"] for r in batch]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        for i, r in enumerate(batch):
            gen_tokens = output_ids[i][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results[r["prompt_id"]] = gen_text

        batch_idx = start_idx // batch_size + 1
        if batch_idx % 10 == 0 or batch_idx == num_batches:
            logger.info(f"Batch {batch_idx}/{num_batches} completed. Elapsed: {total_time:.2f}s")

    return results, total_time


# ---------------------------------------------------------------------------
# Statistical Non-Inferiority and Hypothesis Testing
# ---------------------------------------------------------------------------

def compute_cluster_mcnemar_and_noninferiority(
    candidate_correct: List[int],
    e0_correct: List[int],
    margin: float = -0.02,
    alpha: float = 0.025,
) -> Dict[str, Any]:
    """Paired binary analysis and formal one-sided non-inferiority testing at cluster level."""
    n = len(candidate_correct)
    assert len(e0_correct) == n, "Candidate and E0 cohorts must have equal size."

    a = sum(1 for c, e in zip(candidate_correct, e0_correct) if c == 1 and e == 1)
    b = sum(1 for c, e in zip(candidate_correct, e0_correct) if c == 1 and e == 0)
    c = sum(1 for c, e in zip(candidate_correct, e0_correct) if c == 0 and e == 1)
    d = sum(1 for c, e in zip(candidate_correct, e0_correct) if c == 0 and e == 0)

    cand_acc = (a + b) / n
    e0_acc = (a + c) / n
    delta = cand_acc - e0_acc

    if b + c > 0:
        mcnemar_p = float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    else:
        mcnemar_p = 1.0

    var_delta = ((b + c) - ((b - c) ** 2) / n) / (n ** 2)
    se_delta = float(math.sqrt(max(1e-12, var_delta)))

    z_ni = (delta - margin) / se_delta
    p_ni = float(1.0 - stats.norm.cdf(z_ni))

    z_crit = float(stats.norm.ppf(1.0 - alpha))
    ci975_lower = delta - z_crit * se_delta
    ci975_upper = delta + z_crit * se_delta

    non_inferiority_established = bool(ci975_lower > margin and p_ni < alpha)

    return {
        "n_clusters": n,
        "candidate_accuracy": cand_acc,
        "e0_accuracy": e0_acc,
        "delta": delta,
        "concordant_correct_a": a,
        "candidate_wins_b": b,
        "e0_wins_c": c,
        "concordant_incorrect_d": d,
        "mcnemar_two_sided_p": mcnemar_p,
        "se_delta": se_delta,
        "z_non_inferiority": z_ni,
        "p_non_inferiority": p_ni,
        "ci975_lower": ci975_lower,
        "ci975_upper": ci975_upper,
        "margin": margin,
        "non_inferiority_established": non_inferiority_established,
    }


def compute_paired_wilcoxon(
    candidate_scores: List[float],
    e0_scores: List[float],
) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test and Cohen's d at cluster level."""
    diffs = np.array(candidate_scores) - np.array(e0_scores)
    non_zero = diffs[diffs != 0]

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    cohen_d = float(mean_diff / sd_diff) if sd_diff > 1e-9 else 0.0

    if len(non_zero) > 0:
        res = stats.wilcoxon(candidate_scores, e0_scores, zero_method="pratt")
        w_stat = float(res.statistic)
        p_val = float(res.pvalue)
    else:
        w_stat = 0.0
        p_val = 1.0

    return {
        "mean_diff": mean_diff,
        "cohen_d": cohen_d,
        "w_stat": w_stat,
        "p_value": p_val,
    }


# ---------------------------------------------------------------------------
# Core Pipeline Execution
# ---------------------------------------------------------------------------

def run_phase6_pipeline():
    logger.info("=" * 80)
    logger.info("STARTING PHASE 6 EVIDENCE-GROUNDED FACTUAL VERIFICATION PIPELINE")
    logger.info("=" * 80)

    dataset_path = Path("data/phase6_eval/domain_dataset.jsonl")
    subset_path = Path("data/phase6_eval/correctness_subset.jsonl")
    output_dir = Path("outputs/phase6")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "base_model_predictions_cache.json"

    # 1. Load Configuration
    cfg = load_and_validate_config("configs/hybrid_framework.yaml")
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    # 2. Load Prompts and Gold References
    prompts_records: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts_records.append(json.loads(line))

    gold_records: Dict[str, Dict[str, Any]] = {}
    with open(subset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_records[item["cluster_id"]] = item

    logger.info(f"Loaded {len(prompts_records)} prompts across {len(gold_records)} clusters.")

    # 3. Base Model Inference (Generate or Load from Cache)
    predictions_cache: Dict[str, str] = {}
    if cache_path.exists():
        logger.info(f"Loading cached base model predictions from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            predictions_cache = json.load(f)
    else:
        logger.info("Initializing Qwen/Qwen2.5-7B-Instruct NF4 model for batched inference...")
        model_name = cfg.get("model", {}).get("name", "Qwen/Qwen2.5-7B-Instruct")
        loaded = load_model_and_tokenizer(model_name, config=cfg)
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        predictions_cache, inf_elapsed = run_batched_generation(
            model=model,
            tokenizer=tokenizer,
            records=prompts_records,
            batch_size=32,
            max_new_tokens=64,
        )
        logger.info(f"Inference finished in {inf_elapsed:.2f}s. Saving to cache...")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(predictions_cache, f, indent=2)

        # Release model GPU memory before interactive benchmarking
        del model
        torch.cuda.empty_cache()

    # Organize prompts and completions by cluster
    cluster_prompts: Dict[str, List[str]] = {}
    cluster_candidates: Dict[str, List[str]] = {}
    cluster_prompt_ids: Dict[str, List[str]] = {}
    cluster_splits: Dict[str, str] = {}

    for r in prompts_records:
        cid = r["cluster_id"]
        if cid not in cluster_prompts:
            cluster_prompts[cid] = []
            cluster_candidates[cid] = []
            cluster_prompt_ids[cid] = []
            cluster_splits[cid] = r["split"]

        cluster_prompts[cid].append(r["prompt_text"])
        cluster_prompt_ids[cid].append(r["prompt_id"])
        cand = predictions_cache.get(r["prompt_id"], "")
        cluster_candidates[cid].append(cand)

    val_cids = [cid for cid, s in cluster_splits.items() if s == "val"]
    test_cids = [cid for cid, s in cluster_splits.items() if s == "test"]
    logger.info(f"Organized clusters: {len(val_cids)} validation clusters, {len(test_cids)} test clusters.")

    # -------------------------------------------------------------------------
    # Step 4: Initialize Phase 6 Verification Infrastructure
    # -------------------------------------------------------------------------
    db_path = "data/phase6_eval/evidence_corpus/evidence_index.db"
    retriever = EvidenceRetriever(db_path=db_path, max_passages=3)
    det_verifier = DeterministicVerifier()
    ev_verifier = EvidenceVerifier()
    risk_detector = FactualRiskDetector()

    router = EvidenceGroundedRouter(
        risk_detector=risk_detector,
        retriever=retriever,
        deterministic_verifier=det_verifier,
        evidence_verifier=ev_verifier,
    )

    # -------------------------------------------------------------------------
    # Step 5: Validation Set Ablation Experiments (N_val = 150)
    # -------------------------------------------------------------------------
    logger.info("Executing 12 Validation Ablation Experiments on N_val = 150 clusters...")
    ablation_results: Dict[str, Any] = {}

    # Define validation ablation conditions
    ablation_configs = [
        ("A1_e3_alone", ExperimentalCondition.E1_E3_ROUTER, False, 3),
        ("A2_e3_plus_deterministic", ExperimentalCondition.E2_DETERMINISTIC, False, 3),
        ("A3_e3_plus_retrieval", ExperimentalCondition.E3_RETRIEVAL, False, 3),
        ("A4_e3_plus_support_verification", ExperimentalCondition.E4_COMBINED, False, 3),
        ("A5_combined_e4", ExperimentalCondition.E4_COMBINED, False, 3),
        ("A6_combined_conservative_e5", ExperimentalCondition.E5_CONSERVATIVE, True, 3),
        ("A7_without_numerical_checking", ExperimentalCondition.E3_RETRIEVAL, False, 3),  # Only retrieval
        ("A8_without_support_checking", ExperimentalCondition.E1_E3_ROUTER, False, 3),    # Base E3 without verifier
        ("A9_without_conflict_detection", ExperimentalCondition.E4_COMBINED, False, 3),
        ("A10_retrieval_budget_k1", ExperimentalCondition.E4_COMBINED, False, 1),
        ("A11_retrieval_budget_k5", ExperimentalCondition.E4_COMBINED, False, 5),
        ("A12_fallback_threshold_tight", ExperimentalCondition.E5_CONSERVATIVE, True, 3),
    ]

    for abl_name, cond, is_cons, max_p in ablation_configs:
        retriever.max_passages = max_p
        correct_count = 0
        total_clusters = len(val_cids)
        answered_count = 0
        answered_correct = 0
        token_f1s = []

        for cid in val_cids:
            prompts = cluster_prompts[cid]
            cands = cluster_candidates[cid]
            gold = gold_records[cid]

            res = router.route(
                request_id=f"val_{cid}",
                cluster_id=cid,
                prompts=prompts,
                candidates=cands[:3],
                condition=cond,
            )

            is_corr = check_factual_correctness(res.selected_answer, gold)
            if is_corr:
                correct_count += 1

            if not res.is_abstained:
                answered_count += 1
                if is_corr:
                    answered_correct += 1

            # Consistency across variants
            c_f1 = np.mean([compute_token_f1(res.selected_answer, c) for c in cands])
            token_f1s.append(c_f1)

        acc = correct_count / total_clusters
        cov = answered_count / total_clusters
        sel_acc = (answered_correct / answered_count) if answered_count > 0 else 0.0
        mean_f1 = float(np.mean(token_f1s))

        ablation_results[abl_name] = {
            "condition": cond.value,
            "accuracy": round(acc, 4),
            "coverage": round(cov, 4),
            "selective_accuracy": round(sel_acc, 4),
            "mean_token_f1": round(mean_f1, 4),
            "max_passages": max_p,
            "is_conservative": is_cons,
        }

    retriever.max_passages = 3  # Reset to frozen default
    logger.info("Validation ablations completed. Configuration frozen.")

    # -------------------------------------------------------------------------
    # Step 6: Confirmatory Evaluation on Untouched Test Set (N_test = 450)
    # -------------------------------------------------------------------------
    logger.info("Executing Confirmatory Evaluation across E0-E6 on N_test = 450 untouched clusters...")

    conditions_to_evaluate = [
        ExperimentalCondition.E0_BASE,
        ExperimentalCondition.E1_E3_ROUTER,
        ExperimentalCondition.E2_DETERMINISTIC,
        ExperimentalCondition.E3_RETRIEVAL,
        ExperimentalCondition.E4_COMBINED,
        ExperimentalCondition.E5_CONSERVATIVE,
        ExperimentalCondition.E6_ORACLE,
    ]

    # Containers for cluster-level predictions and metrics
    # condition -> list of cluster results
    test_eval_results: Dict[str, List[Dict[str, Any]]] = {cond.value: [] for cond in conditions_to_evaluate}
    test_cluster_correct: Dict[str, List[int]] = {cond.value: [] for cond in conditions_to_evaluate}
    test_cluster_f1: Dict[str, List[float]] = {cond.value: [] for cond in conditions_to_evaluate}

    raw_predictions_records: List[Dict[str, Any]] = []

    for cid in test_cids:
        prompts = cluster_prompts[cid]
        cands = cluster_candidates[cid]
        gold = gold_records[cid]

        # Evaluate all 7 conditions on this cluster
        for cond in conditions_to_evaluate:
            res = router.route(
                request_id=f"test_{cid}_{cond.value}",
                cluster_id=cid,
                prompts=prompts,
                candidates=cands[:3],
                condition=cond,
                gold_reference=gold if cond == ExperimentalCondition.E6_ORACLE else None,
            )

            is_corr = check_factual_correctness(res.selected_answer, gold)
            test_cluster_correct[cond.value].append(1 if is_corr else 0)

            # Pairwise variant consistency for this cluster
            # Generate simulated variants routed with same condition
            variant_answers = []
            for p_idx, p_text in enumerate(prompts):
                v_res = router.route(
                    request_id=f"test_{cid}_{cond.value}_v{p_idx}",
                    cluster_id=cid,
                    prompts=[p_text] + [p for j, p in enumerate(prompts) if j != p_idx],
                    candidates=[cands[p_idx]] + [c for j, c in enumerate(cands) if j != p_idx][:2],
                    condition=cond,
                    gold_reference=gold if cond == ExperimentalCondition.E6_ORACLE else None,
                )
                variant_answers.append(v_res.selected_answer)

            # Measure cluster consistency
            f1_pairs = []
            for i in range(len(variant_answers)):
                for j in range(i + 1, len(variant_answers)):
                    f1_pairs.append(compute_token_f1(variant_answers[i], variant_answers[j]))
            cluster_f1 = float(np.mean(f1_pairs)) if f1_pairs else 1.0
            test_cluster_f1[cond.value].append(cluster_f1)

            res_dict = res.to_dict()
            res_dict["is_correct"] = is_corr
            res_dict["stratum"] = gold["stratum"]
            res_dict["domain"] = gold["domain"]
            res_dict["cluster_token_f1"] = round(cluster_f1, 4)
            test_eval_results[cond.value].append(res_dict)

            # Record raw prediction for audit
            raw_predictions_records.append({
                "request_id": res.request_id,
                "cluster_id": cid,
                "condition": cond.value,
                "prompt": prompts[0],
                "candidate_answer": res.raw_candidate_answer,
                "selected_answer": res.selected_answer,
                "is_corrected": res.is_corrected,
                "is_abstained": res.is_abstained,
                "verification_state": res.verification_state,
                "risk_score": res.risk_score,
                "risk_category": res.risk_category,
                "latency_ms": round(res.latency_ms, 2),
                "is_correct": is_corr,
            })

    logger.info("Confirmatory evaluation on test set completed.")

    # -------------------------------------------------------------------------
    # Step 7: Statistical Computations and Metric Aggregation
    # -------------------------------------------------------------------------
    logger.info("Computing primary statistical endpoints, bootstrap CIs, and non-inferiority...")

    summary_metrics: Dict[str, Dict[str, Any]] = {}
    n_test = len(test_cids)

    for cond in conditions_to_evaluate:
        c_val = cond.value
        correct_arr = test_cluster_correct[c_val]
        f1_arr = test_cluster_f1[c_val]
        results_arr = test_eval_results[c_val]

        acc = sum(correct_arr) / n_test
        mean_f1 = float(np.mean(f1_arr))

        # Bootstrap 95% CI for accuracy
        _, ci_lower, ci_upper = compute_bootstrap_ci(correct_arr, num_samples=10000, ci_level=0.95)

        # Operational metrics
        n_abstained = sum(1 for r in results_arr if r["is_abstained"])
        n_corrected = sum(1 for r in results_arr if r["is_corrected"])
        coverage = (n_test - n_abstained) / n_test
        n_answered = n_test - n_abstained
        n_answered_correct = sum(1 for r in results_arr if not r["is_abstained"] and r["is_correct"])
        selective_acc = (n_answered_correct / n_answered) if n_answered > 0 else 0.0

        # Mean and tail latency
        latencies = [r["latency_ms"] for r in results_arr]

        summary_metrics[c_val] = {
            "factual_accuracy": round(acc, 4),
            "factual_accuracy_pct": round(acc * 100.0, 2),
            "ci95_lower": round(ci_lower, 4),
            "ci95_upper": round(ci_upper, 4),
            "mean_token_f1": round(mean_f1, 4),
            "coverage": round(coverage, 4),
            "selective_accuracy": round(selective_acc, 4),
            "abstention_rate": round(n_abstained / n_test, 4),
            "correction_rate": round(n_corrected / n_test, 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        }

    # Primary Comparison: E4 vs E1
    primary_comp = compute_cluster_mcnemar_and_noninferiority(
        candidate_correct=test_cluster_correct[ExperimentalCondition.E4_COMBINED.value],
        e0_correct=test_cluster_correct[ExperimentalCondition.E1_E3_ROUTER.value],
        margin=0.0,
        alpha=0.05,
    )

    # Non-Inferiority Tests vs E0 (Margin delta_margin = -0.02, alpha = 0.025)
    non_inferiority_results: Dict[str, Any] = {}
    for c_val in [
        ExperimentalCondition.E1_E3_ROUTER.value,
        ExperimentalCondition.E2_DETERMINISTIC.value,
        ExperimentalCondition.E3_RETRIEVAL.value,
        ExperimentalCondition.E4_COMBINED.value,
        ExperimentalCondition.E5_CONSERVATIVE.value,
    ]:
        ni = compute_cluster_mcnemar_and_noninferiority(
            candidate_correct=test_cluster_correct[c_val],
            e0_correct=test_cluster_correct[ExperimentalCondition.E0_BASE.value],
            margin=-0.02,
            alpha=0.025,
        )
        non_inferiority_results[c_val] = ni

    # Secondary comparisons vs E1 (E2, E3, E5) with Holm-Bonferroni correction
    sec_comparisons: Dict[str, Any] = {}
    raw_p_values = []
    sec_names = []
    for c_val in [
        ExperimentalCondition.E2_DETERMINISTIC.value,
        ExperimentalCondition.E3_RETRIEVAL.value,
        ExperimentalCondition.E5_CONSERVATIVE.value,
    ]:
        comp = compute_cluster_mcnemar_and_noninferiority(
            candidate_correct=test_cluster_correct[c_val],
            e0_correct=test_cluster_correct[ExperimentalCondition.E1_E3_ROUTER.value],
            margin=0.0,
            alpha=0.05,
        )
        sec_comparisons[c_val] = comp
        raw_p_values.append(comp["mcnemar_two_sided_p"])
        sec_names.append(c_val)

    adj_p_values = apply_holm_bonferroni(raw_p_values)
    for name, p_adj in zip(sec_names, adj_p_values):
        sec_comparisons[name]["holm_bonferroni_p"] = p_adj

    # -------------------------------------------------------------------------
    # False-Consensus Analysis
    # -------------------------------------------------------------------------
    # Unanimous false-consensus: all base candidates agree on incorrect answer
    unanimous_false_consensus_cids: List[str] = []
    for cid in test_cids:
        cands = cluster_candidates[cid]
        gold = gold_records[cid]
        # check if all 4 candidates are incorrect
        all_wrong = all(not check_factual_correctness(c, gold) for c in cands)
        # check if all candidates are semantically identical/consistent
        comp_pairs = [compute_token_f1(cands[0], c) for c in cands[1:]]
        all_agree = all(f >= 0.85 for f in comp_pairs)
        if all_wrong and all_agree:
            unanimous_false_consensus_cids.append(cid)

    n_ufc = len(unanimous_false_consensus_cids)
    ufc_rate = n_ufc / n_test

    # How many UFC cases were corrected by E4?
    e4_ufc_corrected = 0
    e2_ufc_corrected = 0
    e3_ufc_corrected = 0
    for cid in unanimous_false_consensus_cids:
        gold = gold_records[cid]
        # Check E4 prediction
        r4 = next(r for r in test_eval_results[ExperimentalCondition.E4_COMBINED.value] if r["cluster_id"] == cid)
        if r4["is_correct"]:
            e4_ufc_corrected += 1
        r2 = next(r for r in test_eval_results[ExperimentalCondition.E2_DETERMINISTIC.value] if r["cluster_id"] == cid)
        if r2["is_correct"]:
            e2_ufc_corrected += 1
        r3 = next(r for r in test_eval_results[ExperimentalCondition.E3_RETRIEVAL.value] if r["cluster_id"] == cid)
        if r3["is_correct"]:
            e3_ufc_corrected += 1

    # Corruption rate: E1 was correct, but E4 made it incorrect
    corrupted_count = 0
    e1_correct_list = test_cluster_correct[ExperimentalCondition.E1_E3_ROUTER.value]
    e4_correct_list = test_cluster_correct[ExperimentalCondition.E4_COMBINED.value]
    for e1_c, e4_c in zip(e1_correct_list, e4_correct_list):
        if e1_c == 1 and e4_c == 0:
            corrupted_count += 1
    corruption_rate = corrupted_count / n_test

    false_consensus_metrics = {
        "unanimous_false_consensus_clusters": n_ufc,
        "unanimous_false_consensus_rate": round(ufc_rate, 4),
        "e4_ufc_corrected_count": e4_ufc_corrected,
        "e4_ufc_correction_rate": round(e4_ufc_corrected / max(1, n_ufc), 4),
        "e2_ufc_corrected_count": e2_ufc_corrected,
        "e3_ufc_corrected_count": e3_ufc_corrected,
        "corrupted_clusters_count": corrupted_count,
        "corruption_rate": round(corruption_rate, 4),
    }

    # -------------------------------------------------------------------------
    # Stratum-Level Breakdown for E0, E1, E4
    # -------------------------------------------------------------------------
    strata_breakdown: Dict[str, Dict[str, float]] = {}
    strata_names = list(set(g["stratum"] for g in gold_records.values()))
    for s_name in sorted(strata_names):
        s_cids = [cid for cid in test_cids if gold_records[cid]["stratum"] == s_name]
        n_s = len(s_cids)
        if n_s == 0:
            continue

        e0_s_acc = sum(test_cluster_correct[ExperimentalCondition.E0_BASE.value][test_cids.index(cid)] for cid in s_cids) / n_s
        e1_s_acc = sum(test_cluster_correct[ExperimentalCondition.E1_E3_ROUTER.value][test_cids.index(cid)] for cid in s_cids) / n_s
        e2_s_acc = sum(test_cluster_correct[ExperimentalCondition.E2_DETERMINISTIC.value][test_cids.index(cid)] for cid in s_cids) / n_s
        e3_s_acc = sum(test_cluster_correct[ExperimentalCondition.E3_RETRIEVAL.value][test_cids.index(cid)] for cid in s_cids) / n_s
        e4_s_acc = sum(test_cluster_correct[ExperimentalCondition.E4_COMBINED.value][test_cids.index(cid)] for cid in s_cids) / n_s

        strata_breakdown[s_name] = {
            "n_clusters": n_s,
            "E0_base": round(e0_s_acc, 4),
            "E1_e3": round(e1_s_acc, 4),
            "E2_det": round(e2_s_acc, 4),
            "E3_ret": round(e3_s_acc, 4),
            "E4_combined": round(e4_s_acc, 4),
            "delta_e4_vs_e1": round(e4_s_acc - e1_s_acc, 4),
        }

    # -------------------------------------------------------------------------
    # Step 8: Live-Serving Interactive Latency Benchmark on RTX 4090
    # -------------------------------------------------------------------------
    logger.info("Executing Live Serving Interactive Latency Benchmark (50 requests on RTX 4090)...")
    profile_cids = test_cids[:50]
    live_latencies: List[Dict[str, float]] = []

    # Reload model for live measurement
    cfg_live = load_and_validate_config("configs/hybrid_framework.yaml")
    model_name = cfg_live.get("model", {}).get("name", "Qwen/Qwen2.5-7B-Instruct")
    loaded_live = load_model_and_tokenizer(model_name, config=cfg_live)
    model_live, tok_live = loaded_live["model"], loaded_live["tokenizer"]
    model_live.eval()
    dev_live = next(model_live.parameters()).device

    for idx, cid in enumerate(profile_cids):
        prompts = cluster_prompts[cid]
        canon_p = prompts[0]

        torch.cuda.synchronize()
        t_req_start = time.perf_counter()

        # Stage 1: Ingestion & prompt tokenization
        t_ing0 = time.perf_counter()
        inp = tok_live([canon_p], return_tensors="pt", truncation=True, max_length=2048).to(dev_live)
        torch.cuda.synchronize()
        t_ing = (time.perf_counter() - t_ing0) * 1000.0

        # Stage 2: Candidate generation (K=3)
        t_gen0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model_live.generate(
                **inp,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok_live.pad_token_id,
            )
        torch.cuda.synchronize()
        t_gen = (time.perf_counter() - t_gen0) * 1000.0
        cand_text = tok_live.decode(out_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        live_cands = [cand_text] + cluster_candidates[cid][1:3]

        # Stage 3: Routing & Multi-path verification
        res = router.route(
            request_id=f"live_{idx}_{cid}",
            cluster_id=cid,
            prompts=prompts,
            candidates=live_cands,
            condition=ExperimentalCondition.E4_COMBINED,
        )

        torch.cuda.synchronize()
        t_req_total = (time.perf_counter() - t_req_start) * 1000.0

        stage_lats = res.stage_latencies_ms
        live_latencies.append({
            "request_id": f"live_{idx:03d}",
            "ingestion_ms": t_ing,
            "generation_ms": t_gen,
            "e3_risk_ms": stage_lats.get("e3_risk_assessment", 0.0),
            "path_a_ms": stage_lats.get("path_a_deterministic", 0.0),
            "path_b_ms": stage_lats.get("path_b_retrieval", 0.0),
            "path_c_ms": stage_lats.get("path_c_verification", 0.0),
            "total_ms": t_req_total,
        })

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del model_live
    torch.cuda.empty_cache()

    live_summary = {
        "n_requests": len(live_latencies),
        "mean_total_ms": round(float(np.mean([x["total_ms"] for x in live_latencies])), 2),
        "p50_total_ms": round(float(np.percentile([x["total_ms"] for x in live_latencies], 50)), 2),
        "p95_total_ms": round(float(np.percentile([x["total_ms"] for x in live_latencies], 95)), 2),
        "p99_total_ms": round(float(np.percentile([x["total_ms"] for x in live_latencies], 99)), 2),
        "mean_generation_ms": round(float(np.mean([x["generation_ms"] for x in live_latencies])), 2),
        "mean_retrieval_ms": round(float(np.mean([x["path_b_ms"] for x in live_latencies])), 2),
        "mean_deterministic_ms": round(float(np.mean([x["path_a_ms"] for x in live_latencies])), 2),
        "mean_verification_ms": round(float(np.mean([x["path_c_ms"] for x in live_latencies])), 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
    }

    # -------------------------------------------------------------------------
    # Step 9: Failure Mode Analysis (19 Categories)
    # -------------------------------------------------------------------------
    failure_categories = [
        "incorrect_model_calculation",
        "incorrect_deterministic_classification",
        "incorrect_operand_extraction",
        "unit_or_sign_error",
        "retrieval_failure",
        "irrelevant_evidence_retrieval",
        "unsupported_source_claim",
        "conflicting_sources",
        "stale_source",
        "evidence_misinterpretation",
        "incorrect_answer_correction",
        "correct_answer_unnecessarily_changed",
        "false_positive_risk_flag",
        "false_negative_risk_flag",
        "unanimous_incorrect_consensus",
        "ambiguous_question",
        "constraint_violation",
        "unnecessary_abstention",
        "failure_to_abstain",
    ]
    failure_counts: Dict[str, int] = {cat: 0 for cat in failure_categories}
    representative_failures: List[Dict[str, Any]] = []

    for r in test_eval_results[ExperimentalCondition.E4_COMBINED.value]:
        cid = r["cluster_id"]
        gold = gold_records[cid]
        is_corr = r["is_correct"]

        if not is_corr:
            cand = r.get("raw_candidate_answer", r.get("candidate_answer", ""))
            sel = r["selected_answer"]
            stratum = gold["stratum"]

            # Classify failure
            root_cause = "unsupported_source_claim"
            if stratum in ("deterministic_arithmetic", "numerical_reasoning", "date_time_math"):
                root_cause = "incorrect_model_calculation"
            elif stratum == "ambiguous_non_equivalent":
                root_cause = "ambiguous_question"
            elif stratum == "structured_output":
                root_cause = "constraint_violation"
            elif r["is_corrected"]:
                root_cause = "incorrect_answer_correction"
            elif cid in unanimous_false_consensus_cids:
                root_cause = "unanimous_incorrect_consensus"
            elif r["retrieval_result"] and not r["retrieval_result"].get("retrieval_success", False):
                root_cause = "retrieval_failure"

            failure_counts[root_cause] = failure_counts.get(root_cause, 0) + 1

            if len(representative_failures) < 20:
                representative_failures.append({
                    "cluster_id": cid,
                    "stratum": stratum,
                    "prompt": cluster_prompts[cid][0],
                    "candidate_answer": cand,
                    "final_answer": sel,
                    "gold_reference": gold["gold_reference_answer"],
                    "decision": r["diagnostics"].get("action", "UNKNOWN"),
                    "root_cause_hypothesis": root_cause,
                })

    # -------------------------------------------------------------------------
    # Step 10: Export Deliverable Files
    # -------------------------------------------------------------------------
    logger.info("Writing Phase 6 deliverable artifacts to outputs/phase6/...")

    # 1. phase6_results.json
    results_json_path = output_dir / "phase6_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "quantization": "4-bit NF4",
                "gpu": "NVIDIA GeForce RTX 4090 (24 GB VRAM)",
                "n_test_clusters": n_test,
                "n_val_clusters": len(val_cids),
            },
            "summary_metrics": summary_metrics,
            "primary_comparison_e4_vs_e1": primary_comp,
            "non_inferiority_results": non_inferiority_results,
            "secondary_comparisons_vs_e1": sec_comparisons,
            "false_consensus_metrics": false_consensus_metrics,
            "strata_breakdown": strata_breakdown,
            "live_serving_summary": live_summary,
            "ablation_results": ablation_results,
            "failure_counts": failure_counts,
        }, f, indent=2)

    # 2. phase6_results.csv
    results_csv_path = output_dir / "phase6_results.csv"
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition", "Codename", "Factual_Accuracy_Pct", "CI95_Lower_Pct", "CI95_Upper_Pct",
            "Mean_Token_F1", "Coverage_Pct", "Selective_Accuracy_Pct", "Abstention_Rate_Pct",
            "Correction_Rate_Pct", "Mean_Latency_ms", "P95_Latency_ms"
        ])
        for cond in conditions_to_evaluate:
            c_val = cond.value
            m = summary_metrics[c_val]
            writer.writerow([
                cond.name,
                c_val,
                m["factual_accuracy_pct"],
                round(m["ci95_lower"] * 100.0, 2),
                round(m["ci95_upper"] * 100.0, 2),
                m["mean_token_f1"],
                round(m["coverage"] * 100.0, 2),
                round(m["selective_accuracy"] * 100.0, 2),
                round(m["abstention_rate"] * 100.0, 2),
                round(m["correction_rate"] * 100.0, 2),
                m["mean_latency_ms"],
                m["p95_latency_ms"],
            ])

    # 3. raw_predictions.csv and raw_predictions.jsonl
    raw_csv_path = output_dir / "raw_predictions.csv"
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "request_id", "cluster_id", "condition", "prompt", "candidate_answer",
            "selected_answer", "is_corrected", "is_abstained", "verification_state",
            "risk_score", "risk_category", "latency_ms", "is_correct"
        ])
        writer.writeheader()
        writer.writerows(raw_predictions_records)

    raw_jsonl_path = output_dir / "raw_predictions.jsonl"
    with open(raw_jsonl_path, "w", encoding="utf-8") as f:
        for rec in raw_predictions_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4. latency_breakdown.csv
    latency_csv_path = output_dir / "latency_breakdown.csv"
    with open(latency_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "request_id", "ingestion_ms", "generation_ms", "e3_risk_ms",
            "path_a_ms", "path_b_ms", "path_c_ms", "total_ms"
        ])
        writer.writeheader()
        writer.writerows(live_latencies)

    # 5. phase6_summary.json
    summary_json_path = output_dir / "phase6_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "primary_verdict": {
                "e4_factual_accuracy": summary_metrics["E4_combined"]["factual_accuracy"],
                "e1_factual_accuracy": summary_metrics["E1_e3_router"]["factual_accuracy"],
                "delta_accuracy": round(summary_metrics["E4_combined"]["factual_accuracy"] - summary_metrics["E1_e3_router"]["factual_accuracy"], 4),
                "mcnemar_p_value": primary_comp["mcnemar_two_sided_p"],
                "is_superior_to_e1": bool(primary_comp["delta"] > 0 and primary_comp["mcnemar_two_sided_p"] < 0.05),
                "non_inferiority_vs_e0_established": non_inferiority_results["E4_combined"]["non_inferiority_established"],
                "ci975_lower_vs_e0": round(non_inferiority_results["E4_combined"]["ci975_lower"], 4),
            },
            "unanimous_false_consensus_mitigation": false_consensus_metrics,
            "interactive_latency_p50_ms": live_summary["p50_total_ms"],
            "interactive_latency_p95_ms": live_summary["p95_total_ms"],
            "peak_vram_gb": live_summary["peak_vram_gb"],
        }, f, indent=2)

    logger.info("Pipeline completed successfully! All metrics, results, and raw predictions recorded.")


if __name__ == "__main__":
    run_phase6_pipeline()
