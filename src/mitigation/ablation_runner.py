"""Unified Ablation Study Runner evaluating conditions E0 through E6.

Conditions:
  E0: Original base model (baseline sensitivity)
  E1: Inference-time semantic comparison and selective verification (training-free)
  E2: Standard LoRA adaptation (ordinary fine-tuning)
  E3: Contrastive LoRA adaptation (learned representation invariance)
  E4: CREME-style targeted editing (layer-aware parameter editing)
  E5: Contrastive LoRA + targeted editing (combined representation + editing)
  E6: Full hybrid with adaptive routing and verification (complete framework)
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from scipy import stats

from src.mitigation.metrics import (
    apply_holm_bonferroni,
    check_concept_accuracy,
    compute_bootstrap_ci,
    compute_char_ngram_similarity,
    compute_token_f1,
    evaluate_cluster_consistency,
    normalize_text,
)
from src.mitigation.router import (
    AdaptiveMitigationRouter,
    MitigationResult,
    RouteDecision,
    SemanticAnswerComparator,
)

logger = logging.getLogger("prompt_sensitivity.mitigation.ablation")


CONDITIONS_METADATA = [
    ("E0", "Original Base Model", "Establish baseline sensitivity"),
    ("E1", "Inference Semantic Verification", "Training-free semantic comparison and selective verification"),
    ("E2", "Standard LoRA Adaptation", "Ordinary parameter-efficient adaptation"),
    ("E3", "Contrastive LoRA Adaptation", "Supervised contrastive representation invariance"),
    ("E4", "CREME-Style Targeted Editing", "Causally guided layer-aware parameter editing (w_pres=5.0)"),
    ("E5", "Contrastive + Targeted Editing", "Combined contrastive invariance plus layer-aware editing"),
    ("E6", "Full Hybrid Framework", "Complete adaptive routing, contrastive adapter, and verification"),
]


def evaluate_predictions_record(
    predictions_by_cluster: Dict[str, Dict[str, str]],
    baseline_predictions_by_cluster: Dict[str, Dict[str, str]],
    gold_map: Dict[str, str],
) -> Dict[str, Any]:
    """Compute cluster-level consistency, anchor preservation, and factual accuracy."""
    cluster_canonical_f1s = []
    cluster_pairwise_f1s = []
    cluster_canonical_emas = []
    cluster_pairwise_emas = []
    cluster_all_identical = []
    cluster_semantic_sims = []
    cluster_anchor_pres = []
    cluster_lengths = []
    cluster_factual_accs = []
    cluster_factual_f1s = []

    unique_answers = set()
    total_answers = 0

    for cid, variants in predictions_by_cluster.items():
        consistency_res = evaluate_cluster_consistency(variants, canonical_key="original")
        cluster_canonical_f1s.append(consistency_res["canonical_token_f1"])
        cluster_pairwise_f1s.append(consistency_res["pairwise_token_f1"])
        cluster_canonical_emas.append(consistency_res["canonical_exact_match"])
        cluster_pairwise_emas.append(consistency_res["pairwise_exact_match"])
        cluster_all_identical.append(consistency_res["all_identical_exact_match"])
        cluster_semantic_sims.append(consistency_res["canonical_ngram_sim"])
        cluster_lengths.append(consistency_res["mean_answer_length"])

        for ans in variants.values():
            total_answers += 1
            unique_answers.add(normalize_text(ans))

        # Anchor preservation: original answer vs baseline original answer
        orig_ans = variants.get("original", "")
        base_orig = baseline_predictions_by_cluster.get(cid, {}).get("original", "")
        pres = compute_token_f1(orig_ans, base_orig)
        cluster_anchor_pres.append(pres)

        # Factual accuracy if cluster is in factual subset
        if cid in gold_map:
            gold = gold_map[cid]
            acc_list = []
            f1_list = []
            for ans in variants.values():
                acc_list.append(1.0 if check_concept_accuracy(ans, gold) else 0.0)
                f1_list.append(compute_token_f1(ans, gold))
            cluster_factual_accs.append(float(np.mean(acc_list)))
            cluster_factual_f1s.append(float(np.mean(f1_list)))

    diversity = len(unique_answers) / max(1, total_answers)

    return {
        "cluster_canonical_f1": cluster_canonical_f1s,
        "cluster_pairwise_f1": cluster_pairwise_f1s,
        "cluster_canonical_ema": cluster_canonical_emas,
        "cluster_pairwise_ema": cluster_pairwise_emas,
        "cluster_all_identical": cluster_all_identical,
        "cluster_semantic_sim": cluster_semantic_sims,
        "cluster_anchor_pres": cluster_anchor_pres,
        "cluster_lengths": cluster_lengths,
        "cluster_factual_acc": cluster_factual_accs,
        "cluster_factual_f1": cluster_factual_f1s,
        "diversity_ratio": diversity,
        "total_clusters": len(predictions_by_cluster),
        "total_answers": total_answers,
    }


def compare_paired_cluster_scores(inter: List[float], base: List[float]) -> Dict[str, Any]:
    """Paired two-sided Wilcoxon signed-rank test between candidate and baseline."""
    if len(inter) != len(base) or len(inter) < 3:
        return {"abs_change": 0.0, "rel_change_pct": 0.0, "p_value": 1.0, "significant": False}

    arr_i = np.array(inter)
    arr_b = np.array(base)
    diffs = arr_i - arr_b

    mean_b = float(np.mean(arr_b))
    mean_i = float(np.mean(arr_i))
    abs_change = mean_i - mean_b
    rel_change = (abs_change / mean_b * 100.0) if mean_b != 0.0 else 0.0

    if np.all(diffs == 0):
        return {"abs_change": 0.0, "rel_change_pct": 0.0, "p_value": 1.0, "significant": False}

    try:
        res = stats.wilcoxon(diffs, alternative="two-sided")
        p_val = float(res.pvalue)
    except Exception:
        res = stats.ttest_rel(arr_i, arr_b)
        p_val = float(res.pvalue)

    return {
        "abs_change": round(abs_change, 4),
        "rel_change_pct": round(rel_change, 2),
        "p_value": round(p_val, 5),
        "significant": bool(p_val < 0.05),
    }
