"""Evaluation metrics, bootstrap confidence intervals, and paired hypothesis testing.

Measures exact agreement, lexical similarity, semantic divergence, anchor preservation,
output diversity, answer length, and statistical significance across prompt clusters.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def normalize_answer(text: str) -> str:
    """Standardize answer text by lowercasing, removing punctuation, and stripping extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def compute_token_f1(pred: str, target: str) -> float:
    """Compute token-level F1 score between prediction and target strings."""
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()

    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    return float(2 * precision * recall / (precision + recall))


def compute_char_ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute cosine similarity of character n-gram frequencies as a lightweight semantic metric."""
    norm_a = normalize_answer(text_a)
    norm_b = normalize_answer(text_b)

    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0

    def get_ngrams(s: str) -> Counter:
        if len(s) < n:
            return Counter([s])
        return Counter([s[i:i + n] for i in range(len(s) - n + 1)])

    ca = get_ngrams(norm_a)
    cb = get_ngrams(norm_b)

    all_keys = set(ca.keys()) | set(cb.keys())
    va = np.array([ca[k] for k in all_keys], dtype=np.float32)
    vb = np.array([cb[k] for k in all_keys], dtype=np.float32)

    norm_prod = (np.linalg.norm(va) * np.linalg.norm(vb))
    if norm_prod == 0:
        return 0.0
    return float(np.dot(va, vb) / norm_prod)


def evaluate_cluster_predictions(
    predictions: List[Dict[str, Any]],
    baseline_predictions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute cluster-level agreement, preservation, diversity, and length metrics.

    Args:
        predictions: List of prediction records for the condition
        baseline_predictions: Optional list of baseline predictions to compute anchor preservation

    Returns:
        Dict mapping metric names to cluster-level distributions and aggregate statistics.
    """
    # Group predictions by cluster
    by_cluster: Dict[str, Dict[str, str]] = {}
    for p in predictions:
        cid = p["cluster_id"]
        vtype = p.get("variant_type", "original")
        by_cluster.setdefault(cid, {})[vtype] = p.get("generated_text", "")

    # Group baseline by cluster if provided
    baseline_by_cluster: Dict[str, Dict[str, str]] = {}
    if baseline_predictions:
        for p in baseline_predictions:
            cid = p["cluster_id"]
            vtype = p.get("variant_type", "original")
            baseline_by_cluster.setdefault(cid, {})[vtype] = p.get("generated_text", "")

    cluster_exact_matches: List[float] = []
    cluster_token_f1s: List[float] = []
    cluster_ngram_sims: List[float] = []
    cluster_preservations: List[float] = []
    cluster_answer_lengths: List[float] = []

    unique_answers: set[str] = set()
    total_answers = 0

    for cid, variants in by_cluster.items():
        orig_ans = variants.get("original", "")
        norm_orig = normalize_answer(orig_ans)

        variant_matches = []
        variant_f1s = []
        variant_sims = []
        lengths = []

        for vtype, ans in variants.items():
            total_answers += 1
            unique_answers.add(normalize_answer(ans))
            lengths.append(len(ans.split()))

            if vtype == "original":
                continue

            norm_ans = normalize_answer(ans)
            # Exact match agreement
            variant_matches.append(1.0 if norm_ans == norm_orig and norm_orig != "" else 0.0)
            # Token F1
            variant_f1s.append(compute_token_f1(ans, orig_ans))
            # N-gram similarity
            variant_sims.append(compute_char_ngram_similarity(ans, orig_ans))

        if variant_matches:
            cluster_exact_matches.append(float(np.mean(variant_matches)))
            cluster_token_f1s.append(float(np.mean(variant_f1s)))
            cluster_ngram_sims.append(float(np.mean(variant_sims)))
        if lengths:
            cluster_answer_lengths.append(float(np.mean(lengths)))

        # Anchor preservation: does the intervention change the original prompt's answer vs baseline?
        if baseline_by_cluster and cid in baseline_by_cluster:
            base_orig = baseline_by_cluster[cid].get("original", "")
            pres_f1 = compute_token_f1(orig_ans, base_orig)
            cluster_preservations.append(pres_f1)

    # Output diversity: ratio of distinct generated answers to total evaluated answers
    diversity_ratio = len(unique_answers) / max(1, total_answers)

    return {
        "cluster_exact_matches": cluster_exact_matches,
        "cluster_token_f1s": cluster_token_f1s,
        "cluster_ngram_sims": cluster_ngram_sims,
        "cluster_preservations": cluster_preservations,
        "cluster_answer_lengths": cluster_answer_lengths,
        "diversity_ratio": diversity_ratio,
        "total_clusters": len(by_cluster),
        "total_evaluated_answers": total_answers,
    }


def compute_bootstrap_ci(
    data: List[float],
    num_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean and cluster-level bootstrap confidence interval."""
    if not data:
        return 0.0, 0.0, 0.0

    arr = np.array(data)
    mean_val = float(np.mean(arr))
    if len(arr) < 2:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(arr)
    for _ in range(num_samples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_val, lower, upper


def compare_conditions_paired(
    intervention_scores: List[float],
    baseline_scores: List[float],
) -> Dict[str, Any]:
    """Perform paired statistical test (Wilcoxon signed-rank and paired t-test)."""
    if len(intervention_scores) != len(baseline_scores) or len(intervention_scores) < 3:
        return {
            "abs_change": 0.0,
            "rel_change_pct": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    inter = np.array(intervention_scores)
    base = np.array(baseline_scores)
    diffs = inter - base

    mean_base = float(np.mean(base))
    mean_inter = float(np.mean(inter))
    abs_change = mean_inter - mean_base
    rel_change = (abs_change / mean_base * 100.0) if mean_base != 0.0 else 0.0

    # If all differences are zero
    if np.all(diffs == 0):
        return {
            "abs_change": 0.0,
            "rel_change_pct": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    try:
        res = stats.wilcoxon(diffs, alternative="two-sided")
        p_val = float(res.pvalue)
    except Exception:
        # Fallback to paired t-test
        res = stats.ttest_rel(inter, base)
        p_val = float(res.pvalue)

    return {
        "abs_change": round(abs_change, 4),
        "rel_change_pct": round(rel_change, 2),
        "p_value": round(p_val, 5),
        "significant": bool(p_val < 0.05),
    }
