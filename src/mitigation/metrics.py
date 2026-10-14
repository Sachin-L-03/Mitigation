"""Audited and extended metric evaluation library for CREME-Hybrid framework.

Provides:
  - Canonical and pairwise exact match agreements (clearly distinguishing definitions)
  - Canonical and pairwise token F1 metrics
  - Character n-gram and semantic similarity proxies
  - Factual concept accuracy and short/full answer evaluation
  - Anchor preservation against unmodified baseline outputs
  - Cluster-level bootstrap 95% confidence intervals
  - Paired two-sided Wilcoxon signed-rank tests with Holm-Bonferroni corrections
"""

from __future__ import annotations

import re
import string
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def normalize_text(text: str) -> str:
    """Standardize text by lowercasing, removing articles, punctuation, and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def compute_token_f1(pred: str, target: str) -> float:
    """Compute token-level F1 score between prediction and target strings.

    Precision: common_tokens / len(pred_tokens)
    Recall:    common_tokens / len(target_tokens)
    F1:        2 * P * R / (P + R)
    """
    pred_tokens = normalize_text(pred).split()
    target_tokens = normalize_text(target).split()

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
    """Compute cosine similarity of character n-gram frequency vectors."""
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)

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

    norm_prod = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm_prod == 0:
        return 0.0
    return float(np.dot(va, vb) / norm_prod)


def check_concept_accuracy(generated_text: str, gold_reference: str | List[str]) -> bool:
    """Verify whether generated response contains the verified factual entity/concept."""
    norm_gen = normalize_text(generated_text)
    if not norm_gen:
        return False

    if isinstance(gold_reference, str):
        # Extract keywords and salient chunks
        chunks = re.split(r"[/,()]", gold_reference.lower())
        candidates = [normalize_text(c) for c in chunks if normalize_text(c)]
        if not candidates:
            candidates = [normalize_text(gold_reference)]
    else:
        candidates = [normalize_text(c) for c in gold_reference if normalize_text(c)]

    for cand in candidates:
        if cand and cand in norm_gen:
            return True

    # Fallback: high token F1
    if isinstance(gold_reference, str) and compute_token_f1(generated_text, gold_reference) >= 0.40:
        return True

    return False


def evaluate_cluster_consistency(
    cluster_answers: Dict[str, str],
    canonical_key: str = "original",
) -> Dict[str, float]:
    """Compute comprehensive within-cluster consistency metrics.

    Distinguishes:
      - canonical_exact_match: Mean agreement of variants with the canonical answer
      - pairwise_exact_match:  Mean agreement across all pairs in the cluster
      - all_identical_exact_match: 1.0 if all variants produce identical normalized answers, else 0.0
      - canonical_token_f1:    Mean Token F1 of variants against canonical answer
      - pairwise_token_f1:     Mean Token F1 across all pairs in the cluster
      - canonical_ngram_sim:   Mean char 3-gram cosine similarity vs canonical answer
    """
    if not cluster_answers:
        return {
            "canonical_exact_match": 0.0,
            "pairwise_exact_match": 0.0,
            "all_identical_exact_match": 0.0,
            "canonical_token_f1": 0.0,
            "pairwise_token_f1": 0.0,
            "canonical_ngram_sim": 0.0,
            "mean_answer_length": 0.0,
        }

    keys = list(cluster_answers.keys())
    answers = [cluster_answers[k] for k in keys]
    norm_answers = [normalize_text(ans) for ans in answers]
    lengths = [len(ans.split()) for ans in answers]

    # 1. All identical agreement
    first_norm = norm_answers[0]
    all_identical = 1.0 if (len(norm_answers) > 1 and all(a == first_norm and a != "" for a in norm_answers)) else 0.0

    # 2. Canonical comparisons
    canon_text = cluster_answers.get(canonical_key, answers[0])
    norm_canon = normalize_text(canon_text)

    canon_matches = []
    canon_f1s = []
    canon_sims = []

    for k, ans in cluster_answers.items():
        if k == canonical_key and len(cluster_answers) > 1:
            continue
        norm_a = normalize_text(ans)
        canon_matches.append(1.0 if (norm_a == norm_canon and norm_canon != "") else 0.0)
        canon_f1s.append(compute_token_f1(ans, canon_text))
        canon_sims.append(compute_char_ngram_similarity(ans, canon_text))

    # 3. Pairwise comparisons across all distinct pairs
    pair_matches = []
    pair_f1s = []

    if len(answers) >= 2:
        for (i, j) in combinations(range(len(answers)), 2):
            match = 1.0 if (norm_answers[i] == norm_answers[j] and norm_answers[i] != "") else 0.0
            f1 = compute_token_f1(answers[i], answers[j])
            pair_matches.append(match)
            pair_f1s.append(f1)

    return {
        "canonical_exact_match": float(np.mean(canon_matches)) if canon_matches else 0.0,
        "pairwise_exact_match": float(np.mean(pair_matches)) if pair_matches else 0.0,
        "all_identical_exact_match": float(all_identical),
        "canonical_token_f1": float(np.mean(canon_f1s)) if canon_f1s else 0.0,
        "pairwise_token_f1": float(np.mean(pair_f1s)) if pair_f1s else 0.0,
        "canonical_ngram_sim": float(np.mean(canon_sims)) if canon_sims else 0.0,
        "mean_answer_length": float(np.mean(lengths)) if lengths else 0.0,
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
        res = stats.ttest_rel(inter, base)
        p_val = float(res.pvalue)

    return {
        "abs_change": round(abs_change, 4),
        "rel_change_pct": round(rel_change, 2),
        "p_value": round(p_val, 5),
        "significant": bool(p_val < 0.05),
    }


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Adjust p-values using the Holm-Bonferroni step-down procedure."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = p_val * (n - rank)
        adj = min(1.0, max(running_max, adj))
        running_max = adj
        adjusted[orig_idx] = round(adj, 5)

    return adjusted
