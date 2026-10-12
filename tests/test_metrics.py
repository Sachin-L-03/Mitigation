"""Unit tests for metrics, bootstrap confidence intervals, and paired significance testing."""

import pytest

from metrics import (
    compare_conditions_paired,
    compute_bootstrap_ci,
    compute_char_ngram_similarity,
    compute_token_f1,
    evaluate_cluster_predictions,
    normalize_answer,
)


def test_normalize_answer():
    """Verify punctuation removal, lowercasing, and article removal."""
    raw = "The quick, brown Fox!"
    norm = normalize_answer(raw)
    assert norm == "quick brown fox"


def test_compute_token_f1():
    """Verify token F1 score computation."""
    assert compute_token_f1("hello world", "hello world") == 1.0
    assert compute_token_f1("hello world", "goodbye world") == pytest.approx(0.5, rel=1e-3)
    assert compute_token_f1("apple", "banana") == 0.0


def test_compute_char_ngram_similarity():
    """Verify character n-gram cosine similarity."""
    assert compute_char_ngram_similarity("paris", "paris") == pytest.approx(1.0, rel=1e-3)
    assert compute_char_ngram_similarity("paris", "parisian") > 0.5
    assert compute_char_ngram_similarity("abc", "xyz") == 0.0


def test_evaluate_cluster_predictions():
    """Verify cluster-level agreement and preservation metrics."""
    preds = [
        {"cluster_id": "c1", "variant_type": "original", "generated_text": "Answer 42"},
        {"cluster_id": "c1", "variant_type": "lexical", "generated_text": "Answer 42"},
        {"cluster_id": "c1", "variant_type": "syntactic", "generated_text": "Answer 42"},
        {"cluster_id": "c2", "variant_type": "original", "generated_text": "Oxygen"},
        {"cluster_id": "c2", "variant_type": "lexical", "generated_text": "Nitrogen"},
    ]
    eval_res = evaluate_cluster_predictions(preds)
    assert eval_res["total_clusters"] == 2
    # Cluster 1 has 100% agreement, cluster 2 has 0% agreement -> mean 50%
    assert eval_res["cluster_exact_matches"][0] == 1.0
    assert eval_res["cluster_exact_matches"][1] == 0.0
    assert eval_res["diversity_ratio"] > 0.0


def test_compute_bootstrap_ci():
    """Verify bootstrap CI bounds."""
    data = [0.8, 0.82, 0.79, 0.81, 0.80]
    mean, low, high = compute_bootstrap_ci(data, num_samples=100)
    assert low <= mean <= high


def test_compare_conditions_paired():
    """Verify paired statistical comparisons and p-value generation."""
    inter = [0.9, 0.85, 0.95, 0.88, 0.92, 0.89, 0.91, 0.93, 0.87, 0.94]
    base = [0.7, 0.65, 0.72, 0.68, 0.71, 0.69, 0.70, 0.73, 0.66, 0.72]
    res = compare_conditions_paired(inter, base)
    assert res["abs_change"] > 0.0
    assert res["rel_change_pct"] > 0.0
    assert "p_value" in res
    assert res["significant"] is True
