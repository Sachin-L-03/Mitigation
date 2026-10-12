"""Unit tests for causal divergence and activation patching metrics."""

import torch

from causal_analysis import (
    compute_cosine_divergence,
    compute_kl_divergence,
    compute_layer_divergences,
    rank_layers_by_causality,
)


def test_compute_cosine_divergence():
    """Verify cosine divergence is 0 for identical vectors and 2 for opposite vectors."""
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([1.0, 0.0, 0.0])
    assert abs(compute_cosine_divergence(v1, v2)) < 1e-6

    v3 = torch.tensor([-1.0, 0.0, 0.0])
    assert abs(compute_cosine_divergence(v1, v3) - 2.0) < 1e-5


def test_compute_kl_divergence():
    """Verify KL divergence between identical logit distributions is zero."""
    logits1 = torch.tensor([2.0, 1.0, -1.0])
    logits2 = torch.tensor([2.0, 1.0, -1.0])
    assert compute_kl_divergence(logits1, logits2) < 1e-5

    # Divergence should be positive for distinct distributions
    logits3 = torch.tensor([-1.0, 5.0, 1.0])
    assert compute_kl_divergence(logits1, logits3) > 0.0


def test_compute_layer_divergences():
    """Verify divergence calculation across clusters."""
    cluster_records = [
        {"cluster_id": "c1", "variant_type": "original", "layers": {4: torch.tensor([1.0, 0.0])}},
        {"cluster_id": "c1", "variant_type": "lexical", "layers": {4: torch.tensor([0.9, 0.1])}},
        {"cluster_id": "c2", "variant_type": "original", "layers": {4: torch.tensor([0.0, 1.0])}},
    ]
    res = compute_layer_divergences(cluster_records, target_layers=[4])
    assert 4 in res
    assert "within_cluster_divergence" in res[4]
    assert "between_cluster_separation" in res[4]
    assert res[4]["within_cluster_divergence"] >= 0.0


def test_rank_layers_by_causality():
    """Verify descending rank ordering by causal specificity."""
    causal_results = {
        8: {"causal_specificity": 0.12},
        16: {"causal_specificity": 0.45},
        24: {"causal_specificity": 0.05},
    }
    ranked = rank_layers_by_causality(causal_results)
    assert ranked[0] == (16, 0.45)
    assert ranked[1] == (8, 0.12)
    assert ranked[2] == (24, 0.05)
