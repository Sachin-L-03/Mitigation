"""Unit tests for optimization and ablation study utilities."""

import pytest
import torch
import numpy as np

from representation_steering import SubspaceSuppressionHook


class DummyDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return (self.linear(x),)


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([DummyDecoderLayer(hidden_dim) for _ in range(2)])

    def forward(self, input_ids=None, attention_mask=None):
        return None


def test_steering_zero_strength_identity():
    """Verify that steering strength alpha = 0.0 is an exact identity pass-through."""
    dim = 8
    model = DummyModel(hidden_dim=dim)
    P = torch.eye(dim)

    hook = SubspaceSuppressionHook(
        model=model,
        target_layer=0,
        projection_matrix=P,
        steering_strength=0.0,
    )

    test_tensor = torch.randn(1, 4, dim)
    with hook:
        out = hook.hook_fn(None, None, (test_tensor,))
        result = out[0]
        assert torch.allclose(result, test_tensor, atol=1e-7)


def test_subspace_comparison_metrics():
    """Verify metrics for comparing two projection subspaces."""
    dim = 8
    # Subspace A: spanned by e1, e2
    V_a = torch.zeros(dim, 2)
    V_a[0, 0] = 1.0
    V_a[1, 1] = 1.0
    P_a = torch.mm(V_a, V_a.T)

    # Subspace B: identical to A
    P_b = P_a.clone()

    frob_diff = torch.norm(P_a - P_b, p="fro").item()
    assert frob_diff == 0.0

    # Subspace C: orthogonal to A (spanned by e3, e4)
    V_c = torch.zeros(dim, 2)
    V_c[2, 0] = 1.0
    V_c[3, 1] = 1.0
    P_c = torch.mm(V_c, V_c.T)

    frob_ortho = torch.norm(P_a - P_c, p="fro").item()
    assert frob_ortho > 0.0


def test_validation_objective():
    """Verify validation selection score properly balances F1 and preservation."""
    from optimization import compute_validation_selection_score

    # Model A: High F1, good preservation
    score_a = compute_validation_selection_score(token_f1=45.0, anchor_preservation=35.0)
    # Model B: Low F1, good preservation
    score_b = compute_validation_selection_score(token_f1=35.0, anchor_preservation=35.0)
    # Model C: High F1, zero preservation
    score_c = compute_validation_selection_score(token_f1=45.0, anchor_preservation=0.0)

    assert score_a > score_b
    assert score_a > score_c
