"""Unit tests for subspace estimation and suppression steering hook."""

import pytest
import torch
import torch.nn as nn

from representation_steering import (
    SubspaceSuppressionHook,
    estimate_sensitivity_subspace,
)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return (self.linear(x),)


class DummyModel(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyDecoderLayer(hidden_dim) for _ in range(2)])

    def forward(self, input_ids=None, attention_mask=None):
        return None


def test_estimate_sensitivity_subspace(tmp_path):
    """Verify SVD subspace estimation produces symmetric projection operator P of correct shape."""
    dim = 8
    train_records = [
        {"cluster_id": "c1", "variant_type": "original", "layers": {0: torch.zeros(dim)}},
        {"cluster_id": "c1", "variant_type": "lexical", "layers": {0: torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])}},
        {"cluster_id": "c2", "variant_type": "original", "layers": {0: torch.zeros(dim)}},
        {"cluster_id": "c2", "variant_type": "syntactic", "layers": {0: torch.tensor([0.5, 0.5, 0, 0, 0, 0, 0, 0])}},
    ]
    save_file = tmp_path / "subspace.pt"
    P, s_vals = estimate_sensitivity_subspace(train_records, target_layer=0, rank=2, save_path=save_file)

    assert P.shape == (dim, dim)
    # Projection operator must be symmetric: P = P^T
    assert torch.allclose(P, P.T, atol=1e-5)
    assert save_file.is_file()


def test_subspace_suppression_hook():
    """Verify that SubspaceSuppressionHook projects out sensitive subspace components."""
    dim = 4
    model = DummyModel(hidden_dim=dim)
    # Projection matrix targeting first dimension
    P = torch.zeros(dim, dim)
    P[0, 0] = 1.0  # Projects out first dimension

    hook = SubspaceSuppressionHook(
        model=model,
        target_layer=0,
        projection_matrix=P,
        steering_strength=1.0,  # Full suppression
        token_position="all",
    )

    # Input tensor with activation [5.0, 3.0, 2.0, 1.0]
    test_tensor = torch.tensor([[[5.0, 3.0, 2.0, 1.0]]])
    with hook:
        out = hook.hook_fn(None, None, (test_tensor,))
        steered = out[0]
        # First dimension should be suppressed to 0, other dimensions untouched
        assert abs(steered[0, 0, 0].item()) < 1e-5
        assert abs(steered[0, 0, 1].item() - 3.0) < 1e-5
        assert abs(steered[0, 0, 2].item() - 2.0) < 1e-5
