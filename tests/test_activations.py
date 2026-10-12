"""Unit tests for activation extraction, pooling, and cache management."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from activations import (
    ActivationCacheManager,
    get_transformer_layers,
    pool_hidden_states,
)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return (self.linear(x),)


class DummyTransformerModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyDecoderLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, input_ids=None, attention_mask=None):
        return None


def test_get_transformer_layers():
    """Verify that get_transformer_layers discovers decoder layers across model structures."""
    model = DummyTransformerModel(num_layers=4)
    layers = get_transformer_layers(model)
    assert len(layers) == 4
    assert isinstance(layers, nn.ModuleList)


def test_pool_hidden_states_last():
    """Verify last token pooling selects non-padded token position."""
    batch_size = 2
    seq_len = 5
    hidden_dim = 16
    hs = torch.randn(batch_size, seq_len, hidden_dim)
    # Batch item 0 has length 3 (indices 0, 1, 2); item 1 has length 5 (indices 0..4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)

    pooled = pool_hidden_states(hs, mask, token_position="last")
    assert pooled.shape == (batch_size, hidden_dim)
    assert torch.allclose(pooled[0], hs[0, 2])
    assert torch.allclose(pooled[1], hs[1, 4])


def test_pool_hidden_states_mean():
    """Verify masked mean pooling."""
    hs = torch.ones(1, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    pooled = pool_hidden_states(hs, mask, token_position="mean")
    assert pooled.shape == (1, 8)
    assert torch.allclose(pooled, torch.ones(1, 8))


def test_activation_cache_manager(tmp_path):
    """Verify cache saving, loading, and manifest updates."""
    mgr = ActivationCacheManager(cache_dir=tmp_path, model_name="test_model")
    assert mgr.is_cached("p1") is False

    acts = {4: torch.randn(1, 16), 8: torch.randn(1, 16)}
    saved_file = mgr.save_prompt_activations(
        prompt_id="p1",
        cluster_id="c1",
        variant_type="original",
        activations_by_layer=acts,
    )
    assert saved_file.is_file()
    assert mgr.is_cached("p1") is True

    loaded = mgr.load_prompt_activations("p1")
    assert loaded is not None
    assert loaded["cluster_id"] == "c1"
    assert 4 in loaded["layers"]
    assert torch.allclose(loaded["layers"][4], acts[4], atol=1e-5)
