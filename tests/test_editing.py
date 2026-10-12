"""Unit tests for parameter editing dataset collation and loss computation."""

import pytest
import torch
import torch.nn as nn

from parameter_editing import collate_editing_batch, compute_editing_loss


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids=None, attention_mask=None):
        emb = self.embedding(input_ids)
        logits = self.head(emb)

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


def test_collate_editing_batch():
    """Verify padding and tensor shaping in collate_editing_batch."""
    s1 = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([-100, 2, 3]),
        "is_original": True,
    }
    s2 = {
        "input_ids": torch.tensor([4, 5]),
        "attention_mask": torch.tensor([1, 1]),
        "labels": torch.tensor([-100, 5]),
        "is_original": False,
    }

    batch = collate_editing_batch([s1, s2])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["is_original"].tolist() == [True, False]


def test_compute_editing_loss():
    """Verify multi-component loss computation."""
    model = DummyCausalLM(vocab_size=10, hidden_dim=8)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "labels": torch.tensor([[-100, 2, 3], [-100, 5, -100]]),
        "is_original": torch.tensor([True, False]),
    }
    loss_weights = {"consistency": 1.0, "preservation": 1.0, "l2_reg": 0.01}

    total_loss, metrics = compute_editing_loss(model, batch, loss_weights)
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0.0
    assert "consistency_loss" in metrics
    assert "preservation_loss" in metrics
    assert "param_reg_loss" in metrics
