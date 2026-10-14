"""Unit tests for contrastive representation module in src/mitigation/contrastive.py."""

import pytest
import torch
import torch.nn.functional as F

from src.mitigation.contrastive import (
    SupervisedContrastiveLoss,
    RepresentationPreservationLoss,
    extract_prompt_embeddings,
)


class MockMiniTransformer(torch.nn.Module):
    """Tiny mock model returning hidden states for testing extraction."""
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, hidden_dim)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        x = self.embedding(input_ids)
        # Return mock hidden states tuple of 2 layers
        class MockOutput:
            def __init__(self, h):
                self.hidden_states = (h, h * 1.5)
        return MockOutput(x)


def test_supcon_loss_identical_clusters_lower_loss():
    """Verify that embeddings closer together within clusters produce lower SupCon loss."""
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)

    # 4 embeddings of dim 8
    # Embeddings 0 and 1 are cluster 0 (close together)
    # Embeddings 2 and 3 are cluster 1 (close together)
    z0 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z1 = torch.tensor([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z2 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z3 = torch.tensor([0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0])

    emb = torch.stack([z0, z1, z2, z3])
    emb = F.normalize(emb, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])

    loss_aligned = loss_fn(emb, labels)
    assert loss_aligned.item() > 0.0
    assert torch.isfinite(loss_aligned)

    # Now misalign them: z1 orthogonal to z0
    z1_bad = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    emb_misaligned = torch.stack([z0, z1_bad, z2, z3])
    emb_misaligned = F.normalize(emb_misaligned, p=2, dim=-1)
    loss_misaligned = loss_fn(emb_misaligned, labels)

    assert loss_misaligned.item() > loss_aligned.item()


def test_supcon_loss_gradients():
    """Verify that gradients are computed cleanly through SupConLoss."""
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)
    emb = torch.randn(6, 16, requires_grad=True)
    norm_emb = F.normalize(emb, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = loss_fn(norm_emb, labels)
    loss.backward()

    assert emb.grad is not None
    assert torch.all(torch.isfinite(emb.grad))


def test_preservation_loss():
    """Verify RepresentationPreservationLoss behavior."""
    loss_fn = RepresentationPreservationLoss()
    z_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # Identical representations -> loss should be 0.0
    loss_zero = loss_fn(z_a, z_b)
    assert pytest.approx(loss_zero.item(), abs=1e-6) == 0.0

    # Orthogonal representations -> loss should be 1.0
    z_c = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss_ortho = loss_fn(z_a, z_c)
    assert pytest.approx(loss_ortho.item(), abs=1e-6) == 1.0


def test_extract_prompt_embeddings_mock():
    """Verify pooling and L2 normalization on mock model."""
    model = MockMiniTransformer(hidden_dim=8)
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    mean_rep = extract_prompt_embeddings(model, input_ids, attention_mask, pooling="mean")
    last_rep = extract_prompt_embeddings(model, input_ids, attention_mask, pooling="last")

    assert mean_rep.shape == (2, 8)
    assert last_rep.shape == (2, 8)

    # Norm must be 1.0
    norms_mean = torch.norm(mean_rep, p=2, dim=-1)
    norms_last = torch.norm(last_rep, p=2, dim=-1)
    assert torch.allclose(norms_mean, torch.ones(2), atol=1e-5)
    assert torch.allclose(norms_last, torch.ones(2), atol=1e-5)
