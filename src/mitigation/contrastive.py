"""Contrastive representation learning module for prompt sensitivity mitigation.

Implements:
  - Masked prompt representation pooling (mean, last-token) with L2 normalization
  - Numerically stable Supervised Contrastive Loss (SupConLoss) with temperature scaling
  - Representation preservation loss against frozen baseline representations
  - Training loop with LoRA adaptation, gradient accumulation, and validation checkpointing
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("prompt_sensitivity.mitigation.contrastive")


def extract_prompt_embeddings(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = "mean",
    layer_idx: int = -1,
) -> torch.Tensor:
    """Extract masked, L2-normalized prompt representations from transformer hidden states.

    Args:
        model: Transformer causal LM or PEFT model
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        pooling: 'mean' or 'last'
        layer_idx: index into hidden_states (-1 for final layer)

    Returns:
        (batch_size, hidden_dim) L2-normalized representation
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    hidden_states = outputs.hidden_states[layer_idx]  # (B, L, D)

    if pooling == "last":
        # Extract last non-padding token representation
        # attention_mask is 1 for real tokens, 0 for pad
        seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
        seq_lengths = torch.clamp(seq_lengths, min=0)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        rep = hidden_states[batch_indices, seq_lengths]
    else:
        # Default: mean pooling over prompt tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        rep = sum_embeddings / sum_mask

    # L2 normalize for contrastive cosine geometry
    norm_rep = F.normalize(rep, p=2, dim=-1)
    return norm_rep


class SupervisedContrastiveLoss(nn.Module):
    """Numerically stable Supervised Contrastive Loss (Khosla et al., 2020).

    Computes:
      L_con = - 1/|P(i)| sum_{p in P(i)} log( exp(s(z_i, z_p)/tau) / sum_{a != i} exp(s(z_i, z_a)/tau) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

        embeddings: (N, D) L2-normalized representations
        labels: (N,) cluster identifier integers
        negative_mask: optional (N, N) bool mask indicating valid negatives (for hard negative contrasts)

        Returns:
            Scalar contrastive loss
        """
        device = embeddings.device
        N = embeddings.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pairwise cosine similarities: s_ij = z_i^T z_j / tau
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask: M_ij = 1 if labels[i] == labels[j] and i != j
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        diag_mask = torch.eye(N, dtype=torch.float32, device=device)
        pos_mask = pos_mask * (1.0 - diag_mask)

        # If no positive pairs exist in the batch, return zero loss
        num_pos_per_row = pos_mask.sum(dim=1)
        valid_rows = num_pos_per_row > 0
        if not valid_rows.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # For numerical stability: subtract max per row
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator mask: exclude self (diagonal)
        denom_mask = 1.0 - diag_mask
        if negative_mask is not None:
            denom_mask = denom_mask * negative_mask.float()

        exp_logits = torch.exp(logits) * denom_mask
        denom = exp_logits.sum(dim=1, keepdim=True) + 1e-9
        log_prob = logits - torch.log(denom)

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / torch.clamp(num_pos_per_row, min=1.0)
        loss = -mean_log_prob_pos[valid_rows].mean()
        return loss


class RepresentationPreservationLoss(nn.Module):
    """Penalizes deviation of canonical prompt representations from frozen base representations."""

    def __init__(self):
        super().__init__()

    def forward(self, edited_rep: torch.Tensor, base_rep: torch.Tensor) -> torch.Tensor:
        """Cosine distance loss: 1 - cosine_sim(edited, base)."""
        cos_sim = F.cosine_similarity(edited_rep, base_rep, dim=-1)
        return torch.mean(1.0 - cos_sim)


class ContrastiveBatchDataset(Dataset):
    """Dataset producing batches of prompt variants and hard negatives for contrastive training."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, max_len: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        return {
            "text": item["text"],
            "cluster_label": item["cluster_int_id"],
            "is_negative": item.get("is_negative", False),
        }


def collate_contrastive_batch(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_len: int = 256,
) -> Dict[str, torch.Tensor]:
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["cluster_label"] for b in batch], dtype=torch.long)
    is_neg = torch.tensor([b["is_negative"] for b in batch], dtype=torch.bool)

    tokenizer.padding_side = "left"
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "is_negative": is_neg,
    }
