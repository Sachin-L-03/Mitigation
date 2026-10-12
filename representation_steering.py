"""Subspace-suppression representation steering for large language models.

Estimates sensitivity subspaces from training activations via SVD and applies
non-invasive inference-time suppression hooks to reduce sensitivity to prompt variations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from activations import get_transformer_layers

logger = logging.getLogger("prompt_sensitivity.steering")


def estimate_sensitivity_subspace(
    train_records: List[Dict[str, Any]],
    target_layer: int,
    rank: int = 2,
    save_path: Optional[str | Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate sensitive nuisance subspace from training prompt activation variations.

    Computes difference vectors between variants and original prompts within clusters,
    then computes principal components via SVD.

    Returns:
        projection_matrix: Projection operator P = V_k V_k^T of shape (hidden_dim, hidden_dim)
        singular_values: Top singular values indicating explained variance
    """
    by_cluster: Dict[str, Dict[str, torch.Tensor]] = {}
    for item in train_records:
        cid = item["cluster_id"]
        vtype = item["variant_type"]
        layers = item["layers"]
        if target_layer in layers:
            by_cluster.setdefault(cid, {})[vtype] = layers[target_layer].float()

    diff_vectors: List[torch.Tensor] = []
    for cid, variants in by_cluster.items():
        if "original" not in variants:
            continue
        orig_vec = variants["original"].view(-1)
        for vtype, var_tensor in variants.items():
            if vtype == "original":
                continue
            var_vec = var_tensor.view(-1)
            diff = var_vec - orig_vec
            diff_vectors.append(diff)

    if not diff_vectors:
        raise ValueError(f"No variant activation pairs found at target layer {target_layer}.")

    D = torch.stack(diff_vectors, dim=0)  # Shape: (num_differences, hidden_dim)
    # Mean-center difference vectors
    D_centered = D - D.mean(dim=0, keepdim=True)

    # Perform SVD
    # D_centered: (N, d) -> U (N, k), S (k), Vh (k, d)
    U, S, Vh = torch.linalg.svd(D_centered, full_matrices=False)

    k = min(rank, Vh.shape[0], Vh.shape[1])
    Vk = Vh[:k, :].T  # Shape: (hidden_dim, k)

    # Projection matrix P = Vk Vk^T (symmetric, shape hidden_dim x hidden_dim)
    P = torch.mm(Vk, Vk.T)

    logger.info(
        f"Estimated subspace for layer {target_layer}: rank={k}, "
        f"top singular values={S[:k].tolist()}"
    )

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "target_layer": target_layer,
            "rank": k,
            "projection_matrix": P,
            "singular_values": S[:k],
            "num_differences": len(diff_vectors),
        }, out)
        logger.info(f"Saved subspace projection operator to {out}")

    return P, S[:k]


class SubspaceSuppressionHook:
    """Non-invasive forward hook that suppresses sensitive subspace components during generation."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: int,
        projection_matrix: torch.Tensor,
        steering_strength: float = 0.5,
        token_position: str = "last",
    ):
        self.model = model
        self.target_layer = target_layer
        self.steering_strength = float(steering_strength)
        self.token_position = token_position

        self.layers = get_transformer_layers(model)
        if target_layer < 0 or target_layer >= len(self.layers):
            raise ValueError(f"Target layer {target_layer} out of range [0, {len(self.layers) - 1}].")

        self.projection_matrix = projection_matrix
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.active: bool = False

    def hook_fn(self, module: nn.Module, inp: Any, out: Any) -> Any:
        if not self.active or self.steering_strength <= 0.0:
            return out

        hs = out[0] if isinstance(out, tuple) else out
        device = hs.device
        dtype = hs.dtype

        P = self.projection_matrix.to(device=device, dtype=torch.float32)

        hs_float = hs.clone().to(torch.float32)
        batch_size, seq_len, hidden_dim = hs_float.shape

        if self.token_position == "last":
            # Suppress sensitivity strictly at the last sequence position
            last_tokens = hs_float[:, -1, :]  # Shape: (batch_size, hidden_dim)
            # Projected component: last_tokens @ P
            proj_component = torch.matmul(last_tokens, P)
            steered_last = last_tokens - (self.steering_strength * proj_component)
            hs_float[:, -1, :] = steered_last

        elif self.token_position == "all":
            # Suppress across all tokens
            flat_hs = hs_float.view(-1, hidden_dim)
            proj_component = torch.matmul(flat_hs, P)
            steered_flat = flat_hs - (self.steering_strength * proj_component)
            hs_float = steered_flat.view(batch_size, seq_len, hidden_dim)

        hs_out = hs_float.to(dtype=dtype)
        if isinstance(out, tuple):
            return (hs_out,) + out[1:]
        return hs_out

    def attach(self) -> None:
        """Attach the suppression hook to the target transformer layer."""
        if self.hook_handle is None:
            target_module = self.layers[self.target_layer]
            self.hook_handle = target_module.register_forward_hook(self.hook_fn)
            self.active = True

    def remove(self) -> None:
        """Safely detach and clean up the hook."""
        self.active = False
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def generate_with_steering(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    target_layer: int,
    projection_matrix: torch.Tensor,
    steering_strength: float = 0.5,
    max_new_tokens: int = 64,
    token_position: str = "last",
) -> Dict[str, Any]:
    """Execute model generation with active representation steering hook."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    hook = SubspaceSuppressionHook(
        model=model,
        target_layer=target_layer,
        projection_matrix=projection_matrix,
        steering_strength=steering_strength,
        token_position=token_position,
    )

    with hook:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return {
        "prompt": prompt,
        "generated_text": gen_text,
        "tokens_generated": len(gen_tokens),
        "steering_layer": target_layer,
        "steering_strength": steering_strength,
    }
