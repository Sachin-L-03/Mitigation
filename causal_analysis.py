"""Causal layer analysis and activation patching between prompt variants.

Measures within-cluster representation divergence, inter-cluster separation,
and executes causal activation patching to identify and rank candidate intervention layers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import get_transformer_layers

logger = logging.getLogger("prompt_sensitivity.causal")


def compute_cosine_divergence(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute 1 - cosine_similarity between two 1D vectors."""
    v1 = vec_a.view(1, -1).float()
    v2 = vec_b.view(1, -1).float()
    cos_sim = F.cosine_similarity(v1, v2).item()
    return float(1.0 - cos_sim)


def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Compute KL divergence KL(Softmax(logits_p) || Softmax(logits_q))."""
    p = F.softmax(logits_p.float().squeeze(0), dim=-1)
    log_p = F.log_softmax(logits_p.float().squeeze(0), dim=-1)
    log_q = F.log_softmax(logits_q.float().squeeze(0), dim=-1)
    kl = (p * (log_p - log_q)).sum().item()
    return float(max(0.0, kl))


def compute_layer_divergences(
    cluster_records: List[Dict[str, Any]],
    target_layers: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute within-cluster and between-cluster representation divergences across layers.

    Args:
        cluster_records: List of dictionaries loaded from activation cache for clusters
        target_layers: Layers to evaluate

    Returns:
        Dict mapping layer -> {'within_divergence': float, 'between_separation': float, 'ratio': float}
    """
    layer_intra: Dict[int, List[float]] = {l: [] for l in target_layers}
    layer_inter: Dict[int, List[float]] = {l: [] for l in target_layers}

    # Group by cluster
    by_cluster: Dict[str, Dict[str, torch.Tensor]] = {}
    for item in cluster_records:
        cid = item["cluster_id"]
        vtype = item["variant_type"]
        layers_dict = item["layers"]
        by_cluster.setdefault(cid, {})[vtype] = layers_dict

    cluster_ids = list(by_cluster.keys())

    for cid, variants in by_cluster.items():
        if "original" not in variants:
            continue
        orig_layers = variants["original"]

        # Within-cluster divergence against original
        for vtype, vlayers in variants.items():
            if vtype == "original":
                continue
            for lyr in target_layers:
                if lyr in orig_layers and lyr in vlayers:
                    div = compute_cosine_divergence(orig_layers[lyr], vlayers[lyr])
                    layer_intra[lyr].append(div)

    # Between-cluster separation
    for i in range(len(cluster_ids)):
        for j in range(i + 1, min(i + 5, len(cluster_ids))):
            cid_a, cid_b = cluster_ids[i], cluster_ids[j]
            if "original" in by_cluster[cid_a] and "original" in by_cluster[cid_b]:
                orig_a = by_cluster[cid_a]["original"]
                orig_b = by_cluster[cid_b]["original"]
                for lyr in target_layers:
                    if lyr in orig_a and lyr in orig_b:
                        div = compute_cosine_divergence(orig_a[lyr], orig_b[lyr])
                        layer_inter[lyr].append(div)

    results: Dict[int, Dict[str, float]] = {}
    for lyr in target_layers:
        mean_intra = float(np.mean(layer_intra[lyr])) if layer_intra[lyr] else 0.0
        mean_inter = float(np.mean(layer_inter[lyr])) if layer_inter[lyr] else 0.0
        ratio = mean_inter / (mean_intra + 1e-9)
        results[lyr] = {
            "within_cluster_divergence": round(mean_intra, 5),
            "between_cluster_separation": round(mean_inter, 5),
            "separation_ratio": round(ratio, 4),
        }

    return results


class CausalPatcher:
    """Executes residual-stream activation patching between original and paraphrased prompts."""

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_transformer_layers(model)

    def get_last_token_activation_and_logits(
        self,
        text: str,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and extract activation at last token position plus output logits."""
        inputs = self.tokenizer(text, return_tensors="pt").to(next(self.model.parameters()).device)
        captured_activation: List[torch.Tensor] = []

        target_layer = self.layers[layer_idx]

        def hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
            hs = out[0] if isinstance(out, tuple) else out
            last_idx = inputs["attention_mask"].sum().long() - 1
            captured_activation.append(hs[0, last_idx, :].detach())

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :].detach().cpu()
            if not captured_activation:
                raise RuntimeError("Failed to capture activation in hook.")
            return captured_activation[0], logits
        finally:
            handle.remove()

    def run_patched_forward(
        self,
        text: str,
        layer_idx: int,
        patch_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass with target layer's last token replaced by patch_vector."""
        inputs = self.tokenizer(text, return_tensors="pt").to(next(self.model.parameters()).device)
        target_layer = self.layers[layer_idx]

        def patch_hook(module: nn.Module, inp: Any, out: Any) -> Any:
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            last_idx = inputs["attention_mask"].sum().long() - 1
            hs[0, last_idx, :] = patch_vector.to(device=hs.device, dtype=hs.dtype)
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs

        handle = target_layer.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :].detach().cpu()
            return logits
        finally:
            handle.remove()


def evaluate_layer_causal_patching(
    patcher: CausalPatcher,
    cluster_pairs: List[Dict[str, str]],
    candidate_layers: List[int],
    sham_control: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Evaluate candidate layers by patching original prompt activations into paraphrases.

    Args:
        patcher: Initialized CausalPatcher
        cluster_pairs: List of dicts with keys 'original_text', 'variant_text', 'cluster_id'
        candidate_layers: List of layer indices to test
        sham_control: Whether to include sham noise baseline

    Returns:
        Dict mapping layer -> Causal metrics and ranking score
    """
    layer_scores: Dict[int, List[float]] = {l: [] for l in candidate_layers}
    layer_sham_scores: Dict[int, List[float]] = {l: [] for l in candidate_layers}

    for pair_idx, pair in enumerate(cluster_pairs):
        orig_text = pair["original_text"]
        var_text = pair["variant_text"]

        for lyr in candidate_layers:
            try:
                # 1. Clean original activation and baseline logits
                orig_act, orig_logits = patcher.get_last_token_activation_and_logits(orig_text, lyr)

                # 2. Clean variant logits (unpatched)
                _, var_logits = patcher.get_last_token_activation_and_logits(var_text, lyr)
                baseline_kl = compute_kl_divergence(orig_logits, var_logits)

                # 3. Patched forward pass
                patched_logits = patcher.run_patched_forward(var_text, lyr, orig_act)
                patched_kl = compute_kl_divergence(orig_logits, patched_logits)

                # Causal restoration: how much did patching reduce divergence toward original
                restoration = baseline_kl - patched_kl
                layer_scores[lyr].append(restoration)

                # 4. Optional sham control (Gaussian noise vector matched to activation norm)
                if sham_control:
                    norm = orig_act.norm().item()
                    sham_act = torch.randn_like(orig_act)
                    sham_act = sham_act / (sham_act.norm() + 1e-9) * norm
                    sham_logits = patcher.run_patched_forward(var_text, lyr, sham_act)
                    sham_kl = compute_kl_divergence(orig_logits, sham_logits)
                    sham_restoration = baseline_kl - sham_kl
                    layer_sham_scores[lyr].append(sham_restoration)

            except Exception as e:
                logger.warning(f"Error evaluating pair {pair_idx} at layer {lyr}: {e}")

    results: Dict[int, Dict[str, Any]] = {}
    for lyr in candidate_layers:
        scores = layer_scores[lyr]
        mean_restoration = float(np.mean(scores)) if scores else 0.0
        std_restoration = float(np.std(scores)) if scores else 0.0

        sham_mean = float(np.mean(layer_sham_scores[lyr])) if layer_sham_scores[lyr] else 0.0
        causal_specificity = mean_restoration - sham_mean

        results[lyr] = {
            "mean_causal_restoration": round(mean_restoration, 5),
            "std_causal_restoration": round(std_restoration, 5),
            "sham_mean_restoration": round(sham_mean, 5),
            "causal_specificity": round(causal_specificity, 5),
            "evaluated_pairs": len(scores),
        }

    return results


def rank_layers_by_causality(causal_results: Dict[int, Dict[str, Any]]) -> List[Tuple[int, float]]:
    """Rank layers in descending order of causal specificity / restoration score."""
    ranked = sorted(
        causal_results.items(),
        key=lambda item: item[1]["causal_specificity"],
        reverse=True,
    )
    return [(k, v["causal_specificity"]) for k, v in ranked]
