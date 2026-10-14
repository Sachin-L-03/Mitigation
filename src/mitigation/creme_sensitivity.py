"""CREME-inspired layer-aware sensitivity analysis and causal patching.

Analyzes transformer representations to identify which layers exhibit high
sensitivity to semantically irrelevant prompt variations while maintaining task separation.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import get_transformer_layers
from causal_analysis import (
    CausalPatcher,
    compute_cosine_divergence,
    compute_kl_divergence,
    evaluate_layer_causal_patching,
    rank_layers_by_causality,
)

logger = logging.getLogger("prompt_sensitivity.mitigation.creme")


def analyze_layer_sensitivity(
    model: nn.Module,
    tokenizer: Any,
    clusters: List[Dict[str, Any]],
    candidate_layers: List[int],
    output_csv_path: Optional[str | Path] = None,
) -> Dict[int, Dict[str, float]]:
    """Compute layer-level sensitivity metrics across candidate layers.

    Metrics per layer:
      1. mean_equiv_distance: Mean normalized activation cosine distance for equivalent pairs
      2. mean_equiv_similarity: Mean cosine similarity for equivalent pairs (1 - distance)
      3. within_cluster_variance: Variance of representations across equivalent variants
      4. between_cluster_separation: Mean cosine distance between different clusters
      5. fisher_ratio: between_cluster_separation / (within_cluster_variance + 1e-9)
    """
    model.eval()
    device = next(model.parameters()).device
    layers = get_transformer_layers(model)

    layer_equiv_dists: Dict[int, List[float]] = {l: [] for l in candidate_layers}
    layer_cluster_reps: Dict[int, Dict[str, List[torch.Tensor]]] = {l: {} for l in candidate_layers}

    for c in clusters:
        cid = c["cluster_id"]
        variants = c.get("variants", [])
        if not variants:
            continue

        for v in variants:
            text = v["text"] if isinstance(v, dict) else getattr(v, "text", "")
            if not text:
                continue

            inputs = tokenizer(text, return_tensors="pt").to(device)

            for lyr_idx in candidate_layers:
                captured: List[torch.Tensor] = []

                def hook_fn(mod: nn.Module, inp: Any, out: Any) -> None:
                    hs = out[0] if isinstance(out, tuple) else out
                    # Mean-pool over prompt tokens
                    mask = inputs["attention_mask"].unsqueeze(-1).expand_as(hs).float()
                    sum_hs = (hs * mask).sum(dim=1)
                    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = F.normalize(sum_hs / sum_mask, p=2, dim=-1)
                    captured.append(pooled[0].detach().cpu())

                handle = layers[lyr_idx].register_forward_hook(hook_fn)
                try:
                    with torch.no_grad():
                        model(**inputs)
                    if captured:
                        layer_cluster_reps[lyr_idx].setdefault(cid, []).append(captured[0])
                finally:
                    handle.remove()

    # Calculate metrics
    layer_metrics: Dict[int, Dict[str, float]] = {}

    for lyr_idx in candidate_layers:
        within_dists = []
        cluster_means = []

        for cid, reps in layer_cluster_reps[lyr_idx].items():
            if len(reps) >= 2:
                # Pairwise distance within cluster
                for i in range(len(reps)):
                    for j in range(i + 1, len(reps)):
                        cos_sim = float(F.cosine_similarity(reps[i].unsqueeze(0), reps[j].unsqueeze(0)).item())
                        within_dists.append(1.0 - cos_sim)
            if reps:
                c_mean = torch.stack(reps).mean(dim=0)
                cluster_means.append(c_mean)

        mean_equiv_dist = float(np.mean(within_dists)) if within_dists else 0.0
        var_within = float(np.var(within_dists)) if within_dists else 0.0

        # Between-cluster separation
        between_dists = []
        if len(cluster_means) >= 2:
            for i in range(len(cluster_means)):
                for j in range(i + 1, min(i + 6, len(cluster_means))):
                    sim = float(F.cosine_similarity(cluster_means[i].unsqueeze(0), cluster_means[j].unsqueeze(0)).item())
                    between_dists.append(1.0 - sim)

        mean_between = float(np.mean(between_dists)) if between_dists else 0.0
        fisher_ratio = mean_between / (mean_equiv_dist + 1e-9)

        layer_metrics[lyr_idx] = {
            "layer": lyr_idx,
            "mean_equiv_distance": round(mean_equiv_dist, 5),
            "mean_equiv_similarity": round(1.0 - mean_equiv_dist, 5),
            "within_cluster_variance": round(var_within, 6),
            "between_cluster_separation": round(mean_between, 5),
            "fisher_ratio": round(fisher_ratio, 4),
        }

    # Save CSV if requested
    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Layer",
                "Mean Equiv Distance",
                "Mean Equiv Similarity",
                "Within Variance",
                "Between Separation",
                "Fisher Ratio",
            ])
            for lyr in candidate_layers:
                m = layer_metrics[lyr]
                writer.writerow([
                    m["layer"],
                    m["mean_equiv_distance"],
                    m["mean_equiv_similarity"],
                    m["within_cluster_variance"],
                    m["between_cluster_separation"],
                    m["fisher_ratio"],
                ])
        logger.info(f"Saved layer sensitivity metrics to {output_csv_path}")

    return layer_metrics
