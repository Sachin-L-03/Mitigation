"""Activation extraction, layer hooks, and incremental disk caching for transformer models.

Supports Qwen2.5, LLaMA-3.1, and compatible architectures with memory-safe incremental extraction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("prompt_sensitivity.activations")


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Safely locate the list of transformer decoder layers across base and PEFT models."""
    curr = model
    # Repeatedly unwrap PEFT, LoRA, and HF CausalLM wrappers until layers are located
    for _ in range(10):
        if hasattr(curr, "layers") and isinstance(curr.layers, (nn.ModuleList, list)):
            return curr.layers
        if hasattr(curr, "decoder") and hasattr(curr.decoder, "layers"):
            return curr.decoder.layers
        if hasattr(curr, "h") and isinstance(curr.h, (nn.ModuleList, list)):
            return curr.h
        if hasattr(curr, "base_model"):
            curr = curr.base_model
            continue
        if hasattr(curr, "model"):
            curr = curr.model
            continue
        if hasattr(curr, "transformer"):
            curr = curr.transformer
            continue
        break

    raise AttributeError(f"Unable to locate transformer layers in model of type {type(model)}")


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    token_position: str = "last",
) -> torch.Tensor:
    """Pool hidden states based on attention mask and chosen token position.

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_len) with 1s for real tokens
        token_position: 'last' (last valid token), 'mean' (masked mean), or 'all'

    Returns:
        Tensor of shape (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim)
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    if token_position == "last":
        # Find index of last token for each item in batch
        last_indices = attention_mask.sum(dim=-1).long() - 1
        last_indices = last_indices.clamp(min=0, max=seq_len - 1)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[batch_indices, last_indices, :]

    elif token_position == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        sum_hidden = (hidden_states * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / lengths

    elif token_position == "all":
        return hidden_states

    else:
        raise ValueError(f"Unsupported token_position '{token_position}'. Choose 'last', 'mean', or 'all'.")


class ActivationExtractor:
    """Manages forward hooks to extract activations from specified layers with zero memory leaks."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[int],
        token_position: str = "last",
    ):
        self.model = model
        self.target_layers = sorted(target_layers)
        self.token_position = token_position
        self.layers = get_transformer_layers(model)

        total_layers = len(self.layers)
        for lyr in self.target_layers:
            if lyr < 0 or lyr >= total_layers:
                raise ValueError(f"Target layer {lyr} out of range [0, {total_layers - 1}].")

        self.storage: Dict[int, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self) -> None:
        self._remove_hooks()
        self.storage.clear()

        for lyr_idx in self.target_layers:
            target_module = self.layers[lyr_idx]

            def make_hook(idx: int) -> Callable:
                def hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
                    hs = out[0] if isinstance(out, tuple) else out
                    # Detach and move to CPU immediately to preserve GPU VRAM
                    self.storage[idx] = hs.detach().cpu()
                return hook_fn

            handle = target_module.register_forward_hook(make_hook(lyr_idx))
            self.hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Run forward pass and return pooled hidden states for target layers.

        Returns:
            Dict mapping layer_index -> pooled Tensor (batch_size, hidden_dim) on CPU.
        """
        self._register_hooks()
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)

            results: Dict[int, torch.Tensor] = {}
            cpu_mask = attention_mask.cpu()

            for lyr_idx in self.target_layers:
                raw_hs = self.storage.get(lyr_idx)
                if raw_hs is None:
                    raise RuntimeError(f"Hook failed to capture activation for layer {lyr_idx}.")
                pooled = pool_hidden_states(raw_hs, cpu_mask, token_position=self.token_position)
                results[lyr_idx] = pooled

            return results
        finally:
            self._remove_hooks()
            self.storage.clear()


class ActivationCacheManager:
    """Handles incremental on-disk caching of activations to avoid GPU memory overflow."""

    def __init__(self, cache_dir: str | Path, model_name: str):
        safe_model_name = model_name.replace("/", "_")
        self.cache_dir = Path(cache_dir) / safe_model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "cache_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def get_cache_path(self, prompt_id: str) -> Path:
        return self.cache_dir / f"{prompt_id}.pt"

    def is_cached(self, prompt_id: str) -> bool:
        return prompt_id in self.manifest and self.get_cache_path(prompt_id).is_file()

    def save_prompt_activations(
        self,
        prompt_id: str,
        cluster_id: str,
        variant_type: str,
        activations_by_layer: Dict[int, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save activations for a single prompt to a standalone .pt file."""
        target_file = self.get_cache_path(prompt_id)

        # Store tensors as fp16 or bf16 on CPU for compact disk footprint
        data = {
            "prompt_id": prompt_id,
            "cluster_id": cluster_id,
            "variant_type": variant_type,
            "layers": {k: v.to(torch.float32) for k, v in activations_by_layer.items()},
            "metadata": metadata or {},
        }
        torch.save(data, target_file)

        self.manifest[prompt_id] = {
            "cluster_id": cluster_id,
            "variant_type": variant_type,
            "layers": list(activations_by_layer.keys()),
            "file": str(target_file.name),
        }
        self._save_manifest()
        return target_file

    def load_prompt_activations(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        target_file = self.get_cache_path(prompt_id)
        if not target_file.is_file():
            return None
        return torch.load(target_file, map_location="cpu", weights_only=False)

    def load_cluster_activations(self, cluster_id: str) -> Dict[str, Dict[str, Any]]:
        """Load all cached prompt activations belonging to a specific cluster."""
        results: Dict[str, Dict[str, Any]] = {}
        for pid, meta in self.manifest.items():
            if meta.get("cluster_id") == cluster_id:
                data = self.load_prompt_activations(pid)
                if data:
                    results[pid] = data
        return results
