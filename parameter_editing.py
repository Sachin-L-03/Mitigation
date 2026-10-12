"""CREME-inspired layer-aware targeted parameter editing.

Learns low-rank parameter updates strictly in causally identified transformer layers,
using base-model anchor responses as pseudo-teachers with preservation regularization.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("prompt_sensitivity.parameter_editing")


def generate_and_cache_pseudo_targets(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_records: List[Any],
    cache_path: str | Path,
    max_new_tokens: int = 64,
) -> Dict[str, Dict[str, Any]]:
    """Generate and cache base-model responses on original prompts to serve as pseudo-targets."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                logger.info(f"Loaded {len(cached)} existing pseudo-targets from {path}")
                return cached
        except Exception:
            pass

    targets: Dict[str, Dict[str, Any]] = {}
    model.eval()
    device = next(model.parameters()).device

    # Extract all original prompts
    orig_records = [r for r in train_records if getattr(r, "variant_type", "") == "original"]
    logger.info(f"Generating pseudo-targets for {len(orig_records)} training anchor prompts...")

    for rec in orig_records:
        cid = rec.cluster_id
        prompt_text = rec.prompt_text
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        targets[cid] = {
            "cluster_id": cid,
            "prompt_id": rec.prompt_id,
            "original_prompt": prompt_text,
            "pseudo_target": gen_text,
            "token_count": len(gen_tokens),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)

    logger.info(f"Cached {len(targets)} pseudo-targets to {path}")
    return targets


def build_layer_aware_lora_model(
    base_model: PreTrainedModel,
    target_layers: List[int],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Any:
    """Create a PEFT LoRA model restricting adapters strictly to causally selected layers."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Restrict layers_to_transform strictly to the causally identified layers
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
        layers_to_transform=sorted(target_layers),
    )

    # Ensure model is ready for k-bit training if quantized
    if getattr(base_model, "is_loaded_in_4bit", False) or getattr(base_model, "is_loaded_in_8bit", False):
        base_model = prepare_model_for_kbit_training(base_model)

    peft_model = get_peft_model(base_model, lora_config)
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"Layer-aware LoRA created on layers {target_layers}: "
        f"{trainable_params:,} trainable / {all_params:,} total parameters ({100 * trainable_params / all_params:.3f}%)"
    )
    return peft_model


class PromptEditingDataset(Dataset):
    """PyTorch Dataset preparing prompt-completion pairs for consistency and preservation training."""

    def __init__(
        self,
        records: List[Any],
        pseudo_targets: Dict[str, Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples: List[Dict[str, Any]] = []

        # Group records by cluster
        clusters: Dict[str, List[Any]] = {}
        for r in records:
            clusters.setdefault(r.cluster_id, []).append(r)

        cluster_list = list(clusters.keys())

        for cid, recs in clusters.items():
            if cid not in pseudo_targets:
                continue
            anchor_resp = pseudo_targets[cid]["pseudo_target"]
            if not anchor_resp:
                continue

            # Find negative target for contrastive discrimination
            neg_cid = cluster_list[(cluster_list.index(cid) + 1) % len(cluster_list)] if len(cluster_list) > 1 else cid
            neg_resp = pseudo_targets.get(neg_cid, {}).get("pseudo_target", "")

            orig_rec = next((r for r in recs if getattr(r, "variant_type", "") == "original"), None)
            var_recs = [r for r in recs if getattr(r, "variant_type", "") != "original"]

            if orig_rec:
                self.samples.append({
                    "cluster_id": cid,
                    "prompt_text": orig_rec.prompt_text,
                    "target_text": anchor_resp,
                    "is_original": True,
                    "negative_target": neg_resp,
                })

            for vr in var_recs:
                self.samples.append({
                    "cluster_id": cid,
                    "prompt_text": vr.prompt_text,
                    "target_text": anchor_resp,
                    "is_original": False,
                    "negative_target": neg_resp,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        prompt = item["prompt_text"]
        target = item["target_text"]
        full_text = f"{prompt} {target}"

        enc_prompt = self.tokenizer(prompt, add_special_tokens=True)
        enc_full = self.tokenizer(full_text, add_special_tokens=True, max_length=self.max_seq_length, truncation=True)

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        # Mask prompt tokens so loss is only computed on target tokens
        prompt_len = len(enc_prompt["input_ids"])
        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_original": item["is_original"],
        }


def collate_editing_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pad batch of variable length sequences."""
    pad_id = 0
    max_len = max(len(b["input_ids"]) for b in batch)

    batch_ids = []
    batch_masks = []
    batch_labels = []
    is_original_list = []

    for b in batch:
        pad_size = max_len - len(b["input_ids"])
        batch_ids.append(torch.cat([b["input_ids"], torch.full((pad_size,), pad_id, dtype=torch.long)]))
        batch_masks.append(torch.cat([b["attention_mask"], torch.full((pad_size,), 0, dtype=torch.long)]))
        batch_labels.append(torch.cat([b["labels"], torch.full((pad_size,), -100, dtype=torch.long)]))
        is_original_list.append(b["is_original"])

    return {
        "input_ids": torch.stack(batch_ids),
        "attention_mask": torch.stack(batch_masks),
        "labels": torch.stack(batch_labels),
        "is_original": torch.tensor(is_original_list, dtype=torch.bool),
    }


def compute_editing_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-component loss: Consistency, Preservation, and Parameter Regularization.

    Loss Terms:
      1. consistency_loss: Cross-entropy of paraphrases generating the anchor completion.
      2. preservation_loss: Cross-entropy of original prompt generating anchor completion.
      3. param_reg_loss: L2 penalty on trainable LoRA adapter parameters.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    is_original = batch["is_original"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift logits and labels for next-token prediction loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    # Token-level losses: shape (batch_size, seq_len - 1)
    batch_size, seq_len, vocab_size = shift_logits.shape
    token_losses = loss_fn(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    token_losses = token_losses.view(batch_size, seq_len)

    # Per-sample loss averaged over non-ignored tokens
    non_ignored = (shift_labels != -100).float()
    sample_losses = (token_losses * non_ignored).sum(dim=-1) / non_ignored.sum(dim=-1).clamp(min=1.0)

    # Split into Consistency (variants) and Preservation (originals)
    var_mask = ~is_original
    consistency_loss = sample_losses[var_mask].mean() if var_mask.any() else torch.tensor(0.0, device=device)
    preservation_loss = sample_losses[is_original].mean() if is_original.any() else torch.tensor(0.0, device=device)

    # Parameter regularization: L2 norm of trainable weights
    l2_reg = torch.tensor(0.0, device=device)
    for p in model.parameters():
        if p.requires_grad:
            l2_reg = l2_reg + (p ** 2).sum()

    w_cons = loss_weights.get("consistency", 1.0)
    w_pres = loss_weights.get("preservation", 1.0)
    w_reg = loss_weights.get("l2_reg", 0.001)

    total_loss = w_cons * consistency_loss + w_pres * preservation_loss + w_reg * l2_reg

    metrics = {
        "total_loss": total_loss.item(),
        "consistency_loss": consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
        "preservation_loss": preservation_loss.item() if isinstance(preservation_loss, torch.Tensor) else preservation_loss,
        "param_reg_loss": l2_reg.item() if isinstance(l2_reg, torch.Tensor) else l2_reg,
    }

    return total_loss, metrics
