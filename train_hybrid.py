"""Training pipeline for CREME-inspired layer-aware parameter editing.

Executes low-rank parameter optimization on causally selected layers with gradient accumulation,
validation checkpointing, and preservation regularization.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import PromptDataset
from model import load_model_and_tokenizer
from parameter_editing import (
    PromptEditingDataset,
    build_layer_aware_lora_model,
    collate_editing_batch,
    compute_editing_loss,
    generate_and_cache_pseudo_targets,
)
from utils import load_and_validate_config, set_seed, setup_logger

logger = logging.getLogger("prompt_sensitivity.train_hybrid")


def train_parameter_editor(
    config: Dict[str, Any],
    target_layers: Optional[List[int]] = None,
    resume_from_checkpoint: bool = True,
) -> Dict[str, Any]:
    """Train layer-aware parameter editor on training data using pseudo-target supervision.

    Args:
        config: Loaded project configuration
        target_layers: Optional explicit layers to edit (defaults to config target_layers)
        resume_from_checkpoint: Whether to skip training if best_adapter already exists

    Returns:
        Summary dict containing best checkpoint path, training metrics, and layer info.
    """
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    edit_cfg = config.get("parameter_editing", {})
    output_dir = Path(edit_cfg.get("output_dir", "outputs/parameter_editing"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_adapter_dir = output_dir / "best_adapter"

    if resume_from_checkpoint and (best_adapter_dir / "adapter_model.safetensors").is_file():
        logger.info(f"Existing adapter checkpoint discovered at {best_adapter_dir}. Skipping re-training.")
        history_file = output_dir / "training_history.json"
        history = {}
        if history_file.is_file():
            with open(history_file, "r") as f:
                history = json.load(f)
        return {"status": "reused_checkpoint", "adapter_dir": str(best_adapter_dir), "history": history}

    layers_to_edit = target_layers or edit_cfg.get("target_layers", [12, 16, 20])
    logger.info(f"Initializing parameter editing on layers: {layers_to_edit}")

    # 1. Load dataset splits
    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "data/splits"))
    train_dataset = PromptDataset.load_jsonl(splits_dir / "train.jsonl")
    val_dataset = PromptDataset.load_jsonl(splits_dir / "val.jsonl")

    # 2. Load model and tokenizer
    model_name = config["model"]["name"]
    logger.info(f"Loading base model '{model_name}' for training...")
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    base_model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    # 3. Generate & cache pseudo-targets on training anchor prompts
    pseudo_target_cache = output_dir / "pseudo_targets.json"
    pseudo_targets = generate_and_cache_pseudo_targets(
        model=base_model,
        tokenizer=tokenizer,
        train_records=train_dataset.records,
        cache_path=pseudo_target_cache,
        max_new_tokens=config.get("generation", {}).get("max_new_tokens", 64),
    )

    # Also generate validation pseudo-targets for evaluation
    val_pseudo_targets = generate_and_cache_pseudo_targets(
        model=base_model,
        tokenizer=tokenizer,
        train_records=val_dataset.records,
        cache_path=output_dir / "val_pseudo_targets.json",
        max_new_tokens=config.get("generation", {}).get("max_new_tokens", 64),
    )

    # 4. Wrap with layer-aware LoRA
    peft_model = build_layer_aware_lora_model(
        base_model=base_model,
        target_layers=layers_to_edit,
        r=edit_cfg.get("r", 16),
        lora_alpha=edit_cfg.get("lora_alpha", 32),
        lora_dropout=edit_cfg.get("lora_dropout", 0.05),
    )

    # 5. Prepare PyTorch datasets
    train_ds = PromptEditingDataset(train_dataset.records, pseudo_targets, tokenizer)
    val_ds = PromptEditingDataset(val_dataset.records, val_pseudo_targets, tokenizer)

    batch_size = edit_cfg.get("batch_size", 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_editing_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_editing_batch)

    # 6. Optimizer and scheduler
    lr = edit_cfg.get("learning_rate", 1e-4)
    optimizer = AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=lr)

    epochs = edit_cfg.get("epochs", 3)
    grad_accum_steps = edit_cfg.get("gradient_accumulation_steps", 8)
    loss_weights = edit_cfg.get("loss_weights", {"consistency": 1.0, "preservation": 1.0, "l2_reg": 0.001})

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")

    logger.info(f"Starting layer-aware editing training: {epochs} epochs, {len(train_loader)} steps/epoch")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        peft_model.train()
        train_epoch_loss = 0.0
        train_cons_loss = 0.0
        train_pres_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            loss, loss_dict = compute_editing_loss(peft_model, batch, loss_weights)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            train_epoch_loss += loss_dict["total_loss"]
            train_cons_loss += loss_dict["consistency_loss"]
            train_pres_loss += loss_dict["preservation_loss"]

            if step % grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_train_cons = train_cons_loss / len(train_loader)
        avg_train_pres = train_pres_loss / len(train_loader)

        # Validation evaluation
        peft_model.eval()
        val_epoch_loss = 0.0
        val_cons_loss = 0.0
        val_pres_loss = 0.0

        with torch.no_grad():
            for v_batch in val_loader:
                v_loss, v_loss_dict = compute_editing_loss(peft_model, v_batch, loss_weights)
                val_epoch_loss += v_loss_dict["total_loss"]
                val_cons_loss += v_loss_dict["consistency_loss"]
                val_pres_loss += v_loss_dict["preservation_loss"]

        avg_val_loss = val_epoch_loss / max(1, len(val_loader))
        avg_val_cons = val_cons_loss / max(1, len(val_loader))
        avg_val_pres = val_pres_loss / max(1, len(val_loader))

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_consistency": round(avg_train_cons, 4),
            "train_preservation": round(avg_train_pres, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_consistency": round(avg_val_cons, 4),
            "val_preservation": round(avg_val_pres, 4),
        }
        history.append(epoch_record)

        logger.info(
            f"[Epoch {epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} (Cons: {avg_train_cons:.4f}, Pres: {avg_train_pres:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (Cons: {avg_val_cons:.4f}, Pres: {avg_val_pres:.4f})"
        )

        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            peft_model.save_pretrained(str(best_adapter_dir))
            logger.info(f"Saved new best adapter checkpoint to {best_adapter_dir}")

    total_time = round(time.time() - start_time, 2)

    # Save training trajectory manifest
    meta_info = {
        "model_name": model_name,
        "target_layers": layers_to_edit,
        "epochs": epochs,
        "best_val_loss": round(best_val_loss, 4),
        "total_training_seconds": total_time,
        "history": history,
        "loss_weights": loss_weights,
    }
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    logger.info(f"Parameter editing training finished in {total_time}s. Best Val Loss: {best_val_loss:.4f}")
    return {
        "status": "completed",
        "adapter_dir": str(best_adapter_dir),
        "best_val_loss": best_val_loss,
        "history": meta_info,
    }
