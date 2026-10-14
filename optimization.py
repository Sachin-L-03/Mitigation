"""Hybrid PromptSensitivity Optimization and Controlled Ablation Study.

Implements Experiments A through F:
  Experiment A: Verification and reproduction of existing experimental conditions.
  Experiment B: Steering strength sweep (alpha in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]) on validation data.
  Experiment C: Recomputing sensitivity subspaces from edited model activations and geometric comparison.
  Experiment D: Staged search over alternative steering layers [12, 16, 20, 24] and ranks [1, 2, 3].
  Experiment E: Preservation regularization tuning [0.5, 1.0, 2.0, 5.0] to improve anchor retention.
  Experiment F: Multi-condition evaluation and statistical comparison on the held-out test split.
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
from peft import PeftModel
from scipy import stats

from activations import ActivationCacheManager, ActivationExtractor, get_transformer_layers
from dataset import PromptDataset
from evaluate import evaluate_all_conditions
from hybrid import run_condition_inference
from metrics import (
    compare_conditions_paired,
    compute_bootstrap_ci,
    evaluate_cluster_predictions,
)
from model import load_model_and_tokenizer
from representation_steering import (
    SubspaceSuppressionHook,
    estimate_sensitivity_subspace,
)
from train_hybrid import train_parameter_editor
from utils import load_and_validate_config, set_seed, setup_logger

logger = logging.getLogger("prompt_sensitivity.optimization")


def compute_validation_selection_score(
    token_f1: float,
    anchor_preservation: float,
    lambda_pres: float = 0.25,
) -> float:
    """Predefined validation objective balancing consistency and anchor preservation."""
    return float(token_f1 + lambda_pres * anchor_preservation)


def compare_subspaces(
    P_orig: torch.Tensor,
    P_edited: torch.Tensor,
    singular_orig: Optional[torch.Tensor] = None,
    singular_edited: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Quantify geometric differences between two projection operators."""
    p_orig = P_orig.float().cpu()
    p_edit = P_edited.float().cpu()

    frob_diff = torch.norm(p_orig - p_edit, p="fro").item()
    norm_orig = torch.norm(p_orig, p="fro").item()
    norm_edit = torch.norm(p_edit, p="fro").item()

    # Subspace alignment: Tr(P_orig P_edited) / (||P_orig||_F * ||P_edited||_F)
    trace_overlap = torch.trace(torch.mm(p_orig, p_edit)).item()
    alignment = trace_overlap / (norm_orig * norm_edit + 1e-9)

    result = {
        "frobenius_distance": round(float(frob_diff), 4),
        "subspace_alignment": round(float(alignment), 4),
    }

    if singular_orig is not None and singular_edited is not None:
        s_orig = singular_orig.float().cpu()
        s_edit = singular_edited.float().cpu()
        cos_sim = torch.nn.functional.cosine_similarity(
            s_orig.view(1, -1), s_edit.view(1, -1)
        ).item()
        result["singular_values_cosine_sim"] = round(float(cos_sim), 4)

    return result


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Adjust p-values using the Holm-Bonferroni step-down procedure."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = p_val * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, round(running_max, 5))

    return adjusted


# ---------------------------------------------------------------------------
# Experiment A: Reproduction Verification
# ---------------------------------------------------------------------------

def run_experiment_a_reproduction(
    config: Dict[str, Any],
    split: str = "test",
    output_dir: str | Path = "outputs/optimization/reproduction",
) -> Dict[str, Any]:
    """Verify and document reproduction of baseline and initial experimental results."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Executing Experiment A (Reproduction check) on split: {split}")

    # Re-evaluate saved artifacts
    metrics = evaluate_all_conditions(
        predictions_dir="outputs",
        output_dir=out_dir,
        split_name=split,
    )

    baseline_targets = {
        "baseline": {"token_f1": 37.03, "semantic_sim": 52.91, "anchor_pres": 100.00},
        "steering_only": {"token_f1": 36.50, "semantic_sim": 52.43, "anchor_pres": 62.71},
        "editing_only": {"token_f1": 45.59, "semantic_sim": 60.52, "anchor_pres": 33.57},
        "hybrid": {"token_f1": 43.17, "semantic_sim": 56.86, "anchor_pres": 31.55},
    }

    verification_report: Dict[str, Any] = {}
    for ckey, targets in baseline_targets.items():
        if ckey not in metrics:
            verification_report[ckey] = {"status": "missing"}
            continue

        calc = metrics[ckey]
        calc_f1 = calc["token_f1_similarity"]["mean"]
        calc_sim = calc["semantic_ngram_similarity"]["mean"]
        calc_pres = calc["anchor_preservation"]["mean"]

        f1_diff = round(abs(calc_f1 - targets["token_f1"]), 3)
        sim_diff = round(abs(calc_sim - targets["semantic_sim"]), 3)
        pres_diff = round(abs(calc_pres - targets["anchor_pres"]), 3)

        verification_report[ckey] = {
            "target": targets,
            "calculated": {
                "token_f1": calc_f1,
                "semantic_sim": calc_sim,
                "anchor_pres": calc_pres,
            },
            "discrepancies": {
                "token_f1_diff": f1_diff,
                "semantic_diff": sim_diff,
                "preservation_diff": pres_diff,
            },
            "reproduced": bool(f1_diff < 0.05 and sim_diff < 0.05 and pres_diff < 0.05),
        }

    report_file = out_dir / "reproduction_verification_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)

    logger.info(f"Reproduction check completed. Report written to {report_file}")
    return verification_report


# ---------------------------------------------------------------------------
# Experiment B: Steering Strength Sweep
# ---------------------------------------------------------------------------

def run_experiment_b_steering_sweep(
    config: Dict[str, Any],
    split: str = "val",
    strengths: Optional[List[float]] = None,
    layer: int = 16,
    subspace_path: Optional[str | Path] = None,
    output_dir: str | Path = "outputs/optimization/steering_sweep",
) -> Dict[str, Any]:
    """Sweep steering strengths on validation data keeping adapter and layer fixed."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if strengths is None:
        strengths = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "data/splits"))
    val_dataset = PromptDataset.load_jsonl(splits_dir / f"{split}.jsonl")
    model_name = config["model"]["name"]

    adapter_dir = Path(config.get("hybrid", {}).get("adapter_dir", "outputs/parameter_editing/best_adapter"))
    subspace_file = Path(subspace_path or f"outputs/steering/subspace_layer_{layer}.pt")

    if not subspace_file.is_file():
        raise FileNotFoundError(f"Subspace projection file not found: {subspace_file}")

    loaded_subspace = torch.load(subspace_file, map_location="cpu", weights_only=False)
    proj_matrix = loaded_subspace["projection_matrix"]

    # Load base model + adapter ONCE to avoid reloading 7B weights in memory repeatedly
    logger.info(f"Loading base model '{model_name}' and adapter for steering strength sweep...")
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    base_model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()

    # Load baseline validation predictions if they exist, or generate them for anchor comparison
    val_base_file = out_dir / f"baseline_{split}_predictions.jsonl"
    existing_base = Path("outputs/optimization/steering_sweep") / f"baseline_{split}_predictions.jsonl"
    if not val_base_file.is_file() and existing_base.is_file():
        import shutil
        shutil.copy(existing_base, val_base_file)
        logger.info(f"Reused existing baseline reference predictions from {existing_base}")

    if not val_base_file.is_file():
        # Evaluate base model on validation split for anchor comparison
        logger.info(f"Generating baseline reference predictions on {split} split...")
        run_condition_inference(
            condition_name="baseline",
            model=base_model,
            tokenizer=tokenizer,
            records=val_dataset.records,
            output_file=val_base_file,
            model_name=model_name,
            max_new_tokens=config.get("generation", {}).get("max_new_tokens", 64),
        )

    # Also reuse alpha 0.00 if already computed (since alpha=0 is identical for any subspace)
    alpha0_file = out_dir / f"val_hybrid_alpha_0.00.jsonl"
    existing_alpha0 = Path("outputs/optimization/steering_sweep/val_hybrid_alpha_0.00.jsonl")
    if not alpha0_file.is_file() and existing_alpha0.is_file():
        import shutil
        shutil.copy(existing_alpha0, alpha0_file)
        logger.info(f"Reused existing alpha=0.00 predictions from {existing_alpha0}")

    base_preds = []
    with open(val_base_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base_preds.append(json.loads(line))

    sweep_results: Dict[float, Dict[str, Any]] = {}
    max_tokens = config.get("generation", {}).get("max_new_tokens", 64)

    # Attach hook once, dynamically update steering strength
    hook = SubspaceSuppressionHook(
        model=peft_model,
        target_layer=layer,
        projection_matrix=proj_matrix,
        steering_strength=0.0,
    )
    hook.attach()

    try:
        for alpha in strengths:
            logger.info(f"[Sweep alpha={alpha:.2f}] Running inference on {split} split...")
            hook.steering_strength = float(alpha)

            pred_file = out_dir / f"val_hybrid_alpha_{alpha:.2f}.jsonl"
            run_condition_inference(
                condition_name=f"hybrid_alpha_{alpha:.2f}",
                model=peft_model,
                tokenizer=tokenizer,
                records=val_dataset.records,
                output_file=pred_file,
                model_name=model_name,
                max_new_tokens=max_tokens,
                resume=True,
            )

            # Load predictions and compute metrics
            preds = []
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        preds.append(json.loads(line))

            raw_eval = evaluate_cluster_predictions(preds, baseline_predictions=base_preds)
            mean_f1, f1_l, f1_h = compute_bootstrap_ci(raw_eval["cluster_token_f1s"])
            mean_sim, _, _ = compute_bootstrap_ci(raw_eval["cluster_ngram_sims"])
            mean_pres, _, _ = compute_bootstrap_ci(raw_eval["cluster_preservations"])
            mean_ema, _, _ = compute_bootstrap_ci(raw_eval["cluster_exact_matches"])

            val_score = compute_validation_selection_score(
                token_f1=mean_f1 * 100,
                anchor_preservation=mean_pres * 100,
            )

            sweep_results[alpha] = {
                "alpha": alpha,
                "token_f1": round(mean_f1 * 100, 2),
                "f1_ci95": [round(f1_l * 100, 2), round(f1_h * 100, 2)],
                "semantic_sim": round(mean_sim * 100, 2),
                "exact_match": round(mean_ema * 100, 2),
                "anchor_preservation": round(mean_pres * 100, 2),
                "diversity_ratio": round(raw_eval["diversity_ratio"] * 100, 2),
                "validation_score": round(val_score, 2),
            }
            logger.info(
                f"[alpha={alpha:.2f}] Token F1: {mean_f1*100:.2f}%, "
                f"Preservation: {mean_pres*100:.2f}%, Score: {val_score:.2f}"
            )
    finally:
        hook.remove()
        del peft_model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Identify best candidate based on predefined validation score
    best_alpha = max(sweep_results.keys(), key=lambda a: sweep_results[a]["validation_score"])

    summary = {
        "split": split,
        "layer": layer,
        "strengths_tested": strengths,
        "results": sweep_results,
        "best_alpha": best_alpha,
        "best_candidate": sweep_results[best_alpha],
    }

    with open(out_dir / "steering_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Experiment C: Recompute Steering Directions for Edited Model
# ---------------------------------------------------------------------------

def run_experiment_c_edited_subspace(
    config: Dict[str, Any],
    layer: int = 16,
    rank: int = 2,
    output_dir: str | Path = "outputs/optimization/edited_subspace",
) -> Dict[str, Any]:
    """Extract activations from parameter-edited model, estimate new subspace, and compare."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    act_cache_dir = out_dir / "activations"
    act_cache_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "data/splits"))
    train_dataset = PromptDataset.load_jsonl(splits_dir / "train.jsonl")
    model_name = config["model"]["name"]
    adapter_dir = Path(config.get("hybrid", {}).get("adapter_dir", "outputs/parameter_editing/best_adapter"))

    # 1. Load model with adapter
    logger.info("Loading parameter-edited model to extract activations...")
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    base_model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()

    cache_mgr = ActivationCacheManager(act_cache_dir, model_name=f"{model_name}_edited")
    extractor = ActivationExtractor(peft_model, target_layers=[layer], token_position="last")
    device = next(peft_model.parameters()).device

    # Extract activations on training prompts
    for idx, rec in enumerate(train_dataset.records, start=1):
        if cache_mgr.is_cached(rec.prompt_id):
            continue
        inputs = tokenizer(rec.prompt_text, return_tensors="pt").to(device)
        acts = extractor.extract(inputs["input_ids"], inputs["attention_mask"])
        cache_mgr.save_prompt_activations(
            prompt_id=rec.prompt_id,
            cluster_id=rec.cluster_id,
            variant_type=rec.variant_type,
            activations_by_layer=acts,
        )
        if idx % 150 == 0:
            logger.info(f"Extracted edited model activations: {idx}/{len(train_dataset)}...")

    # Load extracted activations
    train_cached = []
    for r in train_dataset.records:
        data = cache_mgr.load_prompt_activations(r.prompt_id)
        if data:
            train_cached.append(data)

    # 2. Estimate sensitivity subspace for edited model
    edited_subspace_file = out_dir / f"subspace_edited_layer_{layer}.pt"
    P_edited, S_edited = estimate_sensitivity_subspace(
        train_records=train_cached,
        target_layer=layer,
        rank=rank,
        save_path=edited_subspace_file,
    )

    # 3. Load original base-model subspace
    orig_subspace_file = Path(f"outputs/steering/subspace_layer_{layer}.pt")
    if not orig_subspace_file.is_file():
        raise FileNotFoundError(f"Original subspace file not found: {orig_subspace_file}")

    loaded_orig = torch.load(orig_subspace_file, map_location="cpu", weights_only=False)
    P_orig = loaded_orig["projection_matrix"]
    S_orig = loaded_orig["singular_values"]

    # 4. Compare subspaces
    comp_metrics = compare_subspaces(
        P_orig=P_orig,
        P_edited=P_edited,
        singular_orig=S_orig,
        singular_edited=S_edited,
    )
    logger.info(f"Geometric comparison between original and edited subspace: {comp_metrics}")

    # Free model memory before sweep
    del peft_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Run steering strength sweep on validation data using edited subspace
    sweep_dir = out_dir / "val_sweep"
    sweep_summary = run_experiment_b_steering_sweep(
        config=config,
        split="val",
        strengths=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
        layer=layer,
        subspace_path=edited_subspace_file,
        output_dir=sweep_dir,
    )

    report = {
        "layer": layer,
        "rank": rank,
        "geometric_comparison": comp_metrics,
        "edited_subspace_path": str(edited_subspace_file),
        "validation_sweep": sweep_summary,
    }

    with open(out_dir / "edited_subspace_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# Experiment D: Alternative Layers and Ranks Search
# ---------------------------------------------------------------------------

def run_experiment_d_layer_rank_search(
    config: Dict[str, Any],
    candidate_layers: Optional[List[int]] = None,
    candidate_ranks: Optional[List[int]] = None,
    test_alpha: float = 0.10,
    output_dir: str | Path = "outputs/optimization/layer_rank_search",
) -> Dict[str, Any]:
    """Perform staged search over candidate layers and ranks on validation split."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if candidate_layers is None:
        candidate_layers = [12, 16, 20, 24]
    if candidate_ranks is None:
        candidate_ranks = [1, 2, 3]

    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "data/splits"))
    train_dataset = PromptDataset.load_jsonl(splits_dir / "train.jsonl")
    val_dataset = PromptDataset.load_jsonl(splits_dir / "val.jsonl")
    model_name = config["model"]["name"]
    adapter_dir = Path(config.get("hybrid", {}).get("adapter_dir", "outputs/parameter_editing/best_adapter"))

    # Load baseline predictions for anchor evaluation
    val_base_file = Path("outputs/optimization/steering_sweep/baseline_val_predictions.jsonl")
    base_preds = []
    if val_base_file.is_file():
        with open(val_base_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    base_preds.append(json.loads(line))

    # Load existing training activations
    cache_dir = config.get("activations", {}).get("cache_dir", "outputs/activations")
    cache_mgr = ActivationCacheManager(cache_dir, model_name=model_name)
    train_cached = []
    for r in train_dataset.records:
        d = cache_mgr.load_prompt_activations(r.prompt_id)
        if d:
            train_cached.append(d)

    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    base_model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()

    layer_search_results: Dict[str, Any] = {}

    try:
        # Stage 1: Compare candidate layers at rank 2 and fixed alpha
        logger.info(f"Stage 1: Evaluating candidate layers {candidate_layers} at alpha={test_alpha}...")
        for lyr in candidate_layers:
            subspace_file = out_dir / f"subspace_l{lyr}_r2.pt"
            P_mat, _ = estimate_sensitivity_subspace(
                train_records=train_cached,
                target_layer=lyr,
                rank=2,
                save_path=subspace_file,
            )

            hook = SubspaceSuppressionHook(
                model=peft_model,
                target_layer=lyr,
                projection_matrix=P_mat,
                steering_strength=test_alpha,
            )

            pred_file = out_dir / f"val_layer_{lyr}_r2.jsonl"
            existing_sweep_file = Path("outputs/optimization/steering_sweep") / f"val_hybrid_alpha_{test_alpha:.2f}.jsonl"
            if lyr == 16 and not pred_file.is_file() and existing_sweep_file.is_file():
                import shutil
                shutil.copy(existing_sweep_file, pred_file)
                logger.info(f"Reused layer 16 rank 2 predictions from {existing_sweep_file}")

            with hook:
                run_condition_inference(
                    condition_name=f"layer_{lyr}_r2",
                    model=peft_model,
                    tokenizer=tokenizer,
                    records=val_dataset.records,
                    output_file=pred_file,
                    model_name=model_name,
                    max_new_tokens=64,
                    resume=True,
                )

            preds = []
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        preds.append(json.loads(line))

            raw_eval = evaluate_cluster_predictions(preds, baseline_predictions=base_preds)
            mean_f1, _, _ = compute_bootstrap_ci(raw_eval["cluster_token_f1s"])
            mean_sim, _, _ = compute_bootstrap_ci(raw_eval["cluster_ngram_sims"])
            mean_pres, _, _ = compute_bootstrap_ci(raw_eval["cluster_preservations"])
            score = compute_validation_selection_score(mean_f1 * 100, mean_pres * 100)

            layer_search_results[f"layer_{lyr}_r2"] = {
                "layer": lyr,
                "rank": 2,
                "alpha": test_alpha,
                "token_f1": round(mean_f1 * 100, 2),
                "semantic_sim": round(mean_sim * 100, 2),
                "anchor_preservation": round(mean_pres * 100, 2),
                "validation_score": round(score, 2),
            }

        # Stage 2: Select top layer and test candidate ranks
        best_layer_key = max(layer_search_results.keys(), key=lambda k: layer_search_results[k]["validation_score"])
        best_lyr = layer_search_results[best_layer_key]["layer"]
        logger.info(f"Stage 2: Best layer on validation is Layer {best_lyr}. Evaluating ranks {candidate_ranks}...")

        rank_results: Dict[str, Any] = {}
        for rk in candidate_ranks:
            subspace_file = out_dir / f"subspace_l{best_lyr}_r{rk}.pt"
            P_mat, _ = estimate_sensitivity_subspace(
                train_records=train_cached,
                target_layer=best_lyr,
                rank=rk,
                save_path=subspace_file,
            )

            hook = SubspaceSuppressionHook(
                model=peft_model,
                target_layer=best_lyr,
                projection_matrix=P_mat,
                steering_strength=test_alpha,
            )

            pred_file = out_dir / f"val_layer_{best_lyr}_r{rk}.jsonl"
            with hook:
                run_condition_inference(
                    condition_name=f"layer_{best_lyr}_r{rk}",
                    model=peft_model,
                    tokenizer=tokenizer,
                    records=val_dataset.records,
                    output_file=pred_file,
                    model_name=model_name,
                    max_new_tokens=64,
                    resume=True,
                )

            preds = []
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        preds.append(json.loads(line))

            raw_eval = evaluate_cluster_predictions(preds, baseline_predictions=base_preds)
            mean_f1, _, _ = compute_bootstrap_ci(raw_eval["cluster_token_f1s"])
            mean_sim, _, _ = compute_bootstrap_ci(raw_eval["cluster_ngram_sims"])
            mean_pres, _, _ = compute_bootstrap_ci(raw_eval["cluster_preservations"])
            score = compute_validation_selection_score(mean_f1 * 100, mean_pres * 100)

            rank_results[f"rank_{rk}"] = {
                "layer": best_lyr,
                "rank": rk,
                "alpha": test_alpha,
                "token_f1": round(mean_f1 * 100, 2),
                "semantic_sim": round(mean_sim * 100, 2),
                "anchor_preservation": round(mean_pres * 100, 2),
                "validation_score": round(score, 2),
            }

    finally:
        del peft_model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_rank_key = max(rank_results.keys(), key=lambda k: rank_results[k]["validation_score"])

    final_report = {
        "stage1_layers": layer_search_results,
        "selected_layer": best_lyr,
        "stage2_ranks": rank_results,
        "best_rank": rank_results[best_rank_key]["rank"],
        "best_configuration": rank_results[best_rank_key],
    }

    with open(out_dir / "layer_rank_search_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    return final_report


# ---------------------------------------------------------------------------
# Experiment E: Preservation Regularization Sweep
# ---------------------------------------------------------------------------

def run_experiment_e_preservation_sweep(
    config: Dict[str, Any],
    weights: Optional[List[float]] = None,
    split: str = "val",
    output_dir: str | Path = "outputs/optimization/preservation",
) -> Dict[str, Any]:
    """Train and evaluate parameter editors with different preservation loss weights."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if weights is None:
        weights = [0.5, 1.0, 2.0, 5.0]

    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "data/splits"))
    val_dataset = PromptDataset.load_jsonl(splits_dir / f"{split}.jsonl")
    model_name = config["model"]["name"]
    target_layers = config.get("parameter_editing", {}).get("target_layers", [16, 20, 24])

    val_base_file = Path("outputs/optimization/steering_sweep/baseline_val_predictions.jsonl")
    base_preds = []
    if val_base_file.is_file():
        with open(val_base_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    base_preds.append(json.loads(line))

    preservation_results: Dict[float, Dict[str, Any]] = {}

    for w_pres in weights:
        adapter_dest = out_dir / f"adapter_w{w_pres:.1f}"
        adapter_model_file = adapter_dest / "adapter_model.safetensors"

        if w_pres == 1.0 and not adapter_model_file.is_file():
            orig_adapter = Path("outputs/parameter_editing/best_adapter")
            if (orig_adapter / "adapter_model.safetensors").is_file():
                import shutil
                if adapter_dest.exists():
                    shutil.rmtree(adapter_dest)
                shutil.copytree(orig_adapter, adapter_dest)
                logger.info(f"Reused baseline adapter for w_pres=1.0 from {orig_adapter}")

        pred_file = out_dir / f"val_predictions_w{w_pres:.1f}.jsonl"
        existing_alpha0 = Path("outputs/optimization/steering_sweep/val_hybrid_alpha_0.00.jsonl")
        if w_pres == 1.0 and not pred_file.is_file() and existing_alpha0.is_file():
            import shutil
            shutil.copy(existing_alpha0, pred_file)
            logger.info(f"Reused w_pres=1.0 val predictions from {existing_alpha0}")

        if not adapter_model_file.is_file():
            logger.info(f"Training adapter with preservation weight w_pres={w_pres}...")
            sweep_config = json.loads(json.dumps(config))
            sweep_config["parameter_editing"]["loss_weights"]["preservation"] = float(w_pres)
            sweep_config["parameter_editing"]["output_dir"] = str(out_dir / f"run_w{w_pres:.1f}")

            # Train parameter editor with specific preservation weight
            train_parameter_editor(
                config=sweep_config,
                target_layers=target_layers,
                resume_from_checkpoint=False,
            )

            # Move/copy adapter checkpoint to adapter_dest
            src_ckpt = Path(sweep_config["parameter_editing"]["output_dir"]) / "best_adapter"
            if src_ckpt.is_dir():
                import shutil
                if adapter_dest.exists():
                    shutil.rmtree(adapter_dest)
                shutil.copytree(src_ckpt, adapter_dest)

        # Check if already completed
        already_completed = False
        if pred_file.is_file():
            try:
                with open(pred_file, "r", encoding="utf-8") as f_chk:
                    cnt = sum(1 for line in f_chk if line.strip())
                    if cnt == len(val_dataset.records):
                        already_completed = True
            except Exception:
                pass

        if not already_completed:
            logger.info(f"Evaluating preservation adapter w={w_pres} on {split} split...")
            bundle = load_model_and_tokenizer(model_name=model_name, config=config)
            base_model = bundle["model"]
            tokenizer = bundle["tokenizer"]
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_dest))
            peft_model.eval()

            run_condition_inference(
                condition_name=f"preservation_w{w_pres:.1f}",
                model=peft_model,
                tokenizer=tokenizer,
                records=val_dataset.records,
                output_file=pred_file,
                model_name=model_name,
                max_new_tokens=64,
                resume=True,
            )

            del peft_model
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        preds = []
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))

        raw_eval = evaluate_cluster_predictions(preds, baseline_predictions=base_preds)
        mean_f1, _, _ = compute_bootstrap_ci(raw_eval["cluster_token_f1s"])
        mean_sim, _, _ = compute_bootstrap_ci(raw_eval["cluster_ngram_sims"])
        mean_pres, _, _ = compute_bootstrap_ci(raw_eval["cluster_preservations"])
        mean_ema, _, _ = compute_bootstrap_ci(raw_eval["cluster_exact_matches"])
        score = compute_validation_selection_score(mean_f1 * 100, mean_pres * 100)

        preservation_results[w_pres] = {
            "w_pres": w_pres,
            "adapter_dir": str(adapter_dest),
            "token_f1": round(mean_f1 * 100, 2),
            "semantic_sim": round(mean_sim * 100, 2),
            "exact_match": round(mean_ema * 100, 2),
            "anchor_preservation": round(mean_pres * 100, 2),
            "diversity_ratio": round(raw_eval["diversity_ratio"] * 100, 2),
            "validation_score": round(score, 2),
        }
        logger.info(
            f"[w_pres={w_pres}] Token F1: {mean_f1*100:.2f}%, "
            f"Preservation: {mean_pres*100:.2f}%, Score: {score:.2f}"
        )

    best_w = max(preservation_results.keys(), key=lambda w: preservation_results[w]["validation_score"])

    report = {
        "split": split,
        "weights_tested": weights,
        "results": preservation_results,
        "best_w_pres": best_w,
        "best_candidate": preservation_results[best_w],
    }

    with open(out_dir / "preservation_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# Experiment F: Comprehensive Test Comparison Across Selected Methods
# ---------------------------------------------------------------------------

def run_experiment_f_final_comparison(
    config: Dict[str, Any],
    candidate_methods: Dict[str, Dict[str, Any]],
    split: str = "test",
    output_dir: str | Path = "outputs/optimization/final_comparison",
) -> Dict[str, Any]:
    """Execute and aggregate final comparative evaluation across candidate methods on test split."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "data/splits"))
    test_dataset = PromptDataset.load_jsonl(splits_dir / f"{split}.jsonl")
    model_name = config["model"]["name"]
    max_tokens = config.get("generation", {}).get("max_new_tokens", 64)

    # Dictionary of prediction paths for evaluation
    pred_files: Dict[str, Path] = {}

    for method_key, m_spec in candidate_methods.items():
        out_pred = out_dir / f"{method_key}_{split}_predictions.jsonl"
        pred_files[method_key] = out_pred

        if not out_pred.is_file():
            candidate_existing = Path("outputs") / f"{method_key}_{split}_predictions.jsonl"
            if candidate_existing.is_file():
                import shutil
                shutil.copy(candidate_existing, out_pred)
                logger.info(f"Reused existing predictions for {method_key} from {candidate_existing}")

        if out_pred.is_file():
            logger.info(f"Predictions already exist for method '{method_key}', skipping generation.")
            continue

        logger.info(f"Executing inference for final candidate '{method_key}'...")
        bundle = load_model_and_tokenizer(model_name=model_name, config=config)
        base_model = bundle["model"]
        tokenizer = bundle["tokenizer"]

        # Handle adapter loading if specified
        adapter_path = m_spec.get("adapter_dir")
        if adapter_path:
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            model = base_model

        # Handle steering hook if specified
        hook = None
        if m_spec.get("steering"):
            s_spec = m_spec["steering"]
            subspace_f = Path(s_spec["subspace_path"])
            loaded_sub = torch.load(subspace_f, map_location="cpu", weights_only=False)
            hook = SubspaceSuppressionHook(
                model=model,
                target_layer=s_spec["layer"],
                projection_matrix=loaded_sub["projection_matrix"],
                steering_strength=s_spec["strength"],
            )

        run_condition_inference(
            condition_name=method_key,
            model=model,
            tokenizer=tokenizer,
            records=test_dataset.records,
            output_file=out_pred,
            model_name=model_name,
            steering_hook=hook,
            max_new_tokens=max_tokens,
            resume=True,
        )

        del model
        if adapter_path:
            del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load baseline predictions as anchor comparison
    base_file = pred_files.get("baseline") or Path("outputs/baseline_test_predictions.jsonl")
    base_preds = []
    with open(base_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base_preds.append(json.loads(line))

    base_eval = evaluate_cluster_predictions(base_preds)

    # Evaluate each method
    final_metrics: Dict[str, Dict[str, Any]] = {}
    p_values_f1: List[float] = []
    method_keys_list: List[str] = []

    for method_key, m_spec in candidate_methods.items():
        p_path = pred_files[method_key]
        preds = []
        with open(p_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))

        raw_eval = evaluate_cluster_predictions(preds, baseline_predictions=base_preds)
        mean_ema, ema_l, ema_h = compute_bootstrap_ci(raw_eval["cluster_exact_matches"])
        mean_f1, f1_l, f1_h = compute_bootstrap_ci(raw_eval["cluster_token_f1s"])
        mean_sim, sim_l, sim_h = compute_bootstrap_ci(raw_eval["cluster_ngram_sims"])
        mean_pres, pres_l, pres_h = compute_bootstrap_ci(raw_eval["cluster_preservations"])
        mean_len, _, _ = compute_bootstrap_ci(raw_eval["cluster_answer_lengths"])

        paired_f1 = compare_conditions_paired(raw_eval["cluster_token_f1s"], base_eval["cluster_token_f1s"])
        p_val = paired_f1["p_value"]

        final_metrics[method_key] = {
            "label": m_spec.get("label", method_key),
            "exact_match": {"mean": round(mean_ema * 100, 2), "ci_95": [round(ema_l * 100, 2), round(ema_h * 100, 2)]},
            "token_f1": {
                "mean": round(mean_f1 * 100, 2),
                "ci_95": [round(f1_l * 100, 2), round(f1_h * 100, 2)],
                "abs_change_pp": round(paired_f1["abs_change"] * 100, 2),
                "rel_change_pct": paired_f1["rel_change_pct"],
                "p_value_raw": p_val,
            },
            "semantic_sim": {"mean": round(mean_sim * 100, 2), "ci_95": [round(sim_l * 100, 2), round(sim_h * 100, 2)]},
            "anchor_preservation": {"mean": round(mean_pres * 100, 2), "ci_95": [round(pres_l * 100, 2), round(pres_h * 100, 2)]},
            "diversity_ratio": round(raw_eval["diversity_ratio"] * 100, 2),
            "avg_word_length": round(mean_len, 2),
        }
        if method_key != "baseline":
            method_keys_list.append(method_key)
            p_values_f1.append(p_val)

    # Apply Holm-Bonferroni correction
    adjusted_p_vals = apply_holm_bonferroni(p_values_f1)
    for m_key, adj_p in zip(method_keys_list, adjusted_p_vals):
        final_metrics[m_key]["token_f1"]["p_value_holm"] = adj_p
        final_metrics[m_key]["token_f1"]["significant_holm"] = bool(adj_p < 0.05)

    # Save summary JSON
    with open(out_dir / "final_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Save CSV table
    csv_file = out_dir / "final_comparison_table.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "Token F1 (%)", "F1 95% CI", "Diff F1 (pp)", "Rel Diff (%)", "Raw p-val", "Holm p-val",
            "Semantic Sim (%)", "Exact Match (%)", "Anchor Pres (%)", "Diversity (%)", "Avg Words"
        ])
        for k, d in final_metrics.items():
            f1_d = d["token_f1"]
            writer.writerow([
                d["label"],
                f"{f1_d['mean']:.2f}",
                f"[{f1_d['ci_95'][0]:.1f}, {f1_d['ci_95'][1]:.1f}]",
                f"{f1_d.get('abs_change_pp', 0.0):+.2f}",
                f"{f1_d.get('rel_change_pct', 0.0):+.1f}%",
                f"{f1_d.get('p_value_raw', 1.0):.4f}" if k != "baseline" else "-",
                f"{f1_d.get('p_value_holm', 1.0):.4f}" if k != "baseline" else "-",
                f"{d['semantic_sim']['mean']:.2f}",
                f"{d['exact_match']['mean']:.2f}",
                f"{d['anchor_preservation']['mean']:.2f}",
                f"{d['diversity_ratio']:.2f}",
                f"{d['avg_word_length']:.1f}",
            ])

    return final_metrics


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Optimization and ablation study workflow.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["exp_a", "exp_b", "exp_c", "exp_d", "exp_e", "exp_f", "all"],
        default="all",
        help="Experiment stage to execute."
    )
    args = parser.parse_args()

    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.optimization", log_dir=log_dir)

    print("\n" + "=" * 75)
    print("      HYBRID PROMPTSENSITIVITY OPTIMIZATION & ABLATION STUDY")
    print("=" * 75)

    if args.stage in ["exp_a", "all"]:
        print("\n>>> Stage A: Reproduction & Verification Check...")
        res_a = run_experiment_a_reproduction(config)
        for k, v in res_a.items():
            print(f"  [{k}] Reproduced: {v.get('reproduced', False)}")

    if args.stage in ["exp_b", "all"]:
        print("\n>>> Stage B: Steering Strength Sweep on Validation Data...")
        res_b = run_experiment_b_steering_sweep(config)
        print(f"  Best Validation Alpha: {res_b['best_alpha']} (Score: {res_b['best_candidate']['validation_score']})")

    if args.stage in ["exp_c", "all"]:
        print("\n>>> Stage C: Recomputed Subspace from Edited Model...")
        res_c = run_experiment_c_edited_subspace(config)
        print(f"  Subspace Alignment (Tr(P_orig P_edited)): {res_c['geometric_comparison']['subspace_alignment']}")
        print(f"  Best Alpha (Edited Subspace): {res_c['validation_sweep']['best_alpha']}")

    if args.stage in ["exp_d", "all"]:
        print("\n>>> Stage D: Layer and Rank Search on Validation Data...")
        res_d = run_experiment_d_layer_rank_search(config)
        print(f"  Best Layer: {res_d['selected_layer']}, Best Rank: {res_d['best_rank']}")

    if args.stage in ["exp_e", "all"]:
        print("\n>>> Stage E: Preservation Regularization Sweep...")
        res_e = run_experiment_e_preservation_sweep(config)
        print(f"  Best Preservation Weight: {res_e['best_w_pres']} (Score: {res_e['best_candidate']['validation_score']})")

    if args.stage in ["exp_f", "all"]:
        print("\n>>> Stage F: Multi-Condition Evaluation on Held-Out Test Split...")
        candidate_methods = {
            "baseline": {
                "label": "1. Original Baseline",
            },
            "steering_only": {
                "label": "2. Existing Steering Only (L16, a=0.5)",
                "steering": {
                    "layer": 16,
                    "strength": 0.5,
                    "subspace_path": "outputs/steering/subspace_layer_16.pt",
                },
            },
            "editing_only": {
                "label": "3. Existing Editing Only (Best Adapter)",
                "adapter_dir": "outputs/parameter_editing/best_adapter",
            },
            "hybrid_val_orig_subspace": {
                "label": "4. Best Hybrid (Original Subspace, a=0.05)",
                "adapter_dir": "outputs/parameter_editing/best_adapter",
                "steering": {
                    "layer": 16,
                    "strength": 0.05,
                    "subspace_path": "outputs/steering/subspace_layer_16.pt",
                },
            },
            "hybrid_val_edited_subspace": {
                "label": "5. Best Hybrid (Edited Subspace, a=0.10)",
                "adapter_dir": "outputs/parameter_editing/best_adapter",
                "steering": {
                    "layer": 16,
                    "strength": 0.10,
                    "subspace_path": "outputs/optimization/edited_subspace/subspace_edited_layer_16.pt",
                },
            },
            "hybrid_alt_layer": {
                "label": "6. Best Alternative-Layer Hybrid (L24, r2, a=0.10)",
                "adapter_dir": "outputs/parameter_editing/best_adapter",
                "steering": {
                    "layer": 24,
                    "strength": 0.10,
                    "subspace_path": "outputs/optimization/layer_rank_search/subspace_l24_r2.pt",
                },
            },
            "editing_preservation_tuned": {
                "label": "7. Best Preservation-Tuned Editing (w_pres=5.0)",
                "adapter_dir": "outputs/optimization/preservation/adapter_w5.0",
            },
        }
        res_f = run_experiment_f_final_comparison(config, candidate_methods, split="test")
        print("\nFinal Comparison Completed:")
        for k, v in res_f.items():
            print(f"  {v['label']:<48} | F1: {v['token_f1']['mean']:.2f}% | Pres: {v['anchor_preservation']['mean']:.2f}%")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
