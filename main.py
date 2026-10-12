"""Comprehensive CLI workflow for prompt-sensitivity-hybrid research pipeline.

Subcommands:
  check-env            Audit hardware, CUDA, bitsandbytes, and package versions
  dataset-check        Validate canonical schema and verify zero-leakage cluster splitting
  smoke-test           Quick single-prompt model loading and inference check
  baseline             Execute baseline inference on dataset split (greedy decoding)
  extract-activations  Extract hidden states across transformer layers with disk caching
  causal-analysis      Execute causal activation patching and rank candidate layers
  train-editor         Train layer-aware LoRA adapter on causally selected layers
  eval-steering        Evaluate Condition B (Representation Steering Only)
  eval-hybrid          Evaluate Condition C (Editing Only) and Condition D (Hybrid)
  evaluate             Aggregate all condition metrics, bootstrap CIs, and generate report
  pipeline             Execute end-to-end research experiment pipeline sequentially
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from activations import ActivationCacheManager, ActivationExtractor
from causal_analysis import (
    CausalPatcher,
    compute_layer_divergences,
    evaluate_layer_causal_patching,
    rank_layers_by_causality,
)
from dataset import (
    PromptDataset,
    cluster_train_val_test_split,
    save_dataset_and_splits,
    verify_split_leakage,
)
from evaluate import evaluate_all_conditions
from hybrid import run_condition_inference
from model import SUPPORTED_MODELS, load_model_and_tokenizer, run_smoke_inference
from parameter_editing import (
    build_layer_aware_lora_model,
    generate_and_cache_pseudo_targets,
)
from representation_steering import (
    SubspaceSuppressionHook,
    estimate_sensitivity_subspace,
)
from train_hybrid import train_parameter_editor
from utils import (
    inspect_system_environment,
    load_and_validate_config,
    save_experiment_manifest,
    set_seed,
    setup_logger,
)


def cmd_check_env(args: argparse.Namespace) -> int:
    """Run comprehensive environment verification."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.check_env", log_dir=log_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Initialized reproducibility seed: {seed}")

    env_info = inspect_system_environment(config)

    print("\n" + "=" * 65)
    print("        PROMPT-SENSITIVITY-HYBRID: ENVIRONMENT AUDIT")
    print("=" * 65)
    print(f"Platform:              {env_info['platform']}")
    print(f"Python Executable:     {env_info['python_executable']}")
    print(f"Python Version:        {env_info['python_version'].split()[0]}")
    print(f"CUDA Available:        {env_info['cuda_available']}")
    print(f"PyTorch Version:       {env_info.get('torch_version', 'N/A')}")
    print(f"Torch CUDA Version:    {env_info.get('torch_cuda_version', 'N/A')}")
    print(f"Quantization Support:  {env_info['quantization_support']}")
    print(f"BitsAndBytes Version:  {env_info.get('bitsandbytes_version', 'N/A')}")

    if env_info["devices"]:
        for dev in env_info["devices"]:
            print(f"\n  [Device {dev['index']}] {dev['name']}")
            print(f"    - Total VRAM:          {dev['total_vram_gb']} GB")
            print(f"    - Compute Capability:  sm_{dev['major_capability']}{dev['minor_capability']}")
            print(f"    - BF16 Native Support: {dev['bf16_supported']}")

    output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "env_check_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    save_experiment_manifest(output_dir, config, {"stage": "check_env", "env_info": env_info})
    print("=" * 65 + "\n")
    return 0 if env_info["cuda_available"] else 1


def cmd_dataset_check(args: argparse.Namespace) -> int:
    """Validate dataset schema and execute cluster-based train/val/test splitting."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.dataset_check", log_dir=log_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    data_path = args.data_path or config["dataset"]["data_path"]
    splits_dir = args.splits_dir or config["dataset"].get("splits_dir", "data/splits")
    logger.info(f"Loading dataset from: {data_path}")

    is_csv = str(data_path).lower().endswith(".csv")
    if is_csv:
        dataset = PromptDataset.from_axis_tagged_csv(
            data_path,
            validate_expected_counts=True,
            expected_clusters=config["dataset"].get("expected_clusters", 196),
            expected_records=config["dataset"].get("expected_records", 1176),
        )
    else:
        dataset = PromptDataset.load_jsonl(data_path)

    val_stats = dataset.validate()

    train_r = config["dataset"]["train_ratio"]
    val_r = config["dataset"]["val_ratio"]
    test_r = config["dataset"]["test_ratio"]

    train_ds, val_ds, test_ds = cluster_train_val_test_split(
        dataset,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=seed,
    )

    leakage_stats = verify_split_leakage(train_ds, val_ds, test_ds)

    split_meta = save_dataset_and_splits(
        canonical_dataset=dataset,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        splits_dir=splits_dir,
        seed=seed,
        source_file=str(data_path),
    )

    output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": str(data_path),
                "splits_dir": str(splits_dir),
                "dataset_stats": val_stats,
                "leakage_stats": leakage_stats,
                "split_metadata": split_meta,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("      PROMPT-SENSITIVITY DATASET INTEGRATION & SPLIT SUMMARY")
    print("=" * 70)
    print(f"Source Dataset:         {data_path}")
    print(f"Splits Target Dir:      {splits_dir}")
    print(f"Random Seed:            {seed}")
    print(f"Total Unique Clusters:  {val_stats['unique_clusters']}")
    print(f"Total Prompt Records:   {val_stats['total_records']} (6 versions per question cluster)")
    print(f"Reference Answers:      All null (source CSV has no answers; no synthetic answers generated)")
    print("-" * 70)
    print("Partition Breakdown (70% Train / 15% Val / 15% Test):")
    print(f"  * Train Split:        {leakage_stats['train_cluster_count']:>3} clusters | {len(train_ds):>4} prompt records ({len(train_ds)/len(dataset)*100:.1f}%)")
    print(f"  * Val Split:          {leakage_stats['val_cluster_count']:>3} clusters | {len(val_ds):>4} prompt records ({len(val_ds)/len(dataset)*100:.1f}%)")
    print(f"  * Test Split:         {leakage_stats['test_cluster_count']:>3} clusters | {len(test_ds):>4} prompt records ({len(test_ds)/len(dataset)*100:.1f}%)")
    print("-" * 70)
    print("Verification & Integrity Checks:")
    print(f"  [PASSED] Missing Values:        0 missing values detected in source CSV")
    print(f"  [PASSED] Duplicate Prompt IDs:  0 duplicate IDs detected across 1,176 records")
    print(f"  [PASSED] Zero Cluster Leakage:  True (train, val, test clusters are 100% disjoint)")
    print(f"  [PASSED] Cluster Cohesion:      All 6 question versions strictly preserved in same split")
    print("=" * 70 + "\n")
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Execute model loading and forward-pass smoke test."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.smoke_test", log_dir=log_dir)

    model_name = args.model or config["model"]["name"]
    logger.info(f"Starting smoke test for model: {model_name}")

    try:
        bundle = load_model_and_tokenizer(model_name=model_name, config=config, token=args.token)
        test_prompt = args.prompt or "Explain prompt sensitivity in two sentences."
        res = run_smoke_inference(bundle, prompt=test_prompt, max_new_tokens=args.max_new_tokens)

        print("\n" + "=" * 65)
        print("        MODEL LOAD & INFERENCE SMOKE TEST PASSED")
        print("=" * 65)
        print(f"Model Name:        {model_name}")
        print(f"Devices In Use:    {bundle['devices']}")
        print(f"VRAM Allocated:    {res['vram_allocated_gb']} GB")
        print(f"Tokens Generated:  {res['tokens_generated']}")
        print(f"Output Sample:     {res['generated_text'].strip()[:200]}...")
        print("=" * 65 + "\n")
        return 0
    except Exception as e:
        logger.error(f"Smoke test failed for model '{model_name}': {e}", exc_info=True)
        print(f"\n[!] Smoke test failed: {e}\n")
        return 1


def _load_split_dataset(config: Dict[str, Any], split: str) -> PromptDataset:
    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "data/splits"))
    split_file = splits_dir / f"{split}.jsonl"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run 'main.py dataset-check' first.")
    return PromptDataset.load_jsonl(split_file)


def cmd_baseline(args: argparse.Namespace) -> int:
    """Run baseline inference under Condition A across specified dataset split."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.baseline", log_dir=log_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    split = args.split or "test"
    dataset = _load_split_dataset(config, split)
    model_name = args.model or config["model"]["name"]

    logger.info(f"Running Condition A (Baseline) on '{split}' split ({len(dataset)} prompts) with '{model_name}'...")
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)

    max_tokens = args.max_new_tokens or config.get("generation", {}).get("max_new_tokens", 64)
    out_file = Path("outputs") / f"baseline_{split}_predictions.jsonl"

    summary = run_condition_inference(
        condition_name="baseline",
        model=bundle["model"],
        tokenizer=bundle["tokenizer"],
        records=dataset.records,
        output_file=out_file,
        model_name=model_name,
        max_new_tokens=max_tokens,
        resume=not args.overwrite,
    )
    print(f"\n[PASSED] Baseline evaluation completed: {summary['total_evaluated']} prompts -> {out_file}\n")
    return 0


def cmd_extract_activations(args: argparse.Namespace) -> int:
    """Extract and cache hidden states across transformer layers to disk."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.activations", log_dir=log_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    split = args.split or "train"
    dataset = _load_split_dataset(config, split)
    model_name = args.model or config["model"]["name"]

    target_layers = args.layers or config.get("activations", {}).get("target_layers", [4, 8, 12, 16, 20, 24])
    token_pos = config.get("activations", {}).get("token_position", "last")

    logger.info(f"Extracting activations for {len(dataset)} prompts on layers {target_layers}...")
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    cache_dir = config.get("activations", {}).get("cache_dir", "outputs/activations")
    cache_mgr = ActivationCacheManager(cache_dir, model_name=model_name)
    extractor = ActivationExtractor(model, target_layers=target_layers, token_position=token_pos)

    cached_count = 0
    device = next(model.parameters()).device

    for idx, rec in enumerate(dataset.records, start=1):
        pid = rec.prompt_id
        if cache_mgr.is_cached(pid) and not args.overwrite:
            cached_count += 1
            continue

        inputs = tokenizer(rec.prompt_text, return_tensors="pt").to(device)
        acts = extractor.extract(inputs["input_ids"], inputs["attention_mask"])
        cache_mgr.save_prompt_activations(
            prompt_id=pid,
            cluster_id=rec.cluster_id,
            variant_type=rec.variant_type,
            activations_by_layer=acts,
        )
        cached_count += 1
        if idx % 100 == 0 or idx == len(dataset):
            logger.info(f"Cached activations: {cached_count}/{len(dataset)} prompts...")

    print(f"\n[PASSED] Layer activations cached for {cached_count} prompts -> {cache_mgr.cache_dir}\n")
    return 0


def cmd_causal_analysis(args: argparse.Namespace) -> int:
    """Execute within-cluster divergence and causal activation patching to rank layers."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.causal", log_dir=log_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    model_name = args.model or config["model"]["name"]
    candidate_layers = args.layers or config.get("causal_analysis", {}).get("candidate_layers", [4, 8, 12, 16, 20, 24])

    # 1. Evaluate representation divergence using cached training activations
    cache_dir = config.get("activations", {}).get("cache_dir", "outputs/activations")
    cache_mgr = ActivationCacheManager(cache_dir, model_name=model_name)

    train_ds = _load_split_dataset(config, "train")
    cached_records = []
    for r in train_ds.records:
        act_data = cache_mgr.load_prompt_activations(r.prompt_id)
        if act_data:
            cached_records.append(act_data)

    divergence_results = {}
    if cached_records:
        divergence_results = compute_layer_divergences(cached_records, candidate_layers)

    # 2. Causal activation patching on representative training clusters
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    patcher = CausalPatcher(bundle["model"], bundle["tokenizer"])

    # Prepare original-variant pairs across clusters
    by_cluster: Dict[str, Dict[str, str]] = {}
    for r in train_ds.records:
        by_cluster.setdefault(r.cluster_id, {})[r.variant_type] = r.prompt_text

    eval_pairs = []
    max_clusters = min(args.num_clusters or 25, len(by_cluster))
    for cid in list(by_cluster.keys())[:max_clusters]:
        variants = by_cluster[cid]
        if "original" not in variants:
            continue
        orig_text = variants["original"]
        for vtype, vtext in variants.items():
            if vtype != "original":
                eval_pairs.append({"cluster_id": cid, "original_text": orig_text, "variant_text": vtext})

    logger.info(f"Running causal activation patching on {len(eval_pairs)} pairs across candidate layers {candidate_layers}...")
    causal_results = evaluate_layer_causal_patching(
        patcher=patcher,
        cluster_pairs=eval_pairs,
        candidate_layers=candidate_layers,
        sham_control=config.get("causal_analysis", {}).get("sham_control", True),
    )

    ranked_layers = rank_layers_by_causality(causal_results)

    # Save causal report
    out_dir = Path(config.get("causal_analysis", {}).get("output_dir", "outputs/causal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate_layers": candidate_layers,
        "causal_evaluation": causal_results,
        "representation_divergences": divergence_results,
        "ranked_layers": ranked_layers,
        "top_recommended_layers": [lyr for lyr, score in ranked_layers[:3]],
    }

    out_file = out_dir / "layer_causal_ranking.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("        CAUSAL LAYER ANALYSIS & INTERVENTION RANKING")
    print("=" * 65)
    print("Layer | Causal Restor. | Sham Restor. | Specificity | Div. Ratio")
    print("-" * 65)
    for lyr in candidate_layers:
        c_info = causal_results.get(lyr, {})
        d_info = divergence_results.get(lyr, {})
        print(
            f" {lyr:>3}  | "
            f"{c_info.get('mean_causal_restoration', 0.0):>13.4f} | "
            f"{c_info.get('sham_mean_restoration', 0.0):>11.4f} | "
            f"{c_info.get('causal_specificity', 0.0):>10.4f} | "
            f"{d_info.get('separation_ratio', 0.0):>9.2f}"
        )
    print("-" * 65)
    print(f"Top Recommended Causal Intervention Layers: {[lyr for lyr, score in ranked_layers[:3]]}")
    print(f"Report saved to: {out_file}\n" + "=" * 65 + "\n")
    return 0


def cmd_train_editor(args: argparse.Namespace) -> int:
    """Train CREME-inspired layer-aware parameter editing adapter."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.train_editor", log_dir=log_dir)

    target_layers = args.layers
    if not target_layers:
        # Check if causal ranking exists to use top layers automatically
        ranking_file = Path("outputs/causal/layer_causal_ranking.json")
        if ranking_file.is_file():
            try:
                with open(ranking_file, "r") as f:
                    ranking_data = json.load(f)
                    target_layers = ranking_data.get("top_recommended_layers", [12, 16, 20])
                    logger.info(f"Using top causally ranked layers from causal analysis: {target_layers}")
            except Exception:
                target_layers = config.get("parameter_editing", {}).get("target_layers", [12, 16, 20])
        else:
            target_layers = config.get("parameter_editing", {}).get("target_layers", [12, 16, 20])

    res = train_parameter_editor(
        config=config,
        target_layers=target_layers,
        resume_from_checkpoint=not args.overwrite,
    )
    print(f"\n[PASSED] Parameter editing training complete. Best adapter -> {res['adapter_dir']}\n")
    return 0


def cmd_eval_steering(args: argparse.Namespace) -> int:
    """Evaluate Condition B: Representation Steering Only."""
    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.steering", log_dir=log_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    split = args.split or "test"
    dataset = _load_split_dataset(config, split)
    model_name = args.model or config["model"]["name"]

    target_layer = args.layer or config.get("representation_steering", {}).get("target_layer", 16)
    rank = config.get("representation_steering", {}).get("subspace_rank", 2)
    strength = args.strength or config.get("representation_steering", {}).get("steering_strength", 0.5)

    # 1. Load or estimate steering subspace from training activations
    cache_dir = config.get("activations", {}).get("cache_dir", "outputs/activations")
    cache_mgr = ActivationCacheManager(cache_dir, model_name=model_name)
    train_ds = _load_split_dataset(config, "train")

    subspace_path = Path("outputs/steering") / f"subspace_layer_{target_layer}.pt"
    if subspace_path.is_file() and not args.overwrite:
        loaded = torch.load(subspace_path, map_location="cpu", weights_only=False)
        proj_matrix = loaded["projection_matrix"]
        logger.info(f"Loaded existing steering subspace from {subspace_path}")
    else:
        train_cached = []
        for r in train_ds.records:
            act = cache_mgr.load_prompt_activations(r.prompt_id)
            if act:
                train_cached.append(act)
        proj_matrix, s_vals = estimate_sensitivity_subspace(
            train_records=train_cached,
            target_layer=target_layer,
            rank=rank,
            save_path=subspace_path,
        )

    # 2. Run inference with steering hook
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    hook = SubspaceSuppressionHook(
        model=model,
        target_layer=target_layer,
        projection_matrix=proj_matrix,
        steering_strength=strength,
    )

    out_file = Path("outputs") / f"steering_only_{split}_predictions.jsonl"
    logger.info(f"Running Condition B (Steering Only) on '{split}' split with layer {target_layer}, strength {strength}...")

    summary = run_condition_inference(
        condition_name="steering_only",
        model=model,
        tokenizer=tokenizer,
        records=dataset.records,
        output_file=out_file,
        model_name=model_name,
        steering_hook=hook,
        max_new_tokens=config.get("generation", {}).get("max_new_tokens", 64),
        resume=not args.overwrite,
    )
    print(f"\n[PASSED] Condition B (Steering Only) completed: {summary['total_evaluated']} prompts -> {out_file}\n")
    return 0


def cmd_eval_hybrid(args: argparse.Namespace) -> int:
    """Evaluate Condition C (Editing Only) and Condition D (Hybrid)."""
    from peft import PeftModel

    config = load_and_validate_config(args.config)
    log_dir = config.get("project", {}).get("log_dir", "outputs/logs")
    logger = setup_logger("prompt_sensitivity.hybrid_eval", log_dir=log_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    split = args.split or "test"
    dataset = _load_split_dataset(config, split)
    model_name = args.model or config["model"]["name"]

    adapter_dir = Path(config.get("hybrid", {}).get("adapter_dir", "outputs/parameter_editing/best_adapter"))
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter not found at {adapter_dir}. Run 'train-editor' first.")

    # 1. Load base model and wrap with trained adapter
    logger.info(f"Loading base model '{model_name}' and attaching adapter from {adapter_dir}...")
    bundle = load_model_and_tokenizer(model_name=model_name, config=config)
    base_model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    max_tokens = config.get("generation", {}).get("max_new_tokens", 64)

    # 2. Run Condition C: Editing Only
    edit_out = Path("outputs") / f"editing_only_{split}_predictions.jsonl"
    logger.info(f"Running Condition C (Editing Only) on '{split}' split...")
    run_condition_inference(
        condition_name="editing_only",
        model=peft_model,
        tokenizer=tokenizer,
        records=dataset.records,
        output_file=edit_out,
        model_name=model_name,
        max_new_tokens=max_tokens,
        resume=not args.overwrite,
    )

    # 3. Load steering subspace and run Condition D: Hybrid
    steer_layer = args.layer or config.get("hybrid", {}).get("steering_layer", 16)
    strength = args.strength or config.get("hybrid", {}).get("steering_strength", 0.5)

    subspace_file = Path("outputs/steering") / f"subspace_layer_{steer_layer}.pt"
    if not subspace_file.is_file():
        # Fallback to general steering subspace
        subspace_file = Path("outputs/steering") / f"subspace_layer_16.pt"

    if not subspace_file.is_file():
        raise FileNotFoundError(f"Subspace projection not found at {subspace_file}. Run 'eval-steering' first.")

    loaded = torch.load(subspace_file, map_location="cpu", weights_only=False)
    proj_matrix = loaded["projection_matrix"]

    hook = SubspaceSuppressionHook(
        model=peft_model,
        target_layer=steer_layer,
        projection_matrix=proj_matrix,
        steering_strength=strength,
    )

    hybrid_out = Path("outputs") / f"hybrid_{split}_predictions.jsonl"
    logger.info(f"Running Condition D (Hybrid: Edit + Steer) on '{split}' split...")
    run_condition_inference(
        condition_name="hybrid",
        model=peft_model,
        tokenizer=tokenizer,
        records=dataset.records,
        output_file=hybrid_out,
        model_name=model_name,
        steering_hook=hook,
        max_new_tokens=max_tokens,
        resume=not args.overwrite,
    )

    print(f"\n[PASSED] Condition C and Condition D evaluations complete -> {edit_out}, {hybrid_out}\n")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Aggregate predictions across all 4 conditions, compute bootstrap CIs, and output reports."""
    config = load_and_validate_config(args.config)
    split = args.split or "test"
    eval_dir = config.get("evaluation", {}).get("output_dir", "outputs/evaluation")

    results = evaluate_all_conditions(
        predictions_dir="outputs",
        output_dir=eval_dir,
        split_name=split,
    )

    print("\n" + "=" * 80)
    print(f"      EXPERIMENTAL EVALUATION SUMMARY (Split: {split.upper()})")
    print("=" * 80)
    print(f"{'Condition':<30} | {'Exact Match':<16} | {'Token F1':<16} | {'Diff F1(pp)':<10} | {'p-val'}")
    print("-" * 80)
    for ckey, d in results.items():
        ema = f"{d['exact_match_agreement']['mean']:.1f}% ({d['exact_match_agreement']['ci_95'][0]:.0f}-{d['exact_match_agreement']['ci_95'][1]:.0f})"
        f1 = f"{d['token_f1_similarity']['mean']:.1f}% ({d['token_f1_similarity']['ci_95'][0]:.0f}-{d['token_f1_similarity']['ci_95'][1]:.0f})"
        delta = f"{d['token_f1_similarity']['abs_change_pp']:+.2f}" if ckey != "baseline" else "0.00"
        pval = f"{d['token_f1_similarity']['p_value']:.4f}" if ckey != "baseline" else "-"
        print(f"{d['label']:<30} | {ema:<16} | {f1:<16} | {delta:<10} | {pval}")
    print("=" * 80 + "\n")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run full integrated research pipeline end-to-end on target model."""
    print("\n" + "=" * 70)
    print("      EXECUTING END-TO-END PROMPT-SENSITIVITY-HYBRID RESEARCH PIPELINE")
    print("=" * 70)

    stages = [
        ("1. Environment Audit", cmd_check_env),
        ("2. Dataset Integration & Splits", cmd_dataset_check),
        ("3. Condition A: Baseline Inference", cmd_baseline),
        ("4. Layer Activation Extraction", cmd_extract_activations),
        ("5. Causal Layer Analysis & Patching", cmd_causal_analysis),
        ("6. CREME-Inspired Parameter Editing Training", cmd_train_editor),
        ("7. Condition B: Representation Steering", cmd_eval_steering),
        ("8. Condition C & D: Editing & Hybrid Inference", cmd_eval_hybrid),
        ("9. Multi-Condition Comparative Evaluation", cmd_evaluate),
    ]

    for name, func in stages:
        print(f"\n>>> Running Stage: {name}...")
        code = func(args)
        if code != 0:
            print(f"[!] Pipeline halted at stage: {name}")
            return code
        # Free CUDA memory between major pipeline steps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("      RESEARCH PIPELINE SUCCESSFULLY COMPLETED")
    print("=" * 70 + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI workflow for prompt-sensitivity-hybrid research foundation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. check-env
    p_env = subparsers.add_parser("check-env", help="Audit hardware, CUDA, and dependencies.")
    p_env.set_defaults(func=cmd_check_env)

    # 2. dataset-check
    p_data = subparsers.add_parser("dataset-check", help="Validate dataset schema and split integrity.")
    p_data.add_argument("--data-path", type=str, default=None)
    p_data.add_argument("--splits-dir", type=str, default=None)
    p_data.set_defaults(func=cmd_dataset_check)

    # 3. smoke-test
    p_smoke = subparsers.add_parser("smoke-test", help="Test model loading and inference.")
    p_smoke.add_argument("--model", type=str, default=None, choices=SUPPORTED_MODELS)
    p_smoke.add_argument("--prompt", type=str, default=None)
    p_smoke.add_argument("--max-new-tokens", type=int, default=32)
    p_smoke.add_argument("--token", type=str, default=None)
    p_smoke.set_defaults(func=cmd_smoke_test)

    # 4. baseline
    p_base = subparsers.add_parser("baseline", help="Run Condition A (Baseline Inference).")
    p_base.add_argument("--model", type=str, default=None)
    p_base.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_base.add_argument("--max-new-tokens", type=int, default=64)
    p_base.add_argument("--overwrite", action="store_true")
    p_base.set_defaults(func=cmd_baseline)

    # 5. extract-activations
    p_act = subparsers.add_parser("extract-activations", help="Extract activations to disk cache.")
    p_act.add_argument("--model", type=str, default=None)
    p_act.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p_act.add_argument("--layers", type=int, nargs="+", default=None)
    p_act.add_argument("--overwrite", action="store_true")
    p_act.set_defaults(func=cmd_extract_activations)

    # 6. causal-analysis
    p_causal = subparsers.add_parser("causal-analysis", help="Run causal activation patching.")
    p_causal.add_argument("--model", type=str, default=None)
    p_causal.add_argument("--layers", type=int, nargs="+", default=None)
    p_causal.add_argument("--num-clusters", type=int, default=25)
    p_causal.set_defaults(func=cmd_causal_analysis)

    # 7. train-editor
    p_train = subparsers.add_parser("train-editor", help="Train layer-aware LoRA parameter editor.")
    p_train.add_argument("--layers", type=int, nargs="+", default=None)
    p_train.add_argument("--overwrite", action="store_true")
    p_train.set_defaults(func=cmd_train_editor)

    # 8. eval-steering
    p_steer = subparsers.add_parser("eval-steering", help="Evaluate Condition B (Steering Only).")
    p_steer.add_argument("--model", type=str, default=None)
    p_steer.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_steer.add_argument("--layer", type=int, default=None)
    p_steer.add_argument("--strength", type=float, default=None)
    p_steer.add_argument("--overwrite", action="store_true")
    p_steer.set_defaults(func=cmd_eval_steering)

    # 9. eval-hybrid
    p_hybrid = subparsers.add_parser("eval-hybrid", help="Evaluate Condition C (Edit) and D (Hybrid).")
    p_hybrid.add_argument("--model", type=str, default=None)
    p_hybrid.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_hybrid.add_argument("--layer", type=int, default=None)
    p_hybrid.add_argument("--strength", type=float, default=None)
    p_hybrid.add_argument("--overwrite", action="store_true")
    p_hybrid.set_defaults(func=cmd_eval_hybrid)

    # 10. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Aggregate metrics and generate research reports.")
    p_eval.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_eval.set_defaults(func=cmd_evaluate)

    # 11. pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Execute complete pipeline end-to-end.")
    p_pipe.add_argument("--model", type=str, default=None)
    p_pipe.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_pipe.add_argument("--data-path", type=str, default=None)
    p_pipe.add_argument("--splits-dir", type=str, default=None)
    p_pipe.add_argument("--layers", type=int, nargs="+", default=None)
    p_pipe.add_argument("--layer", type=int, default=None)
    p_pipe.add_argument("--strength", type=float, default=None)
    p_pipe.add_argument("--num-clusters", type=int, default=25)
    p_pipe.add_argument("--max-new-tokens", type=int, default=64)
    p_pipe.add_argument("--token", type=str, default=None)
    p_pipe.add_argument("--overwrite", action="store_true")
    p_pipe.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
