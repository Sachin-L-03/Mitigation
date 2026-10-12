"""Evaluation and statistical comparison pipeline across all 4 experimental conditions.

Generates structured JSON comparison, CSV metrics table, and comprehensive Markdown research report.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from metrics import (
    compute_bootstrap_ci,
    compare_conditions_paired,
    evaluate_cluster_predictions,
)

logger = logging.getLogger("prompt_sensitivity.evaluate")

CONDITIONS = [
    ("baseline", "Condition A: Baseline"),
    ("steering_only", "Condition B: Steering Only"),
    ("editing_only", "Condition C: Editing Only"),
    ("hybrid", "Condition D: Hybrid (Edit + Steer)"),
]


def load_predictions_jsonl(filepath: str | Path) -> List[Dict[str, Any]]:
    path = Path(filepath)
    if not path.is_file():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate_all_conditions(
    predictions_dir: str | Path = "outputs",
    output_dir: str | Path = "outputs/evaluation",
    split_name: str = "test",
) -> Dict[str, Any]:
    """Load predictions for all 4 conditions, compute metrics, bootstrap CIs, and statistical comparisons."""
    pred_path = Path(predictions_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    loaded_preds: Dict[str, List[Dict[str, Any]]] = {}
    for cond_key, cond_label in CONDITIONS:
        filename = f"{cond_key}_{split_name}_predictions.jsonl"
        file_candidate = pred_path / filename
        if not file_candidate.is_file():
            # Check alternative default filename
            file_candidate = pred_path / f"{cond_key}_predictions.jsonl"

        if file_candidate.is_file():
            loaded_preds[cond_key] = load_predictions_jsonl(file_candidate)
            logger.info(f"Loaded {len(loaded_preds[cond_key])} predictions for {cond_key} from {file_candidate}")
        else:
            logger.warning(f"Prediction file not found for condition '{cond_key}': {file_candidate}")

    if "baseline" not in loaded_preds:
        raise FileNotFoundError("Baseline predictions are required to perform comparative evaluation.")

    baseline_raw = loaded_preds["baseline"]
    base_eval = evaluate_cluster_predictions(baseline_raw)

    condition_metrics: Dict[str, Dict[str, Any]] = {}

    for cond_key, cond_label in CONDITIONS:
        if cond_key not in loaded_preds:
            continue

        preds = loaded_preds[cond_key]
        raw_eval = evaluate_cluster_predictions(preds, baseline_predictions=baseline_raw)

        # Compute bootstrap CIs
        mean_ema, ema_low, ema_high = compute_bootstrap_ci(raw_eval["cluster_exact_matches"])
        mean_f1, f1_low, f1_high = compute_bootstrap_ci(raw_eval["cluster_token_f1s"])
        mean_sim, sim_low, sim_high = compute_bootstrap_ci(raw_eval["cluster_ngram_sims"])
        mean_pres, pres_low, pres_high = compute_bootstrap_ci(raw_eval["cluster_preservations"])
        mean_len, len_low, len_high = compute_bootstrap_ci(raw_eval["cluster_answer_lengths"])

        # Paired comparison against baseline
        paired_ema = compare_conditions_paired(raw_eval["cluster_exact_matches"], base_eval["cluster_exact_matches"])
        paired_f1 = compare_conditions_paired(raw_eval["cluster_token_f1s"], base_eval["cluster_token_f1s"])
        paired_sim = compare_conditions_paired(raw_eval["cluster_ngram_sims"], base_eval["cluster_ngram_sims"])

        condition_metrics[cond_key] = {
            "label": cond_label,
            "total_records": len(preds),
            "clusters_evaluated": raw_eval["total_clusters"],
            "exact_match_agreement": {
                "mean": round(mean_ema * 100, 2),
                "ci_95": [round(ema_low * 100, 2), round(ema_high * 100, 2)],
                "abs_change_pp": round(paired_ema["abs_change"] * 100, 2),
                "p_value": paired_ema["p_value"],
                "significant": paired_ema["significant"],
            },
            "token_f1_similarity": {
                "mean": round(mean_f1 * 100, 2),
                "ci_95": [round(f1_low * 100, 2), round(f1_high * 100, 2)],
                "abs_change_pp": round(paired_f1["abs_change"] * 100, 2),
                "p_value": paired_f1["p_value"],
                "significant": paired_f1["significant"],
            },
            "semantic_ngram_similarity": {
                "mean": round(mean_sim * 100, 2),
                "ci_95": [round(sim_low * 100, 2), round(sim_high * 100, 2)],
                "abs_change_pp": round(paired_sim["abs_change"] * 100, 2),
                "p_value": paired_sim["p_value"],
                "significant": paired_sim["significant"],
            },
            "anchor_preservation": {
                "mean": round(mean_pres * 100, 2),
                "ci_95": [round(pres_low * 100, 2), round(pres_high * 100, 2)],
            },
            "output_diversity_ratio": round(raw_eval["diversity_ratio"] * 100, 2),
            "average_answer_words": round(mean_len, 2),
        }

    # Save summary JSON
    summary_file = out_path / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(condition_metrics, f, indent=2)

    # Save CSV comparison
    csv_file = out_path / "condition_comparison.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition",
            "Exact Match (%)",
            "EMA 95% CI",
            "Delta EMA (pp)",
            "Token F1 (%)",
            "F1 95% CI",
            "Delta F1 (pp)",
            "Semantic Sim (%)",
            "Anchor Pres (%)",
            "Diversity (%)",
            "Avg Words",
            "p-value (F1)",
        ])
        for ckey, info in condition_metrics.items():
            writer.writerow([
                info["label"],
                f"{info['exact_match_agreement']['mean']:.2f}",
                f"[{info['exact_match_agreement']['ci_95'][0]:.1f}, {info['exact_match_agreement']['ci_95'][1]:.1f}]",
                f"{info['exact_match_agreement']['abs_change_pp']:+.2f}",
                f"{info['token_f1_similarity']['mean']:.2f}",
                f"[{info['token_f1_similarity']['ci_95'][0]:.1f}, {info['token_f1_similarity']['ci_95'][1]:.1f}]",
                f"{info['token_f1_similarity']['abs_change_pp']:+.2f}",
                f"{info['semantic_ngram_similarity']['mean']:.2f}",
                f"{info['anchor_preservation']['mean']:.2f}",
                f"{info['output_diversity_ratio']:.2f}",
                f"{info['average_answer_words']:.1f}",
                f"{info['token_f1_similarity']['p_value']:.4f}",
            ])

    # Save Markdown report
    md_file = out_path / "experiment_report.md"
    generate_markdown_report(condition_metrics, split_name, md_file)

    logger.info(f"Evaluation complete. Reports generated in {out_path}")
    return condition_metrics


def generate_markdown_report(
    metrics: Dict[str, Dict[str, Any]],
    split_name: str,
    output_path: Path,
) -> None:
    """Generate Markdown research report with comparative tables and findings."""
    lines = [
        "# Prompt-Sensitivity-Hybrid: Research Evaluation Report",
        "",
        f"**Evaluation Split**: `{split_name}` | **Target GPU**: NVIDIA RTX 4090 | **Quantization**: 4-bit NF4",
        "",
        "## 1. Experimental Conditions Comparison",
        "",
        "| Condition | Exact Match (EMA) | Token F1 | Semantic Sim | Anchor Pres. | Diversity | Avg Length | Δ F1 vs Base (pp) | p-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ckey, d in metrics.items():
        label = d["label"]
        ema = f"{d['exact_match_agreement']['mean']:.1f}% [{d['exact_match_agreement']['ci_95'][0]:.1f}-{d['exact_match_agreement']['ci_95'][1]:.1f}]"
        f1 = f"{d['token_f1_similarity']['mean']:.1f}% [{d['token_f1_similarity']['ci_95'][0]:.1f}-{d['token_f1_similarity']['ci_95'][1]:.1f}]"
        sim = f"{d['semantic_ngram_similarity']['mean']:.1f}%"
        pres = f"{d['anchor_preservation']['mean']:.1f}%"
        div = f"{d['output_diversity_ratio']:.1f}%"
        words = f"{d['average_answer_words']:.1f} w"
        delta = f"{d['token_f1_similarity']['abs_change_pp']:+.2f} pp" if ckey != "baseline" else "-"
        pval = f"{d['token_f1_similarity']['p_value']:.4f}" if ckey != "baseline" else "-"
        sig = " *" if d['token_f1_similarity'].get('significant') else ""

        lines.append(f"| **{label}** | {ema} | {f1} | {sim} | {pres} | {div} | {words} | {delta} | {pval}{sig} |")

    lines.extend([
        "",
        "> `*` indicates statistical significance at $p < 0.05$ (Wilcoxon signed-rank test against baseline).",
        "",
        "## 2. Key Findings & Hypothesis Evaluation",
        "",
    ])

    base_f1 = metrics.get("baseline", {}).get("token_f1_similarity", {}).get("mean", 0.0)
    steer_f1 = metrics.get("steering_only", {}).get("token_f1_similarity", {}).get("mean", 0.0)
    edit_f1 = metrics.get("editing_only", {}).get("token_f1_similarity", {}).get("mean", 0.0)
    hybrid_f1 = metrics.get("hybrid", {}).get("token_f1_similarity", {}).get("mean", 0.0)

    lines.append(f"- **Baseline Performance**: Mean Token F1 across paraphrases is **{base_f1:.2f}%**.")
    if "steering_only" in metrics:
        lines.append(f"- **Representation Steering Only**: Token F1 is **{steer_f1:.2f}%** (Δ = {steer_f1 - base_f1:+.2f} pp).")
    if "editing_only" in metrics:
        lines.append(f"- **Parameter Editing Only**: Token F1 is **{edit_f1:.2f}%** (Δ = {edit_f1 - base_f1:+.2f} pp).")
    if "hybrid" in metrics:
        lines.append(f"- **Hybrid Intervention**: Token F1 is **{hybrid_f1:.2f}%** (Δ = {hybrid_f1 - base_f1:+.2f} pp).")

    lines.extend([
        "",
        "## 3. Methodological Notes",
        "- **Reference Answer Notice**: Source dataset contains no gold answers; reference answers are strictly null.",
        "- **Factual Correctness**: High paraphrase agreement reflects invariance to linguistic phrasing, not factual verification.",
        "- **Preservation Regularization**: Monitored via anchor preservation on original prompts to detect collapse or regression.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
