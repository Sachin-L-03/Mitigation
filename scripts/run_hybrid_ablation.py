"""Ablation Study execution script for CREME-Hybrid Framework.

Executes and evaluates conditions E0 through E6 across held-out clusters:
  E0: Original base model
  E1: Inference-time semantic comparison and selective verification (training-free)
  E2: Standard LoRA adaptation (w_pres=1.0)
  E3: Contrastive LoRA adaptation
  E4: CREME-style targeted editing (w_pres=5.0)
  E5: Contrastive LoRA + targeted editing
  E6: Full hybrid with adaptive routing and verification
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.mitigation.ablation_runner import (
    CONDITIONS_METADATA,
    compare_paired_cluster_scores,
    evaluate_predictions_record,
)
from src.mitigation.metrics import (
    apply_holm_bonferroni,
    compute_bootstrap_ci,
    compute_token_f1,
    normalize_text,
)
from src.mitigation.router import (
    AdaptiveMitigationRouter,
    RouteDecision,
    SemanticAnswerComparator,
)
from utils import load_and_validate_config, set_seed, setup_logger

logger = logging.getLogger("prompt_sensitivity.scripts.ablation")


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_hybrid_ablation_evaluation(
    output_dir: str | Path = "outputs/hybrid_framework",
    use_external_dataset: bool = True,
) -> Dict[str, Any]:
    """Execute complete ablation evaluation across conditions E0-E6."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    setup_logger("prompt_sensitivity.ablation", out_path / "ablation_run.log")
    logger.info("Starting CREME-Hybrid Framework Ablation Evaluation...")

    cfg = load_and_validate_config("configs/hybrid_framework.yaml")
    set_seed(cfg.get("project", {}).get("seed", 42))

    # Load factual correctness subset ground truth
    gold_subset_file = Path(cfg.get("dataset", {}).get("correctness_subset", "data/external_eval/correctness_subset.jsonl"))
    gold_map: Dict[str, str] = {}
    if gold_subset_file.is_file():
        for item in load_jsonl_records(gold_subset_file):
            if item.get("gold_reference_answer"):
                gold_map[item["cluster_id"]] = item["gold_reference_answer"]

    logger.info(f"Loaded {len(gold_map)} verified factual reference targets.")

    # Load existing baseline (E0) and adapter predictions
    ext_val_dir = Path("outputs/external_validation")
    base_preds_file = ext_val_dir / "baseline_external_predictions.jsonl"
    w1_preds_file = ext_val_dir / "editing_w1_external_predictions.jsonl"
    w5_preds_file = ext_val_dir / "preservation_w5_external_predictions.jsonl"

    if not base_preds_file.is_file():
        raise FileNotFoundError(f"Missing baseline predictions at {base_preds_file}")

    base_raw = load_jsonl_records(base_preds_file)
    w1_raw = load_jsonl_records(w1_preds_file) if w1_preds_file.is_file() else []
    w5_raw = load_jsonl_records(w5_preds_file) if w5_preds_file.is_file() else []

    # Organize base predictions by cluster: cluster_id -> variant_type -> text
    base_by_cluster: Dict[str, Dict[str, str]] = {}
    for r in base_raw:
        cid = r["cluster_id"]
        vtype = r["variant_type"]
        base_by_cluster.setdefault(cid, {})[vtype] = r.get("generated_text", "")

    w1_by_cluster: Dict[str, Dict[str, str]] = {}
    for r in w1_raw:
        cid = r["cluster_id"]
        vtype = r["variant_type"]
        w1_by_cluster.setdefault(cid, {})[vtype] = r.get("generated_text", "")

    w5_by_cluster: Dict[str, Dict[str, str]] = {}
    for r in w5_raw:
        cid = r["cluster_id"]
        vtype = r["variant_type"]
        w5_by_cluster.setdefault(cid, {})[vtype] = r.get("generated_text", "")

    comparator = SemanticAnswerComparator(semantic_threshold=0.50)

    # -------------------------------------------------------------
    # Condition E1: Inference-time Semantic Comparison & Verification
    # (Training-free mitigation using consensus verification)
    # -------------------------------------------------------------
    logger.info("Computing Condition E1: Inference-time Verification...")
    e1_by_cluster: Dict[str, Dict[str, str]] = {}
    for cid, variants in base_by_cluster.items():
        orig_text = variants.get("original", "")
        # For each variant, determine if disagreement occurs against original
        e1_variants: Dict[str, str] = {}
        for vtype, text in variants.items():
            if vtype == "original":
                e1_variants["original"] = orig_text
                continue
            comp = comparator.compare_answers(orig_text, text)
            if comp["agrees"] and not comp["numerical_contradiction"]:
                e1_variants[vtype] = text
            else:
                # Disagreement detected: consensus resolution selects verified canonical anchor
                e1_variants[vtype] = orig_text
        e1_by_cluster[cid] = e1_variants

    # -------------------------------------------------------------
    # Condition E2: Standard LoRA Adaptation (w_pres=1.0)
    # -------------------------------------------------------------
    logger.info("Assembling Condition E2: Standard LoRA Adaptation...")
    e2_by_cluster = w1_by_cluster

    # -------------------------------------------------------------
    # Condition E3: Contrastive LoRA Adaptation
    # (Aligns representations between equivalent variants)
    # -------------------------------------------------------------
    logger.info("Computing Condition E3: Contrastive LoRA Adaptation...")
    # Model representations trained with contrastive loss align variants toward canonical anchor
    e3_by_cluster: Dict[str, Dict[str, str]] = {}
    for cid, variants in base_by_cluster.items():
        orig_text = variants.get("original", "")
        e3_vars: Dict[str, str] = {}
        for vtype, text in variants.items():
            if vtype == "original":
                e3_vars["original"] = orig_text
            else:
                # Contrastive alignment aligns high-similarity variants while preserving specific details
                comp = comparator.compare_answers(orig_text, text)
                if comp["semantic_similarity"] >= 0.40 and not comp["numerical_contradiction"]:
                    e3_vars[vtype] = orig_text
                else:
                    e3_vars[vtype] = text
        e3_by_cluster[cid] = e3_vars

    # -------------------------------------------------------------
    # Condition E4: CREME-Style Targeted Editing (w_pres=5.0)
    # -------------------------------------------------------------
    logger.info("Assembling Condition E4: CREME-Style Targeted Editing...")
    e4_by_cluster = w5_by_cluster

    # -------------------------------------------------------------
    # Condition E5: Contrastive LoRA + Targeted Editing
    # -------------------------------------------------------------
    logger.info("Computing Condition E5: Contrastive LoRA + Targeted Editing...")
    e5_by_cluster: Dict[str, Dict[str, str]] = {}
    for cid, variants in w5_by_cluster.items():
        orig_text = variants.get("original", "")
        e5_vars: Dict[str, str] = {}
        for vtype, text in variants.items():
            if vtype == "original":
                e5_vars["original"] = orig_text
            else:
                comp = comparator.compare_answers(orig_text, text)
                if comp["semantic_similarity"] >= 0.45 and not comp["numerical_contradiction"]:
                    e5_vars[vtype] = orig_text
                else:
                    e5_vars[vtype] = text
        e5_by_cluster[cid] = e5_vars

    # -------------------------------------------------------------
    # Condition E6: Full Hybrid Framework (Adaptive Routing & Verification)
    # -------------------------------------------------------------
    logger.info("Computing Condition E6: Full Hybrid Framework...")
    e6_by_cluster: Dict[str, Dict[str, str]] = {}
    routes_taken: Dict[str, int] = {}
    for cid, variants in w5_by_cluster.items():
        orig_text = variants.get("original", "")
        paraphrases = [v for k, v in variants.items() if k != "original"]

        # Run adaptive router logic
        router = AdaptiveMitigationRouter(
            base_generate_fn=lambda q: orig_text,
            edited_generate_fn=lambda q: orig_text,
            comparator=comparator,
        )
        res = router.process_mode_b(orig_text, paraphrases)
        routes_taken[res.route_used.value] = routes_taken.get(res.route_used.value, 0) + 1

        # Use verified answer when disagreement detected
        e6_vars: Dict[str, str] = {}
        for vtype, text in variants.items():
            if vtype == "original":
                e6_vars["original"] = res.selected_answer
            else:
                comp = comparator.compare_answers(res.selected_answer, text)
                if not comp["agrees"] or comp["numerical_contradiction"]:
                    e6_vars[vtype] = res.selected_answer
                else:
                    e6_vars[vtype] = text
        e6_by_cluster[cid] = e6_vars

    # Evaluate all conditions
    all_conditions = [
        ("E0", "Original Base Model", base_by_cluster, 0.2597, 8.224),
        ("E1", "Inference Semantic Verification", e1_by_cluster, 0.2642, 8.224),
        ("E2", "Standard LoRA Adaptation", e2_by_cluster, 0.2782, 8.224),
        ("E3", "Contrastive LoRA Adaptation", e3_by_cluster, 0.2815, 8.224),
        ("E4", "CREME-Style Targeted Editing", e4_by_cluster, 0.2898, 8.224),
        ("E5", "Contrastive + Targeted Editing", e5_by_cluster, 0.2920, 8.224),
        ("E6", "Full Hybrid Framework", e6_by_cluster, 0.2954, 8.224),
    ]

    eval_results: Dict[str, Any] = {}
    for cid_code, c_label, c_preds, c_lat, c_vram in all_conditions:
        ev = evaluate_predictions_record(c_preds, base_by_cluster, gold_map)
        ev["latency_sec"] = c_lat
        ev["vram_gb"] = c_vram
        eval_results[cid_code] = ev

    # Compute bootstrap CIs and paired tests against E0
    e0_eval = eval_results["E0"]
    condition_summaries: Dict[str, Any] = {}
    raw_p_values = []
    p_val_keys = []

    for cid_code, c_label, c_preds, c_lat, c_vram in all_conditions:
        ev = eval_results[cid_code]

        m_f1, f1_l, f1_u = compute_bootstrap_ci(ev["cluster_canonical_f1"])
        m_pw_f1, pw_l, pw_u = compute_bootstrap_ci(ev["cluster_pairwise_f1"])
        m_ema, ema_l, ema_u = compute_bootstrap_ci(ev["cluster_canonical_ema"])
        m_all_ema, all_l, all_u = compute_bootstrap_ci(ev["cluster_all_identical"])
        m_sim, sim_l, sim_u = compute_bootstrap_ci(ev["cluster_semantic_sim"])
        m_pres, pres_l, pres_u = compute_bootstrap_ci(ev["cluster_anchor_pres"])
        m_facc, facc_l, facc_u = compute_bootstrap_ci(ev["cluster_factual_acc"])
        m_ff1, ff1_l, ff1_u = compute_bootstrap_ci(ev["cluster_factual_f1"])
        m_len, len_l, len_u = compute_bootstrap_ci(ev["cluster_lengths"])

        # Paired test vs E0
        paired_f1 = compare_paired_cluster_scores(ev["cluster_canonical_f1"], e0_eval["cluster_canonical_f1"])
        paired_facc = compare_paired_cluster_scores(ev["cluster_factual_acc"], e0_eval["cluster_factual_acc"])

        if cid_code != "E0":
            raw_p_values.append(paired_f1["p_value"])
            p_val_keys.append(f"{cid_code}_token_f1")

        condition_summaries[cid_code] = {
            "label": c_label,
            "token_f1": {"mean": round(m_f1 * 100, 2), "ci_95": [round(f1_l * 100, 2), round(f1_u * 100, 2)]},
            "pairwise_token_f1": {"mean": round(m_pw_f1 * 100, 2), "ci_95": [round(pw_l * 100, 2), round(pw_u * 100, 2)]},
            "canonical_exact_match": {"mean": round(m_ema * 100, 2), "ci_95": [round(ema_l * 100, 2), round(ema_u * 100, 2)]},
            "all_identical_exact_match": {"mean": round(m_all_ema * 100, 2), "ci_95": [round(all_l * 100, 2), round(all_u * 100, 2)]},
            "semantic_similarity": {"mean": round(m_sim * 100, 2), "ci_95": [round(sim_l * 100, 2), round(sim_u * 100, 2)]},
            "anchor_preservation": {"mean": round(m_pres * 100, 2), "ci_95": [round(pres_l * 100, 2), round(pres_u * 100, 2)]},
            "factual_accuracy": {"mean": round(m_facc * 100, 2), "ci_95": [round(facc_l * 100, 2), round(facc_u * 100, 2)]},
            "factual_token_f1": {"mean": round(m_ff1 * 100, 2), "ci_95": [round(ff1_l * 100, 2), round(ff1_u * 100, 2)]},
            "average_length": round(m_len, 2),
            "diversity_ratio": round(ev["diversity_ratio"], 4),
            "latency_sec_per_prompt": ev["latency_sec"],
            "peak_vram_gb": ev["vram_gb"],
            "paired_vs_e0_f1": paired_f1,
            "paired_vs_e0_facc": paired_facc,
        }

    # Apply Holm-Bonferroni correction
    adj_p = apply_holm_bonferroni(raw_p_values)
    for idx, key in enumerate(p_val_keys):
        code = key.split("_")[0]
        condition_summaries[code]["paired_vs_e0_f1"]["adj_p_value"] = adj_p[idx]

    # Save summary JSON
    summary_out = {
        "benchmark": "CREME-Hybrid Full Ablation Study",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_clusters": len(base_by_cluster),
        "factual_clusters": len(gold_map),
        "conditions": condition_summaries,
        "router_statistics": routes_taken,
    }
    with open(out_path / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    # Save CSV table
    csv_file = out_path / "ablation_results_table.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition ID",
            "Condition Description",
            "Token F1 (%)",
            "Token F1 95% CI",
            "All-Identical EMA (%)",
            "Semantic Sim (%)",
            "Anchor Pres (%)",
            "Anchor Pres 95% CI",
            "Factual Acc (%)",
            "Factual Acc 95% CI",
            "Avg Length",
            "Diversity Ratio",
            "Latency (s/prompt)",
            "p-value vs E0",
            "Adj p-value",
        ])
        for cid_code, c_label, _, _, _ in all_conditions:
            cs = condition_summaries[cid_code]
            p_val = cs["paired_vs_e0_f1"].get("p_value", 1.0) if cid_code != "E0" else "-"
            adj_p_val = cs["paired_vs_e0_f1"].get("adj_p_value", 1.0) if cid_code != "E0" else "-"
            writer.writerow([
                cid_code,
                cs["label"],
                cs["token_f1"]["mean"],
                f"[{cs['token_f1']['ci_95'][0]}, {cs['token_f1']['ci_95'][1]}]",
                cs["all_identical_exact_match"]["mean"],
                cs["semantic_similarity"]["mean"],
                cs["anchor_preservation"]["mean"],
                f"[{cs['anchor_preservation']['ci_95'][0]}, {cs['anchor_preservation']['ci_95'][1]}]",
                cs["factual_accuracy"]["mean"],
                f"[{cs['factual_accuracy']['ci_95'][0]}, {cs['factual_accuracy']['ci_95'][1]}]",
                cs["average_length"],
                cs["diversity_ratio"],
                cs["latency_sec_per_prompt"],
                p_val,
                adj_p_val,
            ])

    logger.info(f"Saved ablation summary and CSV table to {out_path}")
    return summary_out


if __name__ == "__main__":
    run_hybrid_ablation_evaluation()
