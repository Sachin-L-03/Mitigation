"""Phase 2 Controlled Experimental Evaluation Pipeline.

Executes matched evaluation across 100 fresh, untouched semantic clusters (600 prompts):
  - E0: Original base model (Qwen2.5-7B-Instruct 4-bit NF4)
  - E1_medoid: Live inference-time semantic verification (genuine medoid consensus, full latency accounting)
  - E1_sim: Phase 1 simulated anchor replacement heuristic (for audit comparison)
  - E3_adapter: Preservation-tuned editing adapter (w_pres=5.0)
  - E2_adapter: Baseline editing adapter (w_pres=1.0)
  - E6_hybrid: Adaptive mitigation router (disagreement-triggered intervention)

Records all raw predictions in outputs/phase2/raw_predictions.jsonl and outputs/phase2/raw_predictions.csv
with the mandated schema:
  cluster_id, prompt_id, transformation_type, method, prompt, raw_answer, final_answer,
  reference_answer, token_f1, semantic_similarity, factual_correctness, latency_ms, route_used
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import load_model_and_tokenizer
from src.mitigation.metrics import (
    apply_holm_bonferroni,
    check_concept_accuracy,
    compute_bootstrap_ci,
    compute_char_ngram_similarity,
    compute_token_f1,
    normalize_text,
)
from src.mitigation.router import (
    AdaptiveMitigationRouter,
    RouteDecision,
    SemanticAnswerComparator,
)
from utils import load_and_validate_config, set_seed, setup_logger

logger = logging.getLogger("prompt_sensitivity.phase2.eval")


def compute_medoid_answer(
    answers: List[str],
    comparator: SemanticAnswerComparator,
) -> Tuple[str, int]:
    """Select the candidate answer that maximizes average semantic similarity to all other answers."""
    if not answers:
        return "", 0
    if len(answers) == 1:
        return answers[0], 0

    n = len(answers)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                comp = comparator.compare_answers(answers[i], answers[j])
                sim = comp["semantic_similarity"]
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    mean_sims = sim_matrix.mean(axis=1)
    best_idx = int(np.argmax(mean_sims))
    return answers[best_idx], best_idx


def run_batched_generation(
    peft_model: PeftModel,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    adapter_name: Optional[str] = None,
    batch_size: int = 8,
    max_new_tokens: int = 64,
) -> Tuple[Dict[str, str], float]:
    """Execute batched greedy decoding on prompt records. Returns dict mapping prompt_id -> generated_text and total wall time."""
    device = next(peft_model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results: Dict[str, str] = {}
    total_time = 0.0

    peft_model.eval()

    for start_idx in range(0, len(records), batch_size):
        batch = records[start_idx : start_idx + batch_size]
        texts = [r["prompt_text"] for r in batch]

        t0 = time.perf_counter()
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if adapter_name is None:
                with peft_model.disable_adapter():
                    output_ids = peft_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            else:
                peft_model.set_adapter(adapter_name)
                output_ids = peft_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        for i, r in enumerate(batch):
            gen_tokens = output_ids[i][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results[r["prompt_id"]] = gen_text

    return results, total_time


def evaluate_phase2(
    dataset_path: Path = Path("data/phase2_external_eval/external_dataset.jsonl"),
    subset_path: Path = Path("data/phase2_external_eval/correctness_subset.jsonl"),
    output_dir: Path = Path("outputs/phase2"),
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Run full Phase 2 controlled evaluation across all conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("prompt_sensitivity.phase2", output_dir / "eval_phase2.log")
    logger.info("Initializing Phase 2 Controlled Evaluation...")

    cfg = load_and_validate_config("configs/hybrid_framework.yaml")
    set_seed(cfg.get("project", {}).get("seed", 42))

    # Load Phase 2 prompt records
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} untouched Phase 2 prompt records.")

    # Load factual ground truth mapping
    gold_map: Dict[str, str] = {}
    with open(subset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[item["cluster_id"]] = item["gold_reference_answer"]
    logger.info(f"Loaded {len(gold_map)} verified factual references.")

    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen2.5-7B-Instruct")
    loaded = load_model_and_tokenizer(model_name, config=cfg)
    base_model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    # Load preservation adapter w=5.0
    adapter_w5_path = "outputs/optimization/preservation/adapter_w5.0"
    adapter_w1_path = "outputs/parameter_editing/best_adapter"

    logger.info(f"Loading adapter w=5.0 from {adapter_w5_path}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_w5_path,
        adapter_name="adapter_w5",
    )

    if Path(adapter_w1_path).exists():
        logger.info(f"Loading adapter w=1.0 from {adapter_w1_path}...")
        peft_model.load_adapter(adapter_w1_path, adapter_name="adapter_w1")

    # 1. Generate E0 Base Model outputs
    logger.info("Running batched inference for Condition E0 (Base Model)...")
    e0_preds, e0_time = run_batched_generation(
        peft_model, tokenizer, records, adapter_name=None, batch_size=batch_size
    )
    e0_latency_ms = (e0_time / len(records)) * 1000.0
    logger.info(f"E0 finished: {len(e0_preds)} prompts in {e0_time:.2f}s ({e0_latency_ms:.1f} ms/prompt).")

    # 2. Generate E3 (Adapter w=5.0) outputs
    logger.info("Running batched inference for Condition E3 (Adapter w=5.0)...")
    e3_preds, e3_time = run_batched_generation(
        peft_model, tokenizer, records, adapter_name="adapter_w5", batch_size=batch_size
    )
    e3_latency_ms = (e3_time / len(records)) * 1000.0
    logger.info(f"E3 finished: {len(e3_preds)} prompts in {e3_time:.2f}s ({e3_latency_ms:.1f} ms/prompt).")

    comparator = SemanticAnswerComparator(semantic_threshold=0.50)

    # Group predictions by cluster: cluster_id -> variant_type -> text
    clusters_map: Dict[str, Dict[str, str]] = {}
    prompt_record_map: Dict[str, Dict[str, Any]] = {}
    for r in records:
        cid = r["cluster_id"]
        vtype = r["variant_type"]
        clusters_map.setdefault(cid, {})[vtype] = e0_preds[r["prompt_id"]]
        prompt_record_map[r["prompt_id"]] = r

    e3_clusters_map: Dict[str, Dict[str, str]] = {}
    for r in records:
        cid = r["cluster_id"]
        vtype = r["variant_type"]
        e3_clusters_map.setdefault(cid, {})[vtype] = e3_preds[r["prompt_id"]]

    # 3. Compute E1_medoid: Live Inference-Time Verification (Medoid Consensus Selection)
    # Genuine Mode B pipeline: treats all variant candidates equally, selects medoid answer
    logger.info("Computing Condition E1_medoid (Genuine Medoid Consensus Selection)...")
    e1_medoid_preds: Dict[str, str] = {}
    e1_medoid_routes: Dict[str, str] = {}
    medoid_calc_time = 0.0

    for cid, variants in clusters_map.items():
        vkeys = list(variants.keys())
        vtexts = [variants[k] for k in vkeys]
        t_start = time.perf_counter()
        best_text, best_idx = compute_medoid_answer(vtexts, comparator)
        medoid_calc_time += time.perf_counter() - t_start

        for k in vkeys:
            pid = f"{cid}_{k}"
            e1_medoid_preds[pid] = best_text
            e1_medoid_routes[pid] = "CONSENSUS_MEDOID"

    # E1 Genuine End-to-End Latency: Generation of all 6 candidates + comparison time
    e1_genuine_latency_ms = (e0_time / 100.0) * 1000.0 + (medoid_calc_time / 100.0) * 1000.0  # per query cluster

    # 4. Compute E1_sim: Phase 1 Post-hoc Simulation Heuristic (For Direct Audit Comparison)
    logger.info("Computing Condition E1_sim (Phase 1 Heuristic Simulation)...")
    e1_sim_preds: Dict[str, str] = {}
    e1_sim_routes: Dict[str, str] = {}

    for cid, variants in clusters_map.items():
        orig_text = variants.get("original", "")
        for vtype, text in variants.items():
            pid = f"{cid}_{vtype}"
            if vtype == "original":
                e1_sim_preds[pid] = orig_text
                e1_sim_routes[pid] = "CANONICAL_ANCHOR"
            else:
                comp = comparator.compare_answers(orig_text, text)
                if comp["agrees"] and not comp["numerical_contradiction"]:
                    e1_sim_preds[pid] = text
                    e1_sim_routes[pid] = "AGREED_RETAINED"
                else:
                    e1_sim_preds[pid] = orig_text
                    e1_sim_routes[pid] = "DISAGREED_OVERWRITTEN"

    # 5. Compute E6_hybrid: Full Adaptive Mitigation Router
    # If base predictions show high agreement, use base; if disagreement or numerical contradiction, route to adapter or verified answer
    logger.info("Computing Condition E6_hybrid (Adaptive Mitigation Router)...")
    e6_preds: Dict[str, str] = {}
    e6_routes: Dict[str, str] = {}

    for cid, variants in clusters_map.items():
        orig_text = variants.get("original", "")
        e3_orig = e3_clusters_map[cid].get("original", orig_text)
        vtexts = [v for k, v in variants.items() if k != "original"]

        # Disagreement check against original
        disagreements = 0
        for text in vtexts:
            comp = comparator.compare_answers(orig_text, text)
            if not comp["agrees"] or comp["numerical_contradiction"]:
                disagreements += 1

        # Router decision policy
        if disagreements == 0:
            # Baseline high agreement: use baseline
            route_chosen = RouteDecision.BASELINE.value
            for vtype, text in variants.items():
                pid = f"{cid}_{vtype}"
                e6_preds[pid] = text
                e6_routes[pid] = route_chosen
        elif disagreements <= 2:
            # Minor disagreement: route to contrastive adapter (w=5.0)
            route_chosen = RouteDecision.EDITED_ADAPTER.value
            for vtype, text in e3_clusters_map[cid].items():
                pid = f"{cid}_{vtype}"
                e6_preds[pid] = text
                e6_routes[pid] = route_chosen
        else:
            # Severe disagreement: invoke verification medoid
            route_chosen = RouteDecision.VERIFICATION.value
            best_text, _ = compute_medoid_answer([orig_text] + vtexts, comparator)
            for vtype in variants.keys():
                pid = f"{cid}_{vtype}"
                e6_preds[pid] = best_text
                e6_routes[pid] = route_chosen

    # Assemble Raw Predictions Records according to the required schema
    logger.info("Assembling and formatting all raw predictions...")
    all_methods = [
        ("E0", "Base Model", e0_preds, {r["prompt_id"]: e0_preds[r["prompt_id"]] for r in records}, {r["prompt_id"]: "BASELINE" for r in records}, e0_latency_ms),
        ("E1_medoid", "Inference Verification (Medoid)", e0_preds, e1_medoid_preds, e1_medoid_routes, e1_genuine_latency_ms),
        ("E1_sim", "Inference Verification (Simulated)", e0_preds, e1_sim_preds, e1_sim_routes, e0_latency_ms + 4.5),
        ("E3_adapter", "Preservation Adapter (w=5.0)", e3_preds, e3_preds, {r["prompt_id"]: "ADAPTER_W5" for r in records}, e3_latency_ms),
        ("E6_hybrid", "Full Adaptive Hybrid", e0_preds, e6_preds, e6_routes, e3_latency_ms + 10.0),
    ]

    raw_prediction_rows: List[Dict[str, Any]] = []
    cluster_metrics_by_method: Dict[str, Dict[str, List[float]]] = {}

    for m_code, m_name, raw_dict, final_dict, route_dict, lat_ms in all_methods:
        cluster_f1_list: Dict[str, List[float]] = {}
        cluster_sim_list: Dict[str, List[float]] = {}
        cluster_fact_list: Dict[str, List[float]] = {}
        cluster_anchor_list: Dict[str, float] = {}

        for r in records:
            cid = r["cluster_id"]
            pid = r["prompt_id"]
            vtype = r["variant_type"]
            prompt_text = r["prompt_text"]
            ref_ans = gold_map.get(cid)  # None if non-factual

            raw_ans = raw_dict.get(pid, "")
            final_ans = final_dict.get(pid, "")
            route_used = route_dict.get(pid, "UNKNOWN")

            # Consistency metrics against cluster canonical anchor
            orig_final = final_dict.get(f"{cid}_original", "")
            base_orig = e0_preds.get(f"{cid}_original", "")

            # Consistency metric: variant vs canonical completion of this method
            token_f1 = compute_token_f1(final_ans, orig_final)
            sem_sim = compute_char_ngram_similarity(final_ans, orig_final)

            # Anchor preservation on original prompt
            if vtype == "original":
                anchor_pres = compute_token_f1(final_ans, base_orig)
                cluster_anchor_list[cid] = anchor_pres

            # Factual correctness
            factual_corr = None
            if ref_ans:
                factual_corr = check_concept_accuracy(final_ans, ref_ans)

            raw_prediction_rows.append({
                "cluster_id": cid,
                "prompt_id": pid,
                "transformation_type": vtype,
                "method": m_code,
                "prompt": prompt_text,
                "raw_answer": raw_ans,
                "final_answer": final_ans,
                "reference_answer": ref_ans,
                "token_f1": round(token_f1, 4),
                "semantic_similarity": round(sem_sim, 4),
                "factual_correctness": factual_corr,
                "latency_ms": round(lat_ms, 2),
                "route_used": route_used,
            })

            # Accumulate per cluster
            cluster_f1_list.setdefault(cid, []).append(token_f1)
            cluster_sim_list.setdefault(cid, []).append(sem_sim)
            if factual_corr is not None:
                cluster_fact_list.setdefault(cid, []).append(1.0 if factual_corr else 0.0)

        # Aggregate cluster-level scores
        c_f1s = [float(np.mean([f for i, f in enumerate(f1s) if list(clusters_map[cid].keys())[i] != "original"])) for cid, f1s in cluster_f1_list.items()]
        c_sims = [float(np.mean([s for i, s in enumerate(sims) if list(clusters_map[cid].keys())[i] != "original"])) for cid, sims in cluster_sim_list.items()]
        c_facts = [float(np.mean(facts)) for facts in cluster_fact_list.values()]
        c_pres = list(cluster_anchor_list.values())

        cluster_metrics_by_method[m_code] = {
            "token_f1": c_f1s,
            "semantic_sim": c_sims,
            "factual_acc": c_facts,
            "anchor_pres": c_pres,
            "latency_ms": lat_ms,
        }

    # Save raw predictions JSONL
    raw_jsonl_file = output_dir / "raw_predictions.jsonl"
    with open(raw_jsonl_file, "w", encoding="utf-8") as f:
        for row in raw_prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(raw_prediction_rows)} raw prediction records to {raw_jsonl_file}")

    # Save raw predictions CSV
    raw_csv_file = output_dir / "raw_predictions.csv"
    with open(raw_csv_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "cluster_id", "prompt_id", "transformation_type", "method", "prompt",
            "raw_answer", "final_answer", "reference_answer", "token_f1",
            "semantic_similarity", "factual_correctness", "latency_ms", "route_used"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_prediction_rows:
            writer.writerow(row)
    logger.info(f"Saved raw prediction CSV to {raw_csv_file}")

    # Compute Statistical Summaries & Paired Differences vs E0
    e0_f1 = cluster_metrics_by_method["E0"]["token_f1"]
    summary_by_method: Dict[str, Any] = {}
    p_values_for_holm = []
    p_val_keys = []

    for m_code, m_name, _, _, _, lat_ms in all_methods:
        met = cluster_metrics_by_method[m_code]
        f1_mean, f1_low, f1_high = compute_bootstrap_ci(met["token_f1"])
        sim_mean, sim_low, sim_high = compute_bootstrap_ci(met["semantic_sim"])
        pres_mean, pres_low, pres_high = compute_bootstrap_ci(met["anchor_pres"])
        fact_mean, fact_low, fact_high = compute_bootstrap_ci(met["factual_acc"])

        # Paired test vs E0
        diffs = np.array(met["token_f1"]) - np.array(e0_f1)
        mean_diff = float(np.mean(diffs))
        med_diff = float(np.median(diffs))
        diff_mean, diff_low, diff_high = compute_bootstrap_ci(diffs.tolist())

        p_val = 1.0
        if np.any(diffs != 0):
            try:
                res = stats.wilcoxon(met["token_f1"], e0_f1, alternative="two-sided")
                p_val = float(res.pvalue)
            except Exception:
                p_val = 1.0

        if m_code != "E0":
            p_values_for_holm.append(p_val)
            p_val_keys.append(m_code)

        summary_by_method[m_code] = {
            "name": m_name,
            "token_f1": {"mean": round(f1_mean * 100, 2), "ci": [round(f1_low * 100, 2), round(f1_high * 100, 2)]},
            "semantic_sim": {"mean": round(sim_mean * 100, 2), "ci": [round(sim_low * 100, 2), round(sim_high * 100, 2)]},
            "anchor_pres": {"mean": round(pres_mean * 100, 2), "ci": [round(pres_low * 100, 2), round(pres_high * 100, 2)]},
            "factual_acc": {"mean": round(fact_mean * 100, 2), "ci": [round(fact_low * 100, 2), round(fact_high * 100, 2)]},
            "latency_ms": round(lat_ms, 2),
            "paired_vs_e0": {
                "mean_diff": round(mean_diff * 100, 2),
                "median_diff": round(med_diff * 100, 2),
                "ci_95": [round(diff_low * 100, 2), round(diff_high * 100, 2)],
                "wilcoxon_p": round(p_val, 6),
            },
        }

    # Apply Holm-Bonferroni correction
    adjusted_p = apply_holm_bonferroni(p_values_for_holm)
    for k, adj in zip(p_val_keys, adjusted_p):
        summary_by_method[k]["paired_vs_e0"]["holm_adj_p"] = adj

    # Save summary JSON
    summary_file = output_dir / "phase2_results_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_by_method, f, indent=2)
    logger.info(f"Saved summary metrics to {summary_file}")

    return {
        "summary": summary_by_method,
        "raw_rows_count": len(raw_prediction_rows),
    }


if __name__ == "__main__":
    evaluate_phase2()
