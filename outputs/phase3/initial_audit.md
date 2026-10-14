# Phase 3 Initial Audit Report: Baseline Verification and Asset Inventory

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Target Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Target Model**: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 Precision)  
**Date**: October 12, 2026  
**Auditor**: Senior ML Research Engineer  

---

## 1. Executive Summary & Git Environment State

This initial audit establishes the verified foundation for Phase 3 before implementing any architectural modifications, threshold re-tunings, or dataset expansions.

### 1.1 Git Status & Branch Verification
- **Active Git Branch**: `creme-hybrid-framework`
- **Repository Path**: `c:\Users\Student5\Desktop\Mitigation`
- **Working Tree State**: Uncommitted modifications present on `optimization.py`, `outputs/logs/run.log`, and steering sweep logs. All Phase 1 and Phase 2 outputs are preserved intact in `outputs/hybrid_framework/`, `outputs/external_validation/`, and `outputs/phase2/`.
- **Pre-Execution Test Suite**:
  ```bash
  .venv/Scripts/pytest tests/ -v
  ============================= 67 passed in 4.94s ==============================
  ```
  All 67 unit and integration tests are verified passing with zero regressions.

---

## 2. Inventory of Preserved Artifacts and Checkpoints

All prior artifacts have been audited on disk and verified intact:

### 2.1 Model Checkpoints and Trained Adapters:
- `outputs/parameter_editing/best_adapter/adapter_model.safetensors` (161,539,856 bytes, SHA-256 verified): Preservation-tuned editing adapter with $w_{\text{pres}}=1.0$.
- `outputs/optimization/preservation/adapter_w5.0/adapter_model.safetensors` (161,539,856 bytes, SHA-256 verified): Preservation-tuned editing adapter with $w_{\text{pres}}=5.0$.
- **Finding regarding Contrastive LoRA**: As established in Phase 2, no trained adapter exists for contrastive representation alignment; Phase 1 simulated it via string-copying heuristics.

### 2.2 Dataset Artifacts:
- **Original Dataset**: `data/paraphrased_prompts_axis_tagged_qa.csv` (196 clusters, 1,176 prompt records).
- **Phase 1 Splits**: `data/splits/train.jsonl` (137 clusters), `val.jsonl` (29 clusters), `test.jsonl` (30 clusters).
- **Phase 1 External Eval**: `data/external_eval/external_dataset.jsonl` (100 clusters, 600 records; SHA-256: `911cf2...`).
- **Phase 2 Untouched External Eval**: `data/phase2_external_eval/external_dataset.jsonl` (100 clusters, 600 records; SHA-256: `50418ed32a0ceb3c306f72b85bb583dfd7f6f3d8d887b0c7a7119b80b5fa252a`).
- **Phase 2 Correctness Subset**: `data/phase2_external_eval/correctness_subset.jsonl` (50 factual clusters, 300 records; SHA-256: `a668aa61f64fcec15f5f8fc02f8e1a56c3b425babd5ee577a50ad88ffb585284`).

### 2.3 Phase 2 Prediction Files & Metric Summaries:
- `outputs/phase2/raw_predictions.jsonl` (3,000 records, 3.13 MB)
- `outputs/phase2/raw_predictions.csv` (3,000 records, 2.41 MB)
- `outputs/phase2/phase2_results_summary.json` (Summary statistics across E0, E1_medoid, E1_sim, E3_adapter, E6_hybrid)
- `outputs/phase2/transformation_analysis.csv` & `.md`
- `outputs/phase2/final_phase2_research_report.md`

---

## 3. Audit of the Current E6 Implementation & Decision Rules

In [`scripts/run_phase2_evaluation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_phase2_evaluation.py#L271-L321), condition E6 is implemented as follows:

```python
for cid, variants in clusters_map.items():
    orig_text = variants.get("original", "")
    vtexts = [v for k, v in variants.items() if k != "original"]

    # Disagreement count against canonical prompt completion
    disagreements = 0
    for text in vtexts:
        comp = comparator.compare_answers(orig_text, text)
        if not comp["agrees"] or comp["numerical_contradiction"]:
            disagreements += 1

    # Router decision policy:
    if disagreements == 0:
        route_chosen = RouteDecision.BASELINE.value
        # Use baseline outputs
    elif disagreements <= 2:
        route_chosen = RouteDecision.EDITED_ADAPTER.value
        # Use adapter outputs (w=5.0)
    else:
        route_chosen = RouteDecision.VERIFICATION.value
        # Invoke medoid consensus across candidates
```

### 3.1 Current Router Parameters and Thresholds:
- `semantic_threshold`: $0.50$ (calculated as $0.4 \times \text{Token F1} + 0.6 \times \text{Char 3-gram Sim}$).
- `disagreement_trigger`:
  - 0 disagreements $\to$ `BASELINE`
  - 1–2 disagreements $\to$ `EDITED_ADAPTER`
  - $\ge 3$ disagreements $\to$ `VERIFICATION`

---

## 4. Phase 2 Baseline Results Summary

Across the 100 clusters and 600 prompts in `data/phase2_external_eval/`:

| Metric | E0 (Base Model) | E3 (Adapter $w=5.0$) | E1 (Medoid Consensus) | E6 (Adaptive Router) |
|---|:---:|:---:|:---:|:---:|
| **Within-Cluster Token F1** | 52.78% | 53.26% | 100.00% | 71.30% |
| **Semantic Similarity** | 66.62% | 67.07% | 100.00% | 80.96% |
| **Anchor Preservation** | 100.00%* | 96.93% | 74.79% | 86.92% |
| **Factual Accuracy** | 91.67% | 91.67% | 94.00% | 93.00% |
| **Reported Latency** | 269.1 ms | 287.6 ms | 1,617.6 ms | 297.6 ms** |

*\*E0 anchor preservation is 100% by self-referential definition.*  
*\*\*Note on E6 Latency: In Phase 2, E6 latency was logged as single-pass adapter latency + 10 ms (297.6 ms), but forensic inspection reveals that 30% of queries triggered the 1,617.6 ms verification route, making true average latency $\approx 682.2$ ms. This discrepancy will be systematically resolved in Phase 3.*
