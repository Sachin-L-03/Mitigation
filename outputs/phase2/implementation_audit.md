# Implementation Audit Report: CREME-Hybrid Framework

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Audit Target**: Phase 1 Implementation, Experiment Scripts, Checkpoints, and Metric Functions  
**Date**: October 12, 2026  
**Auditor**: ML Research Engineering Audit Team  

---

## 1. Overview and Executive Summary

This implementation audit independently examines the code, configuration, checkpoints, raw prediction files, and experiment scripts responsible for the reported results in the CREME-Hybrid Prompt Sensitivity framework.

The project report previously claimed the following metrics on 100 independent semantic clusters (600 prompts) with a 40-cluster factual subset:

| Condition | Token F1 | Semantic similarity | Factual accuracy | Latency |
|---|---:|---:|---:|---:|
| **E0: Base model** | 37.23% | 54.82% | 89.58% | 0.2597 s |
| **E1: Inference semantic verification** | 78.76% | 86.11% | 90.42% | 0.2642 s |
| **E2: Standard LoRA** | 37.44% | 55.55% | 90.00% | 0.2782 s |
| **E3: Contrastive LoRA** | 77.82% | 82.15% | 92.08% | 0.2815 s |
| **E4: CREME-style editing** | 37.20% | 55.11% | 90.00% | 0.2898 s |
| **E5: Contrastive LoRA + editing** | 69.99% | 76.68% | 92.50% | 0.2920 s |
| **E6: Full hybrid** | 78.98% | 86.36% | 90.42% | 0.2954 s |

### Primary Audit Findings:
1. **The reported leaps in Token F1 (~37% -> ~79% in E1, E3, E6) are post-processing substitution artifacts**, not outputs directly sampled from a model. In [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py), when disagreement or similarity conditions were met, variant responses were explicitly replaced by copying the exact text of the canonical prompt (`orig_text`).
2. **E3 was not evaluated from a trained Contrastive LoRA checkpoint**. No contrastive adapter weights were trained or loaded in [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py). Instead, E3 was simulated by taking baseline predictions and setting `e3_vars[vtype] = orig_text` whenever semantic similarity $\ge 0.40$.
3. **E1's 100% anchor preservation is tautological by construction**. The evaluation script hardcodes `e1_variants["original"] = orig_text`, bypassing any intervention on canonical prompts.
4. **E1 latency (0.2642 s) omitted multi-pass generation costs**. The latency measured only one forward pass plus Python string comparison, omitting the $K=6$ generations required to produce candidate answers at inference time.
5. **Existing adapters are intact and verified**:
   - `outputs/parameter_editing/best_adapter/adapter_model.safetensors` ($w=1.0$, 161.5 MB)
   - `outputs/optimization/preservation/adapter_w5.0/adapter_model.safetensors` ($w=5.0$, 161.5 MB)

---

## 2. File and Function Attribution Matrix

Every metric and condition reported in the project is mapped to its exact source file and function below:

| Component / Metric | Source File | Line Range | Function / Implementation |
|---|---|---|---|
| **E0 (Base Model)** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 97–101 | Loaded from `outputs/external_validation/baseline_external_predictions.jsonl` |
| **E1 (Inference Verification)** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 121–137 | Post-processing consensus override copying `orig_text` on disagreement |
| **E2 (Standard LoRA $w=1.0$)** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 142–144 | Loaded from `outputs/external_validation/editing_w1_external_predictions.jsonl` |
| **E3 (Contrastive LoRA)** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 149–165 | Post-processing override copying `orig_text` when $\text{sim} \ge 0.40$ |
| **E4 (CREME Editing $w=5.0$)** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 170–172 | Loaded from `outputs/external_validation/preservation_w5_external_predictions.jsonl` |
| **E5 (Contrastive + Editing)** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 177–190 | Post-processing override on $w=5.0$ predictions copying `orig_text` when $\text{sim} \ge 0.45$ |
| **E6 (Full Hybrid)** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 195–223 | [`AdaptiveMitigationRouter.process_mode_b`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/router.py#L180-L240) post-processing override |
| **Semantic Comparison** | [`src/mitigation/router.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/router.py) | 83–115 | [`SemanticAnswerComparator.compare_answers`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/router.py#L83-L115) |
| **Token F1** | [`src/mitigation/metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py) | 36–59 | [`compute_token_f1`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py#L36-L59) |
| **Char 3-gram Similarity** | [`src/mitigation/metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py) | 61–86 | [`compute_char_ngram_similarity`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py#L61-L86) |
| **Exact Match Agreement** | [`src/mitigation/metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py) | 115–184 | [`evaluate_cluster_consistency`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py#L115-L184) |
| **Factual Accuracy** | [`src/mitigation/metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py) | 89–112 | [`check_concept_accuracy`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py#L89-L112) |
| **Anchor Preservation** | [`src/mitigation/ablation_runner.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/ablation_runner.py) | 91–95 | `compute_token_f1(orig_ans, base_orig)` |
| **Cluster Bootstrap CI** | [`src/mitigation/metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py) | 187–212 | [`compute_bootstrap_ci`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py#L187-L212) |
| **Wilcoxon Signed-Rank** | [`src/mitigation/metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py) | 215–258 | [`compare_conditions_paired`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py#L215-L258) |
| **Holm-Bonferroni** | [`src/mitigation/metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py) | 260–276 | [`apply_holm_bonferroni`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/metrics.py#L260-L276) |
| **Reported Latency** | [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py) | 226–232 | Hardcoded tuple values in `all_conditions` |

---

## 3. Detailed Forensic Analysis

### 3.1 Condition E1: Inference Semantic Verification
The implementation in [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py#L122-L137) is:
```python
for cid, variants in base_by_cluster.items():
    orig_text = variants.get("original", "")
    e1_variants = {}
    for vtype, text in variants.items():
        if vtype == "original":
            e1_variants["original"] = orig_text
            continue
        comp = comparator.compare_answers(orig_text, text)
        if comp["agrees"] and not comp["numerical_contradiction"]:
            e1_variants[vtype] = text
        else:
            e1_variants[vtype] = orig_text  # <--- OVERWRITING WITH CANONICAL TEXT
    e1_by_cluster[cid] = e1_variants
```
- **Forensic finding**: 290 out of 500 non-original variants (58.0%) triggered disagreement and were overwritten with `orig_text`.
- **Impact on metrics**: Comparing `orig_text` to `orig_text` yields Token F1 = 1.0 (100.0%) and Exact Match = 1.0. This explains why cluster Token F1 doubled from 37.23% to 78.76%.
- **Deployment disconnect**: In production, a user does not supply the canonical original answer alongside a paraphrase query. Replacing an answer with an already-known gold anchor is an oracle operation unless candidate generations and selection are performed autonomously.

### 3.2 Condition E3: Contrastive LoRA Adaptation
The implementation in [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py#L151-L165) is:
```python
for cid, variants in base_by_cluster.items():
    orig_text = variants.get("original", "")
    e3_vars = {}
    for vtype, text in variants.items():
        if vtype == "original":
            e3_vars["original"] = orig_text
        else:
            comp = comparator.compare_answers(orig_text, text)
            if comp["semantic_similarity"] >= 0.40 and not comp["numerical_contradiction"]:
                e3_vars[vtype] = orig_text  # <--- OVERWRITING WITH CANONICAL TEXT
            else:
                e3_vars[vtype] = text
    e3_by_cluster[cid] = e3_vars
```
- **Forensic finding**: 359 out of 500 variants (71.8%) had semantic similarity $\ge 0.40$ and were overwritten with `orig_text`.
- **Absence of physical adapter**: Despite the report discussing contrastive loss training ($\mathcal{L}_{\text{con}}$), no contrastive adapter weights were trained or loaded in `scripts/run_hybrid_ablation.py`. E3 was evaluated as a simulated representation alignment heuristic.

### 3.3 Latency Accounting Discrepancy
In [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py#L226-L232):
- E0 latency was recorded as **0.2597 s** (the actual batched forward pass time of base Qwen).
- E1 latency was recorded as **0.2642 s** ($+1.7\%$).
- **Discrepancy**: E1's 0.2642 s only accounts for the 0.2597 s base forward pass plus ~4.5 ms of Python string comparison. It completely omits:
  1. Paraphrase generation time (if generating equivalent prompts).
  2. Generation of the $K=6$ candidate answers.
  3. Real end-to-end time: Generating 6 candidates at 0.2597 s per candidate requires **$\approx 1.56$ seconds**, a $600\%$ latency increase, not $1.7\%$.

---

## 4. Summary of Verification Status

| Finding | Original Status | Audited Status | Explanation |
|---|---|---|---|
| E0 Baseline Score | 37.23% F1 | **VERIFIED** | Real generation on 100 external clusters |
| E1 Score (78.76%) | Claimed Genuine | **SIMULATION ARTIFACT** | Overwrote 58% of answers with canonical text |
| E3 Score (77.82%) | Claimed Trained Adapter | **SIMULATION ARTIFACT** | Overwrote 71.8% of answers with canonical text; no adapter loaded |
| E1 Anchor Pres (100%) | Claimed Genuine | **TAUTOLOGICAL** | `e1_variants["original"] = orig_text` by definition |
| E1 Latency (0.2642s) | Claimed End-to-End | **INCOMPLETE ACCOUNTING** | Excludes candidate generation passes |
| Factual Acc (~90%) | Claimed High | **VERIFIED DEFINITION** | Concept presence matching |
| Factual F1 (~9%) | Claimed Low | **VERIFIED DEFINITION** | Length penalty between 48-word text and 1-word gold |
