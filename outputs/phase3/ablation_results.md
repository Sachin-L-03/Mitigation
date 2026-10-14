# Ablation Study and Acceptance Criteria Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Benchmark**: Phase 3 Untouched Benchmark (`data/phase3_final_eval/`, 100 Clusters, 600 Prompts)  
**Date**: October 12, 2026  
**Target Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Status**: Completed and Verified  

---

## 1. Controlled Ablation Architecture

To understand the individual contributions of candidate budget limits, conflict detectors, and adapter parameters, we evaluated seven controlled conditions (A0 through A6) under identical inference conditions on the 100 fresh Phase 3 clusters:

| ID | Condition | Routing Logic | Extra Gens ($K$) | Token F1 (%) | Semantic Sim (%) | Factual Acc (%) | Mean Latency (ms) | Gens / Query |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0** | Base Model | Single-pass unmodified decoding | 0 | 51.80% | 66.86% | 92.00% | 267.9 ms | 1.00 |
| **A1** | Always-On Full Medoid | Unconditional consensus over 6 variants | 5 ($K=6$) | 100.00% | 100.00% | 98.00% | 1,612.2 ms | 6.00 |
| **A2** | Always-On Bounded Medoid | Unconditional consensus over 3 variants | 2 ($K=3$) | 100.00% | 98.84% | 97.60% | 806.6 ms | 3.00 |
| **A3** | Numerical/Entity-Only Router | Trigger medoid only on hard slot conflicts | 2 ($K=3$) | 72.14% | 81.33% | 94.40% | 472.6 ms | 1.76 |
| **A4** | Semantic-Disagreement Router | Trigger medoid when semantic agreement < 0.50 | 2 ($K=3$) | 79.58% | 86.72% | 95.80% | 526.5 ms | 1.96 |
| **A5** | **Cost-Aware Bounded Router (`E6_bounded`)** | Fast path on agreement ($\le 1$), Bounded medoid on conflict | 2 ($K=3$) | **85.11%** | **90.33%** | **97.20%** | **596.5 ms** | **2.22** |
| **A6** | Adapter + Adaptive Router (`E6_multi`) | 3-way route: Base (27%) $\rightarrow$ LoRA (28%) $\rightarrow$ Medoid (45%) | 0 or 2 | 77.63% | 85.42% | 96.00% | 514.8 ms | 1.90 |

---

## 2. Component Contribution Analysis

### 2.1 Bounded Candidate Verification ($K=3$ vs $K=6$)
- **Empirical Finding**: Full medoid consensus across 6 candidates takes 1,612.2 ms. Bounding candidate verification to $K=3$ (the incoming prompt plus 2 paraphrase samples) cuts verification latency from 1,612.2 ms to 806.6 ms (a **$50.0\%$ reduction in verification latency**).
- **Agreement Preservation**: The bounded medoid over 3 candidates achieves 97.60% factual accuracy and selects the identical top medoid as the 6-candidate pool in **92.4% of cases**. The marginal utility of evaluating 3 additional candidates ($K=4,5,6$) is negligible relative to the $2\times$ compute cost.

### 2.2 Hard Slot Conflict Detection (Negation & Numbers)
- **Empirical Finding**: In baseline generations (A0), 10.4% of comparisons exhibited hard contradictions (47 negation contradictions and 5 numerical contradictions).
- **Ablation Comparison (A3 vs A4)**: Routing solely on numerical/entity conflicts (A3) triggers verification in 38% of queries and achieves 72.14% Token F1. Adding semantic similarity checks (A4) triggers verification in 48% of queries and reaches 79.58% Token F1.
- Combining both signals in **A5 (`E6_bounded`)** triggers verification on 61% of queries, pushing Token F1 to 85.11% while completely suppressing all numerical and negation contradictions (0.0%).

### 2.3 Parameter Editing LoRA Adapter Evaluation (A6 vs A5)
- **Investigation of the Preservation Adapter ($w=5.0$)**:
  - The checkpoint `outputs/optimization/preservation/adapter_w5.0` was trained on 137 training clusters using an editing objective combined with preservation loss.
  - On unseen Phase 3 clusters, standalone LoRA achieves 51.43% Token F1 vs 51.80% for base Qwen ($p=0.2707$, not significant).
  - In Condition A6 (`E6_multi`), 28% of queries are routed to the adapter. This yields 77.63% Token F1, which is **$7.48$ pp lower than A5 (`E6_bounded`)**, which avoids routing to the adapter.
- **Architectural Recommendation**: The adapter provides no measurable generalization benefit on novel questions. The recommended deployment system should omit the adapter entirely, running base Qwen with adaptive bounded verification.

---

## 3. Formal Acceptance Criteria Verification

We assess the recommended deployment configuration (**`E6_bounded`**) against the eight mandatory acceptance criteria specified in the Phase 3 master protocol:

| Criterion | Requirement | Empirical Evaluation | Verdict |
|---|---|---|:---:|
| **1. Live Generation** | Gains must be produced by live generation and real routing, not simulation or cached lookups. | Executed 150 real batched inference forward passes on RTX 4090. All answers generated live without mock substitution. | **PASSED** |
| **2. Primary Consistency** | Improves declared Token F1 on unseen clusters. | Token F1 improved from 51.80% to 85.11% ($+33.31$ pp, $p < 10^{-15}$). | **PASSED** |
| **3. Factual Safety** | Factual accuracy does not regress beyond acceptable margin ($\le 2$ pp). | Factual accuracy improved from 92.00% to 97.20% ($+5.20$ pp). No regression observed. | **PASSED** |
| **4. Budget Compliance** | Router respects declared latency and generation limits. | Mean latency is 596.5 ms (well below 1,000 ms ceiling); mean generations per query is 2.22 (below 3.0 limit). | **PASSED** |
| **5. Paired Statistics** | Gains supported by cluster-level paired statistical analysis. | Paired Wilcoxon $W = 0.0$, $p = 1.83 \times 10^{-16}$, Holm-adjusted $p < 10^{-15}$, Cohen's $d = 1.134$. | **PASSED** |
| **6. Reproducibility** | Raw predictions and metrics are reproducible from disk. | 3,000 raw predictions saved in `outputs/phase3/raw_predictions.csv` and `raw_predictions.jsonl`. Deterministic seeds set. | **PASSED** |
| **7. No Cheating** | No copying of canonical completions or gold-answer leakage. | Verified by regression tests in `tests/test_router.py` and `tests/test_audited_metrics.py`. Zero gold leakage during inference. | **PASSED** |
| **8. Superiority vs Simpler** | Outperforms simpler alternatives or provides clearly justified cost advantage. | Outperforms A0 (+33.31 pp), A3 (+12.97 pp), A4 (+5.53 pp), and cuts A1 latency by 63.0% (596.5 ms vs 1,612.2 ms). | **PASSED** |

---

## 4. Final Deployment Recommendation

**Recommended Configuration**: **`E6_bounded` (Cost-Aware Bounded Adaptive Router)**
- **Model**: Base `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) without LoRA parameter edits.
- **Verification Trigger**: Semantic similarity $< 0.50$, OR numerical contradiction, OR negation contradiction, OR entity mismatch.
- **Verification Strategy**: Bounded medoid consensus over $K=3$ candidates (original query plus 2 lightweight paraphrases).
- **Fast-Path Rate**: 39% single-pass (267.9 ms).
- **Verification Rate**: 61% 3-pass (806.6 ms).
- **Expected End-to-End Latency**: 596.5 ms.
- **Expected Token F1**: 85.11% ($+33.31$ pp over base model).
- **Factual Accuracy**: 97.20% ($+5.20$ pp over base model).
