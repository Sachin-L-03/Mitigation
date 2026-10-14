# Phase 4 Consensus and Correctness Forensic Audit

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Audit Target**: "0/1,000 Contradictions" Claim and Incorrect Consensus Dynamics  
**Forensic Records**: [`outputs/phase4_forensic/contradiction_audit.csv`](file:///outputs/phase4_forensic/contradiction_audit.csv) and [`outputs/phase4_forensic/recomputed_metrics.json`](file:///outputs/phase4_forensic/recomputed_metrics.json)  
**Auditor**: ML Research Auditor & Experimental Statistician  
**Date**: October 13, 2026  

---

## 1. Executive Summary

A headline claim in the Phase 4 report is:
> *"In baseline model generations (E0), 34.00% of comparisons produced hard contradictions (38 numerical conflicts, 59 negation reversals, 282 predicate entity divergences). E6_bounded completely suppressed all 340 contradictions, driving the hard contradiction rate to 0.00% (0/1,000)."*

This forensic audit investigates whether this zero-contradiction result represents **genuine semantic contradiction prevention** or **an artifact of assigning an identical shared string across variants**.

### Key Forensic Findings:
1. **The Mechanism of Zero Contradictions**:
   - In `scripts/run_phase4_domain_evaluation.py:246-255`, whenever bounded verification is triggered (147 out of 200 clusters, or 73.5%), the router calculates the medoid answer string `best_text` across $K=3$ candidates.
   - It then executes:
     ```python
     for vtype in variants.keys():
         pid = f"{cid}_{vtype}"
         e6_bounded_preds[pid] = best_text
     ```
   - **All 6 prompt variants in the cluster receive the exact same answer string**.
   - By definition, an answer string cannot contradict itself ($0$ numerical conflicts, $0$ negation reversals, $0$ entity mismatches).
   - In the remaining 53 clusters (26.5%), the queries were routed to the fast path because the baseline answers *already had zero hard conflicts and $\le 1$ minor lexical disagreement*.
   - Therefore, the contradiction rate of $0/1,000$ was achieved by **structural answer sharing**, not by generating 6 diverse, independent answers that miraculously avoided contradicting each other.
2. **Structural Agreement $\neq$ Factual Reliability**:
   - While within-cluster contradiction is 0.00%, **factual accuracy is 76.40%** (strict) and **80.70%** (concept retrieval).
   - In **35 clusters (17.50% of the entire benchmark)**, all generated candidates agreed with each other, but **every candidate was factually false** against the verified literature ground truth (unanimous hallucination).
   - In **2 clusters (1.00%)**, the router took a correct baseline answer and replaced it with an incorrect candidate via medoid majority vote.
   - In **5 clusters (2.50%)**, the router corrected an erroneous baseline answer by selecting an accurate candidate.

---

## 2. End-to-End Execution Trace of E6_bounded

```
+---------------------------------------------------------------------------------------------------+
| 1. Incoming Prompt Cluster (6 Paraphrases: x_0 canonical, x_1..x_5 variants)                     |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. Initial Forward Passes: Base Model Generates Raw Answers (y_0..y_5)                            |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. Disagreement Detection (SemanticAnswerComparator):                                             |
|    - Compare y_0 against y_1..y_5                                                                 |
|    - Check: Numerical conflict, Negation reversal, Entity divergence, SemSim < 0.50               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         | Disagreements <= 1                              | Hard Conflict detected OR
                         | AND 0 Hard Conflicts                            | Disagreements > 1
                         v                                                 v
+---------------------------------------------------+ +---------------------------------------------+
| 4a. FAST PATH (53 clusters, 26.5%)                | | 4b. BOUNDED VERIFICATION (147 clusters, 73.5%) |
|     - Return original raw outputs as-is:          | |     - Bound candidate pool to K=3:          |
|       y_i* = y_i for each variant                 | |       C = {y_0, y_1, y_2}                   |
|     - Latency: 266.3 ms                           | |     - Compute Medoid over C:                |
|     - Contradictions: 0 (pre-screened)            | |       y* = argmin sum(dist(c_i, c_j))       |
+---------------------------------------------------+ |     - Force all 6 variants to share y*:     |
                         |                            |       y_0*..y_5* = y*                       |
                         |                            |     - Latency: 801.9 ms (3x + comparator)   |
                         |                            +---------------------------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 5. Output Verification & Metric Computation:                                                      |
|    - Pairwise contradiction across y_0*..y_5*: EXACTLY 0.00% (By String Identity on 73.5% of CIDs) |
|    - Factual Accuracy against Gold Ground Truth: 76.40% (Strict) / 80.70% (Concept)                |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 Trace Details
- **Candidate Pool Limitation**: The candidate budget $K=3$ was strictly enforced. In line 248 of `run_phase4_domain_evaluation.py`:
  `bounded_candidates = [orig_text] + vtexts[:2]`
  No retries or loops exceeded $K=3$.
- **Medoid Computation**: The distance between candidates combines $1 - \text{SemanticSimilarity}$ with penalty weights for numerical, negation, and entity mismatches.
- **Timing of Contradiction Metric**: Contradiction detection was evaluated on the **final returned outputs** (`final_answer` column in `raw_predictions.jsonl`). Because the final answers in all 147 verified clusters are identical strings, contradictions in those clusters are mathematically zero.

---

## 3. Dissecting Incorrect Consensus Dynamics

Because E6_bounded operates via consensus medoid voting over $K=3$ candidates, the system has structural failure modes when candidate agreement diverges from truth.

### Table 3.1: Consensus and Correctness Breakdown Across $N = 200$ Clusters

| Outcome Category | Definition | Cluster Count | Percentage |
|---|---|:---:|:---:|
| **Accurate Consensus** | Candidates agree, and selected answer is factually correct | 151 | 75.50% |
| **Unanimous Hallucination** | Candidates agree (or medoid selected), but ALL candidates are factually wrong | **35** | **17.50%** |
| **Recovery via Verification** | Baseline answer was wrong, but router selected an accurate candidate | **5** | **2.50%** |
| **Harmful Consensus Override** | Baseline answer was correct, but router selected an incorrect candidate | **2** | **1.00%** |
| **Residual Lexical Variation** | Fast path clusters where answers differ lexically without conflict | 7 | 3.50% |
| **Total** | | **200** | **100.00%** |

### 3.2 Analysis of Harmful Consensus Overrides (2 Clusters)
In 2 out of 200 clusters, the base model answered the canonical question correctly, but due to prompt sensitivity on the first two paraphrases ($x_1, x_2$), generated incorrect answers. The medoid chose the majority incorrect answer, degrading cluster accuracy:

1. **Cluster `phase4_num_014` (Numerical Constraints)**:
   - Question: What is the boiling point of ethanol at 1 atm?
   - Gold reference: `78.37 °C` (or ~`78.4 °C`).
   - Base canonical answer ($y_0$): "78.4 degrees Celsius" (**Correct**).
   - Variant $y_1$: "78.2 C".
   - Variant $y_2$: "78.2 degrees Celsius".
   - Router decision: Medoid selected "78.2 C" because $y_1$ and $y_2$ matched each other closely.
   - Result: Correct answer overridden by slightly erroneous consensus.
2. **Cluster `phase4_sci_027` (Natural Sciences)**:
   - Question: What is the primary function of ribosomes in a eukaryotic cell?
   - Gold reference: `protein synthesis` / `translation`.
   - Base canonical answer ($y_0$): "Protein synthesis via translation of mRNA" (**Correct**).
   - Variant $y_1$: "RNA synthesis and processing".
   - Variant $y_2$: "Transcription of RNA".
   - Router decision: $y_1$ and $y_2$ shared "RNA synthesis", winning the medoid vote.
   - Result: Correct anchor overridden by incorrect biological consensus.

### 3.3 Analysis of Recovery Overrides (5 Clusters)
In 5 clusters, the canonical prompt had produced an incomplete or wrong answer, but paraphrased variants produced the correct entity. The medoid successfully selected the accurate variant:
- Example `phase4_hist_009`: Base canonical answer omitted the treaty year; variants supplied "Treaty of Versailles (1919)". Router medoid picked the complete, accurate version.
- Net effect on correctness: $+5$ corrected minus $-2$ harmed = **$+3$ net clusters corrected** (+1.50 pp).

### 3.4 Unanimous Hallucinations (35 Clusters / 17.5%)
The most critical vulnerability of consensus routing is the **unanimous hallucination**. When an underlying LLM holds a systematic misconception or lacks pretraining knowledge about a specific entity or niche question:
- Paraphrasing the prompt does not elicit the truth; the model hallucinates identically across all paraphrases.
- E6_bounded detects high semantic agreement, declares consensus, and outputs the hallucinated answer with 100% agreement.
- **Example `phase4_sci_038`**:
  - Gold Reference: `Titan` (largest moon of Saturn).
  - All 3 generated candidates: "Enceladus is the largest moon of Saturn."
  - E6_bounded output across all variants: "Enceladus is the largest moon of Saturn."
  - Token F1: **100.00%**. Contradiction Rate: **0.00%**. Factual Correctness: **0.00%**.

---

## 4. Contradiction Audit Table Analysis

From the 340 baseline contradictions documented in [`outputs/phase4_forensic/contradiction_audit.csv`](file:///outputs/phase4_forensic/contradiction_audit.csv):
- **Numerical Conflicts (38 cases)**: E.g., baseline answered "12 pairs of cranial nerves" on variant 1, but "10 pairs" on variant 3.
- **Negation Conflicts (59 cases)**: E.g., baseline answered "Light can propagate through a vacuum" on variant 1, but "Sound waves cannot travel without a medium... Light also requires a medium" on variant 4.
- **Entity Divergences (282 cases)**: E.g., baseline attributed discovery of penicillin to "Louis Pasteur" on variant 2, and "Alexander Fleming" on variant 1.

Under E6_bounded, every single one of these 340 conflicts triggered bounded verification and received the selected medoid answer uniformly across all variants.

---

## 5. Auditor Verdict on the Zero-Contradiction Claim

1. **Validity of Measurement**: The 0.00% contradiction rate is technically true of the final outputs generated by the code.
2. **Scientific Interpretation**:
   - The report's claim that E6_bounded "completely suppressed all contradictions" must be explicitly qualified: **it suppressed contradictions by assigning an identical shared completion to all paraphrased variants within verified clusters**.
   - Zero contradictions **DOES NOT** mean zero factual errors. In 17.5% of cases, the system reliably and consistently agreed on a false statement.
   - Inference-time consensus mitigates *prompt sensitivity*, not *model ignorance*.
