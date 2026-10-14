# Phase 4 Forensic Verification and Research Validity Audit Report

**Repository**: `prompt-sensitivity-hybrid`  
**Git Branch**: `creme-hybrid-framework` (Commit: `7536b4f`)  
**Evaluation Benchmark**: Phase 4 Domain-Shift Benchmark (`data/phase4_domain_eval/`, $N=200$ Semantic Clusters, 1,200 Prompts)  
**Host Hardware**: NVIDIA GeForce RTX 4090 (24,564 MiB VRAM), Windows 11, CUDA 13.3  
**Model Under Audit**: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 precision, unadapted base)  
**Primary Method**: `E6_bounded` (Cost-Aware Bounded Adaptive Router, $K=3$)  
**Role**: Skeptical ML Research Auditor, Experimental Statistician & Systems Engineer  
**Date**: October 13, 2026  
**Auditor Output Directory**: `outputs/phase4_forensic/`  

---

## 1. Executive Summary & Audit Verdict

This forensic audit was commissioned to independently verify the implementation, raw prediction records, statistical calculations, consensus mechanics, and serving measurements of Phase 4. All evaluations were executed directly on repository artifacts without importing precomputed summaries.

### Headline Audit Findings:
1. **Raw Predictions Reconciled (6,000 vs 4,800 records)**:
   - The reported 6,000 live predictions are fully accounted for: `scripts/run_phase4_domain_evaluation.py` ran 5 conditions (E0, E3, E1, E6_bounded, E6_enhanced) across 1,200 prompts ($5 \times 1,200 = 6,000$ rows).
   - Table 3.1 in the original report displayed only the 4 primary baselines (4,800 rows).
   - Both `raw_predictions.csv` and `raw_predictions.jsonl` match with **zero field mismatches**.
2. **Consistency Gains Verified**:
   - `E6_bounded` improves Canonical Token F1 from **57.43% to 90.64%** ($+33.21$ pp, paired Wilcoxon $W=0.0$, $p = 7.15 \times 10^{-26}$, Cohen's $d = 1.456$).
   - Pairwise exact match rises from **5.37% to 75.63%** (+70.26 pp gain).
3. **LoRA Parameter Editing Generalization Debunked**:
   - Preservation LoRA (E3) achieved **57.37% vs 57.43% for base Qwen** ($-0.06$ pp difference, $p = 0.5232$). Parameter editing does not generalize to novel question domains.
4. **Factual Non-Inferiority Claim Fails**:
   - The report claimed that factual accuracy was preserved within a 2-percentage-point margin based on a non-significant two-sided test ($p = 0.442$).
   - A formal one-sided non-inferiority test ($H_0: \delta \le -0.02$ vs $H_1: \delta > -0.02$) yields **$p_{\text{NI}} = 0.1978$** for strict factual accuracy (lower bound $-3.58$ pp) and **$p_{\text{NI}} = 0.1634$** for concept retrieval (lower bound $-3.11$ pp).
   - **The data does NOT establish factual non-inferiority within 2 percentage points**.
5. **Zero-Contradiction Result is Structural, Not Preventative**:
   - Hard contradictions fell from 340 (34.00%) in baseline to 0 (0.00%) in E6_bounded because verified clusters (73.5%) were assigned the identical medoid string across all 6 variants.
   - Zero contradictions does NOT mean zero factual errors: in **35 clusters (17.50%)**, the candidates agreed on an answer that was universally false against verified literature references (unanimous hallucination).
6. **Analytical Latency Boundary**:
   - The reported **660.1 ms** latency was derived from an analytical cost model on offline batched passes ($266.3$ ms fast path, $801.9$ ms verification), not measured on an interactive live serving endpoint.

### Overall Recommendation:
`E6_bounded` is an **empirically validated research candidate for prompt sensitivity mitigation**, but **NOT a production-ready system**. It stabilizes lexical consistency and suppresses slot contradictions via medoid voting, but cannot guarantee factual reliability, lacks NLI-based constraint handling, and exhibits blindspots on shared hallucinations.

---

## 2. Master Forensic Claim Reconciliation Table

| Claim | Original Value | Recomputed Value | Status | Evidence | Impact on Conclusions |
|---|:---:|:---:|:---:|---|---|
| **E0 Baseline Token F1** | 57.43% | 57.43% | **Verified** | `scripts/audit_phase4_metrics.py:364` | Baseline sensitivity confirmed. |
| **E3 LoRA Token F1** | 57.37% | 57.37% | **Verified** | `scripts/audit_phase4_metrics.py:364` | LoRA editing does not generalize ($p=0.52$). |
| **E6_bounded Token F1** | 90.64% | 90.64% | **Verified** | `scripts/audit_phase4_metrics.py:364` | $+33.21$ pp gain independently verified. |
| **Pairwise Exact Match** | 77.30% | 75.63% (Pairwise)<br>77.30% (Canonical) | **Verified with qualification** | `scripts/audit_phase4_metrics.py:368-369` | Original report mislabelled Canonical EM as Pairwise EM. True pairwise EM is 75.63%. |
| **All-Identical Cluster Rate** | Implied high | 74.00% (148/200) | **Verified** | `scripts/audit_phase4_metrics.py:370` | 74% of clusters achieve identical outputs across all 6 variants. |
| **Anchor Preservation** | 83.47% | 83.47% | **Verified** | `scripts/audit_phase4_metrics.py:371` | Router preserves baseline anchor in 83.5% of cases. |
| **Concept Retrieval Rate** | 80.70% (as "Factual Acc") | 80.70% | **Verified** | `scripts/audit_phase4_metrics.py:373` | Substring entity retrieval matches report. |
| **Contextual Factual Accuracy** | Not reported | 76.40% (vs 77.20% E0) | **New Metric** | `scripts/audit_phase4_metrics.py:374` | Penalizing negation/numeric conflicts lowers factual accuracy by 0.80 pp. |
| **Factual Non-Inferiority ($\le 2$ pp)**| Claimed preserved ($p=0.442$) | Fails ($p_{\text{NI}}=0.198$, lower bound $-3.58$ pp) | **Incorrect** | `outputs/phase4_forensic/statistical_reanalysis.md` | Non-inferiority is NOT established. Two-sided test fallacy corrected. |
| **Zero Contradictions (0/1,000)** | 0.00% | 0.00% | **Verified with qualification** | `scripts/audit_phase4_metrics.py:375` | 0 contradictions achieved by assigning identical medoid string across variants. |
| **Unanimous Hallucinations** | Not highlighted | 35 clusters (17.50%) | **New Finding** | `scripts/audit_phase4_metrics.py:381` | Candidates agree on false answers in 17.5% of benchmark. |
| **Mean Latency** | 660.1 ms | 660.1 ms (Analytical) | **Verified with qualification** | `outputs/phase4_forensic/deployment_reanalysis.md` | Calculated analytically from batched generation, not live serving. |
| **Latency Savings vs E1** | 58.8% | 58.82% | **Verified** | $1 - (660.1 / 1603.1) = 0.5882$ | Mathematical calculation holds for analytical model. |
| **Generations / Query** | 2.47 | 2.47 | **Verified** | $(53\times 1 + 147\times 3)/200 = 2.47$ | Generation budget verified. |
| **Detector Stress Accuracy** | 80.0% | 80.0% (20/25) | **Verified** | `outputs/phase4_forensic/detector_validation.md` | Confusion matrix verified: TP=7, TN=13, FP=2, FN=3. |
| **Alias Fast-Path Routing** | Claimed STRESS_04/19 on fast path | STRESS_04/19 routed to verification | **Incorrect narrative** | `outputs/phase4/paraphrase_stress_test.csv:5,20` | Report narrative stated exact opposite of raw CSV records. |
| **Total Test Suite** | 79 passing tests | 87 passing tests | **Verified & Expanded** | `tests/test_phase4_forensic.py` | 8 new forensic tests added; 100% green. |

---

## 3. Answers to the 10 Primary Audit Questions

### Question 1: Can the reported Phase 4 metrics be reproduced from the actual raw prediction files?
**Answer**: **YES, with one labelling qualification.**
Canonical Token F1 (57.43% $\to$ 90.64%), Semantic Similarity (70.70% $\to$ 93.82%), Anchor Preservation (83.47%), Concept Retrieval (80.70%), and Contradiction counts (340 $\to$ 0) reproduce exactly from `raw_predictions.jsonl`. However, the column labelled "Pairwise EM" in Table 3.1 was actually **Canonical Exact Match** (77.30%). True pairwise exact match across all 15 pairs is **75.63%**.

### Question 2: Are the 6,000 reported predictions accounted for when the four displayed conditions appear to require only 4,800 records?
**Answer**: **YES.**
The generation script `scripts/run_phase4_domain_evaluation.py` ran 5 conditions: `E0` (1,200), `E3` (1,200), `E1` (1,200), `E6_bounded` (1,200), and `E6_enhanced` (1,200), yielding exactly 6,000 records. Table 3.1 of the report displayed only the 4 primary conditions (4,800 records). Condition `E6_enhanced` made identical routing decisions to `E6_bounded` on this benchmark. Both CSV and JSONL contain all 6,000 rows with zero record mismatches.

### Question 3: Are Token F1, semantic similarity, pairwise exact match, anchor preservation, and factual accuracy computed correctly?
**Answer**: **YES in implementation, but conflated in reporting.**
- Token F1 is computed per prompt variant against the canonical answer, averaged per cluster, and then macro-averaged across 200 clusters.
- Semantic similarity combines 0.40 Token F1 and 0.60 char 3-gram cosine similarity with a 0.70 containment floor.
- Pairwise EM was conflated with Canonical EM in report tables.
- "Factual Accuracy" in the report represents keyword/concept substring retrieval. Contextual factual accuracy (penalizing contradiction with gold) is 76.40% for E6_bounded vs 77.20% for baseline.

### Question 4: Are all statistical tests performed at the correct independent unit, accounting for six paraphrases within each semantic cluster?
**Answer**: **YES.**
Statistical tests were executed using the semantic cluster ($N=200$) as the independent unit, properly avoiding Type I error inflation from pooling 1,200 prompt records. The Wilcoxon signed-rank test on matched cluster-level Token F1 yields $W=0.0$, $p = 7.15 \times 10^{-26}$, Cohen's $d = 1.456$.

### Question 5: Does the evidence establish factual non-inferiority within a 2-percentage-point margin?
**Answer**: **NO.**
The original report committed a classic statistical fallacy by citing a non-significant two-sided test ($p=0.442$) as evidence that the safety margin was respected. A formal one-sided non-inferiority test ($H_0: \delta \le -0.02$) at $\alpha=0.025$ yields $t_{\text{NI}} = 0.8514$, $p_{\text{NI}} = 0.1978$ (97.5% lower bound $-3.58$ pp). The sample of 200 clusters is underpowered to establish non-inferiority within 2.0 pp.

### Question 6: Does the reported zero-contradiction result reflect true contradiction suppression, shared-answer selection, or a limitation in the contradiction metric?
**Answer**: **SHARED-ANSWER SELECTION.**
In all 147 clusters routed to verification (73.5%), all 6 variants were assigned the identical medoid string `best_text`. Identical strings cannot contradict each other. In the remaining 53 fast-path clusters (26.5%), answers were pre-screened to have 0 hard conflicts. The contradiction metric evaluated final outputs; hence, zero contradictions was guaranteed by construction on verified clusters.

### Question 7: What fraction of the time does the router select an incorrect consensus answer?
**Answer**: **17.50% (35 out of 200 clusters).**
In 35 clusters, all generated candidates agreed on a false completion (unanimous hallucination). Additionally, in 2 clusters (1.00%), the router medoid took a correct baseline answer and replaced it with an incorrect candidate. In 5 clusters (2.50%), the router corrected an erroneous baseline answer.

### Question 8: Does the reported 660.1 ms latency include the entire end-to-end serving path?
**Answer**: **NO.**
The 660.1 ms figure was computed from an analytical cost model applied to offline batch-size-8 passes: $266.3$ ms (fast path) and $801.9$ ms (verification path). It excludes network I/O, server queue latency, streaming decodes, and runtime paraphrase generation.

### Question 9: Does the 25-case detector stress test support the strength of the report's claims?
**Answer**: **NO.**
While the raw numbers (80% accuracy, 78% precision, 70% recall, 0.74 F1) are mathematically verified, a 25-case sample has wide confidence intervals ([59.3%, 93.2%]). Furthermore, the report claimed entity aliases kept queries on the fast path, whereas the CSV explicitly records STRESS_04 and STRESS_19 as false positives that triggered verification. Regex-based detection fails on mathematical signs, multi-slot answers, and prompt constraints.

### Question 10: What conclusions can honestly be stated after all discrepancies are resolved?
**Answer**:
1. `E6_bounded` reliably and powerfully stabilizes LLM response consistency across paraphrased prompts ($+33.21$ pp Token F1, $+70.26$ pp pairwise exact match).
2. Parameter editing via LoRA does not generalize to unseen domains.
3. Zero contradictions reflects structural string assignment, not universal factual correctness.
4. Factual non-inferiority within 2.0 pp cannot be claimed from the current sample size ($N=200$).
5. Bounded verification introduces a $2.47\times$ compute multiplier over baseline generation.

---

## 4. Technical Architecture & Inference Specification

- **Model Checkpoint**: `Qwen/Qwen2.5-7B-Instruct` (Unadapted base).
- **Quantization**: 4-bit NormalFloat4 (NF4), double quantization, compute dtype `torch.bfloat16`.
- **Decoding Protocol**: Greedy decoding (`do_sample=False`, `max_new_tokens=64`).
- **Candidate Budget**: Strictly bounded to $K \le 3$ generations per query.
- **Routing Invariant**:
  - If disagreements $\le 1$ and 0 hard conflicts: Fast path ($K=1$, 266.3 ms).
  - If disagreements $> 1$ or hard conflict detected: Bounded verification ($K=3$, 801.9 ms).

---

## 5. Summary of Deliverables Created

All forensic audit artifacts reside in [`outputs/phase4_forensic/`](file:///outputs/phase4_forensic/):
1. [`artifact_inventory.md`](file:///outputs/phase4_forensic/artifact_inventory.md): Cryptographic hashes and provenance for all Phase 4 files.
2. [`prediction_record_audit.csv`](file:///outputs/phase4_forensic/prediction_record_audit.csv): Condition-by-condition accounting of all 6,000 raw prediction rows.
3. [`recomputed_metrics.json`](file:///outputs/phase4_forensic/recomputed_metrics.json): Independently recomputed metric dictionary.
4. [`metric_reconciliation.md`](file:///outputs/phase4_forensic/metric_reconciliation.md): Comprehensive reconciliation table and discrepancy analysis.
5. [`contradiction_audit.csv`](file:///outputs/phase4_forensic/contradiction_audit.csv): 340 row-level records of detected baseline contradictions.
6. [`statistical_reanalysis.md`](file:///outputs/phase4_forensic/statistical_reanalysis.md): Formal non-inferiority testing and cluster-level paired statistics.
7. [`consensus_and_correctness_audit.md`](file:///outputs/phase4_forensic/consensus_and_correctness_audit.md): Trace of structural agreement vs factual accuracy.
8. [`router_implementation_audit.md`](file:///outputs/phase4_forensic/router_implementation_audit.md): Code path, candidate generation, and adapter isolation trace.
9. [`detector_validation.md`](file:///outputs/phase4_forensic/detector_validation.md): 25-case stress test confusion matrix and narrative discrepancy audit.
10. [`deployment_reanalysis.md`](file:///outputs/phase4_forensic/deployment_reanalysis.md): Analytical latency boundary and serving workload analysis.
11. [`dataset_leakage_audit.md`](file:///outputs/phase4_forensic/dataset_leakage_audit.md): Manifest hash verification and zero-leakage confirmation.
12. [`test_results.md`](file:///outputs/phase4_forensic/test_results.md): Regression suite execution report (87/87 passed).
13. [`final_forensic_report.md`](file:///outputs/phase4_forensic/final_forensic_report.md): This comprehensive final report.
14. [`forensic_summary.json`](file:///outputs/phase4_forensic/forensic_summary.json): Machine-readable summary of all audited findings.
15. [`scripts/audit_phase4_metrics.py`](file:///scripts/audit_phase4_metrics.py): Standalone independent audit evaluation script.
16. [`tests/test_phase4_forensic.py`](file:///tests/test_phase4_forensic.py): 8 focused regression tests for audited behaviors.
17. [`data/phase4_forensic_challenge/diagnostic_cases.jsonl`](file:///data/phase4_forensic_challenge/diagnostic_cases.jsonl): 10 synthetic diagnostic challenge cases.

---

## 6. Exact Reproduction Commands

```powershell
# 1. Activate project environment
.venv\Scripts\Activate.ps1

# 2. Run independent metric evaluation script directly from raw predictions
python scripts/audit_phase4_metrics.py

# 3. Execute full regression test suite (87 tests)
pytest tests/ -v
```

---

## 7. Final Classification Verdict

**Classification**: `E6_bounded` should be categorized as an **Empirically Validated Research Candidate**.
- **Validated**: It robustly eliminates prompt sensitivity and within-cluster contradiction on out-of-distribution questions with a bounded compute budget.
- **Not Production-Ready**: It cannot be certified for safety-critical production deployment until:
  1. The two-sided test fallacy is corrected and factual non-inferiority is tested on an adequately powered sample ($N \ge 1,900$ clusters).
  2. The regex-based disagreement detector is replaced with an NLI model capable of handling mathematical signs, constraints, and multi-slot answers.
  3. Consensus routing is paired with retrieval-augmented generation (RAG) to eliminate the 17.5% blindspot on unanimous model hallucinations.
