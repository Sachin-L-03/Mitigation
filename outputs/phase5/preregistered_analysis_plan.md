# Phase 5 Preregistered Analysis Plan: Factual Reliability, False-Consensus Mitigation, and Live Deployment Validation

**Document Status:** FROZEN PRIOR TO FINAL TEST INFERENCE  
**Repository:** `prompt-sensitivity-hybrid`  
**Branch:** `creme-hybrid-framework`  
**Protocol Version:** 1.0.0  
**Timestamp:** 2026-10-13  
**Authors:** Senior ML Research Engineer, Experimental Statistician & Skeptical Research Auditor  

---

## 1. Executive Summary and Preregistration Scope

This document specifies the preregistered experimental and statistical protocol for **Phase 5** of the prompt-sensitivity hybrid mitigation research project.

Following the forensic audit of Phase 4 (`outputs/phase4_forensic/`), which established that:
1. `E6_bounded` achieved +33.21 percentage points Token F1 consistency improvement over `E0`, but
2. In 17.5% of clusters (35/200), candidate generations exhibited unanimous incorrect consensus (false consensus),
3. Medoid voting in two clusters converted a correct baseline response into an incorrect consensus answer,
4. Factual non-inferiority within a 2-percentage-point margin was NOT established (lower confidence bound was $-3.58$ pp, failing the $-2.00$ pp threshold),
5. The 660.1 ms latency was an analytical estimate rather than a live-serving interactive measurement, and
6. The existing disagreement detector failed on negative numerical signs, entity aliases, and multi-slot constraints.

Phase 5 investigates whether factual reliability and false-consensus robustness can be improved while preserving the high paraphrase consistency of bounded routing, under rigorous out-of-distribution evaluation.

**Crucial Preregistration Commitment:**  
This analysis plan is committed and frozen *before* conducting inference or evaluating the untouched Phase 5 test dataset (`data/phase5_eval/`, test split: 226 clusters, 904 prompts). No thresholds, scoring rules, or hypotheses will be modified after test inspection. Any subsequent adjustments will be explicitly labeled as post-hoc exploratory analyses.

---

## 2. Experimental Conditions

Phase 5 evaluates six distinct conditions:

| Condition Code | Method Name | Description | Inference Inputs |
| :--- | :--- | :--- | :--- |
| **E0** | Base Model | Unmodified `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4). Single generation per prompt. Reference baseline for accuracy, consistency, and latency. | Input prompt only. |
| **E1** | Existing `E6_bounded` | Historical Phase 4 bounded adaptive router ($K=3$, $\tau_{\text{sim}}=0.50$, medoid selection). Evaluated with historical logic. | Input prompt + variants. |
| **E2** | Improved Disagreement Detector | Bounded routing ($K=3$) equipped with enhanced detector for numerical signs, units, dates, entity aliases, polarity, and slot extraction. | Input prompt + variants. |
| **E3** | Factual-Risk-Aware Router | Adaptive router incorporating multi-feature factual risk estimation (detecting suspicious consensus, vague hedging, constraint violations, and slot disagreement). | Input prompt + variants. |
| **E4** | Conservative Abstention Policy | Risk-aware router that emits explicit uncertainty/abstention fallbacks when risk score exceeds validation-tuned threshold $\theta_{\text{risk}}$. | Input prompt + variants. |
| **E5** | Oracle Diagnostic Upper Bound | Offline diagnostic upper bound using gold reference labels to measure theoretical ceiling if factual errors were perfectly identifiable. | Offline analysis only (never live). |

*Note on E6 (External Verifier):* Evaluated as an optional diagnostic condition only if an independent, reproducible knowledge source is available without gold leakage.

---

## 3. Dataset Architecture and Split Integrity

The Phase 5 evaluation dataset (`data/phase5_eval/`) is structured as follows:
- **Total Clusters:** 300 independent semantic clusters across 10 domains.
- **Variants per Cluster:** 4 semantically equivalent prompt formulations (Original, Lexical, Syntactic, Telegraphic/Constraint) = 1,200 prompt records.
- **Hard Negatives:** 30 pairs of semantically non-equivalent prompts to benchmark prompt equivalence discrimination.
- **Domain Distribution:** Science (40), Math (35), History (35), Geography (35), Dates (30), Units (30), Everyday (30), Structured (25), Negation (20), Constraints (20).

### Split Separation
1. **Validation Split ($N_{\text{val}} = 74$ clusters, 296 prompts):**  
   Exclusively used for feature engineering, ablation studies, and risk threshold calibration ($\theta_{\text{risk}}$).
2. **Test Split ($N_{\text{test}} = 226$ clusters, 904 prompts):**  
   Frozen and strictly untouched until all detector parameters and thresholds are locked.

### Cryptographic Dataset Integrity
- `data/phase5_eval/domain_dataset.jsonl`: SHA-256 `197ee41696e1295c0804206f2d4a45ffb15a50990f9cd22d0c7feedd0ae4602f`
- `data/phase5_eval/correctness_subset.jsonl`: SHA-256 `3f2882182da94b198cc34152b522b42d0dc52783a41fa0e0847ddaff85607740`
- `data/phase5_eval/hard_negatives.jsonl`: SHA-256 `1d49e2ae6f90e68a9650413009afbfefffba2820112d9c3feac024160f1f453a`
- **Leakage Audit:** 0 exact or near-match prompt overlaps with 4,084 historical prompts from Phases 1–4.

---

## 4. Primary and Secondary Endpoints

### 4.1 Primary Confirmatory Endpoint
- **Cluster-Level Factual Accuracy ($A_{\text{fact}}$):**  
  Proportion of clusters in the untouched test set where the model's completion satisfies the verified reference answer criteria:
  $$A_{\text{fact}}(M) = \frac{1}{N_{\text{clusters}}} \sum_{i=1}^{N_{\text{clusters}}} y_{i}(M)$$
  where $y_i(M) \in \{0, 1\}$ is the cluster-level factual correctness indicator.

### 4.2 Secondary Endpoints
1. **Paraphrase Consistency (Token F1):** Mean Token F1 across all pairwise prompt variants within each cluster.
2. **Semantic Similarity:** Mean character 3-gram/4-gram Jaccard similarity across variants.
3. **Cluster Agreement Rate:** Proportion of clusters where all $V=4$ variants receive identical strings.
4. **Unanimous False-Consensus Rate:** Proportion of clusters where all candidates agree on a response that is factually incorrect ($y_i = 0$ while variance across candidates is 0).
5. **Contradiction Rates:** Rate of pairwise numerical, negation, and entity contradictions across variants.
6. **Abstention Rate ($\rho_{\text{abstain}}$):** Percentage of test requests where E4 emits a fallback or structured uncertainty response.
7. **Effective Accuracy at Coverage ($A_{\text{cov}}$):** Factual accuracy evaluated exclusively on non-abstained requests.
8. **Constraint and Structured Output Compliance:** Exact schema/format adherence on structured and multi-constraint clusters.
9. **Live-Serving Latency:** End-to-end interactive response latency measured on RTX 4090 using synchronized GPU events (mean, median, p95, p99).
10. **Resource Footprint:** Peak VRAM consumption (GB) and total generated tokens per query.

---

## 5. Statistical Hypothesis Testing Protocol

### 5.1 Independent Analysis Unit
The **semantic question cluster $C_i$** is the sole independent unit of statistical analysis ($N=226$). Paraphrase prompts within a cluster are repeated observations and are aggregated at the cluster level prior to hypothesis testing.

### 5.2 Confirmatory Statistical Tests
1. **Factual Correctness (Binary Paired Differences):**  
   Evaluated using **McNemar’s exact test** for paired binary outcomes and paired cluster differences:
   $$d_i = y_i(\text{Candidate}) - y_i(\text{E0})$$
2. **Consistency Metrics (Continuous Paired Differences):**  
   Evaluated using the two-sided **Wilcoxon signed-rank test** with zero-difference exclusion.
3. **Bootstrap Confidence Intervals:**  
   10,000 cluster-level bootstrap resamples. 95% two-sided confidence intervals computed using the percentile and BCa methods.
4. **Family-Wise Error Rate Control:**  
   The **Holm-Bonferroni step-down procedure** will be applied across all confirmatory paired comparisons against E0 to maintain an overall family-wise error rate of $\alpha = 0.05$.

---

## 6. Factual Non-Inferiority Testing Protocol

### 6.1 Formal Hypotheses
Let $\delta = p_{\text{Candidate}} - p_{\text{E0}}$ denote the difference in true factual accuracy between a candidate method and baseline E0.  
The prespecified non-inferiority margin is $\Delta_{\text{margin}} = -0.02$ ($-2.0$ percentage points).

The formal one-sided statistical test is:
$$H_0: \delta \le -0.02 \quad \text{vs.} \quad H_1: \delta > -0.02$$

### 6.2 Decision Rule
Non-inferiority is declared **established** if and only if:
1. The one-sided p-value satisfies $p_{\text{NI}} < 0.025$ (equivalent to a two-sided $\alpha = 0.05$ framework), and
2. The lower bound of the 95% two-sided (or 97.5% one-sided) confidence interval for $\delta$ strictly exceeds $-0.02$:
   $$\text{CI}_{\text{lower}}(\delta) > -0.020$$

**Explicit Methodological Constraint:**  
A failure to reject the null hypothesis of superiority in a two-sided test ($p > 0.05$) shall NOT be interpreted or reported as evidence of non-inferiority. If $\text{CI}_{\text{lower}}(\delta) \le -0.02$, non-inferiority is formally declared **FAILED**.

---

## 7. Handling of Abstentions and Missing Data

1. **Abstention Scoring in Primary Metric:**  
   In the primary factual accuracy metric ($A_{\text{fact}}$), abstentions are scored as $y_i = 0$. This prevents artificial metric inflation through selective answering.
2. **Coverage-Accuracy Trade-off Reporting:**  
   E4 performance will be documented along the complete coverage-accuracy Pareto curve:
   $$\text{Coverage} = 1 - \rho_{\text{abstain}}, \quad \text{Selective Accuracy} = \frac{\sum_{i \in \text{Covered}} y_i}{N_{\text{covered}}}$$
3. **Missing Data / Generation Failures:**  
   Any timeout, memory error, or unparseable empty generation will be treated as an incorrect prediction ($y_i = 0$). No missing values will be imputed or dropped.

---

## 8. Latency Measurement Boundaries

Live-serving latency must reflect interactive end-to-end production conditions:
1. **Clock Measurement:** `time.perf_counter()` synchronized with `torch.cuda.synchronize()` before and after the execution pipeline.
2. **Pipeline Scope:**
   - Input ingestion and prompt formatting.
   - Initial candidate generation ($K=1$ or $K=3$).
   - Tokenization and KV-cache allocation.
   - Detector execution (slot extraction, constraint validation, risk classification).
   - Selection / medoid computation.
   - Response formatting and emission.
3. **Analytical Latency Prohibition:** Analytical estimates from batched forward passes will not be substituted for live single-request interactive latency.

---

## 9. Criteria for System Selection

To qualify as a superior deployable mitigation over existing E6_bounded:
1. **Factual Non-Inferiority:** Must satisfy $\text{CI}_{\text{lower}}(\delta_{\text{E0}}) > -0.02$.
2. **Paraphrase Consistency:** Must maintain Token F1 $\ge 85.0\%$ across prompt variants.
3. **False-Consensus Reduction:** Must measurably reduce the cluster-level false-consensus rate relative to E1 ($p < 0.05$).
4. **Latency Feasibility:** Mean interactive serving latency must not exceed 1,500 ms on the target RTX 4090 GPU.

*Protocol frozen on 2026-10-13. Ready for validation ablation and module implementation.*
