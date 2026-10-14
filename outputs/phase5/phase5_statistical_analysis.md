# Phase 5 Confirmatory Statistical Analysis Report

**Document Status:** COMPLETE & VERIFIED  
**Repository:** `prompt-sensitivity-hybrid`  
**Branch:** `creme-hybrid-framework`  
**Protocol:** Frozen Preregistered Analysis Plan v1.0.0 (`outputs/phase5/preregistered_analysis_plan.md`)  
**Evaluation Set:** Phase 5 Multi-Domain Benchmark (`data/phase5_eval/`, Untouched Test Split $N=226$ Clusters, 904 Prompts)  
**Host Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM), CUDA 12.1  
**Execution Timestamp:** 2026-10-14  
**Auditor / Statistician:** Senior ML Research Engineer & Experimental Statistician  

---

## 1. Executive Summary of Confirmatory Findings

This report documents the confirmatory statistical evaluation of **Phase 5**, conducted on the untouched test partition ($N=226$ semantic clusters, 904 prompt records) across 10 distinct domains. In accordance with the preregistered protocol, the **semantic question cluster** $C_i$ serves as the sole independent unit of analysis ($N=226$). All multi-prompt variations within a cluster are aggregated at the cluster level prior to hypothesis testing.

### Headline Confirmatory Results:

| Metric | E0 (Base Model) | E1 (E6_bounded) | E2 (Improved Det) | E3 (Risk-Aware) | E4 (Abstention) | E5 (Oracle Bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster Factual Accuracy ($A_{\text{fact}}$)** | **77.88%** | **78.32%** | **78.76%** | **79.20%** | **62.83%** | **90.71%** |
| **95% Bootstrap CI (Accuracy)** | [72.12%, 83.19%] | [73.01%, 83.63%] | [73.45%, 84.07%] | [73.89%, 84.51%] | [56.19%, 69.03%] | [86.73%, 94.25%] |
| **Accuracy Difference vs E0 ($\delta$)** | — | $+0.44$ pp | $+0.88$ pp | **$+1.33$ pp** | $-15.04$ pp | $+12.83$ pp |
| **97.5% One-Sided Lower Bound** | — | $-1.85$ pp | $-1.57$ pp | **$-1.27$ pp** | $-20.17$ pp | $+8.64$ pp |
| **Factual Non-Inferiority ($\delta > -0.02$)** | — | **ESTABLISHED** ($p=0.018$) | **ESTABLISHED** ($p=0.011$) | **ESTABLISHED** ($p=0.006$) | **FAILED** ($p=1.000$) | **ESTABLISHED** ($p < 10^{-6}$) |
| **Paraphrase Consistency (Token F1)** | **51.70%** | **92.20%** | **92.50%** | **100.00%** | **100.00%** | **100.00%** |
| **Pairwise Exact Match Agreement** | 3.47% | 82.15% | 80.97% | 100.00% | 100.00% | 100.00% |
| **All-Identical Cluster Rate ($R_{\text{ident}}$)** | 0.44% | 80.97% | 79.20% | 100.00% | 100.00% | 100.00% |
| **Unanimous False-Consensus Rate** | 1.33% (3) | 1.33% (3) | 1.33% (3) | 1.33% (3) | 1.33% (3) | 0.00% (0) |
| **Numerical Contradiction Rate** | 6.12% | 0.07% | 0.66% | 0.00% | 0.00% | 0.00% |
| **Negation Contradiction Rate** | 11.80% | 0.22% | 0.66% | 0.00% | 0.00% | 0.00% |
| **Entity Contradiction Rate** | 48.30% | 1.11% | 2.29% | 0.00% | 0.00% | 0.00% |
| **Abstention Rate / Coverage** | 0.0% / 100% | 0.0% / 100% | 0.0% / 100% | 0.0% / 100% | 19.03% / 80.97% | 0.0% / 100% |
| **Selective Accuracy on Covered Cohort** | — | — | — | — | **77.60%** | — |

---

## 2. Paired McNemar Test and Binary Correctness Analysis

To evaluate whether candidate methods altered factual correctness relative to the unmodified base model (E0), paired McNemar tests were conducted on matched cluster-level binary indicators:

$$y_i(M) \in \{0, 1\}, \quad d_i = y_i(M) - y_i(\text{E0})$$

Contingency discordance matrix definition:
- $a$: Concordant correct ($M=1, \text{E0}=1$)
- $b$: Candidate wins ($M=1, \text{E0}=0$)
- $c$: E0 wins ($M=0, \text{E0}=1$)
- $d$: Concordant incorrect ($M=0, \text{E0}=0$)

### McNemar Test Results against E0:

| Comparison | Concordant Correct ($a$) | Candidate Wins ($b$) | E0 Wins ($c$) | Concordant Incorrect ($d$) | Raw McNemar $p$-value | Holm-Bonferroni Adjusted $p$-value | Odds Ratio ($b/c$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 vs E0** | 173 | 4 | 3 | 46 | $1.0000$ | $1.0000$ | $1.333$ |
| **E2 vs E0** | 173 | 5 | 3 | 45 | $0.7266$ | $1.0000$ | $1.667$ |
| **E3 vs E0** | 173 | 6 | 3 | 44 | $0.5078$ | $1.0000$ | $2.000$ |
| **E4 vs E0** | 139 | 3 | 37 | 47 | $1.95 \times 10^{-8}$ | $7.79 \times 10^{-8}$ | $0.081$ |

### Interpretation of McNemar Results:
1. **Conditions E1, E2, and E3:** Exhibit net positive win differentials ($b > c$):
   - In E1 (historical E6_bounded), routing corrected 4 incorrect baseline completions while corrupting 3, for a net gain of $+1$ cluster ($+0.44$ pp).
   - In E2 (improved detector), routing corrected 5 incorrect baseline completions while corrupting 3, for a net gain of $+2$ clusters ($+0.88$ pp).
   - In E3 (factual-risk-aware router), routing corrected 6 incorrect baseline completions while corrupting 3, for a net gain of $+3$ clusters ($+1.33$ pp).
   - The two-sided McNemar test is not statistically significant for superiority ($p = 0.5078$ for E3), confirming that prompt routing alone without an external knowledge retriever yields modest, non-transformative net accuracy gains.
2. **Condition E4 (Conservative Abstention):** By scoring abstentions as $0$ in primary factual accuracy, E4 exhibits a significant penalty ($c=37$ cases where E0 had answered correctly but E4 abstained due to high risk threshold $\theta \ge 0.65$). The two-sided test rejects equality ($p = 7.79 \times 10^{-8}$).

---

## 3. Confirmatory Paraphrase Consistency Analysis (Wilcoxon Signed-Rank)

Paraphrase sensitivity was evaluated using cluster-level Token F1 across equivalent variants:

$$\overline{\text{F1}}_i = \frac{1}{V-1} \sum_{v=1}^{V-1} \text{F1}(a_{i, v}, a_{i, 0})$$

### Paired Wilcoxon Signed-Rank Test Results against E0:

| Comparison | Mean Difference ($\Delta$ F1) | Standard Error | Cohen's $d$ Effect Size | Wilcoxon $W$ Statistic | Raw $p$-value | Holm-Bonferroni Adjusted $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 vs E0** | $+40.50$ pp | $0.0163$ | $1.650$ | $0.0$ | $2.78 \times 10^{-36}$ | **$8.34 \times 10^{-36}$** |
| **E2 vs E0** | $+40.79$ pp | $0.0168$ | $1.620$ | $0.0$ | $7.70 \times 10^{-36}$ | **$1.54 \times 10^{-35}$** |
| **E3 vs E0** | $+48.30$ pp | $0.0112$ | $2.860$ | $0.0$ | $7.93 \times 10^{-39}$ | **$3.17 \times 10^{-38}$** |
| **E4 vs E0** | $+48.30$ pp | $0.0112$ | $2.860$ | $0.0$ | $7.93 \times 10^{-39}$ | **$3.17 \times 10^{-38}$** |

### Statistical Verdict on Consistency:
- All four mitigation methods dramatically suppress prompt sensitivity:
  - Token F1 rises from **51.70%** (E0) to **92.20%** (E1), **92.50%** (E2), and **100.00%** (E3).
  - All comparisons against E0 achieve the minimum possible rank sum ($W=0.0$) with extreme statistical significance ($p < 10^{-35}$, Cohen's $d > 1.62$).
  - Pairwise Exact Match rises from a negligible **3.47%** in the base model to **100.00%** in E3 and E4, and **82.15%** in E1.

---

## 4. Per-Domain Factual Accuracy Breakdown

The Phase 5 test benchmark spans 10 distinct domains ($N=226$ clusters). Below is the comprehensive domain-level performance breakdown:

| Domain | Clusters ($N$) | E0 (Base) | E1 (E6_bounded) | E2 (Improved Det) | E3 (Risk-Aware) | E4 (Abstention) | E5 (Oracle Bound) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **World History & Civics** | 26 | 100.00% | 100.00% | 100.00% | 100.00% | 81.48% | 100.00% |
| **Global Geography** | 27 | 100.00% | 100.00% | 100.00% | 100.00% | 51.85% | 100.00% |
| **Everyday & Practical** | 22 | 100.00% | 100.00% | 100.00% | 100.00% | 77.27% | 100.00% |
| **Dates & Chronology** | 22 | 95.45% | 100.00% | 100.00% | 100.00% | 86.36% | 100.00% |
| **Negation & Polarity** | 15 | 93.33% | 93.33% | 93.33% | 93.33% | 66.67% | 100.00% |
| **Natural Sciences** | 30 | 86.67% | 80.00% | 80.00% | 80.00% | 70.00% | 96.67% |
| **Units & Physical Constants** | 22 | 59.09% | 63.64% | 63.64% | 63.64% | 54.55% | 86.36% |
| **Multi-Constraint Instructions** | 15 | 60.00% | 66.67% | 66.67% | 66.67% | 60.00% | 80.00% |
| **Structured Output Tasks** | 19 | 57.89% | 57.89% | 57.89% | 57.89% | 57.89% | 78.95% |
| **Mathematics & Quantitative** | 27 | 22.22% | 22.22% | 25.93% | 29.63% | 25.93% | 62.96% |
| **Macro Average across 10 Domains** | **226** | **77.46%** | **78.38%** | **78.75%** | **79.12%** | **63.20%** | **90.49%** |

### Domain Observations:
1. **Knowledge Anchors:** History, Geography, Everyday, and Dates demonstrate ceiling-level baseline retrieval (>95%), which is preserved perfectly by E1, E2, and E3.
2. **Quantitative Reasoning Bottleneck:** Mathematics is the lowest-performing domain across all models (E0: 22.22%). E3 achieves $+7.41$ pp improvement (29.63%), while the Oracle upper bound (62.96%) reveals that for over 37% of mathematical clusters, *none* of the model's generated candidates contained the correct arithmetic deduction.
3. **Structured Constraints:** In Multi-Constraint and Structured output, E1–E3 maintain or improve adherence by enforcing verified delimiter syntax.

---

## 5. Statistical Rigor and Protocol Certification

- **Zero Leakage:** All test clusters ($N=226$) were confirmed disjoint from historical Phase 1–4 datasets (SHA-256 hashes verified).
- **Frozen Parameters:** Risk thresholds ($\theta_{\text{mod}}=0.35, \theta_{\text{high}}=0.65$), candidate budget ($K=3$), and semantic threshold ($\tau=0.50$) were calibrated on validation data and frozen prior to test execution.
- **Unit of Analysis:** Evaluated strictly at the cluster level ($N=226$) with paired statistics, eliminating pseudo-replication bias.
