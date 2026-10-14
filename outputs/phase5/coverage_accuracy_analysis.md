# Phase 5 Coverage vs. Accuracy Trade-Off Analysis

**Document Status:** COMPLETE & VERIFIED  
**Repository:** `prompt-sensitivity-hybrid`  
**Protocol Reference:** `outputs/phase5/preregistered_analysis_plan.md` (Section 7) & RQ3  
**Evaluation Set:** Phase 5 Benchmark (Validation $N=74$, Test $N=226$)  
**Author:** ML Research Engineer & Decision-Theoretic Statistician  
**Date:** 2026-10-14  

---

## 1. Executive Summary

This report evaluates the selective prediction and abstention dynamics of **Condition E4** (`ConservativeAbstentionPolicy`). In practical AI deployments, providing a structured uncertainty signal or admitting "I am uncertain" is far preferable to confidently generating a false answer. However, in benchmark evaluation where abstentions are scored as $0.0$, abstaining inherently depresses the gross factual accuracy metric.

### Headline Findings:
1. **Validation Threshold Sweep Calibration:**
   - As the risk threshold $\theta$ varies from $0.20$ to $0.80$, coverage scales monotonically from $17.57\%$ to $94.59\%$.
   - At high selectivity ($\theta = 0.20$), selective accuracy reaches **92.31%**, demonstrating that low-risk scores are exceptionally strong predictors of factual correctness.
   - At the frozen threshold $\theta_{\text{high}} = 0.65$, validation coverage is **79.73%** with selective accuracy of **74.58%**.
2. **Confirmatory Test Set Performance ($\theta_{\text{high}} = 0.65$):**
   - **Abstention Rate ($\rho_{\text{abstain}}$):** **19.03%** (43 of 226 test clusters).
   - **Coverage ($\text{Cov}$):** **80.97%** (183 of 226 test clusters answered).
   - **Selective Factual Accuracy ($A_{\text{selective}}$):** **77.60%** (142 of 183 answered clusters correct).
   - **Gross Factual Accuracy ($A_{\text{fact}}$, abstention = 0):** **62.83%** (142 of 226 total clusters).
3. **Operational Implication:**
   - Scoring abstentions as zero imposes a $-15.04$ percentage point penalty on the gross accuracy metric relative to baseline E0 (77.88%).
   - E4 is **not** designed to maximize raw benchmark accuracy; it is designed for **safety-critical deployments** where emitting a false answer has asymmetric negative utility.

---

## 2. Validation Threshold Sweep and Pareto Frontier

On the validation dataset ($N_{\text{val}} = 74$), the risk threshold $\theta \in [0.20, 0.80]$ was swept to map the complete Pareto frontier between operational coverage and selective accuracy:

| Risk Threshold ($\theta$) | Operational Coverage | Selective Factual Accuracy | Effective Gross Accuracy ($A_{\text{eff}}$) | Qualitative Regime |
| :---: | :---: | :---: | :---: | :--- |
| **0.20** | **17.57%** (13/74) | **92.31%** (12/13) | 16.22% | Ultra-conservative / High Precision |
| **0.30** | 20.27% (15/74) | 86.67% (13/15) | 17.57% | Strict Verification Gate |
| **0.35** | 20.27% (15/74) | 86.67% (13/15) | 17.57% | **Frozen $\theta_{\text{moderate}}$ (Verification Threshold)** |
| **0.40** | 21.62% (16/74) | 87.50% (14/16) | 18.92% | Conservative Operation |
| **0.50** | 48.65% (36/74) | 80.56% (29/36) | 39.19% | Balanced Utility |
| **0.60** | 77.03% (57/74) | 73.68% (42/57) | 56.76% | High Coverage |
| **0.65** | **79.73%** (59/74) | **74.58%** (44/59) | **59.46%** | **Frozen $\theta_{\text{high}}$ (Abstention Threshold)** |
| **0.70** | 83.78% (62/74) | 75.81% (47/62) | 63.51% | Permissive Gate |
| **0.80** | 94.59% (70/74) | 77.14% (54/70) | 72.97% | Near-Complete Coverage |

### Mathematical Definitions:
$$\text{Coverage}(\theta) = \frac{N_{\text{answered}}(\theta)}{N_{\text{total}}}, \quad A_{\text{selective}}(\theta) = \frac{\sum_{i \in \text{Answered}} y_i}{N_{\text{answered}}(\theta)}, \quad A_{\text{eff}}(\theta) = \frac{\sum_{i \in \text{Answered}} y_i}{N_{\text{total}}}$$

---

## 3. Confirmatory Test Set Selective Performance (E4)

On the untouched Phase 5 test dataset ($N=226$), with frozen threshold $\theta_{\text{high}} = 0.65$:

```
Total Test Cohort: N = 226 clusters
   ├── Answered Cohort (Coverage = 80.97%, N = 183)
   │     ├── Factually Correct: 142 clusters (Selective Accuracy = 77.60%)
   │     └── Factually Incorrect: 41 clusters (22.40%)
   └── Abstained Cohort (Abstention Rate = 19.03%, N = 43)
         ├── Scored as 0.0 in primary benchmark accuracy
         └── Emitted machine-readable structured uncertainty fallback
```

### Risk Category Breakdown on Test Cohort:
- **Low Risk ($\text{Risk} < 0.35$):** 163 clusters (72.12%) $\to$ Accuracy: **79.75%** (130/163).
- **Moderate Risk ($0.35 \le \text{Risk} < 0.65$):** 20 clusters (8.85%) $\to$ Accuracy: **60.00%** (12/20).
- **High Risk ($\text{Risk} \ge 0.65$):** 43 clusters (19.03%) $\to$ Routed to **Abstention Fallback** (Scored 0.0).

The monotonic degradation in accuracy across risk categories ($79.75\% \to 60.00\% \to \text{Abstained}$) confirms that `FactualRiskDetector` successfully orders clusters by factual reliability.

---

## 4. Cost-Utility Decision Matrix

In high-stakes applications (e.g. medical informatics, legal discovery, technical engineering), the cost of an error is strictly greater than the cost of an abstention:
$$U = C_{\text{correct}} \cdot N_{\text{correct}} - C_{\text{error}} \cdot N_{\text{error}} - C_{\text{abstain}} \cdot N_{\text{abstain}}$$

Assuming normalized costs: $C_{\text{correct}} = +1.0$, $C_{\text{abstain}} = -0.1$, and varying the penalty of false answers $C_{\text{error}}$:

| Error Penalty ($C_{\text{error}}$) | Utility of E0 (Base Model) | Utility of E3 (Risk-Aware Router) | Utility of E4 (Conservative Abstention) | Optimal Policy |
| :---: | :---: | :---: | :---: | :---: |
| **$-1.0$ (Symmetric Utility)** | $+126.0$ | **$+132.0$** | $+96.7$ | **E3 Wins** |
| **$-2.0$ (Double Penalty)** | $+76.0$ | **$+85.0$** | $+55.7$ | **E3 Wins** |
| **$-3.0$ (Triple Penalty)** | $+26.0$ | **$+38.0$** | $+14.7$ | **E3 Wins** |
| **$-5.0$ (High Stakes)** | $-74.0$ | $-56.0$ | **$-67.3$** | **E4 Outperforms E0** |
| **$-10.0$ (Safety Critical)** | $-324.0$ | $-291.0$ | **$-272.3$** | **E4 Wins** |

### Policy Recommendation:
- **General Purpose & Benchmarks:** Deploy **Condition E3** (Full Coverage, Highest Accuracy 79.20%, Zero Contradictions, 100% Token F1).
- **Safety-Critical / Regulated Tasks:** Deploy **Condition E4** with $\theta_{\text{high}} = 0.65$ (80.97% Coverage, structured fallback codes emitted for upstream human review).
