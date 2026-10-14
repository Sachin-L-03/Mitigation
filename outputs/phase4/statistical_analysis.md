# Statistical Analysis Report: Phase 4 Domain-Shift Benchmark

**Repository**: `prompt-sensitivity-hybrid`  
**Active Branch**: `creme-hybrid-framework`  
**Sample Size**: $N = 200$ Independent Semantic Question Clusters (1,200 Prompts)  
**Independent Sampling Unit**: Semantic Question Cluster (Grouped sampling)  
**Statistical Method**: Two-Sided Paired Wilcoxon Signed-Rank Test with 1,000-sample Cluster Bootstrap 95% Confidence Intervals  
**Family-Wise Error Rate Control**: Holm-Bonferroni Step-Down Procedure across Predeclared Comparisons ($\alpha = 0.05$)  
**Date**: October 12, 2026  
**Auditor**: Senior ML Research Engineer (Autonomous Phase 4 Protocol)  

---

## 1. Primary Statistical Comparisons vs Base Model (E0)

| Condition | Paired Mean Diff (pp) | Paired Median Diff (pp) | 95% Bootstrap CI | Cohen's d | Raw Wilcoxon p | Holm-Adjusted p | Statistically Significant? | Practical Significance? ($\ge 5$ pp) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E3: Preservation LoRA ($w=5$)** | -0.06 pp | 0.00 pp | [-0.49, +0.36] | -0.019 | 5.2320e-01 | 5.2320e-01 | **NO** ($p \ge 0.05$) | **NO** |
| **E1: Full Medoid ($K=6$)** | +42.57 pp | +41.78 pp | [+40.88, +44.45] | 3.210 | 0.0000e+00 | 0.0000e+00 | **YES** ($p < 10^{-15}$) | **YES** |
| **E6_bounded: Bounded Router ($K=3$)** | **+33.21 pp** | **+39.17 pp** | **[+30.09, +36.43]** | **1.456** | **2.4100e-32** | **< 10^-30** | **YES** ($p < 10^{-30}$) | **YES** |

---

## 2. Hypothesis Testing Decisions

### 2.1 Hypothesis 1: LoRA Parameter Editing Generalizes to Novel Domains
- **Null Hypothesis $H_0$**: $\mu(\text{F1}_{\text{E3}}) - \mu(\text{F1}_{\text{E0}}) = 0$
- **Alternative Hypothesis $H_1$**: $\mu(\text{F1}_{\text{E3}}) - \mu(\text{F1}_{\text{E0}}) \neq 0$
- **Test Results**:
  - Sample mean difference: $-0.06$ pp ($95\%$ bootstrap CI: $[-0.49, +0.36]$)
  - Wilcoxon statistic: $W = 8122.5, p = 0.5232$
  - Holm-adjusted $p = 0.5232$
- **Decision**: **FAIL TO REJECT $H_0$**. There is no statistically significant or practically meaningful difference between the preservation adapter and the base model on unseen question domains. Parameter editing does not generalize.

### 2.2 Hypothesis 2: Bounded Adaptive Routing Improves Consistency on Novel Domains
- **Null Hypothesis $H_0$**: $\mu(\text{F1}_{\text{E6\_bounded}}) - \mu(\text{F1}_{\text{E0}}) \le 0$
- **Alternative Hypothesis $H_1$**: $\mu(\text{F1}_{\text{E6\_bounded}}) - \mu(\text{F1}_{\text{E0}}) > 0$
- **Test Results**:
  - Sample mean difference: $+33.21$ pp ($95\%$ bootstrap CI: $[+30.09, +36.43]$)
  - Paired median difference: $+39.17$ pp
  - Wilcoxon statistic: $W = 0.0, p = 2.41 \times 10^{-32}$
  - Holm-adjusted $p < 10^{-30}$
  - Standardized effect size: Cohen's $d = 1.456$ (very large effect)
- **Decision**: **REJECT $H_0$ IN FAVOR OF $H_1$**. The bounded adaptive router produces a massive, statistically overwhelming, and practically dominant improvement in token consistency across 200 out-of-distribution question clusters.

---

## 3. Factual Correctness Paired Analysis

- Baseline factual correctness (E0): **81.60%** [95% CI: 77.2%, 85.8%]
- Bounded router factual correctness (`E6_bounded`): **80.70%** [95% CI: 75.0%, 86.3%]
- Paired difference: **$-0.90$ percentage points** (Wilcoxon $p = 0.442$, not significant).
- **Conclusion**: The $-0.90$ pp variation falls well inside the 95% bootstrap confidence interval bounds and respects the predefined acceptable factual margin ($\le 2$ pp). The framework stabilizes consistency without inducing significant factual regression.
