# Phase 4 Statistical Reanalysis & Factual Non-Inferiority Audit

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Sample Size**: $N = 200$ Independent Semantic Question Clusters (1,200 Prompts)  
**Independent Evaluation Unit**: Semantic Question Cluster  
**Auditor**: ML Research Auditor & Experimental Statistician  
**Date**: October 13, 2026  

---

## 1. Experimental Design & Independent Unit Audit

### 1.1 The Unit-of-Analysis Problem
In multi-prompt sensitivity evaluation, an LLM is queried with multiple paraphrases of the same underlying semantic question. In the Phase 4 domain-shift benchmark, each of the $N = 200$ question clusters contains exactly 6 prompt variants (1 canonical prompt + 5 paraphrases), yielding 1,200 total prompts.

**Auditor Verification of Sampling Unit**:
- If each prompt ($N=1,200$) or each prompt pair ($N=3,000$) were treated as an independent observation, degrees of freedom would be artificially inflated by $6\times$ to $15\times$, causing catastrophic Type I error inflation and spuriously tiny p-values.
- **Audit Finding**: In [`scripts/run_phase4_domain_evaluation.py:358-410`](file:///scripts/run_phase4_domain_evaluation.py), the evaluation pipeline correctly averaged metric scores across the 5 variant comparisons within each cluster first, producing exactly **$N = 200$ paired cluster-level observations**.
- The independent unit utilized in the statistical tests is therefore **defensible and correct** ($N = 200$).

---

## 2. Token Consistency Statistical Verification (E6_bounded vs E0)

### 2.1 Cluster-Level Distribution Characteristics
- Independent observations: $N = 200$ clusters.
- Baseline (E0) Cluster Mean Token F1: **57.43%** ($\text{SD} = 12.87\%$, $\text{Median} = 56.67\%$).
- Bounded Router (E6_bounded) Cluster Mean Token F1: **90.64%** ($\text{SD} = 18.23\%$, $\text{Median} = 100.00\%$).
- Paired cluster differences: $\Delta_i = \text{F1}_{i, \text{E6}} - \text{F1}_{i, \text{E0}}$ for $i \in \{1, \dots, 200\}$.

### 2.2 Paired Test Statistics
- **Sample Mean Paired Difference**: **$+33.21$ percentage points** ($+0.3321$).
- **Sample Median Paired Difference**: **$+39.17$ percentage points** ($+0.3917$).
- **Standard Deviation of Paired Difference**: $\text{SD}_{\Delta} = 22.81$ pp ($0.2281$).
- **Standard Error of Mean Difference**: $\text{SE}_{\Delta} = \frac{0.2281}{\sqrt{200}} = 0.01613$ (1.61 pp).
- **95% Bootstrap Confidence Interval (1,000 Cluster Resamples)**: **[+30.09 pp, +36.43 pp]**.
- **Parametric Paired t-test**: $t = 20.59$, $\text{df} = 199$, $p = 1.15 \times 10^{-52}$.
- **Non-Parametric Wilcoxon Signed-Rank Test**:
  - In 147 clusters, $\text{F1}_{\text{E6}} > \text{F1}_{\text{E0}}$ ($\Delta_i > 0$).
  - In 53 clusters, $\text{F1}_{\text{E6}} = \text{F1}_{\text{E0}}$ ($\Delta_i = 0$, fast path routed).
  - In 0 clusters, $\text{F1}_{\text{E6}} < \text{F1}_{\text{E0}}$ ($\Delta_i < 0$).
  - Because there are zero negative differences, the sum of negative ranks is $W = 0.0$.
  - Exact Wilcoxon p-value: **$p = 7.15 \times 10^{-26}$** (asymptotically reported as $2.41 \times 10^{-32}$).
  - Holm-Bonferroni adjusted p-value across the 3 predeclared baseline comparisons: **$p < 10^{-25}$**.
- **Standardized Effect Size**:
  $$\text{Cohen's } d = \frac{\overline{\Delta}}{\text{SD}_{\Delta}} = \frac{0.3321}{0.2281} = \mathbf{1.4558}$$
  (Classified as an exceptionally large effect size, $d > 0.80$).

**Verdict**: The $+33.21$ pp gain in Token F1 achieved by `E6_bounded` over baseline `E0` is mathematically verified, statistically overwhelming, and practically dominant.

---

## 3. Preservation LoRA (E3) Statistical Reanalysis

### 3.1 Parameter Editing Generalization Test
- Baseline E0 Mean F1: **57.43%**.
- Preservation Adapter E3 ($w_{\text{pres}}=5.0$) Mean F1: **57.37%**.
- Paired Mean Difference: **$-0.06$ pp** ($-0.0006$).
- 95% Bootstrap Confidence Interval: **[-0.49 pp, +0.36 pp]**.
- Wilcoxon Signed-Rank Statistic: $W = 8122.5$, $p = 0.5232$.
- Holm-Adjusted p-value: $p = 0.5232$.
- Cohen's $d = -0.019$.

**Verdict**: Fails to reject the null hypothesis of zero difference ($p = 0.5232$). Parameter editing via preservation-tuned LoRA provides zero measurable robustness gain across out-of-distribution question domains.

---

## 4. Factual Non-Inferiority Reanalysis (CRITICAL AUDIT FINDING)

### 4.1 Fallacy in the Original Phase 4 Report
In Section 3 of `outputs/phase4/statistical_analysis.md`, the author reported:
> "Baseline factual correctness (E0): 81.60%  
> Bounded router factual correctness (E6_bounded): 80.70%  
> Paired difference: -0.90 percentage points (Wilcoxon p = 0.442, not significant).  
> Conclusion: The -0.90 pp variation falls well inside the 95% bootstrap confidence interval bounds and respects the predefined acceptable factual margin (<= 2 pp)."

**Auditor Forensic Critique**:
This conclusion commits a fundamental, textbook error in experimental statistics: **conflating "absence of evidence of a difference" with "evidence of non-inferiority."**

A non-significant two-sided test ($p = 0.442$) tests the null hypothesis $H_0: \mu_{\text{E6}} = \mu_{\text{E0}}$ against the alternative $H_1: \mu_{\text{E6}} \neq \mu_{\text{E0}}$. Failing to reject $H_0$ means only that the study cannot detect a difference from zero; it DOES NOT demonstrate that the degradation is smaller than 2 percentage points.

To demonstrate non-inferiority within a prespecified margin $\delta_{\text{margin}} = -0.02$ ($-2$ percentage points) at $\alpha = 0.025$ (one-sided), one must perform a formal one-sided non-inferiority test:
- **Null Hypothesis $H_0$ (Inferior)**: $\delta \le -0.02$ (the method degrades factual accuracy by 2 pp or more).
- **Alternative Hypothesis $H_1$ (Non-Inferior)**: $\delta > -0.02$ (the degradation, if any, is strictly less than 2 pp).

For non-inferiority to be established at one-sided $\alpha = 0.025$, the **lower bound of the two-sided 95% confidence interval** (equivalent to the 97.5% one-sided lower confidence bound) for the paired difference $\delta$ must be **strictly greater than $-0.02$** ($-2.00$ pp).

### 4.2 Formal Independent Non-Inferiority Test Calculations

#### 1. Contextual Factual Accuracy (Strict Slot & Conflict Scoring)
- $N = 200$ clusters.
- Baseline E0 Mean: **77.20%** ($0.7720$).
- Bounded Router E6_bounded Mean: **76.40%** ($0.7640$).
- Paired Difference: $\delta = \text{Mean}(\text{E6}) - \text{Mean}(\text{E0}) = -0.0080$ ($-0.80$ pp).
- Standard Deviation of Paired Difference: $\text{SD}_{\delta} = 0.1993$ (19.93 pp).
- Standard Error: $\text{SE}_{\delta} = \frac{0.1993}{\sqrt{200}} = 0.01409$ (1.409 pp).
- Two-sided 95% Confidence Interval for $\delta$:
  $$\delta \pm t_{0.975, 199} \times \text{SE}_{\delta} = -0.0080 \pm 1.9719 \times 0.01409 = [-0.0358, +0.0198] \implies \mathbf{[-3.58\text{ pp}, +1.98\text{ pp}]}$$
- 97.5% One-Sided Lower Confidence Bound: **$-3.58$ pp** ($-0.0358$).
- Non-Inferiority Test Statistic:
  $$t_{\text{NI}} = \frac{\delta - \delta_{\text{margin}}}{\text{SE}_{\delta}} = \frac{-0.0080 - (-0.0200)}{0.01409} = \frac{+0.0120}{0.01409} = \mathbf{+0.8514}$$
- One-Sided Non-Inferiority p-value:
  $$p_{\text{NI}} = 1 - \Phi_{t_{199}}(0.8514) = \mathbf{0.1978}$$
- Threshold: $\alpha = 0.025$.
- **Statistical Decision**:
  - Is $p_{\text{NI}} < 0.025$? **NO** ($0.1978 \gg 0.025$).
  - Is the lower confidence bound ($ -3.58$ pp) $> -2.00$ pp? **NO** ($-3.58\text{ pp} < -2.00\text{ pp}$).
  - **Result: NON-INFERIORITY IS NOT ESTABLISHED.**

#### 2. Concept Retrieval Accuracy (Keyword/Entity Substring Scoring)
- $N = 200$ clusters.
- Baseline E0 Mean: **81.60%** ($0.8160$).
- Bounded Router E6_bounded Mean: **80.70%** ($0.8070$).
- Paired Difference: $\delta = -0.0090$ ($-0.90$ pp).
- Standard Deviation of Paired Difference: $\text{SD}_{\delta} = 0.1584$ (15.84 pp).
- Standard Error: $\text{SE}_{\delta} = \frac{0.1584}{\sqrt{200}} = 0.01120$ (1.120 pp).
- Two-sided 95% Confidence Interval for $\delta$:
  $$\delta \pm 1.9719 \times \text{SE}_{\delta} = -0.0090 \pm 1.9719 \times 0.01120 = [-0.0311, +0.0131] \implies \mathbf{[-3.11\text{ pp}, +1.31\text{ pp}]}$$
- 97.5% One-Sided Lower Confidence Bound: **$-3.11$ pp** ($-0.0311$).
- Non-Inferiority Test Statistic:
  $$t_{\text{NI}} = \frac{-0.0090 - (-0.0200)}{0.01120} = \frac{+0.0110}{0.01120} = \mathbf{+0.9830}$$
- One-Sided Non-Inferiority p-value:
  $$p_{\text{NI}} = 1 - \Phi_{t_{199}}(0.9830) = \mathbf{0.1634}$$
- Threshold: $\alpha = 0.025$.
- **Statistical Decision**:
  - Is $p_{\text{NI}} < 0.025$? **NO** ($0.1634 \gg 0.025$).
  - Is the lower confidence bound ($-3.11$ pp) $> -2.00$ pp? **NO** ($-3.11\text{ pp} < -2.00\text{ pp}$).
  - **Result: NON-INFERIORITY IS NOT ESTABLISHED.**

---

## 5. Non-Inferiority Audit Summary Table

| Metric | E0 Mean | E6_bounded Mean | Mean Diff ($\delta$) | 95% 2-Sided CI | 97.5% 1-Sided Lower Bound | Non-Inferiority Margin | $t_{\text{NI}}$ | 1-Sided $p_{\text{NI}}$ ($\alpha=0.025$) | Non-Inferiority Established? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Factual Accuracy (Strict)** | 77.20% | 76.40% | -0.80 pp | [-3.58 pp, +1.98 pp] | -3.58 pp | -2.00 pp | +0.8514 | 0.1978 | **NO** ($p \ge 0.025$) |
| **Concept Retrieval** | 81.60% | 80.70% | -0.90 pp | [-3.11 pp, +1.31 pp] | -3.11 pp | -2.00 pp | +0.9830 | 0.1634 | **NO** ($p \ge 0.025$) |

---

## 6. Methodological Conclusions and Recommendations

1. **Why the Claim Failed**:
   - The sample size of $N=200$ clusters (with paired difference $\text{SD} \approx 16\text{--}20$ pp) is statistically underpowered to reject a non-inferiority margin of 2 percentage points.
   - To have 80% statistical power to establish non-inferiority within 2.0 pp when the true degradation is 0.85 pp and $\text{SD} = 18$ pp, an experiment requires at least:
     $$N \ge \frac{(z_{0.975} + z_{0.80})^2 \times \sigma^2}{(\delta - \delta_{\text{margin}})^2} = \frac{(1.96 + 0.84)^2 \times (0.18)^2}{(0.02 - 0.0085)^2} \approx 1,920 \text{ clusters}$$
   - With only $N=200$ clusters, the confidence interval is $\pm 2.2\text{--}2.8$ pp wide, so any point estimate near $-0.8$ pp will have a lower bound extending beyond $-3.0$ pp.
2. **Correct Research Language**:
   - Researchers must **NOT** state that "E6_bounded is proven non-inferior within 2 percentage points."
   - The scientifically truthful statement is:
     > *"In this sample of 200 clusters, factual accuracy decreased by 0.80 percentage points (from 77.20% to 76.40%). However, due to sample variance, the 95% confidence interval spans [-3.58 pp, +1.98 pp]. Formal one-sided non-inferiority testing at the 2-percentage-point margin fails to reach significance ($p = 0.198$). The data cannot rule out a factual degradation as large as 3.58 percentage points."*
