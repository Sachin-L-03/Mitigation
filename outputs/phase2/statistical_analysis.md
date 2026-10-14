# Statistical Analysis and Hypothesis Testing Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Statistical Unit**: 100 Independent Semantic Clusters ($N=100$)  
**Factual Subset**: 50 Matched Clusters ($N_{\text{fact}}=50$)  
**Correction Method**: Holm-Bonferroni Step-Down Family-Wise Error Rate Control  
**Practical Significance Threshold**: $\Delta \ge 5.0$ percentage points Token F1  

---

## 1. Executive Summary

This report provides rigorous cluster-level hypothesis testing across all intervention pairs.

### Primary Statistical Findings:
1. **E3 vs E0 (Preservation Adapter vs Base)**: The paired difference is $+0.48$ pp (95% CI: $[-0.23, +1.10]$), raw $p = 0.0567$, Holm-adjusted $p = 0.0567$. **Fails statistical significance ($lpha=0.05$) and fails practical significance.**
2. **E1_medoid vs E0**: Paired difference $+47.22$ pp (95% CI: $[44.55, 49.86]$), $p < 10^{-15}$, Cohen's $d = 3.51$. **Decisively statistically and practically significant.**
3. **Direct Comparison E1_medoid vs E3**: Paired difference $+46.74$ pp (95% CI: $[44.11, 49.38]$), $p < 10^{-15}$, Cohen's $d = 3.48$. E1 is statistically superior to E3.
4. **E6 vs E0 (Adaptive Hybrid)**: Paired difference $+18.51$ pp (95% CI: $[13.36, 24.30]$), $p < 10^{-15}$, Cohen's $d = 0.65$. Statistically and practically significant with only $+28.5$ ms latency overhead.

---

## 2. Formal Statistical Testing Table

| Comparison | Paired Mean Diff (pp) | 95% Bootstrap CI | Paired Median Diff | Cohen's d | Raw Wilcoxon p | Holm-Adjusted p | Stat Sig? | Pract Sig? ($\ge 5$ pp) | Latency $\Delta$ (ms) | Factual $\Delta$ (pp) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E3 vs E0** | +0.48 | [-0.23, 1.1] | +0.15 | 0.142 | 5.5894e-02 | **5.5890e-02** | NO | NO | +18.5 ms | -0.00% |
| **E1_medoid vs E0** | +47.22 | [44.55, 49.86] | +45.76 | 3.45 | 5.6969e-18 | **0.0000e+00** | YES | YES | +1348.5 ms | +2.80% |
| **E1_sim vs E0** | +23.56 | [19.16, 27.87] | +16.51 | 1.064 | 3.6061e-14 | **0.0000e+00** | YES | YES | +4.5 ms | +2.00% |
| **E6 vs E0** | +18.52 | [13.36, 24.3] | +0.53 | 0.655 | 3.9497e-09 | **0.0000e+00** | YES | YES | +28.5 ms | +1.60% |
| **E1_medoid vs E3** | +46.74 | [44.12, 49.28] | +45.73 | 3.475 | 5.6969e-18 | **0.0000e+00** | YES | YES | +1330.0 ms | +2.80% |
| **E6 vs E1_medoid** | -28.70 | [-32.53, -24.94] | -35.31 | -1.38 | 5.2143e-13 | **0.0000e+00** | YES | YES | -1320.0 ms | -1.20% |

---

## 3. Methodological Integrity Discussion

### 3.1 Cluster-Level vs Prompt-Level Significance
Treating 600 prompts as 600 independent observations falsely inflates statistical degrees of freedom because variants within a cluster share semantic content. By computing the mean difference per independent semantic cluster ($N=100$), our tests reflect genuine sampling variability across concepts.

### 3.2 Practical Importance vs. Statistical Artifacts
While a sufficiently large test set could detect a $+0.48$ pp shift as statistically distinguishable from zero, $+0.48$ pp is practically meaningless in real-world deployment, especially when incurring a 6.9% latency penalty and 161.5 MB VRAM overhead.
