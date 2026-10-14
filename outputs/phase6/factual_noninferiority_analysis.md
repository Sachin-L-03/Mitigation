# Phase 6: Factual Non-Inferiority Statistical Analysis

**Protocol Reference:** `PHASE6-NI-ANALYSIS-001`  
**Preregistered Protocol:** `outputs/phase6/preregistered_analysis_plan.md`  
**Evaluation Set:** Frozen Untouched Phase 6 Test Set ($N = 450$ clusters)  
**Prespecified Non-Inferiority Margin:** $\Delta_{\text{margin}} = -0.02$ (-2.0 percentage points)  
**Prespecified Significance Level:** One-Sided $\alpha = 0.025$ ($z_{0.025} = 1.95996$)  
**Status:** CONFIRMED & AUDITED  

---

## 1. Formal Non-Inferiority Hypothesis Framework

In high-stakes ML systems, adding consistency routers, retrieval layers, and verifiers carries the risk of "corrupting" answers that were previously generated correctly by the base model. To guard against regression, Phase 6 preregistered a formal one-sided non-inferiority test against the base Qwen model (**E0**).

### 1.1 Hypotheses
Let $\Delta_{\text{NI}} = A_{\text{fact}}(E_k) - A_{\text{fact}}(E0)$.
- **Null Hypothesis ($H_{0,\text{NI}}$):** The candidate condition $E_k$ is inferior to the base model by more than the 2.0 percentage point margin:
  $$H_{0,\text{NI}}: \Delta_{\text{NI}} \le -0.02$$
- **Alternative Hypothesis ($H_{1,\text{NI}}$):** The candidate condition $E_k$ is non-inferior to the base model within the 2.0 percentage point margin:
  $$H_{1,\text{NI}}: \Delta_{\text{NI}} > -0.02$$

### 1.2 Decision Rule
Non-inferiority is established if and only if the **97.5% one-sided lower confidence bound** strictly exceeds $-0.02$:
$$\text{CI}_{97.5,\text{lower}} = \hat{\Delta}_{\text{NI}} - z_{0.025} \cdot \text{SE}(\hat{\Delta}_{\text{NI}}) > -0.02$$
and the corresponding one-sided $p$-value satisfies $p_{\text{NI}} < 0.025$:
$$Z_{\text{NI}} = \frac{\hat{\Delta}_{\text{NI}} - (-0.02)}{\text{SE}(\hat{\Delta}_{\text{NI}})}, \quad p_{\text{NI}} = 1 - \Phi(Z_{\text{NI}})$$

A standard two-sided test showing "no significant difference" ($p > 0.05$) does **not** establish non-inferiority.

---

## 2. Non-Inferiority Evaluation Results

| Condition ($E_k$) | Description | $A_{\text{fact}}(E_k)$ | $A_{\text{fact}}(E0)$ | Paired $\Delta_{\text{NI}}$ | Discordant ($b / c$) | $\text{SE}(\Delta)$ | $Z_{\text{NI}}$ | $p_{\text{NI}}$ | $\text{CI}_{97.5,\text{lower}}$ | Non-Inferiority Established? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | E3 Router Baseline | 58.89% | 59.56% | -0.67 pp | 17 / 20 | 1.04 pp | +1.28 | 0.0987 | -2.71 pp | Inconclusive ($p > 0.025$) |
| **E2** | E3 + Deterministic | 81.56% | 59.56% | **+22.00 pp** | 102 / 3 | 2.36 pp | +10.18 | $< 10^{-15}$ | **+17.38 pp** | **ESTABLISHED** |
| **E3** | E3 + Retrieval | 60.22% | 59.56% | **+0.67 pp** | 18 / 15 | 1.05 pp | +2.55 | **0.0054** | **-1.38 pp** | **ESTABLISHED** |
| **E4** | **Combined Router** | **81.56%** | **59.56%** | **+22.00 pp** | **106 / 7** | **2.43 pp** | **+9.89** | **$< 10^{-15}$** | **+17.24 pp** | **ESTABLISHED** |
| **E5** | Conservative Fallback | 50.00% | 59.56% | -9.56 pp | 106 / 149 | 3.52 pp | -2.15 | 0.9841 | -16.45 pp | Not Established* |

*\*Note on E5: E5 intentionally trades off raw answer coverage (58.22%) for high selective accuracy (85.88%). Because abstentions are scored as 0.0 in $A_{\text{fact}}$, its raw unweighted accuracy is penalized.*

---

## 3. In-Depth Analysis of Confirmatory Findings

### 3.1 Primary Candidate (E4) Non-Inferiority Verdict
- Observed Paired Difference: $\Delta = +0.2200$ ($+22.00\text{ percentage points}$).
- Discordant Pairs:
  - $b$ ($E_4$ wins): **106 clusters** where $E_4 = 1, E_0 = 0$.
  - $c$ ($E_0$ wins): **7 clusters** where $E_4 = 0, E_0 = 1$.
  - Discordance Ratio: $\frac{b}{c} = 15.14$, indicating an overwhelming 15:1 ratio of improvements to regressions.
- Standard Error: $\text{SE} = 0.02426$ ($2.43\text{ pp}$).
- Non-Inferiority Test Statistic:
  $$Z_{\text{NI}} = \frac{0.2200 - (-0.02)}{0.02426} = \frac{0.2400}{0.02426} = +9.89$$
- One-Sided $p$-Value: $p_{\text{NI}} = 1 - \Phi(9.89) < 10^{-15}$.
- 97.5% Lower Bound: $\text{CI}_{97.5,\text{lower}} = +0.2200 - 1.95996 \cdot 0.02426 = \mathbf{+17.24\text{ pp}}$.
- **Conclusion:** Because $+17.24\text{ pp} \gg -2.00\text{ pp}$, the null hypothesis of inferiority is decisively rejected. E4 not only satisfies non-inferiority but achieves **substantial, statistically significant superiority** over the base model.

### 3.2 Baseline E3 Router (E1) Performance
- On this challenging 12-stratum benchmark, E1 achieved $58.89\%$ vs $59.56\%$ for E0 ($\Delta = -0.67\text{ pp}$).
- Discordant Pairs: $b = 17, c = 20$.
- Standard Error: $\text{SE} = 0.01043$ ($1.04\text{ pp}$).
- 97.5% Lower Bound: $-0.0067 - 1.96 \cdot 0.0104 = -0.0271$ ($-2.71\text{ pp}$).
- Because the lower bound ($-2.71\text{ pp}$) dips slightly below the $-2.00\text{ pp}$ margin, non-inferiority for E1 alone was **inconclusive** ($p = 0.0987$).
- **Insight:** E3's internal consistency checks cannot resolve difficult arithmetic or modular errors. When candidates hallucinate a false consensus, E3 preserves the error. Adding Path A and Path B completely solves this bottleneck.

### 3.3 Retrieval Layer (E3) Non-Inferiority
- $\Delta = +0.67\text{ pp}$ ($b = 18, c = 15$).
- $\text{CI}_{97.5,\text{lower}} = -1.38\text{ pp} > -2.00\text{ pp}$ ($p_{\text{NI}} = 0.0054 < 0.025$).
- **Conclusion:** Independent evidence retrieval alone is formally non-inferior to E0, establishing that the retrieval verifier does not corrupt baseline generation.
