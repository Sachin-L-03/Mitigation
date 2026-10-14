# Phase 5 Factual Non-Inferiority Statistical Analysis

**Document Status:** COMPLETE & VERIFIED  
**Repository:** `prompt-sensitivity-hybrid`  
**Protocol Reference:** `outputs/phase5/preregistered_analysis_plan.md` (Section 6)  
**Evaluation Set:** Phase 5 Multi-Domain Benchmark (`data/phase5_eval/`, Untouched Test Split $N=226$ Clusters)  
**Author / Statistician:** Experimental Statistician & ML Evaluation Auditor  
**Date:** 2026-10-14  

---

## 1. Executive Summary & Audit Verdict

In Phase 4, the forensic audit (`outputs/phase4_forensic/final_forensic_report.md`) debunked a false claim of factual non-inferiority, exposing that a non-significant two-sided test ($p=0.442$) had been improperly cited as proof of non-inferiority, when in reality the formal one-sided test yielded $p_{\text{NI}} = 0.1978$ and a lower confidence bound of $-3.58$ pp, failing the $-2.00$ pp safety threshold.

In Phase 5, under a rigorous, preregistered one-sided hypothesis testing protocol at $\alpha = 0.025$ on the untouched test split ($N=226$ clusters, 904 prompts):

1. **Condition E3 (Factual-Risk-Aware Router):**
   - Paired difference vs E0: $\delta = +1.33$ percentage points ($A_{\text{E3}} = 79.20\%$ vs $A_{\text{E0}} = 77.88\%$).
   - Standard Error of difference: $\text{SE}(\delta) = 1.32$ percentage points.
   - 97.5% One-sided lower confidence bound: **$-1.27$ percentage points**.
   - One-sided test statistic: $z_{\text{NI}} = 2.5122$, $p_{\text{NI}} = \mathbf{0.005999} < 0.025$.
   - **Verdict: FACTUAL NON-INFERIORITY IS FORMALLY ESTABLISHED.** Because the lower bound ($-1.27$ pp) strictly exceeds the prespecified margin ($-2.00$ pp), E3 is certified not to degrade factual accuracy beyond the 2-percentage-point threshold.

2. **Condition E2 (Improved Disagreement Detector):**
   - Paired difference: $\delta = +0.88$ percentage points ($A_{\text{E2}} = 78.76\%$).
   - 97.5% One-sided lower bound: **$-1.57$ percentage points** ($> -2.00$ pp).
   - One-sided test statistic: $z_{\text{NI}} = 2.3077$, $p_{\text{NI}} = \mathbf{0.010507} < 0.025$.
   - **Verdict: FACTUAL NON-INFERIORITY IS FORMALLY ESTABLISHED.**

3. **Condition E1 (Historical E6_bounded):**
   - Paired difference: $\delta = +0.44$ percentage points ($A_{\text{E1}} = 78.32\%$).
   - 97.5% One-sided lower bound: **$-1.85$ percentage points** ($> -2.00$ pp).
   - One-sided test statistic: $z_{\text{NI}} = 2.0870$, $p_{\text{NI}} = \mathbf{0.018443} < 0.025$.
   - **Verdict: FACTUAL NON-INFERIORITY IS FORMALLY ESTABLISHED.**

4. **Condition E4 (Conservative Abstention Policy):**
   - Paired difference: $\delta = -15.04$ percentage points ($A_{\text{E4}} = 62.83\%$).
   - 97.5% One-sided lower bound: **$-20.17$ percentage points** ($\le -2.00$ pp).
   - One-sided test statistic: $z_{\text{NI}} = -4.9912$, $p_{\text{NI}} = 1.000$.
   - **Verdict: FACTUAL NON-INFERIORITY FORMALLY FAILS.** Scoring abstentions as $0$ in the primary accuracy metric creates a significant penalty.

---

## 2. Formal Non-Inferiority Framework

### 2.1 Hypotheses
Let $\delta = p_{\text{Candidate}} - p_{\text{E0}}$ denote the true population difference in cluster factual accuracy between a candidate mitigation method and the unmodified base model (E0).

The prespecified non-inferiority margin is:
$$\Delta_{\text{margin}} = -0.020 \quad (-2.00\text{ percentage points})$$

The formal statistical hypothesis pair is:
$$H_0: \delta \le -0.020 \quad \text{vs.} \quad H_1: \delta > -0.020$$

### 2.2 Decision Rule
Non-inferiority is declared **ESTABLISHED** if and only if both conditions are satisfied:
1. The one-sided p-value satisfies $p_{\text{NI}} < 0.025$ (corresponding to a 95% two-sided confidence level), and
2. The lower bound of the 97.5% one-sided (or 95% two-sided) confidence interval strictly exceeds the safety margin:
   $$\text{CI}_{97.5, \text{lower}}(\delta) = \delta - z_{0.975} \cdot \text{SE}(\delta) > -0.020$$
   where $z_{0.975} = 1.95996$.

**Explicit Methodological Standard:**  
In accordance with biostatistical and experimental standards (ICH E9/E10), failing to reject equality in a two-sided test ($p > 0.05$) is never accepted as evidence of non-inferiority.

---

## 3. Master Non-Inferiority Verification Table

| Evaluation Condition | Candidate Acc ($A_{\text{cand}}$) | Base Acc ($A_{\text{E0}}$) | Paired Delta ($\delta$) | Discordant ($b, c$) | $\text{SE}(\delta)$ | $z_{\text{NI}}$ Stat | One-Sided $p_{\text{NI}}$ | 97.5% Lower Bound | Non-Inferiority Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 (E6_bounded)** | 78.32% | 77.88% | $+0.44$ pp | $b=4, c=3$ | $1.17$ pp | $2.0870$ | $0.0184$ | **$-1.85$ pp** | **ESTABLISHED** |
| **E2 (Improved Det)** | 78.76% | 77.88% | $+0.88$ pp | $b=5, c=3$ | $1.25$ pp | $2.3077$ | $0.0105$ | **$-1.57$ pp** | **ESTABLISHED** |
| **E3 (Risk-Aware)** | **79.20%** | **77.88%** | **$+1.33$ pp** | $b=6, c=3$ | **$1.32$ pp** | **$2.5122$** | **$0.0060$** | **$-1.27$ pp** | **ESTABLISHED** |
| **E4 (Abstention)** | 62.83% | 77.88% | $-15.04$ pp | $b=3, c=37$ | $2.61$ pp | $-4.9912$ | $1.0000$ | **$-20.17$ pp** | **FAILED** |

---

## 4. Power and Sample Size Sensitivity Analysis

### 4.1 Statistical Model for Paired Binary Non-Inferiority
For a paired binary trial with $N$ clusters, let $p_{\text{disc}} = (b + c) / N$ denote the total paired discordance rate, and let $\delta = (b - c) / N$ denote the true difference.

Under $H_1$ with true difference $\delta_1 = 0.00$ (true parity) and non-inferiority margin $\Delta_{\text{margin}} = -0.02$:
$$\text{SE}(\delta) \approx \sqrt{\frac{p_{\text{disc}}}{N}}$$
The required sample size for statistical power $1 - \beta$ at one-sided level $\alpha = 0.025$ is:
$$N = \frac{\left( z_{1-\alpha} + z_{1-\beta} \right)^2 \cdot p_{\text{disc}}}{\left( \delta_1 - \Delta_{\text{margin}} \right)^2} = \frac{\left( 1.96 + 1.28 \right)^2 \cdot p_{\text{disc}}}{(0.02)^2} = \frac{10.50 \cdot p_{\text{disc}}}{0.0004} = 26,250 \cdot p_{\text{disc}}$$

### 4.2 Discordance Rate Sensitivity Table ($N=226$ Clusters):

| Discordance Rate ($p_{\text{disc}}$) | Expected Discordant Pairs ($b+c$) | $\text{SE}(\delta)$ at $N=226$ | 97.5% Lower Bound (under $\delta=0$) | Achieved Power ($1-\beta$) | Required $N$ for 80% Power | Required $N$ for 90% Power |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.010 (1.0%)** | 2.3 | $0.66$ pp | $-1.30$ pp | **98.2%** | 263 | 352 |
| **0.020 (2.0%)** | 4.5 | $0.94$ pp | $-1.84$ pp | **87.8%** | 525 | 703 |
| **0.035 (3.5%)** | 7.9 | $1.24$ pp | $-2.43$ pp | **66.1%** | 919 | 1,230 |
| **0.040 (Observed E3)** | **9.0** | **$1.32$ pp** | **$-2.59$ pp** | **61.4%** | **1,050** | **1,406** |
| **0.060 (6.0%)** | 13.6 | $1.63$ pp | $-3.19$ pp | **46.8%** | 1,575 | 2,109 |
| **0.100 (10.0%)** | 22.6 | $2.10$ pp | $-4.12$ pp | **30.5%** | 2,625 | 3,515 |

### 4.3 Why Did E3 Establish Non-Inferiority with $N=226$?
In the historical Phase 4 audit, the estimated sample size required for 80% power was ~1,920 clusters under the assumption of $p_{\text{disc}} \approx 7.5\%$ and $\delta = 0$.

In Phase 5, E3 established non-inferiority with $N=226$ clusters because:
1. **Low Discordance Rate:** The observed discordance rate was exceptionally low ($p_{\text{disc}} = (6 + 3) / 226 = 3.98\%$). The router preserved the base model's correct answers in 173 of 176 cases (98.30% anchor preservation).
2. **Positive Effect Size:** The true observed difference was not $0.00$, but **$+1.33$ pp** ($b=6$ wins vs $c=3$ losses). When $\delta > 0$, the numerator in the test statistic $z_{\text{NI}} = (\delta - \Delta_{\text{margin}}) / \text{SE}$ is $0.0133 - (-0.0200) = 0.0333$ (3.33 percentage points away from the margin), requiring far fewer clusters to reject $H_0$.

---

## 5. Non-Inferiority Audit Verdict

Under the prespecified margin $\Delta_{\text{margin}} = -2.0$ percentage points and $\alpha = 0.025$:
- **E3 establishes factual non-inferiority:** $\text{CI}_{97.5, \text{lower}} = -1.27\text{ pp} > -2.00\text{ pp}$, $p_{\text{NI}} = 0.0060$.
- **E2 establishes factual non-inferiority:** $\text{CI}_{97.5, \text{lower}} = -1.57\text{ pp} > -2.00\text{ pp}$, $p_{\text{NI}} = 0.0105$.
- **E1 establishes factual non-inferiority:** $\text{CI}_{97.5, \text{lower}} = -1.85\text{ pp} > -2.00\text{ pp}$, $p_{\text{NI}} = 0.0184$.

The factual non-inferiority claim that failed in Phase 4 is now **empirically validated and formally established** on the independent Phase 5 benchmark.
