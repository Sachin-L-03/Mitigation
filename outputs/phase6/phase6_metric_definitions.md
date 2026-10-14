# Phase 6: Formal Metric Definitions and Evaluation Standards

**Specification Code:** `PHASE6-METRICS-001`  
**Date:** October 14, 2026  
**Auditor:** Experimental Statistician & Evaluation Engineer  
**Status:** FROZEN  

---

## 1. Overview and Core Unit of Evaluation

All primary, secondary, and operational metrics are evaluated at the level of the **semantic question cluster** ($N$), which constitutes the independent experimental unit. Each cluster $i \in \{1, \dots, N\}$ consists of 4 prompt variants ($V = 4$: `original`, `lexical`, `syntactic`, `concise`).

Evaluating at the cluster level eliminates pseudoreplication and ensures unbiased hypothesis testing.

---

## 2. Factual Accuracy Metrics

### 2.1 Strict Cluster Factual Accuracy ($A_{\text{fact}}$)
The primary endpoint of the evaluation. For candidate condition $E_k$ over cluster $i$, let $Y_i \in \{0, 1\}$ denote whether the selected canonical answer satisfies the gold reference:
$$A_{\text{fact}}(E_k) = \frac{1}{N} \sum_{i=1}^N Y_i$$

**Scoring Criteria:**
1. **`numerical_exact`:** Requires exact signed numerical equivalence:
   $$\text{Prediction} \equiv \text{Gold Reference} \quad (\text{e.g. } -31 \neq 31, \quad 373.15 \equiv 373.15\text{ K})$$
2. **`entity_match`:** Requires canonical entity match or inclusion in verified `acceptable_aliases`.
3. **`slot_match`:** Requires all delimited slots / structured values to be accurately present.
4. **Strict Abstention Scoring:** Any output containing `[UNCERTAINTY_FALLBACK]` is unconditionally scored as **$0.0$ (Incorrect)** in $A_{\text{fact}}$.

### 2.2 Paired Factual Accuracy Difference ($\Delta_{\text{fact}}$)
For matched conditions evaluated across identical clusters:
$$\Delta_{\text{fact}}(E_A, E_B) = A_{\text{fact}}(E_A) - A_{\text{fact}}(E_B) = \frac{b - c}{N}$$
where $b$ is the number of clusters where $E_A = 1, E_B = 0$ (wins for $E_A$), and $c$ is the number of clusters where $E_A = 0, E_B = 1$ (wins for $E_B$).

### 2.3 Paired Bootstrap Confidence Intervals
Evaluated via 10,000 paired cluster bootstrap resamples. For each resample $b \in \{1, \dots, 10000\}$, compute $\Delta^{(b)}$. The two-sided 95% confidence interval is $[\Delta_{(250)}, \Delta_{(9750)}]$.

### 2.4 Formal Non-Inferiority Criterion ($H_{\text{NI}}$)
Non-inferiority against baseline $E_0$ is formally established if and only if the **97.5% one-sided lower confidence bound** exceeds the prespecified margin $\Delta_{\text{margin}} = -0.02$ (-2.0 percentage points):
$$\text{CI}_{97.5,\text{lower}} = \Delta_{\text{fact}} - z_{0.025} \cdot \text{SE}_{\Delta} > -0.02$$
where $\text{SE}_{\Delta} = \frac{\sqrt{b + c - \frac{(b - c)^2}{N}}}{N}$, and $z_{0.025} = 1.95996$.

---

## 3. Evidence Quality Metrics

| Metric | Mathematical Definition | Target / Benchmark Standard |
| :--- | :--- | :--- |
| **Retrieval Success Rate** | $\frac{\text{Count}(\text{Passages} > 0)}{\text{Total Queries}}$ | $\ge 90\%$ for eligible factual queries |
| **Evidence-Support Precision** | $\frac{\text{Supported Claims Verified Correct}}{\text{Total Supported Claims}}$ | $\ge 95\%$ |
| **Contradiction Detection Rate** | $\frac{\text{Contradicted Outcomes Identified}}{\text{Total Factually Incorrect Candidates}}$ | Higher is better |
| **Insufficient-Evidence Rate** | $\frac{\text{Count}(\text{INSUFFICIENT\_EVIDENCE})}{\text{Total Unverified Queries}}$ | Correct classification when passages are absent |
| **Source Conflict Rate** | $\frac{\text{Count}(\text{Conflicting Passages})}{\text{Total Retrieved Query Sets}}$ | Identifies divergent sources |
| **Correction Success Rate** | $\frac{\text{Incorrect Answers Altered to Correct}}{\text{Total Corrected Answers}}$ | $\ge 90\%$ |
| **Corruption Rate** | $\frac{\text{Correct Answers Altered to Incorrect}}{\text{Total Clusters}}$ | $\le 1.0\%$ |

---

## 4. Prompt Robustness and Consistency Metrics

### 4.1 Mean Pairwise Token F1 ($S_{\text{cons}}$)
For each cluster $i$, let $A_{i, j}$ and $A_{i, k}$ denote completions for prompt variants $j$ and $k$ ($j < k \le 4$):
$$\text{Token F1}(A_{i, j}, A_{i, k}) = \frac{2 \cdot P \cdot R}{P + R}$$
$$\bar{S}_{\text{cons}} = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{6} \sum_{1 \le j < k \le 4} \text{Token F1}(A_{i, j}, A_{i, k}) \right)$$

### 4.2 All-Identical Cluster Rate
Proportion of clusters where all 4 prompt variants produce identical strings after normalization:
$$\text{All-Identical Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( A_{i, 1} = A_{i, 2} = A_{i, 3} = A_{i, 4} \right)$$

### 4.3 Semantic Contradiction Rates
- **Numerical Contradiction Rate:** Fraction of clusters where candidate outputs contain mutually conflicting numbers for identical attributes.
- **Entity Contradiction Rate:** Fraction of clusters where candidate outputs mention non-synonymous named entities.
- **Negation Contradiction Rate:** Fraction of clusters where one variant asserts a claim and another asserts its logical negation.

---

## 5. False-Consensus Metrics

1. **Unanimous False-Consensus Rate ($R_{\text{ufc}}$):**
   $$R_{\text{ufc}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\forall j: Y_{i, j} = 0 \quad \text{and} \quad \forall j, k: \text{F1}(A_{i, j}, A_{i, k}) \ge 0.85\right)$$
2. **False-Consensus Correction Rate ($C_{\text{ufc}}$):**
   $$C_{\text{ufc}} = \frac{\text{UFC Clusters Corrected by Verification}}{\text{Total UFC Clusters}}$$
3. **All-Candidate-Wrong Recovery Rate ($R_{\text{recov}}$):**
   Fraction of clusters where all $K=3$ generated candidates were factually wrong, but independent evidence retrieval or deterministic computation allowed the system to synthesize or select the correct answer.

---

## 6. Operational Performance Metrics

- **Answer Coverage ($C$):** $C = \frac{N_{\text{answered}}}{N}$ where $N_{\text{answered}} = N - N_{\text{abstained}}$.
- **Abstention Rate ($R_{\text{abst}}$):** $R_{\text{abst}} = \frac{N_{\text{abstained}}}{N} = 1 - C$.
- **Selective Accuracy ($A_{\text{sel}}$):** Accuracy strictly over answered cases:
  $$A_{\text{sel}} = \frac{1}{N_{\text{answered}}} \sum_{i \in \text{Answered}} Y_i$$
- **Interactive Serving Latency:** Measured using hardware-synchronized timers (`torch.cuda.synchronize()`). Reported across p50, p95, and p99 percentiles for total response time and stage components (Ingestion, Generation, Path A, Path B, Path C, Medoid).
- **Interactive Throughput:** Requests per second under real single-user sequential serving on an NVIDIA RTX 4090.
