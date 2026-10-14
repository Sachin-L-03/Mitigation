# Phase 5 Metric Definitions and Statistical Formulas

**Document Status:** COMPLETE & FROZEN  
**Repository:** `prompt-sensitivity-hybrid`  
**Protocol Version:** 1.0.0  
**Audit Date:** 2026-10-13  
**Auditor:** Experimental Statistician & ML Evaluation Auditor  

---

## 1. Overview and Analysis Units

In Phase 5, all confirmatory statistical claims use the **semantic question cluster** $C_i$ as the primary unit of analysis ($i = 1, \dots, N_{\text{clusters}}$). Prompt-level observations represent repeated measurements within clusters.

---

## 2. Factual Accuracy Metrics

### 2.1 Cluster-Level Factual Correctness ($y_i$)
For cluster $C_i$ with canonical prompt $p_{i, 0}$ and completion $a_i(M)$ under method $M$:
$$y_i(M) = \begin{cases} 1 & \text{if } a_i(M) \text{ satisfies verified reference criteria for } C_i \\ 0 & \text{otherwise} \end{cases}$$

Reference satisfaction rules:
1. **Entity Match:** Token overlap or normalized substring match with canonical answer or any verified acceptable alias in `acceptable_answers`.
2. **Exact Numeric:** Canonical numerical value and unit match, respecting mathematical sign (e.g. $-273.15$ requires negative sign).
3. **Slot Match:** Delimited multi-slot values match all designated slots (e.g. `CH4 / NH3` matches `CH4` and `NH3` in designated slots).
4. **Format Match:** Adherence to requested structured output template.

### 2.2 Population Factual Accuracy ($A_{\text{fact}}$)
$$A_{\text{fact}}(M) = \frac{1}{N} \sum_{i=1}^N y_i(M)$$
Where $N = 226$ in the confirmatory test cohort. Abstentions are explicitly scored as $y_i = 0$ in this metric.

---

## 3. Consistency and Robustness Metrics

### 3.1 Token F1 Across Paraphrases
For prompt variant $v \in \{1, \dots, V-1\}$ and canonical prompt completion $a_{i, 0}$:
$$\text{Precision}_{i, v} = \frac{|\text{Tokens}(a_{i, v}) \cap \text{Tokens}(a_{i, 0})|}{|\text{Tokens}(a_{i, v})|}, \quad \text{Recall}_{i, v} = \frac{|\text{Tokens}(a_{i, v}) \cap \text{Tokens}(a_{i, 0})|}{|\text{Tokens}(a_{i, 0})|}$$
$$\text{F1}_{i, v} = \frac{2 \cdot \text{Precision}_{i, v} \cdot \text{Recall}_{i, v}}{\text{Precision}_{i, v} + \text{Recall}_{i, v} + \epsilon}$$
Cluster-level Token F1 is the mean over the $V-1$ variant pairs:
$$\overline{\text{F1}}_i = \frac{1}{V-1} \sum_{v=1}^{V-1} \text{F1}_{i, v}$$

### 3.2 Semantic Similarity (Character N-Gram Jaccard)
Character 3-gram and 4-gram Jaccard similarity between candidate completions:
$$\text{Sim}_{\text{ngram}}(a, b) = \frac{|\text{Grams}(a) \cap \text{Grams}(b)|}{|\text{Grams}(a) \cup \text{Grams}(b)|}$$

### 3.3 Cluster All-Identical Completion Rate ($R_{\text{ident}}$)
$$R_{\text{ident}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( a_{i, 0} = a_{i, 1} = \dots = a_{i, V-1} \right)$$

---

## 4. False-Consensus and Contradiction Metrics

### 4.1 Unanimous False-Consensus Rate ($R_{\text{false\_cons}}$)
Measures the critical failure mode where all candidate completions agree on an answer that is factually false:
$$R_{\text{false\_cons}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( y_i = 0 \;\land\; \min_{u \neq v} \text{Sim}(a_{i, u}, a_{i, v}) \ge \tau_{\text{agree}} \right)$$
where $\tau_{\text{agree}} = 0.80$ Token F1 across candidate generations.

### 4.2 Numerical Contradiction Rate ($R_{\text{num\_contra}}$)
Fraction of pairwise variant comparisons containing non-overlapping numerical quantities or sign reversals (e.g. $+42$ vs $-42$).

### 4.3 Negation Contradiction Rate ($R_{\text{neg\_contra}}$)
Fraction of variant pairs where one completion asserts a predicate and another explicitly negates it with shared substantive nouns.

### 4.4 Entity Substitution Rate ($R_{\text{ent\_contra}}$)
Fraction of variant pairs containing non-synonymous named entities in the same syntactic role.

---

## 5. Abstention and Selective Prediction Metrics

### 5.1 Coverage ($\text{Cov}$)
$$\text{Cov} = 1 - \frac{N_{\text{abstained}}}{N} = \frac{N_{\text{answered}}}{N}$$

### 5.2 Selective Factual Accuracy ($A_{\text{selective}}$)
Accuracy evaluated strictly on answered cases:
$$A_{\text{selective}} = \frac{\sum_{i \in \text{Answered}} y_i}{N_{\text{answered}}}$$

### 5.3 Risk-Conditional Accuracy
Accuracy conditioned on detector risk classification:
$$A(r) = \frac{\sum_{i: \text{Risk}(i) = r} y_i}{N_{r}}, \quad r \in \{\text{Low}, \text{Moderate}, \text{High}\}$$

---

## 6. Live Serving and Latency Metrics

All latency measurements record interactive request processing on the RTX 4090:
- **Mean Latency ($\mu_L$):** Arithmetic mean wall-clock time (ms).
- **Median Latency ($p_{50}$):** 50th percentile response time.
- **Tail Latencies ($p_{95}, p_{99}$):** 95th and 99th percentile response times.
- **Generations per Query ($K_{\text{gen}}$):** Average number of LLM forward generation passes executed per user query.
- **Tokens per Query:** Average sum of input prompt tokens and generated output tokens.
- **Peak VRAM:** Maximum allocated GPU memory (GB) during inference.

---

## 7. Statistical Testing and Non-Inferiority Formulas

### 7.1 Paired Cluster Difference ($\delta$)
$$\delta = A_{\text{fact}}(\text{Candidate}) - A_{\text{fact}}(E0)$$

### 7.2 McNemar's Test for Paired Factual Outcomes
For discordant clusters where Candidate and E0 differ:
- $b$: Clusters where Candidate is correct and E0 is incorrect.
- $c$: Clusters where Candidate is incorrect and E0 is correct.
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad p = \text{Exact Binomial}(b; b+c, 0.5)$$

### 7.3 Formal Non-Inferiority Test
- Margin: $\Delta_{\text{margin}} = -0.020$ (-2.0 pp).
- Test: $H_0: \delta \le -0.02$ vs $H_1: \delta > -0.02$.
- Lower Confidence Bound:
$$\text{CI}_{\text{lower}} = \delta - z_{1-\alpha} \cdot \text{SE}(\delta)$$
where $\text{SE}(\delta) = \sqrt{\frac{b + c - (b - c)^2 / N}{N^2}}$, and $z_{0.975} = 1.960$.
- **Decision Rule:** Non-inferiority is declared if and only if $\text{CI}_{\text{lower}} > -0.020$ and $p_{\text{NI}} < 0.025$.

---

## 8. Summary Checklist

All metrics and mathematical formulations defined above are frozen for Phase 5 reporting.
