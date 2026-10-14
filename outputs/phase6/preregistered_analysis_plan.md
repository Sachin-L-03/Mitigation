# Phase 6: Preregistered Statistical Analysis Plan

**Protocol Reference:** `PHASE6-EXP-EV-001`  
**Date of Preregistration:** October 14, 2026  
**Status:** FROZEN PRIOR TO FINAL TEST EVALUATION  
**Repository:** `prompt-sensitivity-hybrid`  
**Branch:** `creme-hybrid-framework`  
**Base Model:** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 Quantization)  
**Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM), CUDA 12.1  
**Benchmark Artifacts:** `data/phase6_eval/`  
**Output Directory:** `outputs/phase6/`  

---

## 1. Research Objectives and Formal Hypotheses

The goal of Phase 6 is to determine whether independent evidence retrieval (Path B) and deterministic computation (Path A) can mitigate factual errors and unanimous false consensus missed by E3's internal consistency checks, while preserving E3's paraphrase consistency gains.

### 1.1 Primary Hypothesis ($H_1$)
- **Null Hypothesis ($H_{0,\text{primary}}$):** The factual accuracy of the combined evidence-grounded router (E4) is less than or equal to that of the baseline consistency router (E1) on the untouched Phase 6 test set:
  $$\Delta_{\text{fact}} = A_{\text{fact}}(E4) - A_{\text{fact}}(E1) \le 0$$
- **Alternative Hypothesis ($H_{1,\text{primary}}$):** The combined evidence-grounded router (E4) achieves statistically significant superiority in factual accuracy over E1:
  $$\Delta_{\text{fact}} = A_{\text{fact}}(E4) - A_{\text{fact}}(E1) > 0$$

### 1.2 Non-Inferiority Hypothesis vs Base Model ($H_{\text{NI}}$)
- **Null Hypothesis ($H_{0,\text{NI}}$):** The factual accuracy of candidate condition $E_k$ is inferior to the base model (E0) by more than 2.0 percentage points:
  $$\Delta_{\text{NI}} = A_{\text{fact}}(E_k) - A_{\text{fact}}(E0) \le -0.02$$
- **Alternative Hypothesis ($H_{1,\text{NI}}$):** Candidate condition $E_k$ is non-inferior to E0 within a 2.0 percentage point margin at one-sided $\alpha = 0.025$:
  $$\Delta_{\text{NI}} = A_{\text{fact}}(E_k) - A_{\text{fact}}(E0) > -0.02$$

### 1.3 Secondary Hypotheses
- **$H_{2}$ (Deterministic Gain):** Path A deterministic verification (E2) corrects arithmetic, modular, unit, and date errors compared to E1.
- **$H_{3}$ (Retrieval Gain):** Path B external evidence retrieval (E3) corrects factual general knowledge, science, and entity errors compared to E1.
- **$H_{4}$ (Consistency Preservation):** Paraphrase consistency (Token F1 across prompt variants) of E4 is non-inferior to E1 ($\Delta_{\text{F1}} \ge -0.01$).
- **$H_{5}$ (False Consensus Mitigation):** Unanimous incorrect consensus rate under E4 is strictly lower than under E1.

---

## 2. Experimental Conditions

The evaluation strictly compares 7 prespecified conditions:

| Condition | Codename | Architecture & Routing Logic | Target Capability |
| :--- | :--- | :--- | :--- |
| **E0** | Base Model | Single greedy generation per prompt | Baseline single-turn model capability |
| **E1** | E3 Risk-Aware Router | Candidate generation ($K=3$), medoid selection, slot check, no tools | Phase 5 consistency baseline |
| **E2** | E3 + Deterministic | E1 followed by Path A (Safe AST / SymPy / Unit converter) | Deterministic arithmetic & constraint correction |
| **E3** | E3 + Retrieval | E1 followed by Path B (SQLite FTS5 BM25 retrieval) | External evidence factual grounding |
| **E4** | Combined Router | E1 + Path A + Path B (deterministic first, retrieval fallback) | Primary deployment candidate |
| **E5** | Conservative Fallback | E4 with explicit abstention (`[UNCERTAINTY_FALLBACK]`) on unresolved high risk | High-precision / zero-hallucination deployment |
| **E6** | Oracle Diagnostic | Offline upper bound using gold reference labels strictly in offline scoring | Theoretical performance ceiling |

---

## 3. Independent Unit of Analysis and Sample Sizing

### 3.1 Unit of Analysis
The independent experimental unit is the **semantic question cluster** ($N$), NOT individual prompt variants or pairwise comparisons.
- Each cluster comprises 4 prompt variants (`original`, `lexical`, `syntactic`, `concise`).
- Cluster-level factual accuracy is defined as the mean correctness across the 4 variants (or canonical accuracy where evaluated at the cluster medoid).
- Paraphrase consistency is measured by pairwise Token F1 and cluster variance across the 4 variants within the cluster.

### 3.2 Dataset Stratification and Partitioning
The Phase 6 benchmark (`data/phase6_eval/`) contains exactly 600 clusters (2,400 prompts) stratified equally across 12 distinct domains:
1. `deterministic_arithmetic` (50 clusters)
2. `numerical_reasoning` (50 clusters)
3. `date_time_math` (50 clusters)
4. `general_knowledge_history` (50 clusters)
5. `natural_sciences` (50 clusters)
6. `entity_relations` (50 clusters)
7. `everyday_practical_facts` (50 clusters)
8. `multi_constraint` (50 clusters)
9. `structured_output` (50 clusters)
10. `ambiguous_non_equivalent` (50 clusters)
11. `competing_hypotheses` (50 clusters)
12. `unanimous_hallucination_diagnostic` (50 clusters)

**Partitioning:**
- **Validation Split ($N_{\text{val}} = 150$ clusters, 600 prompts):** 25% of dataset. Used strictly for threshold tuning, ablation audits, and prompt checking.
- **Test Split ($N_{\text{test}} = 450$ clusters, 1,800 prompts):** 75% of dataset. Frozen and untouched until all configurations and code are locked.

### 3.3 Power Analysis
For a paired cluster design with $N=450$ clusters:
- With discordant pair proportion $p_{\text{disc}} = (n_{10} + n_{01}) / N \approx 0.12$ (54 discordant clusters),
- A true accuracy advantage of $+3.5$ percentage points ($\Delta = 0.035$) yields power $(1 - \beta) > 0.92$ at two-sided $\alpha = 0.05$.
- For non-inferiority testing with margin $\Delta_{\text{margin}} = -0.02$ at one-sided $\alpha = 0.025$, power exceeds $0.98$ for $\Delta \ge 0.00$.

---

## 4. Prespecified Metrics

### 4.1 Factual Accuracy ($A_{\text{fact}}$)
Evaluated offline against gold references.
- **Scoring Rules:**
  - `numerical_exact`: Requires exact signed numerical equivalence.
  - `entity_match`: Requires canonical entity or acceptable alias match.
  - `slot_match`: Requires all specified delimited slots to be present and accurate.
  - **Abstention Rule:** Any response containing `[UNCERTAINTY_FALLBACK]` is strictly scored as **0.0 (Incorrect)** in the primary factual accuracy metric.

### 4.2 Selective Accuracy & Coverage
- **Coverage ($C$):** Fraction of clusters where the system outputs an answer ($1 - \text{Abstention Rate}$).
- **Selective Accuracy ($A_{\text{sel}}$):** Factual accuracy computed strictly over answered clusters.

### 4.3 Paraphrase Consistency ($S_{\text{cons}}$)
- **Mean Pairwise Token F1:** Harmonic mean of precision and recall across all $\binom{4}{2} = 6$ pairs of prompt completions within a cluster.
- **All-Identical Rate:** Fraction of clusters where all 4 prompt variants produce identical canonical answers.
- **Contradiction Rates:** Fraction of clusters exhibiting numerical, entity, or negation contradictions.

### 4.4 False-Consensus Metrics
- **Unanimous False-Consensus Rate:** Proportion of clusters where all 4 base model completions agree on an incorrect factual claim.
- **Consensus Correction Rate:** Proportion of unanimous false-consensus clusters successfully corrected by verification.
- **Corruption Rate:** Proportion of initially correct answers improperly altered to incorrect answers by verification.

### 4.5 Operational Latency
- Measured using high-resolution hardware clocks with CUDA synchronization (`torch.cuda.synchronize()`).
- Reported: Mean, Median (p50), 95th percentile (p95), 99th percentile (p99).
- Stage latency breakdown: Ingestion, Candidate Generation, Path A (Deterministic), Path B (Retrieval), Path C (Verifier), Total End-to-End.

---

## 5. Statistical Testing and Multiplicity Control

1. **Primary Comparison (E4 vs E1):**
   - Paired McNemar exact test on cluster correctness indicator ($Y_i \in \{0, 1\}$).
   - 95% Two-sided Confidence Interval for $\Delta_{\text{fact}}$ using 10,000 paired cluster bootstrap resamples.
   - Prespecified significance threshold: $\alpha = 0.05$.

2. **Non-Inferiority Testing (E4 vs E0, E1 vs E0):**
   - 97.5% One-sided lower confidence bound for $\Delta_{\text{NI}} = A(E_k) - A(E0)$.
   - Rejection rule: Reject $H_{0,\text{NI}}$ if $\text{CI}_{97.5,\text{lower}} > -0.02$.
   - One-sided $p$-value from normal approximation of paired discordant pairs with non-inferiority offset:
     $$Z_{\text{NI}} = \frac{(n_{10} - n_{01}) - N \cdot \Delta_{\text{margin}}}{\sqrt{n_{10} + n_{01}}}$$

3. **Multiplicity Adjustment:**
   - Secondary comparisons (E2 vs E1, E3 vs E1, E5 vs E1) controlled using Holm-Bonferroni step-down procedure at FWER $\alpha = 0.05$.

4. **Consistency Comparison:**
   - Wilcoxon signed-rank test on paired cluster-level mean Token F1 scores.

---

## 6. Execution Protocol and Missing Data Handling

- **Zero Gold Reference Leakage:** Gold references (`correctness_subset.jsonl`) are strictly sequestered from the inference pipeline and inaccessible to the retriever, verifier, and model.
- **Hardware & Seed Locking:** Random seed frozen to 42; temperature = 0.0 (greedy decoding); top_p = 1.0; max_new_tokens = 64.
- **Missing or Failed Tool Calls:** If deterministic evaluation fails or errors, Path A returns `is_eligible=False`. If retrieval returns no passages, Path B returns `retrieval_success=False` with status `NO_RELEVANT_PASSAGES_FOUND`. Under no circumstances may synthetic or mock data be substituted.
