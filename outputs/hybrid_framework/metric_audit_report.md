# Comprehensive Metric Audit and Formulation Report

**Project**: `prompt-sensitivity-hybrid`  
**Component**: Framework Metric Formulation, Mathematical Audit, and Discrepancy Resolution  
**Date**: October 12, 2026  

---

## 1. Executive Summary: Investigation of Metric Discrepancies

A critical discrepancy was identified in previous experimental reports:
- **Reported Factual Accuracy**: $\approx 90.00\%$
- **Reported Factual Token F1**: $\approx 9.02\%$

### Root Cause Analysis
1. **Asymmetric Length in Ground Truth vs. Model Generation**:
   - In [`data/external_eval/correctness_subset.jsonl`](file:///c:/Users/Student5/Desktop/Mitigation/data/external_eval/correctness_subset.jsonl), the ground-truth target `gold_reference_answer` is a concise entity (e.g., `"Glutamate"`, `"Ohm"`, or `"Angiotensin-converting enzyme (ACE)"`), typically containing **$1$ to $4$ words**.
   - The model, operating under instruction-following and greedy generation with `max_new_tokens=64`, generates multi-sentence technical answers or structured options (e.g., `"A. Acetylcholine B. Dopamine C. GABA D. Glutamate\nAnswer: D\nThe primary excitatory neurotransmitter in the mammalian central nervous system is glutamate..."`), averaging **$48.8$ words**.
2. **Mathematical Breakdown of Token F1**:
   $$\text{Precision} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{target}}|}{|\text{Tokens}_{\text{pred}}|} \approx \frac{1}{48} \approx 0.0208$$
   $$\text{Recall} = \frac{|\text{Tokens}_{\text{pred}} \cap \text{Tokens}_{\text{target}}|}{|\text{Tokens}_{\text{target}}|} = \frac{1}{1} = 1.0000$$
   $$\text{Token F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.0208 \times 1.0}{0.0208 + 1.0} \approx \frac{0.0416}{1.0208} \approx 0.0407 \quad (4.07\%)$$
3. **Resolution**:
   - `concept_match_accuracy` measures whether the verified factual entity/concept is present in the response ($90\%$ recall).
   - Direct string Token F1 between a 50-word answer and a 1-word target is mathematically dominated by precision penalty.
   - For factual evaluation, we formulate:
     a) **Concept Recall / Factual Entity Accuracy**: Verified concept presence (normalized substring and entity alias match).
     b) **Concise Extracted F1**: Evaluated after extracting the concise answer entity from the model output.
     c) **Full-Answer F1**: Evaluated against the full reference explanation when gold explanations are available.

---

## 2. Rigorous Metric Definitions

### 2.1 Within-Cluster Semantic Consistency ($C$)
Let a semantic cluster $\mathcal{C}$ contain $K$ prompt variants $\{x_1, \dots, x_K\}$ where $x_1$ is the canonical prompt, with completions $\{y_1, \dots, y_K\}$.

1. **Pairwise Exact Match Agreement (EMA)**:
   $$\text{EMA}_{\text{pairwise}}(\mathcal{C}) = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \mathbb{I}[f_{\text{norm}}(y_i) = f_{\text{norm}}(y_j)]$$
2. **Canonical Exact Match Agreement**:
   $$\text{EMA}_{\text{canonical}}(\mathcal{C}) = \frac{1}{K-1} \sum_{j=2}^K \mathbb{I}[f_{\text{norm}}(y_j) = f_{\text{norm}}(y_1)]$$
3. **All-Identical Cluster Agreement**:
   $$\text{EMA}_{\text{all}}(\mathcal{C}) = \mathbb{I}\left[ \bigwedge_{j=2}^K (f_{\text{norm}}(y_j) = f_{\text{norm}}(y_1)) \right]$$
4. **Pairwise Token F1**:
   $$\text{F1}_{\text{pairwise}}(\mathcal{C}) = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \text{TokenF1}(y_i, y_j)$$
5. **Canonical Token F1**:
   $$\text{F1}_{\text{canonical}}(\mathcal{C}) = \frac{1}{K-1} \sum_{j=2}^K \text{TokenF1}(y_j, y_1)$$
6. **Semantic Embedding / N-gram Similarity**:
   $$\text{Sim}_{\text{ngram}}(\mathcal{C}) = \frac{1}{K-1} \sum_{j=2}^K \cos\left(\mathbf{v}_{3\text{gram}}(y_j), \mathbf{v}_{3\text{gram}}(y_1)\right)$$

### 2.2 Anchor Preservation ($P$)
Anchor preservation measures whether the mitigation alters the model's behavior on the original canonical prompt relative to the **unmodified baseline model**:
$$\text{Preservation}(\mathcal{C}) = \text{TokenF1}\left(y_{\text{mitigated}}(x_{\text{orig}}), y_{\text{base}}(x_{\text{orig}})\right)$$
*Note: For the base model itself, this is definitionally $100.0\%$.*

### 2.3 Output Diversity ($D$)
$$\text{Diversity Ratio} = \frac{|\{f_{\text{norm}}(y_i) : i = 1, \dots, N_{\text{total}}\}|}{N_{\text{total}}}$$

### 2.4 Statistical Testing & Bootstrap Inference
- **Independent Sampling Unit**: The cluster $\mathcal{C}$ is the primary independent unit.
- **Bootstrap 95% Confidence Intervals**: 1,000 cluster-level resamples with replacement.
- **Paired Hypothesis Tests**: Two-sided Wilcoxon signed-rank test on cluster-level paired differences:
  $$d_c = \text{Metric}_{\text{mitigated}}(\mathcal{C}_c) - \text{Metric}_{\text{base}}(\mathcal{C}_c)$$
- **Family-Wise Error Rate**: Holm-Bonferroni step-down correction across all tested hypothesis pairs.
