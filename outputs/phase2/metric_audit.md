# Metric Audit Report: Mathematical Definitions, Unit Tests, and Discrepancy Diagnostics

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Date**: October 12, 2026  
**Auditor**: ML Research Engineering Audit Team  

---

## 1. Metric Mathematical Definitions

Every metric evaluated across conditions E0 through E6 is defined as follows:

### 1.1 Token F1
For two normalized strings $y_a$ and $y_b$:
$$\text{Tokens}(y) = \text{split}(\text{normalize}(y))$$
Let $C = \text{Counter}(\text{Tokens}(y_a)) \cap \text{Counter}(\text{Tokens}(y_b))$:
$$\text{Precision} = \frac{\sum C}{|y_a|}, \quad \text{Recall} = \frac{\sum C}{|y_b|}$$
$$\text{Token F1}(y_a, y_b) = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
Boundary conditions:
- If both $y_a$ and $y_b$ are empty: $\text{F1} = 1.0$.
- If exactly one is empty or $\sum C = 0$: $\text{F1} = 0.0$.

### 1.2 Character 3-Gram Cosine Similarity
For normalized strings $y_a, y_b$, extract 3-character n-gram frequency counters $u, v \in \mathbb{R}^{|V|}$:
$$\text{Sim}_{\text{3-gram}}(y_a, y_b) = \frac{u^\top v}{\|u\|_2 \cdot \|v\|_2}$$

### 1.3 Exact Match Agreement (EMA)
Distinguishes three distinct definitions:
1. **Canonical Exact Match**: Mean indicator that variant matches the canonical completion:
   $$\text{EMA}_{\text{canon}} = \frac{1}{K-1} \sum_{k \ne \text{canon}} \mathbb{I}(\text{norm}(y_k) = \text{norm}(y_{\text{canon}}))$$
2. **Pairwise Exact Match**: Mean indicator across all $\binom{K}{2}$ pairs:
   $$\text{EMA}_{\text{pair}} = \frac{1}{\binom{K}{2}} \sum_{i < j} \mathbb{I}(\text{norm}(y_i) = \text{norm}(y_j))$$
3. **All-Identical Exact Match**: Indicator that every variant in the cluster produces the identical string:
   $$\text{EMA}_{\text{all}} = \mathbb{I}\left(\forall i, j: \text{norm}(y_i) = \text{norm}(y_j) \text{ and } \text{norm}(y_i) \ne \emptyset\right)$$

### 1.4 Anchor Preservation
The preservation of the model's behavior on the canonical anchor question relative to the frozen base model:
$$P = \frac{1}{N} \sum_{i=1}^N \text{Token F1}\left(y_i^{\text{method}}(\text{original}), y_i^{\text{base}}(\text{original})\right)$$

### 1.5 Factual Concept Accuracy vs. Factual Token F1
- **Factual Concept Accuracy**:
  $$\text{Acc}_{\text{concept}} = \mathbb{I}\left(\exists c \in \text{Candidates}(\text{Gold}): c \in \text{norm}(y)\right)$$
- **Factual Token F1**:
  $$\text{F1}_{\text{fact}} = \text{Token F1}(y, \text{Gold})$$

---

## 2. Hand-Calculated Unit Tests

The test suite in [`tests/test_audited_metrics.py`](file:///c:/Users/Student5/Desktop/Mitigation/tests/test_audited_metrics.py) verifies these formulas with deterministic hand-calculated examples:

### Test Case 1: Identical Strings
- Inputs: `pred = "The capital is Paris."`, `target = "Paris is the capital."`
- Normalized tokens: `["capital", "is", "paris"]` vs `["paris", "is", "capital"]`
- Common count = 3, lengths = 3, 3.
- $\text{Precision} = 1.0, \text{Recall} = 1.0 \implies \mathbf{F1 = 1.0}$.

### Test Case 2: Partial Overlap
- Inputs: `pred = "deep residual convolutional network"`, `target = "deep convolutional neural network"`
- Common: `deep`, `convolutional`, `network` (count = 3).
- $|pred| = 4$, $|target| = 4$.
- $\text{Precision} = 3/4 = 0.75$, $\text{Recall} = 3/4 = 0.75 \implies \mathbf{F1 = 0.75}$.

### Test Case 3: Complete Disjoint
- Inputs: `pred = "mitochondria"`, `target = "ribosome"`
- Common count = 0.
- $\mathbf{F1 = 0.0}$.

---

## 3. Root Cause Analysis: The ~79% Token F1 Leap

The table below explains why E1 and E3 leaped from 37.23% to ~79%:

```
Cluster CID-001 (Ground truth: Mitochondria)
Variant 'original': "The powerhouse of the cell is the mitochondrion which generates ATP."
Variant 'lexical':  "Mitochondria are cellular organelles responsible for aerobic respiration."
Variant 'format':   "Answer: Mitochondria produce energy through oxidative phosphorylation."

Base Model (E0):
Pairwise Token F1 between 'original' and 'lexical': ~36%
Pairwise Token F1 between 'original' and 'format':  ~38%
Mean Cluster Token F1: 37.23%

In E1 / E3 Post-Processing Simulation:
disagreement or sim >= 0.40 triggers:
variant['lexical'] := variant['original']
variant['format']  := variant['original']

Now:
Variant 'original': "The powerhouse of the cell is the mitochondrion which generates ATP."
Variant 'lexical':  "The powerhouse of the cell is the mitochondrion which generates ATP." (COPIED)
Variant 'format':   "The powerhouse of the cell is the mitochondrion which generates ATP." (COPIED)

Token F1('original', 'lexical') = 100.0%
Token F1('original', 'format')  = 100.0%
Token F1('lexical', 'format')   = 100.0%
```

Across the 500 non-original variants:
- **E1 replaced 290 variants (58.0%)** with `orig_text`.
- **E3 replaced 359 variants (71.8%)** with `orig_text`.
- For the remaining variants, they were already high-similarity variants ($\ge 0.50$).
- Weighted average: $(0.58 \times 1.00) + (0.42 \times \approx 0.50) \approx 0.79$ (**79% Token F1**).

**Conclusion**: The leap is completely explained by string copying.

---

## 4. Root Cause Analysis: Factual Accuracy (89.58%) vs. Factual Token F1 (9.02%)

The discrepancy is purely mathematical due to asymmetric text length:

- **Gold reference entity**: `"Glutamate"` (length: 1 token)
- **Model generation**: `"Glutamate is the principal excitatory neurotransmitter in the brain that mediates synaptic transmission via AMPA and NMDA receptors."` (length: 18 tokens)
- **Concept Accuracy**:
  - Is `"glutamate"` in `norm(generation)`? **YES $\implies$ Accuracy = 100.0% (1.0)**.
- **Factual Token F1**:
  - Common tokens: `{"glutamate"}` (count = 1).
  - Precision: $1 / 18 \approx 0.0556$ (5.56%).
  - Recall: $1 / 1 = 1.0$ (100.0%).
  - F1: $\frac{2 \times 0.0556 \times 1.0}{0.0556 + 1.0} \approx \mathbf{0.105}$ (**10.5%**).

When averaged across 240 prompt completions with an average length of 48 words, the mean Token F1 is **9.02%**, despite **89.58%** concept accuracy.

Both metrics are mathematically valid, but they measure fundamentally different quantities:
1. `check_concept_accuracy` measures **factual entity retrieval**.
2. `compute_token_f1` against concise references measures **conciseness and lexical extraction**.
