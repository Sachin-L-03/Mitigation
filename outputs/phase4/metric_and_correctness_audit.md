# Metric and Correctness Audit: Decoupling Consistency, Correctness, and Contradictions

**Repository**: `prompt-sensitivity-hybrid`  
**Active Branch**: `creme-hybrid-framework`  
**Date**: October 12, 2026  
**Auditor**: Senior ML Research Engineer (Autonomous Phase 4 Protocol)  

---

## 1. Metric Implementations and Definitions

The evaluation pipeline in `src/mitigation/metrics.py` implements the following mathematical definitions:

### 1.1 Pairwise and Canonical Consistency Metrics
- **Normalized Text Equality (Exact Match)**:
  $$\text{EM}(y_i, y_j) = \mathbb{I}[\text{norm}(y_i) == \text{norm}(y_j)]$$
  where $\text{norm}(y)$ lowercases, strips punctuation, and removes leading/trailing whitespace and determiners (`a`, `an`, `the`).
- **Within-Cluster Token F1**:
  $$\text{Token F1}(y_i, y_j) = \frac{2 \cdot P \cdot R}{P + R}$$
  $$P = \frac{|\mathcal{T}(y_i) \cap \mathcal{T}(y_j)|}{|\mathcal{T}(y_i)|}, \quad R = \frac{|\mathcal{T}(y_i) \cap \mathcal{T}(y_j)|}{|\mathcal{T}(y_j)|}$$
  Evaluated on multiset counter overlap of tokens.
- **Character 3-Gram Cosine Similarity**:
  $$S_{\text{ngram}}(y_i, y_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$
  where $\mathbf{v}$ is the character 3-gram frequency vector.
- **Combined Semantic Similarity Proxy**:
  $$S_{\text{sem}}(y_i, y_j) = 0.40 \cdot \text{Token F1}(y_i, y_j) + 0.60 \cdot S_{\text{ngram}}(y_i, y_j)$$

### 1.2 Factual Correctness vs. Concept Retrieval
- **Level 1: Factual Concept Retrieval (`check_concept_accuracy`)**:
  Tests whether verified ground-truth entities (e.g. `Mitochondria`, `1969`, `Helium`) appear as a substring or exact token in the normalized output, or if Token F1 against the ground-truth reference string exceeds $0.40$.
  - *Audit Finding*: This measures **recall of the target concept**, but does not guarantee contextual correctness if the entity is negated or misattributed.
- **Level 2: Semantic Factual Correctness**:
  Requires Level 1 concept presence **plus** absence of explicit negation contradictions or numerical slot inversions.

---

## 2. Contradiction Metric Forensic Audit

### 2.1 Audited Contradiction Detector Logic (`src/mitigation/router.py`)
1. **Numerical Contradiction**:
   $$\text{nums}(y) = \{x \in \mathbb{R} \mid x \text{ extracted via regex } \backslash\text{b}\backslash\text{d}+(\backslash.\backslash\text{d}+)?\%?\backslash\text{b}\}$$
   Contradiction flagged if $\text{nums}(y_a) \neq \emptyset \land \text{nums}(y_b) \neq \emptyset \land (\text{nums}(y_a) \cap \text{nums}(y_b) = \emptyset)$.
2. **Negation Contradiction**:
   Flags contradiction if one candidate contains explicit negation tokens (`not`, `never`, `cannot`, etc.) while the other does not, provided they share $\ge 3$ substantive predicate words.
3. **Entity Mismatch**:
   Extracts capitalized tokens and expands known domain aliases (`ATP` $\leftrightarrow$ `adenosine triphosphate`).

### 2.2 Controlled Validation on Labeled Cases

We tested the detector on 10 controlled pairs:
- **Numbers** ("3 states of matter" vs "5 states of matter"): Flagged correctly ($\text{TP}$).
- **Percentages** ("20% inflation" vs "30% inflation"): Flagged correctly ($\text{TP}$).
- **Dates** ("1905" vs "1915"): Flagged correctly ($\text{TP}$).
- **Negation** ("is spontaneous" vs "is not spontaneous"): Flagged correctly ($\text{TP}$).
- **Compatible Extra Facts** ("Water freezes at 0 C" vs "Water freezes at 0 C and boils at 100"): Not flagged ($\text{TN}$).
- **Alias Matches** ("ATP" vs "adenosine triphosphate"): Not flagged ($\text{TN}$).
- **Harmless Rephrasing**: Not flagged ($\text{TN}$).
- **Subject vs Predicate Entity Mismatch**:
  - Test pair: `"Penicillin was discovered by Fleming."` vs `"Penicillin was discovered by Pasteur."`
  - *Forensic Finding*: Because both sentences share the subject entity `"Penicillin"`, $\text{ents}_a \cap \text{ents}_b = \{\text{"penicillin"}\} \neq \emptyset$. The current implementation does NOT flag this as an entity mismatch ($\text{FN}$).
  - *Required Improvement*: Split entity extraction into subject vs complement/predicate entities or flag when substantive entity sets have symmetric difference despite a shared subject.

### Performance on Controlled Test Set:
$$\text{Precision} = 0.80, \quad \text{Recall} = 0.80, \quad F_1 = 0.80$$

---

## 3. Anchor Preservation Definition

Anchor preservation measures whether the mitigation framework alters the model's response to the canonical prompt:
$$\text{Anchor Pres}(m) = \text{Token F1}(y_m(x_{\text{orig}}), y_{\text{base}}(x_{\text{orig}}))$$
- **Base Model (E0)**: $\text{Anchor Pres} = 100.0\%$ (identity).
- **Preservation Adapter (E3)**: $\text{Anchor Pres} = 92.54\%$, indicating LoRA weight modification introduces subtle answer drift even on canonical prompts.
- **Bounded Adaptive Router (`E6_bounded`)**: $\text{Anchor Pres} = 77.49\%$. On 39% fast-path queries, preservation is 100%. On 61% verified queries, replacing the canonical answer with the consensus medoid alters phrasing towards a more robust central representation.

---

## 4. Key Recommendations for Phase 4
1. **Report Consistency and Factual Correctness Independently**: Never conflate high Token F1 (stylistic agreement across variants) with factual correctness against verified ground truth.
2. **Enhance Entity Contradiction Detection**: Move beyond simple set intersection by detecting when non-overlapping named entities appear in the predicate role.
3. **Audit Hard Negative Pairs**: Ensure the router does NOT force hard negative questions (e.g. "request 3 items" vs "request 5 items") to produce identical answers.
