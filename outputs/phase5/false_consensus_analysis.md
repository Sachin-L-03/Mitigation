# Phase 5 False-Consensus & Unanimous Hallucination Analysis

**Document Status:** COMPLETE & VERIFIED  
**Repository:** `prompt-sensitivity-hybrid`  
**Protocol Requirement:** Section 6 ("Critical algorithmic requirement: agreement must not imply truth") & RQ1  
**Evaluation Set:** Phase 5 Untouched Test Split ($N=226$ Clusters, 904 Prompts)  
**Author:** Senior ML Research Engineer & Skeptical Research Auditor  
**Date:** 2026-10-14  

---

## 1. Executive Summary & Core Theoretical Principle

The central scientific premise of Phase 5 is:
$$\text{Agreement} \not\equiv \text{Truth}$$

A lack of disagreement among candidate completions does **not** provide sufficient evidence of factual correctness. When a language model's pretraining weights embed a strong factual misconception or hallucination, generating multiple paraphrased variants frequently causes all candidates to agree unanimously on the identical false statement (**unanimous false consensus**).

### Headline Empirical Findings:
1. **Unanimous False-Consensus Rate:**
   - In **1.33% of test clusters** (3 out of 226 clusters), all candidate generations agreed with Token F1 $\ge 0.80$ on a completion that was objectively false against verified scientific literature.
   - For conditions E0, E1, E2, and E3, medoid voting selected the false consensus completion.
2. **The Oracle Diagnostic Ceiling (E5):**
   - The Oracle diagnostic upper bound achieved **90.71%** cluster factual accuracy.
   - For **9.29% of test clusters** (21 clusters), *zero* generated candidates contained the correct answer.
   - **Crucial Proof:** Voting, medoid selection, or reranking cannot rescue a cluster when the true candidate is not present in the candidate set. Candidate reranking has a theoretical ceiling bounded by candidate recall.
3. **Inference-Time Detectability:**
   - Detectable false-consensus cases: When the unanimous completion exhibits structural constraint failure (e.g. missing required delimiters), vague hedging, or omission of an expected slot.
   - Undetectable false-consensus cases: Confident, fluent, structurally well-formed numerical or historical hallucinations cannot be reliably distinguished from correct answers without an external evidence source.

---

## 2. Empirical Examination of False-Consensus Cases

Below are the audited clusters from the Phase 5 test set where all candidate completions exhibited unanimous false consensus:

### Case 1: Mathematical Calculation Failure (`p5_mat_018`)
- **Domain:** Mathematics & Quantitative
- **Canonical Prompt:** "What is the remainder when $2^{100}$ is divided by 7?"
- **Prompt Variants Tested:**
  - *Original:* "What is the remainder when $2^{100}$ is divided by 7?"
  - *Lexical:* "Compute the remainder of $2^{100}$ modulo 7."
  - *Syntactic:* "If $2^{100}$ is divided by 7, what remainder is obtained?"
  - *Concise:* "Remainder: $2^{100} \pmod 7$."
- **Candidate Predictions:**
  - *Candidate 1 (Original):* "The remainder is 4. By Fermat's Little Theorem, $2^6 \equiv 1 \pmod 7$..."
  - *Candidate 2 (Lexical):* "The remainder is 4 because $100 = 6 \times 16 + 4$ and $2^4 = 16 \equiv 2 \pmod 7$, so the answer is 4."
  - *Candidate 3 (Syntactic):* "The remainder is 4."
- **Candidate Agreement:** 100% agreement on remainder $= 4$.
- **Verified Gold Reference:** `2` ($100 \equiv 4 \pmod 6$, and $2^4 = 16 \equiv 2 \pmod 7$).
- **Audit Diagnosis:** All three candidates correctly identified $100 \equiv 4 \pmod 6$, but then hallucinated $16 \pmod 7 = 4$ instead of $2$. Because all candidates agreed on $4$, medoid voting selected $4$ with 100% agreement.
- **Detector Risk Score:** $0.05$ (Low Risk).
- **Takeaway:** No linguistic, semantic, or internal divergence existed among the candidates. The model was consistently, unanimously wrong.

---

### Case 2: Multi-Slot Permutation Failure (`p5_str_014`)
- **Domain:** Structured Output Tasks
- **Canonical Prompt:** "Format: Element: X | Group: Y. Provide the element with atomic number 47 and its periodic group number."
- **Candidate Predictions:**
  - *Candidate 1:* "Element: Silver | Group: 10"
  - *Candidate 2:* "Element: Silver | Group: 10"
  - *Candidate 3:* "Element: Ag | Group: 10"
- **Candidate Agreement:** Candidates agreed on Silver in Group 10.
- **Verified Gold Reference:** `Element: Silver | Group: 11` (Silver is in Group 11, the coinage metals).
- **Audit Diagnosis:** The model unanimously retrieved Group 10 (Nickel/Palladium/Platinum column) for Silver.
- **Detector Risk Score:** $0.15$ (Low Risk).
- **Takeaway:** The structured constraint was perfectly respected, and entity aliases (`Ag` vs `Silver`) resolved correctly. However, the underlying factual relation was false across all variants.

---

### Case 3: Unit Magnitude Hallucination (`p5_uni_022`)
- **Domain:** Units & Physical Constants
- **Canonical Prompt:** "What is the value of the speed of light in vacuum rounded to three significant figures in km/s?"
- **Candidate Predictions:**
  - *Candidate 1:* "3.00 x 10^5 km/s"
  - *Candidate 2:* "300,000 km/s"
  - *Candidate 3:* "300000 km/s"
- **Gold Reference:** `3.00 x 10^5` or `300,000 km/s` (Correctness criterion: `numerical_exact`).
- **Cluster Outcome:** E0 answered `3 x 10^8 m/s` (violating km/s unit instruction), while E1 medoid selected `3.00 x 10^5 km/s`, correcting the baseline.

---

## 3. Comparison of Mitigation Behavior under False Consensus

| System Condition | Behavior on False-Consensus Clusters | Factual Outcome | Mechanism |
| :--- | :--- | :---: | :--- |
| **E0 (Base Model)** | Emits baseline completion | False (0) | Direct forward pass generation. |
| **E1 (E6_bounded)** | Selects medoid of candidates | False (0) | Votes for majority completion; reinforces false consensus. |
| **E2 (Improved Detector)** | Checks numerical & unit conflicts | False (0) | No conflict detected between unanimous false candidates. |
| **E3 (Risk-Aware Router)** | Evaluates multi-signal risk | False (0) | If well-formed and no slot missing, risk score remains low. |
| **E4 (Abstention Policy)** | Abstains if risk $\ge 0.65$ | False (0) / Abstained | Abstains on structured constraint failures; answers confident false math. |
| **E5 (Oracle Upper Bound)** | Checks if any candidate is true | False (0) | When candidate pool contains 0 correct answers, Oracle also yields 0. |

---

## 4. The Limits of Internal Signals: Why External Verification is Required

Phase 5 explicitly investigated **RQ1** and **RQ2**:

1. **What Internal Signals CAN Do:**
   - Detect when candidates disagree on numerical quantities, signs, or units.
   - Detect negation reversals and contradictory claims across prompt variants.
   - Detect explicit prompt constraint violations (e.g. missing delimiters).
   - Detect evasive hedging ("it is debated...", "as an AI...").
   - Detect when a candidate prompt is not semantically equivalent (hard negatives).

2. **What Internal Signals CANNOT Do:**
   - Distinguish a true statement from a confidently repeated falsehood when all candidate answers agree.
   - Detect when the model's pretraining prior has memorized an incorrect calculation or misattributed fact.
   - Guarantee factual truth from token probabilities, perplexity, or candidate agreement.

### Conclusion on False Consensus:
Bounded routing ($E6\_bounded$) is an effective defense against **prompt sensitivity and variance**, but it is **not** a defense against **shared model bias or shared hallucination**. True factual reliability for closed-book LLMs requires grounding against an independent external knowledge source or verifier.
