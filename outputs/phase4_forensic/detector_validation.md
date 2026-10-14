# Phase 4 Disagreement Detector Forensic Validation

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Audit Target**: 25-Case Paraphrase Robustness Stress Test  
**Source Data**: [`outputs/phase4/paraphrase_stress_test.csv`](file:///outputs/phase4/paraphrase_stress_test.csv) and [`outputs/phase4/paraphrase_stress_test.md`](file:///outputs/phase4/paraphrase_stress_test.md)  
**Auditor**: ML Research Auditor & Experimental Statistician  
**Date**: October 13, 2026  

---

## 1. Executive Summary

Phase 4 evaluated the `E6_bounded` disagreement detector across 25 edge cases spanning lexical paraphrasing, syntactic transformation, entity aliases, numeric conflicts, negation reversals, and hard negatives. This audit recomputes the confusion matrix, verifies metric consistency, and exposes a critical discrepancy between the raw CSV data and the published report narrative.

### Headline Audit Findings:
1. **Mathematical Consistency**: The published conflict detection figures (**80.0% Accuracy**, **77.8% Precision**, **70.0% Recall**, **0.737 F1**) are mathematically consistent with the 25 rows in `paraphrase_stress_test.csv`.
2. **Major Narrative Discrepancy Found**:
   - In Section 3 of `outputs/phase4/paraphrase_stress_test.md`, the author claimed:
     > *"Alias Handling (STRESS_04, STRESS_19): The domain alias map successfully resolves ATP <-> adenosine triphosphate and BGP <-> Border Gateway Protocol, keeping these queries on the fast path without unnecessary generation."*
   - **The Raw Data Directly Contradicts This Claim**:
     - `STRESS_04`: `expected_conflict = False`, `detected_conflict = True`, `route_assigned = VERIFICATION_K3`, `verdict = FALSE_POSITIVE`.
     - `STRESS_19`: `expected_conflict = False`, `detected_conflict = True`, `route_assigned = VERIFICATION_K3`, `verdict = FALSE_POSITIVE`.
     - Both cases **FAILED** the detector, triggered false-positive entity mismatches, and were dispatched to `VERIFICATION_K3`. The written report claimed the exact opposite of what the CSV recorded.
3. **Route-Level vs Conflict-Level Mismatches**:
   - In 8 cases (`STRESS_02, STRESS_03, STRESS_13, STRESS_14, STRESS_18, STRESS_20, STRESS_22, STRESS_23`), `detected_conflict = False`, but because Semantic Similarity was $< 0.50$, the router dispatched them to `VERIFICATION_K3`. The report labelled these verdicts `CORRECT_FAST_PATH` despite their being routed to verification.

---

## 2. Recomputed Confusion Matrix and Verification

### Table 2.1: 25-Case Conflict Detector Confusion Matrix

| | Actually No Conflict (`expected = False`) | Actually Conflict (`expected = True`) | Total |
|---|:---:|:---:|:---:|
| **Predicted No Conflict** (`detected = False`) | **TN = 13** | **FN = 3** | 16 |
| **Predicted Conflict** (`detected = True`) | **FP = 2** | **TP = 7** | 9 |
| **Total** | **15** | **10** | **25** |

### Table 2.2: Recomputed Metrics vs Published Report

| Metric | Formula | Recomputed Exact | Published Report | Discrepancy | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{7 + 13}{25}$ | **80.00%** | 80.0% | 0.00 pp | **Verified** |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{7}{7 + 2}$ | **77.78%** | 78% | -0.22 pp (rounding) | **Verified** |
| **Recall** | $\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{7}{7 + 3}$ | **70.00%** | 70% | 0.00 pp | **Verified** |
| **F1 Score** | $\frac{2 \times P \times R}{P + R} = \frac{2 \times 0.7778 \times 0.70}{1.4778}$ | **0.7368** | 0.74 (or 0.737) | 0.000 | **Verified** |
| **False Positive Rate** | $\frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{2}{15}$ | **13.33%** | Not reported | N/A | **New Metric** |
| **False Negative Rate** | $\frac{\text{FN}}{\text{TP} + \text{FN}} = \frac{3}{10}$ | **30.00%** | Not reported | N/A | **New Metric** |

---

## 3. Failure Mode Categorization

### 3.1 False Negatives (3 Cases, Missed Contradictions)
When the detector fails to flag a real contradiction, the system risks emitting an unverified, conflicting answer:
1. **`STRESS_11` (Hard Negative Constraint Count)**:
   - Question asks for 3 items vs 5 items.
   - Failure Cause: Answers share words, achieving Token F1 = 0.74. The regex numerical detector looks for standalone numbers, but does not parse grammatical item counts in lists.
2. **`STRESS_17` (Multi-Part Slot Disagreement)**:
   - Prompt: "What is the capital and currency of Australia?"
   - Candidate A: "Canberra and Australian Dollar".
   - Candidate B: "Sydney and Australian Dollar".
   - Failure Cause: The shared currency ("Australian Dollar") elevated Token F1 to 0.90, blinding the detector to the capital city contradiction ("Canberra" vs "Sydney").
3. **`STRESS_25` (Mathematical Sign Inversion)**:
   - Candidate A: "The roots are x = +4".
   - Candidate B: "The roots are x = -4".
   - Failure Cause: Stripping punctuation removed the minus sign (`-`), converting `-4` to `4` and producing a false numerical match.

### 3.2 False Positives (2 Cases, Unnecessary Verification Invocations)
When the detector hallucinates a contradiction, the system wastes compute running bounded verification on identical factual answers:
1. **`STRESS_04` (Entity Alias Equivalence)**:
   - Candidate A: "ATP provides cellular energy".
   - Candidate B: "Adenosine triphosphate drives metabolic reactions".
   - Failure Cause: Capitalized regex entity extraction extracted `ATP` from A, but lowercase normalization in B failed to match the expanded alias set before set intersection.
2. **`STRESS_19` (Protocol Abbreviation Equivalence)**:
   - Candidate A: "BGP manages routing between autonomous systems".
   - Candidate B: "Border Gateway Protocol connects Internet domains".
   - Failure Cause: Exact substring alias replacement failed due to title casing differences.

---

## 4. Diagnostic Challenge Set Construction

To validate fixes for the audited edge cases without altering the untouched Phase 4 primary evaluation set, a 10-case diagnostic challenge set was constructed at `data/phase4_forensic_challenge/diagnostic_cases.jsonl`:
- **Diagnostics 01-03**: Mathematical sign errors (`+` vs `-`, negative exponents).
- **Diagnostics 04-06**: Multi-slot questions with 1 agreeing slot and 1 conflicting slot.
- **Diagnostics 07-08**: List count constraints (requesting 3 items vs requesting 5 items).
- **Diagnostics 09-10**: Multi-word biological and networking acronym aliases with case variations.

---

## 5. Auditor Recommendation on Disagreement Detection

1. **Statistical Power**: A 25-case stress test has wide binomial confidence intervals:
   - 80.0% accuracy on $N=25$ yields a 95% Clopper-Pearson CI of **[59.3%, 93.2%]**.
   - A detector with true accuracy of 65% could easily produce 20/25 correct cases by chance ($p \approx 0.08$).
   - The report's confident assertions of detector robustness are over-generalized from a tiny diagnostic sample.
2. **Regex Limitations**: Pure regex-based entity and numerical extraction is fragile against mathematical signs, multi-slot answers, and prompt constraint changes. Production deployment requires embedding-based NLI or fine-tuned cross-encoder verification.
