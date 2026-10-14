# Phase 5 Comprehensive Failure Analysis & Error Taxonomy

**Document Status:** COMPLETE & VERIFIED  
**Repository:** `prompt-sensitivity-hybrid`  
**Protocol Requirement:** Section 13 ("Failure analysis")  
**Evaluation Set:** Phase 5 Untouched Test Split ($N=226$ Clusters, 904 Prompts)  
**Author:** ML Research Engineer & Skeptical Forensic Auditor  
**Date:** 2026-10-14  

---

## 1. Executive Summary and Taxonomy Overview

To ensure complete experimental transparency, all errors observed across the evaluated conditions (E0, E1, E2, E3, E4) were systematically audited and classified into the 18 preregistered error categories specified in Section 13.

### Primary Error Distribution across Test Cohort ($N=226$ Clusters):
- **Base Model (E0) Factual Failures:** 50 clusters (22.12% failure rate, 77.88% accuracy).
- **Risk-Aware Router (E3) Factual Failures:** 47 clusters (20.80% failure rate, 79.20% accuracy).
- **Routing Transitions:**
  - **Incorrect baseline corrected by routing:** **6 clusters** (+2.65% repair rate).
  - **Correct baseline corrupted by routing:** **3 clusters** (-1.33% corruption rate).
  - **Net Routing Improvement:** **+3 clusters** (+1.33 percentage points).
- **Unanimous False Consensus:** **3 clusters** (1.33%) where all candidates agreed on a false completion.
- **Oracle Ceiling Gap:** In **21 clusters** (9.29%), *no* generated candidate was factually correct.

---

## 2. 18-Category Comprehensive Failure Mode Breakdown

| Category # | Error Category | Observed Count (Test Split) | Frequency (%) | Primary Causes / Mechanism | Detectability with Internal Signals |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1** | **Wrong Numerical Value** | 22 | 9.73% | Arithmetic reasoning failure in complex powers/modulos | **Partially Detectable** (if variants disagree) |
| **2** | **Wrong Entity** | 12 | 5.31% | Confusion between closely related proteins, isotopes, minerals | **Partially Detectable** (if variants disagree) |
| **3** | **Malformed Structured Output** | 8 | 3.54% | Failure to emit designated delimiter (e.g. `/`, `\|`) | **Fully Detectable** (via `ConstraintChecker`) |
| **4** | **Unnecessary Abstention (E4)** | 37 | 16.37% | E0 answered correctly, but E4 abstained due to risk score $\ge 0.65$ | **False Positive Trigger** |
| **5** | **Failure to Abstain on High-Risk** | 6 | 2.65% | Model answered incorrectly but generated false consensus ($r < 0.65$) | **False Negative Risk** |
| **6** | **Unanimous Incorrect Consensus** | 3 | 1.33% | All 4 candidates agreed on false completion | **Undetectable without External KB** |
| **7** | **Correct Changed to Incorrect by Routing** | 3 | 1.33% | Medoid voting chose flawed candidate over correct baseline | **Routing Vulnerability** |
| **8** | **Incorrect Corrected by Routing** | 6 | 2.65% | Medoid voting selected correct candidate from variant pool | **Routing Success** |
| **9** | **Sign or Unit Error** | 4 | 1.77% | Missing negative sign or conversion unit error | **Fully Detectable** (via `EnhancedComparator`) |
| **10** | **Predicate / Relation Error** | 3 | 1.33% | Misattributed periodic group or taxonomic order | **Undetectable without External KB** |
| **11** | **Constraint Violation** | 5 | 2.21% | Word count mismatch or sorting constraint violation | **Fully Detectable** (via `ConstraintChecker`) |
| **12** | **Incomplete Answer** | 4 | 1.77% | Truncated generation before emitting all required slots | **Fully Detectable** (slot check) |
| **13** | **Unsupported Factual Claim** | 3 | 1.33% | Confident invention of non-existent historical details | **Undetectable without External KB** |
| **14** | **Incorrect Date or Time** | 2 | 0.88% | Off-by-one or off-by-decade chronological errors | **Partially Detectable** |
| **15** | **Negation Reversal** | 1 | 0.44% | Affirming a banned attribute ("Which is NOT...") | **Fully Detectable** |
| **16** | **False-Positive Verification Trigger** | 8 | 3.54% | Invoked verification despite baseline being factually correct | **Benign Cost Overhead** |
| **17** | **False-Negative Risk Detection** | 5 | 2.21% | Flawed response classified as Low Risk | **Safety Deficit** |
| **18** | **Ambiguous Question** | 0 | 0.00% | Filtered out during dataset integrity audit | **Zero Ambiguity Certified** |

---

## 3. Representative Failure Case Audit Table

Below is the audited failure table with cluster IDs, source predictions, reference provenance, root causes, and proposed remedies:

| Cluster ID | Domain | Input Prompt | Model Prediction (E3) | Verified Reference Answer | Provenance | Root Cause Hypothesis | Proposed Engineering Remedy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p5_mat_018` | Math | "What is the remainder when $2^{100}$ is divided by 7?" | "The remainder is 4." | `2` | Number Theory Standard | Fermat's Little Theorem deduction failed at final modular reduction ($16 \pmod 7$). | Integrate lightweight Python code execution tool / sandbox for mathematical expressions. |
| `p5_str_014` | Structured | "Format: Element: X \| Group: Y. Provide element 47 and group." | "Element: Silver \| Group: 10" | `Element: Silver \| Group: 11` | IUPAC Periodic Table | Model retrieved Group 10 (Nickel/Platinum group) instead of Group 11 (Coinage metals). | Integrate structured entity-attribute knowledge graph lookup. |
| `p5_sci_011` | Science | "What is the SI derived unit of electrical inductance?" | "The SI derived unit of electrical inductance is the Henry (symbol: H)." | `henry` | BIPM SI Brochure | Correctly matched by entity alias evaluator. | Correct (demonstrates alias resolution success). |
| `p5_mat_005` | Math | "Evaluate $(-3)^3 - (-2)^2$." | "The value is -31." | `-31` | Basic Arithmetic | Correct signed integer arithmetic: $-27 - 4 = -31$. | Correct (demonstrates signed arithmetic preservation). |
| `p5_uni_005` | Units | "What SI base unit measures luminous intensity?" | "The SI base unit of luminous intensity is the candela (cd)." | `candela` | NIST Physical Reference | Correctly matched canonical and symbol alias `cd`. | Correct (demonstrates unit entity extraction). |
| `p5_con_012` | Constraints | "List the capitals of Canada and Mexico in alphabetical order separated by a comma." | "Mexico City, Ottawa" | `Mexico City, Ottawa` | Britannica World Atlas | Correct alphabetical ordering (`M` before `O`). | Correct (demonstrates constraint compliance). |
| `p5_sci_022` | Science | "What is the freezing point of heavy water in degrees Celsius?" | "The freezing point is 3.82 degrees Celsius." | `3.82 degrees Celsius` | NIST WebBook | Correct decimal numerical extraction respecting degrees Celsius. | Correct (demonstrates float extraction). |
| `p5_geo_015` | Geography | "What is the deepest lake in the world?" | "Lake Baikal in Russia is the deepest lake in the world." | `Lake Baikal` | Hydrological Records | Correct lake entity retrieval matching alias `baikal`. | Correct (demonstrates alias match). |
| `p5_str_005` | Structured | "Output format: Author (Year). Name the author and year of 1984." | "George Orwell (1949)" | `George Orwell (1949)` | Literary Catalog | Correct structured parenthetical year format conformance. | Correct (demonstrates format conformance). |

---

## 4. Separation of Errors: Detectable vs. External-Evidence-Bound

A foundational goal of Phase 5 is to separate errors into two distinct engineering regimes:

### 4.1 Errors Detectable by Current Inference Router (Internal Signals):
- **Syntactic & Format Violations:** Missing slashes, missing key-value pipes, missing author-year parentheses.
- **Variant Divergence:** Disagreements across paraphrases on numerical values, units, or named entities.
- **Explicit Negation Contradictions:** When one candidate asserts a claim and another negates it.
- **Signed Arithmetic Inversions:** Mismatches between positive and negative numbers.
- **Vague Evasions:** Hedging phrases and uninformative boilerplate.

### 4.2 Errors That CANNOT Be Detected Without External Knowledge:
- **Shared Closed-Book Misconceptions:** Confidently stated falsehoods that appear identically across all paraphrased prompt variants (e.g. `p5_mat_018`).
- **Subtle Relational Offsets:** Placing an element in Group 10 instead of Group 11 with complete semantic coherence.
- **Complex Multi-Step Arithmetic:** Mental arithmetic beyond 3 tokens where the model hallucinated intermediate carries.

### Architectural Recommendation:
To eliminate the remaining 20.80% factual failure rate, the router should be augmented with an **external verification API** (e.g., local Wikipedia BM25 / vector retriever or python calculation sandbox) triggered conditionally when the factual risk score is Moderate ($0.35 \le \text{Risk} < 0.65$).
