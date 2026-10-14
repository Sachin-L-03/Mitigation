# Comprehensive Failure Analysis: Phase 4 Domain-Shift Benchmark

**Repository**: `prompt-sensitivity-hybrid`  
**Active Branch**: `creme-hybrid-framework`  
**Evaluation Scope**: 200 Semantic Question Clusters (1,200 Prompts)  
**Date**: October 12, 2026  
**Auditor**: Senior ML Research Engineer (Autonomous Phase 4 Protocol)  

---

## 1. Failure Taxonomy and Empirical Distribution

Out of 200 evaluated clusters on the out-of-distribution benchmark, failures and suboptimal routing decisions were cataloged into 6 categories (recorded in [`outputs/phase4/failure_cases.csv`](file:///outputs/phase4/failure_cases.csv)):

| Category | Cases Observed | Frequency (% of 200 Clusters) | Severity | Primary Detectability | Remediation Status |
|---|:---:|:---:|---|:---:|---|
| **1. Unanimous Hallucinations** | 8 | 4.0% | High | Undetectable at inference | Consensus cannot resolve universal base model blindspots; requires external retrieval (RAG). |
| **2. Unnecessary Verification (False Positives)** | 12 | 6.0% | Low | Detectable (Formatting) | Solved by containment checks for telegraphic queries. |
| **3. Numerical Precision & Rounding Drift** | 9 | 4.5% | Medium | Detectable (Numbers) | Solved by 10% approximate tolerance when lexical qualifiers (`approx`, `about`) are present. |
| **4. Delimiter & Punctuation Inconsistencies** | 11 | 5.5% | Low | Detectable (Tokens) | Solved by delimiter normalization before computing token F1. |
| **5. Suboptimal Medoid Selection** | 6 | 3.0% | Medium | Detectable (Similarity) | Bounded medoid occasionally selects slightly more verbose answer over concise canonical. |
| **6. Paraphrase Constraint Drift** | 4 | 2.0% | Medium | Detectable (Length/Tokens) | Bounded candidate pool ($K=3$) restricts paraphrase generation to high-fidelity transformations. |

---

## 2. Deep Dive into Critical Failure Modes

### 2.1 Unanimous Hallucinations (E.g. `sci_017`, `geo_038`)
- **Phenomenon**: On obscure or ambiguous trivia questions (e.g. naming specific dates of minor ancient events), all $K=3$ generations from `Qwen2.5-7B-Instruct` produced the identical incorrect historical attribution.
- **Root Cause**: The underlying base model's internal pretraining weights have a strong prior towards the misconception. Consensus among internal generations reflects the model's coherent hallucination.
- **Conclusion**: Multi-candidate medoid voting can only resolve *inconsistencies*; it cannot inject external factual knowledge not possessed by the base model. For factual reliability on rare entities, inference-time verification must be paired with retrieval-augmented generation (RAG).

### 2.2 Numerical Precision vs Rounding (E.g. `cst_012`, `eve_017`)
- **Phenomenon**: Original answer stated marathon distance as `42.195 km`; variant answer stated `42.2 km` or `42 kilometers`.
- **Root Cause**: Literal set intersection of numbers $\{42.195\} \cap \{42.2\} = \emptyset$ triggered a numerical contradiction.
- **Remediation**: In `EnhancedAnswerComparator`, numbers within 10% relative tolerance are exempted from hard conflict when approximate markers (`about`, `roughly`, `nearly`) appear in the text.

### 2.3 Delimiter and Punctuation Mismatches in Structured Outputs (E.g. `str_002`, `str_006`)
- **Phenomenon**: On structured-output tasks (e.g., "provide format: Symbol: X, Number: Y"), one variant produced `Symbol: Fe | Number: 26` (pipe delimiter) while another produced `Symbol: Fe, Number: 26` (comma delimiter).
- **Impact**: Reduced Token F1 slightly due to token differences, despite perfect semantic alignment.
- **Remediation**: Strip punctuation and common delimiters (`|`, `;`, `,`, `:`) during equivalence normalization.
