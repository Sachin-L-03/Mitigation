# Phase 6: Unanimous False-Consensus Audit and Mitigation Analysis

**Audit Report:** `PHASE6-AUDIT-FC-001`  
**Evaluation Set:** Frozen Untouched Phase 6 Test Set ($N = 450$ clusters)  
**Status:** CONFIRMED & AUDITED  

---

## 1. The Phenomenon of Unanimous False Consensus

A critical vulnerability of inference-time consistency methods (such as self-consistency, majority voting, bounded medoids, and Phase 5's E3 router) is **unanimous false consensus**:
- When a language model possesses systematic training distribution biases or parametric blind spots, it does not generate diverse errors.
- Instead, across all prompt paraphrases, the model confidently and unanimously generates the **exact same factually incorrect claim**.
- Under standard internal consensus rules, unanimous agreement produces a maximum consistency score ($S_{\text{cons}} = 1.0$) and minimum disagreement ($D = 0$). The router happily outputs the hallucination with high confidence.

Phase 6 was specifically designed to evaluate whether external evidence retrieval (Path B) and deterministic computation (Path A) can break this unanimous false consensus.

---

## 2. Quantitative False-Consensus Findings

Across the 450 untouched test clusters:

| Metric | Measured Value | Percentage / Proportion | Analytical Significance |
| :--- | :---: | :---: | :--- |
| **Total Test Clusters** | 450 | 100.0% | Complete independent evaluation cohort |
| **Unanimous False-Consensus Clusters** | **19** | **4.22%** | Base model produced unanimous incorrect consensus |
| **E1 Baseline Correction Count** | **0 / 19** | **0.0%** | Internal consistency router cannot correct false consensus |
| **E2 (Path A) Correction Count** | **19 / 19** | **100.0%** | Deterministic tools broke 100% of arithmetic false consensus |
| **E3 (Path B) Correction Count** | **0 / 19** | **0.0%** | Retrieval did not overlap with purely arithmetic UFC clusters |
| **E4 (Combined) Correction Count** | **19 / 19** | **100.0%** | Complete mitigation of unanimous false consensus |
| **Clusters Corrupted by Verification** | **4** | **0.89%** | Correct answers accidentally altered (corruption rate $< 1\%$) |
| **Net Factual Win / Loss Ratio** | **106 / 4** | **26.5 : 1** | For every 1 corruption, 26.5 errors were corrected |

---

## 3. Case Studies: Anatomy of Unanimous False Consensus

### Case Study 1: Modular Exponentiation Blind Spot
- **Cluster ID:** `p6_s12_diag_001`
- **Canonical Prompt:** *"What is the remainder when 2^100 is divided by 7?"*
- **Paraphrase Variants:**
  - *Lexical:* "Calculate 2^100 modulo 7."
  - *Syntactic:* "When 2^100 is divided by 7, what remainder is obtained?"
  - *Concise:* "2^100 mod 7?"
- **Base Model Generations:**
  - Variant 1: `The remainder is 4.`
  - Variant 2: `4`
  - Variant 3: `It leaves a remainder of 4.`
  - Variant 4: `4.`
- **Internal Consensus Outcome (E1):**
  - Pairwise Token F1: $1.0000$. Disagreement: $0.00$.
  - Medoid Selected: `"4"`.
  - Reference Answer: `"2"` (Proof: $2^6 \equiv 1 \pmod 7 \implies 2^{100} = (2^6)^{16} \cdot 2^4 = 16 \equiv 2$).
  - **E1 Result:** `INCORRECT`. The internal consistency check was completely blind to the hallucination.
- **Evidence-Grounded Outcome (E4):**
  - Path A parsed the expression into deterministic AST: `pow(2, 100, 7)`.
  - Deterministic evaluation returned `2` in $4 \ \mu\text{s}$.
  - Candidate `"4"` was classified as `CONTRADICTED` with $1.0$ confidence.
  - Final Answer: `"2"`.
  - **E4 Result:** `CORRECT` (False consensus resolved).

---

### Case Study 2: Signed Negative Arithmetic and Operator Precedence
- **Cluster ID:** `p6_s01_arith_026`
- **Canonical Prompt:** *"What is (-3)^3 - (-2)^2?"*
- **Paraphrase Variants:**
  - *Lexical:* "Evaluate the mathematical expression (-3)^3 - (-2)^2."
  - *Syntactic:* "Calculate the exact numerical value of (-3)^3 - (-2)^2."
  - *Concise:* "Evaluate (-3)^3 - (-2)^2?"
- **Base Model Generations:**
  - Variant 1: `(-3)^3 - (-2)^2 = -27 - (-4) = -23`
  - Variant 2: `-23`
  - Variant 3: `The result is -23.`
  - Variant 4: `-23`
- **Failure Analysis:** Base Qwen systematically mishandled the negative sign preceding $(-2)^2$, miscalculating $-(-2)^2 = -(-4) = +4$ instead of $-(+4) = -4$.
- **Internal Consensus (E1):** Unanimous agreement on `-23`. Selected: `-23`. **INCORRECT**.
- **Evidence-Grounded Outcome (E4):**
  - Safe AST evaluator parsed node structure: `Sub(Pow(Constant(-3), Constant(3)), Pow(Constant(-2), Constant(2)))`.
  - Evaluated: $-27 - 4 = -31$.
  - Classified candidate as `CONTRADICTED`.
  - Final Answer: `"-31"`.
  - **E4 Result:** `CORRECT`.

---

### Case Study 3: Chronological Year Duration
- **Cluster ID:** `p6_s03_date_001`
- **Canonical Prompt:** *"How many years elapsed between 1776 and 1789?"*
- **Base Model Generations:**
  - All 4 variants unanimously answered: `"15 years"`.
- **Internal Consensus (E1):** Accepted `"15 years"`. **INCORRECT** ($|1789 - 1776| = 13$).
- **Evidence-Grounded Outcome (E4):**
  - Path A parsed year tokens: `|1789 - 1776| = 13`.
  - Final Answer: `"13"`.
  - **E4 Result:** `CORRECT`.

---

## 4. Audit of Corruption Cases ($N = 4$)

Four clusters were corrupted by the verification layer (E1 was correct, but E4 made an error):
1. **Multi-Constraint Prompt `p6_s08_mcon_005` (Grover Cleveland):** The prompt requested the president who served two non-consecutive terms. E1 produced "Grover Cleveland". Path B retrieved a passage that mentioned both "Grover Cleveland" and "Benjamin Harrison" (who served between the two terms). The verifier extracted "Benjamin Harrison" as an entity match, corrupting the answer.
2. **Date Prompt `p6_s04_hist_005` (Fall of Western Rome):** Base model answered "476 AD". The retriever retrieved an analytical essay discussing "Odoacer in 480 AD" as an alternative thesis, causing an unwanted year substitution.
3. **Structured Prompt `p6_s09_stct_002` (Gold):** Base model answered "Au / 79". The verifier flagged a spacing anomaly in one variant.
4. **General Knowledge `p6_s04_hist_003` (WWI Prime Minister):** Candidate answered "H. H. Asquith". Retrieved passage mentioned "David Lloyd George took over in 1916". The verifier misidentified Lloyd George as the answer for 1914.

**Corruption Rate:** $\frac{4}{450} = \mathbf{0.89\%}$.
This corruption rate is well below the preregistered risk tolerance threshold of $2.0\%$, and is vastly outweighed by the **106 genuine error corrections** (26.5 corrections per corruption).
