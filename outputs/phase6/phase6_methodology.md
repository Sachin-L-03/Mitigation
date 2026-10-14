# Phase 6: Methodology — Evidence-Grounded Factual Verification & False-Consensus Mitigation

**Methodology Document:** `PHASE6-METH-001`  
**Author:** ML Research Engineer & Systems Verification Specialist  
**Status:** FROZEN  

---

## 1. Motivation and Problem Formulation

In Phase 5, the factual-risk-aware bounded router (E3) achieved strong paraphrase consistency ($S_{\text{cons}} = 100\%$ pairwise Token F1) and established statistical non-inferiority against the base Qwen model ($\Delta_{\text{NI}} = -0.58\text{ pp}, \text{CI}_{97.5,\text{lower}} = -1.27\text{ pp} > -2.00\text{ pp}$).

However, forensic analysis revealed two fundamental limitations inherent in purely internal consistency methods:
1. **Unanimous False Consensus:** When an LLM harbors systematic parametric training biases, all generated paraphrase candidates frequently agree on the **same incorrect factual claim** (e.g., modular arithmetic $2^{100}\pmod 7$, subtle dates, or common misconceptions). Internal consensus mechanisms (medoid selection, majority voting) enthusiastically accept this unanimous false consensus because internal agreement is 100%.
2. **Formalizable Math and Constraint Inaccuracy:** Autoregressive language models inherently struggle with exact multi-step arithmetic, modular arithmetic, exact unit transformations, and strict syntactic constraints, producing confident but factually invalid results.

Phase 6 resolves these limitations without degrading paraphrase consistency by introducing a modular, gold-free verification layer comprising three complementary paths:
- **Path A (Deterministic Computation and Validation):** For formalizable arithmetic, modular arithmetic, percentages, unit conversions, date math, and structured-output constraints.
- **Path B (Independent Evidence Retrieval):** For external factual queries, retrieving authoritative passages from a local SQLite FTS5 database with BM25 ranking.
- **Path C (Evidence-Aware Answer Verification):** Harmonizes candidate answers with retrieved and calculated evidence, classifying outcomes into `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `PARTIALLY_SUPPORTED`, `AMBIGUOUS_QUESTION`, or `VERIFICATION_ERROR`.

---

## 2. Architectural Design of the Three Verification Paths

```
                             User Prompt Variant
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │  Base Model Candidate Generation     │
                   │           (K=3 Candidates)           │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │   E3 Risk Detector & Medoid Selector │
                   │  (Computes Risk Score, Semantic Slot)│
                   └──────────────────┬───────────────────┘
                                      │
                         Tentative Candidate Answer
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
   ┌──────────────────────┐                       ┌──────────────────────┐
   │        Path A        │                       │        Path B        │
   │    Deterministic     │                       │  Independent Evidence│
   │  Computation Engine  │                       │   Retrieval (FTS5)   │
   │ (Modular, AST, Unit, │                       │  (Top-K Passages,    │
   │  Date, Constraints)  │                       │   Conflict Detect)   │
   └──────────┬───────────┘                       └───────────┬──────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │                Path C                │
                   │       Evidence-Aware Verifier        │
                   │  - Predicate Extraction              │
                   │  - Contradiction vs Support Check    │
                   │  - Principled Evidence Correction    │
                   │  - Conservative Fallback Policy      │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                            Final Verified Output
```

### 2.1 Path A — Deterministic Computation and Validation
Implemented in `src/mitigation/deterministic_verifier.py`:
- **Eligibility Classifier:** Identifies whether the input prompt can be unambiguously formalized into a deterministic operation.
- **Safe AST Evaluator (`SafeASTEvaluator`):** Whitelists basic binary and unary arithmetic operators (`Add`, `Sub`, `Mult`, `Div`, `FloorDiv`, `Mod`, `Pow`, `USub`, `UAdd`). Strictly forbids `Call`, `Import`, `Attribute`, `Exec`, and `Eval` nodes to prevent arbitrary code execution vulnerabilities.
- **Modular Exponentiation:** Evaluates expressions of form $b^e \pmod m$ using Python's native `pow(base, exp, mod)`, computing astronomical powers (e.g. $2^{100}\pmod 7$) in microseconds with $100\%$ precision.
- **Unit Conversions:** Implements exact conversion formulas across Temperature ($^\circ\text{C} \leftrightarrow \text{K}, ^\circ\text{F} \leftrightarrow ^\circ\text{C}$), Speed ($\text{mph} \leftrightarrow \text{km/h}, \text{km/h} \leftrightarrow \text{m/s}$), Distance ($\text{miles} \leftrightarrow \text{km}, \text{inches} \leftrightarrow \text{cm}$), Area ($\text{acres} \leftrightarrow \text{sq ft}$), Volume ($\text{gallons} \leftrightarrow \text{L}, \text{pints} \leftrightarrow \text{fl oz}$), and Mass ($\text{lbs} \leftrightarrow \text{kg}$).
- **Date Duration:** Computes exact elapsed calendar years $|Y_2 - Y_1|$.
- **Structured Output Validation:** Validates delimiter requirements (e.g. `Item / Category`), sorting constraints (alphabetical ordering), and slot structure.

### 2.2 Path B — Independent Evidence Retrieval
Implemented in `src/mitigation/evidence_retriever.py`:
- **Authoritative Evidence Index (`evidence_index.db`):** SQLite FTS5 database indexing verified reference corpora with Porter stemmer and BM25 ranking.
- **Provenance Logging:** Every retrieved passage records:
  - `source_id`: Unique identifier (e.g. `doc_sci_001`).
  - `title`: Document title.
  - `document_path`: Verified filesystem path in `data/phase6_eval/evidence_corpus/docs/`.
  - `content`: Authentic factual passage.
  - `provenance`: Curated authority (BIPM, NIST, IUPAC, CODATA, Encyclopaedia Britannica, PDG, NASA).
  - `retrieval_timestamp`: ISO 8601 UTC timestamp.
  - `score`: BM25 relevance score.
- **Bounded Retrieval Budget:** Top-$K$ passages (default $K_{\text{pass}} = 3$) to maintain latency and token constraints.
- **Source Conflict Detection:** Evaluates whether retrieved passages cite conflicting numeric quantities or contradictory relations for the queried entity.

### 2.3 Path C — Evidence-Aware Answer Verification
Implemented in `src/mitigation/evidence_verifier.py`:
- **State Classification:**
  - `SUPPORTED`: Direct or predicate support in retrieved evidence or deterministic computation.
  - `CONTRADICTED`: Evidence directly refutes the candidate answer with an authoritative alternative.
  - `PARTIALLY_SUPPORTED`: Evidence confirms some slots but lacks confirmation on others.
  - `INSUFFICIENT_EVIDENCE`: Retrieved passages are uninformative or no relevant documents were retrieved.
  - `AMBIGUOUS_QUESTION`: Question is inherently ambiguous with multiple valid interpretations.
  - `VERIFICATION_ERROR`: Tool failure or runtime error.
- **Correction Policy:** An answer is corrected **only** when reliable evidence or deterministic computation explicitly contradicts the candidate AND provides an unambiguous target value.
- **Conservative Fallback Policy:** Under condition E5, cases flagged with High Factual Risk that lack verified evidence emit an explicit abstention (`[UNCERTAINTY_FALLBACK]`).

---

## 3. Experimental Conditions

1. **E0 — Base Model:** Single greedy completion per prompt ($K=1$). Baseline model performance.
2. **E1 — E3 Risk-Aware Router:** Consistency baseline from Phase 5 ($K=3$, medoid selection, slot check, no tools).
3. **E2 — E3 + Deterministic Verification:** E1 followed by Path A for eligible numerical/math/unit/date/constraint queries.
4. **E3 — E3 + External Evidence Retrieval:** E1 followed by Path B for factual queries requiring external knowledge.
5. **E4 — Combined Evidence-Grounded Router:** Primary candidate: Path A first; if not eligible, Path B retrieval; evidence-aware verifier.
6. **E5 — Conservative Evidence-Aware Fallback:** E4 with explicit abstention on unresolved high-risk claims.
7. **E6 — Offline Oracle Diagnostic:** Offline upper bound using gold reference labels strictly for diagnostic benchmarking.

---

## 4. Preservation of Consistency Baseline

The E3 router logic is preserved identically to Phase 5:
- Same risk detector (`FactualRiskDetector`)
- Same semantic comparator (`EnhancedSemanticComparator`)
- Same risk scoring and thresholding ($\tau_{\text{high}} = 0.70, \tau_{\text{mod}} = 0.40$)
- Zero modification to historical Phase 1–5 artifacts.
