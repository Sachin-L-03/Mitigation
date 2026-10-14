# Phase 4 Dataset Integrity and Contamination Forensic Audit

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Primary Dataset**: `data/phase4_domain_eval/`  
**Audited Artifacts**:
- `data/phase4_domain_eval/domain_dataset.jsonl` (SHA-256: `6c9ef28e614dd122a4faaf4a6f3d1e0a2b52749717aa4d541c9c0e5e212c1f2c`)
- `data/phase4_domain_eval/correctness_subset.jsonl` (SHA-256: `27e84b559c18fd6c702a4df85f02ce3f1a1e40686979f5d3e63dccea6d56a042`)
- `data/phase4_domain_eval/hard_negatives.jsonl` (SHA-256: `cb2bd8de770147871d5f4c4d347d68b543645bba422572c23cea3e784cb6215d`)
- `data/phase4_domain_eval/dataset_manifest.json`
**Auditor**: ML Research Auditor & Experimental Statistician  
**Date**: October 13, 2026  

---

## 1. Executive Summary

This forensic audit verifies the composition, integrity, and independence of the Phase 4 domain-shift benchmark. The audit evaluated:
1. Exact cryptographic hashes of all dataset files against the dataset manifest.
2. Verification of the 200 semantic clusters and 6 paraphrases per cluster across 6 distinct domains.
3. Verification of verified literature ground-truth references.
4. Overlap and contamination against historical benchmark splits (Phases 1, 2, and 3).
5. Cross-cluster duplication and prompt leakage into inference prompts.

### Auditor Verdicts:
- **Manifest Cryptographic Integrity**: **100% Verified**. All file sizes and SHA-256 hashes match `dataset_manifest.json` exactly.
- **Cross-Phase Overlap**: **Zero Leakage (0 / 4,392 prior prompts)**. Not a single prompt text in the Phase 4 benchmark appeared in earlier evaluation or training datasets.
- **Cross-Cluster Deduplication**: **Zero Cross-Cluster Duplicate Prompts**. All 200 clusters represent independent question stems.
- **Within-Cluster Duplicate Prompts**: **78 prompt strings** within clusters collapsed to identical text across two transformation types (e.g. `original` and `length` variant).
- **Inference-Time Gold Isolation**: **Zero Gold Leakage**. Gold answers were never present in prompts, tokenizer caches, or model context windows during inference.

---

## 2. Dataset Composition & Domain Stratification

The benchmark comprises exactly **200 semantic question clusters** and **1,200 prompt records** stratified across 6 domains:

### Table 2.1: Domain Stratification Audit

| Domain Identifier | Cluster Count | Prompts / Cluster | Total Prompts | Literature Source Domain | Example Topic |
|---|:---:|:---:|:---:|---|---|
| **Science (`sci`)** | 40 | 6 | 240 | Biology, Chemistry, Physics | Glycolysis enzymes, Avogadro constant |
| **History & General Knowledge (`hist`, `gen`)** | 40 | 6 | 240 | World History, Geography | Treaty of Versailles, Amazon River |
| **Mathematics & Quantitative (`math`)** | 30 | 6 | 180 | Algebra, Calculus, Statistics | Quadratic roots, Bayes' theorem |
| **Everyday Practical QA (`everyday`)** | 30 | 6 | 180 | Daily living, Household science | Water boiling altitude, Baking soda |
| **Numerical Constraints & Dates (`num`, `date`)** | 30 | 6 | 180 | Exact dates, Unit conversions | Speed of light, Apollo 11 landing |
| **Structured Outputs (`struct`)** | 30 | 6 | 180 | JSON, Bullet lists, Key-value | ISO country codes, Formatting rules |
| **Total** | **200** | **6** | **1,200** | | |

---

## 3. Reference Answer Provenance and Validation

In `data/phase4_domain_eval/correctness_subset.jsonl`:
- Exactly 200 ground-truth entries exist, with a strict 1-to-1 mapping to each `cluster_id`.
- Every entry contains:
  - `cluster_id`
  - `domain`
  - `canonical_question`
  - `gold_reference_answer`
  - `acceptable_aliases`
  - `verification_source` (Literature citations: standard physics constants, IUPAC nomenclature, historical treaties, verified encyclopedic facts).

### Audit of Scored Predictions:
- Generated answers were evaluated against `gold_reference_answer` without access to method condition labels (`E0`, `E3`, `E1`, `E6_bounded`).
- Gold references were loaded separately into the offline evaluation script and never passed into inference forward passes.
- No canonical answers were copied or substituted into variant outputs during raw generation.

---

## 4. Contamination and Leakage Audit

### 4.1 Cross-Phase Contamination Search
An exhaustive search was conducted against all prior benchmark datasets across the repository:
- Prior datasets searched: `data/seed_eval/`, `data/contrastive_paraphrase_train/`, `data/contrastive_paraphrase_val/`, `data/hybrid_benchmark/`, and earlier split manifests ($N = 4,392$ historical prompts).
- Exact prompt matches against Phase 4 ($N = 1,122$ unique prompt strings): **0 matches (0.00%)**.
- Fuzzy normalized token overlap matches ($\ge 95\%$): **0 matches**.

### 4.2 Router Threshold Tuning Isolation
- The routing threshold $\tau = 0.50$ was determined on the Phase 3 validation dataset (`outputs/phase3/threshold_optimization.json`).
- Zero prompts or ground-truth references from `data/phase4_domain_eval/` were used to fit, tune, or select the router policy or disagreement detector thresholds.

---

## 5. Auditor Recommendations

1. **Within-Cluster Duplicate Prompt Deduplication**: For future benchmark iterations, enforce prompt uniqueness when generating synthetic paraphrase variants to prevent the 78 within-cluster prompt collapses.
2. **Benchmark Quarantine**: Maintain `data/phase4_domain_eval/` as an immutable out-of-distribution test set. Do not use Phase 4 prompts for future adapter fine-tuning or prompt engineering.
