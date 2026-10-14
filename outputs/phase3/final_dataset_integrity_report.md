# Final Dataset Integrity and Cryptographic Split Discipline Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Target Directory**: `data/phase3_final_eval/`  
**Date**: October 12, 2026  
**Auditor**: Senior ML Research Engineer  

---

## 1. Executive Summary

To establish whether the E6 adaptive router and inference-time mitigation techniques generalize to genuinely unseen prompts without any risk of test-set tuning or leakage, an untouched final evaluation benchmark was created under `data/phase3_final_eval/`.

Because Phase 2 evaluation data (`data/phase2_external_eval/`) was used in Phase 3 to audit failure cases and optimize routing thresholds ($\tau = 0.50$, $K = 3$), it has been formally reclassified as **development/validation data**.

The Phase 3 benchmark contains **100 brand-new independent semantic clusters (600 prompt records)** and a **50-cluster factual correctness subset (300 prompt records)** with verified ground-truth reference answers and explicit academic provenance.

A comprehensive cross-dataset duplicate audit was conducted against all **2,376 pre-existing prompt records**:
1. `data/paraphrased_prompts_axis_tagged_qa.csv` (1,176 records)
2. `data/external_eval/external_dataset.jsonl` (600 records)
3. `data/phase2_external_eval/external_dataset.jsonl` (600 records)

### Audit Outcome:
- **Exact Normalized Matches**: **0 (0.00%)**
- **Near-Duplicate Matches (Jaccard > 0.85)**: **0 (0.00%)**
- **Zero-Overlap Verification**: **PASSED (`True`)**

---

## 2. Dataset Architecture and Linguistic Variations

Each semantic cluster contains 6 systematically controlled variations of the same underlying query across 6 linguistic axes:

| Linguistic Axis | Transformation Description | Example Transformation (Cluster P3-001) |
|---|---|---|
| `original` | Neutral, direct canonical question | "What is the boundary around a black hole beyond which no matter or radiation can escape?" |
| `lexical` | Domain-appropriate synonym substitutions | "Which perimeter surrounding a black hole marks the threshold where escape velocity exceeds light speed?" |
| `syntactic` | Passive/active inversion, cleft sentence | "In general relativity, by what name is the boundary of a black hole known from which no radiation can escape?" |
| `pragmatic` | Persona / conversational framing | "As a gravitational physicist, identify the defining boundary surface of a black hole." |
| `length` | Detailed background context elaboration | "Formulated in Schwarzschild geometry as the radius where spacetime curvature prevents any outward null geodesics, what is the formal name of the causal boundary enclosing a black hole singularity?" |
| `format` | Strict structured schema / output constraint | "Relativistic Astrophysics:\nObject: Black hole causal boundary\nBoundary name:" |

---

## 3. Curated Scientific Domains and Provenance

The 100 clusters cover advanced scientific and technical domains:
- **Astrophysics & Relativistic Mechanics**: Event horizon, Oort cloud, James Webb Space Telescope L2 point, Pulsars.
- **Molecular Genetics & Cell Biology**: Rough ER, CRISPR-Cas9, DNA polymerase III, Centrosomes.
- **Quantum & Condensed Matter Physics**: Pauli exclusion principle, Photoelectric effect, Superconductivity.
- **Cardiology & Hematology**: Statins (HMG-CoA reductase), AV node, Hemoglobin, Warfarin.
- **Computer Architecture & Systems**: Amdahl's law, Virtual memory page faults, QuickSort, Hash collisions, TCP.
- **Thermodynamics & Physical Chemistry**: Le Chatelier's principle, Triple point, Enthalpy, Adiabatic processes.

---

## 4. Cryptographic Manifest and Dataset Freeze

The benchmark files have been cryptographically hashed and frozen in `data/phase3_final_eval/dataset_manifest.json`:

```json
{
  "benchmark": "Phase 3 Final Untouched Generalization Test",
  "created_at": "2026-10-12T22:50:00Z",
  "total_clusters": 100,
  "total_prompt_records": 600,
  "factual_clusters_count": 50,
  "axes": [
    "original",
    "lexical",
    "syntactic",
    "pragmatic",
    "length",
    "format"
  ],
  "files": {
    "external_dataset_jsonl": {
      "filename": "external_dataset.jsonl",
      "sha256": "426be23df949e295e86971167732d84713c72b226eec102ffeb1b12b557876a3",
      "records": 600
    },
    "correctness_subset_jsonl": {
      "filename": "correctness_subset.jsonl",
      "sha256": "921867c8052145cfbe094f6f7091fe21199a0937a44f932bf172771cb87d605c",
      "clusters": 50
    }
  },
  "overlap_audit": {
    "total_new_records": 600,
    "total_existing_records_audited": 2376,
    "exact_matches": 0,
    "near_matches": 0,
    "zero_overlap_verified": true
  },
  "frozen": true
}
```

This dataset is locked as the final untouched evaluation test set.
