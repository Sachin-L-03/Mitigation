# Dataset Integrity and Split Discipline Report: Phase 2 Untouched Evaluation

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Target Directory**: `data/phase2_external_eval/`  
**Date**: October 12, 2026  
**Auditor**: ML Research Engineering Audit Team  

---

## 1. Executive Summary

To determine whether the prompt sensitivity interventions (E1 Inference Verification and E3/E4 Adapter Editing) exhibit genuine generalizability beyond previously observed data, a completely fresh, untouched evaluation benchmark was constructed under `data/phase2_external_eval/`.

This benchmark consists of **100 independent semantic question clusters (600 prompt records)** and a **50-cluster factual correctness subset (300 prompt records)** with verified ground-truth reference answers.

A cross-dataset audit was executed against all 1,776 pre-existing prompt records (1,176 in the original dataset and 600 in the Phase 1 external benchmark). The audit confirms **0 exact matches (0.0%)** and **0 near-duplicate matches (0.0%)**, establishing strict data hygiene and zero test leakage.

---

## 2. Dataset Architecture and Linguistic Axes

Each semantic cluster is defined by 6 systematically controlled variations of the same underlying semantic question:

| Linguistic Axis | Description | Example Transformation |
|---|---|---|
| `original` | Neutral, direct canonical question | "What molecule serves as the primary chemical energy currency in cellular metabolism?" |
| `lexical` | Domain-appropriate synonym substitutions | "Which biochemical molecule acts as the principal cellular energy currency during metabolic processes?" |
| `syntactic` | Passive/active inversion, cleft sentence | "In cellular metabolism, what molecule is utilized as the primary currency of chemical energy?" |
| `pragmatic` | Persona / conversational framing | "As a molecular biochemist, identify the molecule that functions as the universal energy currency in living cells." |
| `length` | Detailed background context elaboration | "Intracellular metabolic reactions require high-energy phosphoanhydride bonds to drive non-spontaneous processes. What specific nucleotide derivative serves as the primary chemical energy currency of eukaryotic cells?" |
| `format` | Strict structured schema / output constraint | "Cellular Biochemistry Query:\nFunction: Primary chemical energy currency\nMolecule name:" |

### Difficult Stress Transformations Included:
1. **Negations**: e.g., "Which organ does NOT produce digestive enzymes: liver, pancreas, or gallbladder?" (P2-051).
2. **Quantities & Constants**: e.g., "What is the speed of light in a vacuum?" (P2-065), "Planck's constant" (P2-003).
3. **Physical Units**: e.g., "SI unit of magnetic flux density" (Tesla, P2-013), "SI unit of electrical capacitance" (Farad, P2-071).
4. **Boundary Conditions**: e.g., "Absolute zero" (P2-005), "Carnot cycle" (P2-030).
5. **Exact Output Constraints**: e.g., "Answer concisely:", "Provide the standard name:".

---

## 3. Academic Domains and Provenance

The 100 clusters span foundational science, engineering, and mathematics disciplines:
- **Biochemistry & Pharmacology**: ATP, Penicillin, Reverse transcriptase, Insulin, Naloxone.
- **Quantum Physics & Electromagnetism**: Planck's constant, Tesla, Superconductivity, Blackbody radiation.
- **Computer Systems & Architecture**: Translation Lookaside Buffer (TLB), BGP routing, Deadlock, Dijkstra's algorithm.
- **Cellular & Molecular Genetics**: Ribosomes, Lysosomes, Uracil, Telomeres, Peptidoglycan.
- **Geophysics & Astrophysics**: Moho discontinuity, Chandrasekhar limit, Cosmic Microwave Background, Basalt.
- **Immunology & Hematology**: IgG, Histamine, SA node.
- **Mathematics & Complexity**: Prime numbers, P vs NP, Archimedes' principle, Reynolds number.

---

## 4. Leakage Audit and Duplicate Detection

The audit script [`scripts/build_phase2_external_dataset.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/build_phase2_external_dataset.py) compared all 600 Phase 2 prompts against every existing record in:
1. `data/paraphrased_prompts_axis_tagged_qa.csv` (196 clusters $\times$ 6 = 1,176 records)
2. `data/external_eval/external_dataset.jsonl` (100 clusters $\times$ 6 = 600 records)
Total existing records audited: **1,776 prompt records**.

### Audit Results:
- **Exact Matches**: **0 (0.00%)**
- **Near-Duplicate Matches (Token Jaccard > 0.85)**: **0 (0.00%)**
- **Zero-Overlap Verification**: **PASSED (`true`)**

*(Note: An initial candidate prompt for P2-086 regarding "ordinary table salt" was flagged during pre-freeze auditing as matching Phase 1 `ext_009`. It was immediately replaced with "chemical element with highest electrical conductivity (Silver)", restoring zero overlap).*

---

## 5. Cryptographic Fingerprints and Split Manifest

The dataset files have been frozen with the following SHA-256 checksums:

```json
{
  "benchmark": "Phase 2 Untouched External Generalization Test",
  "created_at": "2026-10-12T22:25:00Z",
  "total_clusters": 100,
  "total_prompt_records": 600,
  "factual_clusters_count": 50,
  "files": {
    "external_dataset_jsonl": {
      "filename": "data/phase2_external_eval/external_dataset.jsonl",
      "sha256": "50418ed32a0ceb3c306f72b85bb583dfd7f6f3d8d887b0c7a7119b80b5fa252a",
      "records": 600
    },
    "correctness_subset_jsonl": {
      "filename": "data/phase2_external_eval/correctness_subset.jsonl",
      "sha256": "a668aa61f64fcec15f5f8fc02f8e1a56c3b425babd5ee577a50ad88ffb585284",
      "clusters": 50
    }
  },
  "frozen": true
}
```

This benchmark is frozen as the final, untouched evaluation dataset for Phase 2.
