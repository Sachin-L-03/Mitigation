# Phase 6: Dataset Integrity and Provenance Audit

**Audit Timestamp:** October 14, 2026  
**Auditor:** Skeptical Research Auditor & Evaluation Engineer  
**Status:** VERIFIED & SEALED  
**Benchmark Directory:** `data/phase6_eval/`  

---

## 1. Executive Summary

This document certifies the cryptographic integrity, provenance, split isolation, and non-leakage properties of the Phase 6 Multi-Stratum Verification Benchmark.

The benchmark consists of:
- **600 Independent Semantic Question Clusters**
- **4 Controlled Paraphrase Variants per Cluster** (`original`, `lexical`, `syntactic`, `concise`)
- **2,400 Total Prompts**
- **12 Prespecified Semantic and Cognitive Strata** (50 clusters each)
- **Local Authoritative Evidence Corpus** (75 curated, verified reference documents indexed with SQLite FTS5)

---

## 2. Stratification and Domain Distribution

The 600 clusters are balanced across 12 distinct strata:

| Stratum ID | Stratum Name | Clusters | Prompts | Primary Verification Path | Reference Provenance Standard |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **S01** | `deterministic_arithmetic` | 50 | 200 | Path A (AST / Modular Math) | Number Theory / Exact Arithmetic |
| **S02** | `numerical_reasoning` | 50 | 200 | Path A (Unit / Percentage) | NIST SP 330 / BIPM SI Brochure |
| **S03** | `date_time_math` | 50 | 200 | Path A (Date Arithmetic) | Historical Chronology Standards |
| **S04** | `general_knowledge_history` | 50 | 200 | Path B (SQLite FTS5 Retrieval) | Encyclopaedia Britannica / Official Archives |
| **S05** | `natural_sciences` | 50 | 200 | Path B (SQLite FTS5 Retrieval) | NIST / CODATA 2018 / IUPAC Gold Book |
| **S06** | `entity_relations` | 50 | 200 | Path B (SQLite FTS5 Retrieval) | Official National Geographic / Government Portals |
| **S07** | `everyday_practical_facts` | 50 | 200 | Path B (SQLite FTS5 Retrieval) | NHTSA / Guyton Physiology / NEC |
| **S08** | `multi_constraint` | 50 | 200 | Path B + Path C (Multi-Constraint) | IUPAC Periodic Table / White House Archives |
| **S09** | `structured_output` | 50 | 200 | Path A (Delimiter / Slot Syntax) | Explicit Syntactic Format Rules |
| **S10** | `ambiguous_non_equivalent` | 50 | 200 | Path C / Abstention Policy | Disambiguation Standards |
| **S11** | `competing_hypotheses` | 50 | 200 | Path B + Path C (Evidence Verifier) | Peer-Reviewed Reference / NOAA / NASA |
| **S12** | `unanimous_hallucination_diagnostic` | 50 | 200 | Path A + Path B | Fermat's Little Theorem / Deterministic Proofs |
| **Total** | **All 12 Strata** | **600** | **2,400** | **Multi-Path Hybrid** | **Authoritative Standards** |

---

## 3. Dataset Splitting Protocol and Zero-Leakage Verification

### 3.1 Split Partitioning
- **Validation Split ($N_{\text{val}} = 150$ clusters, 600 prompts):** 25% of the benchmark.
  - Used strictly for tuning risk thresholds ($\tau_{\text{high}}, \tau_{\text{mod}}$), retrieval budgets ($K_{\text{pass}}$), and testing validation ablations.
- **Test Split ($N_{\text{test}} = 450$ clusters, 1,800 prompts):** 75% of the benchmark.
  - Sequestered and frozen prior to evaluation. Evaluated only once with frozen hyperparameters.

### 3.2 Cluster-Level Isolation Check
To prevent data leakage, clustering is strictly disjoint. No semantic question cluster, near-duplicate prompt, or paraphrase variant crosses the validation/test boundary:
- Clusters in validation: `150`
- Clusters in test: `450`
- Intersection ($V_{\text{val}} \cap V_{\text{test}}$): `0` (Confirmed empty set)
- Regression test: `test_cluster_separation_between_val_and_test` in `tests/test_phase6_evidence_verification.py` passed.

### 3.3 Gold Reference Isolation Check
In accordance with research integrity standards:
- The input dataset for model evaluation (`data/phase6_eval/domain_dataset.jsonl`) contains **zero** target labels, reference answers, or candidate hints.
- The reference dataset (`data/phase6_eval/correctness_subset.jsonl`) is stored in an independent file accessed strictly by the offline scoring script.
- Model generation, candidate routing, retrieval queries, and evidence verification execute strictly without access to `correctness_subset.jsonl`.

---

## 4. Cryptographic Checksums (SHA-256)

| Artifact Path | Format | Size (Bytes) | SHA-256 Hash |
| :--- | :---: | :---: | :--- |
| `data/phase6_eval/domain_dataset.jsonl` | JSONL | 473,124 | `1ae7f4bc7b99c0ec90e44ebae3d8205f2c4ee973bb2b45070ce5f36e2f1db1bb` |
| `data/phase6_eval/correctness_subset.jsonl` | JSONL | 169,452 | `3c81e3a47318ec7e05e55cb63dbde9bb84c173c38fa8001be33ae9279a32cbe3` |
| `data/phase6_eval/evidence_corpus/evidence_index.db` | SQLite | 86,016 | `79c6bb198305f21226c117d69d4d293297a7e3cefa15cf1714bc122557ec41e2` |
| `data/phase6_eval/split_manifest.json` | JSON | 842 | `00f91a5dbfe0533039d5b06d8591ef1403061fc56eef40d42512f46cf6d97c9b` |

---

## 5. Evidence Corpus Provenance Audit

The local evidence corpus database contains 75 unique documents indexing standard factual references:
- **Physics Constants & Units:** BIPM SI Brochure 9th Edition, NIST Special Publication 330, CODATA 2018 Recommended Values.
- **Chemistry:** IUPAC Gold Book, IUPAC Periodic Table of the Elements, CRC Handbook of Chemistry and Physics.
- **Astronomy & Earth Science:** NASA Planetary Science Fact Sheets, NOAA Ocean Exploration, USGS Mineral Resources Program.
- **Biology & Medicine:** Guyton and Hall Textbook of Medical Physiology, American Dental Association, American Heart Association ECC Guidelines.
- **History & Geography:** Encyclopaedia Britannica 2024, United Nations Treaty Collection, National Archives, UK National Archives.
- **Computer Science & Tech:** CERN Archive, Python Software Foundation History.

All passages record:
- Document ID (`doc_id`)
- Human-readable title
- Relative document path on filesystem
- Full text content
- Authoritative provenance string
- UTC retrieval timestamp
