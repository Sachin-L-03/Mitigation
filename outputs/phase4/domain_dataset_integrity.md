# Domain-Shift Dataset Integrity and Provenance Report

**Repository**: `prompt-sensitivity-hybrid`  
**Active Branch**: `creme-hybrid-framework`  
**Dataset Directory**: `data/phase4_domain_eval/`  
**Date**: October 12, 2026  
**Auditor**: Senior ML Research Engineer (Autonomous Phase 4 Protocol)  

---

## 1. Benchmark Architecture and Domain Coverage

The Phase 4 Domain-Shift Benchmark expands evaluation beyond prior single-domain distributions to **200 independent semantic question clusters** comprising **1,200 prompt records** across 6 linguistic transformation axes:

| Domain | Clusters | Prompts | Description & Scope | Verified Factual References |
|---|:---:|:---:|---|:---:|
| **1. Natural Sciences** | 40 | 240 | Cellular biology, biochemistry, inorganic chemistry, classical & particle physics, astronomy. | 40 (100%) |
| **2. Mathematics & Quantitative** | 30 | 180 | Arithmetic, algebra, geometry, calculus, statistics, number theory, discrete math. | 30 (100%) |
| **3. History & Geography** | 40 | 240 | World geography, ancient and modern history, capitals, treaties, oceanic cartography. | 40 (100%) |
| **4. Everyday Practical QA** | 30 | 180 | Common standards, weights, measures, time calculations, domestic chemistry, tools. | 30 (100%) |
| **5. Numerical Constraints & Units** | 30 | 180 | Precise dates, temperatures, unit conversions, player counts, chemical constants. | 30 (100%) |
| **6. Structured Output & Multi-Constraint** | 30 | 180 | Formatted lists, delimiter-separated outputs, paired attributes, ordered tuples. | 30 (100%) |
| **Total** | **200** | **1,200** | **Complete Multi-Domain Distribution** | **200 (100%)** |

---

## 2. Transformation Axes per Semantic Cluster

Each of the 200 semantic question clusters contains 6 linguistic variations:
1. `original`: Canonical baseline formulation.
2. `lexical`: Systematic synonym substitution on question frame and predicates.
3. `syntactic`: Clause inversion, fronting, and passive framing.
4. `pragmatic`: Conversational framing with politeness and exploratory context.
5. `length`: Telegraphic compression stripping filler words.
6. `format`: Structured instruction wrapper with explicit direct-answer prompt.

---

## 3. Cryptographic Leakage and Non-Contamination Audit

To guarantee that Phase 4 results are strictly out-of-distribution and untouched by any prior experimental iteration, an exhaustive programmatic search was performed comparing all 1,200 Phase 4 prompt records against **2,960 historical prompt records** across the entire repository:
- `data/splits/canonical_dataset.jsonl` (Original 1,176 records)
- `data/splits/train.jsonl` (822 records)
- `data/splits/val.jsonl` (174 records)
- `data/splits/test.jsonl` (180 records)
- `data/external_eval/external_dataset.jsonl` (Phase 1 600 records)
- `data/phase2_external_eval/external_dataset.jsonl` (Phase 2 600 records)
- `data/phase3_final_eval/external_dataset.jsonl` (Phase 3 600 records)

### Overlap Audit Results:
- **Total Historical Prompts Queried**: 2,960 unique prompt strings
- **Exact String Matches**: **0 (0.00%)**
- **Normalized Token Matches**: **0 (0.00%)**
- **Near-Duplicate Cosine Matches ($>0.85$)**: **0 (0.00%)**

---

## 4. Cryptographic Manifest and File Hashes

Frozen SHA-256 cryptographic hashes for the Phase 4 benchmark files:

```json
{
  "dataset_name": "Phase 4 Domain-Shift Benchmark",
  "total_clusters": 200,
  "total_prompts": 1200,
  "files": {
    "domain_dataset.jsonl": {
      "path": "data/phase4_domain_eval/domain_dataset.jsonl",
      "sha256": "6c9ef28e614dd122a4faaf4a6f3d1e0a2b52749717aa4d541c9c0e5e212c1f2c",
      "bytes": 442566
    },
    "correctness_subset.jsonl": {
      "path": "data/phase4_domain_eval/correctness_subset.jsonl",
      "sha256": "27e84b559c18fd6c702a4df85f02ce3f1a1e40686979f5d3e63dccea6d56a042",
      "bytes": 43046
    },
    "hard_negatives.jsonl": {
      "path": "data/phase4_domain_eval/hard_negatives.jsonl",
      "sha256": "cb2bd8de770147871d5f4c4d347d68b543645bba422572c23cea3e784cb6215d",
      "bytes": 1651
    }
  }
}
```

---

## 5. Hard Negative Integrity

To ensure that the router does not collapse distinct semantic questions into identical answers, `hard_negatives.jsonl` defines controlled constraint perturbations:
- `neg_001` (sci_032): Valence electrons of Carbon (4) vs Nitrogen (5) $\rightarrow$ Entity substitution.
- `neg_002` (cst_002): Requesting 3 vs 5 liquid elements $\rightarrow$ Numerical constraint change.
- `neg_003` (sci_003): Speed of light vs speed of sound in vacuum $\rightarrow$ Physical concept negation.
- `neg_004` (math_008): Legs 3, 4 (hypotenuse 5) vs legs 5, 12 (hypotenuse 13) $\rightarrow$ Parameter change.
- `neg_005` (geo_003): Apollo 11 (landed on Moon) vs Apollo 13 (aborted, did not land) $\rightarrow$ Historical mission negation.

The mitigation router must treat all hard negative pairs as distinct, refusing to enforce identical answers when task constraints differ.
