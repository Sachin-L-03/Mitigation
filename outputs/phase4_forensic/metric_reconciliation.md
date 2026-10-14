# Phase 4 Metric Reconciliation Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Audit Target**: `outputs/phase4/final_phase4_research_report.md` and `outputs/phase4/phase4_results_summary.json`  
**Forensic Script**: [`scripts/audit_phase4_metrics.py`](file:///scripts/audit_phase4_metrics.py)  
**Primary Dataset**: `data/phase4_domain_eval/domain_dataset.jsonl` (200 clusters, 1,200 prompts)  
**Verification References**: `data/phase4_domain_eval/correctness_subset.jsonl` (200 gold answers)  
**Date**: October 13, 2026  
**Auditor**: ML Research Auditor & Experimental Statistician  

---

## 1. Executive Summary

This document provides a forensic line-by-line reconciliation of all reported Phase 4 metrics against independently recomputed metrics generated directly from raw model outputs ([`outputs/phase4/raw_predictions.jsonl`](file:///outputs/phase4/raw_predictions.jsonl)). Precomputed summary files and report claims were quarantined during this audit.

### Headline Reconciliation Summary:
1. **Token F1 (Canonical)**:
   - Reported: **57.43%** (E0), **57.37%** (E3), **100.00%** (E1), **90.64%** (E6_bounded).
   - Recomputed: **57.43%** (E0), **57.37%** (E3), **100.00%** (E1), **90.64%** (E6_bounded).
   - Status: **Verified**.
2. **Exact Match Disambiguation**:
   - The original report published a single "Pairwise EM" column with values: E0 = 11.10%, E3 = 10.80%, E1 = 100.00%, E6_bounded = 77.30%.
   - Independent audit reveals:
     - **Pairwise Exact Match** (across all 15 pairs per cluster): E0 = **5.37%**, E3 = **5.23%**, E1 = **100.00%**, E6_bounded = **75.63%**.
     - **Canonical Exact Match** (variants vs canonical prompt): E0 = **11.10%**, E3 = **10.80%**, E1 = **100.00%**, E6_bounded = **77.30%**.
     - **All-Identical Cluster Rate** (all 6 variants producing identical answers): E0 = **0.50%** (1/200), E3 = **0.50%** (1/200), E1 = **100.00%** (200/200), E6_bounded = **74.00%** (148/200).
   - Status: **Corrected with qualification**. The published "Pairwise EM" column was actually **Canonical Exact Match**. True pairwise EM is 75.63% for E6_bounded.
3. **Factual Correctness vs Concept Retrieval**:
   - Level 1 Concept Retrieval: E0 = **81.60%**, E3 = **81.40%**, E1 = **80.00%**, E6_bounded = **80.70%** (matches published "Factual Acc").
   - Level 2 Contextual Factual Accuracy (penalizing internal negation and numerical slot contradictions against gold): E0 = **77.20%**, E3 = **76.80%**, E1 = **75.50%**, E6_bounded = **76.40%**.
   - Status: **Verified with qualification**. Published factual accuracy used keyword/concept substring retrieval rather than strict slot-level evaluation.
4. **Contradiction Suppression**:
   - Reported: 340 hard contradictions in E0 (34.00%), 0 in E6_bounded (0.00%).
   - Recomputed: 340 hard contradictions in E0 (38 numerical, 59 negation, 282 entity divergences), 0 in E6_bounded.
   - Status: **Verified in execution, Qualified in interpretation**. E6_bounded achieves 0 contradictions because 73.5% of clusters trigger verification and are assigned identical shared strings, and the remaining 26.5% fast-path clusters were already contradiction-free.

---

## 2. Comprehensive Reconciliation Table

| Metric | Condition | Original Published | Independently Recomputed | Difference | Implementation Location | Auditor Verdict |
|---|---|:---:|:---:|:---:|---|---|
| **Canonical Token F1** | E0 (Base) | 57.43% | 57.43% | 0.00 pp | `scripts/audit_phase4_metrics.py:364` | **Verified** |
| | E3 (Adapter) | 57.37% | 57.37% | 0.00 pp | `scripts/audit_phase4_metrics.py:364` | **Verified** |
| | E1 (Full Medoid) | 100.00% | 100.00% | 0.00 pp | `scripts/audit_phase4_metrics.py:364` | **Verified** |
| | E6_bounded | 90.64% | 90.64% | 0.00 pp | `scripts/audit_phase4_metrics.py:364` | **Verified** |
| **Pairwise Token F1** | E0 (Base) | Not reported | 50.13% | N/A | `scripts/audit_phase4_metrics.py:365` | **New Metric** |
| | E3 (Adapter) | Not reported | 50.08% | N/A | `scripts/audit_phase4_metrics.py:365` | **New Metric** |
| | E1 (Full Medoid) | Not reported | 100.00% | N/A | `scripts/audit_phase4_metrics.py:365` | **New Metric** |
| | E6_bounded | Not reported | 88.56% | N/A | `scripts/audit_phase4_metrics.py:365` | **New Metric** |
| **Semantic Similarity** | E0 (Base) | 70.70% | 70.67% | -0.03 pp | `scripts/audit_phase4_metrics.py:366` | **Verified** |
| | E3 (Adapter) | 70.54% | 70.52% | -0.02 pp | `scripts/audit_phase4_metrics.py:366` | **Verified** |
| | E1 (Full Medoid) | 100.00% | 100.00% | 0.00 pp | `scripts/audit_phase4_metrics.py:366` | **Verified** |
| | E6_bounded | 93.82% | 93.81% | -0.01 pp | `scripts/audit_phase4_metrics.py:366` | **Verified** |
| **Canonical Exact Match** | E0 (Base) | 11.10% (as "Pairwise EM") | 11.10% | 0.00 pp | `scripts/audit_phase4_metrics.py:368` | **Mislabelled in Report** |
| | E3 (Adapter) | 10.80% (as "Pairwise EM") | 10.80% | 0.00 pp | `scripts/audit_phase4_metrics.py:368` | **Mislabelled in Report** |
| | E1 (Full Medoid) | 100.00% | 100.00% | 0.00 pp | `scripts/audit_phase4_metrics.py:368` | **Verified** |
| | E6_bounded | 77.30% (as "Pairwise EM") | 77.30% | 0.00 pp | `scripts/audit_phase4_metrics.py:368` | **Mislabelled in Report** |
| **Pairwise Exact Match** | E0 (Base) | Not isolated | 5.37% | N/A | `scripts/audit_phase4_metrics.py:369` | **Disambiguated** |
| | E3 (Adapter) | Not isolated | 5.23% | N/A | `scripts/audit_phase4_metrics.py:369` | **Disambiguated** |
| | E1 (Full Medoid) | 100.00% | 100.00% | 0.00 pp | `scripts/audit_phase4_metrics.py:369` | **Verified** |
| | E6_bounded | Not isolated | 75.63% | N/A | `scripts/audit_phase4_metrics.py:369` | **Disambiguated** |
| **All-Identical Clusters** | E0 (Base) | Not isolated | 0.50% (1/200) | N/A | `scripts/audit_phase4_metrics.py:370` | **Disambiguated** |
| | E3 (Adapter) | Not isolated | 0.50% (1/200) | N/A | `scripts/audit_phase4_metrics.py:370` | **Disambiguated** |
| | E1 (Full Medoid) | 100.00% | 100.00% (200/200) | 0.00 pp | `scripts/audit_phase4_metrics.py:370` | **Verified** |
| | E6_bounded | Not isolated | 74.00% (148/200) | N/A | `scripts/audit_phase4_metrics.py:370` | **Disambiguated** |
| **Anchor Preservation** | E0 (Base) | 100.00% | 100.00% | 0.00 pp | `scripts/audit_phase4_metrics.py:371` | **Verified** (by def) |
| | E3 (Adapter) | 95.19% | 95.19% | 0.00 pp | `scripts/audit_phase4_metrics.py:371` | **Verified** |
| | E1 (Full Medoid) | 81.44% | 81.44% | 0.00 pp | `scripts/audit_phase4_metrics.py:371` | **Verified** |
| | E6_bounded | 83.47% | 83.47% | 0.00 pp | `scripts/audit_phase4_metrics.py:371` | **Verified** |
| **Concept Retrieval** | E0 (Base) | 81.60% (as "Factual Acc") | 81.60% | 0.00 pp | `scripts/audit_phase4_metrics.py:373` | **Verified** |
| | E3 (Adapter) | 81.40% (as "Factual Acc") | 81.40% | 0.00 pp | `scripts/audit_phase4_metrics.py:373` | **Verified** |
| | E1 (Full Medoid) | 80.00% (as "Factual Acc") | 80.00% | 0.00 pp | `scripts/audit_phase4_metrics.py:373` | **Verified** |
| | E6_bounded | 80.70% (as "Factual Acc") | 80.70% | 0.00 pp | `scripts/audit_phase4_metrics.py:373` | **Verified** |
| **Contextual Factual Acc**| E0 (Base) | Not reported | 77.20% | N/A | `scripts/audit_phase4_metrics.py:374` | **New Metric** |
| | E3 (Adapter) | Not reported | 76.80% | N/A | `scripts/audit_phase4_metrics.py:374` | **New Metric** |
| | E1 (Full Medoid) | Not reported | 75.50% | N/A | `scripts/audit_phase4_metrics.py:374` | **New Metric** |
| | E6_bounded | Not reported | 76.40% | N/A | `scripts/audit_phase4_metrics.py:374` | **New Metric** |
| **Hard Contradictions** | E0 (Base) | 340 (34.00%) | 340 (34.00%) | 0 | `scripts/audit_phase4_metrics.py:375` | **Verified** |
| | E3 (Adapter) | 338 (33.80%) | 338 (33.80%) | 0 | `scripts/audit_phase4_metrics.py:375` | **Verified** |
| | E1 (Full Medoid) | 0 (0.00%) | 0 (0.00%) | 0 | `scripts/audit_phase4_metrics.py:375` | **Verified** |
| | E6_bounded | 0 (0.00%) | 0 (0.00%) | 0 | `scripts/audit_phase4_metrics.py:375` | **Verified** |

---

## 3. Detailed Forensic Analysis of Discrepancies

### 3.1 Exact Match Conflation
In Table 3.1 of `outputs/phase4/final_phase4_research_report.md`, column 5 is titled `Pairwise EM (%)` and reports:
- E0: 11.10%
- E3: 10.80%
- E6_bounded: 77.30%

**Forensic Finding**:
In `src/mitigation/metrics.py:150-165` and `scripts/run_phase4_domain_evaluation.py:331-360`, the code compared each of the 5 paraphrases against `orig_final` (the cluster's canonical variant answer):
$$\text{Canonical EM} = \frac{1}{5} \sum_{i=1}^5 \mathbb{I}(\text{norm}(y_i) = \text{norm}(y_{\text{original}}))$$
This calculation evaluates agreement with the canonical variant, NOT agreement between all $\binom{6}{2} = 15$ pairs.
When all 15 unordered pairs in each cluster are evaluated:
$$\text{Pairwise EM} = \frac{1}{15} \sum_{1 \le j < k \le 6} \mathbb{I}(\text{norm}(y_j) = \text{norm}(y_k))$$
- Baseline E0 Pairwise EM is **5.37%** (not 11.10%).
- Bounded router E6_bounded Pairwise EM is **75.63%** (not 77.30%).
- The all-identical cluster rate is **74.00%** (148/200 clusters produce identical answers across all 6 variants).

The report mislabelled Canonical Exact Match as Pairwise Exact Match. The recomputed pairwise metric confirms that E6_bounded still delivers a massive +70.26 pp increase in pairwise exact match over baseline.

### 3.2 Token F1 Aggregation Definition
The report stated Token F1 is 57.43% for E0 and 90.64% for E6_bounded.
Audit traces the exact aggregation implementation:
1. For each cluster $c \in \{1, \dots, 200\}$:
   - Evaluates the 5 variant answers $y_{c, 1}, \dots, y_{c, 5}$ against the canonical answer $y_{c, \text{orig}}$.
   - Normalizes text by lowercasing, stripping punctuation, and removing leading/trailing whitespace.
   - Computes token-level precision, recall, and harmonic mean $F_1(y_{c, i}, y_{c, \text{orig}})$.
   - Computes cluster mean: $\overline{F}_{1, c} = \frac{1}{5} \sum_{i=1}^5 F_1(y_{c, i}, y_{c, \text{orig}})$.
2. Overall metric is the macro-average across all 200 clusters:
   $$\text{Macro F1} = \frac{1}{200} \sum_{c=1}^{200} \overline{F}_{1, c} = 90.64\%$$
When evaluated pairwise across all 15 combinations without privileging the canonical prompt, Macro Pairwise Token F1 is **88.56%** for E6_bounded vs **50.13%** for E0 (+38.43 pp gain). Both aggregation methods confirm substantial consistency gains.

### 3.3 Reference Answer Handling and Factual Accuracy
- Verified that all 200 clusters in `data/phase4_domain_eval/correctness_subset.jsonl` possess exactly one verified ground truth reference.
- Verified that reference answers were **NEVER provided in inference prompts or context windows**. Model generations were produced strictly from the prompt text.
- Scoring procedure: `check_concept_accuracy` checks whether any salient comma/slash separated chunk of the gold reference appears as a substring in the normalized generation, or fallback Token F1 $\ge 0.40$.
- Because `check_concept_accuracy` is a substring entity retrieval metric, we introduced `factual_accuracy` in `scripts/audit_phase4_metrics.py`, which checks both entity presence AND the absence of numerical and negation conflicts against the gold reference.
- Under strict factual accuracy, E0 scores **77.20%** and E6_bounded scores **76.40%** ($-0.80$ pp). Under concept retrieval, E0 scores **81.60%** and E6_bounded scores **80.70%** ($-0.90$ pp).

---

## 4. Verification Conclusion

All primary Token F1, semantic similarity, anchor preservation, and contradiction count figures published in the Phase 4 report match the raw prediction records. However:
1. Column "Pairwise EM" must be recognized as Canonical Exact Match.
2. "Factual Accuracy" represents keyword/entity concept retrieval, which drops by 0.80 pp to 0.90 pp under E6_bounded.
3. Zero contradictions is achieved by string assignment on verified clusters, eliminating divergence by construction.
