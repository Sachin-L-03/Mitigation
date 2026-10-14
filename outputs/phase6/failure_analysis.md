# Phase 6: Comprehensive Failure Mode Analysis & Error Audit

**Artifact Path:** `outputs/phase6/failure_analysis.md`  
**Evaluation Set:** Test Split ($N_{\text{test}} = 450$ semantic question clusters, 1,800 prompts)  
**Evaluated Condition:** Primary Deployment Candidate E4 (Combined Evidence-Grounded Router)  
**Execution Timestamp:** 2026-10-14T07:03:41 UTC  

---

## 1. Executive Summary

In Phase 6, Condition E4 (Combined Evidence-Grounded Router) achieved a factual accuracy of **81.56%** (367 / 450 clusters correct), representing a net reduction in errors from 185 clusters (under E1 baseline, 58.89% accuracy) down to **83 clusters**.

In accordance with Section 11 of the Phase 6 specification, this document provides a complete audit of the remaining 83 failure cases. All failures were mapped into the 19 prespecified taxonomic failure categories. Furthermore, we investigate the **4 corruption cases** where the baseline E1 router was correct but the verification pipeline altered the answer to an incorrect output, and classify which errors can be resolved via stronger tooling versus those that fundamentally require human review.

---

## 2. Global Failure Category Taxonomy

Table 1 displays the incidence count across all 19 preregistered failure categories for Condition E4 across the 450 test clusters.

### Table 1: Distribution of Failure Categories under E4 ($N_{\text{test}} = 450$)

| Failure Category | Test Count | % of All Failures ($N=83$) | Primary Strata Involved | Resolvable via Tooling vs. Human Review |
| :--- | :---: | :---: | :--- | :--- |
| `ambiguous_question` | **38** | 45.78% | `ambiguous_non_equivalent` | Requires human clarification / prompt disclaimers |
| `unsupported_source_claim` | **25** | 30.12% | `everyday_practical_facts`, `general_knowledge`, `entity_relations` | Resolvable via broader external knowledge corpus |
| `constraint_violation` | **8** | 9.64% | `structured_output`, `multi_constraint` | Resolvable via constrained grammar decoding |
| `incorrect_answer_correction` | **8** | 9.64% | `multi_constraint`, `everyday_facts` | Resolvable via stricter multi-sentence predicate checking |
| `incorrect_model_calculation` | **4** | 4.82% | `deterministic_arithmetic`, `numerical_reasoning` | Resolvable via extended formal word-problem parsers |
| `correct_answer_unnecessarily_changed`| **4** | 4.82% | Overlaps with `incorrect_answer_correction` | Resolvable via higher replacement confidence thresholds |
| `incorrect_deterministic_classification` | 0 | 0.00% | N/A | Path A classifier had 0 false-positive formalizations |
| `incorrect_operand_extraction` | 0 | 0.00% | N/A | Regex and AST operand extraction was 100% accurate |
| `unit_or_sign_error` | 0 | 0.00% | N/A | Explicitly handled by Pint & SymPy sign rules |
| `retrieval_failure` | 0 | 0.00% | N/A | SQLite FTS5 index had 100% query execution success |
| `irrelevant_evidence_retrieval` | 0 | 0.00% | N/A | BM25 token-overlap filtering prevented off-topic fetches |
| `conflicting_sources` | 0 | 0.00% | N/A | Curated local index contained zero contradictory docs |
| `stale_source` | 0 | 0.00% | N/A | Static factual corpus with verified timestamps |
| `evidence_misinterpretation` | 0 | 0.00% | N/A | Exact predicate extraction prevented hallucinated readings |
| `false_positive_risk_flag` | 0 | 0.00% | N/A | No valid responses flagged inappropriately |
| `false_negative_risk_flag` | 0 | 0.00% | N/A | All high-risk numerical/date tasks properly routed |
| `unanimous_incorrect_consensus` | 0 | 0.00% | N/A | **100% of UFC cases (19/19) were corrected!** |
| `unnecessary_abstention` | 0 | 0.00% | N/A | E4 operates under full coverage (0 abstentions) |
| `failure_to_abstain` | 0 | 0.00% | N/A | Applies to E5 conservative analysis |
| **Total Failures** | **83** | **100.0%** | — | — |

*(Note: Certain failures involve overlapping categories, such as the 4 corruption cases which are jointly logged under `incorrect_answer_correction` and `correct_answer_unnecessarily_changed`.)*

---

## 3. Stratum-Level Failure Breakdown

Table 2 maps the 83 failed clusters to their respective evaluation strata.

### Table 2: Error Counts by Stratum under E4 ($N_{\text{test}} = 450$)

| Stratum | Total Clusters | Correct ($E_4$) | Failed ($E_4$) | Stratum Accuracy | Dominant Root Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ambiguous_non_equivalent` | 38 | 0 | **38** | 0.00% | Intentional semantic ambiguity; no single answer valid |
| `multi_constraint` | 38 | 21 | **17** | 55.26% | Partial constraint satisfaction without external verifier |
| `structured_output` | 38 | 30 | **8** | 78.95% | Minor JSON formatting deviations vs reference schema |
| `everyday_practical_facts` | 38 | 31 | **7** | 81.58% | Corpus lacked specific niche factual assertions |
| `general_knowledge_history`| 37 | 33 | **4** | 89.19% | Complex historical timelines with unindexed details |
| `entity_relations` | 37 | 33 | **4** | 89.19% | Missing secondary relational triplets |
| `deterministic_arithmetic` | 37 | 34 | **3** | 91.89% | Complex word problem phrasing fell back to model |
| `numerical_reasoning` | 37 | 36 | **1** | 97.30% | Multi-currency conversion rate unstated in prompt |
| `competing_hypotheses` | 38 | 37 | **1** | 97.37% | Fine-grained scientific theory distinction |
| `date_time_math` | 37 | 37 | **0** | **100.00%** | Zero failures (Path A solved all cases) |
| `natural_sciences` | 37 | 37 | **0** | **100.00%** | Zero failures (Path B grounded all claims) |
| `unanimous_hallucination` | 38 | 38 | **0** | **100.00%** | Zero failures (Path A corrected all false consensus) |
| **Total** | **450** | **367** | **83** | **81.56%** | — |

---

## 4. In-Depth Root Cause Analysis & Representative Case Studies

### 4.1 Ambiguous & Non-Equivalent Prompts (`ambiguous_question`, $N = 38$)
- **Mechanism:** In Stratum 10 (`ambiguous_non_equivalent`), prompt variants within the cluster are intentionally designed to have multiple conflicting interpretations (e.g., "What is the capital of Georgia?" referring to the US state vs. the Caucasus nation; or "When was the constitution signed?" without specifying the country).
- **Behavior:** Because E4 operates under a 100% coverage mandate, it chooses one valid candidate answer. However, because the cluster evaluation requires equivalence across all prompt variants, the cluster fails strict equivalence and is scored 0.0.
- **Remedy:** As shown in the coverage analysis, Condition E5 correctly abstains on **100% of these 38 clusters**. For production systems, ambiguous questions must trigger disambiguation dialogues rather than arbitrary single-entity completions.

### 4.2 Missing Corpus Evidence (`unsupported_source_claim`, $N = 25$)
- **Mechanism:** In practical and historical facts, the prompt asked for factual specifics that were not present in the 75 curated documents in `evidence_index.db`.
- **Case Example (`cluster_hist_029`):**
  - *Prompt:* "In what year was the Treaty of Utrecht signed by Great Britain and Spain?"
  - *Candidate Answer:* "1714"
  - *Retrieved Evidence:* Document covered the War of the Spanish Succession broadly, but lacked the specific treaty ratification date passage.
  - *Verification Decision:* `INSUFFICIENT_EVIDENCE`.
  - *Pipeline Action:* E4 retained the E3 candidate answer ("1714") without modification.
  - *Evaluation Reference:* "1713" (Great Britain and Spain signed in July 1713).
  - *Root Cause:* Without relevant external evidence, Path B safely refrained from rewriting, leaving the base model's uncorrected error intact.

### 4.3 Constraint Violations in Structured Prompts (`constraint_violation`, $N = 8$)
- **Case Example (`cluster_struct_014`):**
  - *Prompt:* "Return a JSON object with keys 'city', 'country', and 'population_millions' for Tokyo."
  - *Generated Answer:* `{"city": "Tokyo", "country": "Japan", "population": 37.4}` (Used key `population` instead of `population_millions`).
  - *Root Cause:* The base LLM generated a semantically accurate JSON object that violated the exact schema key constraint. Path A's JSON validator identified the syntax as valid JSON, but the reference checker rejected the key mismatch.
  - *Remedy:* Integrate a formal JSON-schema constraint decoder (e.g., Outlines or Guidance) into candidate generation.

### 4.4 Audit of the 4 Corruption Cases (`correct_answer_unnecessarily_changed`)
Across the 450 test clusters, there were exactly **4 clusters** where E1 (baseline) was correct, but E4 was incorrect. In McNemar terms, these represent the 4 discordant losses ($c = 4$). We conducted a deep forensic audit on each case:
1. **Case 1 (`cluster_multi_007`):** Prompt asked for a list of three European countries meeting two geographical criteria. E1 provided a correct triplet. Path B retrieved an article describing only two of the countries and flagged `PARTIALLY_SUPPORTED`, triggering an over-conservative edit that truncated the list.
2. **Case 2 (`cluster_multi_019`):** Prompt required an answer formatted under a specific negative constraint ("Do not mention X"). Path C's verification step appended a clarifying clause citing evidence that inadvertently referenced entity X, triggering a constraint failure.
3. **Case 3 (`cluster_everyday_012`):** Base model generated a colloquial measurement ("about 2 cups"). Path A extracted numerical operands and converted to standard SI units ("0.473 liters"), which failed the rigid string-matching criteria of the colloquial reference label.
4. **Case 4 (`cluster_entity_021`):** An ambiguous biographical query where E1 produced the primary historical figure, but retrieved evidence cited a namesake junior relative, causing a misaligned candidate overwrite.

**Takeaway:** The corruption rate is exceptionally low: **0.89%** (4 out of 450 clusters). This is vastly outweighed by the **106 corrected clusters** ($b = 106$), yielding a net win ratio of **26.5 to 1**.

---

## 5. Architectural Capability Bounding

Table 3 categorizes the remaining errors by the engineering interventions required to resolve them.

### Table 3: Resolution Pathway for Remaining 83 Failures

| Failure Group | Error Count | Percentage | Required Mitigation Technique |
| :--- | :---: | :---: | :--- |
| **Semantic Ambiguity** | 38 | 45.78% | Interactive disambiguation / Conservative Fallback (E5) |
| **Corpus Coverage Limits**| 25 | 30.12% | Expansion of external knowledge base / live web search adapter |
| **Schema & Negative Constraints** | 8 | 9.64% | Constrained decoding / JSON grammar enforcement |
| **Multi-Sentence Reasoning** | 8 | 9.64% | Multi-hop graph retrieval & compositional verification |
| **Complex Word Problems** | 4 | 4.82% | Semantic parser for natural-language word-problem extraction |
| **Total** | **83** | **100.0%** | — |

---

## 6. Conclusion

The failure analysis confirms that the Phase 6 verification layer did not introduce runaway corruptions or unstable behavioral regressions. The dominant remaining failure mode (45.78%) is unresolvable semantic ambiguity, which Condition E5 correctly addresses by abstaining. In factual domains, failures are predominantly caused by finite local corpus size rather than verification logic failures.
