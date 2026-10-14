# Phase 4 Test Suite and Regression Verification Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Execution Environment**: Windows 11, PowerShell, Python 3.11.9, pytest-9.1.1  
**Auditor**: ML Research Auditor & Experimental Statistician  
**Date**: October 13, 2026  

---

## 1. Executive Summary

This report documents the verification of the project's regression test suite and the addition of targeted forensic tests designed to prevent regression on the defects, methodological ambiguities, and statistical fallacies discovered during the Phase 4 forensic audit.

### Summary of Results:
- **Baseline Test Count**: **79 passing tests** (0 failed, 0 skipped, run time 4.90s).
- **New Forensic Regression Tests Added**: **8 targeted tests** in [`tests/test_phase4_forensic.py`](file:///tests/test_phase4_forensic.py).
- **Final Test Count**: **87 passing tests** (0 failed, 0 skipped, run time 4.98s).
- **Status**: **100% Green**. Zero tests were deleted, weakened, or skipped.

---

## 2. Inventory of Forensic Regression Tests Added

The new test suite [`tests/test_phase4_forensic.py`](file:///tests/test_phase4_forensic.py) explicitly tests the audited behaviors identified in Phase 4:

### Table 2.1: New Forensic Tests

| Test Function | Audited Issue / Defect Covered | Validation Logic |
|---|---|---|
| `test_exact_match_disambiguation` | Conflation of Canonical EM, Pairwise EM, and All-Identical rate | Evaluates a controlled 4-variant cluster with known divergent agreements, ensuring canonical EM (33.3%), pairwise EM (33.3%), and all-identical rate (0.0%) are strictly distinct. |
| `test_non_inferiority_testing_logic` | Two-sided difference test fallacy used to claim non-inferiority | Simulates paired difference observations with small sample mean drop ($-0.8$ pp) and verifies that failing to reject equality ($p > 0.05$) does NOT establish non-inferiority ($p_{\text{NI}} > 0.025$, lower bound $< -2.0$ pp). |
| `test_csv_jsonl_record_reconciliation` | Discrepancy between CSV and JSONL raw prediction rows | Reconciles all 6,000 rows between `raw_predictions.csv` and `raw_predictions.jsonl`, verifying exact match on `prompt_id`, `cluster_id`, `method`, `final_answer`, and `token_f1`. |
| `test_analytical_latency_accounting` | Latency boundary calculation and multiplier | Verifies that $L = \frac{53 \times 266.33 + 147 \times (3 \times 266.33 + 3.0)}{200} = 660.1$ ms and that generations per query equals 2.47, confirming the analytical model. |
| `test_unanimous_hallucination_behavior` | Blindspot where unanimous agreement hides factual error | Feeds a cluster where all candidates hallucinate "Nero" for Mount Vesuvius; verifies that comparator finds agreement ($\ge 0.50$ similarity, 0 contradictions) while factual accuracy against "Titus" is correctly scored as False (0%). |
| `test_alias_resolution_avoids_false_positives` | False-positive entity mismatches on domain aliases | Verifies that biological and networking aliases (`ATP` vs `adenosine triphosphate`, `BGP` vs `Border Gateway Protocol`) with shared substantive context do not trigger false-positive entity contradictions. |
| `test_bounded_candidate_budget_strictness` | Unbounded candidate generation or runaway loops | Supplies 10 candidates to `compute_bounded_medoid`; verifies that candidate evaluation is strictly truncated to `max_k = 3`. |
| `test_math_negative_sign_contradiction` | Mathematical sign errors ($+4$ vs $-4$) stripped by punctuation normalization | Tests that sign divergence between positive and negative numbers is flagged and not discarded as a false match. |

---

## 3. Test Execution Log

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Student5\Desktop\Mitigation\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Student5\Desktop\Mitigation
plugins: anyio-4.15.1
collecting ... collected 87 items

tests/test_activations.py::test_get_transformer_layers PASSED            [  1%]
tests/test_activations.py::test_pool_hidden_states_last PASSED           [  2%]
tests/test_activations.py::test_pool_hidden_states_mean PASSED           [  3%]
tests/test_activations.py::test_activation_cache_manager PASSED          [  4%]
tests/test_audited_metrics.py::test_hand_calculated_token_f1 PASSED      [  5%]
tests/test_audited_metrics.py::test_hand_calculated_concept_accuracy PASSED [  6%]
tests/test_audited_metrics.py::test_hand_calculated_exact_match_definitions PASSED [  8%]
tests/test_audited_metrics.py::test_hand_calculated_holm_bonferroni PASSED [  9%]
tests/test_audited_metrics.py::test_bootstrap_ci_bounds PASSED           [ 10%]
tests/test_causal.py::test_compute_cosine_divergence PASSED              [ 11%]
tests/test_causal.py::test_compute_kl_divergence PASSED                  [ 12%]
tests/test_causal.py::test_compute_layer_divergences PASSED              [ 13%]
tests/test_causal.py::test_rank_layers_by_causality PASSED               [ 14%]
tests/test_config.py::test_valid_config_loads_successfully PASSED        [ 16%]
tests/test_config.py::test_missing_config_raises_file_not_found PASSED   [ 17%]
tests/test_config.py::test_invalid_split_ratios_rejected PASSED          [ 18%]
tests/test_config.py::test_fallback_allowed_true_rejected PASSED         [ 19%]
tests/test_config.py::test_unsupported_model_rejected PASSED             [ 20%]
tests/test_contrastive.py::test_supcon_loss_identical_clusters_lower_loss PASSED [ 21%]
tests/test_contrastive.py::test_supcon_loss_gradients PASSED             [ 22%]
tests/test_contrastive.py::test_preservation_loss PASSED                 [ 24%]
tests/test_contrastive.py::test_extract_prompt_embeddings_mock PASSED    [ 25%]
tests/test_dataset.py::test_prompt_record_valid PASSED                   [ 26%]
tests/test_dataset.py::test_prompt_record_invalid_fields[-id_1-text-original] PASSED [ 27%]
tests/test_dataset.py::test_prompt_record_invalid_fields[   -id_1-text-original] PASSED [ 28%]
tests/test_dataset.py::test_prompt_record_invalid_fields[1--text-original] PASSED [ 29%]
tests/test_dataset.py::test_prompt_record_invalid_fields[1-id_1--original] PASSED [ 31%]
tests/test_dataset.py::test_prompt_record_invalid_fields[1-id_1-text-] PASSED [ 32%]
tests/test_dataset.py::test_duplicate_prompt_id_rejected PASSED          [ 33%]
tests/test_dataset.py::test_csv_loading_and_expected_counts PASSED       [ 34%]
tests/test_dataset.py::test_csv_variant_mapping PASSED                   [ 35%]
tests/test_dataset.py::test_csv_preserves_original_cluster_ids PASSED    [ 36%]
tests/test_dataset.py::test_csv_missing_values_rejected PASSED           [ 37%]
tests/test_dataset.py::test_cluster_split_ratio_and_cohesion PASSED      [ 39%]
tests/test_dataset.py::test_cluster_split_deterministic_reproducibility PASSED [ 40%]
tests/test_dataset.py::test_saved_split_files_on_disk PASSED             [ 41%]
tests/test_editing.py::test_collate_editing_batch PASSED                 [ 42%]
tests/test_editing.py::test_compute_editing_loss PASSED                  [ 43%]
tests/test_external_validation.py::test_external_dataset_integrity PASSED [ 44%]
tests/test_external_validation.py::test_factual_correctness_subset_integrity PASSED [ 45%]
tests/test_external_validation.py::test_zero_overlap_with_original_dataset PASSED [ 47%]
tests/test_external_validation.py::test_factual_accuracy_matching_logic PASSED [ 48%]
tests/test_external_validation.py::test_adapter_checkpoints_exist_and_valid PASSED [ 49%]
tests/test_hybrid_framework_integration.py::test_end_to_end_mock_ablation_pipeline PASSED [ 50%]
tests/test_metrics.py::test_normalize_answer PASSED                      [ 51%]
tests/test_metrics.py::test_compute_token_f1 PASSED                      [ 52%]
tests/test_metrics.py::test_compute_char_ngram_similarity PASSED         [ 54%]
tests/test_metrics.py::test_evaluate_cluster_predictions PASSED          [ 55%]
tests/test_metrics.py::test_compute_bootstrap_ci PASSED                  [ 56%]
tests/test_metrics.py::test_compare_conditions_paired PASSED             [ 57%]
tests/test_mitigation_dataset.py::test_hard_negative_generation PASSED   [ 58%]
tests/test_mitigation_dataset.py::test_validate_prompt_equivalence PASSED [ 59%]
tests/test_mitigation_dataset.py::test_cluster_dataset_pairs_and_leakage PASSED [ 60%]
tests/test_model.py::test_supported_models PASSED                        [ 62%]
tests/test_model.py::test_validate_model_target PASSED                   [ 63%]
tests/test_model.py::test_resolve_dtype PASSED                           [ 64%]
tests/test_model.py::test_build_quantization_config PASSED               [ 65%]
tests/test_model.py::test_cuda_prerequisites_enforcement PASSED          [ 66%]
tests/test_optimization.py::test_steering_zero_strength_identity PASSED  [ 67%]
tests/test_optimization.py::test_subspace_comparison_metrics PASSED      [ 68%]
tests/test_optimization.py::test_validation_objective PASSED             [ 70%]
tests/test_phase4_forensic.py::test_exact_match_disambiguation PASSED    [ 71%]
tests/test_phase4_forensic.py::test_non_inferiority_testing_logic PASSED [ 72%]
tests/test_phase4_forensic.py::test_csv_jsonl_record_reconciliation PASSED [ 73%]
tests/test_phase4_forensic.py::test_analytical_latency_accounting PASSED [ 74%]
tests/test_phase4_forensic.py::test_unanimous_hallucination_behavior PASSED [ 75%]
tests/test_phase4_forensic.py::test_alias_resolution_avoids_false_positives PASSED [ 77%]
tests/test_phase4_forensic.py::test_bounded_candidate_budget_strictness PASSED [ 78%]
tests/test_phase4_forensic.py::test_math_negative_sign_contradiction PASSED [ 79%]
tests/test_phase4_robustness.py::test_no_canonical_answer_copying PASSED [ 80%]
tests/test_phase4_robustness.py::test_zero_gold_leakage_in_mock_inference PASSED [ 81%]
tests/test_phase4_robustness.py::test_numerical_contradiction_detection PASSED [ 82%]
tests/test_phase4_robustness.py::test_predicate_entity_conflict_detection PASSED [ 83%]
tests/test_phase4_robustness.py::test_negation_contradiction_detection PASSED [ 85%]
tests/test_phase4_robustness.py::test_bounded_medoid_candidate_budget PASSED [ 86%]
tests/test_phase4_robustness.py::test_phase4_dataset_zero_leakage PASSED [ 87%]
tests/test_phase4_robustness.py::test_latency_accounting_logic PASSED    [ 88%]
tests/test_router.py::test_semantic_comparator_numerical_contradiction PASSED [ 89%]
tests/test_router.py::test_router_mode_a_single_question PASSED          [ 90%]
tests/test_router.py::test_router_mode_b_high_agreement PASSED           [ 91%]
tests/test_router.py::test_router_mode_b_disagreement_routes_to_adapter PASSED [ 93%]
tests/test_router.py::test_entity_alias_resolution_avoids_false_positive PASSED [ 94%]
tests/test_router.py::test_negation_contradiction_detection PASSED       [ 95%]
tests/test_router.py::test_numerical_percentage_contradiction PASSED     [ 96%]
tests/test_router.py::test_bounded_medoid_selection PASSED               [ 97%]
tests/test_steering.py::test_estimate_sensitivity_subspace PASSED        [ 98%]
tests/test_steering.py::test_subspace_suppression_hook PASSED            [100%]

============================= 87 passed in 4.98s ==============================
```
