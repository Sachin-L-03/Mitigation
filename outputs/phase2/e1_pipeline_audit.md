# E1 Inference-Time Pipeline Audit Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Target Module**: `src/mitigation/router.py`, `scripts/run_hybrid_ablation.py`  
**Date**: October 12, 2026  
**Auditor**: ML Research Engineering Audit Team  

---

## 1. Overview of the Claimed E1 Pipeline

Condition E1 was introduced as **Inference-Time Semantic Verification**, a training-free mitigation technique that operates purely at inference time to resolve sensitivity variations across semantically equivalent queries.

The project report claimed:
- Token F1: **78.76%** (vs 37.23% for E0 base model)
- Semantic Similarity: **86.11%** (vs 54.82% for E0 base model)
- Anchor Preservation: **100.00%**
- Latency: **0.2642 s** ($+1.7\%$ over E0 base model)

---

## 2. Actual E1 Algorithm: Audited Pseudocode

Based on forensic inspection of [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py#L121-L138), the actual algorithm executed was:

```python
# Actual Algorithm Executed in Phase 1 (Post-hoc Consensus Simulation)
def evaluate_condition_e1(cached_predictions_by_cluster, semantic_comparator):
    e1_by_cluster = {}
    
    for cluster_id, variants in cached_predictions_by_cluster.items():
        # ORACLE ASSUMPTION: The original prompt's answer is already generated and known
        orig_text = variants["original"]
        e1_variants = {}
        
        for variant_type, text in variants.items():
            if variant_type == "original":
                # TAUTOLOGICAL ANCHOR PRESERVATION:
                # The canonical prompt's answer is always preserved untouched
                e1_variants["original"] = orig_text
                continue
            
            # Step 1: Compare the variant answer against the canonical original answer
            comp = semantic_comparator.compare_answers(orig_text, text)
            
            # Step 2: Decide whether to keep or overwrite
            if comp["agrees"] and not comp["numerical_contradiction"]:
                # If answers agree and have no numeric contradiction, retain variant output
                e1_variants[variant_type] = text
            else:
                # DISAGREEMENT DETECTED:
                # Overwrite the variant answer with the exact string of the canonical original!
                e1_variants[variant_type] = orig_text
                
        e1_by_cluster[cluster_id] = e1_variants
        
    return e1_by_cluster
```

---

## 3. Disconnect Between Theoretical Pipeline and Actual Execution

The project documentation in [`src/mitigation/router.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/router.py) specified two operational deployment modes:

### Mode A: Single User Prompt
1. User provides prompt $q$.
2. Model generates initial answer $y_0$.
3. System generates $K$ paraphrases $\{p_1, \dots, p_K\}$ of $q$.
4. Model generates answers $\{y_1, \dots, y_K\}$ for each paraphrase.
5. `SemanticAnswerComparator` computes pairwise agreement across all candidates.
6. If consensus exists, select medoid or majority answer; if disagreement, trigger verification or adapter.

### Mode B: Explicit Paraphrase Set
1. User supplies a cluster of known equivalent prompts $\{q_1, \dots, q_K\}$.
2. Model generates candidate answers for each.
3. System selects the medoid candidate that minimizes semantic disagreement across the set.

### What Actually Ran in Phase 1:
- In [`scripts/run_hybrid_ablation.py`](file:///c:/Users/Student5/Desktop/Mitigation/scripts/run_hybrid_ablation.py), **neither Mode A nor Mode B was executed as an autonomous generation pipeline**.
- Instead, the script read `baseline_external_predictions.jsonl` from disk, designated `variants["original"]` as the oracle canonical answer, and whenever a variant disagreed with it, replaced the variant's answer with `variants["original"]`.

---

## 4. Key Scientific Flaws in the Phase 1 E1 Evaluation

1. **Oracle Canonical Anchor**: The pipeline assumed that `original` is a privileged anchor whose response is superior to all other variants. In real deployment, the incoming query might be a syntactic or lexical variant; the system does not know which query is the "original canonical" version.
2. **String Overwriting (Copying)**: Replacing variant answers with the anchor string converts a disagreement resolution problem into a string-copying operation.
3. **Artificial Anchor Preservation**: The metric $P = \text{sim}(y_{\text{canon}}^{\text{E1}}, y_{\text{canon}}^{\text{base}})$ was $100\%$ simply because line 128 explicitly assigns `e1_variants["original"] = orig_text`.
4. **No External Verification Model**: The method did not use a separate verifier LLM, API, or retriever; it used purely rule-based regex and token overlap.

---

## 5. Correct Formulation for Genuine Inference Verification

To evaluate genuine inference verification without cheating:
1. **Medoid Consensus Selection**: When given candidate answers $\{y_1, \dots, y_K\}$, compute pairwise semantic similarity matrix $S_{ij}$. Select the candidate $y^*$ that maximizes mean similarity to all other candidates:
   $$y^* = \arg\max_{y_i} \sum_{j=1}^K S_{ij}$$
2. **No Privileged Anchor**: Treat all variants symmetrically without assuming $y_{\text{original}}$ is the gold standard.
3. **End-to-End Accounting**: Account for all generation passes and comparison latency.
