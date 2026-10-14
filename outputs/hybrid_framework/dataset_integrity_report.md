# Dataset Integrity and Split Audit Report

**Project**: `prompt-sensitivity-hybrid`  
**Date**: October 12, 2026  
**Status**: Zero Leakage Verified  

---

## 1. Dataset Overview and Schema

The project utilizes two distinct dataset assets:

### 1.1 In-Distribution PromptSensitivity Dataset
- **Source**: [`data/paraphrased_prompts_axis_tagged_qa.csv`](file:///c:/Users/Student5/Desktop/Mitigation/data/paraphrased_prompts_axis_tagged_qa.csv)
- **Total Semantic Clusters**: 196
- **Total Prompt Records**: 1,176 ($6$ variants per cluster)
- **Linguistic Variation Axes**: `original`, `lexical`, `syntactic`, `pragmatic`, `length`, `format`.
- **Cluster Splits**:
  - **Train**: 137 clusters (822 records, 70%)
  - **Validation**: 29 clusters (174 records, 15%)
  - **Test (In-Distribution)**: 30 clusters (180 records, 15%)
- **Split Audit**: Cluster-stratified; zero cross-split cluster or text overlap.

### 1.2 Independent Out-of-Distribution Generalization Benchmark
- **Source**: [`data/external_eval/external_dataset.jsonl`](file:///c:/Users/Student5/Desktop/Mitigation/data/external_eval/external_dataset.jsonl)
- **Total Semantic Clusters**: 100 ($600$ prompt records)
- **Factual Correctness Subset**: 40 clusters ($240$ records) with verified domain reference answers.
- **Leakage Verification**: Automated cross-audit against all 1,176 in-distribution prompts demonstrated:
  - Exact normalized matches: **$0$**
  - Near-duplicate matches (token Jaccard $> 0.60$ and char 3-gram $> 0.65$): **$0$**

---

## 2. Hard Negative Pair Construction

To prevent contrastive representation learning from collapsing meaningful instruction distinctions into identical embeddings, [`src/mitigation/dataset.py`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/dataset.py) implements rule-based hard negative variant generation:

1. **Negation Contrast**:
   - Inverts requirements (e.g., *"Which of the following is NOT true regarding..."*).
   - Prevents the representation space from ignoring logical negation operators.
2. **Quantity Contrast**:
   - Modifies cardinality constraints (e.g., *"List exactly three distinct examples..."* vs. *"List five..."*).
3. **Scope & Entity Contrast**:
   - Modifies temporal, geographic, or categorical scopes (e.g., *"prior to 1900"* vs. modern context).

---

## 3. Split Hashes and Reproducibility

| Split | Cluster Count | Record Count | SHA-256 Cluster Hash | Role in Framework |
|---|:---:|:---:|:---:|---|
| **Train** | 137 | 822 | `9c43b81ef3908864` | Contrastive training & LoRA editing |
| **Validation** | 29 | 174 | `d55283995874c76b` | Model selection, layer search, router tuning |
| **Test (In-Dist)** | 30 | 180 | `4738da7a192bf135` | In-distribution ablation evaluation |
| **External Eval** | 100 | 600 | `b9e0783f98c43912` | Final untouched out-of-distribution evaluation |

*Rule Enforcement*: The final external test set is strictly sequestered from all hyperparameter tuning, layer selection, checkpoint selection, and routing threshold calibration.
