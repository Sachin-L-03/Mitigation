# Independent Generalization and Cross-Model Validation Report

**Project**: `prompt-sensitivity-hybrid`  
**Evaluation Scope**: Out-of-Distribution Independent Generalization and Factual Correctness Audit  
**Date**: October 12, 2026  
**Hardware Target**: NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Model Under Test**: `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 precision)  
**Cross-Model Candidate**: `meta-llama/Llama-3.1-8B-Instruct` (Status: Access Blocked)  

---

## Executive Summary

This study conducts an independent out-of-distribution evaluation and factual correctness audit of the best candidate intervention identified in prior experiments: **preservation-tuned parameter editing with $w_{\text{pres}}=5.0$** (`outputs/optimization/preservation/adapter_w5.0`), compared against the **baseline parameter editing adapter with $w_{\text{pres}}=1.0$** (`outputs/parameter_editing/best_adapter`) and the **unmodified base model** (`Qwen/Qwen2.5-7B-Instruct`).

Prior in-distribution evaluations on 30 test clusters (180 prompts) reported an increase in Token F1 from 37.03% to 45.82%. To test whether this reflects genuine prompt invariance or in-distribution memorization, we constructed an independent benchmark of **100 brand-new semantic clusters (600 prompt records)** with verified zero overlap against the original training data. Furthermore, a **40-cluster (240 prompt) factual correctness subset** with independently verified ground-truth reference answers was evaluated to separate linguistic paraphrase consistency from factual accuracy.

### Key Finding

| Condition | Token F1 (%) | 95% Bootstrap CI | Semantic Sim (%) | Anchor Pres (%) | Factual Acc (%) | 95% Bootstrap CI | Latency (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Condition A: Original Baseline** | **37.23%** | [35.36%, 39.15%] | 54.82% | 100.00%* | 89.58% | [82.50%, 95.42%] | **0.2597** |
| **Condition B: Editing Only ($w=1.0$)** | 37.44% | [35.43%, 39.47%] | 55.55% | 91.97% | 90.00% | [83.33%, 95.83%] | 0.2782 |
| **Condition C: Preservation-Tuned ($w=5.0$)** | 37.20% | [35.20%, 39.22%] | 55.11% | 92.24% | 90.00% | [83.33%, 95.83%] | 0.2898 |

*\*Baseline anchor preservation is 100.0% by self-referential definition.*

**Core Conclusion**: On completely independent, out-of-distribution semantic clusters, **the preservation-tuned method does not generalize**. The within-cluster Token F1 of the preservation-tuned adapter (37.20%) is statistically indistinguishable from the original baseline (37.23%, paired Wilcoxon $p = 0.6100$, Holm-Bonferroni adjusted $p = 1.00$). While both adapters preserve factual correctness (~90.00% accuracy), they offer no prompt invariance benefits beyond the training distribution while incurring an 11.6% inference latency penalty.

---

## 1. Audit and Freeze of Existing Artifacts

Before initiating new evaluations, all previously trained checkpoints, configurations, and metric routines were audited and frozen:

1. **Checkpoints**:
   - `outputs/parameter_editing/best_adapter/adapter_model.safetensors` (161.5 MB, verified intact).
   - `outputs/optimization/preservation/adapter_w5.0/adapter_model.safetensors` (161.5 MB, verified intact).
2. **Metric Definitions**:
   - **Within-Cluster Consistency Proxies**: Token F1, Exact Match Agreement (EMA), and character 3-gram cosine similarity between variant completions and the model's completion on the original prompt within the same semantic cluster.
   - **Anchor Preservation**: Evaluated strictly as the Token F1 between the intervention's completion on the `original` prompt and the *unmodified baseline model's* completion on that identical prompt. In prior validation sweeps, $w=1.0$ was compared to itself yielding 100.0%; this is now explicitly labeled as a self-referenced control artifact, whereas true test anchor preservation relative to base Qwen was 33.57% ($w=1.0$) and 38.58% ($w=5.0$).
3. **Statistical Integrity**: All comparisons use cluster-matched observations with two-sided Wilcoxon signed-rank tests and Holm-Bonferroni family-wise error rate corrections.

---

## 2. Independent Evaluation Dataset Construction

To eliminate data leakage, an independent benchmark was constructed under `data/external_eval/`:

- **Scale**: 100 distinct semantic question clusters, each featuring an `original` prompt and 5 systematic linguistic variants (`lexical`, `syntactic`, `pragmatic`, `length`, `format`), totaling **600 prompt records**.
- **Domains Covered**: Molecular biology, neurobiology, clinical cardiology, classical mechanics, electromagnetism, geophysics, paleomagnetism, atmospheric science, organic chemistry, cognitive architecture, and applied engineering.
- **Linguistic Axes**:
  - `lexical`: Domain-matched synonym substitutions and phrasing shifts.
  - `syntactic`: Passive/active inversions, fronted adverbial clauses, and cleft sentence structures.
  - `pragmatic`: Contextual personas (e.g., clinician, tutor, telecommunications engineer).
  - `length`: Detailed domain elaboration providing extensive scientific background.
  - `format`: Structured query-response schemas (e.g., `Question: ...\nAnswer with exact name:`).
- **Leakage & Overlap Audit**:
  - Script: [`dataset_external.py`](file:///c:/Users/Student5/Desktop/Mitigation/dataset_external.py)
  - Report: [`outputs/external_validation/dataset_overlap_audit.json`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/dataset_overlap_audit.json)
  - Existing prompt records audited: 1,176.
  - Total cross-comparisons: 117,600.
  - Exact normalized matches: **0 (0.0%)**.
  - Near-duplicate matches (token Jaccard > 0.60, char 3-gram > 0.65): **0 (0.0%)**.
  - **Zero overlap verified**.

---

## 3. Factual Correctness Evaluation Subset

The original dataset contained no reference answers. We created a separate, verified subset to evaluate whether interventions maintain factual integrity:

- **File**: [`data/external_eval/correctness_subset.jsonl`](file:///c:/Users/Student5/Desktop/Mitigation/data/external_eval/correctness_subset.jsonl)
- **Scale**: 40 semantic clusters (240 prompt records).
- **Ground Truth**: Each cluster is assigned an independently verified ground-truth reference answer derived from standard scientific literature (e.g., *Glutamate* for CNS excitatory transmission; *ACE* for angiotensin conversion; *Mitochondria* for oxidative phosphorylation; *Ohm* for resistance; *O negative* for universal RBC donation; *Insulin* for beta-cell glucose reduction; *101.325 kPa* for standard sea-level pressure).
- **Evaluation Criteria**:
  - **Factual Accuracy**: Presence of the gold entity/concept within the generated completion (case-insensitive normalized matching).
  - **Factual Token F1**: Word-level F1 between generated completion and reference answer.
- **Distinction from Consistency**: Paraphrase consistency measures invariant outputs across phrasing variants. A model can be 100% consistent while consistently hallucinating. Therefore, factual correctness is reported strictly as a domain accuracy metric against verified facts.

---

## 4. Empirical Evaluation on Qwen2.5-7B-Instruct

All 3 conditions were evaluated under identical conditions:
- **Precision**: 4-bit NF4 (`bitsandbytes`, double quantization, bfloat16 compute).
- **Decoding**: Greedy decoding (`do_sample=False`, `temperature=0.0`, `max_new_tokens=64`).
- **Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM), ~12.5 GB allocated during batched generation.

### Detailed Metric Breakdown

| Metric | Condition A (Baseline) | Condition B ($w=1.0$) | Condition C ($w=5.0$) |
|---|:---:|:---:|:---:|
| **Within-Cluster Token F1** | **37.23%** $\pm$ 1.90% | **37.44%** $\pm$ 2.02% | **37.20%** $\pm$ 2.01% |
| *95% Bootstrap CI* | [35.36%, 39.15%] | [35.43%, 39.47%] | [35.20%, 39.22%] |
| **Exact Match Agreement (EMA)** | 0.00% | 0.00% | 0.00% |
| **Semantic Similarity (Char 3-gram)** | 54.82% $\pm$ 2.02% | 55.55% $\pm$ 2.00% | 55.11% $\pm$ 2.15% |
| *95% Bootstrap CI* | [52.68%, 56.72%] | [53.44%, 57.44%] | [52.90%, 57.19%] |
| **Anchor Preservation** | 100.00%* | 91.97% $\pm$ 2.70% | 92.24% $\pm$ 3.06% |
| *95% Bootstrap CI* | [100.0%, 100.0%] | [89.19%, 94.59%] | [88.94%, 95.05%] |
| **Factual Accuracy (40 Clusters)** | 89.58% $\pm$ 6.46% | 90.00% $\pm$ 6.25% | 90.00% $\pm$ 6.25% |
| *95% Bootstrap CI* | [82.50%, 95.42%] | [83.33%, 95.83%] | [83.33%, 95.83%] |
| **Factual Token F1** | 9.02% | 9.01% | 9.02% |
| **Average Answer Length** | 48.76 words | 48.91 words | 48.88 words |
| **Output Diversity Ratio** | 0.8017 | 0.7917 | 0.8067 |
| **Non-Empty Response Rate** | 100.0% | 100.0% | 100.0% |
| **Inference Latency** | 0.2597 s/prompt | 0.2782 s/prompt | 0.2898 s/prompt |

*\*Baseline anchor preservation is 100.0% by self-referential definition.*

### Statistical Hypothesis Testing (Paired Cluster Observations)

All tests are cluster-level paired two-sided Wilcoxon signed-rank tests:

1. **Condition B ($w=1.0$) vs. Condition A (Baseline)**:
   - **Token F1**: Absolute change $+0.21$ pp ($+0.59\%$ relative), $p = 0.4445$ (Not Significant; Holm-Bonferroni $p_{\text{adj}} = 1.00$).
   - **Semantic Similarity**: Absolute change $+0.73$ pp ($+1.33\%$ relative), $p = 0.0063$ (Significant without correction, but marginal effect size).
   - **Factual Accuracy**: Absolute change $+0.42$ pp ($+0.47\%$ relative), $p = 0.3173$ (Not Significant).

2. **Condition C ($w=5.0$) vs. Condition A (Baseline)**:
   - **Token F1**: Absolute change $-0.03$ pp ($-0.06\%$ relative), $p = 0.6100$ (Not Significant; Holm-Bonferroni $p_{\text{adj}} = 1.00$).
   - **Semantic Similarity**: Absolute change $+0.29$ pp ($+0.52\%$ relative), $p = 0.1376$ (Not Significant).
   - **Factual Accuracy**: Absolute change $+0.42$ pp ($+0.47\%$ relative), $p = 0.3173$ (Not Significant).

3. **Condition C ($w=5.0$) vs. Condition B ($w=1.0$)**:
   - **Token F1**: Absolute change $-0.24$ pp ($-0.65\%$ relative), $p = 0.7843$ (Not Significant).
   - **Anchor Preservation**: Absolute change $+0.27$ pp ($+0.29\%$ relative), $p = 0.7731$ (Not Significant).
   - **Factual Accuracy**: Absolute change $0.00$ pp, $p = 1.0000$ (Identical).

---

## 5. Cross-Model Evaluation Status: Meta LLaMA-3.1-8B-Instruct

We checked access to `meta-llama/Llama-3.1-8B-Instruct` via the Hugging Face API:

- **Authentication Check**: Programmatic probing via `AutoConfig.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')` failed with:
  `401 Client Error: Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`.
- **Root Cause**: The repository is gated by Meta and requires an active, approved Hugging Face access token (`HF_TOKEN`). No token was provided in the local environment.
- **Execution Protocol**: Per Section 5 of the project instructions, this external blocker is formally documented, all Qwen experiments were fully completed, and model weights were not substituted with unauthorized alternatives.

---

## 6. Synthesis and Answers to the Core Research Questions

### 1. Does the preservation-tuned method generalize to new semantic questions?
**No.**  
On 100 independent semantic clusters (600 prompts) with zero training overlap, the preservation-tuned adapter ($w_{\text{pres}}=5.0$) achieves **37.20% Token F1**, compared to **37.23%** for the unmodified baseline model. The absolute change of $-0.03$ percentage points is completely within the noise floor ($p = 0.6100$). The 45.82% Token F1 reported on the preliminary test set was a consequence of in-distribution dataset similarity rather than true semantic invariance.

### 2. Does it retain correctness while improving consistency?
**It retains correctness, but fails to improve consistency out-of-distribution.**  
On the 40 verified factual QA clusters, factual accuracy remained high and stable:
- Baseline: 89.58% accuracy
- $w_{\text{pres}}=1.0$: 90.00% accuracy
- $w_{\text{pres}}=5.0$: 90.00% accuracy  
The difference ($+0.42$ pp, $p = 0.3173$) demonstrates that parameter editing does not induce catastrophic forgetting or factual degradation. However, because within-cluster consistency did not improve out-of-distribution (37.20% vs. 37.23%), the intended dual objective was not realized on independent data.

### 3. Does the improvement transfer to Llama?
**Empirical execution on Llama was blocked due to repository gating (401 Client Error).**  
However, the empirical evidence from Qwen indicates that low-rank parameter editing trained on 137 specific question clusters acts as localized memorization rather than a universal sensitivity-mitigation mechanism. Therefore, transferring the same methodology to Llama without broad, multi-task training data is strongly predicted to yield the same out-of-distribution limitation.

### 4. Is the improvement statistically reliable on the independent evaluation set?
**No.**  
The paired Wilcoxon signed-rank test yields $p = 0.6100$, and the Holm-Bonferroni adjusted $p$-value is $1.00$. The 95% bootstrap confidence interval for the preservation-tuned adapter ([35.20%, 39.22%]) almost completely overlaps the baseline model's confidence interval ([35.36%, 39.15%]). There is no statistically reliable difference.

### 5. Which method should be retained?
**Retain the unmodified Baseline model (Condition A) for production and general deployment.**  
Both the baseline editing adapter ($w=1.0$) and the preservation-tuned adapter ($w=5.0$) add operational complexity and an **11.6% increase in per-prompt latency** (0.260s vs. 0.290s) without providing any statistically significant consistency gains on unseen questions. Parameter editing should only be retained if deployed strictly within the specific question domain on which it was fine-tuned.

---

## 7. Deliverables Checklist

The following deliverables have been generated and saved under `outputs/external_validation/`:
- [`dataset_overlap_audit.json`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/dataset_overlap_audit.json): Verification of 0 exact and 0 near matches.
- [`correctness_evaluation_report.json`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/correctness_evaluation_report.json): Factual accuracy and token F1 against gold references.
- [`baseline_external_predictions.jsonl`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/baseline_external_predictions.jsonl): 600 completions for Condition A.
- [`editing_w1_external_predictions.jsonl`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/editing_w1_external_predictions.jsonl): 600 completions for Condition B.
- [`preservation_w5_external_predictions.jsonl`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/preservation_w5_external_predictions.jsonl): 600 completions for Condition C.
- [`per_cluster_metrics.json`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/per_cluster_metrics.json): Granular metric logs for all 100 clusters across all conditions.
- [`external_validation_table.csv`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/external_validation_table.csv): Structured summary table.
- [`external_validation_summary.json`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/external_validation_summary.json): Complete bootstrap intervals and hypothesis test results.
- [`external_validation_report.md`](file:///c:/Users/Student5/Desktop/Mitigation/outputs/external_validation/external_validation_report.md): This comprehensive research report.
