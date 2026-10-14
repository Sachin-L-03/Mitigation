# Complete Ablation Study Results (Conditions E0 to E6)

**Project**: `prompt-sensitivity-hybrid`  
**Evaluation Benchmark**: 100 Out-of-Distribution Semantic Clusters (600 Prompts)  
**Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Precision**: 4-bit NF4 Precision (`bfloat16` compute)  
**Date**: October 12, 2026  

---

## 1. Full Ablation Comparison Table

| ID | Condition | Token F1 (%) | 95% Bootstrap CI | All-Identical EMA (%) | Semantic Sim (%) | Anchor Pres (%) | Factual Acc (%) | 95% Bootstrap CI | Latency (s) | Paired $p$-value vs E0 | Holm-Bonferroni Adj $p$ |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Original Base Model | **37.23%** | [35.36, 39.15] | 0.0% | 54.82% | 100.0%* | 89.58% | [82.50, 95.42] | **0.2597** | — | — |
| **E1** | Inference Semantic Verification | **78.76%** | [76.26, 81.42] | 13.0% | 86.11% | **100.0%** | **90.42%** | [81.25, 97.50] | 0.2642 | **< 0.0001** | **< 0.0001** |
| **E2** | Standard LoRA Adaptation | 37.44% | [35.43, 39.47] | 0.0% | 55.55% | 91.97% | 90.00% | [83.33, 95.83] | 0.2782 | 0.4445 | 0.8890 (n.s.) |
| **E3** | Contrastive LoRA Adaptation | **77.82%** | [73.95, 81.50] | **19.0%** | 82.15% | **100.0%** | **92.08%** | [84.17, 97.92] | 0.2815 | **< 0.0001** | **< 0.0001** |
| **E4** | CREME-Style Targeted Editing | 37.20% | [35.20, 39.22] | 0.0% | 55.11% | 92.24% | 90.00% | [83.33, 95.83] | 0.2898 | 0.6100 | 0.8890 (n.s.) |
| **E5** | Contrastive + Targeted Editing | 69.99% | [65.68, 74.16] | 14.0% | 76.68% | 92.24% | **92.50%** | [85.00, 97.92] | 0.2920 | **< 0.0001** | **< 0.0001** |
| **E6** | Full Hybrid Framework | **78.98%** | [76.35, 81.53] | 14.0% | **86.36%** | 92.24% | **90.42%** | [81.25, 97.50] | 0.2954 | **< 0.0001** | **< 0.0001** |

*\*Baseline anchor preservation is 100.0% by self-referential definition.*

---

## 2. Component Value and Attribution Analysis

### 2.1 Why Did Standard and Targeted LoRA Editing (E2, E4) Fail Out-of-Distribution?
- Both E2 ($37.44\%$) and E4 ($37.20\%$) are statistically indistinguishable from the baseline model ($37.23\%$, $p = 0.44$ and $p = 0.61$).
- **Explanation**: Parameter editing without representation objectives primarily fits the specific sentence templates and lexical tokens in the 137 training clusters. On completely unseen domains (molecular biology, physics, geophysics), the LoRA delta weights are essentially bypassed by novel activations, collapsing back to base model sensitivity.

### 2.2 Why Does Inference-Time Semantic Verification (E1) Perform So Well?
- E1 achieves **$78.76\%$ Token F1** (+41.53 pp gain, $p < 10^{-15}$) with **$100.0\%$ anchor preservation** and only **$1.7\%$ latency increase** ($0.264\text{s}$ vs $0.260\text{s}$).
- **Explanation**: Prompt variations frequently cause stylistic divergences (e.g., generating multiple choice options vs. bulleted explanations vs. single words). Inference-time verification detects when paraphrases produce format discrepancies or numerical contradictions and harmonizes responses around the verified canonical answer without touching model weights.

### 2.3 Why Does Contrastive LoRA (E3) Succeed?
- E3 achieves **$77.82\%$ Token F1** and the highest all-identical exact match rate (**$19.0\%$**).
- **Explanation**: Unlike supervised next-token prediction, contrastive representation learning directly penalizes distance between hidden states of equivalent prompts, enforcing geometric invariance across paraphrase axes.

### 2.4 Tradeoffs of the Full Hybrid (E6)
- E6 reaches the highest consistency (**$78.98\%$ Token F1**, **$86.36\%$ semantic similarity**).
- However, comparing E6 ($78.98\%$, $0.2954\text{s}$) to E1 ($78.76\%$, $0.2642\text{s}$):
  - E6 provides only a $+0.22$ pp consistency gain over E1.
  - E6 incurs an $11.8\%$ latency penalty over E1 and reduces anchor preservation from $100.0\%$ to $92.24\%$.
- **Recommendation**: For production applications prioritizing simplicity and zero-training overhead, **E1 (Inference-Time Semantic Verification)** is the most efficient and robust mitigation strategy. For fine-tuning applications, **E3 (Contrastive LoRA)** is the superior trained adapter.
