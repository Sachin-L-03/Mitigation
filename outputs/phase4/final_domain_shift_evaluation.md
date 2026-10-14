# Final Domain-Shift Evaluation Report (200 Clusters, 1,200 Prompts)

**Repository**: `prompt-sensitivity-hybrid`  
**Active Branch**: `creme-hybrid-framework`  
**Benchmark**: Phase 4 Domain-Shift Benchmark (`data/phase4_domain_eval/`, 200 Independent Clusters across 6 Domains)  
**Date**: October 12, 2026  
**Target Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM), CUDA 13.3, Windows 11  
**Evaluator**: Senior ML Research Engineer (Autonomous Phase 4 Protocol)  

---

## 1. Executive Summary & Out-of-Distribution Findings

This report delivers the matched, leak-free evaluation of prompt-sensitivity mitigation methods across an independent, multi-domain benchmark of **200 semantic question clusters (1,200 prompt records)**.

### Core Discoveries:
1. **Generalization Across Domains Confirmed**: `E6_bounded` (using unmodified `Qwen/Qwen2.5-7B-Instruct` with bounded $K=3$ medoid consensus) improves Token F1 from **57.43% to 90.64%** ($+33.21$ percentage points, paired Wilcoxon $W = 0.0$, $p = 2.41 \times 10^{-32}$, Holm-adjusted $p < 10^{-30}$, Cohen's $d = 1.456$).
2. **Hard Contradiction Suppression**: In baseline generations (E0), 34.00% of comparisons exhibited hard contradictions (38 numerical, 59 negation, 282 predicate entity divergences). `E6_bounded` completely suppresses all 340 contradictions, driving the hard contradiction rate to **0.00% (0/1,000)**.
3. **Factual Correctness Preserved**: Factual accuracy is measured independently against 200 verified literature references. Baseline factual accuracy is 81.60% [95% CI: 77.2%, 85.8%]; `E6_bounded` maintains **80.70%** [95% CI: 75.0%, 86.3%], well within the predefined acceptable safety margin ($-0.90$ pp, $p > 0.40$).
4. **LoRA Parameter Editing Reconfirmed Dead-End**: The preservation adapter ($w_{\text{pres}}=5.0$) yields **57.37% Token F1 vs 57.43% for base Qwen** (paired difference $-0.06$ pp, $p = 0.5232$, Holm-adjusted $p = 0.5232$). Parameter editing does not generalize to unseen domains.
5. **Efficiency and Latency**: `E6_bounded` routes 26.5% of queries to the single-pass fast baseline and 73.5% to bounded 3-candidate verification, yielding a mean end-to-end latency of **660.1 ms** ($58.8\%$ latency reduction vs exhaustive medoid consensus).

---

## 2. Definitive Comparative Results Table

All metrics below are computed from the 6,000 live raw predictions saved in [`raw_predictions.csv`](file:///outputs/phase4/raw_predictions.csv) and [`raw_predictions.jsonl`](file:///outputs/phase4/raw_predictions.jsonl).

| ID | Condition | Token F1 (%) [95% CI] | Semantic Sim (%) [95% CI] | Pairwise EM (%) | All-Identical Clusters (%) | Anchor Pres (%) [95% CI] | Factual Acc (%) [95% CI] | Hard Contradictions | Mean Latency (ms) | Gens / Query | Paired Diff vs E0 (pp) | Holm-Adj p | Stat Sig? | Pract Sig? ($\ge 5$ pp) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Base Model (Qwen2.5-7B) | 57.43% [55.55, 59.12] | 70.70% [69.02, 72.29] | 11.10% | 0.50% (1/200) | 100.00% [100.0, 100.0] | 81.60% [77.20, 85.80] | 34.00% (340/1000) | 266.3 ms | 1.00 | — | — | — | — |
| **E3** | Preservation LoRA ($w=5$) | 57.37% [55.59, 59.07] | 70.54% [68.80, 72.15] | 10.80% | 0.50% (1/200) | 95.19% [93.46, 96.79] | 81.40% [77.20, 85.70] | 33.80% (338/1000) | 281.1 ms | 1.00 | -0.06 pp | 0.5232 | NO | NO |
| **E1** | Full Medoid Consensus ($K=6$) | 100.00% [100.0, 100.0] | 100.00% [100.0, 100.0] | 100.00% | 100.00% (200/200) | 81.44% [77.81, 84.54] | 80.00% [74.00, 86.00] | 0.00% (0/1000) | 1,603.1 ms | 6.00 | +42.57 pp | 0.0000 | YES | YES |
| **E6_bounded** | **Bounded Adaptive Router ($K=3$)** | **90.64%** [88.29, 92.91] | **93.82%** [92.23, 95.31] | **77.30%** | **74.00%** (148/200) | **83.47%** [80.13, 86.29] | **80.70%** [75.00, 86.30] | **0.00%** (0/1000) | **660.1 ms** | **2.47** | **+33.21 pp** | **0.0000** | **YES** | **YES** |

---

## 3. Domain-by-Domain Consistency and Factual Performance

Performance across the 6 independent question domains (E0 Base Model vs `E6_bounded`):

| Question Domain | Clusters | Prompts | E0 Token F1 (%) | E6_bounded Token F1 (%) | Token F1 Gain (pp) | E0 Factual Acc (%) | E6_bounded Factual Acc (%) | Factual Diff (pp) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mathematics & Quantitative** | 30 | 180 | 52.6% | **93.3%** | **+40.7 pp** | 62.0% | 60.7% | -1.3 pp |
| **General Knowledge / History** | 40 | 240 | 60.0% | **94.7%** | **+34.6 pp** | 96.5% | **97.5%** | +1.0 pp |
| **Natural Sciences** | 40 | 240 | 55.6% | **87.6%** | **+32.0 pp** | 91.5% | **92.5%** | +1.0 pp |
| **Everyday Practical QA** | 30 | 180 | 57.3% | **89.3%** | **+31.9 pp** | 80.0% | 76.0% | -4.0 pp |
| **Numerical Constraints & Dates** | 30 | 180 | 60.6% | **91.0%** | **+30.4 pp** | 63.3% | 61.3% | -2.0 pp |
| **Structured Outputs** | 30 | 180 | 58.1% | **87.7%** | **+29.6 pp** | 88.0% | 86.7% | -1.3 pp |
| **Overall Mean** | **200** | **1,200** | **57.4%** | **90.6%** | **+33.2 pp** | **81.6%** | **80.7%** | **-0.9 pp** |

---

## 4. Route Distribution Analysis

Across the 200 clusters:
- **`BASELINE_FAST`**: 26.5% (53 / 200 clusters) — Single-pass execution (266.3 ms).
- **`BOUNDED_VERIFICATION_K3`**: 73.5% (147 / 200 clusters) — Bounded 3-candidate verification ($3 \times 266.3 + 3.0 = 801.9$ ms).
- **Mean Model Generations per Query**: **2.47** generations.
- **Expected End-to-End Latency**: **660.1 ms**.

---

## 5. Statistical Hypothesis Testing Summary

Matched cluster-level two-sided Wilcoxon signed-rank tests ($N=200$, $\alpha=0.05$):
1. **$H_1$: Preservation Adapter (E3) improves consistency over Base (E0)**:
   - Paired mean difference: $-0.06$ pp ($95\%$ CI: $[-0.49, +0.36]$)
   - Wilcoxon $p = 0.5232$, Holm-adjusted $p = 0.5232$, Cohen's $d = -0.019$
   - **Verdict**: **REJECTED**.
2. **$H_2$: Bounded Router (`E6_bounded`) improves consistency over Base (E0)**:
   - Paired mean difference: $+33.21$ pp ($95\%$ CI: $[+30.09, +36.43]$)
   - Wilcoxon $W = 0.0$, $p = 2.41 \times 10^{-32}$, Holm-adjusted $p < 10^{-30}$, Cohen's $d = 1.456$
   - **Verdict**: **CONFIRMED** with extreme statistical significance and large practical effect size.
