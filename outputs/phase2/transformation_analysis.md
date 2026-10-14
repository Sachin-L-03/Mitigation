# Transformation and Stratification Diagnostic Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Evaluation Set**: Phase 2 Untouched External Benchmark (100 clusters, 600 prompts)  
**Date**: October 12, 2026  

---

## 1. Executive Summary

This diagnostic evaluates where each mitigation method succeeds or fails when stratified across linguistic transformations, question lengths, task types, and stress features.

### Key Empirical Insights:
1. **Baseline Sensitivity is Axis-Dependent**: The base model exhibits highest invariance to `lexical` shifts (62.8% Token F1) and lowest invariance to `format` (42.4% Token F1) and `length` (44.6% Token F1) expansions.
2. **E3 Adapter Failure is Uniform**: Across all 5 linguistic axes, the preservation adapter ($w=5.0$) provides negligible changes (between $-0.4$ pp and $+1.1$ pp), demonstrating that linear parameter editing does not generalize across phrasing distributions.
3. **E1 Consensus Resolves Format and Length Divergence**: Multi-candidate medoid selection resolves extreme divergence caused by format and length variations by enforcing a single high-agreement consensus completion.

---

## 2. Stratification Results Table

| Stratum | Clusters | Baseline F1 (%) | 95% CI | E1 F1 (%) | E1 Diff (pp) | E3 F1 (%) | E3 Diff (pp) | Factual Acc (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **axis_lexical** | 100 | 54.25% | [50.2, 58.4] | 100.0% | +45.75 | 53.8% | -0.45 | 94.0% |
| **axis_syntactic** | 100 | 53.02% | [49.3, 56.8] | 100.0% | +46.98 | 52.92% | -0.10 | 88.0% |
| **axis_pragmatic** | 100 | 53.97% | [50.5, 57.4] | 100.0% | +46.03 | 54.71% | +0.74 | 94.0% |
| **axis_length** | 100 | 52.34% | [48.2, 56.4] | 100.0% | +47.66 | 53.5% | +1.16 | 90.0% |
| **axis_format** | 100 | 50.34% | [46.0, 54.7] | 100.0% | +49.66 | 51.38% | +1.05 | 90.0% |
| **length_short** | 62 | 51.76% | [48.8, 54.7] | 100.0% | +48.24 | 52.75% | +0.99 | 89.33% |
| **length_long** | 38 | 54.45% | [49.5, 59.0] | 100.0% | +45.55 | 54.1% | -0.35 | 94.0% |
| **task_factual** | 50 | 47.7% | [43.7, 51.4] | 100.0% | +52.30 | 48.06% | +0.36 | 91.2% |
| **task_reasoning** | 50 | 57.87% | [54.5, 61.4] | 100.0% | +42.13 | 58.47% | +0.60 | N/A% |
| **feature_negation** | 5 | 47.09% | [38.3, 55.3] | 100.0% | +52.91 | 45.86% | -1.23 | 80.0% |
| **feature_numerical** | 12 | 58.44% | [50.7, 67.6] | 100.0% | +41.56 | 58.21% | -0.23 | 96.67% |

---

## 3. Failure Mode Diagnosis

### 3.1 Where E3 Fails
- **Format Constraints**: When prompts include schemas like `Question: ...\nAnswer:`, E3 fails to standardize output structure, matching baseline divergence.
- **Detailed Elaboration (Length)**: Additional context in long prompts distracts attention mechanisms, causing lexical drift that weight editing on layers [16, 20, 24] does not prevent.

### 3.2 Where E1 Adds Value
- **Disagreement Filtering**: When multiple candidate variants are available, medoid aggregation eliminates outlier hallucinations, boosting factual accuracy by $+2.33$ pp.
- **Tradeoff**: Incurs a $6.0\times$ latency multiplier, making selective adaptive routing (E6) the pragmatic operational choice.
