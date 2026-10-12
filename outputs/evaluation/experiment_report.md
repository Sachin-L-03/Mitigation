# Prompt-Sensitivity-Hybrid: Research Evaluation Report

**Evaluation Split**: `test` | **Target GPU**: NVIDIA RTX 4090 | **Quantization**: 4-bit NF4

## 1. Experimental Conditions Comparison

| Condition | Exact Match (EMA) | Token F1 | Semantic Sim | Anchor Pres. | Diversity | Avg Length | Δ F1 vs Base (pp) | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A: Baseline** | 1.3% [0.0-3.3] | 37.0% [32.0-42.2] | 52.9% | 100.0% | 98.3% | 52.1 w | - | - |
| **Condition B: Steering Only** | 1.3% [0.0-3.3] | 36.5% [31.2-42.0] | 52.4% | 62.7% | 98.9% | 52.6 w | -0.53 pp | 0.3285 |
| **Condition C: Editing Only** | 2.0% [0.0-4.7] | 45.6% [41.1-50.8] | 60.5% | 33.6% | 97.8% | 52.7 w | +8.56 pp | 0.0076 * |
| **Condition D: Hybrid (Edit + Steer)** | 4.0% [1.3-8.0] | 43.2% [38.3-48.7] | 56.9% | 31.6% | 95.6% | 52.6 w | +6.14 pp | 0.0879 |

> `*` indicates statistical significance at $p < 0.05$ (Wilcoxon signed-rank test against baseline).

## 2. Key Findings & Hypothesis Evaluation

- **Baseline Performance**: Mean Token F1 across paraphrases is **37.03%**.
- **Representation Steering Only**: Token F1 is **36.50%** (Δ = -0.53 pp).
- **Parameter Editing Only**: Token F1 is **45.59%** (Δ = +8.56 pp).
- **Hybrid Intervention**: Token F1 is **43.17%** (Δ = +6.14 pp).

## 3. Methodological Notes
- **Reference Answer Notice**: Source dataset contains no gold answers; reference answers are strictly null.
- **Factual Correctness**: High paraphrase agreement reflects invariance to linguistic phrasing, not factual verification.
- **Preservation Regularization**: Monitored via anchor preservation on original prompts to detect collapse or regression.
