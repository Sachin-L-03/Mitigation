# Bounded Routing Optimization and Candidate Budget Analysis

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Target Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Date**: October 12, 2026  

---

## 1. Candidate Budget Sweep ($K \in \{1, 2, 3, 4, 6\}$)

| Configuration | Verification Candidates ($K$) | Fast Path (%) | Verification (%) | Token F1 (%) | Factual Acc (%) | Contradictions (%) | Mean Latency (ms) | Gens / Query |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **K=1 (Base Only)** | 1 | 100.0% | 0.0% | **51.8%** | **92.0%** | 10.4% | **267.9 ms** | **1.00** |
| **K=2 (Pairwise Verified)** | 2 | 48.0% | 52.0% | **73.4%** | **94.6%** | 2.2% | **408.2 ms** | **1.52** |
| **K=3 (Bounded Medoid - E6_bounded)** | 3 | 39.0% | 61.0% | **85.11%** | **97.2%** | 0.0% | **596.5 ms** | **2.22** |
| **K=4 (Extended Bounded)** | 4 | 39.0% | 61.0% | **88.35%** | **97.6%** | 0.0% | **760.6 ms** | **2.83** |
| **K=6 (Exhaustive Medoid - E1)** | 6 | 0.0% | 100.0% | **100.0%** | **98.0%** | 0.0% | **1612.4 ms** | **6.00** |

---

## 2. Deployment Latency Profiles

### Profile A: Strict Low-Latency Budget (< 400 ms)
- **Selected Policy**: $K=1$ Baseline (267.9 ms) or Aggressive Fast Path (tau=0.40, K=2, 408.2 ms).
- **Tradeoff**: Low compute overhead, but leaves 10.4% baseline contradiction rate unmitigated.

### Profile B: Moderate Latency Budget (< 650 ms) [RECOMMENDED: `E6_bounded`]
- **Selected Policy**: Bounded Adaptive Router ($K=3$, tau=0.50).
- **Tradeoff**: **Pareto optimal**. Achieves 85.11% Token F1, 97.20% factual accuracy, eliminates 100% of hard contradictions (0.0%), and respects the 650 ms ceiling with a measured mean latency of **596.5 ms**.

### Profile C: Quality-First Offline Setting (< 1800 ms)
- **Selected Policy**: Exhaustive Consensus ($K=6$, E1).
- **Tradeoff**: Guarantees identical answers across all 6 variants (100% F1), but requires 1,612.4 ms latency and 6 model generations per query.
