# Latency, Compute Cost, and Memory Benchmark Analysis

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Hardware**: NVIDIA GeForce RTX 4090 (24,564 MiB VRAM), CUDA 13.3, Windows 11  
**Date**: October 12, 2026  

---

## 1. Latency Distribution Profiling

| Configuration | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Slowdown vs Base | Latency Savings vs E1 |
|---|:---:|:---:|:---:|:---:|:---:|
| **K=1 (Base Only)** | 267.9 ms | 267.9 ms | 267.9 ms | 1.00x | 83.4% |
| **K=2 (Pairwise Verified)** | 408.2 ms | 537.8 ms | 537.8 ms | 1.52x | 74.7% |
| **K=3 (Bounded Medoid - E6_bounded)** | 596.5 ms | 806.7 ms | 806.7 ms | 2.23x | 63.0% |
| **K=4 (Extended Bounded)** | 760.6 ms | 1075.6 ms | 1075.6 ms | 2.84x | 52.8% |
| **K=6 (Exhaustive Medoid - E1)** | 1612.4 ms | 1612.4 ms | 1612.4 ms | 6.02x | 0.0% |

---

## 2. GPU VRAM Allocation and Compute Footprint

- **Base Model Weights (4-bit NF4)**: 5.62 GB
- **Peak KV Cache during Generation (Batch Size 8, Max Tokens 128)**: 6.95 GB
- **Total Peak Allocated VRAM**: **12.63 GB** (51.4% of 24 GB RTX 4090 capacity)
- **Idle Background VRAM**: 1.16 GB
- **Cold-Start Model Load Time**: 7.82 seconds
- **Warm Single-Batch Forward Pass**: ~1.85 seconds for 8 prompts (231 ms / prompt)
