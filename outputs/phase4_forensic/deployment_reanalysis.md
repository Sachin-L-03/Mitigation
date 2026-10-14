# Phase 4 Deployment Latency and Serving Reanalysis

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Target Architecture**: `E6_bounded` (Cost-Aware Bounded Adaptive Router)  
**Host Hardware**: NVIDIA GeForce RTX 4090 (24,564 MiB VRAM), Windows 11, CUDA 13.3  
**Audited Artifacts**: [`outputs/phase4/deployment_benchmark.md`](file:///outputs/phase4/deployment_benchmark.md), [`outputs/phase4/latency_and_cost_analysis.md`](file:///outputs/phase4/latency_and_cost_analysis.md), [`scripts/run_phase4_domain_evaluation.py`](file:///scripts/run_phase4_domain_evaluation.py)  
**Auditor**: ML Systems & Serving Infrastructure Auditor  
**Date**: October 13, 2026  

---

## 1. Executive Summary

Phase 4 published detailed serving claims:
- E0 Latency: **266.3 ms**
- E3 Latency: **281.1 ms**
- E1 Latency: **1,603.1 ms**
- E6_bounded Latency: **660.1 ms**
- E6_bounded Generations / Query: **2.47**
- Latency Savings vs E1: **58.8%**
- Cold Start: **7.82 s**
- Peak VRAM: **12.63 GB**
- Peak Throughput: **477.8 tokens/sec**

This forensic audit traced the exact code instrumentation that produced these numbers.

### Headline Auditor Verdict:
1. **Mathematical Consistency**: The published numbers ($660.1$ ms, $2.47$ generations, $58.8\%$ savings) are algebraically exact given the underlying assumptions.
2. **Timing Boundary Clarification**:
   - The reported latency of $660.1$ ms was **NOT measured per-request on a live, single-stream interactive serving endpoint**.
   - Instead, it was derived from an **analytical cost model applied to offline batched forward passes**:
     - 1,200 baseline prompts were evaluated in batch size 8 in $319.6$ seconds, yielding an amortized forward pass time of $e_0 = \frac{319.6}{1200} \times 1000 = \mathbf{266.33\text{ ms}}$.
     - For clusters routed to the fast path (53 clusters, 26.5%), latency was assigned as $L_{\text{fast}} = e_0 = 266.3$ ms.
     - For clusters routed to bounded verification (147 clusters, 73.5%), latency was calculated analytically as $L_{\text{verif}} = 3.0 \times e_0 + 3.0\text{ ms} = 802.0$ ms.
     - Mean latency: $\frac{53 \times 266.33 + 147 \times 802.0}{200} = \mathbf{660.1\text{ ms}}$.
3. **Throughput Misclassification**:
   - The reported $477.8$ tokens/sec is **batched generation throughput** across 8 parallel streams ($8 \times 128\text{ tokens} / 2.143\text{ s}$).
   - Single-user interactive throughput is **$172.6$ tokens/sec** ($32\text{ tokens} / 185.4\text{ ms}$). Presenting batched throughput without distinguishing it from single-user responsiveness can mislead production deployment sizing.

---

## 2. Forensic Tracing of Latency Instrumentation

In `scripts/run_phase4_domain_evaluation.py:180-255`:
```python
# Baseline batched generation
e0_preds, e0_time = run_batched_generation(peft_model, tokenizer, records, adapter_name=None)
e0_latency_ms = (e0_time / len(records)) * 1000.0  # -> 266.3 ms

# Analytical routing assignment
if disagreements <= 1 and not has_conflict:
    route = "BASELINE_FAST"
    lat = e0_latency_ms                              # 266.3 ms
else:
    route = "BOUNDED_VERIFICATION_K3"
    lat = e0_latency_ms * 3.0 + 3.0                  # 801.9 ms
```

### Table 2.1: Analytical Component Breakdown

| Condition | Invocations / Formula | Fast Path ($N=53$) | Verification Path ($N=147$) | Weighted Mean | Reported Value | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **E0 (Base)** | $1 \times \text{pass}$ | 266.3 ms | 266.3 ms | 266.3 ms | 266.3 ms | **Verified** (Amortized Batch) |
| **E3 (Adapter)**| $1 \times \text{pass} (\text{w/ LoRA})$ | 281.1 ms | 281.1 ms | 281.1 ms | 281.1 ms | **Verified** (Amortized Batch) |
| **E1 (Full Medoid)**| $6 \times \text{pass} + 5.0\text{ ms}$ | 1,603.1 ms | 1,603.1 ms | 1,603.1 ms | 1,603.1 ms | **Verified** (Analytical $6\times$) |
| **E6_bounded** | $266.3\text{ ms}$ (26.5%) vs $801.9\text{ ms}$ (73.5%) | 266.3 ms | 801.9 ms | **660.1 ms** | **660.1 ms** | **Verified** (Analytical Model) |
| **Generations/Req** | $1$ (26.5%) vs $3$ (73.5%) | 1.00 | 3.00 | **2.47** | **2.47** | **Verified** |

### Table 2.2: Latency Savings Calculation Verification
$$\text{Savings} = 1 - \frac{\text{Latency}_{\text{E6\_bounded}}}{\text{Latency}_{\text{E1}}} = 1 - \frac{660.1}{1603.1} = 1 - 0.41176 = \mathbf{0.58824 \implies 58.82\%}$$
The published savings of **58.8%** is mathematically verified.

---

## 3. Serving Boundary & Workload Distinction

In realistic microservice deployment (e.g. FastAPI / Triton / vLLM), total latency encompasses boundaries outside the offline batched forward pass:

```
[Client Request] 
      |---> Network Ingest & Request Deserialization (~5-15 ms)
      |---> Tokenization & Chat Templating (~2-5 ms)
      |---> Initial Candidate Generation y_0 (~200-280 ms)
      |---> Pairwise Disagreement Detection (~2-4 ms)
      |         |
      |         +---> [Fast Path Decision] ---> Response Formatting (~1-3 ms) ---> [Client Receive]
      |         |                               (Total E2E: ~210-305 ms)
      |         v
      |     [Verification Decision]
      |     |---> Paraphrase Dispatch & Candidate Generation (y_1, y_2)
      |     |     (Sequential HF: +400-550 ms; Parallel vLLM: +80-120 ms)
      |     |---> Medoid Distance Computation (~2-4 ms)
      |     +---> Response Formatting (~1-3 ms) ---> [Client Receive]
      |           (Total E2E Sequential: ~650-860 ms)
```

### Table 3.1: Interactive Single-Stream vs Amortized Batched Serving

| Regime | Concurrency | Mean Decode Time | Throughput | Overhead Excluded in Report |
|---|:---:|:---:|:---:|---|
| **Single-Stream Interactive** | 1 | 185.4 ms ($K=1$) / 560 ms ($K=3$) | 172.6 tok/s | HTTP/gRPC roundtrip, tokenizer dispatch |
| **Amortized Batched (Reported)** | 8 | 266.3 ms ($K=1$) / 801.9 ms ($K=3$) | 477.8 tok/s | Sequential queue wait times |
| **Continuous Batching (vLLM)** | Dynamic | ~120 ms ($K=1$) / ~340 ms ($K=3$) | > 900 tok/s | Requires PagedAttention serving framework |

---

## 4. GPU Memory & Cold-Start Profile

- **Base Weights Footprint**: 5.62 GB (Qwen2.5-7B in 4-bit NF4).
- **Peak KV Cache Footprint**: 5.85 GB (Batch size 8, sequence length 512).
- **CUDA Runtime / Context**: 1.16 GB.
- **Total Peak VRAM**: **12.63 GB** (51.4% of 24 GB RTX 4090 capacity).
- **VRAM Headroom**: **11.93 GB** (48.6% unallocated).
- **Cold Start**: **7.82 seconds** (loading 4 checkpoint shards from local NVMe SSD into GPU memory).

---

## 5. Auditor Recommendations for Production Serving

1. **Explicitly Label Latency as Analytical Multiplier**: The 660.1 ms latency must be reported as "amortized analytical serving time based on batch-size-8 generation", not "end-to-end client latency."
2. **Transition to Continuous Batching**: In native Hugging Face sequential loops, $K=3$ generates 3 passes sequentially ($3 \times 266$ ms). Deploying with vLLM enables parallel evaluation of candidate paraphrases $x_1$ and $x_2$ on the same GPU engine, cutting bounded verification latency from 801.9 ms to ~340 ms.
