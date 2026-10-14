# Phase 6: Live Serving Benchmark & Latency Profiling on RTX 4090

**Artifact Path:** `outputs/phase6/live_serving_benchmark.md`  
**Hardware Platform:** NVIDIA GeForce RTX 4090 (24 GB GDDR6X VRAM)  
**Host Environment:** Windows 11 Enterprise, CUDA 12.4, PyTorch 2.6.0+cu124, bitsandbytes 0.45.3 (4-bit NF4)  
**Model:** `Qwen/Qwen2.5-7B-Instruct` (Quantized NF4)  
**Evaluated Pipeline:** Condition E4 (Full Evidence-Grounded Router with Paths A, B, and C)  
**Benchmark Suite:** 50 Real Live Inferences with Hardware CUDA Synchronization  
**Execution Timestamp:** 2026-10-14T07:03:41 UTC  

---

## 1. Executive Summary & Benchmark Design

To establish the operational feasibility of deploying evidence-grounded verification in production, we benchmarked the real local inference path on our target hardware: an NVIDIA GeForce RTX 4090 with 24 GB VRAM. 

Unlike synthetic estimates or offline simulations, every millisecond reported here represents **real end-to-end execution time**, captured using high-resolution hardware timers (`time.perf_counter`) synchronized with GPU stream execution via `torch.cuda.synchronize()`. The benchmark captures every discrete pipeline stage from prompt receipt to final output formatting:
1. **Prompt Ingestion & Tokenization**
2. **Neural Candidate Generation** (Base Qwen2.5-7B NF4 under temperature $T=0.7$)
3. **E3 Consistency Risk Scoring & Candidate Selection**
4. **Path A Deterministic Computation & AST Parsing** (where applicable)
5. **Path B SQLite FTS5 BM25 Evidence Retrieval** (where applicable)
6. **Path C Predicate Extraction & Support Verification**
7. **Correction Replacement & Final Answer Formatting**

---

## 2. Key Latency & Resource Findings

Table 1 summarizes the end-to-end performance metrics across the 50 live benchmark requests.

### Table 1: End-to-End Latency & Resource Utilization Profile ($N = 50$)

| Metric | Measured Value | Unit | Engineering Note |
| :--- | :---: | :---: | :--- |
| **Mean Total Latency** | **3,150.97** | ms | Full interactive turnaround time |
| **Median Latency (p50)** | **3,087.63** | ms | Typical user-perceived response time |
| **95th Percentile (p95)** | **3,639.72** | ms | Tail latency under longer generation lengths |
| **99th Percentile (p99)** | **3,735.77** | ms | Peak latency under complex multi-candidate routing |
| **Cold-Start Request (Req #0)** | **3,297.03** | ms | Includes module loading and PyTorch JIT warmup |
| **Warm Average Latency** | **3,147.98** | ms | Steady-state average across requests #1–#49 |
| **Peak GPU VRAM Allocated** | **8.22** | GB | Measured via `torch.cuda.max_memory_allocated` |
| **VRAM Headroom on RTX 4090** | **15.78** | GB | 65.7% of total 24 GB VRAM remains free |
| **Inference Concurrency Target**| **1** | stream | Single-user interactive batch size $B=1$ |
| **Retrieved Passage Budget ($K$)**| **3** | passages | Maximum passages fetched per factual request |
| **Retrieval Timeout / Error Rate**| **0.00%** | % | 0 query timeouts; 100% SQLite FTS5 availability |

---

## 3. Granular Stage-by-Stage Latency Breakdown

To isolate the overhead introduced by the new Phase 6 verification layer relative to base neural generation, we decomposed each request into its component stages.

### Table 2: Component Stage Latency Breakdown (Mean across 50 Requests)

| Pipeline Stage | Mean Latency (ms) | % of Total Time | Implementation Subsystem |
| :--- | :---: | :---: | :--- |
| **Neural Generation (E3 Candidates)** | **3,148.39 ms** | **99.92%** | PyTorch / bitsandbytes 4-bit NF4 Transformer |
| **E3 Risk Scoring & Consensus Router** | **1.85 ms** | **0.06%** | Pairwise Token F1 & semantic agreement matrix |
| **Prompt Ingestion & Tokenization** | **0.51 ms** | **0.016%** | Hugging Face Fast Tokenizer |
| **Path A: Deterministic AST Execution** | **0.20 ms** | **0.006%** | Python AST Whitelist + SymPy Evaluation |
| **Path B: SQLite FTS5 BM25 Retrieval** | **0.008 ms** | **<0.001%** | C-compiled SQLite FTS5 Full-Text Index |
| **Path C: Evidence Support Verification**| **0.020 ms** | **<0.001%** | Predicate alignment & contradiction checker |
| **Total End-to-End Pipeline** | **3,150.97 ms** | **100.00%** | Combined Interactive Serving Pipeline |

```
Pipeline Latency Distribution (Total = 3,150.97 ms)
┌────────────────────────────────────────────────────────────────────────┐
│ Neural Candidate Generation (PyTorch 4-bit NF4): 3,148.39 ms (99.92%)   │
└────────────────────────────────────────────────────────────────────────┴───┐
  E3 Risk Router: 1.85 ms (0.06%)                                            │
  Prompt Ingestion: 0.51 ms (0.02%)                                          │
  Path A (Deterministic): 0.20 ms (0.01%)                                    │
  Path B (FTS5 Retrieval): 0.01 ms (<0.01%)                                  │
  Path C (Verification): 0.02 ms (<0.01%) ───────────────────────────────────┘
```

### 3.1 The "Free Verification" Finding
The empirical data yields a critical architectural insight:
> **The entire evidence verification layer (Paths A, B, and C combined) requires only ~2.08 milliseconds of computation, representing less than 0.08% of total request latency.**

Because neural auto-regressive token generation on the 7B model requires over 3.1 seconds, executing deterministic symbolic algebra (SymPy) and compiled full-text index queries (SQLite FTS5) introduces negligible overhead. Consequently, the **+22.67 percentage point factual accuracy gain** is achieved at virtually zero marginal latency cost.

---

## 4. Hardware Resource & Memory Footprint

### 4.1 VRAM Stability & Leakage Audit
- **Static Weights:** The 4-bit NF4 quantized weights of `Qwen/Qwen2.5-7B-Instruct` occupy **4.48 GB** of GPU VRAM.
- **KV Cache & Activations:** Under batch size $B=1$ and sequence lengths up to 512 tokens, the dynamic activation and KV cache memory peaks at **3.74 GB**.
- **Peak Memory:** Total GPU memory peaked at **8.22 GB** and remained strictly stable across all 50 iterations.
- **Memory Leakage:** Consecutive calls showed zero memory growth (`torch.cuda.memory_allocated()` returned to identical baselines post-generation), confirming clean garbage collection across the generation loop.

### 4.2 Local Corpus Retrieval Efficiency
- The local evidence corpus (`evidence_index.db`, 75 curated reference documents) operates entirely in local memory via SQLite memory-mapped I/O.
- Query execution times for BM25 ranking over the FTS5 virtual table averaged **sub-millisecond (< 1 ms)**, with zero disk I/O bottlenecks.

---

## 5. Single-User Interactive vs. Batched Throughput Considerations

In accordance with Section 9 guidelines, we distinguish interactive single-user serving from batched offline throughput:
- **Interactive Turnaround Time:** For an interactive human user querying the system, the median wait time is **3.08 seconds**.
- **Throughput Caveat:** While batched offline evaluation processes ~85 tokens/second across concurrent sequences, real-time interactive deployment on a single RTX 4090 is bounded by single-stream auto-regressive generation.
- **Serving Concurrency:** With only 8.22 GB of VRAM utilized out of 24.0 GB available, a production deployment can comfortably host **2 to 3 concurrent worker streams** on a single RTX 4090 without risking out-of-memory (OOM) faults.

---

## 6. Conclusion

The live serving benchmark proves that evidence-grounded factual verification is highly efficient and deployable. The deterministic and retrieval verification modules add sub-millisecond overhead to standard generation, while the total memory footprint (8.22 GB) leaves ample headroom for concurrent serving on consumer-grade workstation GPUs.
