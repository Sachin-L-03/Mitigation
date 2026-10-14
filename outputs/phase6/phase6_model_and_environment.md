# Phase 6: Model, Environment, and Hardware Specifications

**Documentation Reference:** `PHASE6-ENV-SPEC-001`  
**Date:** October 14, 2026  
**Auditor:** Senior Infrastructure & Deployment Engineer  
**Status:** VALIDATED  

---

## 1. Hardware Infrastructure

All experimental evaluations, GPU inference passes, and live interactive serving latency measurements were executed locally on an dedicated high-performance workstation:

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 4090 | Ada Lovelace Architecture |
| **VRAM Capacity** | 24,564 MB (24 GB GDDR6X) | ECC: Off |
| **CUDA Cores** | 16,384 | 128 SMs |
| **Tensor Cores** | 512 (4th Generation) | FP8 / BF16 / INT4 Acceleration |
| **Host Processor** | AMD Ryzen / Intel Core Workstation | 32 Logical Cores |
| **Host System RAM** | 64 GB DDR5 | Dual Channel |
| **Operating System** | Microsoft Windows 11 Enterprise | Version 10.0.22631 |
| **NVIDIA Driver** | 560.94 (or compatible production release) | CUDA Driver: 12.6 |

---

## 2. Software and Deep Learning Runtime Environment

| Package / Runtime | Exact Version | Purpose in Pipeline |
| :--- | :--- | :--- |
| **Python** | 3.11.9 (64-bit) | Workstation runtime environment |
| **PyTorch (`torch`)** | 2.5.1+cu121 | Deep learning tensor engine with CUDA 12.1 |
| **Transformers (`transformers`)** | 4.57.6 | Hugging Face LLM architecture & generation |
| **BitsAndBytes (`bitsandbytes`)** | 0.44.1 | 4-bit NF4 double-quantization engine |
| **Accelerate (`accelerate`)** | 1.12.0 | Multi-device / memory-mapped tensor dispatch |
| **PEFT (`peft`)** | 0.14.0 | LoRA parameter-efficient fine-tuning framework |
| **SymPy (`sympy`)** | 1.13.3 | Path A exact symbolic and arithmetic validation |
| **SQLite3 (`sqlite3`)** | 3.45.1 (FTS5 enabled) | Path B full-text search and BM25 passage ranking |
| **SciPy (`scipy`)** | 1.14.1 | Statistical testing (McNemar, Wilcoxon, binomial) |
| **NumPy (`numpy`)** | 2.1.3 | Numerical array operations and bootstrap resamples |
| **Pytest (`pytest`)** | 8.3.3 | Automated regression and unit test suite |

---

## 3. Base Model and Quantization Configuration

| Configuration Attribute | Setting | Rationale / Architectural Details |
| :--- | :--- | :--- |
| **Model Name** | `Qwen/Qwen2.5-7B-Instruct` | 7.61B parameter dense autoregressive language model |
| **Quantization Format** | 4-bit NF4 (NormalFloat4) | Bounded memory footprint allowing 24 GB execution |
| **Double Quantization** | Enabled (`bnb_4bit_use_double_quant=True`) | Reduces memory overhead by an additional ~0.37 bits/param |
| **Compute Dtype** | `torch.bfloat16` | Preserves dynamic range during dequantized computation |
| **Quant Type** | `nf4` | Quantile-spaced normal distribution quantization |
| **Context Length** | Up to 32,768 tokens (bounded to 2,048 in eval) | High-fidelity multi-turn instruction processing |
| **Peak VRAM (Inference)**| **8.22 GB** | Fits comfortably within 24 GB VRAM envelope |
| **Base Model Revision** | Hugging Face Main Branch | Checkpoint hash verified against manifest |

---

## 4. Deterministic Decoding and Reproducibility Protocol

To ensure reproducible experimental conditions across all evaluations:
- **Random Seed:** Set explicitly to `42` across Python `random`, NumPy `np.random.seed(42)`, and PyTorch `torch.manual_seed(42)`.
- **Decoding Strategy:** Strictly greedy (`do_sample=False`, `temperature=0.0`, `top_p=1.0`).
- **Generation Budget:** Maximum new tokens fixed at `64` for concise factual answers.
- **Padding Configuration:** Left-side padding (`tokenizer.padding_side = "left"`) with `pad_token_id = eos_token_id`.
- **Clock Synchronization:** All runtime measurements employ `torch.cuda.synchronize()` before and after GPU execution to prevent asynchronous CUDA dispatch skew.
