# Phase 5 Model and Environment Configuration Report

**Document Status:** COMPLETE & VERIFIED  
**Repository:** `prompt-sensitivity-hybrid`  
**Branch:** `creme-hybrid-framework`  
**Audit Date:** 2026-10-13  
**Auditor:** ML Systems & Inference Engineer  

---

## 1. Hardware Architecture

Inference and serving benchmarks are executed locally on the dedicated workstation environment:

| Property | Value |
| :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 4090 |
| **Compute Capability** | 8.9 (Ada Lovelace) |
| **Total Dedicated VRAM** | 24.0 GB (23.988 GB reported by PyTorch) |
| **CUDA Driver / Runtime** | CUDA 12.1 |
| **Host Operating System** | Windows 11 Enterprise (x64) |
| **Shell Environment** | PowerShell 7 / Windows Terminal |

---

## 2. Software Dependencies and Package Versions

The execution environment runs within a isolated Python virtual environment (`.venv`):

| Package / Component | Version | Role in Phase 5 Pipeline |
| :--- | :--- | :--- |
| **Python** | 3.11.9 | Core runtime |
| **PyTorch** | 2.5.1+cu121 | Tensor execution and CUDA kernel dispatch |
| **Transformers** | 4.57.6 | Model loading, generation pipeline, tokenizer |
| **PEFT** | 0.21.2 | Parameter-efficient fine-tuning / LoRA support |
| **bitsandbytes** | 0.50.2 | 4-bit NF4 quantized matrix multiplication |
| **Accelerate** | 1.12.0 | Device placement and memory management |
| **NumPy** | 1.26.4 | Array computations and similarity matrix operations |
| **SciPy** | 1.14.1 | Statistical testing (Wilcoxon, McNemar, bootstrap) |
| **Pytest** | 9.1.1 | Regression test suite |

---

## 3. Base Model Specification and Quantization Parameters

The base model is unmodified across all primary evaluations:

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Model Identifier** | `Qwen/Qwen2.5-7B-Instruct` | Official release checkpoint |
| **Architecture** | Qwen2ForCausalLM (7B parameters) | Decoder-only Transformer |
| **Quantization Format** | 4-bit NormalFloat4 (NF4) | `bitsandbytes` NF4 quantization |
| **Double Quantization** | Enabled (`bnb_4bit_use_double_quant=True`) | Quantizes quantization constants |
| **Compute Data Type** | `bfloat16` (`torch.bfloat16`) | Native tensor compute dtype on RTX 4090 |
| **Tokenizer Padding Side** | Left (`tokenizer.padding_side = "left"`) | Required for batched causal LM generation |
| **Pad Token ID** | Set to `tokenizer.eos_token_id` | Prevents pad token assertion errors |

---

## 4. Inference and Generation Parameters

To ensure reproducibility across all experimental runs, generation parameters are frozen:

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Greedy Decoding** | `do_sample=False` | Deterministic outputs for paired cluster comparisons |
| **Temperature** | `0.0` | Eliminates random sampling noise |
| **Maximum New Tokens** | `64` | Sufficient for concise answers, definitions, and slots |
| **Random Seed** | `42` | Global seed set via `torch.manual_seed(42)` and `np.random.seed(42)` |
| **Batch Size (Offline Eval)** | `8` | Fits comfortably within RTX 4090 24GB VRAM |
| **KV-Cache** | Enabled (`use_cache=True`) | Accelerates multi-token generation passes |

---

## 5. Baselines and Historical Checkpoint Verification

To maintain reproducibility and continuity with Phases 1–4:
- **Baseline Model Checkpoint:** Pre-cached local weights for `Qwen/Qwen2.5-7B-Instruct`.
- **Preservation LoRA Adapter Checkpoint:** `outputs/optimization/preservation/adapter_w5.0/` verified and readable.
- **Phase 4 Artifacts:** Fully preserved under `outputs/phase4/` and `outputs/phase4_forensic/`. No historical files modified or removed.

---

## 6. Verification Summary

The model and environment configuration meets all hardware, memory, and software prerequisites for executing Phase 5 experiments with zero deviation from preregistered standards.
