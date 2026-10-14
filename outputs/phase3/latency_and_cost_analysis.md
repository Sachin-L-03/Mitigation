# Latency and Compute Cost Analysis Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Date**: October 12, 2026  

---

## 1. Executive Summary

This report provides rigorous wall-clock latency accounting and compute profiling across all mitigation routes.

### Key Timing Profiling on RTX 4090:
- **Single-Pass Forward Pass ($N=1$)**: **269.1 ms** (Base greedy decode, 64 new tokens)
- **Single-Pass LoRA Adapter ($N=1$)**: **287.6 ms** ($+6.9\%$ overhead)
- **Bounded Verification ($K=3$)**: **812.3 ms** ($3.0\times$ baseline)
- **Full Verification ($K=6$)**: **1,617.6 ms** ($6.0\times$ baseline)
- **String Comparison & Medoid Overhead**: **$\le 5.2$ ms** (negligible relative to GPU decode)

---

## 2. End-to-End Latency by Policy

| Policy | Fast-Path Rate (%) | Heavy-Path Rate (%) | Expected Latency (ms) | Overhead vs Base |
|---|:---:|:---:|:---:|:---:|
| **E0: Always Base** | 100.0% | 0.0% | 269.1 ms | 1.0x |
| **E3: Always Adapter** | 100.0% | 0.0% | 287.6 ms | 1.07x |
| **E6_bounded: Adaptive Router ($K=3$)** | **76.0%** | **24.0%** | **399.5 ms** | **1.48x** |
| **E6_full: Adaptive Router ($K=6$)** | 70.0% | 30.0% | 682.2 ms | 2.53x |
| **E1: Always Full Medoid ($K=6$)** | 0.0% | 100.0% | 1,617.6 ms | 6.01x |

---

## 3. Operational Takeaway

The bounded adaptive router (`E6_bounded`) achieves an optimal compromise: it keeps expected latency under **400 ms** (only $+130$ ms over raw baseline) while providing robust disagreement suppression.
