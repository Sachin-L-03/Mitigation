# Causal Patching and Targeted Parameter Editing Report

**Project**: `prompt-sensitivity-hybrid`  
**Component**: Causal Activation Patching and CREME-Style Parameter Editing  
**Date**: October 12, 2026  

---

## 1. Causal Activation Patching Protocol

To verify that the representation divergence identified in layers $[16, 20, 24]$ causally mediates output answer variation, we implemented residual-stream activation patching using [`causal_analysis.py`](file:///c:/Users/Student5/Desktop/Mitigation/causal_analysis.py):

1. **Unpatched Baseline**:
   - Run original prompt $x_{\text{orig}} \to y_{\text{orig}}$, record activations $h_{\text{orig}}^{(L)}$ and output logits $\mathbf{z}_{\text{orig}}$.
   - Run variant prompt $x_{\text{var}} \to y_{\text{var}}$, record output logits $\mathbf{z}_{\text{var}}$.
   - Compute baseline divergence: $D_{\text{KL}}(\mathbf{z}_{\text{orig}} \| \mathbf{z}_{\text{var}})$.
2. **Causal Patching**:
   - Run forward pass on $x_{\text{var}}$, replacing activation at layer $L$ with $h_{\text{orig}}^{(L)}$.
   - Record patched logits $\mathbf{z}_{\text{patch}}$ and compute $D_{\text{KL}}(\mathbf{z}_{\text{orig}} \| \mathbf{z}_{\text{patch}})$.
   - Causal restoration: $\Delta_{\text{KL}} = D_{\text{KL}}(\text{baseline}) - D_{\text{KL}}(\text{patched})$.
3. **Sham Control**:
   - Replace activation with Gaussian noise vector matched to the empirical $L_2$ norm of $h_{\text{orig}}^{(L)}$.
   - Causal Specificity: $\text{Specificity} = \Delta_{\text{KL}}(\text{patch}) - \Delta_{\text{KL}}(\text{sham})$.

---

## 2. Controlled Patching Results

| Layer | Baseline $D_{\text{KL}}$ | Patched $D_{\text{KL}}$ | Sham $D_{\text{KL}}$ | Causal Restoration | Causal Specificity |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Layer 12** | 1.842 | 1.718 | 1.835 | +0.124 | +0.117 |
| **Layer 16** | 1.954 | **1.767** | 1.948 | **+0.187** | **+0.181** |
| **Layer 20** | 2.105 | 1.943 | 2.099 | +0.162 | +0.156 |
| **Layer 24** | 2.312 | 2.174 | 2.308 | +0.138 | +0.134 |

---

## 3. Targeted Parameter Editing Outcomes

Using the causally identified layers $[16, 20, 24]$, we evaluated two parameter editing configurations:
1. **$w_{\text{pres}}=1.0$ (Baseline Editing)**: LoRA adapters attached to layers $[16, 20, 24]$ with standard preservation weight.
2. **$w_{\text{pres}}=5.0$ (Preservation-Tuned Editing)**: High preservation regularization.

### Performance on In-Distribution vs. Out-of-Distribution Clusters

| Split | Base Model F1 | Edited ($w=1.0$) F1 | Edited ($w=5.0$) F1 | Generalization Gap |
|---|:---:|:---:|:---:|:---:|
| **In-Distribution Test (30 clusters)** | $37.03\%$ | $45.59\%$ | **$45.82\%$** | $+8.79\text{ pp (apparent gain)}$ |
| **Out-of-Distribution (100 clusters)** | **$37.23\%$** | $37.44\%$ | $37.20\%$ | **$-0.03\text{ pp (vanishing gain)}$** |

### Critical Scientific Conclusion
Targeted LoRA parameter editing on 137 specific training clusters produces **distribution-specific memorization** rather than universal prompt invariance. While it alters weights to align the 6 specific paraphrase styles present in the training set, on novel domains (biomedical, geophysical, physics), the weights exert negligible influence on representations, causing performance to revert to the base model.
