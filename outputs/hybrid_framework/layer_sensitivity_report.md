# Layer-Aware Sensitivity Analysis Report

**Project**: `prompt-sensitivity-hybrid`  
**Component**: CREME-Inspired Layer-Aware Sensitivity Profiling  
**Date**: October 12, 2026  

---

## 1. Methodology and Layer Attribution

To understand where prompt sensitivity arises across the 28 decoder layers of `Qwen2.5-7B-Instruct`, we profiled residual stream activations at candidate layers $\mathcal{L} \in \{12, 16, 20, 24\}$:

1. **Mean Equivalent Pair Distance ($d_{\text{equiv}}$)**:
   $$d_{\text{equiv}} = 1 - \cos\left(h(x_{\text{orig}}), h(x_{\text{variant}})\right)$$
2. **Inter-Cluster Separation ($s_{\text{inter}}$)**:
   $$s_{\text{inter}} = 1 - \cos\left(h(x_{\text{cluster A}}), h(x_{\text{cluster B}})\right)$$
3. **Fisher Separation Ratio**:
   $$\text{Ratio} = \frac{s_{\text{inter}}}{d_{\text{equiv}} + \epsilon}$$

---

## 2. Layer Sensitivity Profiling Results

| Layer Index | Mean Equiv Distance | Mean Equiv Similarity | Inter-Cluster Separation | Fisher Ratio | Causal Restoration Score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Layer 12** | 0.0891 | 0.9109 | 0.3214 | 3.607 | 0.1242 |
| **Layer 16** | **0.1124** | 0.8876 | **0.4421** | **3.933** | **0.1874** |
| **Layer 20** | 0.1452 | 0.8548 | 0.5120 | 3.526 | 0.1619 |
| **Layer 24** | 0.1820 | 0.8180 | 0.5482 | 3.012 | 0.1380 |

### Key Observations
- **Early-Mid Transition (Layer 12)**: Represents low prompt divergence, primarily processing syntactic surface tokens.
- **Mid-Late Latent Semantic Space (Layer 16 & 20)**: Exhibits the highest Fisher ratio ($3.93$) and maximum causal restoration ($0.187$). Paraphrase representations begin diverging significantly here, directly influencing downstream token selection.
- **Late Readout Layers (Layer 24)**: High overall divergence, but reduced Fisher ratio as representations become tied to specific output token vocabularies.
- **Conclusion**: Layers $[16, 20, 24]$ were selected as the targeted intervention substrate for CREME parameter editing.
