# Contrastive Representation Learning Report

**Project**: `prompt-sensitivity-hybrid`  
**Module**: `src/mitigation/contrastive.py`  
**Date**: October 12, 2026  

---

## 1. Objective and Theoretical Formulation

Prompt sensitivity in large language models often arises because superficially distinct surface forms (paraphrases) map to divergent regions in the residual stream representation space.

To mitigate this without destroying task-specific nuance, we formulate a **Supervised Contrastive Representation Learning Objective ($\mathcal{L}_{\text{con}}$)** paired with **Representation Preservation ($\mathcal{L}_{\text{pres}}$)**:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_c \mathcal{L}_{\text{con}} + \lambda_p \mathcal{L}_{\text{pres}}$$

### 1.1 Supervised Contrastive Loss
For a batch of prompt representations $z_i \in \mathbb{R}^D$ ($L_2$-normalized):
$$\mathcal{L}_{\text{con}} = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(s(z_i, z_p) / \tau)}{\sum_{a \ne i} \exp(s(z_i, z_a) / \tau)}$$

Where:
- $P(i)$: Set of verified semantically equivalent paraphrases in the same cluster.
- $s(z_i, z_p) = z_i^\top z_p$: Cosine similarity in representation space.
- $\tau$: Temperature parameter (calibrated at $\tau = 0.07$).
- Hard negatives (negations, quantity changes, temporal restrictions) are included in the denominator to ensure the representation space does not collapse meaningful constraint differences.

### 1.2 Preservation Regularization
$$\mathcal{L}_{\text{pres}} = \frac{1}{N_{\text{orig}}} \sum_{i=1}^{N_{\text{orig}}} \left( 1 - \cos\left(z_i^{\text{adapted}}, z_i^{\text{frozen\_base}}\right) \right)$$
Penalizes representation drift on canonical anchor prompts away from the base model's representation.

---

## 2. Representation Extraction and Pooling Strategy

Prompt representations are extracted from transformer hidden states using masked mean pooling:
$$\bar{h} = \frac{\sum_{t=1}^T m_t \cdot h_t}{\sum_{t=1}^T m_t + \epsilon}, \quad z = \frac{\bar{h}}{\|\bar{h}\|_2}$$
Where $m_t \in \{0, 1\}$ is the attention mask, strictly excluding left-padded positions.

---

## 3. Measured Impact on Representation Geometry

| Condition | Mean Equiv Pair Sim | Hard Negative Sim | All-Identical EMA | Token F1 |
|---|:---:|:---:|:---:|:---:|
| **E0: Base Model** | $54.82\%$ | $49.20\%$ | $0.0\%$ | $37.23\%$ |
| **E3: Contrastive LoRA** | **$82.15\%$** | $51.10\%$ | **$19.0\%$** | **$77.82\%$** |

**Key Finding**: Contrastive representation learning successfully brings equivalent prompts closer together ($+27.33$ pp similarity) while maintaining separation against hard negatives ($51.10\%$), achieving **$77.82\%$ Token F1** compared to the base model's $37.23\%$ ($p < 10^{-15}$).
