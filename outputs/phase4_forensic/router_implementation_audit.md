# Phase 4 Router and Model Inference Forensic Audit

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Audit Target**: `E6_bounded` (Cost-Aware Bounded Adaptive Router) Implementation  
**Audited Code**: [`src/mitigation/router.py`](file:///src/mitigation/router.py), [`scripts/run_phase4_domain_evaluation.py`](file:///scripts/run_phase4_domain_evaluation.py), [`model.py`](file:///model.py)  
**Auditor**: ML Systems & Research Auditor  
**Date**: October 13, 2026  

---

## 1. Executive Summary

This forensic audit traces the execution of `E6_bounded` from incoming prompt ingest to final token generation. The purpose is to determine whether the inference pipeline conforms to the stated specifications, whether LoRA weights or gold references leaked into runtime contexts, and whether operational constraints (such as the $K=3$ candidate budget and non-equivalent constraint preservation) are strictly enforced in code.

### Auditor Verdicts:
- **Model Checkpoint**: Verified unmodified `Qwen/Qwen2.5-7B-Instruct`.
- **Quantization**: Verified 4-bit NormalFloat4 (NF4) via `bitsandbytes`.
- **Adapter Independence**: Verified that the preservation adapter was disabled during E0 and E6_bounded runs (`with peft_model.disable_adapter()`).
- **Gold-Reference Isolation**: Verified zero gold-reference leakage into prompts or runtime routing contexts.
- **Candidate Budget**: Verified strictly bounded to at most $K=3$ candidate generations.
- **Threshold Origin**: Verified $\tau = 0.50$ was selected on validation data in Phase 3, not retrofitted to the Phase 4 test benchmark.

---

## 2. Model Loading, Precision, and Runtime Environment

### 2.1 Model Checkpoint & Quantization
In `model.py:44-78` and `scripts/run_phase4_domain_evaluation.py:163-176`:
- **Model Name**: `Qwen/Qwen2.5-7B-Instruct` loaded from Hugging Face cache.
- **Quantization Config**:
  ```python
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
  )
  ```
- **Device Placement**: Explicitly mapped to `device_map="auto"`, resolving to `cuda:0` (NVIDIA GeForce RTX 4090, 24,564 MiB VRAM).
- **Tokenizer Setup**: Left-padded (`tokenizer.padding_side = "left"`) with `pad_token_id = tokenizer.eos_token_id = 151643`.

### 2.2 Adapter Isolation Verification
In `scripts/run_phase4_domain_evaluation.py:171-180`:
```python
peft_model = PeftModel.from_pretrained(
    base_model,
    adapter_w5_path,
    adapter_name="adapter_w5",
)
```
Although `peft_model` loaded the `adapter_w5` checkpoint for Condition E3, for Condition E0 and Condition E6_bounded the script invoked:
```python
if adapter_name is None:
    with peft_model.disable_adapter():
        output_ids = peft_model.generate(...)
```
**Verification**: Verified via PyTorch hook inspection that `disable_adapter()` bypasses all LoRA weight additions $\Delta W = B \times A$. Outputs for E0 and E6_bounded were generated solely by the base pretrained parameters.

### 2.3 Generation Parameters
- `do_sample = False` (Greedy deterministic decoding).
- `temperature = 1.0` (Ignored under greedy decoding).
- `max_new_tokens = 64` (Sufficient for factual completions; caps latency).
- Random seed: `torch.manual_seed(42)` and `np.random.seed(42)`.

---

## 3. Disagreement Detection Pipeline (`SemanticAnswerComparator`)

The detector implements a multi-stage comparison pipeline in `src/mitigation/router.py:144-197`:

### 3.1 Lexical & Substring Similarity
1. **Token F1**: Token-level harmonic mean of precision and recall after punctuation and article removal.
2. **Char 3-gram Similarity**: Cosine similarity between character 3-gram frequency vectors.
3. **Composite Similarity**:
   $$\text{SemanticSim} = 0.40 \times \text{TokenF1} + 0.60 \times \text{Char3gramSim}$$
4. **Telegraphic Containment Override**: If normalized string $A \subseteq B$ or $B \subseteq A$, semantic similarity is bounded below by 0.70 to prevent short factual answers (e.g. "Rome") from failing comparison against verbose complete sentences (e.g. "The capital of Italy is Rome").

### 3.2 Numerical Consistency Engine
- Extracts all numerical patterns `\b\d+(?:\.\d+)?%?\b`.
- If both answers contain non-empty sets of extracted numbers $N_A$ and $N_B$, a numerical contradiction is flagged if and only if $N_A \cap N_B = \emptyset$.
- E.g., `{12}` vs `{10}` $\implies$ `num_contradiction = True`.
- E.g., `{200}` vs `{200}` $\implies$ `num_contradiction = False`.

### 3.3 Negation Reversal Engine
- Checks presence of negation tokens (`not`, `no`, `never`, `cannot`, `cant`, `wont`, `neither`, `nor`, `lacks`, `absent`, `without`).
- If one answer contains a negation token and the other does not, it computes substantive common words:
  $$\text{CommonSubstantive} = (\text{Tokens}_A \cap \text{Tokens}_B) \setminus \text{Stopwords} \setminus \text{NegationWords}$$
- If $|\text{CommonSubstantive}| \ge 2$, it flags `neg_contradiction = True`.
- E.g., "Light can propagate in vacuum" vs "Light cannot propagate in vacuum" shares `{"light", "propagate", "vacuum"}`, triggering a negation contradiction.

### 3.4 Entity Consistency & Alias Map
- Extracts capitalized entities and normalizes against a domain alias dictionary (`ATP` $\leftrightarrow$ `adenosine triphosphate`, `BGP` $\leftrightarrow$ `Border Gateway Protocol`, etc.).
- Flags mismatch if extracted entities have zero overlap or if non-alias predicate substitutions occur (e.g., "Pasteur" vs "Fleming").

---

## 4. Bounded Medoid Selection ($K=3$)

In `src/mitigation/router.py:199-226`:
- Candidate pool: $C = \{y_0, y_1, y_2\}$ (exactly 3 candidate completions).
- Computes pairwise similarity matrix:
  $$S_{i, j} = \text{compare\_answers}(y_i, y_j)[\text{"semantic\_similarity"}]$$
- Selects medoid index:
  $$i^* = \arg\max_{i} \frac{1}{K} \sum_{j=1}^K S_{i, j}$$
- **Budget Invariant**: The candidate list is truncated via `candidates[:max_k]` where `max_k = 3`. No generation loops, retries, or verification cascades ever exceed 3 generations per query.

---

## 5. Constraint Differentiation & Hard Negatives

A critical requirement of an adaptive router is distinguishing:
1. **Equivalent Paraphrases**: Should be routed to consensus when answers diverge.
2. **Intentional Constraint Changes (Hard Negatives)**: E.g., "List 3 European capitals" vs "List 5 European capitals", or "Explain why X happens" vs "Explain why X does NOT happen".

### Audit of Hard Negative Handling:
- In test case `STRESS_11` ([`outputs/phase4/paraphrase_stress_test.csv`](file:///outputs/phase4/paraphrase_stress_test.csv)):
  - Input: Prompt requesting 3 items vs variant requesting 5 items.
  - Expected Router Behavior: `REJECT_IDENTICAL` (refuse to force identical answers).
  - Observed Behavior: The detector flagged lexical length difference, but because candidate outputs had Token F1 = 0.74, the simple numerical detector missed the list-count constraint divergence (`detected_conflict = False`).
- **Auditor Finding**: The current regex-based numerical and negation detector operates primarily on completion text, not on input prompt constraints. When prompts contain distinct constraint counts (3 vs 5), `E6_bounded` lacks a dedicated prompt constraint parser and risks forcing consensus if completions overlap lexically. This represents a documented limitation for production deployment.

---

## 6. Threshold $\tau = 0.50$ Provenance

- Investigated whether the threshold $\tau = 0.50$ was tuned against the Phase 4 test benchmark.
- **Trace**:
  - In `outputs/phase3/routing_threshold_optimization.md`, a grid search was conducted over $\tau \in [0.30, 0.70]$ on the Phase 3 validation split ($N=60$ clusters).
  - $\tau = 0.50$ was selected as the optimal Pareto operating point balancing consistency gain ($+33$ pp) against verification overhead ($< 650$ ms).
  - The Phase 4 script imported this preset threshold without modification.
- **Verdict**: Provenance is clean. The threshold was not retroactively tuned on Phase 4 test data.
