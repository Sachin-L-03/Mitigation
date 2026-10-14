# Phase 1–6 Research Findings & Presentation Claims Guide

**Artifact Path:** `outputs/final_visualizations/RESEARCH_FINDINGS.md`  
**Repository:** `prompt-sensitivity-hybrid`  
**Branch:** `creme-hybrid-framework`  
**Execution Timestamp:** 2026-10-14T07:25:00 UTC  

---

## 1. Core Research Questions

This research investigates the dual challenge of **prompt sensitivity** and **factual reliability** in instruction-tuned large language models (`Qwen/Qwen2.5-7B-Instruct`):
1. **The Consistency Question:** Why do minor, semantically invariant paraphrases of a user prompt cause severe output instability and slot-level contradictions?
2. **The Factual Correctness Question:** When an inference-time consistency method enforces agreement across paraphrases (e.g., via majority voting or bounded medoids), does it produce true answers, or does it merely reinforce shared parametric hallucinations?
3. **The Evidence-Grounding Question:** Can deterministic symbolic computation (Path A) and independent evidence retrieval (Path B) correct factual errors and unanimous false consensus while preserving high paraphrase consistency at negligible serving overhead?

---

## 2. Methodological Evolution & Lessons Learned Across Phases 1–6

The trajectory of this project provides several critical scientific lessons regarding the evaluation and mitigation of prompt sensitivity:

### Lesson 1: Parameter Editing Does Not Generalize (Phases 1 & 2)
- **Early Approach:** Attempted to suppress sensitivity subspaces via representation steering and LoRA weight updates on training prompt variations.
- **Forensic Discovery:** While LoRA editing stabilized outputs on in-distribution phrasing, it failed completely on unseen prompt domains (achieving $57.37\%$ Token F1 vs. $57.43\%$ for unadapted base Qwen, $p = 0.52$). Parameter editing does not provide generalized invariance.

### Lesson 2: Internal Consensus Can Be a False-Truth Trap (Phases 3, 4 & 5)
- **Early Approach:** Shifted from weight editing to inference-time routing. Selected the central medoid answer among candidate generations ($E6_{\text{bounded}}$, $E3$ router).
- **Forensic Discovery:** Bounded medoid selection dramatically boosted lexical consistency (Token F1 rose to $90.64\%$) and suppressed output contradictions to $0.0\%$. However, forensic auditing revealed that zero contradictions was a structural artifact (assigning identical medoids to all variants guaranteed zero contradictions by construction). In $17.50\%$ of clusters, the model unanimously agreed on a false claim. Internal consistency cannot detect errors when all sampled candidates share the same bias.

### Lesson 3: The Statistical Two-Sided Fallacy in Safety Benchmarking (Phase 4 Forensic)
- **Forensic Discovery:** Phase 4 claimed that factual non-inferiority was preserved within a 2-percentage-point margin based on a non-significant two-sided t-test ($p = 0.442$). A formal one-sided non-inferiority test ($H_0: \delta \le -0.02$) revealed $p_{\text{NI}} = 0.198$ (lower bound $-3.58\text{ pp}$), proving that non-inferiority was not established. Prespecified one-sided tests are mandatory.

### Lesson 4: Independent External Verification Breaks False Consensus (Phase 6)
- **Breakthrough Approach:** Introduced modular tri-path verification:
  - **Path A (Deterministic):** Safe AST whitelist, SymPy modular math ($b^e \pmod m$), Pint unit conversions, Python `datetime` offsets.
  - **Path B (Retrieval):** Fast SQLite FTS5 BM25 search over verified evidence documents.
  - **Path C (Support Verifier):** Predicate alignment checking that strictly overwrites candidates only when contradicted by verified evidence.
- **Empirical Breakthrough:** Factual accuracy surged from **58.89% to 81.56%** (+22.67 pp, $p = 9.24 \times 10^{-27}$). Non-inferiority was established beyond doubt ($+17.24\text{ pp}$ lower bound vs. $-2.00\text{ pp}$ margin, $p < 10^{-15}$). All 19 unanimous false-consensus cases were corrected.

---

## 3. The Strongest Empirical Results Supported by Evidence

When presenting this research to academic, industrial, or scientific audiences, the following findings are definitively supported by raw prediction records and confirmatory statistical tests:

1. **Massive Factual Accuracy Superiority:**  
   Condition E4 (Combined Router) achieved **81.56%** cluster-level accuracy versus **58.89%** for E1 baseline ($\Delta = +22.67\text{ pp}$, McNemar exact $p = 9.24 \times 10^{-27}$, 106 wins vs. 4 losses on $N = 450$ test clusters).
2. **Conclusive Factual Non-Inferiority vs. Base Qwen:**  
   The lower bound of the one-sided 97.5% confidence interval for E4 vs. E0 is **$+17.24\text{ percentage points}$**, vastly outperforming the prespecified non-inferiority margin of $-2.00\text{ percentage points}$ ($Z_{\text{NI}} = 9.89, p_{\text{NI}} < 10^{-15}$).
3. **100% Elimination of Unanimous False Consensus in Tested Clusters:**  
   In 19 test clusters (4.22%), all candidate generations unanimously agreed on an incorrect claim. E1 corrected 0/19 (0.0%), while E4 corrected **19/19 (100.0%)** via deterministic evaluation.
4. **Sub-Millisecond Verification Overhead:**  
   On physical NVIDIA RTX 4090 hardware, neural auto-regressive generation requires **3,148.39 ms** (99.92% of total turnaround time), while the entire verification layer (Paths A, B, and C combined) requires only **2.08 ms (< 0.08% overhead)**. Verification is computationally negligible.
5. **Exceptionally Low Verification Corruption:**  
   Verification altered an initially correct baseline answer to an incorrect output in only **4 clusters (0.89%)**, yielding a net win-to-loss ratio of **26.5 to 1**.

---

## 4. Key Metric Distinctions: A Guide for Scientific Audiences

To prevent misinterpretation, presenters must maintain strict conceptual clarity between four distinct dimensions of model quality:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        METRIC DISAMBIGUATION                           │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. Paraphrase Consistency      │ Do equivalent prompts produce identical│
│    (Token F1 / Pairwise EM)    │ or mutually compatible answers?       │
├────────────────────────────────┼───────────────────────────────────────┤
│ 2. Primary Factual Accuracy    │ Does the output match verified gold   │
│    (Gross Cluster Accuracy)    │ reference truth? (Abstentions = 0.0)   │
├────────────────────────────────┼───────────────────────────────────────┤
│ 3. Answer Coverage             │ What percentage of incoming queries    │
│    (% Answered vs. Abstained)  │ does the system attempt to answer?    │
├────────────────────────────────┼───────────────────────────────────────┤
│ 4. Selective Accuracy          │ What is the accuracy of the system     │
│    (Accuracy on Answered)      │ strictly when it chooses to answer?   │
└────────────────────────────────┴───────────────────────────────────────┘
```

- **Gross Accuracy vs. Selective Accuracy in E5:**  
  Under Condition E5 (Conservative Fallback), the system abstains on 41.78% of queries (including 100% of ambiguous queries). Under strict scoring rules where abstention is scored 0.0, E5's gross accuracy is **50.00%**. However, among the queries it answers (Coverage = 58.22%), its selective accuracy reaches **85.88%**. Presenters must never report 85.88% without explicitly stating that it applies to 58.22% coverage.

---

## 5. What Can and Cannot Be Claimed in Presentations

### What Can Defensibly Be Claimed:
- *"Deterministic execution tools (SymPy, safe AST, Pint, datetime) solve LLM arithmetic, modular math, and calendar hallucinations, boosting factual accuracy by up to +89 percentage points in those specific domains."*
- *"Internal consistency routers cannot detect false consensus when all sampled candidates agree on a parametric error; external verification is necessary to break unanimous hallucinations."*
- *"The verification layer introduces virtually zero latency overhead (2.08 ms, <0.08% of total runtime) on modern GPU hardware."*
- *"Condition E4 establishes factual non-inferiority over Base Qwen with a lower 97.5% bound of +17.24 pp, easily meeting the -2.00 pp safety margin."*
- *"Condition E5 provides a high-reliability operational mode that correctly abstains on 100% of ambiguous questions, achieving 85.88% selective accuracy."*

### What Should NOT Yet Be Claimed:
- **Do NOT claim that retrieval alone is statistically superior to E1:** Condition E3 achieved a +1.33 pp gain, but the exact McNemar two-sided p-value is **$p = 0.180$** ($b=10, c=4$), which is **statistically non-significant**.
- **Do NOT claim that E4 outperforms E2 in aggregate accuracy:** Both E2 (Deterministic) and E4 (Combined) report **81.56%** accuracy (367 / 450). While case-level audits reveal that E4 corrected 4 entity hallucinations that E2 missed, E4 also corrupted 4 entity answers that E2 got right, resulting in net aggregate parity.
- **Do NOT claim the system is production-ready for arbitrary web open-domain facts:** The evidence index was tested on 75 curated reference documents. Live open-web retrieval involves noise, adversarial documents, and ranking drift that require further hardening.
- **Do NOT claim that 0% contradictions implies 100% factual truth:** As demonstrated in Phase 4, assigning identical strings guarantees 0 contradictions even when the string is completely false.

---

## 6. Recommended Next Research Directions

1. **Constrained Grammar Decoding for Structured Outputs:**  
   Integrate JSON schema guidance (e.g., Outlines or Guidance) into candidate generation to eliminate the 8 schema violation failures in Stratum 9.
2. **Interactive Disambiguation Dialogues:**  
   Convert the 38 ambiguous failures (Stratum 10) into multi-turn clarifications rather than single-turn refusals.
3. **Live Web Search Retrieval Adapter:**  
   Expand Path B beyond the static 75-document SQLite index by incorporating a live search API with strict source provenance and contradiction filtering.
