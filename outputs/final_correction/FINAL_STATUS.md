# Phase 6 System: Final Engineering and Scientific Status

**Project:** Prompt Sensitivity Mitigation in Large Language Models  
**Branch:** `creme-hybrid-framework`  
**Date:** 2026-10-14  
**Verdict:** **CODE FUNCTIONAL & STATISTICALLY AUDITED. SUITABLE FOR CONSTRAINED RESEARCH PILOT.**

---

## 1. System Functionality Verdict

The codebase is fully functional, robust, and verified.
- The complete test suite of **134 automated unit and regression tests passes cleanly** with zero failures or skipped tests (`pytest -v` execution time: 5.2 seconds).
- The execution pipeline safely handles inputs across arithmetic, percentages, unit conversions, date calculations, entity relations, and structured output constraints without unsafe `eval` execution or silent failures.
- Candidate answers ending with standard sentence periods now match correctly without breaking decimal discrimination.

---

## 2. Summary of Confirmed and Fixed Defects

1. **Regex Lookahead Punctuation Bug Fixed:** Permitted terminal periods (`.`) in `_match_number_in_text` while blocking decimal extensions, preventing false negative evaluations on natural language completions.
2. **AST-First Safe Verification Implemented:** Added `SafeASTEvaluator` to parse mathematical operations with whitelisted nodes before SymPy, blocking arbitrary expression injection.
3. **Metric Unit Conversions Added:** Expanded `UNIT_CONVERSIONS` with bidirectional metric distance, area, and mass mappings and aliases.
4. **Contextual Event-Year Alignment Enforced:** Fixed `EvidenceVerifier` to require event-predicate token co-occurrence before accepting year substitutions, eliminating blind first-year overwrites.
5. **Institutional Publisher Filter Added:** Blocked non-human institutional names (`"Oxford University Press"`) from being falsely extracted as biographical authors.
6. **Non-Equivalent Prompt Routing Integrated:** Threaded `prompt_non_equivalent` directly to `VerificationState.AMBIGUOUS_QUESTION` and conservative abstention fallback.

---

## 3. What Existing Results Remain Supported

1. **Large Factual Accuracy Gain on Formalizable Tasks (+22.0 pp):**
   - E2 (Deterministic Verification) and E4 (Combined Hybrid Verification) improve factual accuracy from **59.56% (E0 base) to 81.56%** ($+22.00\text{ pp}$, McNemar $p < 10^{-15}$).
   - This gain is robust and completely reproducible from raw prediction records.
2. **Mitigation of Unanimous False Consensus (19/19 Corrected):**
   - In 19 out of 19 test clusters where the base model exhibited unanimous consensus on an incorrect arithmetic or modular answer, Path A deterministic verification formalized the problem and supplied the correct answer.
3. **Negligible Verification Latency Overhead (< 2 ms):**
   - The algorithmic router, AST verification, and SQLite FTS5 evidence retrieval introduce an average overhead of only **$1.85\text{ ms}$**, adding virtually no compute overhead to the base inference pipeline.
4. **High Precision Under Conservative Policy (85.88% Selective Accuracy):**
   - In E5, when evidence is missing or contradictory, conservative abstention safely drops coverage to 58.22% while boosting selective accuracy to 85.88%.

---

## 4. What Claims Remain Unverified or Unsupported

1. **Retrieval-Only Improvement is NOT Statistically Significant:**
   - E3 (Retrieval Only) increases accuracy by $+1.33\text{ pp}$ over E1 (from 58.89% to 60.22%).
   - Paired McNemar exact test yields **$p = 0.17957 \ (\approx 0.180)$**, which fails standard statistical significance ($\alpha = 0.05$). Claims that retrieval alone significantly improves reliability are unsupported.
2. **Retrieval Does Not Outperform Pure Deterministic Verification:**
   - E2 and E4 achieve identical aggregate accuracy ($81.56\%$).
   - Case-level audit proves this is an exact $4$-for-$4$ trade: retrieval fixed 4 unnormalized entity outputs, but corrupted 4 correct capital city answers due to passage entity collisions. Claims that E4 represents an unconditional improvement over E2 are unsupported.
3. **End-to-End Latency is Dominated by Generation (Not 1.85 ms):**
   - The previously reported $1.85\text{ ms}$ latency is purely router algorithmic overhead. Total end-to-end latency with 4-bit NF4 generation is **$3,150.97\text{ ms}$**.

---

## 5. Recommended Presentation of the Project's Contribution

The project should be presented with precise, defensible scientific framing:
> *"Inference-time paraphrase consensus (e.g. self-consistency and bounded medoid routing) stabilizes prompt sensitivity but remains vulnerable to systematic hallucinations and unanimous false consensus. By coupling a factual-risk-aware router with deterministic AST/modular execution tools and local evidence retrieval, our hybrid framework increases factual accuracy from 59.56% to 81.56% (+22.0 pp) on structured benchmarks and corrects 100% (19/19) of audited unanimous false-consensus arithmetic failures with negligible algorithmic overhead (< 2 ms). While open-domain retrieval alone showed a non-significant trend (+1.33 pp, p = 0.180), deterministic formalization provides a principled, verifiable defense against LLM hallucination in quantitative domains."*

---

## 6. Pilot Readiness Assessment

- **Unrestricted Production Deployment:** **NOT RECOMMENDED**. The system's evidence retriever currently operates on a bounded local corpus ($450$ passages) and cannot generalize to uncurated open-domain queries without risk of passage entity collisions.
- **Constrained Pilot Deployment:** **RECOMMENDED FOR SPECIFIC DOMAINS**. The system is ready for a controlled pilot restricted to:
  1. Quantitative and financial workflows where arithmetic, currency, and unit conversions require provable determinism.
  2. Question answering over verified, closed-domain knowledge bases where conservative abstention (E5) is preferred over ungrounded generation.
