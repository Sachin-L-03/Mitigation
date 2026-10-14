# Factual Evaluation Audit and Ground-Truth Integrity Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Target Evaluation**: Factual Correctness Metrics, Grading Methodology, and Benchmark Provenance  
**Date**: October 12, 2026  
**Auditor**: Senior ML Research Engineer  

---

## 1. Executive Summary

This audit critically examines how **factual correctness** has been defined, automated, and interpreted throughout the CREME-Hybrid project.

Prior reports claimed:
- Baseline factual accuracy: $\approx 90\text{--}92\%$
- Verification factual accuracy: $\approx 94.00\%$
- Factual Token F1: $\approx 9.02\%$

### Core Audit Finding:
The project's primary automated factual metric (`check_concept_accuracy`) is an **entity-presence recall filter**, not a full semantic correctness grader. It checks whether key substrings of the gold answer (e.g., `"glutamate"`, `"mitochondria"`, `"insulin"`) appear anywhere within the generated text.

While effective at detecting whether the model retrieved the core concept, **concept presence does not guarantee full factual correctness or the absence of subtle contradictions**. 

---

## 2. Decoupling Factual Dimensions

To prevent misleading claims, we formally separate five distinct factual evaluation outcomes:

| Outcome Level | Metric / Criteria | What It Measures | Automation Feasibility |
|---|---|---|---|
| **Level 1: Concept Retrieval** | `check_concept_accuracy` | Does the response contain the essential named entity or keyword? | Fully automated (string/regex) |
| **Level 2: Semantic Answer Match** | Gold-aligned Token F1 & Embedding Sim | Does the answer directly and concisely answer the question asked? | Automated with length normalization |
| **Level 3: Contradiction Freedom** | `numerical_contradiction` & `negation_contradiction` | Does the response make claims that contradict the gold target? | Automated rule-based detectors |
| **Level 4: Constraint Satisfaction** | Format & unit verification | Does the response obey explicit length, schema, or unit requirements? | Automated regex schema checks |
| **Level 5: Source Grounding** | Literature provenance | Is the answer supported by standard scientific literature? | Curated ground-truth targets |

---

## 3. Provenance and Scale of the Factual Subset

In Phase 2's benchmark (`data/phase2_external_eval/correctness_subset.jsonl`), **50 distinct clusters (300 prompt records)** were curated with verified academic references across:
- **Biochemistry**: Adenosine triphosphate (ATP), Ribosomes, Benzene, Uracil.
- **Pharmacology**: Penicillin (Alexander Fleming, 1928), Naloxone (Narcan).
- **Physics**: Planck's constant, Absolute zero, Tesla, Mohorovicic discontinuity.
- **Computer Science**: Translation Lookaside Buffer (TLB), Dijkstra's algorithm, BGP.
- **Pathology & Immunology**: IgG, Histamine, SA node.

### Audit of Verified References:
Every gold reference was cross-referenced against standard textbook sources (e.g., *Lehninger Principles of Biochemistry*, *Guyton and Hall Textbook of Medical Physiology*, *Jackson Classical Electrodynamics*).

---

## 4. Limitations of Pure Automated Concept Matching

We inspected 50 representative model completions where `check_concept_accuracy` returned `True`:

### Case Study A: True Positive Concept Retrieval (Ideal Case)
- **Question**: "What is the primary excitatory neurotransmitter in the mammalian central nervous system?"
- **Gold**: "Glutamate"
- **Model Completion**: "Glutamate is the principal excitatory neurotransmitter in the mammalian central nervous system, mediating the vast majority of fast excitatory synaptic transmissions."
- **Audit**: `concept_present = True`, `contradiction = False`, `correct_answer = True`.

### Case Study B: Concept Present with Verbose Hedging (Length Discrepancy)
- **Question**: "What is the SI unit of electrical resistance?"
- **Gold**: "Ohm"
- **Model Completion**: "Electrical resistance is a measure of opposition to current flow. In the International System of Units (SI), the unit is the ohm (symbol: Ω), named after Georg Simon Ohm."
- **Audit**: `concept_present = True` (1.0). However, because the completion is 32 words and the gold target is 1 word, Token F1 is only **$11.8\%$** despite being 100% correct.

### Case Study C: Concept Present in Wrong Context (Potential False Alarm)
- **Question**: "Which organ does NOT produce digestive enzymes: liver, pancreas, or gallbladder?"
- **Gold**: "Gallbladder"
- **Model Completion**: "The pancreas and liver produce digestive juices, but the gallbladder stores bile. Therefore, the gallbladder does not produce digestive enzymes."
- **Audit**: All three organ names are present. A naive keyword check for "pancreas" would trigger `True` even if the answer claimed the pancreas was the correct answer. The enhanced parser ensures polarity and concluding clause checks are applied.

---

## 5. Standardized Factual Grading Protocol for Phase 3

For all Phase 3 benchmarks, reporting must explicitly distinguish:
1. **Concept Accuracy (`concept_acc`)**: Percentage of completions containing the verified gold entity.
2. **Contradiction Rate (`contra_rate`)**: Percentage of completions containing opposing numerical, entity, or polarity values.
3. **Strict Composite Accuracy (`strict_acc`)**: Percentage of completions that satisfy:
   $$\text{Concept Present} \land \neg(\text{Numerical Contradiction}) \land \neg(\text{Negation Reversal})$$
