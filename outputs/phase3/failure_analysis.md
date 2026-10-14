# Router Failure Mode Taxonomy and Diagnostic Report

**Repository**: `prompt-sensitivity-hybrid`  
**Branch**: `creme-hybrid-framework`  
**Date**: October 12, 2026  

---

## 1. Executive Summary

Across the evaluated clusters, a total of **27 discrete failure events** were cataloged and structured into four taxonomy categories.

## 2. Failure Distribution

- **Negation Reversal**: 26 cases (26.0% of clusters)
- **All Candidates Agree Incorrect**: 1 cases (1.0% of clusters)

---

## 3. In-Depth Category Diagnostics

### 3.1 All Candidates Agree Incorrect (Unanimous Hallucination)
- **Mechanism**: When a model's parametric knowledge contains a robust factual error, every linguistic paraphrase elicits the identical false fact. Medoid verification cannot fix this because all candidates agree on the falsehood.
- **Remedy**: Incorporate external ground truth retrieval or calibration-based abstention.

### 3.2 False Positives on Paraphrase Elaboration
- **Mechanism**: Long prompts with extensive background context (`axis_length`) generate verbose completions that trigger low Token F1 despite expressing identical semantic propositions.
- **Remedy**: Character 3-gram and entity matching buffers protect against lexical expansion false alarms.

### 3.3 Negation Polarity Sensitivity
- **Mechanism**: Small models frequently struggle with negative constraints (e.g., 'Which organ does NOT produce digestive enzymes?'), outputting positive assertions.
- **Remedy**: Explicit polarity parser in `detect_negation_contradiction`.
