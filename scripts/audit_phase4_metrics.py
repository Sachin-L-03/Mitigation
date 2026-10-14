"""Independent Forensic Metric Audit Script for Phase 4.

Directly loads:
  - outputs/phase4/raw_predictions.jsonl
  - data/phase4_domain_eval/correctness_subset.jsonl

Recomputes from ground raw text:
  1. Exact Match Agreements:
     - Pairwise exact match (all 15 pairs per cluster)
     - Canonical exact match (variants vs canonical answer)
     - All-identical cluster rate (all 6 variants identical)
  2. Token F1:
     - Canonical Token F1 (variants vs canonical original)
     - Pairwise Token F1 (macro-average over all pairs)
  3. Semantic Similarity:
     - Char 3-gram cosine similarity
     - Combined semantic proxy
  4. Anchor Preservation:
     - Token F1 of final canonical answer vs baseline canonical answer
     - Exact match preservation
  5. Factual Correctness vs Concept Retrieval:
     - Level 1: Concept retrieval (keyword / entity match)
     - Level 2: Factual accuracy (concept match + no negation reversal + no number conflict)
  6. Contradiction & Consensus Breakdown:
     - Hard numerical contradictions
     - Polarity negation contradictions
     - Predicate entity substitutions
     - Cases where router changes a correct baseline answer to an incorrect candidate
     - Cases where router fixes an incorrect baseline answer
     - Unanimous hallucinations (all candidates agree but wrong against ground truth)

Outputs:
  - outputs/phase4_forensic/recomputed_metrics.json
  - outputs/phase4_forensic/contradiction_audit.csv
  - outputs/phase4_forensic/metric_reconciliation.md
"""

from __future__ import annotations

import csv
import json
import re
import string
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

OUTPUT_DIR = Path("outputs/phase4_forensic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """Standardize text by lowercasing, removing articles, punctuation, and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def compute_token_f1(pred: str, target: str) -> float:
    """Compute token-level F1 score between prediction and target."""
    pred_tokens = normalize_text(pred).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    return float(2 * precision * recall / (precision + recall))


def compute_char_ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute cosine similarity of character n-gram frequency vectors."""
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0

    def get_ngrams(s: str) -> Counter:
        if len(s) < n:
            return Counter([s])
        return Counter([s[i : i + n] for i in range(len(s) - n + 1)])

    ca = get_ngrams(norm_a)
    cb = get_ngrams(norm_b)
    all_keys = set(ca.keys()) | set(cb.keys())
    va = np.array([ca[k] for k in all_keys], dtype=np.float32)
    vb = np.array([cb[k] for k in all_keys], dtype=np.float32)
    norm_prod = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm_prod == 0:
        return 0.0
    return float(np.dot(va, vb) / norm_prod)


def check_concept_retrieval(generated_text: str, gold_reference: str) -> bool:
    """Verify whether generated response contains the factual concept/entity."""
    norm_gen = normalize_text(generated_text)
    if not norm_gen:
        return False
    chunks = re.split(r"[/,()]", gold_reference.lower())
    candidates = [normalize_text(c) for c in chunks if normalize_text(c)]
    if not candidates:
        candidates = [normalize_text(gold_reference)]
    for cand in candidates:
        if cand and cand in norm_gen:
            return True
    if compute_token_f1(generated_text, gold_reference) >= 0.40:
        return True
    return False


STOPWORDS = {
    "the", "this", "that", "these", "those", "it", "its", "which", "what",
    "how", "why", "who", "whom", "where", "when", "there", "here", "answer",
    "question", "unit", "and", "but", "for", "with", "from",
}
NEGATION_WORDS = {"not", "no", "never", "cannot", "cant", "wont", "neither", "nor", "lacks", "absent", "without"}
ENTITY_ALIASES = {
    "atp": "adenosine triphosphate",
    "adenosine triphosphate": "atp",
    "tlb": "translation lookaside buffer",
    "translation lookaside buffer": "tlb",
    "bgp": "border gateway protocol",
    "border gateway protocol": "bgp",
    "french": "france",
    "france": "french",
}


def extract_numbers(text: str) -> Set[str]:
    matches = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    return {m.rstrip("%") for m in matches}


def extract_entities(text: str) -> Set[str]:
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", text)
    clean = {t.lower() for t in tokens if len(t) > 2 and t.lower() not in STOPWORDS}
    expanded = set(clean)
    for e in clean:
        if e in ENTITY_ALIASES:
            expanded.add(ENTITY_ALIASES[e])
    return expanded


def detect_contradictions(text_a: str, text_b: str) -> Dict[str, bool]:
    # 1. Numerical
    nums_a = extract_numbers(text_a)
    nums_b = extract_numbers(text_b)
    num_contra = bool(nums_a and nums_b and not (nums_a & nums_b))

    # 2. Negation
    tokens_a = set(normalize_text(text_a).split())
    tokens_b = set(normalize_text(text_b).split())
    has_neg_a = bool(tokens_a & NEGATION_WORDS)
    has_neg_b = bool(tokens_b & NEGATION_WORDS)
    neg_contra = False
    if has_neg_a != has_neg_b:
        common = (tokens_a & tokens_b) - STOPWORDS - NEGATION_WORDS
        if len(common) >= 2:
            neg_contra = True

    # 3. Entity
    ents_a = extract_entities(text_a)
    ents_b = extract_entities(text_b)
    ent_contra = False
    f1 = compute_token_f1(text_a, text_b)
    if ents_a and ents_b and not (ents_a & ents_b):
        ent_contra = True
    elif ents_a and ents_b and f1 < 0.80:
        canon_b = {ENTITY_ALIASES.get(x, x) for x in ents_b} | ents_b
        canon_a = {ENTITY_ALIASES.get(x, x) for x in ents_a} | ents_a
        diff_a = {x for x in ents_a if ENTITY_ALIASES.get(x, x) not in canon_b and x not in canon_b}
        diff_b = {x for x in ents_b if ENTITY_ALIASES.get(x, x) not in canon_a and x not in canon_a}
        if diff_a and diff_b:
            ent_contra = True

    is_hard = num_contra or neg_contra or ent_contra
    return {
        "hard_contradiction": is_hard,
        "numerical": num_contra,
        "negation": neg_contra,
        "entity": ent_contra,
    }


def main():
    print("=" * 70)
    print("STARTING INDEPENDENT FORENSIC AUDIT OF PHASE 4 METRICS")
    print("=" * 70)

    # 1. Load gold reference map
    gold_map = {}
    with open("data/phase4_domain_eval/correctness_subset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[item["cluster_id"]] = item["gold_reference_answer"]

    # 2. Load raw predictions
    with open("outputs/phase4/raw_predictions.jsonl", "r", encoding="utf-8") as f:
        raw_rows = [json.loads(line) for line in f if line.strip()]

    methods = ["E0", "E3", "E1", "E6_bounded", "E6_enhanced"]
    cids = sorted(list(set(r["cluster_id"] for r in raw_rows)))

    recomputed = {}
    contradiction_audit_rows = []

    # Store baseline canonical answers for anchor preservation
    e0_canonical_answers = {}
    for r in raw_rows:
        if r["method"] == "E0" and r["transformation_type"] == "original":
            e0_canonical_answers[r["cluster_id"]] = r["final_answer"]

    for m in methods:
        m_rows = [r for r in raw_rows if r["method"] == m]
        cluster_data = {}
        for r in m_rows:
            cluster_data.setdefault(r["cluster_id"], {})[r["transformation_type"]] = r["final_answer"]

        # Metric accumulators
        c_token_f1_canon = []
        c_token_f1_pairwise = []
        c_sem_sim_canon = []
        c_sem_sim_pairwise = []
        c_em_canon = []
        c_em_pairwise = []
        c_all_identical = []
        c_anchor_pres_f1 = []
        c_anchor_pres_em = []
        c_concept_retrieval = []
        c_factual_accuracy = []

        total_comparisons = 0
        hard_contradictions = 0
        num_contradictions = 0
        neg_contradictions = 0
        ent_contradictions = 0

        # Consensus correctness tracking
        router_overrode_correct = 0
        router_fixed_incorrect = 0
        unanimous_hallucinations = 0

        for cid in cids:
            variants = cluster_data[cid]
            canon_ans = variants["original"]
            base_canon = e0_canonical_answers[cid]
            gold_ref = gold_map.get(cid, "")

            # Anchor preservation
            c_anchor_pres_f1.append(compute_token_f1(canon_ans, base_canon))
            c_anchor_pres_em.append(1.0 if normalize_text(canon_ans) == normalize_text(base_canon) else 0.0)

            # Variant vs Canonical comparisons
            v_keys = [k for k in variants.keys() if k != "original"]
            v_f1s = []
            v_sims = []
            v_ems = []
            v_concepts = []
            v_facts = []

            for vk in v_keys:
                ans_v = variants[vk]
                f1_val = compute_token_f1(ans_v, canon_ans)
                sim_val = 0.40 * f1_val + 0.60 * compute_char_ngram_similarity(ans_v, canon_ans)
                em_val = 1.0 if normalize_text(ans_v) == normalize_text(canon_ans) else 0.0

                v_f1s.append(f1_val)
                v_sims.append(sim_val)
                v_ems.append(em_val)

                # Contradiction check against canonical
                contra = detect_contradictions(canon_ans, ans_v)
                total_comparisons += 1
                if contra["hard_contradiction"]:
                    hard_contradictions += 1
                if contra["numerical"]:
                    num_contradictions += 1
                if contra["negation"]:
                    neg_contradictions += 1
                if contra["entity"]:
                    ent_contradictions += 1

                # Row-level audit logging for detected contradictions
                if contra["hard_contradiction"] and m in ["E0", "E6_bounded"]:
                    contradiction_audit_rows.append({
                        "method": m,
                        "cluster_id": cid,
                        "variant_type": vk,
                        "canonical_answer": canon_ans,
                        "variant_answer": ans_v,
                        "numerical_conflict": contra["numerical"],
                        "negation_conflict": contra["negation"],
                        "entity_conflict": contra["entity"],
                        "gold_reference": gold_ref,
                    })

                # Correctness
                concept_ok = check_concept_retrieval(ans_v, gold_ref)
                v_concepts.append(1.0 if concept_ok else 0.0)
                # Factual correctness: concept match AND not contradictory to gold
                contra_gold = detect_contradictions(ans_v, gold_ref)
                fact_ok = concept_ok and not contra_gold["negation"] and not contra_gold["numerical"]
                v_facts.append(1.0 if fact_ok else 0.0)

            c_token_f1_canon.append(float(np.mean(v_f1s)))
            c_sem_sim_canon.append(float(np.mean(v_sims)))
            c_em_canon.append(float(np.mean(v_ems)))
            c_concept_retrieval.append(float(np.mean(v_concepts)))
            c_factual_accuracy.append(float(np.mean(v_facts)))

            # Pairwise comparisons across all 6 answers (15 pairs)
            all_answers = list(variants.values())
            pair_f1s = []
            pair_sims = []
            pair_ems = []
            for a1, a2 in combinations(all_answers, 2):
                f1_p = compute_token_f1(a1, a2)
                sim_p = 0.40 * f1_p + 0.60 * compute_char_ngram_similarity(a1, a2)
                em_p = 1.0 if normalize_text(a1) == normalize_text(a2) else 0.0
                pair_f1s.append(f1_p)
                pair_sims.append(sim_p)
                pair_ems.append(em_p)

            c_token_f1_pairwise.append(float(np.mean(pair_f1s)))
            c_sem_sim_pairwise.append(float(np.mean(pair_sims)))
            c_em_pairwise.append(float(np.mean(pair_ems)))

            # All identical check
            first_norm = normalize_text(all_answers[0])
            all_id = 1.0 if all(normalize_text(a) == first_norm and first_norm != "" for a in all_answers) else 0.0
            c_all_identical.append(all_id)

            # Consensus & incorrect override tracking (for E6_bounded vs E0)
            if m == "E6_bounded":
                base_orig = e0_canonical_answers[cid]
                base_correct = check_concept_retrieval(base_orig, gold_ref)
                e6_correct = check_concept_retrieval(canon_ans, gold_ref)

                if base_correct and not e6_correct:
                    router_overrode_correct += 1
                elif not base_correct and e6_correct:
                    router_fixed_incorrect += 1

                # Unanimous hallucination: all 6 answers agree (or medoid selected) but none are correct
                if all_id == 1.0 and not e6_correct:
                    unanimous_hallucinations += 1

        recomputed[m] = {
            "token_f1_canonical": round(float(np.mean(c_token_f1_canon)) * 100, 2),
            "token_f1_pairwise": round(float(np.mean(c_token_f1_pairwise)) * 100, 2),
            "semantic_sim_canonical": round(float(np.mean(c_sem_sim_canon)) * 100, 2),
            "semantic_sim_pairwise": round(float(np.mean(c_sem_sim_pairwise)) * 100, 2),
            "exact_match_canonical": round(float(np.mean(c_em_canon)) * 100, 2),
            "exact_match_pairwise": round(float(np.mean(c_em_pairwise)) * 100, 2),
            "all_identical_cluster_rate": round(float(np.mean(c_all_identical)) * 100, 2),
            "anchor_preservation_f1": round(float(np.mean(c_anchor_pres_f1)) * 100, 2),
            "anchor_preservation_em": round(float(np.mean(c_anchor_pres_em)) * 100, 2),
            "concept_retrieval_rate": round(float(np.mean(c_concept_retrieval)) * 100, 2),
            "factual_accuracy": round(float(np.mean(c_factual_accuracy)) * 100, 2),
            "hard_contradiction_rate": round(hard_contradictions / total_comparisons * 100, 2),
            "numerical_contradictions": num_contradictions,
            "negation_contradictions": neg_contradictions,
            "entity_contradictions": ent_contradictions,
            "router_overrode_correct": router_overrode_correct,
            "router_fixed_incorrect": router_fixed_incorrect,
            "unanimous_hallucinations": unanimous_hallucinations,
        }

    # Save recomputed metrics JSON
    json_path = OUTPUT_DIR / "recomputed_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recomputed, f, indent=2)
    print(f"Saved {json_path}")

    # Save contradiction audit CSV
    csv_path = OUTPUT_DIR / "contradiction_audit.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method", "cluster_id", "variant_type", "canonical_answer",
            "variant_answer", "numerical_conflict", "negation_conflict",
            "entity_conflict", "gold_reference"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in contradiction_audit_rows:
            writer.writerow(row)
    print(f"Saved {csv_path} ({len(contradiction_audit_rows)} rows)")

    # Print summary comparison
    print("\nRECOMPUTED METRIC SUMMARY:")
    for m in ["E0", "E3", "E1", "E6_bounded"]:
        r = recomputed[m]
        print(f"[{m}] Token F1 (Canon): {r['token_f1_canonical']}% | Pairwise EM: {r['exact_match_pairwise']}% | All-Identical: {r['all_identical_cluster_rate']}% | Fact: {r['factual_accuracy']}% | Contradictions: {r['hard_contradiction_rate']}%")

    if "E6_bounded" in recomputed:
        e6_r = recomputed["E6_bounded"]
        print(f"\nConsensus Dynamics: Router overrode correct: {e6_r['router_overrode_correct']} clusters | Router fixed incorrect: {e6_r['router_fixed_incorrect']} clusters | Unanimous hallucinations: {e6_r['unanimous_hallucinations']} clusters")


if __name__ == "__main__":
    main()
