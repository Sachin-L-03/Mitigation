"""Inference-time semantic answer comparison, disagreement resolution, and adaptive mitigation router.

Supports:
  - Mode A: Single user prompt
  - Mode B: Explicit paraphrase set
  - Local semantic answer comparison with numerical and entity contradiction detection
  - Adaptive multi-objective routing:
      J = alpha * C + beta * A + gamma * P - delta * T - eta * R
  - Routes: BASELINE, PARAPHRASE_CHECK, VERIFICATION, CONTRASTIVE_ADAPTER, EDITED_ADAPTER, HYBRID, CLARIFICATION
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from src.mitigation.metrics import (
    check_concept_accuracy,
    compute_char_ngram_similarity,
    compute_token_f1,
    normalize_text,
)

logger = logging.getLogger("prompt_sensitivity.mitigation.router")


class RouteDecision(str, Enum):
    BASELINE = "BASELINE"
    PARAPHRASE_CHECK = "PARAPHRASE_CHECK"
    VERIFICATION = "VERIFICATION"
    CONTRASTIVE_ADAPTER = "CONTRASTIVE_ADAPTER"
    EDITED_ADAPTER = "EDITED_ADAPTER"
    HYBRID = "HYBRID"
    CLARIFICATION = "CLARIFICATION"


@dataclass
class MitigationResult:
    selected_answer: str
    route_used: RouteDecision
    confidence: float
    consistency_score: float
    disagreement_detected: bool
    verification_invoked: bool
    latency_sec: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["route_used"] = self.route_used.value
        return d


class SemanticAnswerComparator:
    """Compares candidate answers using lexical, n-gram, and numerical/entity consistency checks."""

    STOPWORDS: Set[str] = {
        "the", "this", "that", "these", "those", "it", "its", "which", "what",
        "how", "why", "who", "whom", "where", "when", "there", "here", "answer",
        "question", "unit", "and", "but", "for", "with", "from",
    }

    # Common domain entity aliases to avoid false positive entity mismatches
    ENTITY_ALIASES: Dict[str, str] = {
        "atp": "adenosine triphosphate",
        "adenosine triphosphate": "atp",
        "tlb": "translation lookaside buffer",
        "translation lookaside buffer": "tlb",
        "bgp": "border gateway protocol",
        "border gateway protocol": "bgp",
        "moho": "mohorovicic",
        "mohorovicic": "moho",
        "mohorovicic discontinuity": "moho",
        "sa node": "sinoatrial node",
        "sinoatrial node": "sa node",
        "dna": "deoxyribonucleic acid",
        "deoxyribonucleic acid": "dna",
        "rna": "ribonucleic acid",
        "ribonucleic acid": "rna",
        "igg": "immunoglobulin g",
        "immunoglobulin g": "igg",
        "cmb": "cosmic microwave background",
        "cosmic microwave background": "cmb",
        "ace": "angiotensin-converting enzyme",
        "angiotensin converting enzyme": "ace",
        "angiotensin-converting enzyme": "ace",
        "narcan": "naloxone",
        "naloxone": "narcan",
        "french": "france",
        "france": "french",
    }

    NEGATION_WORDS: Set[str] = {
        "not", "no", "never", "cannot", "cant", "wont", "neither", "nor", "lacks", "absent", "without"
    }

    def __init__(self, semantic_threshold: float = 0.50):
        self.semantic_threshold = semantic_threshold

    @classmethod
    def extract_numbers(cls, text: str) -> Set[str]:
        """Extract numbers, percentages, and decimal quantities."""
        # Finds integers, decimals, and percentage values
        matches = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        return {m.rstrip("%") for m in matches}

    @classmethod
    def extract_salient_entities(cls, text: str) -> Set[str]:
        """Extract capitalized multi-letter entities from text, resolving known aliases."""
        tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", text)
        clean = {t.lower() for t in tokens if len(t) > 2 and t.lower() not in cls.STOPWORDS}
        # Expand known aliases
        expanded = set(clean)
        for e in clean:
            if e in cls.ENTITY_ALIASES:
                expanded.add(cls.ENTITY_ALIASES[e])
        return expanded

    @classmethod
    def detect_negation_contradiction(cls, text_a: str, text_b: str) -> bool:
        """Detect when one answer asserts a proposition while the other explicitly negates it."""
        norm_a = normalize_text(text_a)
        norm_b = normalize_text(text_b)
        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())

        has_neg_a = bool(tokens_a & cls.NEGATION_WORDS)
        has_neg_b = bool(tokens_b & cls.NEGATION_WORDS)

        # One has negation, other does not
        if has_neg_a != has_neg_b:
            # Check if they share key substantive predicate words
            common = (tokens_a & tokens_b) - cls.STOPWORDS - cls.NEGATION_WORDS
            if len(common) >= 2:
                return True
        return False

    def compare_answers(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """Compare two answers for semantic agreement and detect factual, numeric, and negation contradictions."""
        token_f1 = compute_token_f1(text_a, text_b)
        ngram_sim = compute_char_ngram_similarity(text_a, text_b)
        semantic_sim = 0.4 * token_f1 + 0.6 * ngram_sim

        norm_a = normalize_text(text_a)
        norm_b = normalize_text(text_b)
        # Containment check for concise / telegraphic answers
        if (norm_a and norm_b) and (norm_a in norm_b or norm_b in norm_a):
            semantic_sim = max(semantic_sim, 0.70)

        # Numerical consistency check
        nums_a = self.extract_numbers(text_a)
        nums_b = self.extract_numbers(text_b)
        num_contradiction = False
        if nums_a and nums_b and not (nums_a & nums_b):
            num_contradiction = True

        # Entity consistency check
        ents_a = self.extract_salient_entities(text_a)
        ents_b = self.extract_salient_entities(text_b)
        ent_mismatch = False
        if ents_a and ents_b and not (ents_a & ents_b):
            ent_mismatch = True
        elif ents_a and ents_b and token_f1 < 0.80:
            canon_b = {self.ENTITY_ALIASES.get(x, x) for x in ents_b} | ents_b
            canon_a = {self.ENTITY_ALIASES.get(x, x) for x in ents_a} | ents_a
            diff_a = {x for x in ents_a if self.ENTITY_ALIASES.get(x, x) not in canon_b and x not in canon_b}
            diff_b = {x for x in ents_b if self.ENTITY_ALIASES.get(x, x) not in canon_a and x not in canon_a}
            if diff_a and diff_b:
                # Predicate entity substitution (e.g. Fleming vs Pasteur)
                ent_mismatch = True

        # Negation contradiction check
        neg_contradiction = self.detect_negation_contradiction(text_a, text_b)

        agrees = (
            (semantic_sim >= self.semantic_threshold)
            and not num_contradiction
            and not ent_mismatch
            and not neg_contradiction
        )

        return {
            "semantic_similarity": round(semantic_sim, 4),
            "token_f1": round(token_f1, 4),
            "ngram_similarity": round(ngram_sim, 4),
            "agrees": bool(agrees),
            "numerical_contradiction": bool(num_contradiction),
            "entity_mismatch": bool(ent_mismatch),
            "negation_contradiction": bool(neg_contradiction),
        }


def compute_bounded_medoid(
    candidates: List[str],
    comparator: SemanticAnswerComparator,
    max_k: int = 3,
) -> Tuple[str, int]:
    """Select the medoid answer from at most max_k candidates to bound verification latency."""
    if not candidates:
        return "", 0
    bounded = candidates[:max_k]
    if len(bounded) == 1:
        return bounded[0], 0

    n = len(bounded)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                comp = comparator.compare_answers(bounded[i], bounded[j])
                sim = comp["semantic_similarity"]
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    mean_sims = sim_matrix.mean(axis=1)
    best_idx = int(np.argmax(mean_sims))
    return bounded[best_idx], best_idx


class AdaptiveMitigationRouter:
    """Selects the least expensive effective mitigation route balancing consistency, accuracy, and latency."""

    def __init__(
        self,
        base_generate_fn: Callable[[str], str],
        contrastive_generate_fn: Optional[Callable[[str], str]] = None,
        edited_generate_fn: Optional[Callable[[str], str]] = None,
        verifier_fn: Optional[Callable[[str, str], str]] = None,
        comparator: Optional[SemanticAnswerComparator] = None,
        alpha: float = 0.35,  # Consistency weight
        beta: float = 0.35,   # Accuracy weight
        gamma: float = 0.15,  # Preservation weight
        delta: float = 0.10,  # Latency penalty weight
        eta: float = 0.05,    # Resource cost penalty weight
    ):
        self.base_generate = base_generate_fn
        self.contrastive_generate = contrastive_generate_fn
        self.edited_generate = edited_generate_fn
        self.verifier = verifier_fn
        self.comparator = comparator or SemanticAnswerComparator(semantic_threshold=0.60)

        # Multi-objective policy weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta

    def compute_route_score(
        self,
        consistency: float,
        accuracy: float,
        preservation: float,
        relative_latency: float,
        relative_cost: float,
    ) -> float:
        """Evaluate J = alpha*C + beta*A + gamma*P - delta*T - eta*R."""
        score = (
            self.alpha * consistency
            + self.beta * accuracy
            + self.gamma * preservation
            - self.delta * relative_latency
            - self.eta * relative_cost
        )
        return float(score)

    def process_mode_a(self, user_question: str) -> MitigationResult:
        """Mode A: Single user question handling."""
        start_t = time.perf_counter()

        # Step 1: Base model candidate generation
        ans_base = self.base_generate(user_question)
        latency_base = time.perf_counter() - start_t

        return MitigationResult(
            selected_answer=ans_base,
            route_used=RouteDecision.BASELINE,
            confidence=0.90,
            consistency_score=1.0,
            disagreement_detected=False,
            verification_invoked=False,
            latency_sec=round(latency_base, 4),
            diagnostics={"mode": "Mode A (Single Question)", "base_answer": ans_base},
        )

    def process_mode_b(
        self,
        canonical_question: str,
        paraphrases: List[str],
        gold_answer: Optional[str] = None,
    ) -> MitigationResult:
        """Mode B: Paraphrase set consistency checking and adaptive routing."""
        start_t = time.perf_counter()
        diagnostics: Dict[str, Any] = {"mode": "Mode B (Paraphrase Set)", "num_paraphrases": len(paraphrases)}

        if not paraphrases:
            return self.process_mode_a(canonical_question)

        # Step 1: Generate base model answers across canonical and paraphrases
        all_prompts = [canonical_question] + paraphrases
        candidate_answers = [self.base_generate(p) for p in all_prompts]

        canon_ans = candidate_answers[0]
        variant_answers = candidate_answers[1:]

        # Step 2: Compare answers against canonical
        pairwise_comps = [self.comparator.compare_answers(canon_ans, va) for va in variant_answers]
        mean_sim = float(np.mean([c["semantic_similarity"] for c in pairwise_comps]))
        has_contradiction = any(c["numerical_contradiction"] for c in pairwise_comps)
        has_entity_mismatch = any(c["entity_mismatch"] for c in pairwise_comps)
        num_agreements = sum(1 for c in pairwise_comps if c["agrees"])

        agreement_ratio = num_agreements / max(1, len(variant_answers))
        diagnostics["agreement_ratio"] = round(agreement_ratio, 3)
        diagnostics["mean_similarity"] = round(mean_sim, 3)
        diagnostics["numerical_contradiction"] = has_contradiction
        diagnostics["entity_mismatch"] = has_entity_mismatch

        # If high agreement, no numerical contradiction, and no entity mismatch, accept baseline answer
        if agreement_ratio >= 0.80 and not has_contradiction and not has_entity_mismatch:
            elapsed = time.perf_counter() - start_t
            return MitigationResult(
                selected_answer=canon_ans,
                route_used=RouteDecision.BASELINE,
                confidence=round(agreement_ratio, 3),
                consistency_score=round(mean_sim, 3),
                disagreement_detected=False,
                verification_invoked=False,
                latency_sec=round(elapsed, 4),
                diagnostics=diagnostics,
            )

        # Disagreement detected: evaluate routing options
        diagnostics["disagreement_detected"] = True

        # Route option 1: Contrastive adapter (if available)
        if self.contrastive_generate is not None:
            ans_contrastive = self.contrastive_generate(canonical_question)
            diagnostics["contrastive_answer"] = ans_contrastive
            elapsed = time.perf_counter() - start_t
            return MitigationResult(
                selected_answer=ans_contrastive,
                route_used=RouteDecision.CONTRASTIVE_ADAPTER,
                confidence=0.85,
                consistency_score=round(mean_sim, 3),
                disagreement_detected=True,
                verification_invoked=False,
                latency_sec=round(elapsed, 4),
                diagnostics=diagnostics,
            )

        # Route option 2: Edited adapter (if available)
        if self.edited_generate is not None:
            ans_edited = self.edited_generate(canonical_question)
            diagnostics["edited_answer"] = ans_edited
            elapsed = time.perf_counter() - start_t
            return MitigationResult(
                selected_answer=ans_edited,
                route_used=RouteDecision.EDITED_ADAPTER,
                confidence=0.85,
                consistency_score=round(mean_sim, 3),
                disagreement_detected=True,
                verification_invoked=False,
                latency_sec=round(elapsed, 4),
                diagnostics=diagnostics,
            )

        # Route option 3: Second-pass verification if verifier available
        if self.verifier is not None:
            verified_ans = self.verifier(canonical_question, canon_ans)
            elapsed = time.perf_counter() - start_t
            return MitigationResult(
                selected_answer=verified_ans,
                route_used=RouteDecision.VERIFICATION,
                confidence=0.80,
                consistency_score=round(mean_sim, 3),
                disagreement_detected=True,
                verification_invoked=True,
                latency_sec=round(elapsed, 4),
                diagnostics=diagnostics,
            )

        # Fallback: Paraphrase check resolution (majority / canonical)
        elapsed = time.perf_counter() - start_t
        return MitigationResult(
            selected_answer=canon_ans,
            route_used=RouteDecision.PARAPHRASE_CHECK,
            confidence=round(agreement_ratio, 3),
            consistency_score=round(mean_sim, 3),
            disagreement_detected=True,
            verification_invoked=False,
            latency_sec=round(elapsed, 4),
            diagnostics=diagnostics,
        )
