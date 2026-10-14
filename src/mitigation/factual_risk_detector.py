"""Enhanced inference-time factual risk detector, constraint validator, and semantic comparator.

Addresses Phase 4 weaknesses:
  - Signed numerical quantities (-273.15 vs +273.15)
  - Unit and percentage extraction
  - Entity alias resolution and predicate substitution
  - Negation and polarity contradiction
  - Multi-slot extraction and delimiter conformance
  - Prompt equivalence vs hard-negative discrimination
  - Factual risk scoring for false-consensus / unanimous hallucination mitigation
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.mitigation.metrics import (
    compute_char_ngram_similarity,
    compute_token_f1,
    normalize_text,
)

logger = logging.getLogger("prompt_sensitivity.mitigation.factual_risk")


class RiskCategory(str, Enum):
    LOW_RISK = "LOW_RISK"
    MODERATE_RISK = "MODERATE_RISK"
    HIGH_RISK = "HIGH_RISK"


@dataclass
class FactualRiskAssessment:
    request_id: str
    cluster_id: str
    candidate_ids: List[str]
    detected_semantic_conflicts: List[str]
    numeric_values: Dict[str, Set[str]]
    entity_conflicts: bool
    negation_conflicts: bool
    numerical_conflicts: bool
    constraint_violations: List[str]
    prompt_non_equivalent: bool
    vague_hedging_detected: bool
    missing_factual_slot: bool
    risk_score: float
    risk_category: RiskCategory
    decision: str
    decision_reason: str
    verification_invoked: bool
    abstention_fallback_invoked: bool
    candidate_count: int
    total_generation_count: int
    selected_answer: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_category"] = self.risk_category.value
        d["numeric_values"] = {k: list(v) for k, v in self.numeric_values.items()}
        return d


class EnhancedSemanticComparator:
    """Detects subtle contradictions across candidate completions."""

    STOPWORDS: Set[str] = {
        "the", "this", "that", "these", "those", "it", "its", "which", "what",
        "how", "why", "who", "whom", "where", "when", "there", "here", "answer",
        "question", "unit", "and", "but", "for", "with", "from", "state", "list",
        "provide", "identify", "according", "following", "given", "is", "are", "was",
        "were", "in", "of", "to", "by", "as", "at", "an", "a", "calculate", "evaluate",
        "compute", "determine", "find", "solve", "please",
    }

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
        "copenhagen": "kobenhavn",
        "kobenhavn": "copenhagen",
        "warsaw": "warszawa",
        "warszawa": "warsaw",
        "lisbon": "lisboa",
        "lisboa": "lisbon",
        "brasilia": "brasilia",
        "brasília": "brasilia",
        "tokyo": "tokyo",
        "denali": "mount mckinley",
        "mount mckinley": "denali",
        "nile": "nile river",
        "nile river": "nile",
        "sahara": "sahara desert",
        "sahara desert": "sahara",
        "danube": "danube river",
        "danube river": "danube",
        "baikal": "lake baikal",
        "lake baikal": "baikal",
        "victoria": "lake victoria",
        "lake victoria": "victoria",
        "caspian": "caspian sea",
        "caspian sea": "caspian",
        "fleming": "alexander fleming",
        "alexander fleming": "fleming",
        "orwell": "george orwell",
        "george orwell": "orwell",
        "van rossum": "guido van rossum",
        "guido van rossum": "van rossum",
        "suez": "suez canal",
        "suez canal": "suez",
        "gibraltar": "strait of gibraltar",
        "strait of gibraltar": "gibraltar",
    }

    NEGATION_WORDS: Set[str] = {
        "not", "no", "never", "cannot", "cant", "wont", "neither", "nor", "lacks",
        "absent", "without", "none", "non", "fails", "excluding", "except", "prohibited",
    }

    HEDGING_PATTERNS: List[re.Pattern] = [
        re.compile(r"\b(it is debated|debated by historians|no consensus|depending on|depends on the context)\b", re.I),
        re.compile(r"\b(as an ai|as a language model|i cannot provide|it is difficult to say)\b", re.I),
        re.compile(r"\b(some argue|others claim|open to interpretation|various definitions)\b", re.I),
    ]

    def __init__(self, semantic_threshold: float = 0.50):
        self.semantic_threshold = semantic_threshold

    @classmethod
    def extract_signed_numbers(cls, text: str) -> Set[str]:
        """Extract numbers with optional negative/positive sign, decimals, scientific notation, and percentages."""
        # Handles: -273.15, +42, 1.6 x 10^-19, 1.6e-19, 50%, 43560, etc.
        # Preserve sign!
        clean_text = text.replace("−", "-")  # normalize unicode minus
        # Regex for numbers including explicit negative sign
        pattern = r"(?<!\w)[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?"
        matches = re.findall(pattern, clean_text)
        results = set()
        for m in matches:
            s = m.strip()
            if s and s not in {"+", "-"}:
                # Normalize "+5" to "5", but keep "-5"
                if s.startswith("+"):
                    s = s[1:]
                results.add(s)
        return results

    @classmethod
    def extract_units(cls, text: str) -> Set[str]:
        """Extract SI and common physical measurement units, normalizing plurals."""
        pattern = r"\b(watts?|joules?|newtons?|pascals?|kelvins?|celsius|fahrenheit|amperes?|volts?|ohms?|hertz|becquerels?|sieverts?|grays?|candelas?|steradians?|radians?|katals?|moles?|meters?|kilometers?|centimeters?|millimeters?|grams?|kilograms?|pounds?|ounces?|fluid ounces?|pints?|gallons?|liters?|milliliters?|acres?|tons?|seconds?|minutes?|hours?|years?|days?|w|j|n|pa|k|c|v|hz|bq|cd|sr|m|km|cm|mm|kg|g|lbs?|oz|ft|yd)\b"
        matches = re.findall(pattern, text.lower())
        results = set()
        for m in matches:
            m = m.strip()
            if m in {"w", "watt", "watts"}:
                results.add("watt")
            elif m in {"j", "joule", "joules"}:
                results.add("joule")
            elif m in {"n", "newton", "newtons"}:
                results.add("newton")
            elif m in {"pa", "pascal", "pascals"}:
                results.add("pascal")
            elif m in {"k", "kelvin", "kelvins"}:
                results.add("kelvin")
            elif m in {"c", "celsius"}:
                results.add("celsius")
            elif m in {"v", "volt", "volts"}:
                results.add("volt")
            elif m in {"ohm", "ohms"}:
                results.add("ohm")
            elif m in {"hz", "hertz"}:
                results.add("hertz")
            elif m in {"m", "meter", "meters"}:
                results.add("meter")
            elif m in {"kg", "kilogram", "kilograms"}:
                results.add("kilogram")
            elif m in {"g", "gram", "grams"}:
                results.add("gram")
            elif m in {"s", "sec", "second", "seconds"}:
                results.add("second")
            elif m.endswith("s") and len(m) > 3 and not m.endswith("ss"):
                results.add(m[:-1])
            else:
                results.add(m)
        return results

    @classmethod
    def extract_salient_entities(cls, text: str) -> Set[str]:
        """Extract capitalized multi-letter entities from text, resolving known aliases."""
        tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", text)
        clean = {t.lower() for t in tokens if len(t) > 2 and t.lower() not in cls.STOPWORDS}
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

        if has_neg_a != has_neg_b:
            common = (tokens_a & tokens_b) - cls.STOPWORDS - cls.NEGATION_WORDS
            if len(common) >= 1:
                return True
        return False

    @classmethod
    def detect_vague_hedging(cls, text: str) -> bool:
        """Detect uninformative hedges or evasions."""
        for pattern in cls.HEDGING_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def compare_answers(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """Compare two answers for semantic agreement, signed numbers, units, entities, and polarity."""
        token_f1 = compute_token_f1(text_a, text_b)
        ngram_sim = compute_char_ngram_similarity(text_a, text_b)
        semantic_sim = 0.4 * token_f1 + 0.6 * ngram_sim

        norm_a = normalize_text(text_a)
        norm_b = normalize_text(text_b)
        if (norm_a and norm_b) and (norm_a in norm_b or norm_b in norm_a):
            semantic_sim = max(semantic_sim, 0.70)

        # 1. Signed numbers check
        nums_a = self.extract_signed_numbers(text_a)
        nums_b = self.extract_signed_numbers(text_b)
        num_contradiction = False
        if nums_a and nums_b:
            # Check overlap considering stripped percentage/formatting
            overlap = nums_a & nums_b
            if not overlap:
                # Also check stripped percentage versions
                nums_a_clean = {n.rstrip("%") for n in nums_a}
                nums_b_clean = {n.rstrip("%") for n in nums_b}
                if not (nums_a_clean & nums_b_clean):
                    num_contradiction = True

        # 2. Units check
        units_a = self.extract_units(text_a)
        units_b = self.extract_units(text_b)
        unit_mismatch = False
        if units_a and units_b and not (units_a & units_b):
            unit_mismatch = True

        # 3. Entity check
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
                ent_mismatch = True

        # 4. Negation check
        neg_contradiction = self.detect_negation_contradiction(text_a, text_b)

        agrees = (
            (semantic_sim >= self.semantic_threshold)
            and not num_contradiction
            and not unit_mismatch
            and not ent_mismatch
            and not neg_contradiction
        )

        return {
            "semantic_similarity": round(semantic_sim, 4),
            "token_f1": round(token_f1, 4),
            "ngram_similarity": round(ngram_sim, 4),
            "agrees": bool(agrees),
            "numerical_contradiction": bool(num_contradiction),
            "unit_mismatch": bool(unit_mismatch),
            "entity_mismatch": bool(ent_mismatch),
            "negation_contradiction": bool(neg_contradiction),
        }


class ConstraintChecker:
    """Checks output format, delimiter constraints, and slot compliance."""

    @classmethod
    def check_constraints(cls, prompt: str, answer: str) -> List[str]:
        violations = []
        p_lower = prompt.lower()

        # Delimited slash constraint: "separated by a slash" or "X / Y"
        if "separated by a slash" in p_lower or " / " in prompt:
            if "/" not in answer:
                violations.append("missing_slash_delimiter")
            else:
                parts = [p.strip() for p in answer.split("/") if p.strip()]
                if len(parts) < 2:
                    violations.append("insufficient_slash_slots")

        # Delimited pipe constraint: "Symbol: X | Number: Y"
        if "symbol:" in p_lower and "number:" in p_lower:
            if "|" not in answer and ("symbol" not in answer.lower() or "number" not in answer.lower()):
                violations.append("missing_pipe_key_value_format")

        # Delimited author (year) format: "Author (Year)"
        if "author (year)" in p_lower or "creator (year)" in p_lower or "discoverer (year)" in p_lower:
            if not re.search(r"\b\w+\s*\(\d{4}\)", answer):
                violations.append("missing_parenthetical_year_format")

        # Alphabetical order constraint
        if "in alphabetical order" in p_lower:
            # check items separated by commas
            items = [x.strip().lower() for x in re.split(r"[,;]", answer) if x.strip()]
            if len(items) >= 2:
                if items != sorted(items):
                    violations.append("alphabetical_order_violation")

        # Exactly N words constraint
        m_words = re.search(r"exactly (\d+) words?", p_lower)
        if m_words:
            expected_n = int(m_words.group(1))
            actual_n = len(answer.split())
            if actual_n != expected_n:
                violations.append(f"word_count_mismatch_expected_{expected_n}_got_{actual_n}")

        return violations


class PromptEquivalenceChecker:
    """Detects when candidate prompts are NOT semantically equivalent (hard negatives)."""

    CONTRASTING_PAIRS = [
        ("capital", "largest city"),
        ("largest city", "capital"),
        ("celsius", "fahrenheit"),
        ("fahrenheit", "celsius"),
        ("highest mountain", "longest river"),
        ("longest river", "deepest lake"),
        ("atomic number", "mass number"),
        ("protons", "electrons"),
        ("positive", "negative"),
        ("maximum", "minimum"),
    ]

    @classmethod
    def are_equivalent(cls, prompt_a: str, prompt_b: str) -> bool:
        pa = prompt_a.lower()
        pb = prompt_b.lower()
        if pa == pb:
            return True

        # Check contrasting target attributes
        for term1, term2 in cls.CONTRASTING_PAIRS:
            if (term1 in pa and term2 in pb) or (term2 in pa and term1 in pb):
                return False

        # If they ask about different distinct countries or entities
        ents_a = EnhancedSemanticComparator.extract_salient_entities(prompt_a)
        ents_b = EnhancedSemanticComparator.extract_salient_entities(prompt_b)
        if ents_a and ents_b and not (ents_a & ents_b):
            # Different main entities
            return False

        return True


class FactualRiskDetector:
    """Inference-time multi-signal factual risk estimator.

    Computes a risk score without access to gold labels or references.
    Signals:
      1. Candidate slot / semantic disagreements
      2. Numerical contradictions (including signs)
      3. Entity substitution conflicts
      4. Negation / polarity contradictions
      5. Explicit prompt constraint violations
      6. Vague hedging or evasion patterns
      7. Missing factual slot when prompt demands one
      8. Prompt non-equivalence
    """

    def __init__(
        self,
        semantic_threshold: float = 0.50,
        risk_threshold_moderate: float = 0.35,
        risk_threshold_high: float = 0.65,
    ):
        self.comparator = EnhancedSemanticComparator(semantic_threshold=semantic_threshold)
        self.risk_threshold_moderate = risk_threshold_moderate
        self.risk_threshold_high = risk_threshold_high

    def assess_cluster(
        self,
        request_id: str,
        cluster_id: str,
        prompts: List[str],
        candidates: List[str],
    ) -> FactualRiskAssessment:
        candidate_ids = [f"cand_{i}" for i in range(len(candidates))]
        if not candidates:
            return FactualRiskAssessment(
                request_id=request_id,
                cluster_id=cluster_id,
                candidate_ids=[],
                detected_semantic_conflicts=["no_candidates"],
                numeric_values={},
                entity_conflicts=False,
                negation_conflicts=False,
                numerical_conflicts=False,
                constraint_violations=["empty_candidates"],
                prompt_non_equivalent=False,
                vague_hedging_detected=False,
                missing_factual_slot=True,
                risk_score=1.0,
                risk_category=RiskCategory.HIGH_RISK,
                decision="ABSTAIN_EMPTY",
                decision_reason="No candidates produced",
                verification_invoked=False,
                abstention_fallback_invoked=True,
                candidate_count=0,
                total_generation_count=0,
                selected_answer="",
            )

        # 1. Prompt Equivalence Check
        prompt_non_eq = False
        if len(prompts) >= 2:
            for i in range(len(prompts)):
                for j in range(i + 1, len(prompts)):
                    if not PromptEquivalenceChecker.are_equivalent(prompts[i], prompts[j]):
                        prompt_non_eq = True
                        break
                if prompt_non_eq:
                    break

        # 2. Extract numeric values across candidates
        numeric_dict = {f"cand_{i}": self.comparator.extract_signed_numbers(c) for i, c in enumerate(candidates)}

        # 3. Pairwise comparisons across candidates
        semantic_conflicts = []
        num_conflicts = False
        ent_conflicts = False
        neg_conflicts = False
        disagreement_count = 0
        n_pairs = 0

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                n_pairs += 1
                comp = self.comparator.compare_answers(candidates[i], candidates[j])
                if not comp["agrees"]:
                    disagreement_count += 1
                if comp["numerical_contradiction"]:
                    num_conflicts = True
                    semantic_conflicts.append(f"numerical_contradiction_{i}_{j}")
                if comp["entity_mismatch"]:
                    ent_conflicts = True
                    semantic_conflicts.append(f"entity_mismatch_{i}_{j}")
                if comp["negation_contradiction"]:
                    neg_conflicts = True
                    semantic_conflicts.append(f"negation_contradiction_{i}_{j}")

        pair_disagree_rate = disagreement_count / max(1, n_pairs) if n_pairs > 0 else 0.0

        # 4. Constraint validation
        all_violations = []
        for p, c in zip(prompts, candidates):
            v = ConstraintChecker.check_constraints(p, c)
            all_violations.extend(v)
        unique_violations = sorted(list(set(all_violations)))

        # 5. Vague hedging check
        vague_hedging = any(self.comparator.detect_vague_hedging(c) for c in candidates)

        # 6. Missing factual slot check:
        # If prompt specifically asks for year, number, or unit, but candidates contain none
        missing_slot = False
        canon_p = prompts[0].lower()
        if "what year" in canon_p or "in what year" in canon_p or "calendar year" in canon_p:
            # Requires 3-4 digit year
            has_year = any(re.search(r"\b\d{3,4}\b", c) for c in candidates)
            if not has_year:
                missing_slot = True
        elif "si unit" in canon_p or "measured quantity" in canon_p:
            has_unit = any(self.comparator.extract_units(c) for c in candidates)
            if not has_unit:
                missing_slot = True
        elif "how many" in canon_p:
            has_num = any(self.comparator.extract_signed_numbers(c) for c in candidates)
            if not has_num:
                missing_slot = True

        # 7. Multi-Signal Factual Risk Score Computation (Inference-time, Gold-Free)
        risk_components = []
        if prompt_non_eq:
            risk_components.append(0.50)  # prompt mismatch
        if num_conflicts:
            risk_components.append(0.40)  # hard numerical conflict
        if ent_conflicts:
            risk_components.append(0.35)  # entity mismatch
        if neg_conflicts:
            risk_components.append(0.40)  # negation contradiction
        if unique_violations:
            risk_components.append(min(0.50, 0.20 * len(unique_violations)))  # constraint failures
        if vague_hedging:
            risk_components.append(0.35)  # uninformative evasion
        if missing_slot:
            risk_components.append(0.45)  # missing expected slot
        if pair_disagree_rate > 0.0:
            risk_components.append(0.30 * pair_disagree_rate)

        # Composite risk score: bounded in [0, 1]
        if not risk_components:
            raw_risk = 0.05  # low baseline risk
        else:
            # 1 - prod(1 - c) probability-style pooling
            prod = 1.0
            for rc in risk_components:
                prod *= (1.0 - rc)
            raw_risk = 1.0 - prod

        risk_score = float(np.clip(raw_risk, 0.0, 1.0))

        # Risk categorization
        if risk_score >= self.risk_threshold_high:
            risk_category = RiskCategory.HIGH_RISK
        elif risk_score >= self.risk_threshold_moderate:
            risk_category = RiskCategory.MODERATE_RISK
        else:
            risk_category = RiskCategory.LOW_RISK

        # Determine medoid candidate as tentative selected answer
        selected_answer = self._select_medoid(candidates)

        decision = "ACCEPT_CONSENSUS"
        reason = "Candidates consistent and low factual risk"
        verification_invoked = False
        abstention_invoked = False

        if prompt_non_eq:
            decision = "ROUTED_INDEPENDENT"
            reason = "Prompts are semantically non-equivalent; independent answers preserved"
        elif risk_category == RiskCategory.HIGH_RISK:
            decision = "HIGH_RISK_TRIGGER"
            reason = f"High factual risk ({risk_score:.2f}) due to conflicts/violations"
        elif risk_category == RiskCategory.MODERATE_RISK:
            decision = "MODERATE_RISK_VERIFY"
            reason = f"Moderate factual risk ({risk_score:.2f}); verification recommended"
            verification_invoked = True

        return FactualRiskAssessment(
            request_id=request_id,
            cluster_id=cluster_id,
            candidate_ids=candidate_ids,
            detected_semantic_conflicts=semantic_conflicts,
            numeric_values=numeric_dict,
            entity_conflicts=ent_conflicts,
            negation_conflicts=neg_conflicts,
            numerical_conflicts=num_conflicts,
            constraint_violations=unique_violations,
            prompt_non_equivalent=prompt_non_eq,
            vague_hedging_detected=vague_hedging,
            missing_factual_slot=missing_slot,
            risk_score=round(risk_score, 4),
            risk_category=risk_category,
            decision=decision,
            decision_reason=reason,
            verification_invoked=verification_invoked,
            abstention_fallback_invoked=abstention_invoked,
            candidate_count=len(candidates),
            total_generation_count=len(candidates),
            selected_answer=selected_answer,
        )

    def _select_medoid(self, candidates: List[str]) -> str:
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        n = len(candidates)
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    comp = self.comparator.compare_answers(candidates[i], candidates[j])
                    sim = comp["semantic_similarity"]
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        mean_sims = sim_matrix.mean(axis=1)
        best_idx = int(np.argmax(mean_sims))
        return candidates[best_idx]
