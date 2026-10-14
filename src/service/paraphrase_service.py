"""Meaning-preserving paraphrase generation and validation service.

Generates 4–5 diverse paraphrases along established linguistic axes (Lexical,
Syntactic, Pragmatic, Concise) while strictly preserving entities, quantities,
dates, units, negations, constraints, and output formats.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.mitigation.factual_risk_detector import PromptEquivalenceChecker

logger = logging.getLogger("prompt_sensitivity.frontend.paraphrase_service")


class ParaphraseService:
    """Generates and validates meaning-preserving prompt paraphrases."""

    AXES = [
        ("Original Prompt", "User-submitted canonical prompt"),
        ("Lexical Formalization", "Formal vocabulary and academic phrasings"),
        ("Syntactic Inversion", "Grammatical clause restructuring and inversion"),
        ("Pragmatic Context", "Domain-framed and analytical query perspective"),
        ("Concise Formulation", "Minimalist, direct formulation preserving all constraints"),
    ]

    @classmethod
    def generate_paraphrases(
        cls,
        prompt: str,
        target_count: int = 5,
    ) -> Dict[str, Any]:
        """Generate target_count paraphrases with the original prompt strictly as index 0.

        Args:
            prompt: User-submitted prompt
            target_count: Total prompt count desired (4 or 5)

        Returns:
            Structured dictionary with original prompt, validated variants, and warnings.
        """
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt cannot be empty.")

        target_count = max(4, min(5, target_count))

        # Always include original as item 0
        variants: List[Dict[str, Any]] = [
            {
                "id": 0,
                "text": clean_prompt,
                "axis": cls.AXES[0][0],
                "description": cls.AXES[0][1],
                "is_original": True,
                "is_valid": True,
                "validation_notes": "Canonical prompt as entered by user.",
            }
        ]

        # Generate candidates along distinct linguistic axes
        generated_candidates = cls._generate_linguistic_variants(clean_prompt)

        # Validate candidates against original
        used_texts = {clean_prompt.lower()}
        variant_idx = 1

        for axis_name, axis_desc, cand_text in generated_candidates:
            if variant_idx >= target_count:
                break

            if cand_text.lower() in used_texts:
                continue

            # Validate equivalence
            is_valid, note = cls._validate_paraphrase(clean_prompt, cand_text)

            variants.append({
                "id": variant_idx,
                "text": cand_text,
                "axis": axis_name,
                "description": axis_desc,
                "is_original": False,
                "is_valid": is_valid,
                "validation_notes": note,
            })
            used_texts.add(cand_text.lower())
            variant_idx += 1

        # If we need more variants to reach target_count, synthesize additional valid variant
        while len(variants) < target_count:
            idx = len(variants)
            fallback_text = cls._synthesize_fallback(clean_prompt, idx)
            is_valid, note = cls._validate_paraphrase(clean_prompt, fallback_text)
            variants.append({
                "id": idx,
                "text": fallback_text,
                "axis": f"Variant {idx + 1}",
                "description": "Systematic transformation",
                "is_original": False,
                "is_valid": is_valid,
                "validation_notes": note,
            })

        return {
            "original_prompt": clean_prompt,
            "count": len(variants),
            "paraphrases": variants,
            "warning": (
                "Notice: Automatic paraphrase validation is heuristic and imperfect. "
                "Please inspect and edit the generated variants below before running live inference."
            ),
        }

    @classmethod
    def _generate_linguistic_variants(cls, p: str) -> List[Tuple[str, str, str]]:
        """Synthesize high-fidelity linguistic paraphrases across formal axes."""
        p_strip = p.rstrip("?.! ")
        p_lower = p.lower()
        variants = []

        # 1. Modular arithmetic pattern: "What is the remainder when a^b is divided by m?"
        m_mod = re.search(r"remainder\s+(?:when|of)\s+(\d+\^\d+)\s+(?:is\s+divided\s+by|mod(?:ulo)?)\s+(\d+)", p_lower)
        if m_mod:
            expr, mod = m_mod.group(1), m_mod.group(2)
            variants.append((cls.AXES[1][0], cls.AXES[1][1], f"Calculate the exact value of {expr} modulo {mod}."))
            variants.append((cls.AXES[2][0], cls.AXES[2][1], f"When {expr} is divided by {mod}, what remainder is obtained?"))
            variants.append((cls.AXES[3][0], cls.AXES[3][1], f"As a number theory problem, find {expr} mod {mod}."))
            variants.append((cls.AXES[4][0], cls.AXES[4][1], f"{expr} mod {mod}?"))
            return variants

        # 2. Arithmetic expressions: "What is (-3)^3 - (-2)^2?" or "Evaluate X"
        if re.search(r"[\^\+\-\*\/]", p) and any(d in p for d in "0123456789"):
            clean_expr = re.sub(r"(?i)^(what is|evaluate|calculate|compute)\s*", "", p_strip).strip()
            variants.append((cls.AXES[1][0], cls.AXES[1][1], f"Evaluate the mathematical expression: {clean_expr}."))
            variants.append((cls.AXES[2][0], cls.AXES[2][1], f"Calculate the exact numerical result of {clean_expr}."))
            variants.append((cls.AXES[3][0], cls.AXES[3][1], f"Determine the evaluated value for {clean_expr}."))
            variants.append((cls.AXES[4][0], cls.AXES[4][1], f"{clean_expr} = ?"))
            return variants

        # 3. Date / Duration arithmetic: "How many years elapsed between Y1 and Y2?"
        m_date = re.search(r"between\s+(\d{3,4})\s+and\s+(\d{3,4})", p_lower)
        if m_date:
            y1, y2 = m_date.group(1), m_date.group(2)
            variants.append((cls.AXES[1][0], cls.AXES[1][1], f"What is the difference in years from {y1} to {y2}?"))
            variants.append((cls.AXES[2][0], cls.AXES[2][1], f"Calculate the number of years that passed between {y1} and {y2}."))
            variants.append((cls.AXES[3][0], cls.AXES[3][1], f"From {y1} to {y2}, how many years elapsed chronologically?"))
            variants.append((cls.AXES[4][0], cls.AXES[4][1], f"Years between {y1} and {y2}?"))
            return variants

        # 4. Percentages: "What is X% of Y?"
        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", p_lower)
        if m_pct:
            pct, total = m_pct.group(1), m_pct.group(2)
            variants.append((cls.AXES[1][0], cls.AXES[1][1], f"Calculate {pct} percent of {total}."))
            variants.append((cls.AXES[2][0], cls.AXES[2][1], f"Find the value corresponding to {pct}% of {total}."))
            variants.append((cls.AXES[3][0], cls.AXES[3][1], f"If a quantity is {total}, determine {pct}% of it."))
            variants.append((cls.AXES[4][0], cls.AXES[4][1], f"{pct}% of {total}?"))
            return variants

        # 5. General entity relation / factual query:
        # e.g., "What is the capital city of Australia?"
        if p_lower.startswith("what is the ") or p_lower.startswith("what is "):
            core = re.sub(r"(?i)^what is (?:the )?", "", p_strip).strip()
            variants.append((cls.AXES[1][0], cls.AXES[1][1], f"Identify the {core}."))
            variants.append((cls.AXES[2][0], cls.AXES[2][1], f"Which entity serves as the {core}?"))
            variants.append((cls.AXES[3][0], cls.AXES[3][1], f"Please state the exact {core}."))
            variants.append((cls.AXES[4][0], cls.AXES[4][1], f"Specify the {core}?"))
            return variants

        if p_lower.startswith("who ") or p_lower.startswith("which "):
            core = p_strip
            variants.append((cls.AXES[1][0], cls.AXES[1][1], f"Identify: {core}."))
            variants.append((cls.AXES[2][0], cls.AXES[2][1], f"Provide the exact answer for: {core}?"))
            variants.append((cls.AXES[3][0], cls.AXES[3][1], f"State clearly {core.lower()}."))
            variants.append((cls.AXES[4][0], cls.AXES[4][1], f"{core}?"))
            return variants

        # 6. Fallback general rephrasings
        variants.append((cls.AXES[1][0], cls.AXES[1][1], f"Please answer the following query: {p_strip}."))
        variants.append((cls.AXES[2][0], cls.AXES[2][1], f"Provide the precise response to: {p_strip}?"))
        variants.append((cls.AXES[3][0], cls.AXES[3][1], f"In terms of factual correctness, {p_strip}?"))
        variants.append((cls.AXES[4][0], cls.AXES[4][1], f"{p_strip}?"))
        return variants

    @classmethod
    def _synthesize_fallback(cls, p: str, idx: int) -> str:
        clean = p.rstrip("?.! ")
        templates = [
            f"Provide the exact answer to: {clean}?",
            f"Please specify: {clean}.",
            f"What is the verified response for {clean}?",
            f"Regarding {clean}, what is the correct result?",
        ]
        return templates[idx % len(templates)]

    @classmethod
    def _validate_paraphrase(cls, original: str, paraphrase: str) -> Tuple[bool, str]:
        """Verify meaning preservation, quantity preservation, and attribute consistency."""
        # 1. Equivalence check via PromptEquivalenceChecker
        if not PromptEquivalenceChecker.are_equivalent(original, paraphrase):
            return False, "Warning: Contrasting attributes or mismatched target entities detected."

        # 2. Check numerical quantities preservation (numbers must not change)
        orig_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", original))
        para_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", paraphrase))
        if orig_nums and not orig_nums.issubset(para_nums):
            missing = orig_nums - para_nums
            return False, f"Warning: Numerical quantities {missing} from the original prompt are missing."

        # 3. Negation preservation
        orig_has_neg = bool(re.search(r"\b(not|never|no|neither|none|without)\b", original.lower()))
        para_has_neg = bool(re.search(r"\b(not|never|no|neither|none|without)\b", paraphrase.lower()))
        if orig_has_neg != para_has_neg:
            return False, "Warning: Polarity/negation mismatch between prompts."

        return True, "Preserves meaning, entities, quantities, and polarity."
