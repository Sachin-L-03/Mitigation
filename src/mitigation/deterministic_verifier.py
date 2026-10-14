"""Deterministic computation, mathematical formalization, and constraint validation engine.

Path A of Phase 6:
  - Formulates unambiguous mathematical, modular arithmetic, percentage, unit conversion,
    and date calculation queries into deterministic operations.
  - Safe evaluation using AST whitelist and Sympy (no unrestricted eval or shell execution).
  - Explicit delimiter, sorting, and length constraint validation.
  - Returns structured verification records with suggested corrections when confidence is high.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import sympy

logger = logging.getLogger("prompt_sensitivity.mitigation.deterministic")


@dataclass
class DeterministicVerificationResult:
    is_eligible: bool
    task_type: str
    formalized_expression: Optional[str]
    computed_value: Optional[str]
    candidate_matches: bool
    suggested_correction: Optional[str]
    confidence: float
    execution_time_ms: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafeASTEvaluator:
    """Safely evaluates basic arithmetic expressions without eval() or exec()."""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression_str: str) -> Optional[float]:
        clean_expr = expression_str.replace("^", "**").replace("×", "*").replace("÷", "/")
        try:
            tree = ast.parse(clean_expr, mode="eval")
            return cls._eval_node(tree.body)
        except Exception as e:
            logger.debug(f"AST eval failed on '{expression_str}': {e}")
            return None

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Disallowed constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op_type = type(node.op)
            if op_type in cls.ALLOWED_OPERATORS:
                if op_type in {ast.Div, ast.FloorDiv, ast.Mod} and right == 0:
                    raise ZeroDivisionError("Division by zero")
                if op_type == ast.Pow and (right > 1000 or (left == 0 and right < 0)):
                    raise OverflowError("Exponent too large for simple AST")
                return cls.ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Disallowed operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in cls.ALLOWED_OPERATORS:
                return cls.ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"Disallowed unary operator: {op_type}")
        else:
            raise ValueError(f"Disallowed AST node: {type(node)}")


class DeterministicVerifier:
    """Validates and deterministically calculates answers for formalizable tasks."""

    UNIT_CONVERSIONS: Dict[str, Any] = {
        # Temperature
        ("celsius", "kelvin"): lambda c: round(c + 273.15, 2),
        ("kelvin", "celsius"): lambda k: round(k - 273.15, 2),
        ("fahrenheit", "celsius"): lambda f: round((f - 32) * 5 / 9, 2),
        ("celsius", "fahrenheit"): lambda c: round((c * 9 / 5) + 32, 2),
        # Speed
        ("km/h", "m/s"): lambda v: round(v / 3.6, 2),
        ("m/s", "km/h"): lambda v: round(v * 3.6, 2),
        ("mph", "km/h"): lambda v: round(v * 1.609344, 2),
        # Distance
        ("miles", "km"): lambda d: round(d * 1.609344, 2),
        ("miles", "kilometers"): lambda d: round(d * 1.609344, 2),
        ("inches", "cm"): lambda d: round(d * 2.54, 2),
        ("feet", "meters"): lambda d: round(d * 0.3048, 2),
        ("yards", "meters"): lambda d: round(d * 0.9144, 2),
        ("meters", "km"): lambda m: round(m / 1000.0, 4),
        ("km", "meters"): lambda k: round(k * 1000.0, 4),
        ("cm", "meters"): lambda c: round(c / 100.0, 4),
        ("meters", "cm"): lambda m: round(m * 100.0, 4),
        # Area
        ("acres", "sq_ft"): lambda a: int(round(a * 43560)),
        ("acres", "square feet"): lambda a: int(round(a * 43560)),
        ("hectares", "acres"): lambda h: round(h * 2.47105, 2),
        # Volume
        ("gallons", "liters"): lambda g: round(g * 3.78541, 2),
        ("pints", "fluid ounces"): lambda p: int(round(p * 16)),
        # Mass
        ("pounds", "kilograms"): lambda m: round(m * 0.45359237, 2),
        ("kilograms", "pounds"): lambda m: round(m / 0.45359237, 2),
        ("ounces", "grams"): lambda m: round(m * 28.3495, 2),
        ("grams", "kilograms"): lambda g: round(g / 1000.0, 4),
        ("kilograms", "grams"): lambda k: round(k * 1000.0, 4),
    }

    UNIT_ALIASES: Dict[str, str] = {
        "meter": "meters",
        "metre": "meters",
        "metres": "meters",
        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",
        "mile": "miles",
        "foot": "feet",
        "inch": "inches",
        "gallon": "gallons",
        "liter": "liters",
        "litre": "liters",
        "litres": "liters",
        "pound": "pounds",
        "kilogram": "kilograms",
        "ounce": "ounces",
        "gram": "grams",
        "c": "celsius",
        "f": "fahrenheit",
        "k": "kelvin",
    }

    @classmethod
    def _match_number_in_text(cls, val_str: str, text: str) -> bool:
        """Robustly match numerical values avoiding sign confusion, decimal errors, or operand confusion."""
        clean_text = text.replace("−", "-").replace("–", "-").strip()
        clean_val = val_str.replace("−", "-").replace("–", "-").strip()
        try:
            target_num = float(clean_val)
        except ValueError:
            return False

        is_target_negative = target_num < 0
        abs_target_str = re.escape(clean_val.lstrip("+-"))

        # 1. Check explicit answer/result markers (e.g., 'remainder is 4', 'answer is -23', '= 2')
        marker_pattern = r"(?:remainder|answer|result|value|equals|=)\s*(?:is\s*)?([+-]?\d+(?:\.\d+)?)"
        markers = re.findall(marker_pattern, clean_text, flags=re.I)
        if markers:
            try:
                last_marker_val = float(markers[-1])
                if abs(last_marker_val - target_num) < 1e-6:
                    return True
                else:
                    return False
            except ValueError:
                pass

        # 2. Strict signed regex boundary check (block digits or decimal continuations, allow terminal periods)
        abs_raw = clean_val.lstrip("+-")
        abs_targets = [re.escape(abs_raw)]
        if target_num.is_integer():
            int_abs = str(abs(int(target_num)))
            abs_targets.append(re.escape(int_abs))
            abs_targets.append(re.escape(int_abs) + r"\.0")
        abs_targets = list(set(abs_targets))

        for abs_target_str in abs_targets:
            if is_target_negative:
                # Must be preceded by minus
                pattern = r"(?<![\w\.])-" + abs_target_str + r"(?!\d|(?:\.\d))"
            else:
                # Must NOT be preceded by minus or digit/dot
                pattern = r"(?<![\w\.\-])" + abs_target_str + r"(?!\d|(?:\.\d))"

            if re.search(pattern, clean_text):
                return True

        return False

    @classmethod
    def verify_candidate(
        cls,
        prompt: str,
        candidate_answer: str,
    ) -> DeterministicVerificationResult:
        """Analyze prompt and candidate answer; evaluate deterministically if eligible."""
        t0 = time.perf_counter()
        p_lower = prompt.lower()
        cand_lower = candidate_answer.lower()

        # 1. Modular Arithmetic
        # Pattern: "remainder when a^b is divided by m" or "a^b mod m"
        mod_match1 = re.search(r"remainder\s+(?:when|of)\s+(\d+)\s*\^\s*(\d+)\s+(?:is\s+divided\s+by|mod(?:ulo)?)\s+(\d+)", p_lower)
        mod_match2 = re.search(r"(\d+)\s*\^\s*(\d+)\s*(?:mod(?:ulo)?|%)\s*(\d+)", p_lower)
        if mod_match1 or mod_match2:
            m = mod_match1 or mod_match2
            base = int(m.group(1))
            exp = int(m.group(2))
            mod = int(m.group(3))
            if mod > 0:
                val = pow(base, exp, mod)
                val_str = str(val)
                elapsed = (time.perf_counter() - t0) * 1000.0

                cand_matches = cls._match_number_in_text(val_str, candidate_answer)
                correction = val_str if not cand_matches else None
                return DeterministicVerificationResult(
                    is_eligible=True,
                    task_type="modular_arithmetic",
                    formalized_expression=f"pow({base}, {exp}, {mod})",
                    computed_value=val_str,
                    candidate_matches=cand_matches,
                    suggested_correction=correction,
                    confidence=1.0,
                    execution_time_ms=elapsed,
                    reason=f"Modular arithmetic verified deterministically: {base}^{exp} mod {mod} = {val}",
                )

        # 2. Basic / Compound Arithmetic Expressions
        # Pattern: "evaluate (-3)^3 - (-2)^2" or "calculate 48 * 25 / 12"
        arith_expr_match = re.search(r"(?:evaluate|calculate|compute|solve|what is)\s+([0-9\s\+\-\*\/\^\(\)\.\%]+)\??$", p_lower)
        if arith_expr_match:
            expr_str = arith_expr_match.group(1).strip().rstrip("?")
            if any(op in expr_str for op in ["+", "-", "*", "/", "^"]) and re.search(r"\d", expr_str):
                try:
                    # Safe AST execution first
                    safe_val = SafeASTEvaluator.evaluate(expr_str)
                    if safe_val is not None:
                        if safe_val.is_integer():
                            val_str = str(int(safe_val))
                        else:
                            val_str = f"{safe_val:.4f}".rstrip("0").rstrip(".")
                    else:
                        # Fallback to SymPy only on sanitized numeric expressions (no identifiers/code)
                        clean_s = expr_str.replace("^", "**")
                        if re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.]+", clean_s):
                            res_val = sympy.sympify(clean_s)
                            if res_val.is_number:
                                if res_val.is_integer:
                                    val_str = str(int(res_val))
                                else:
                                    val_str = f"{float(res_val):.4f}".rstrip("0").rstrip(".")
                            else:
                                val_str = None
                        else:
                            val_str = None

                    if val_str is not None:
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        cand_matches = cls._match_number_in_text(val_str, candidate_answer)
                        correction = val_str if not cand_matches else None
                        return DeterministicVerificationResult(
                            is_eligible=True,
                            task_type="arithmetic",
                            formalized_expression=expr_str,
                            computed_value=val_str,
                            candidate_matches=cand_matches,
                            suggested_correction=correction,
                            confidence=1.0,
                            execution_time_ms=elapsed,
                            reason=f"Arithmetic expression evaluated: {expr_str} = {val_str}",
                        )
                except Exception as e:
                    logger.debug(f"Arithmetic parsing failed on '{expr_str}': {e}")

        # 3. Percentages & Ratios
        # "What is X% of Y?" or "Calculate X percent of Y"
        pct_match = re.search(r"(?:what is|calculate|compute|find|determine)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", p_lower)
        pct_rev_match = re.search(r"(\d+(?:\.\d+)?)\s+is what (?:percent|percentage) of\s+(\d+(?:\.\d+)?)", p_lower)
        if pct_match:
            pct = float(pct_match.group(1))
            total = float(pct_match.group(2))
            val = (pct / 100.0) * total
            val_str = str(int(val)) if val.is_integer() else f"{val:.2f}".rstrip("0").rstrip(".")
            elapsed = (time.perf_counter() - t0) * 1000.0
            cand_matches = cls._match_number_in_text(val_str, candidate_answer)
            return DeterministicVerificationResult(
                is_eligible=True,
                task_type="percentage",
                formalized_expression=f"({pct} / 100) * {total}",
                computed_value=val_str,
                candidate_matches=cand_matches,
                suggested_correction=val_str if not cand_matches else None,
                confidence=1.0,
                execution_time_ms=elapsed,
                reason=f"Percentage computed: {pct}% of {total} = {val_str}",
            )
        elif pct_rev_match:
            part = float(pct_rev_match.group(1))
            whole = float(pct_rev_match.group(2))
            if whole != 0:
                val = (part / whole) * 100.0
                val_str = str(int(val)) if val.is_integer() else f"{val:.2f}".rstrip("0").rstrip(".")
                elapsed = (time.perf_counter() - t0) * 1000.0
                cand_matches = cls._match_number_in_text(val_str, candidate_answer)
                return DeterministicVerificationResult(
                    is_eligible=True,
                    task_type="percentage",
                    formalized_expression=f"({part} / {whole}) * 100",
                    computed_value=val_str,
                    candidate_matches=cand_matches,
                    suggested_correction=val_str if not cand_matches else None,
                    confidence=1.0,
                    execution_time_ms=elapsed,
                    reason=f"Percentage ratio computed: {part} / {whole} = {val_str}%",
                )

        # 4. Unit Conversions
        # "Convert X [unitA] to [unitB]" or "How many [unitB] in X [unitA]"
        conv_match = re.search(r"convert\s+([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z\/\s]+?)\s+(?:to|in|into)\s+([a-zA-Z\/\s]+)", p_lower)
        alt_conv_match = re.search(r"(?:how many|what is)\s+([a-zA-Z\/\s]+?)\s+(?:in|are in|for)\s+([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z\/\s]+)", p_lower)
        if conv_match or alt_conv_match:
            if conv_match:
                qty = float(conv_match.group(1))
                u_from = conv_match.group(2).strip().lower()
                u_to = conv_match.group(3).strip().lower().rstrip("?")
            else:
                u_to = alt_conv_match.group(1).strip().lower().rstrip("?")
                qty = float(alt_conv_match.group(2))
                u_from = alt_conv_match.group(3).strip().lower().rstrip("?")

            u_from = cls.UNIT_ALIASES.get(u_from, u_from)
            u_to = cls.UNIT_ALIASES.get(u_to, u_to)
            key = (u_from, u_to)
            if key in cls.UNIT_CONVERSIONS:
                fn = cls.UNIT_CONVERSIONS[key]
                val = fn(qty)
                val_str = str(int(val)) if (isinstance(val, (int, float)) and val == int(val)) else str(val)
                elapsed = (time.perf_counter() - t0) * 1000.0
                cand_matches = cls._match_number_in_text(val_str, candidate_answer)
                return DeterministicVerificationResult(
                    is_eligible=True,
                    task_type="unit_conversion",
                    formalized_expression=f"{key[0]}->{key[1]}({qty})",
                    computed_value=val_str,
                    candidate_matches=cand_matches,
                    suggested_correction=val_str if not cand_matches else None,
                    confidence=1.0,
                    execution_time_ms=elapsed,
                    reason=f"Unit converted: {qty} {u_from} = {val_str} {u_to}",
                )

        # 5. Date Arithmetic
        # "How many years elapsed between year X and year Y?" or "From X to Y, how many years?"
        date_diff_match = re.search(r"(?:how many years|years (?:elapsed|passed|between)|number of years from|difference in years between)\s+(?:between\s+|from\s+)?(\d{3,4})\s+(?:and|to)\s+(\d{3,4})", p_lower)
        if date_diff_match:
            y1 = int(date_diff_match.group(1))
            y2 = int(date_diff_match.group(2))
            diff = abs(y2 - y1)
            diff_str = str(diff)
            elapsed = (time.perf_counter() - t0) * 1000.0
            cand_matches = cls._match_number_in_text(diff_str, candidate_answer)
            return DeterministicVerificationResult(
                is_eligible=True,
                task_type="date_math",
                formalized_expression=f"abs({y2} - {y1})",
                computed_value=diff_str,
                candidate_matches=cand_matches,
                suggested_correction=diff_str if not cand_matches else None,
                confidence=1.0,
                execution_time_ms=elapsed,
                reason=f"Date duration calculated: |{y2} - {y1}| = {diff} years",
            )

        # 6. Structured Output Format Constraints
        violations = []
        if "separated by a slash" in p_lower or " / " in prompt:
            if "/" not in candidate_answer:
                violations.append("missing_slash_delimiter")
            elif len([x.strip() for x in candidate_answer.split("/") if x.strip()]) < 2:
                violations.append("insufficient_slash_slots")

        if "symbol:" in p_lower and "number:" in p_lower:
            if "|" not in candidate_answer or ("symbol" not in cand_lower and "number" not in cand_lower):
                violations.append("missing_pipe_key_value_format")

        if "in alphabetical order" in p_lower:
            items = [x.strip().lower() for x in re.split(r"[,;]", candidate_answer) if x.strip()]
            if len(items) >= 2 and items != sorted(items):
                violations.append("alphabetical_order_violation")

        m_words = re.search(r"exactly (\d+) words?", p_lower)
        if m_words:
            expected_n = int(m_words.group(1))
            actual_n = len(candidate_answer.split())
            if actual_n != expected_n:
                violations.append(f"word_count_mismatch_{actual_n}_vs_{expected_n}")

        elapsed = (time.perf_counter() - t0) * 1000.0
        if violations:
            return DeterministicVerificationResult(
                is_eligible=True,
                task_type="constraint_validation",
                formalized_expression="format_and_constraint_rules",
                computed_value=None,
                candidate_matches=False,
                suggested_correction=None,
                confidence=0.90,
                execution_time_ms=elapsed,
                reason=f"Constraint violations detected: {', '.join(violations)}",
            )

        # Non-formalizable or open-ended factual query
        return DeterministicVerificationResult(
            is_eligible=False,
            task_type="unformalized",
            formalized_expression=None,
            computed_value=None,
            candidate_matches=True,
            suggested_correction=None,
            confidence=0.0,
            execution_time_ms=elapsed,
            reason="Query is qualitative, conceptual, or non-deterministic",
        )
