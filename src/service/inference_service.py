"""Inference and side-by-side comparison service.

Orchestrates live baseline model generation, Phase 6 evidence-grounded
router execution, pairwise consistency evaluation, semantic divergence analysis,
and metric synthesis on identical prompt sets.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.run_phase6_evaluation import check_factual_correctness
from src.mitigation.evidence_grounded_router import (
    EvidenceGroundedRouter,
    EvidenceRouterResult,
    ExperimentalCondition,
)
from src.mitigation.factual_risk_detector import EnhancedSemanticComparator
from src.mitigation.metrics import compute_token_f1, normalize_text
from src.service.model_service import ModelManager

logger = logging.getLogger("prompt_sensitivity.frontend.inference_service")


class InferenceService:
    """Coordinates execution and comparison of baseline and mitigated models."""

    def __init__(self, router: Optional[EvidenceGroundedRouter] = None):
        self.model_manager = ModelManager.get_instance()
        self.router = router or EvidenceGroundedRouter()
        self.comparator = EnhancedSemanticComparator()

    def run_baseline(
        self,
        session_id: str,
        prompts: List[str],
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute baseline model generation across all approved prompts."""
        t_start = time.perf_counter()
        results: List[Dict[str, Any]] = []

        clean_ref = reference_answer.strip() if reference_answer and reference_answer.strip() else None
        gold_item = self._build_gold_item(clean_ref) if clean_ref else None

        for idx, p_text in enumerate(prompts):
            t0 = time.perf_counter()
            err_msg = None
            try:
                gen_text, lat_ms = self.model_manager.generate(p_text)
            except Exception as e:
                logger.error(f"Baseline generation failed for prompt {idx}: {e}")
                gen_text = ""
                lat_ms = (time.perf_counter() - t0) * 1000.0
                err_msg = str(e)

            is_corr = None
            if gold_item and gen_text:
                is_corr = check_factual_correctness(gen_text, gold_item)

            results.append({
                "prompt_index": idx,
                "prompt_text": p_text,
                "answer": gen_text,
                "latency_ms": round(lat_ms, 2),
                "is_correct": is_corr,
                "status": "error" if err_msg else "success",
                "error": err_msg,
            })

        total_lat_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "session_id": session_id,
            "method": "baseline",
            "model_name": self.model_manager.model_name,
            "mock_mode": self.model_manager.mock_mode,
            "total_latency_ms": round(total_lat_ms, 2),
            "results": results,
            "reference_answer": clean_ref,
        }

    def run_mitigation(
        self,
        session_id: str,
        prompts: List[str],
        baseline_candidates: Optional[List[str]] = None,
        condition: str = "E4_combined",
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Phase 6 mitigation framework on the approved prompt set."""
        t_start = time.perf_counter()

        # If baseline candidates are not supplied, generate them first
        if not baseline_candidates or len(baseline_candidates) != len(prompts):
            base_run = self.run_baseline(session_id, prompts, reference_answer)
            baseline_candidates = [r["answer"] for r in base_run["results"]]

        clean_ref = reference_answer.strip() if reference_answer and reference_answer.strip() else None
        gold_item = self._build_gold_item(clean_ref) if clean_ref else None

        try:
            exp_cond = ExperimentalCondition(condition)
        except ValueError:
            exp_cond = ExperimentalCondition.E4_COMBINED

        results: List[Dict[str, Any]] = []

        for idx, p_text in enumerate(prompts):
            t0 = time.perf_counter()
            err_msg = None
            base_ans = baseline_candidates[idx]

            # Reorder prompts so prompt idx is primary, preserving surrounding context
            aligned_prompts = [p_text] + [p for j, p in enumerate(prompts) if j != idx]
            aligned_cands = [base_ans] + [c for j, c in enumerate(baseline_candidates) if j != idx]

            try:
                res: EvidenceRouterResult = self.router.route(
                    request_id=f"{session_id}_m_{idx}",
                    cluster_id=session_id,
                    prompts=aligned_prompts,
                    candidates=aligned_cands[:3],
                    condition=exp_cond,
                    gold_reference=gold_item if exp_cond == ExperimentalCondition.E6_ORACLE else None,
                )
                final_ans = res.selected_answer
                v_state = res.verification_state
                is_corrected = res.is_corrected
                is_abstained = res.is_abstained
                corr_source = res.correction_source
                route_desc = self._format_route_description(res)
                supp_passages = res.supporting_passages
                det_res = res.deterministic_result
                overhead_ms = res.latency_ms
            except Exception as e:
                logger.error(f"Mitigation router failed for prompt {idx}: {e}")
                final_ans = base_ans
                v_state = "VERIFICATION_ERROR"
                is_corrected = False
                is_abstained = False
                corr_source = "router_exception"
                route_desc = "Router Error"
                supp_passages = []
                det_res = None
                overhead_ms = 0.0
                err_msg = str(e)

            mitigation_lat_ms = (time.perf_counter() - t0) * 1000.0

            is_corr = None
            if gold_item and final_ans:
                is_corr = check_factual_correctness(final_ans, gold_item)

            semantic_rel = self._classify_semantic_relation(base_ans, final_ans)

            results.append({
                "prompt_index": idx,
                "prompt_text": p_text,
                "baseline_answer": base_ans,
                "answer": final_ans,
                "latency_ms": round(mitigation_lat_ms, 2),
                "verification_overhead_ms": round(overhead_ms, 2),
                "verification_route": route_desc,
                "verification_state": v_state,
                "is_corrected": is_corrected,
                "is_abstained": is_abstained,
                "correction_source": corr_source,
                "semantic_relation": semantic_rel,
                "supporting_passages": supp_passages,
                "deterministic_result": det_res,
                "is_correct": is_corr,
                "status": "error" if err_msg else "success",
                "error": err_msg,
            })

        total_lat_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "session_id": session_id,
            "method": "mitigation",
            "condition": exp_cond.value,
            "model_name": self.model_manager.model_name,
            "mock_mode": self.model_manager.mock_mode,
            "total_latency_ms": round(total_lat_ms, 2),
            "results": results,
            "reference_answer": clean_ref,
        }

    def run_both(
        self,
        session_id: str,
        prompts: List[str],
        condition: str = "E4_combined",
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute baseline and mitigation sequentially, producing unified comparison data."""
        base_resp = self.run_baseline(session_id, prompts, reference_answer)
        base_cands = [r["answer"] for r in base_resp["results"]]

        mit_resp = self.run_mitigation(
            session_id=session_id,
            prompts=prompts,
            baseline_candidates=base_cands,
            condition=condition,
            reference_answer=reference_answer,
        )

        metrics = self.compute_comparison_metrics(
            baseline_results=base_resp["results"],
            mitigation_results=mit_resp["results"],
            reference_answer=reference_answer,
        )

        return {
            "session_id": session_id,
            "prompts": prompts,
            "reference_answer": reference_answer,
            "baseline": base_resp,
            "mitigation": mit_resp,
            "metrics": metrics,
        }

    def compute_comparison_metrics(
        self,
        baseline_results: List[Dict[str, Any]],
        mitigation_results: List[Dict[str, Any]],
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute answer consistency, factual accuracy, contradictions, and latency comparisons."""
        n_prompts = len(baseline_results)
        if n_prompts == 0:
            return {}

        base_answers = [r.get("answer", "") for r in baseline_results]
        mit_answers = [r.get("answer", "") for r in mitigation_results]

        # 1. Answer Consistency (Pairwise Token F1)
        base_f1_pairs = []
        for i in range(n_prompts):
            for j in range(i + 1, n_prompts):
                base_f1_pairs.append(compute_token_f1(base_answers[i], base_answers[j]))
        base_consistency = float(np.mean(base_f1_pairs)) if base_f1_pairs else 1.0

        mit_f1_pairs = []
        for i in range(n_prompts):
            for j in range(i + 1, n_prompts):
                mit_f1_pairs.append(compute_token_f1(mit_answers[i], mit_answers[j]))
        mit_consistency = float(np.mean(mit_f1_pairs)) if mit_f1_pairs else 1.0

        # 2. Factual Accuracy (if reference available)
        clean_ref = reference_answer.strip() if reference_answer and reference_answer.strip() else None
        base_correct_count = None
        mit_correct_count = None
        base_acc = None
        mit_acc = None

        if clean_ref:
            base_correct_count = sum(1 for r in baseline_results if r.get("is_correct") is True)
            mit_correct_count = sum(1 for r in mitigation_results if r.get("is_correct") is True)
            base_acc = round((base_correct_count / n_prompts) * 100.0, 2)
            mit_acc = round((mit_correct_count / n_prompts) * 100.0, 2)

        # 3. Contradictions & Differences between Baseline and Mitigation
        n_same = 0
        n_paraphrase = 0
        n_different = 0
        n_contradictions = 0

        for b_ans, m_ans in zip(base_answers, mit_answers):
            rel = self._classify_semantic_relation(b_ans, m_ans)
            if rel == "identical":
                n_same += 1
            elif rel == "paraphrase":
                n_paraphrase += 1
            elif rel == "contradiction":
                n_contradictions += 1
                n_different += 1
            else:
                n_different += 1

        # 4. Latency Statistics
        base_lats = [r.get("latency_ms", 0.0) for r in baseline_results]
        mit_lats = [r.get("latency_ms", 0.0) for r in mitigation_results]
        overheads = [r.get("verification_overhead_ms", 0.0) for r in mitigation_results]

        median_base_lat = float(np.median(base_lats)) if base_lats else 0.0
        median_mit_lat = float(np.median(mit_lats)) if mit_lats else 0.0
        mean_overhead = float(np.mean(overheads)) if overheads else 0.0

        # 5. Verification Pathway Breakdown
        path_counts = {
            "deterministic_ast": 0,
            "evidence_retrieval": 0,
            "consensus_retained": 0,
            "conservative_abstention": 0,
            "other": 0,
        }
        for r in mitigation_results:
            state = r.get("verification_state", "")
            route = r.get("verification_route", "")
            if "Deterministic" in route:
                path_counts["deterministic_ast"] += 1
            elif "Evidence" in route or state == "SUPPORTED":
                path_counts["evidence_retrieval"] += 1
            elif r.get("is_abstained"):
                path_counts["conservative_abstention"] += 1
            elif "Consensus" in route or state == "INSUFFICIENT_EVIDENCE":
                path_counts["consensus_retained"] += 1
            else:
                path_counts["other"] += 1

        return {
            "n_prompts": n_prompts,
            "consistency": {
                "baseline_token_f1_pct": round(base_consistency * 100.0, 2),
                "mitigation_token_f1_pct": round(mit_consistency * 100.0, 2),
                "change_pp": round((mit_consistency - base_consistency) * 100.0, 2),
            },
            "factual_accuracy": {
                "evaluated": clean_ref is not None,
                "reference_answer": clean_ref,
                "baseline_correct": base_correct_count,
                "mitigation_correct": mit_correct_count,
                "baseline_accuracy_pct": base_acc,
                "mitigation_accuracy_pct": mit_acc,
                "denominator": n_prompts,
            },
            "relations": {
                "same_answer_count": n_same,
                "paraphrase_count": n_paraphrase,
                "different_answer_count": n_different,
                "contradiction_count": n_contradictions,
            },
            "latency": {
                "median_baseline_ms": round(median_base_lat, 2),
                "median_mitigation_ms": round(median_mit_lat, 2),
                "mean_verification_overhead_ms": round(mean_overhead, 2),
            },
            "verification_breakdown": path_counts,
        }

    def _classify_semantic_relation(self, ans_a: str, ans_b: str) -> str:
        """Classify relationship: identical, paraphrase, contradiction, or different."""
        norm_a = normalize_text(ans_a)
        norm_b = normalize_text(ans_b)

        if norm_a == norm_b:
            return "identical"

        # Check numerical / sign conflict first
        nums_a = re.findall(r"[+-]?\d+(?:\.\d+)?", ans_a.replace("−", "-"))
        nums_b = re.findall(r"[+-]?\d+(?:\.\d+)?", ans_b.replace("−", "-"))
        if nums_a and nums_b and nums_a[-1] != nums_b[-1]:
            return "contradiction"

        # Check negation conflict
        neg_a = bool(re.search(r"\b(not|never|no|neither|none|cannot|without)\b", ans_a.lower()))
        neg_b = bool(re.search(r"\b(not|never|no|neither|none|cannot|without)\b", ans_b.lower()))
        if neg_a != neg_b:
            return "contradiction"

        f1 = compute_token_f1(ans_a, ans_b)
        if f1 >= 0.75:
            return "paraphrase"

        if f1 >= 0.45:
            return "paraphrase"

        return "different"

    @staticmethod
    def _format_route_description(res: EvidenceRouterResult) -> str:
        """Format a human-readable description of the verification route."""
        source = res.correction_source or ""
        state = res.verification_state

        if "deterministic" in source or (res.deterministic_result and res.deterministic_result.get("is_eligible")):
            return "Path A: Deterministic AST Tooling"
        elif "evidence" in source or res.supporting_passages:
            return "Path B/C: Retrieved Evidence Grounding"
        elif res.is_abstained:
            return "Path C: Conservative Abstention Fallback"
        elif state == "E3_CONSENSUS":
            return "Path B: Risk-Aware Medoid Consensus"
        elif state == "INSUFFICIENT_EVIDENCE":
            return "Path C: Unverified Consensus Fallback"
        return f"Route: {state}"

    @staticmethod
    def _build_gold_item(ref_str: str) -> Dict[str, Any]:
        """Construct evaluation gold item from user-entered reference string."""
        clean = ref_str.strip()
        is_num = bool(re.search(r"^[+-]?\d+(?:\.\d+)?$", clean))
        return {
            "gold_reference_answer": clean,
            "acceptable_aliases": [clean, clean.lower(), clean.capitalize()],
            "correctness_criterion": "numerical_exact" if is_num else "entity_match",
        }
