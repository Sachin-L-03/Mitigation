"""Evidence-Grounded Factual Verification and False-Consensus Mitigation Router.

Core router for Phase 6:
  - Integrates E3 consistency baseline with Path A (Deterministic), Path B (Retrieval), and Path C (Verifier).
  - Implements experimental conditions:
      E0: Base Model (single generation)
      E1: E3 Factual-Risk-Aware Bounded Router (consistency baseline)
      E2: E3 + Path A Deterministic Verification
      E3: E3 + Path B Independent Evidence Retrieval
      E4: Combined Evidence-Grounded Router (Path A + Path B + Path C)
      E5: Conservative Evidence-Aware Fallback (abstention on high risk / unverified)
      E6: Offline Oracle Diagnostic Upper Bound (evaluation only)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.mitigation.deterministic_verifier import (
    DeterministicVerificationResult,
    DeterministicVerifier,
)
from src.mitigation.evidence_retriever import EvidenceRetriever, RetrievalResult
from src.mitigation.evidence_verifier import (
    EvidenceVerificationAssessment,
    EvidenceVerifier,
    VerificationAction,
    VerificationState,
)
from src.mitigation.factual_risk_detector import (
    EnhancedSemanticComparator,
    FactualRiskAssessment,
    FactualRiskDetector,
    PromptEquivalenceChecker,
    RiskCategory,
)

logger = logging.getLogger("prompt_sensitivity.mitigation.evidence_router")


class ExperimentalCondition(str, Enum):
    E0_BASE = "E0_base"
    E1_E3_ROUTER = "E1_e3_router"
    E2_DETERMINISTIC = "E2_deterministic"
    E3_RETRIEVAL = "E3_retrieval"
    E4_COMBINED = "E4_combined"
    E5_CONSERVATIVE = "E5_conservative"
    E6_ORACLE = "E6_oracle"


@dataclass
class EvidenceRouterResult:
    request_id: str
    cluster_id: str
    condition: ExperimentalCondition
    selected_answer: str
    raw_candidate_answer: str
    is_corrected: bool
    is_abstained: bool
    correction_source: Optional[str]
    verification_state: str
    risk_score: float
    risk_category: str
    latency_ms: float
    stage_latencies_ms: Dict[str, float]
    deterministic_result: Optional[Dict[str, Any]] = None
    retrieval_result: Optional[Dict[str, Any]] = None
    supporting_passages: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["condition"] = self.condition.value
        return d


class EvidenceGroundedRouter:
    """Multi-path verification and false-consensus mitigation router."""

    def __init__(
        self,
        risk_detector: Optional[FactualRiskDetector] = None,
        retriever: Optional[EvidenceRetriever] = None,
        deterministic_verifier: Optional[DeterministicVerifier] = None,
        evidence_verifier: Optional[EvidenceVerifier] = None,
    ):
        self.risk_detector = risk_detector or FactualRiskDetector()
        self.retriever = retriever or EvidenceRetriever()
        self.deterministic_verifier = deterministic_verifier or DeterministicVerifier()
        self.evidence_verifier = evidence_verifier or EvidenceVerifier()

    def route(
        self,
        request_id: str,
        cluster_id: str,
        prompts: List[str],
        candidates: List[str],
        condition: ExperimentalCondition,
        gold_reference: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRouterResult:
        """Execute specified experimental condition over prompt variants and candidate answers."""
        t_start = time.perf_counter()
        stage_latencies: Dict[str, float] = {}

        if not candidates:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return EvidenceRouterResult(
                request_id=request_id,
                cluster_id=cluster_id,
                condition=condition,
                selected_answer="",
                raw_candidate_answer="",
                is_corrected=False,
                is_abstained=True,
                correction_source="empty_candidates",
                verification_state=VerificationState.VERIFICATION_ERROR.value,
                risk_score=1.0,
                risk_category=RiskCategory.HIGH_RISK.value,
                latency_ms=elapsed,
                stage_latencies_ms={"total": elapsed},
            )

        canonical_prompt = prompts[0] if prompts else ""

        # -------------------------------------------------------------
        # Condition E0: Base Model (Single Generation)
        # -------------------------------------------------------------
        if condition == ExperimentalCondition.E0_BASE:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return EvidenceRouterResult(
                request_id=request_id,
                cluster_id=cluster_id,
                condition=condition,
                selected_answer=candidates[0],
                raw_candidate_answer=candidates[0],
                is_corrected=False,
                is_abstained=False,
                correction_source=None,
                verification_state="UNVERIFIED",
                risk_score=0.0,
                risk_category="NONE",
                latency_ms=elapsed,
                stage_latencies_ms={"routing": elapsed},
                diagnostics={"strategy": "single_generation_base"},
            )

        # -------------------------------------------------------------
        # Step 1: E3 Factual-Risk-Aware Router Baseline (Candidates K >= 1)
        # -------------------------------------------------------------
        t_risk0 = time.perf_counter()
        risk_assessment = self.risk_detector.assess_cluster(
            request_id=request_id,
            cluster_id=cluster_id,
            prompts=prompts,
            candidates=candidates,
        )
        stage_latencies["e3_risk_assessment"] = (time.perf_counter() - t_risk0) * 1000.0

        e3_selected = risk_assessment.selected_answer
        raw_answer = e3_selected

        # Condition E1: Return E3 Baseline directly
        if condition == ExperimentalCondition.E1_E3_ROUTER:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return EvidenceRouterResult(
                request_id=request_id,
                cluster_id=cluster_id,
                condition=condition,
                selected_answer=e3_selected,
                raw_candidate_answer=raw_answer,
                is_corrected=False,
                is_abstained=False,
                correction_source=None,
                verification_state="E3_CONSENSUS",
                risk_score=risk_assessment.risk_score,
                risk_category=risk_assessment.risk_category.value,
                latency_ms=elapsed,
                stage_latencies_ms=stage_latencies,
                diagnostics={"e3_decision": risk_assessment.decision},
            )

        # -------------------------------------------------------------
        # Condition E6: Offline Oracle Diagnostic Upper Bound
        # -------------------------------------------------------------
        if condition == ExperimentalCondition.E6_ORACLE:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            if gold_reference:
                oracle_ans = gold_reference.get("gold_reference_answer", e3_selected)
                is_corr = oracle_ans.strip().lower() != e3_selected.strip().lower()
                return EvidenceRouterResult(
                    request_id=request_id,
                    cluster_id=cluster_id,
                    condition=condition,
                    selected_answer=oracle_ans,
                    raw_candidate_answer=raw_answer,
                    is_corrected=is_corr,
                    is_abstained=False,
                    correction_source="offline_gold_oracle",
                    verification_state="ORACLE_VERIFIED",
                    risk_score=0.0,
                    risk_category="NONE",
                    latency_ms=elapsed,
                    stage_latencies_ms={"oracle": elapsed},
                    diagnostics={"is_diagnostic_only": True},
                )

        # -------------------------------------------------------------
        # Multi-Path Verification Execution
        # -------------------------------------------------------------
        det_result: Optional[DeterministicVerificationResult] = None
        ret_result: Optional[RetrievalResult] = None

        # Path A: Deterministic Tool Check (Eligible for E2, E4, E5)
        if condition in (
            ExperimentalCondition.E2_DETERMINISTIC,
            ExperimentalCondition.E4_COMBINED,
            ExperimentalCondition.E5_CONSERVATIVE,
        ):
            t_det0 = time.perf_counter()
            det_result = self.deterministic_verifier.verify_candidate(
                prompt=canonical_prompt,
                candidate_answer=e3_selected,
            )
            stage_latencies["path_a_deterministic"] = (time.perf_counter() - t_det0) * 1000.0

        # Path B: Evidence Retrieval Check (Eligible for E3, E4, E5)
        # Avoid redundant retrieval in E4/E5 if Path A already decisively confirmed or corrected
        skip_retrieval = False
        if condition in (ExperimentalCondition.E4_COMBINED, ExperimentalCondition.E5_CONSERVATIVE):
            if det_result and det_result.is_eligible and det_result.confidence >= 0.99:
                skip_retrieval = True

        if not skip_retrieval and condition in (
            ExperimentalCondition.E3_RETRIEVAL,
            ExperimentalCondition.E4_COMBINED,
            ExperimentalCondition.E5_CONSERVATIVE,
        ):
            t_ret0 = time.perf_counter()
            ret_result = self.retriever.retrieve(query=canonical_prompt)
            stage_latencies["path_b_retrieval"] = (time.perf_counter() - t_ret0) * 1000.0

        # Path C: Evidence Verifier Harmonization
        t_ver0 = time.perf_counter()
        is_conservative = (condition == ExperimentalCondition.E5_CONSERVATIVE)
        verification_assessment = self.evidence_verifier.verify(
            request_id=request_id,
            prompt=canonical_prompt,
            candidate_answer=e3_selected,
            deterministic_res=det_result,
            retrieval_res=ret_result,
            conservative_fallback=is_conservative,
            prompt_non_equivalent=risk_assessment.prompt_non_equivalent,
        )
        stage_latencies["path_c_verification"] = (time.perf_counter() - t_ver0) * 1000.0

        elapsed = (time.perf_counter() - t_start) * 1000.0
        stage_latencies["total"] = elapsed

        return EvidenceRouterResult(
            request_id=request_id,
            cluster_id=cluster_id,
            condition=condition,
            selected_answer=verification_assessment.verified_answer,
            raw_candidate_answer=raw_answer,
            is_corrected=verification_assessment.is_corrected,
            is_abstained=verification_assessment.is_abstained,
            correction_source=verification_assessment.correction_source,
            verification_state=verification_assessment.verification_state.value,
            risk_score=risk_assessment.risk_score,
            risk_category=risk_assessment.risk_category.value,
            latency_ms=elapsed,
            stage_latencies_ms=stage_latencies,
            deterministic_result=det_result.to_dict() if det_result else None,
            retrieval_result=ret_result.to_dict() if ret_result else None,
            supporting_passages=verification_assessment.supporting_passages,
            diagnostics={
                "action": verification_assessment.action.value,
                "justification": verification_assessment.justification,
                "e3_decision": risk_assessment.decision,
            },
        )
