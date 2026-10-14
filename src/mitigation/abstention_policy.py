"""Conservative abstention and fallback policy for Phase 5.

Implements E4:
  - High risk cases trigger explicit structured abstention or uncertainty emission
  - Evaluates abstention separately from incorrect completions
  - Preserves downstream machine-readable reason codes
  - Handles prompt non-equivalence without forcing consensus
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.mitigation.factual_risk_detector import (
    FactualRiskAssessment,
    FactualRiskDetector,
    RiskCategory,
)

logger = logging.getLogger("prompt_sensitivity.mitigation.abstention")


@dataclass
class PolicyExecutionResult:
    request_id: str
    cluster_id: str
    final_answer: str
    is_abstained: bool
    fallback_type: str
    risk_score: float
    risk_category: str
    assessment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConservativeAbstentionPolicy:
    """Manages routing actions, fallbacks, and uncertainty emissions based on factual risk."""

    UNCERTAINTY_PREFIX = "[UNCERTAINTY_FALLBACK]"

    def __init__(
        self,
        risk_detector: Optional[FactualRiskDetector] = None,
        abstain_on_high_risk: bool = True,
        uncertainty_format: str = "structured_tag",
    ):
        self.detector = risk_detector or FactualRiskDetector()
        self.abstain_on_high_risk = abstain_on_high_risk
        self.uncertainty_format = uncertainty_format

    def execute_policy(
        self,
        request_id: str,
        cluster_id: str,
        prompts: List[str],
        candidates: List[str],
        baseline_answer: str,
    ) -> PolicyExecutionResult:
        assessment = self.detector.assess_cluster(
            request_id=request_id,
            cluster_id=cluster_id,
            prompts=prompts,
            candidates=candidates,
        )

        # 1. Prompt Non-Equivalence (Hard Negative)
        if assessment.prompt_non_equivalent:
            return PolicyExecutionResult(
                request_id=request_id,
                cluster_id=cluster_id,
                final_answer=baseline_answer,
                is_abstained=False,
                fallback_type="INDEPENDENT_RETENTION",
                risk_score=assessment.risk_score,
                risk_category=assessment.risk_category.value,
                assessment=assessment.to_dict(),
            )

        # 2. High Risk Abstention
        if self.abstain_on_high_risk and assessment.risk_category == RiskCategory.HIGH_RISK:
            fallback_msg = (
                f"{self.UNCERTAINTY_PREFIX} High factual risk detected ({assessment.risk_score:.2f}). "
                f"Conflicts: {', '.join(assessment.detected_semantic_conflicts or assessment.constraint_violations or ['disagreement'])}. "
                "Confidence insufficient for reliable factual completion."
            )
            assessment.abstention_fallback_invoked = True
            assessment.decision = "ABSTAIN_UNCERTAINTY"
            return PolicyExecutionResult(
                request_id=request_id,
                cluster_id=cluster_id,
                final_answer=fallback_msg,
                is_abstained=True,
                fallback_type="EXPLICIT_ABSTENTION",
                risk_score=assessment.risk_score,
                risk_category=assessment.risk_category.value,
                assessment=assessment.to_dict(),
            )

        # 3. Low/Moderate Risk: emit consensus medoid answer
        return PolicyExecutionResult(
            request_id=request_id,
            cluster_id=cluster_id,
            final_answer=assessment.selected_answer,
            is_abstained=False,
            fallback_type="NONE_ANSWER_EMITTED",
            risk_score=assessment.risk_score,
            risk_category=assessment.risk_category.value,
            assessment=assessment.to_dict(),
        )
