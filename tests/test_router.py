"""Unit tests for adaptive mitigation router in src/mitigation/router.py."""

import pytest
from src.mitigation.router import (
    AdaptiveMitigationRouter,
    MitigationResult,
    RouteDecision,
    SemanticAnswerComparator,
)


def test_semantic_comparator_numerical_contradiction():
    """Verify comparator detects numerical contradictions between answers."""
    comparator = SemanticAnswerComparator(semantic_threshold=0.50)

    # Answers with same number
    res_agree = comparator.compare_answers("The vehicle travels at 100 km/h.", "The vehicle speed is 100 km/h.")
    assert res_agree["numerical_contradiction"] is False
    assert res_agree["agrees"] is True

    # Answers with conflicting numbers
    res_disagree = comparator.compare_answers("The vehicle travels at 100 km/h.", "The vehicle travels at 50 km/h.")
    assert res_disagree["numerical_contradiction"] is True
    assert res_disagree["agrees"] is False


def test_router_mode_a_single_question():
    """Verify Mode A routes single questions cleanly to BASELINE."""
    base_fn = lambda q: f"Answer to {q}"
    router = AdaptiveMitigationRouter(base_generate_fn=base_fn)

    res = router.process_mode_a("What is photosynthesis?")
    assert res.route_used == RouteDecision.BASELINE
    assert "photosynthesis" in res.selected_answer
    assert res.disagreement_detected is False


def test_router_mode_b_high_agreement():
    """Verify Mode B with high agreement across variants selects BASELINE."""
    base_fn = lambda q: "Mitochondria generate ATP via oxidative phosphorylation."
    router = AdaptiveMitigationRouter(base_generate_fn=base_fn)

    res = router.process_mode_b(
        canonical_question="Which organelle generates ATP?",
        paraphrases=["What organelle produces ATP in cells?"],
    )
    assert res.route_used == RouteDecision.BASELINE
    assert res.disagreement_detected is False
    assert res.consistency_score > 0.90


def test_router_mode_b_disagreement_routes_to_adapter():
    """Verify Mode B routes to contrastive/edited adapter when variant disagreement is detected."""
    # Base generates conflicting outputs
    responses = {
        "What is the SI unit of resistance?": "The unit is Ohm.",
        "Which SI unit measures electrical resistance?": "The unit is Ampere.",
    }
    base_fn = lambda q: responses.get(q, "Unknown")
    contrastive_fn = lambda q: "Ohm (symbol: Ω)"

    router = AdaptiveMitigationRouter(
        base_generate_fn=base_fn,
        contrastive_generate_fn=contrastive_fn,
    )

    res = router.process_mode_b(
        canonical_question="What is the SI unit of resistance?",
        paraphrases=["Which SI unit measures electrical resistance?"],
    )
    assert res.disagreement_detected is True
    assert res.route_used == RouteDecision.CONTRASTIVE_ADAPTER
    assert "Ohm" in res.selected_answer


def test_entity_alias_resolution_avoids_false_positive():
    """Verify alias mapping avoids declaring entity mismatch between abbreviation and full name."""
    comparator = SemanticAnswerComparator(semantic_threshold=0.40)
    # ATP vs Adenosine Triphosphate
    res = comparator.compare_answers(
        "The primary cellular energy currency is ATP.",
        "The primary cellular energy currency is adenosine triphosphate.",
    )
    assert res["entity_mismatch"] is False
    assert res["agrees"] is True


def test_negation_contradiction_detection():
    """Verify comparator detects explicit negation contradiction on same predicate."""
    comparator = SemanticAnswerComparator(semantic_threshold=0.40)
    # Affirms vs Negates
    res = comparator.compare_answers(
        "The gallbladder produces digestive enzymes for intestinal absorption.",
        "The gallbladder does not produce digestive enzymes for intestinal absorption.",
    )
    assert res["negation_contradiction"] is True
    assert res["agrees"] is False


def test_numerical_percentage_contradiction():
    """Verify comparator detects conflicting percentage quantities."""
    comparator = SemanticAnswerComparator(semantic_threshold=0.40)
    res = comparator.compare_answers(
        "The component accounts for approximately 25% of total mass.",
        "The component accounts for approximately 75% of total mass.",
    )
    assert res["numerical_contradiction"] is True
    assert res["agrees"] is False


def test_bounded_medoid_selection():
    """Verify bounded medoid selects candidate maximizing similarity without copying."""
    from src.mitigation.router import compute_bounded_medoid
    comparator = SemanticAnswerComparator(semantic_threshold=0.40)

    candidates = [
        "Mitochondria generate cellular energy in the form of ATP.",
        "Mitochondria produce ATP as the main energy currency of the cell.",
        "The ribosome synthesizes proteins using messenger RNA.",  # Outlier
    ]
    best_text, best_idx = compute_bounded_medoid(candidates, comparator, max_k=3)
    assert best_idx in [0, 1]
    assert "Mitochondria" in best_text
