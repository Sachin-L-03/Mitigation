"""End-to-end integration test with mocked components verifying complete pipeline execution.

Does not require downloading full 7B model or GPU memory.
"""

import pytest
import numpy as np

from src.mitigation.dataset import (
    ClusterDataset,
    EquivalenceStatus,
    PromptVariant,
    NegativeVariant,
    SemanticCluster,
)
from src.mitigation.router import (
    AdaptiveMitigationRouter,
    RouteDecision,
    SemanticAnswerComparator,
)
from src.mitigation.ablation_runner import (
    evaluate_predictions_record,
    compare_paired_cluster_scores,
)
from src.mitigation.metrics import compute_bootstrap_ci


def test_end_to_end_mock_ablation_pipeline():
    """Verify end-to-end pipeline execution from clusters to metrics and router."""
    # 1. Create miniature dataset
    clusters = [
        SemanticCluster(
            cluster_id=f"mock_{i}",
            canonical_prompt=f"Question {i}?",
            variants=[
                PromptVariant(f"mock_{i}_orig", f"Question {i}?", "original", EquivalenceStatus.VERIFIED),
                PromptVariant(f"mock_{i}_v1", f"Paraphrase {i}?", "lexical", EquivalenceStatus.VERIFIED),
            ],
            negatives=[NegativeVariant(f"mock_{i}_neg", f"Not question {i}?", "negation", "Opposite")],
            gold_reference_answer="GoldEntity",
            split="val",
        )
        for i in range(5)
    ]
    ds = ClusterDataset(clusters)
    assert len(ds) == 5

    # 2. Mock model generators
    def mock_base_gen(prompt: str) -> str:
        return f"Base answer to {prompt} containing GoldEntity"

    def mock_contrastive_gen(prompt: str) -> str:
        return f"Consistent GoldEntity answer"

    router = AdaptiveMitigationRouter(
        base_generate_fn=mock_base_gen,
        contrastive_generate_fn=mock_contrastive_gen,
        comparator=SemanticAnswerComparator(semantic_threshold=0.50),
    )

    # 3. Test router execution
    res_a = router.process_mode_a("Question 0?")
    assert res_a.route_used == RouteDecision.BASELINE
    assert "GoldEntity" in res_a.selected_answer

    res_b = router.process_mode_b("Question 0?", ["Paraphrase 0?"])
    assert res_b.selected_answer is not None
    assert res_b.latency_sec >= 0.0

    # 4. Mock condition predictions
    base_preds = {
        f"mock_{i}": {
            "original": f"Base answer {i} with GoldEntity",
            "lexical": f"Base answer {i} variant with GoldEntity",
        }
        for i in range(5)
    }

    hybrid_preds = {
        f"mock_{i}": {
            "original": f"Consistent answer {i} with GoldEntity",
            "lexical": f"Consistent answer {i} with GoldEntity",
        }
        for i in range(5)
    }

    gold_map = {f"mock_{i}": "GoldEntity" for i in range(5)}

    # 5. Evaluate predictions
    base_eval = evaluate_predictions_record(base_preds, base_preds, gold_map)
    hybrid_eval = evaluate_predictions_record(hybrid_preds, base_preds, gold_map)

    assert len(base_eval["cluster_canonical_f1"]) == 5
    assert len(hybrid_eval["cluster_canonical_f1"]) == 5
    assert hybrid_eval["diversity_ratio"] > 0.0

    # 6. Paired comparison
    comp = compare_paired_cluster_scores(
        hybrid_eval["cluster_canonical_f1"],
        base_eval["cluster_canonical_f1"],
    )
    assert "abs_change" in comp
    assert "p_value" in comp

    # 7. Bootstrap CI
    mean, low, high = compute_bootstrap_ci(hybrid_eval["cluster_canonical_f1"], num_samples=100)
    assert low <= mean <= high
