"""Unit tests for src/mitigation/dataset.py."""

import pytest
from pathlib import Path
from src.mitigation.dataset import (
    ClusterDataset,
    EquivalenceStatus,
    NegativeVariant,
    PromptVariant,
    SemanticCluster,
    audit_splits_leakage,
    generate_hard_negatives,
    validate_prompt_equivalence,
)


def test_hard_negative_generation():
    """Verify that hard negative generation creates meaningful constraint changes."""
    text = "What is the capital of France?"
    negs = generate_hard_negatives(text, "c_001")
    assert len(negs) >= 2

    types = {n.contrast_type for n in negs}
    assert "quantity" in types
    assert "scope" in types
    # Confirm negative texts differ from canonical
    for n in negs:
        assert n.text != text
        assert len(n.text) > 0


def test_validate_prompt_equivalence():
    """Verify automated equivalence validator detects negation flips and identical texts."""
    orig = "What is the capital of France?"
    para = "Which city is France's capital?"
    neg = "What is NOT the capital of France?"

    assert validate_prompt_equivalence(orig, orig) == EquivalenceStatus.VERIFIED
    assert validate_prompt_equivalence(orig, para) == EquivalenceStatus.VERIFIED
    assert validate_prompt_equivalence(orig, neg) == EquivalenceStatus.NON_EQUIVALENT


def test_cluster_dataset_pairs_and_leakage():
    """Verify positive/negative pair extraction and zero-leakage split audit."""
    c1 = SemanticCluster(
        cluster_id="c_1",
        canonical_prompt="How does photosynthesis work?",
        variants=[
            PromptVariant("c_1_orig", "How does photosynthesis work?", "original", EquivalenceStatus.VERIFIED),
            PromptVariant("c_1_v1", "What is the mechanism of photosynthesis?", "lexical", EquivalenceStatus.VERIFIED),
        ],
        negatives=[
            NegativeVariant("c_1_neg", "How does cellular respiration work?", "scope", "Opposite process")
        ],
        split="train",
    )

    c2 = SemanticCluster(
        cluster_id="c_2",
        canonical_prompt="What causes earthquakes?",
        variants=[
            PromptVariant("c_2_orig", "What causes earthquakes?", "original", EquivalenceStatus.VERIFIED),
            PromptVariant("c_2_v1", "Why do seismic events happen?", "lexical", EquivalenceStatus.VERIFIED),
        ],
        split="val",
    )

    c3 = SemanticCluster(
        cluster_id="c_3",
        canonical_prompt="What is quantum entanglement?",
        variants=[
            PromptVariant("c_3_orig", "What is quantum entanglement?", "original", EquivalenceStatus.VERIFIED),
        ],
        split="test",
    )

    ds_train = ClusterDataset([c1])
    ds_val = ClusterDataset([c2])
    ds_test = ClusterDataset([c3])

    # Check pair extraction
    pairs = ds_train.get_contrastive_pairs()
    pos_pairs = [p for p in pairs if p["label"] == 1]
    neg_pairs = [p for p in pairs if p["label"] == 0]
    assert len(pos_pairs) == 1
    assert len(neg_pairs) == 2  # each verified variant paired with the hard negative

    # Check leakage audit
    audit = audit_splits_leakage(ds_train, ds_val, ds_test)
    assert audit["zero_leakage_verified"] is True
    assert audit["train_clusters"] == 1
    assert audit["val_clusters"] == 1
    assert audit["test_clusters"] == 1
