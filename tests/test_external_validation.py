"""Unit tests for independent external dataset integrity, duplicate audit, and correctness evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from dataset_external import (
    ExternalPromptRecord,
    audit_dataset_overlap,
    compute_char_3gram_similarity,
    compute_token_jaccard,
    load_external_records,
    normalize_prompt_text,
)
from evaluate_external import check_factual_accuracy, extract_keywords_from_gold


def test_external_dataset_integrity():
    """Verify that external dataset contains exactly 100 clusters with 6 variants each."""
    data_path = Path("data/external_eval/external_dataset.jsonl")
    assert data_path.is_file(), "external_dataset.jsonl must exist"

    records = load_external_records(data_path)
    assert len(records) == 600, f"Expected 600 records, got {len(records)}"

    by_cluster = {}
    for r in records:
        by_cluster.setdefault(r.cluster_id, []).append(r.variant_type)

    assert len(by_cluster) == 100, f"Expected 100 clusters, got {len(by_cluster)}"

    expected_variants = {"original", "lexical", "syntactic", "pragmatic", "length", "format"}
    for cid, vtypes in by_cluster.items():
        assert set(vtypes) == expected_variants, f"Cluster {cid} missing variant types: {set(vtypes)}"


def test_factual_correctness_subset_integrity():
    """Verify that factual correctness subset contains 40 clusters with gold reference answers."""
    subset_path = Path("data/external_eval/correctness_subset.jsonl")
    assert subset_path.is_file(), "correctness_subset.jsonl must exist"

    records = load_external_records(subset_path)
    assert len(records) == 240, f"Expected 240 factual prompt records, got {len(records)}"

    by_cluster = {}
    for r in records:
        assert r.is_factual_subset is True
        assert r.gold_reference_answer is not None and len(r.gold_reference_answer.strip()) > 0
        by_cluster.setdefault(r.cluster_id, []).append(r)

    assert len(by_cluster) == 40, f"Expected 40 factual clusters, got {len(by_cluster)}"


def test_zero_overlap_with_original_dataset():
    """Verify that overlap audit between original CSV and external dataset reports zero leakage."""
    audit_path = Path("outputs/external_validation/dataset_overlap_audit.json")
    assert audit_path.is_file(), "dataset_overlap_audit.json must exist"

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["exact_matches_count"] == 0, "Exact matches must be zero"
    assert audit["near_matches_count"] == 0, "Near matches must be zero"
    assert audit["exact_overlap_zero"] is True
    assert audit["near_overlap_zero"] is True


def test_factual_accuracy_matching_logic():
    """Verify factual concept matching logic on representative positive and negative cases."""
    gold_glu = "Glutamate"
    assert check_factual_accuracy("The primary excitatory neurotransmitter is glutamate.", gold_glu) is True
    assert check_factual_accuracy("Glutamate plays a central role in synaptic transmission.", gold_glu) is True
    assert check_factual_accuracy("The answer is GABA, an inhibitory transmitter.", gold_glu) is False

    gold_ace = "Angiotensin-converting enzyme (ACE)"
    assert check_factual_accuracy("Angiotensin-converting enzyme converts Ang I to Ang II.", gold_ace) is True
    assert check_factual_accuracy("ACE inhibitors block this enzyme.", gold_ace) is True
    assert check_factual_accuracy("Renin is produced by juxtaglomerular cells.", gold_ace) is False


def test_adapter_checkpoints_exist_and_valid():
    """Verify both w1 and w5 adapter checkpoints exist and contain valid configuration."""
    w1_dir = Path("outputs/parameter_editing/best_adapter")
    w5_dir = Path("outputs/optimization/preservation/adapter_w5.0")

    for ad_dir in [w1_dir, w5_dir]:
        assert (ad_dir / "adapter_model.safetensors").is_file()
        cfg_file = ad_dir / "adapter_config.json"
        assert cfg_file.is_file()
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            assert cfg.get("peft_type") == "LORA"
