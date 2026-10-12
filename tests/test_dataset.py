"""Automated unit tests for dataset schema, axis-tagged CSV loading, variant mapping, and zero-leakage cluster splitting."""

import json
from pathlib import Path
import pytest

from dataset import (
    AXIS_COLUMNS,
    PromptDataset,
    PromptRecord,
    cluster_train_val_test_split,
    save_dataset_and_splits,
    verify_split_leakage,
)


def test_prompt_record_valid():
    """Verify that a well-formed PromptRecord instantiates and validates correctly."""
    rec = PromptRecord(
        cluster_id="1",
        prompt_id="1_original",
        prompt_text="What is machine learning?",
        variant_type="original",
        reference_answer=None,
        perturbation_metadata={"axis": "original", "severity": 0},
    )
    rec.validate()
    d = rec.to_dict()
    assert d["cluster_id"] == "1"
    assert d["prompt_id"] == "1_original"
    assert d["variant_type"] == "original"
    assert d["reference_answer"] is None

    # Test deserialization
    rec2 = PromptRecord.from_dict(d)
    assert rec2.prompt_text == rec.prompt_text
    assert rec2.reference_answer is None
    assert rec2.variant_type == "original"


@pytest.mark.parametrize(
    "cluster_id,prompt_id,prompt_text,variant_type",
    [
        ("", "id_1", "text", "original"),
        ("   ", "id_1", "text", "original"),
        ("1", "", "text", "original"),
        ("1", "id_1", "", "original"),
        ("1", "id_1", "text", ""),
    ],
)
def test_prompt_record_invalid_fields(cluster_id, prompt_id, prompt_text, variant_type):
    """Ensure invalid field types or empty strings raise ValueError."""
    with pytest.raises(ValueError):
        PromptRecord(
            cluster_id=cluster_id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            variant_type=variant_type,
        ).validate()


def test_duplicate_prompt_id_rejected():
    """Verify that duplicate prompt_ids in the dataset are rejected with ValueError."""
    ds = PromptDataset(
        [
            PromptRecord("1", "p1", "text 1", "original"),
            PromptRecord("1", "p1", "duplicate p1", "lexical"),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate prompt_id detected"):
        ds.validate()


def test_csv_loading_and_expected_counts():
    """Verify that paraphrased_prompts_axis_tagged_qa.csv loads exactly 196 clusters and 1,176 records."""
    csv_file = Path("data/paraphrased_prompts_axis_tagged_qa.csv")
    assert csv_file.is_file(), "Source CSV must exist at data/paraphrased_prompts_axis_tagged_qa.csv"

    dataset = PromptDataset.from_axis_tagged_csv(csv_file, validate_expected_counts=True)
    stats = dataset.validate()

    assert stats["unique_clusters"] == 196, f"Expected 196 clusters, got {stats['unique_clusters']}"
    assert stats["total_records"] == 1176, f"Expected 1,176 prompt records, got {stats['total_records']}"
    assert stats["unique_prompts"] == 1176, "All prompt IDs must be unique"

    # Verify each cluster has exactly 6 records
    for cluster_id, count in stats["cluster_counts"].items():
        assert count == 6, f"Cluster '{cluster_id}' expected 6 records, got {count}"


def test_csv_variant_mapping():
    """Verify variant_type mapping: 1 original + 5 linguistic variants per cluster."""
    csv_file = Path("data/paraphrased_prompts_axis_tagged_qa.csv")
    dataset = PromptDataset.from_axis_tagged_csv(csv_file)

    cluster_to_variants = {}
    for r in dataset.records:
        cluster_to_variants.setdefault(r.cluster_id, []).append(r.variant_type)
        # Verify reference_answer is strictly None (null) - no fake answers
        assert r.reference_answer is None, f"Expected reference_answer to be None, got {r.reference_answer}"

    expected_variant_set = {"original", "lexical", "syntactic", "pragmatic", "length", "format"}
    for cluster_id, variants in cluster_to_variants.items():
        assert set(variants) == expected_variant_set
        assert len(variants) == 6


def test_csv_preserves_original_cluster_ids():
    """Verify that original CSV 'id' values (e.g. '1', '2', ..., '196') are preserved as cluster_ids."""
    csv_file = Path("data/paraphrased_prompts_axis_tagged_qa.csv")
    dataset = PromptDataset.from_axis_tagged_csv(csv_file)
    clusters = dataset.get_clusters()

    expected_ids = {str(i) for i in range(1, 197)}
    assert clusters == expected_ids, f"Cluster IDs mismatch: difference={clusters.symmetric_difference(expected_ids)}"


def test_csv_missing_values_rejected(tmp_path):
    """Verify that missing values in any CSV column cause validation to raise ValueError."""
    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", encoding="utf-8") as f:
        f.write("id,original,lexical,syntactic,pragmatic,length,format\n")
        f.write("1,What is water?,What is H2O?,,How about water?,Water?,WATER:\n")  # syntactic is empty

    with pytest.raises(ValueError, match="Missing or empty variant 'syntactic'"):
        PromptDataset.from_axis_tagged_csv(bad_csv)


def test_cluster_split_ratio_and_cohesion():
    """Verify 70/15/15 ratio, cluster cohesion, and zero leakage."""
    csv_file = Path("data/paraphrased_prompts_axis_tagged_qa.csv")
    dataset = PromptDataset.from_axis_tagged_csv(csv_file)

    train_ds, val_ds, test_ds = cluster_train_val_test_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    leakage = verify_split_leakage(train_ds, val_ds, test_ds)
    assert leakage["leakage_free"] is True

    # Exact expected counts for 196 clusters at 70/15/15:
    assert leakage["train_cluster_count"] == 137
    assert leakage["val_cluster_count"] == 29
    assert leakage["test_cluster_count"] == 30
    assert len(train_ds) == 137 * 6  # 822
    assert len(val_ds) == 29 * 6    # 174
    assert len(test_ds) == 30 * 6   # 180
    assert len(train_ds) + len(val_ds) + len(test_ds) == 1176

    # Verify all 6 variants of each cluster remain strictly together
    train_clusters = train_ds.get_clusters()
    val_clusters = val_ds.get_clusters()
    test_clusters = test_ds.get_clusters()

    for r in train_ds.records:
        assert r.cluster_id in train_clusters
        assert r.cluster_id not in val_clusters
        assert r.cluster_id not in test_clusters

    for r in val_ds.records:
        assert r.cluster_id in val_clusters
        assert r.cluster_id not in train_clusters
        assert r.cluster_id not in test_clusters

    for r in test_ds.records:
        assert r.cluster_id in test_clusters
        assert r.cluster_id not in train_clusters
        assert r.cluster_id not in val_clusters


def test_cluster_split_deterministic_reproducibility():
    """Verify that splitting with the same seed produces identical subset records."""
    csv_file = Path("data/paraphrased_prompts_axis_tagged_qa.csv")
    dataset = PromptDataset.from_axis_tagged_csv(csv_file)

    train_1, val_1, test_1 = cluster_train_val_test_split(dataset, seed=42)
    train_2, val_2, test_2 = cluster_train_val_test_split(dataset, seed=42)

    assert [r.prompt_id for r in train_1] == [r.prompt_id for r in train_2]
    assert [r.prompt_id for r in val_1] == [r.prompt_id for r in val_2]
    assert [r.prompt_id for r in test_1] == [r.prompt_id for r in test_2]


def test_saved_split_files_on_disk():
    """Verify that files written to data/splits/ match schema and counts."""
    splits_dir = Path("data/splits")
    assert (splits_dir / "canonical_dataset.jsonl").is_file()
    assert (splits_dir / "train.jsonl").is_file()
    assert (splits_dir / "val.jsonl").is_file()
    assert (splits_dir / "test.jsonl").is_file()
    assert (splits_dir / "split_metadata.json").is_file()

    # Load canonical dataset from disk and verify
    ds_disk = PromptDataset.load_jsonl(splits_dir / "canonical_dataset.jsonl")
    assert len(ds_disk) == 1176
    assert len(ds_disk.get_clusters()) == 196

    # Verify split_metadata.json
    with open(splits_dir / "split_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["random_seed"] == 42
    assert meta["total_clusters"] == 196
    assert meta["total_records"] == 1176
    assert meta["splits"]["train"]["clusters"] == 137
    assert meta["splits"]["train"]["records"] == 822
    assert meta["splits"]["val"]["clusters"] == 29
    assert meta["splits"]["val"]["records"] == 174
    assert meta["splits"]["test"]["clusters"] == 30
    assert meta["splits"]["test"]["records"] == 180
    assert meta["leakage_verification"]["leakage_free"] is True
