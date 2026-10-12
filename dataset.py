"""Dataset schema, validation, and cluster-based splitting for prompt-sensitivity-hybrid.

Guarantees zero train/validation/test leakage across prompt clusters and
manages conversion of wide axis-tagged prompt CSV datasets into canonical schema.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

AXIS_COLUMNS: List[str] = ["lexical", "syntactic", "pragmatic", "length", "format"]
EXPECTED_COLUMNS: List[str] = ["id", "original"] + AXIS_COLUMNS


@dataclass
class PromptRecord:
    """Canonical schema for an individual prompt record in prompt-sensitivity-hybrid."""
    cluster_id: str
    prompt_id: str
    prompt_text: str
    variant_type: str = "original"
    reference_answer: Optional[str] = None
    perturbation_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        """Validate integrity and schema constraints of the prompt record."""
        if not isinstance(self.cluster_id, str) or not self.cluster_id.strip():
            raise ValueError(f"cluster_id must be a non-empty string, got: {self.cluster_id!r}")
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError(f"prompt_id must be a non-empty string, got: {self.prompt_id!r}")
        if not isinstance(self.prompt_text, str) or not self.prompt_text.strip():
            raise ValueError(f"prompt_text must be a non-empty string, got: {self.prompt_text!r}")
        if not isinstance(self.variant_type, str) or not self.variant_type.strip():
            raise ValueError(f"variant_type must be a non-empty string, got: {self.variant_type!r}")
        if self.reference_answer is not None and not isinstance(self.reference_answer, str):
            raise ValueError(f"reference_answer must be None or string, got: {type(self.reference_answer)}")
        if not isinstance(self.perturbation_metadata, dict):
            raise ValueError(f"perturbation_metadata must be a dictionary, got: {type(self.perturbation_metadata)}")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be None or a dictionary, got: {type(self.metadata)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to canonical dictionary representation."""
        return {
            "cluster_id": self.cluster_id,
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "variant_type": self.variant_type,
            "reference_answer": self.reference_answer,
            "perturbation_metadata": self.perturbation_metadata,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptRecord:
        """Create PromptRecord from dictionary with schema validation."""
        required_fields = ["cluster_id", "prompt_id", "prompt_text"]
        for rf in required_fields:
            if rf not in data:
                raise ValueError(f"Missing required field in prompt record: '{rf}'")

        variant_type = data.get("variant_type")
        if variant_type is None and "perturbation_metadata" in data and "axis" in data["perturbation_metadata"]:
            variant_type = data["perturbation_metadata"]["axis"]
        elif variant_type is None:
            variant_type = "original"

        record = cls(
            cluster_id=str(data["cluster_id"]),
            prompt_id=str(data["prompt_id"]),
            prompt_text=str(data["prompt_text"]),
            variant_type=str(variant_type),
            reference_answer=data.get("reference_answer"),
            perturbation_metadata=dict(data.get("perturbation_metadata", {})),
            metadata=data.get("metadata"),
        )
        record.validate()
        return record


class PromptDataset:
    """In-memory collection of PromptRecord entries with validation and split support."""

    def __init__(self, records: Optional[List[PromptRecord]] = None):
        self.records: List[PromptRecord] = records or []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PromptRecord:
        return self.records[idx]

    def __iter__(self):
        return iter(self.records)

    def append(self, record: PromptRecord) -> None:
        record.validate()
        self.records.append(record)

    def validate(self) -> Dict[str, Any]:
        """Validate entire dataset, checking individual records and uniqueness of prompt_id."""
        if not self.records:
            raise ValueError("Dataset is empty.")

        seen_prompt_ids: Set[str] = set()
        clusters: Set[str] = set()
        cluster_counts: Dict[str, int] = {}
        variant_counts: Dict[str, int] = {}

        for idx, rec in enumerate(self.records):
            try:
                rec.validate()
            except ValueError as e:
                raise ValueError(
                    f"Invalid record at index {idx} (prompt_id={getattr(rec, 'prompt_id', None)}): {e}"
                ) from e

            if rec.prompt_id in seen_prompt_ids:
                raise ValueError(f"Duplicate prompt_id detected: '{rec.prompt_id}' at index {idx}")
            seen_prompt_ids.add(rec.prompt_id)

            clusters.add(rec.cluster_id)
            cluster_counts[rec.cluster_id] = cluster_counts.get(rec.cluster_id, 0) + 1
            variant_counts[rec.variant_type] = variant_counts.get(rec.variant_type, 0) + 1

        return {
            "total_records": len(self.records),
            "unique_clusters": len(clusters),
            "unique_prompts": len(seen_prompt_ids),
            "cluster_counts": cluster_counts,
            "variant_counts": variant_counts,
        }

    def get_clusters(self) -> Set[str]:
        """Return the set of unique cluster_ids present in the dataset."""
        return {r.cluster_id for r in self.records}

    def save_jsonl(self, filepath: str | Path) -> None:
        """Serialize dataset records to a JSONL file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, filepath: str | Path) -> PromptDataset:
        """Load and validate dataset from a JSONL file."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        records: List[PromptRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    rec = PromptRecord.from_dict(data)
                    records.append(rec)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSONL line {line_no} from {path}: {e}") from e

        dataset = cls(records=records)
        dataset.validate()
        return dataset

    @classmethod
    def from_axis_tagged_csv(
        cls,
        filepath: str | Path,
        validate_expected_counts: bool = False,
        expected_clusters: int = 196,
        expected_records: int = 1176,
    ) -> PromptDataset:
        """Parse axis-tagged prompt variations CSV into a canonical PromptDataset.

        Each unique 'id' becomes one semantic question cluster.
        'original' is mapped as variant_type='original'.
        The other 5 columns ('lexical', 'syntactic', 'pragmatic', 'length', 'format')
        are mapped as its linguistic variants.
        All reference_answers are set strictly to None (null).
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Source CSV file not found: {path}")

        records: List[PromptRecord] = []

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in (reader.fieldnames or [])]

            for expected_col in EXPECTED_COLUMNS:
                if expected_col not in headers:
                    raise ValueError(
                        f"CSV is missing expected column '{expected_col}'. Found: {headers}"
                    )

            for row_no, row in enumerate(reader, start=2):
                cluster_raw = row.get("id")
                if cluster_raw is None or not str(cluster_raw).strip():
                    raise ValueError(f"Missing or empty 'id' in CSV at row {row_no}")
                cluster_id = str(cluster_raw).strip()

                original_text = row.get("original")
                if original_text is None or not str(original_text).strip():
                    raise ValueError(f"Missing or empty 'original' prompt for cluster '{cluster_id}' at row {row_no}")
                original_text = str(original_text).strip()

                # Add canonical original prompt record
                records.append(
                    PromptRecord(
                        cluster_id=cluster_id,
                        prompt_id=f"{cluster_id}_original",
                        prompt_text=original_text,
                        variant_type="original",
                        reference_answer=None,
                        perturbation_metadata={"axis": "original", "is_variant": False},
                    )
                )

                # Add the 5 linguistic variant records
                for axis in AXIS_COLUMNS:
                    variant_text = row.get(axis)
                    if variant_text is None or not str(variant_text).strip():
                        raise ValueError(
                            f"Missing or empty variant '{axis}' for cluster '{cluster_id}' at row {row_no}"
                        )
                    records.append(
                        PromptRecord(
                            cluster_id=cluster_id,
                            prompt_id=f"{cluster_id}_{axis}",
                            prompt_text=str(variant_text).strip(),
                            variant_type=axis,
                            reference_answer=None,
                            perturbation_metadata={
                                "axis": axis,
                                "is_variant": True,
                                "original_prompt": original_text,
                            },
                        )
                    )

        dataset = cls(records=records)
        stats = dataset.validate()

        if validate_expected_counts:
            if stats["unique_clusters"] != expected_clusters:
                raise ValueError(
                    f"Expected {expected_clusters} unique clusters, found {stats['unique_clusters']}"
                )
            if stats["total_records"] != expected_records:
                raise ValueError(
                    f"Expected {expected_records} prompt records, found {stats['total_records']}"
                )

        return dataset


def cluster_train_val_test_split(
    dataset: PromptDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[PromptDataset, PromptDataset, PromptDataset]:
    """Split a dataset into train, validation, and test subsets partitioned strictly by cluster_id.

    This ensures that all 6 prompt versions of a semantic question remain together
    in the same split, eliminating data leakage across partitions.
    """
    dataset.validate()

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total_ratio})"
        )
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative, and train_ratio must be strictly positive.")

    # Group records by cluster_id
    cluster_to_records: Dict[str, List[PromptRecord]] = {}
    for r in dataset.records:
        cluster_to_records.setdefault(r.cluster_id, []).append(r)

    # Sort deterministically by natural or numerical order before shuffling
    unique_clusters = sorted(
        cluster_to_records.keys(),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    num_clusters = len(unique_clusters)

    if num_clusters < 3:
        raise ValueError(
            f"At least 3 distinct cluster_ids are required to create train/val/test splits without leakage, found {num_clusters}."
        )

    # Deterministically shuffle clusters using the provided seed
    rng = random.Random(seed)
    shuffled_clusters = list(unique_clusters)
    rng.shuffle(shuffled_clusters)

    # Compute cluster counts per partition
    n_train = max(1, int(round(num_clusters * train_ratio)))
    n_val = max(1, int(round(num_clusters * val_ratio)))
    if n_train + n_val >= num_clusters:
        n_train = num_clusters - 2
        n_val = 1
    n_test = num_clusters - (n_train + n_val)
    if n_test <= 0 and test_ratio > 0:
        if n_train > 1:
            n_train -= 1
            n_test = 1
        elif n_val > 1:
            n_val -= 1
            n_test = 1

    train_clusters = set(shuffled_clusters[:n_train])
    val_clusters = set(shuffled_clusters[n_train : n_train + n_val])
    test_clusters = set(shuffled_clusters[n_train + n_val :])

    # Strict leakage verification
    leakage_check = verify_split_leakage(train_clusters, val_clusters, test_clusters)
    if not leakage_check["leakage_free"]:
        raise RuntimeError(f"Split leakage detected during partition: {leakage_check}")

    train_records = [r for c in shuffled_clusters[:n_train] for r in cluster_to_records[c]]
    val_records = [r for c in shuffled_clusters[n_train : n_train + n_val] for r in cluster_to_records[c]]
    test_records = [r for c in shuffled_clusters[n_train + n_val :] for r in cluster_to_records[c]]

    return PromptDataset(train_records), PromptDataset(val_records), PromptDataset(test_records)


def verify_split_leakage(
    train_clusters: Set[str] | PromptDataset,
    val_clusters: Set[str] | PromptDataset,
    test_clusters: Set[str] | PromptDataset,
) -> Dict[str, Any]:
    """Verify that there is zero cluster overlap between train, val, and test splits."""
    def extract_clusters(source: Set[str] | PromptDataset) -> Set[str]:
        if isinstance(source, PromptDataset):
            return source.get_clusters()
        return set(source)

    train_set = extract_clusters(train_clusters)
    val_set = extract_clusters(val_clusters)
    test_set = extract_clusters(test_clusters)

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    is_clean = (len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0)

    return {
        "leakage_free": is_clean,
        "train_cluster_count": len(train_set),
        "val_cluster_count": len(val_set),
        "test_cluster_count": len(test_set),
        "train_val_overlap": sorted(train_val_overlap, key=lambda x: int(x) if x.isdigit() else x),
        "train_test_overlap": sorted(train_test_overlap, key=lambda x: int(x) if x.isdigit() else x),
        "val_test_overlap": sorted(val_test_overlap, key=lambda x: int(x) if x.isdigit() else x),
    }


def save_dataset_and_splits(
    canonical_dataset: PromptDataset,
    train_dataset: PromptDataset,
    val_dataset: PromptDataset,
    test_dataset: PromptDataset,
    splits_dir: str | Path = "data/splits",
    seed: int = 42,
    source_file: str = "data/paraphrased_prompts_axis_tagged_qa.csv",
) -> Dict[str, Any]:
    """Save canonical dataset and split files under data/splits/ alongside split metadata."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save canonical full dataset
    canonical_file = out_dir / "canonical_dataset.jsonl"
    canonical_dataset.save_jsonl(canonical_file)

    # 2. Save individual split files
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    test_file = out_dir / "test.jsonl"

    train_dataset.save_jsonl(train_file)
    val_dataset.save_jsonl(val_file)
    test_dataset.save_jsonl(test_file)

    # 3. Verify zero leakage
    leakage = verify_split_leakage(train_dataset, val_dataset, test_dataset)
    if not leakage["leakage_free"]:
        raise RuntimeError(f"Split leakage detected while saving splits: {leakage}")

    # 4. Save metadata JSON
    metadata = {
        "random_seed": seed,
        "source_file": str(source_file),
        "splits_dir": str(out_dir),
        "total_clusters": len(canonical_dataset.get_clusters()),
        "total_records": len(canonical_dataset),
        "splits": {
            "train": {
                "clusters": leakage["train_cluster_count"],
                "records": len(train_dataset),
                "cluster_ids": sorted(list(train_dataset.get_clusters()), key=lambda x: int(x) if x.isdigit() else x),
            },
            "val": {
                "clusters": leakage["val_cluster_count"],
                "records": len(val_dataset),
                "cluster_ids": sorted(list(val_dataset.get_clusters()), key=lambda x: int(x) if x.isdigit() else x),
            },
            "test": {
                "clusters": leakage["test_cluster_count"],
                "records": len(test_dataset),
                "cluster_ids": sorted(list(test_dataset.get_clusters()), key=lambda x: int(x) if x.isdigit() else x),
            },
        },
        "leakage_verification": leakage,
    }

    meta_file = out_dir / "split_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata
