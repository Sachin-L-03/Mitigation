"""Cluster-aware dataset management, negative pair construction, and leakage prevention.

Supports:
  - Variable numbers of equivalent paraphrases per cluster
  - Explicit equivalence statuses (VERIFIED, PROBABLE, AMBIGUOUS, NON_EQUIVALENT)
  - Hard negative pair generation across semantic constraints (quantities, entities, negations, units, scopes)
  - Strict cluster-level split separation and cross-split leakage auditing
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("prompt_sensitivity.mitigation.dataset")


class EquivalenceStatus(str, Enum):
    VERIFIED = "verified"
    PROBABLE = "probable"
    AMBIGUOUS = "ambiguous"
    NON_EQUIVALENT = "non_equivalent"


@dataclass
class PromptVariant:
    prompt_id: str
    text: str
    variant_type: str
    status: EquivalenceStatus = EquivalenceStatus.VERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NegativeVariant:
    negative_id: str
    text: str
    contrast_type: str  # e.g., 'quantity', 'entity', 'negation', 'unit', 'detail'
    target_difference: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCluster:
    cluster_id: str
    canonical_prompt: str
    variants: List[PromptVariant] = field(default_factory=list)
    negatives: List[NegativeVariant] = field(default_factory=list)
    gold_reference_answer: Optional[str] = None
    task_category: str = "general_qa"
    domain: str = "general"
    split: str = "train"
    provenance: str = "PromptSensitivity"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "canonical_prompt": self.canonical_prompt,
            "variants": [
                {
                    "prompt_id": v.prompt_id,
                    "text": v.text,
                    "variant_type": v.variant_type,
                    "status": v.status.value,
                    "metadata": v.metadata,
                }
                for v in self.variants
            ],
            "negatives": [
                {
                    "negative_id": n.negative_id,
                    "text": n.text,
                    "contrast_type": n.contrast_type,
                    "target_difference": n.target_difference,
                    "metadata": n.metadata,
                }
                for n in self.negatives
            ],
            "gold_reference_answer": self.gold_reference_answer,
            "task_category": self.task_category,
            "domain": self.domain,
            "split": self.split,
            "provenance": self.provenance,
        }


def generate_hard_negatives(canonical_text: str, cluster_id: str) -> List[NegativeVariant]:
    """Generate meaningful negative variants that alter core task constraints.

    Contrast types:
      - 'negation': Inverts requirement (e.g., 'Which of the following is NOT...')
      - 'quantity': Changes numerical constraint (e.g., 'List two...' vs 'List five...')
      - 'scope': Narrows or widens target (e.g., 'specifically in pediatric patients')
    """
    negatives: List[NegativeVariant] = []

    # 1. Negation contrast
    if not re.search(r"\b(not|never|except|unlikely)\b", canonical_text, re.IGNORECASE):
        # Create negative prompt asking for the opposite or what is NOT true
        neg_text = f"Which of the following is NOT true regarding: {canonical_text.rstrip('?')}"
        negatives.append(NegativeVariant(
            negative_id=f"{cluster_id}_neg_negation",
            text=neg_text,
            contrast_type="negation",
            target_difference="Inverts the query to identify exceptions or false statements",
        ))

    # 2. Quantity contrast
    q_text = f"List exactly three distinct examples or causes for: {canonical_text.rstrip('?')}"
    negatives.append(NegativeVariant(
        negative_id=f"{cluster_id}_neg_quantity",
        text=q_text,
        contrast_type="quantity",
        target_difference="Imposes an explicit 3-item numerical constraint",
    ))

    # 3. Scope contrast
    scope_text = f"Explain the historical context prior to 1900 for: {canonical_text.rstrip('?')}"
    negatives.append(NegativeVariant(
        negative_id=f"{cluster_id}_neg_scope",
        text=scope_text,
        contrast_type="scope",
        target_difference="Restricts temporal scope to pre-1900 historical context",
    ))

    return negatives


def validate_prompt_equivalence(canonical_text: str, variant_text: str) -> EquivalenceStatus:
    """Automated heuristic validation of semantic equivalence."""
    if not variant_text or not canonical_text:
        return EquivalenceStatus.NON_EQUIVALENT

    norm_c = canonical_text.lower()
    norm_v = variant_text.lower()

    if norm_c == norm_v:
        return EquivalenceStatus.VERIFIED

    # Check for accidental negation introduced
    has_neg_c = bool(re.search(r"\b(not|never|neither|nor|except)\b", norm_c))
    has_neg_v = bool(re.search(r"\b(not|never|neither|nor|except)\b", norm_v))
    if has_neg_c != has_neg_v:
        return EquivalenceStatus.NON_EQUIVALENT

    # Ratio of token length disparity
    len_c = len(norm_c.split())
    len_v = len(norm_v.split())
    if len_v > 4 * len_c or len_c > 4 * len_v:
        return EquivalenceStatus.AMBIGUOUS

    return EquivalenceStatus.VERIFIED


class ClusterDataset:
    """Manages clusters, verified variants, negative pairs, and split integrity."""

    def __init__(self, clusters: Optional[List[SemanticCluster]] = None):
        self.clusters: List[SemanticCluster] = clusters or []
        self._cluster_map: Dict[str, SemanticCluster] = {c.cluster_id: c for c in self.clusters}

    def __len__(self) -> int:
        return len(self.clusters)

    def __iter__(self):
        return iter(self.clusters)

    def get_cluster(self, cluster_id: str) -> Optional[SemanticCluster]:
        return self._cluster_map.get(cluster_id)

    @classmethod
    def from_csv(cls, csv_path: str | Path, split_assignments: Optional[Dict[str, str]] = None) -> "ClusterDataset":
        """Load from original wide PromptSensitivity CSV format."""
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Source CSV not found: {path}")

        variant_cols = ["lexical", "syntactic", "pragmatic", "length", "format"]
        clusters: List[SemanticCluster] = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["id"].strip()
                canonical = row["original"].strip()
                split = split_assignments.get(cid, "train") if split_assignments else "train"

                variants: List[PromptVariant] = [
                    PromptVariant(
                        prompt_id=f"{cid}_original",
                        text=canonical,
                        variant_type="original",
                        status=EquivalenceStatus.VERIFIED,
                    )
                ]

                for col in variant_cols:
                    if col in row and row[col].strip():
                        v_text = row[col].strip()
                        status = validate_prompt_equivalence(canonical, v_text)
                        variants.append(PromptVariant(
                            prompt_id=f"{cid}_{col}",
                            text=v_text,
                            variant_type=col,
                            status=status,
                        ))

                negatives = generate_hard_negatives(canonical, cid)

                cluster = SemanticCluster(
                    cluster_id=cid,
                    canonical_prompt=canonical,
                    variants=variants,
                    negatives=negatives,
                    gold_reference_answer=None,
                    split=split,
                    provenance="paraphrased_prompts_axis_tagged_qa.csv",
                )
                clusters.append(cluster)

        logger.info(f"Loaded {len(clusters)} clusters from CSV {path}")
        return cls(clusters)

    @classmethod
    def from_jsonl(cls, jsonl_path: str | Path) -> "ClusterDataset":
        """Load clusters from canonical jsonl."""
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset JSONL not found: {path}")

        clusters: List[SemanticCluster] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    variants = [
                        PromptVariant(
                            prompt_id=v["prompt_id"],
                            text=v["text"],
                            variant_type=v["variant_type"],
                            status=EquivalenceStatus(v.get("status", "verified")),
                            metadata=v.get("metadata", {}),
                        )
                        for v in d.get("variants", [])
                    ]
                    negatives = [
                        NegativeVariant(
                            negative_id=n["negative_id"],
                            text=n["text"],
                            contrast_type=n["contrast_type"],
                            target_difference=n["target_difference"],
                            metadata=n.get("metadata", {}),
                        )
                        for n in d.get("negatives", [])
                    ]
                    clusters.append(SemanticCluster(
                        cluster_id=d["cluster_id"],
                        canonical_prompt=d["canonical_prompt"],
                        variants=variants,
                        negatives=negatives,
                        gold_reference_answer=d.get("gold_reference_answer"),
                        task_category=d.get("task_category", "general_qa"),
                        domain=d.get("domain", "general"),
                        split=d.get("split", "train"),
                        provenance=d.get("provenance", "jsonl"),
                    ))
        return cls(clusters)

    def save_jsonl(self, output_path: str | Path) -> None:
        """Export dataset to JSONL."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for c in self.clusters:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.clusters)} clusters to {out}")

    def filter_by_split(self, split_name: str) -> "ClusterDataset":
        filtered = [c for c in self.clusters if c.split == split_name]
        return ClusterDataset(filtered)

    def get_contrastive_pairs(self) -> List[Dict[str, Any]]:
        """Construct positive pairs and hard negative pairs for contrastive learning.

        Only positive pairs with status == VERIFIED are included.
        """
        pairs = []
        for c in self.clusters:
            verified_vars = [v for v in c.variants if v.status == EquivalenceStatus.VERIFIED]
            # Construct positive pairs between all verified variants within the cluster
            for i in range(len(verified_vars)):
                for j in range(i + 1, len(verified_vars)):
                    pairs.append({
                        "cluster_id": c.cluster_id,
                        "text_a": verified_vars[i].text,
                        "text_b": verified_vars[j].text,
                        "label": 1,
                        "type": "positive",
                    })

            # Construct negative pairs against hard negatives
            for v in verified_vars:
                for neg in c.negatives:
                    pairs.append({
                        "cluster_id": c.cluster_id,
                        "text_a": v.text,
                        "text_b": neg.text,
                        "label": 0,
                        "type": f"hard_negative_{neg.contrast_type}",
                    })
        return pairs


def audit_splits_leakage(train_ds: ClusterDataset, val_ds: ClusterDataset, test_ds: ClusterDataset) -> Dict[str, Any]:
    """Audit for cross-split cluster or text leakage."""
    train_cids = {c.cluster_id for c in train_ds}
    val_cids = {c.cluster_id for c in val_ds}
    test_cids = {c.cluster_id for c in test_ds}

    overlap_train_val = train_cids & val_cids
    overlap_train_test = train_cids & test_cids
    overlap_val_test = val_cids & test_cids

    def get_texts(ds: ClusterDataset) -> Set[str]:
        texts = set()
        for c in ds:
            texts.add(c.canonical_prompt.lower().strip())
            for v in c.variants:
                texts.add(v.text.lower().strip())
        return texts

    train_texts = get_texts(train_ds)
    val_texts = get_texts(val_ds)
    test_texts = get_texts(test_ds)

    text_leak_train_val = train_texts & val_texts
    text_leak_train_test = train_texts & test_texts
    text_leak_val_test = val_texts & test_texts

    def hash_cids(cids: Set[str]) -> str:
        s = ",".join(sorted(cids))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    report = {
        "train_clusters": len(train_cids),
        "val_clusters": len(val_cids),
        "test_clusters": len(test_cids),
        "cluster_overlap_train_val": len(overlap_train_val),
        "cluster_overlap_train_test": len(overlap_train_test),
        "cluster_overlap_val_test": len(overlap_val_test),
        "text_leak_train_val": len(text_leak_train_val),
        "text_leak_train_test": len(text_leak_train_test),
        "text_leak_val_test": len(text_leak_val_test),
        "zero_leakage_verified": bool(
            len(overlap_train_val) == 0
            and len(overlap_train_test) == 0
            and len(overlap_val_test) == 0
            and len(text_leak_train_val) == 0
            and len(text_leak_train_test) == 0
            and len(text_leak_val_test) == 0
        ),
        "split_hashes": {
            "train": hash_cids(train_cids),
            "val": hash_cids(val_cids),
            "test": hash_cids(test_cids),
        },
    }
    return report
