"""External independent evaluation dataset schema, integrity checks, and overlap audit.

Provides loader and verification tools for out-of-distribution generalization testing.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("prompt_sensitivity.dataset_external")


@dataclass
class ExternalPromptRecord:
    cluster_id: str
    prompt_id: str
    prompt_text: str
    variant_type: str
    gold_reference_answer: Optional[str] = None
    is_factual_subset: bool = False
    domain: str = "general"
    creation_method: str = "curated_paraphrase"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_prompt_text(text: str) -> str:
    """Standardize prompt text for duplicate and overlap detection."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def compute_token_jaccard(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two texts."""
    tokens_a = set(normalize_prompt_text(text_a).split())
    tokens_b = set(normalize_prompt_text(text_b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def compute_char_3gram_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity of character 3-grams."""
    norm_a = normalize_prompt_text(text_a)
    norm_b = normalize_prompt_text(text_b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0

    def ngrams(s: str) -> Counter:
        if len(s) < 3:
            return Counter([s])
        return Counter([s[i:i+3] for i in range(len(s) - 2)])

    c_a = ngrams(norm_a)
    c_b = ngrams(norm_b)
    all_keys = set(c_a.keys()) | set(c_b.keys())

    v_a = np.array([c_a[k] for k in all_keys], dtype=np.float32)
    v_b = np.array([c_b[k] for k in all_keys], dtype=np.float32)
    norm = np.linalg.norm(v_a) * np.linalg.norm(v_b)
    if norm == 0:
        return 0.0
    return float(np.dot(v_a, v_b) / norm)


def audit_dataset_overlap(
    existing_source_path: str | Path,
    external_records: List[ExternalPromptRecord],
    jaccard_threshold: float = 0.60,
    ngram_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Audit overlap between existing 196-cluster dataset and new external dataset.

    Returns:
        Audit report containing total comparisons, exact matches, near-duplicate matches,
        and overlap percentage.
    """
    existing_prompts: List[Dict[str, str]] = []
    source = Path(existing_source_path)

    if source.suffix.lower() == ".csv":
        with open(source, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            variant_cols = ["original", "lexical", "syntactic", "pragmatic", "length", "format"]
            for row in reader:
                cid = row["id"]
                for vcol in variant_cols:
                    if vcol in row and row[vcol].strip():
                        existing_prompts.append({
                            "cluster_id": cid,
                            "variant_type": vcol,
                            "prompt_text": row[vcol].strip(),
                        })
    elif source.suffix.lower() == ".jsonl":
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_prompts.append({
                        "cluster_id": item["cluster_id"],
                        "variant_type": item["variant_type"],
                        "prompt_text": item["prompt_text"],
                    })

    logger.info(f"Loaded {len(existing_prompts)} existing prompt records for overlap audit.")

    exact_matches: List[Dict[str, Any]] = []
    near_matches: List[Dict[str, Any]] = []
    total_comparisons = 0

    existing_norm_map = {
        normalize_prompt_text(p["prompt_text"]): p for p in existing_prompts
    }

    for ext_rec in external_records:
        norm_ext = normalize_prompt_text(ext_rec.prompt_text)

        # 1. Exact normalized match check
        if norm_ext in existing_norm_map:
            match = existing_norm_map[norm_ext]
            exact_matches.append({
                "external_prompt_id": ext_rec.prompt_id,
                "external_cluster_id": ext_rec.cluster_id,
                "existing_cluster_id": match["cluster_id"],
                "existing_prompt_text": match["prompt_text"],
                "external_prompt_text": ext_rec.prompt_text,
            })

        # 2. Near-duplicate check against existing original prompts
        for ex in existing_prompts:
            if ex["variant_type"] == "original":
                total_comparisons += 1
                jaccard = compute_token_jaccard(ext_rec.prompt_text, ex["prompt_text"])
                if jaccard > jaccard_threshold:
                    ngram_sim = compute_char_3gram_similarity(ext_rec.prompt_text, ex["prompt_text"])
                    if ngram_sim > ngram_threshold:
                        near_matches.append({
                            "external_prompt_id": ext_rec.prompt_id,
                            "external_cluster_id": ext_rec.cluster_id,
                            "existing_cluster_id": ex["cluster_id"],
                            "jaccard_similarity": round(jaccard, 3),
                            "ngram_similarity": round(ngram_sim, 3),
                            "external_text": ext_rec.prompt_text,
                            "existing_text": ex["prompt_text"],
                        })

    report = {
        "existing_prompt_count": len(existing_prompts),
        "external_prompt_count": len(external_records),
        "total_comparisons_checked": total_comparisons,
        "exact_matches_count": len(exact_matches),
        "near_matches_count": len(near_matches),
        "exact_overlap_zero": bool(len(exact_matches) == 0),
        "near_overlap_zero": bool(len(near_matches) == 0),
        "exact_matches": exact_matches,
        "near_matches": near_matches,
    }

    logger.info(
        f"Dataset overlap audit result: exact_matches={len(exact_matches)}, "
        f"near_matches={len(near_matches)}"
    )
    return report


def load_external_records(filepath: str | Path) -> List[ExternalPromptRecord]:
    """Load ExternalPromptRecord items from a JSONL file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"External dataset file not found: {path}")

    records: List[ExternalPromptRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                records.append(ExternalPromptRecord(
                    cluster_id=data["cluster_id"],
                    prompt_id=data["prompt_id"],
                    prompt_text=data["prompt_text"],
                    variant_type=data["variant_type"],
                    gold_reference_answer=data.get("gold_reference_answer"),
                    is_factual_subset=data.get("is_factual_subset", False),
                    domain=data.get("domain", "general"),
                    creation_method=data.get("creation_method", "curated_paraphrase"),
                ))
    return records

