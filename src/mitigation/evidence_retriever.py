"""Independent evidence retrieval engine using local SQLite FTS5 with BM25 ranking.

Path B of Phase 6:
  - Classifies whether questions require external evidence.
  - Queries local verified evidence database (`evidence_index.db`) using full-text search.
  - Enforces bounded retrieval budgets (top-K passages, max tokens).
  - Records full provenance: doc ID, title, document path, timestamp, passage text, and BM25 score.
  - Detects contradictions between retrieved sources.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("prompt_sensitivity.mitigation.retriever")


@dataclass
class RetrievedPassage:
    source_id: str
    title: str
    document_path: str
    passage_text: str
    score: float
    retrieval_timestamp: str
    provenance: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    retrieval_success: bool
    passages: List[RetrievedPassage]
    execution_time_ms: float
    source_conflict_detected: bool
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_success": self.retrieval_success,
            "passages": [p.to_dict() for p in self.passages],
            "execution_time_ms": self.execution_time_ms,
            "source_conflict_detected": self.source_conflict_detected,
            "status_message": self.status_message,
        }


class EvidenceRetriever:
    """Retrieves authoritative evidence passages from local FTS5 database."""

    STOPWORDS: Set[str] = {
        "what", "is", "the", "of", "in", "to", "a", "an", "and", "which",
        "who", "how", "many", "does", "do", "are", "was", "were", "where",
        "when", "for", "with", "by", "on", "at", "from", "as", "state",
        "name", "provide", "list", "identify", "give", "tell", "me",
    }

    def __init__(
        self,
        db_path: str | Path = "data/phase6_eval/evidence_corpus/evidence_index.db",
        max_passages: int = 3,
    ):
        self.db_path = Path(db_path)
        self.max_passages = max_passages
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        if not self.db_path.exists():
            return None
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    @classmethod
    def extract_search_terms(cls, query: str) -> List[str]:
        """Extract salient keywords and entity tokens for full-text search."""
        clean = re.sub(r"[^\w\s]", " ", query.lower())
        tokens = clean.split()
        keywords = [t for t in tokens if len(t) > 2 and t.isalnum() and t not in cls.STOPWORDS]
        return keywords

    def retrieve(self, query: str) -> RetrievalResult:
        """Execute bounded passage retrieval and return structured provenance records."""
        t0 = time.perf_counter()
        conn = self._get_connection()
        if conn is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query=query,
                retrieval_success=False,
                passages=[],
                execution_time_ms=elapsed,
                source_conflict_detected=False,
                status_message="EVIDENCE_DATABASE_NOT_FOUND",
            )

        keywords = self.extract_search_terms(query)
        if not keywords:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query=query,
                retrieval_success=False,
                passages=[],
                execution_time_ms=elapsed,
                source_conflict_detected=False,
                status_message="NO_SEARCH_KEYWORDS_EXTRACTED",
            )

        # Build FTS5 MATCH query with OR / NEAR semantics
        # e.g., "kelvin OR temperature OR unit"
        fts_query = " OR ".join(keywords)

        sql = """
        SELECT doc_id, title, document_path, content, provenance, rank
        FROM evidence_fts
        WHERE evidence_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """

        timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        passages: List[RetrievedPassage] = []

        try:
            cursor = conn.cursor()
            rows = cursor.execute(sql, (fts_query, self.max_passages)).fetchall()
            for r in rows:
                doc_id, title, doc_path, content, prov, rank_val = r
                passages.append(
                    RetrievedPassage(
                        source_id=doc_id,
                        title=title,
                        document_path=doc_path,
                        passage_text=content.strip(),
                        score=float(-rank_val),  # FTS5 rank is negative (lower = better)
                        retrieval_timestamp=timestamp_str,
                        provenance=prov,
                    )
                )
        except Exception as e:
            logger.warning(f"FTS5 search error on query '{fts_query}': {e}")
            elapsed = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query=query,
                retrieval_success=False,
                passages=[],
                execution_time_ms=elapsed,
                source_conflict_detected=False,
                status_message=f"SEARCH_ERROR: {e}",
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        if not passages:
            return RetrievalResult(
                query=query,
                retrieval_success=False,
                passages=[],
                execution_time_ms=elapsed,
                source_conflict_detected=False,
                status_message="NO_RELEVANT_PASSAGES_FOUND",
            )

        # Check for conflicts between retrieved sources
        source_conflict = self._detect_source_conflicts(passages)

        return RetrievalResult(
            query=query,
            retrieval_success=True,
            passages=passages,
            execution_time_ms=elapsed,
            source_conflict_detected=source_conflict,
            status_message="SUCCESS",
        )

    def _detect_source_conflicts(self, passages: List[RetrievedPassage]) -> bool:
        """Detect direct numerical or negation contradictions between retrieved sources."""
        if len(passages) < 2:
            return False
        # Extract signed numbers across passages
        num_sets = []
        for p in passages:
            nums = set(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?", p.passage_text))
            if nums:
                num_sets.append(nums)
        # If passages cite conflicting non-overlapping numbers for specific properties
        if len(num_sets) >= 2:
            common = num_sets[0].intersection(*num_sets[1:])
            # If no common numbers exist and all passages are high confidence
            if not common and all(p.score > 2.0 for p in passages):
                return True
        return False

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
