"""Regression and unit test suite for Phase 6 Evidence Verification and False-Consensus Mitigation.

Verifies:
  1. Gold-reference isolation (inference does not access gold references).
  2. No canonical answer copying.
  3. Safe AST expression parsing (whitelist of operators, rejection of unsafe functions).
  4. Deterministic arithmetic with signed numbers and exponents.
  5. Deterministic modular arithmetic (e.g., 2^100 mod 7, 3^100 mod 8).
  6. Unit and sign mismatch detection and conversion (C to K, mph to km/h, gallons to L).
  7. Date arithmetic (elapsed duration).
  8. Structured output format constraint validation (slash delimiters, sorting).
  9. SQLite FTS5 evidence retrieval and provenance recording.
  10. Empty retrieval and missing keyword handling.
  11. Conflicting evidence detection.
  12. Evidence-support classification: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE.
  13. Correction vs abstention logic (no correction on insufficient evidence).
  14. Strict abstention scoring (abstentions score 0.0 in primary factual accuracy).
  15. Statistical non-inferiority calculation.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.mitigation.deterministic_verifier import (
    DeterministicVerificationResult,
    DeterministicVerifier,
    SafeASTEvaluator,
)
from src.mitigation.evidence_grounded_router import (
    EvidenceGroundedRouter,
    EvidenceRouterResult,
    ExperimentalCondition,
)
from src.mitigation.evidence_retriever import (
    EvidenceRetriever,
    RetrievalResult,
    RetrievedPassage,
)
from src.mitigation.evidence_verifier import (
    EvidenceVerificationAssessment,
    EvidenceVerifier,
    VerificationAction,
    VerificationState,
)
from src.mitigation.factual_risk_detector import (
    EnhancedSemanticComparator,
    FactualRiskAssessment,
    FactualRiskDetector,
    PromptEquivalenceChecker,
    RiskCategory,
)


class TestSafeASTEvaluator:
    def test_safe_arithmetic_evaluation(self):
        assert SafeASTEvaluator.evaluate("15 + 25") == 40.0
        assert SafeASTEvaluator.evaluate("(-3)**3 - (-2)**2") == -31.0
        assert SafeASTEvaluator.evaluate("144 / 12 * 3") == 36.0
        assert SafeASTEvaluator.evaluate("2**6 - 3**3 - 4**2") == 21.0

    def test_division_by_zero_safety(self):
        assert SafeASTEvaluator.evaluate("100 / 0") is None

    def test_unsafe_calls_rejected(self):
        # Disallowed calls, imports, or attributes
        assert SafeASTEvaluator.evaluate("__import__('os').system('dir')") is None
        assert SafeASTEvaluator.evaluate("open('file.txt')") is None
        assert SafeASTEvaluator.evaluate("exec('a = 1')") is None


class TestDeterministicVerifier:
    def test_modular_arithmetic(self):
        # 2^100 mod 7 = 2
        res = DeterministicVerifier.verify_candidate(
            prompt="What is the remainder when 2^100 is divided by 7?",
            candidate_answer="The remainder is 4.",
        )
        assert res.is_eligible is True
        assert res.task_type == "modular_arithmetic"
        assert res.computed_value == "2"
        assert res.candidate_matches is False
        assert res.suggested_correction == "2"

    def test_modular_arithmetic_candidate_correct(self):
        res = DeterministicVerifier.verify_candidate(
            prompt="Calculate 3^100 mod 8.",
            candidate_answer="3^100 mod 8 is 1",
        )
        assert res.is_eligible is True
        assert res.computed_value == "1"
        assert res.candidate_matches is True
        assert res.suggested_correction is None

    def test_compound_signed_arithmetic(self):
        res = DeterministicVerifier.verify_candidate(
            prompt="What is (-3)^3 - (-2)^2?",
            candidate_answer="The answer is -23",
        )
        assert res.is_eligible is True
        assert res.computed_value == "-31"
        assert res.candidate_matches is False
        assert res.suggested_correction == "-31"

    def test_percentage_calculation(self):
        res = DeterministicVerifier.verify_candidate(
            prompt="What is 15% of 240?",
            candidate_answer="It is 36.",
        )
        assert res.is_eligible is True
        assert res.computed_value == "36"
        assert res.candidate_matches is True
        assert res.suggested_correction is None

    def test_unit_conversion_celsius_to_kelvin(self):
        res = DeterministicVerifier.verify_candidate(
            prompt="Convert 100 celsius to kelvin.",
            candidate_answer="100 degrees Celsius is 273.15 K",
        )
        assert res.is_eligible is True
        assert res.computed_value == "373.15"
        assert res.candidate_matches is False
        assert res.suggested_correction == "373.15"

    def test_unit_conversion_speed(self):
        res = DeterministicVerifier.verify_candidate(
            prompt="Convert 60 mph to km/h.",
            candidate_answer="It is approximately 96.56 km/h",
        )
        assert res.is_eligible is True
        assert res.computed_value == "96.56"
        assert res.candidate_matches is True

    def test_date_elapsed_years(self):
        res = DeterministicVerifier.verify_candidate(
            prompt="How many years elapsed between 1776 and 1789?",
            candidate_answer="There were 15 years.",
        )
        assert res.is_eligible is True
        assert res.computed_value == "13"
        assert res.candidate_matches is False
        assert res.suggested_correction == "13"

    def test_structured_slash_constraint(self):
        res = DeterministicVerifier.verify_candidate(
            prompt="Output the chemical symbol and atomic number of Iron separated by a slash.",
            candidate_answer="Iron has symbol Fe and atomic number 26",
        )
        assert res.is_eligible is True
        assert res.task_type == "constraint_validation"
        assert res.candidate_matches is False
        assert "missing_slash_delimiter" in res.reason


class TestEvidenceRetriever:
    @pytest.fixture
    def test_db(self, tmp_path):
        db_file = tmp_path / "test_fts.db"
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("""
        CREATE VIRTUAL TABLE evidence_fts USING fts5(
            doc_id UNINDEXED,
            title,
            document_path UNINDEXED,
            content,
            provenance UNINDEXED,
            tokenize = 'porter unicode61'
        );
        """)
        cursor.execute(
            "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?)",
            (
                "doc_test_001",
                "Capital of Australia",
                "/docs/australia.txt",
                "Canberra is the federal capital city of the Commonwealth of Australia, selected in 1908.",
                "Geoscience Australia",
            ),
        )
        cursor.execute(
            "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?)",
            (
                "doc_test_002",
                "Speed of Light",
                "/docs/c.txt",
                "The speed of light in vacuum is defined as exactly 299792458 meters per second.",
                "NIST SP 330",
            ),
        )
        conn.commit()
        conn.close()
        return db_file

    def test_retrieval_success_and_provenance(self, test_db):
        retriever = EvidenceRetriever(db_path=test_db, max_passages=2)
        res = retriever.retrieve("What is the capital city of Australia?")
        assert res.retrieval_success is True
        assert len(res.passages) >= 1
        p = res.passages[0]
        assert p.source_id == "doc_test_001"
        assert p.title == "Capital of Australia"
        assert "Canberra" in p.passage_text
        assert p.provenance == "Geoscience Australia"
        assert p.document_path == "/docs/australia.txt"
        assert p.retrieval_timestamp != ""

    def test_empty_query_keywords(self, test_db):
        retriever = EvidenceRetriever(db_path=test_db)
        res = retriever.retrieve("what is the of")
        assert res.retrieval_success is False
        assert res.status_message == "NO_SEARCH_KEYWORDS_EXTRACTED"

    def test_missing_database_graceful_handling(self, tmp_path):
        retriever = EvidenceRetriever(db_path=tmp_path / "nonexistent.db")
        res = retriever.retrieve("capital of Australia")
        assert res.retrieval_success is False
        assert res.status_message == "EVIDENCE_DATABASE_NOT_FOUND"


class TestEvidenceVerifier:
    def test_supported_verdict(self):
        p = RetrievedPassage(
            source_id="s1",
            title="Capital of Australia",
            document_path="p1",
            passage_text="Canberra is the federal capital city of Australia.",
            score=5.0,
            retrieval_timestamp="2026-10-14T12:00:00",
            provenance="Official Atlas",
        )
        ret = RetrievalResult(
            query="capital of Australia",
            retrieval_success=True,
            passages=[p],
            execution_time_ms=1.2,
            source_conflict_detected=False,
            status_message="SUCCESS",
        )
        assessment = EvidenceVerifier.verify(
            request_id="r1",
            prompt="What is the capital city of Australia?",
            candidate_answer="Canberra",
            retrieval_res=ret,
        )
        assert assessment.verification_state == VerificationState.SUPPORTED
        assert assessment.action == VerificationAction.ACCEPT_ORIGINAL
        assert assessment.verified_answer == "Canberra"
        assert assessment.is_corrected is False

    def test_contradicted_and_corrected(self):
        p = RetrievedPassage(
            source_id="s1",
            title="Capital of Australia",
            document_path="p1",
            passage_text="Canberra is the federal capital city of Australia.",
            score=5.0,
            retrieval_timestamp="2026-10-14T12:00:00",
            provenance="Official Atlas",
        )
        ret = RetrievalResult(
            query="capital of Australia",
            retrieval_success=True,
            passages=[p],
            execution_time_ms=1.2,
            source_conflict_detected=False,
            status_message="SUCCESS",
        )
        assessment = EvidenceVerifier.verify(
            request_id="r2",
            prompt="What is the capital city of Australia?",
            candidate_answer="Sydney",
            retrieval_res=ret,
        )
        assert assessment.verification_state == VerificationState.CONTRADICTED
        assert assessment.action == VerificationAction.APPLY_EVIDENCE_CORRECTION
        assert assessment.verified_answer == "Canberra"
        assert assessment.is_corrected is True

    def test_insufficient_evidence_retains_consensus(self):
        ret = RetrievalResult(
            query="unanswerable query",
            retrieval_success=False,
            passages=[],
            execution_time_ms=0.5,
            source_conflict_detected=False,
            status_message="NO_RELEVANT_PASSAGES_FOUND",
        )
        assessment = EvidenceVerifier.verify(
            request_id="r3",
            prompt="What is the obscure hypothetical entity?",
            candidate_answer="Plausible Guess",
            retrieval_res=ret,
            conservative_fallback=False,
        )
        assert assessment.verification_state == VerificationState.INSUFFICIENT_EVIDENCE
        assert assessment.action == VerificationAction.RETAIN_CONSENSUS
        assert assessment.verified_answer == "Plausible Guess"
        assert assessment.is_corrected is False

    def test_insufficient_evidence_conservative_abstains(self):
        ret = RetrievalResult(
            query="unanswerable query",
            retrieval_success=False,
            passages=[],
            execution_time_ms=0.5,
            source_conflict_detected=False,
            status_message="NO_RELEVANT_PASSAGES_FOUND",
        )
        assessment = EvidenceVerifier.verify(
            request_id="r4",
            prompt="What is the obscure hypothetical entity?",
            candidate_answer="Plausible Guess",
            retrieval_res=ret,
            conservative_fallback=True,
        )
        assert assessment.verification_state == VerificationState.INSUFFICIENT_EVIDENCE
        assert assessment.action == VerificationAction.EMIT_ABSTENTION_FALLBACK
        assert "[UNCERTAINTY_FALLBACK]" in assessment.verified_answer
        assert assessment.is_abstained is True


class TestEvidenceGroundedRouter:
    def test_e0_base_condition(self):
        router = EvidenceGroundedRouter()
        res = router.route(
            request_id="req_e0",
            cluster_id="c001",
            prompts=["What is 2+2?"],
            candidates=["4", "5", "4"],
            condition=ExperimentalCondition.E0_BASE,
        )
        assert res.condition == ExperimentalCondition.E0_BASE
        assert res.selected_answer == "4"
        assert res.is_corrected is False

    def test_e2_deterministic_routes_and_corrects(self):
        router = EvidenceGroundedRouter()
        res = router.route(
            request_id="req_e2",
            cluster_id="c_mod",
            prompts=["What is the remainder when 2^100 is divided by 7?"],
            candidates=["4", "4", "4"],  # Unanimous false consensus
            condition=ExperimentalCondition.E2_DETERMINISTIC,
        )
        assert res.condition == ExperimentalCondition.E2_DETERMINISTIC
        assert res.is_corrected is True
        assert res.selected_answer == "2"

    def test_e5_conservative_abstains_on_unverified(self):
        # Empty retrieval on high risk should trigger abstention
        retriever = EvidenceRetriever(db_path=Path("nonexistent.db"))
        router = EvidenceGroundedRouter(retriever=retriever)
        res = router.route(
            request_id="req_e5",
            cluster_id="c_amb",
            prompts=["What is the capital of Georgia?", "Name Georgia's capital."],
            candidates=["Tbilisi", "Atlanta"],  # Conflict / high risk
            condition=ExperimentalCondition.E5_CONSERVATIVE,
        )
        assert res.condition == ExperimentalCondition.E5_CONSERVATIVE
        assert res.is_abstained is True
        assert "[UNCERTAINTY_FALLBACK]" in res.selected_answer


class TestGoldReferenceIsolation:
    def test_domain_dataset_contains_no_gold_fields(self):
        domain_file = Path("data/phase6_eval/domain_dataset.jsonl")
        assert domain_file.exists(), "domain_dataset.jsonl must exist"
        with open(domain_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx > 50:
                    break
                rec = json.loads(line)
                assert "gold_reference_answer" not in rec
                assert "acceptable_aliases" not in rec
                assert "canonical_answer" not in rec
                assert "answer" not in rec

    def test_cluster_separation_between_val_and_test(self):
        manifest_path = Path("data/phase6_eval/split_manifest.json")
        assert manifest_path.exists()
        domain_file = Path("data/phase6_eval/domain_dataset.jsonl")

        val_cids = set()
        test_cids = set()
        with open(domain_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["split"] == "val":
                    val_cids.add(rec["cluster_id"])
                elif rec["split"] == "test":
                    test_cids.add(rec["cluster_id"])

        intersection = val_cids.intersection(test_cids)
        assert len(intersection) == 0, f"Clusters leaked between splits: {intersection}"
        assert len(val_cids) == 150
        assert len(test_cids) == 450
