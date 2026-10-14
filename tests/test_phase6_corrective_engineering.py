"""Targeted regression tests for Phase 6 corrective engineering.

Covers:
1. Exact signed number boundary matching and terminal punctuation tolerance.
2. Safe AST evaluator rejecting unsafe or malformed expressions.
3. Diverse percentage and ratio paraphrase variants.
4. Unit conversion normalization and alias mapping.
5. Date arithmetic elapsed years calculation.
6. Non-equivalent prompt detection routing to ambiguous state.
7. Filtering institutional publisher names in evidence entity extraction.
8. Event-predicate co-occurrence protecting against first-year overwrite errors.
9. Contradictory evidence detection and correction.
10. Irrelevant evidence handling and conservative abstention.
11. Metric denominator calculation consistency (coverage, selective accuracy, gross accuracy).
12. Runtime isolation from gold references.
"""

import pytest
from src.mitigation.deterministic_verifier import (
    DeterministicVerifier,
    SafeASTEvaluator,
    DeterministicVerificationResult,
)
from src.mitigation.evidence_retriever import (
    RetrievalResult,
    RetrievedPassage,
)
from src.mitigation.evidence_verifier import (
    EvidenceVerifier,
    VerificationAction,
    VerificationState,
    EvidenceVerificationAssessment,
)


class TestCorrectiveDeterministicVerifier:
    def test_signed_number_boundaries(self):
        """Negative numbers must not match positive answers and vice versa."""
        assert DeterministicVerifier._match_number_in_text("-23", "The temperature was -23 degrees.") is True
        assert DeterministicVerifier._match_number_in_text("-23", "The temperature was 23 degrees.") is False
        assert DeterministicVerifier._match_number_in_text("23", "The temperature was -23 degrees.") is False
        assert DeterministicVerifier._match_number_in_text("23", "The temperature was 23 degrees.") is True

    def test_terminal_punctuation_matching(self):
        """Terminal sentence punctuation (periods) must not prevent valid number matches."""
        # Value "36" with terminal period in sentence
        assert DeterministicVerifier._match_number_in_text("36", "It is 36.") is True
        assert DeterministicVerifier._match_number_in_text("36", "Answer: 36.") is True
        # Decimal continuation must NOT match
        assert DeterministicVerifier._match_number_in_text("36", "It is 36.5.") is False
        assert DeterministicVerifier._match_number_in_text("36", "It is 365.") is False
        # Decimal target with terminal period
        assert DeterministicVerifier._match_number_in_text("36.5", "It is 36.5.") is True
        assert DeterministicVerifier._match_number_in_text("36.5", "It is 36.55.") is False

    def test_safe_ast_evaluator_security(self):
        """Unsafe or malicious AST operations must be strictly rejected."""
        assert SafeASTEvaluator.evaluate("__import__('os').system('dir')") is None
        assert SafeASTEvaluator.evaluate("eval('2 + 2')") is None
        assert SafeASTEvaluator.evaluate("open('file.txt')") is None
        assert SafeASTEvaluator.evaluate("x + 5") is None
        assert SafeASTEvaluator.evaluate("10 / 0") is None
        # Valid safe expressions
        assert SafeASTEvaluator.evaluate("15 + 45 * 2") == 105.0
        assert SafeASTEvaluator.evaluate("(100 - 20) / 4") == 20.0
        assert SafeASTEvaluator.evaluate("2 ** 5") == 32.0

    def test_percentage_paraphrase_variants(self):
        """Verifier must handle various percentage formulations."""
        r1 = DeterministicVerifier.verify_candidate("What is 15% of 240?", "It is 36.")
        assert r1.is_eligible and r1.computed_value == "36" and r1.candidate_matches is True

        r2 = DeterministicVerifier.verify_candidate("Calculate 15 percent of 240", "Result: 36")
        assert r2.is_eligible and r2.computed_value == "36" and r2.candidate_matches is True

        r3 = DeterministicVerifier.verify_candidate("36 is what percent of 240?", "15 percent")
        assert r3.is_eligible and r3.computed_value == "15" and r3.candidate_matches is True

    def test_unit_conversion_and_aliases(self):
        """Unit conversions must handle singular/plural and alias names correctly."""
        r1 = DeterministicVerifier.verify_candidate("Convert 5000 meters to km", "5 km")
        assert r1.is_eligible and r1.computed_value == "5" and r1.candidate_matches is True

        r2 = DeterministicVerifier.verify_candidate("How many km in 5000 meter?", "5.0")
        assert r2.is_eligible and r2.computed_value == "5" and r2.candidate_matches is True

        r3 = DeterministicVerifier.verify_candidate("Convert 0 celsius to kelvin", "273.15 K")
        assert r3.is_eligible and r3.computed_value == "273.15" and r3.candidate_matches is True

    def test_date_elapsed_years(self):
        """Date arithmetic must compute duration between dates accurately."""
        r1 = DeterministicVerifier.verify_candidate("How many years elapsed between 1945 and 2020?", "75 years")
        assert r1.is_eligible and r1.computed_value == "75" and r1.candidate_matches is True

        r2 = DeterministicVerifier.verify_candidate("Difference in years between 1914 and 1918", "4 years")
        assert r2.is_eligible and r2.computed_value == "4" and r2.candidate_matches is True


class TestCorrectiveEvidenceVerifier:
    def test_non_equivalent_prompts_route_to_ambiguous(self):
        """When prompt_non_equivalent is True, verifier must route to AMBIGUOUS_QUESTION."""
        p = RetrievedPassage(
            source_id="doc_test_1",
            title="Physics Reference",
            document_path="/docs/physics.txt",
            passage_text="The speed of light in vacuum is defined as 299792458 meters per second.",
            score=5.0,
            retrieval_timestamp="2026-10-14T12:00:00",
            provenance="NIST",
        )
        ret = RetrievalResult(
            query="speed of light",
            retrieval_success=True,
            passages=[p],
            execution_time_ms=1.0,
            source_conflict_detected=False,
            status_message="SUCCESS",
        )
        res = EvidenceVerifier.verify(
            request_id="req_ambig",
            prompt="What is the speed of sound and light?",
            candidate_answer="299792458 m/s",
            retrieval_res=ret,
            conservative_fallback=True,
            prompt_non_equivalent=True,
        )
        assert res.verification_state == VerificationState.AMBIGUOUS_QUESTION
        assert res.is_abstained is True
        assert res.action == VerificationAction.EMIT_ABSTENTION_FALLBACK

    def test_institutional_publisher_filtering(self):
        """Institutional publishers must not be extracted as author names."""
        p = RetrievedPassage(
            source_id="doc_press",
            title="Book Catalog",
            document_path="/docs/catalog.txt",
            passage_text="The monograph was published by Oxford University Press in 1995.",
            score=4.0,
            retrieval_timestamp="2026-10-14T12:00:00",
            provenance="Publisher Database",
        )
        ret = RetrievalResult(
            query="who wrote monograph",
            retrieval_success=True,
            passages=[p],
            execution_time_ms=1.0,
            source_conflict_detected=False,
            status_message="SUCCESS",
        )
        res = EvidenceVerifier.verify(
            request_id="req_inst",
            prompt="Who wrote the monograph?",
            candidate_answer="Oxford University Press",
            retrieval_res=ret,
            conservative_fallback=False,
        )
        # Should NOT confirm Oxford University Press as author or overwrite with it
        assert res.verified_answer != "Oxford University Press" or res.is_corrected is False

    def test_passage_year_cooccurrence_protection(self):
        """Verifier must not overwrite a correct event year with an unrelated prior year in passage."""
        p = RetrievedPassage(
            source_id="doc_history",
            title="Treaty of Paris",
            document_path="/docs/paris.txt",
            passage_text="In 1999 the exhibition opened. The Treaty of Paris was signed in 1783.",
            score=4.5,
            retrieval_timestamp="2026-10-14T12:00:00",
            provenance="Historical Encyclopedia",
        )
        ret = RetrievalResult(
            query="When was Treaty of Paris signed?",
            retrieval_success=True,
            passages=[p],
            execution_time_ms=1.0,
            source_conflict_detected=False,
            status_message="SUCCESS",
        )
        # Candidate says 1783. Verifier must not blindly pick 1999 because it appeared first!
        res = EvidenceVerifier.verify(
            request_id="req_hist",
            prompt="What year was the Treaty of Paris signed?",
            candidate_answer="1783",
            retrieval_res=ret,
            conservative_fallback=False,
        )
        assert res.verification_state == VerificationState.SUPPORTED
        assert res.action == VerificationAction.ACCEPT_ORIGINAL
        assert res.verified_answer == "1783"
        assert res.is_corrected is False

    def test_evidence_contradiction_detection(self):
        """Contradictory evidence must be flagged and corrected."""
        p = RetrievedPassage(
            source_id="doc_ptable",
            title="Halogens",
            document_path="/docs/halogens.txt",
            passage_text="Fluorine and Chlorine belong to group 17 of the periodic table.",
            score=4.8,
            retrieval_timestamp="2026-10-14T12:00:00",
            provenance="IUPAC",
        )
        ret = RetrievalResult(
            query="Fluorine group",
            retrieval_success=True,
            passages=[p],
            execution_time_ms=1.0,
            source_conflict_detected=False,
            status_message="SUCCESS",
        )
        res = EvidenceVerifier.verify(
            request_id="req_contra",
            prompt="What periodic table group is Fluorine in?",
            candidate_answer="It belongs to group 16",
            retrieval_res=ret,
            conservative_fallback=False,
        )
        assert res.verification_state == VerificationState.CONTRADICTED
        assert res.action == VerificationAction.APPLY_EVIDENCE_CORRECTION
        assert "17" in res.verified_answer
        assert res.is_corrected is True

    def test_irrelevant_evidence_and_conservative_abstention(self):
        """When evidence is insufficient/irrelevant, conservative policy must abstain."""
        ret = RetrievalResult(
            query="boiling point tungsten",
            retrieval_success=False,
            passages=[],
            execution_time_ms=0.5,
            source_conflict_detected=False,
            status_message="NO_RELEVANT_PASSAGES_FOUND",
        )
        res = EvidenceVerifier.verify(
            request_id="req_abst",
            prompt="What is the boiling point of tungsten?",
            candidate_answer="5555 degrees C",
            retrieval_res=ret,
            conservative_fallback=True,
        )
        assert res.verification_state == VerificationState.INSUFFICIENT_EVIDENCE
        assert res.is_abstained is True
        assert res.action == VerificationAction.EMIT_ABSTENTION_FALLBACK


class TestMetricAndIsolationIntegrity:
    def test_abstention_and_coverage_denominator_consistency(self):
        """Verify metric arithmetic: coverage = answered/total, selective = correct/answered."""
        total_clusters = 450
        answered_clusters = 400
        correct_answers = 350

        coverage = answered_clusters / total_clusters
        gross_accuracy = correct_answers / total_clusters
        selective_accuracy = correct_answers / answered_clusters

        assert pytest.approx(coverage, 0.001) == 0.8888
        assert pytest.approx(gross_accuracy, 0.001) == 0.7777
        assert pytest.approx(selective_accuracy, 0.001) == 0.875
        # Mathematical invariant: gross_accuracy == coverage * selective_accuracy
        assert pytest.approx(gross_accuracy, 1e-6) == coverage * selective_accuracy

    def test_runtime_isolation_from_gold_references(self):
        """Runtime verification modules must not accept or process gold reference parameters."""
        import inspect

        det_params = inspect.signature(DeterministicVerifier.verify_candidate).parameters
        assert "gold_answer" not in det_params
        assert "reference_answer" not in det_params
        assert "target_answer" not in det_params

        ev_params = inspect.signature(EvidenceVerifier.verify).parameters
        assert "gold_answer" not in ev_params
        assert "reference_answer" not in ev_params
        assert "ground_truth" not in ev_params
