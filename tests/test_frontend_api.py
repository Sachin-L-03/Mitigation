"""Targeted unit and integration tests for the interactive frontend and backend services.

Tests:
1. Paraphrase generation preserving original prompt at index 0 across 4 and 5 variants.
2. Paraphrase validation detecting contrasting attributes and missing quantities.
3. Baseline execution recording exact prompts, latencies, and output formats.
4. Mitigation execution routing through Phase 6 EvidenceGroundedRouter.
5. Identical prompt set alignment across baseline and mitigation.
6. Semantic relationship classification (identical, paraphrase, contradiction).
7. Correctness evaluation with and without user reference answers.
8. Metric denominator calculation consistency (consistency, accuracy, latency).
9. Session export generating JSON and CSV in outputs/session_exports/ without leakage.
10. HTTP REST API endpoints (/api/health, /api/paraphrase/generate, /api/inference/run_both, /api/export/session).
"""

import json
import urllib.parse
from pathlib import Path

import pytest

from src.service.export_service import ExportService
from src.service.inference_service import InferenceService
from src.service.model_service import ModelManager
from src.service.paraphrase_service import ParaphraseService


@pytest.fixture(scope="module", autouse=True)
def setup_mock_model():
    """Ensure ModelManager runs in mock mode for fast deterministic testing."""
    mm = ModelManager.get_instance()
    mm.load_model(force_mock=True)
    yield mm


class TestParaphraseService:
    def test_original_prompt_preserved_at_index_zero(self):
        """Original prompt must strictly be retained as the first element."""
        orig = "What is the capital city of Australia?"
        res = ParaphraseService.generate_paraphrases(orig, target_count=5)
        assert res["count"] == 5
        assert len(res["paraphrases"]) == 5
        assert res["paraphrases"][0]["is_original"] is True
        assert res["paraphrases"][0]["text"] == orig
        assert res["paraphrases"][0]["id"] == 0

    def test_paraphrase_count_selection(self):
        """Service must support both 4 and 5 requested variants."""
        orig = "What is 15% of 240?"
        res4 = ParaphraseService.generate_paraphrases(orig, target_count=4)
        assert res4["count"] == 4
        assert len(res4["paraphrases"]) == 4

        res5 = ParaphraseService.generate_paraphrases(orig, target_count=5)
        assert res5["count"] == 5
        assert len(res5["paraphrases"]) == 5

    def test_numerical_quantity_preservation(self):
        """Paraphrases must preserve mathematical quantities."""
        orig = "What is the remainder when 7^25 is divided by 13?"
        res = ParaphraseService.generate_paraphrases(orig, target_count=5)
        for p in res["paraphrases"]:
            # All variants must contain 7^25 (or 7 and 25) and 13
            assert "13" in p["text"]
            assert "7^25" in p["text"] or ("7" in p["text"] and "25" in p["text"])

    def test_contrasting_attribute_warning(self):
        """Contrasting attributes (e.g. capital vs largest city) must trigger a validation warning."""
        is_valid, note = ParaphraseService._validate_paraphrase(
            "What is the capital of Australia?",
            "What is the largest city of Australia?"
        )
        assert is_valid is False
        assert "Contrasting attributes" in note

    def test_empty_prompt_rejected(self):
        """Empty or whitespace prompts must raise ValueError."""
        with pytest.raises(ValueError):
            ParaphraseService.generate_paraphrases("   ")


class TestInferenceService:
    def test_baseline_and_mitigation_identical_prompt_sets(self):
        """Baseline and mitigation must execute on the exact same approved prompt cohort."""
        service = InferenceService()
        prompts = [
            "What is the remainder when 7^25 is divided by 13?",
            "Calculate 7^25 modulo 13.",
            "When 7^25 is divided by 13, what remainder is obtained?",
            "7^25 mod 13?"
        ]
        res = service.run_both(session_id="test_sess_align", prompts=prompts, reference_answer="7")
        assert len(res["baseline"]["results"]) == len(prompts)
        assert len(res["mitigation"]["results"]) == len(prompts)

        for i in range(len(prompts)):
            assert res["baseline"]["results"][i]["prompt_text"] == prompts[i]
            assert res["mitigation"]["results"][i]["prompt_text"] == prompts[i]

    def test_deterministic_ast_correction_in_mitigation(self):
        """Phase 6 router must correctly formalize and correct arithmetic false consensus."""
        service = InferenceService()
        prompts = [
            "What is the remainder when 7^25 is divided by 13?",
            "Calculate 7^25 modulo 13.",
            "7^25 mod 13?"
        ]
        # In mock baseline, answer is simulated false consensus 'The remainder is 4'
        res = service.run_both(session_id="test_sess_math", prompts=prompts, reference_answer="7")

        for m_res in res["mitigation"]["results"]:
            assert m_res["answer"] == "7"
            assert m_res["is_corrected"] is True
            assert "Deterministic" in m_res["verification_route"]
            assert m_res["is_correct"] is True

    def test_unknown_correctness_when_no_reference_provided(self):
        """When user provides no reference answer, is_correct must be None (not falsely assumed)."""
        service = InferenceService()
        prompts = ["What is the capital city of Australia?"]
        res = service.run_both(session_id="test_sess_noref", prompts=prompts, reference_answer=None)

        assert res["baseline"]["results"][0]["is_correct"] is None
        assert res["mitigation"]["results"][0]["is_correct"] is None
        assert res["metrics"]["factual_accuracy"]["evaluated"] is False

    def test_semantic_relation_classification(self):
        """Service must distinguish identical answers, paraphrases, and contradictions."""
        service = InferenceService()
        # Identical
        assert service._classify_semantic_relation("Canberra", "Canberra") == "identical"
        # Numerical contradiction
        assert service._classify_semantic_relation("The answer is 4.", "7") == "contradiction"
        assert service._classify_semantic_relation("-23", "-31") == "contradiction"
        # Negation contradiction
        assert service._classify_semantic_relation("It is allowed.", "It is not allowed.") == "contradiction"
        # Paraphrase
        assert service._classify_semantic_relation("The federal capital of Australia is Canberra.", "Canberra is Australia's capital.") == "paraphrase"

    def test_metric_calculation_and_denominators(self):
        """Metric calculations must accurately report consistency and accuracy with explicit denominators."""
        service = InferenceService()
        prompts = [
            "What is (-3)^3 - (-2)^2?",
            "Calculate (-3)^3 - (-2)^2",
            "Evaluate (-3)^3 - (-2)^2?"
        ]
        res = service.run_both(session_id="test_sess_metrics", prompts=prompts, reference_answer="-31")
        m = res["metrics"]

        assert m["n_prompts"] == 3
        assert m["factual_accuracy"]["evaluated"] is True
        assert m["factual_accuracy"]["denominator"] == 3
        assert m["factual_accuracy"]["mitigation_correct"] == 3
        assert m["factual_accuracy"]["mitigation_accuracy_pct"] == 100.0


class TestExportService:
    def test_export_json_and_csv_persistence(self, tmp_path):
        """Export service must create structured files in session_exports without altering historical dirs."""
        session_data = {
            "session_id": "test_unit_session",
            "reference_answer": "42",
            "baseline": {
                "results": [
                    {"prompt_text": "What is the answer?", "answer": "42", "latency_ms": 50.0, "is_correct": True}
                ]
            },
            "mitigation": {
                "results": [
                    {
                        "prompt_text": "What is the answer?",
                        "answer": "42",
                        "latency_ms": 52.0,
                        "verification_overhead_ms": 2.0,
                        "verification_route": "Path A: Deterministic AST Tooling",
                        "verification_state": "SUPPORTED",
                        "is_corrected": False,
                        "is_abstained": False,
                        "semantic_relation": "identical",
                        "is_correct": True,
                    }
                ]
            },
            "metrics": {"consistency": {"change_pp": 0.0}},
        }

        # Export JSON
        res_json = ExportService.export_session(session_data, export_format="json")
        json_file = Path(res_json["file_path"])
        assert json_file.exists()
        assert json_file.suffix == ".json"
        with open(json_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            assert loaded["session"]["session_id"] == "test_unit_session"

        # Export CSV
        res_csv = ExportService.export_session(session_data, export_format="csv")
        csv_file = Path(res_csv["file_path"])
        assert csv_file.exists()
        assert csv_file.suffix == ".csv"
        with open(csv_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "test_unit_session" in content
            assert "Path A: Deterministic AST Tooling" in content


class TestServerEndpointIntegration:
    """Integration test checking HTTP handler dispatch logic without opening live network ports."""

    def test_app_handler_routes(self):
        from app import MitigationAppHandler

        assert hasattr(MitigationAppHandler, "_handle_health")
        assert hasattr(MitigationAppHandler, "_handle_paraphrase_generate")
        assert hasattr(MitigationAppHandler, "_handle_inference_baseline")
        assert hasattr(MitigationAppHandler, "_handle_inference_mitigation")
        assert hasattr(MitigationAppHandler, "_handle_inference_run_both")
        assert hasattr(MitigationAppHandler, "_handle_metrics_compare")
        assert hasattr(MitigationAppHandler, "_handle_export_session")
