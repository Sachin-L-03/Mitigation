"""Session export service supporting JSON and CSV formats.

Saves complete comparison sessions in outputs/session_exports/ without
overwriting historical research artifacts or exposing sensitive credentials.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("prompt_sensitivity.frontend.export_service")


class ExportService:
    """Exports interactive comparison sessions to persistent files."""

    EXPORT_DIR = Path("outputs/session_exports")

    @classmethod
    def export_session(
        cls,
        session_data: Dict[str, Any],
        export_format: str = "json",
    ) -> Dict[str, Any]:
        """Export session data to outputs/session_exports/.

        Args:
            session_data: Dictionary containing session_id, prompts, baseline, mitigation, and metrics.
            export_format: 'json' or 'csv'

        Returns:
            Dictionary with filename, file_path, format, and status.
        """
        cls.EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        session_id = session_data.get("session_id", "unnamed_session")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_format = export_format.lower()

        if export_format == "csv":
            filename = f"session_{timestamp}_{session_id}.csv"
            file_path = cls.EXPORT_DIR / filename
            cls._write_csv(file_path, session_data)
        else:
            filename = f"session_{timestamp}_{session_id}.json"
            file_path = cls.EXPORT_DIR / filename
            cls._write_json(file_path, session_data)

        logger.info(f"Exported session {session_id} to {file_path}")

        return {
            "status": "success",
            "filename": filename,
            "file_path": str(file_path),
            "format": export_format,
            "timestamp": timestamp,
        }

    @classmethod
    def _write_json(cls, path: Path, data: Dict[str, Any]) -> None:
        """Write session data to structured JSON file."""
        export_payload = {
            "metadata": {
                "exported_at": datetime.datetime.now().isoformat(),
                "project": "Prompt Sensitivity Mitigation Research Demo",
                "framework_version": "Phase 6 Hybrid Verification",
            },
            "session": data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def _write_csv(cls, path: Path, data: Dict[str, Any]) -> None:
        """Write aligned prompt comparison records to CSV."""
        baseline_results = data.get("baseline", {}).get("results", [])
        mitigation_results = data.get("mitigation", {}).get("results", [])

        fieldnames = [
            "session_id",
            "prompt_index",
            "prompt_text",
            "reference_answer",
            "baseline_answer",
            "mitigated_answer",
            "baseline_latency_ms",
            "mitigation_latency_ms",
            "verification_overhead_ms",
            "verification_route",
            "verification_state",
            "semantic_relation",
            "baseline_correct",
            "mitigated_correct",
            "is_corrected",
            "is_abstained",
            "correction_source",
        ]

        rows = []
        n = max(len(baseline_results), len(mitigation_results))
        ref_ans = data.get("reference_answer", "")

        for i in range(n):
            b_item = baseline_results[i] if i < len(baseline_results) else {}
            m_item = mitigation_results[i] if i < len(mitigation_results) else {}

            rows.append({
                "session_id": data.get("session_id", ""),
                "prompt_index": i,
                "prompt_text": b_item.get("prompt_text") or m_item.get("prompt_text", ""),
                "reference_answer": ref_ans or "",
                "baseline_answer": b_item.get("answer", ""),
                "mitigated_answer": m_item.get("answer", ""),
                "baseline_latency_ms": b_item.get("latency_ms", ""),
                "mitigation_latency_ms": m_item.get("latency_ms", ""),
                "verification_overhead_ms": m_item.get("verification_overhead_ms", ""),
                "verification_route": m_item.get("verification_route", ""),
                "verification_state": m_item.get("verification_state", ""),
                "semantic_relation": m_item.get("semantic_relation", ""),
                "baseline_correct": b_item.get("is_correct", ""),
                "mitigated_correct": m_item.get("is_correct", ""),
                "is_corrected": m_item.get("is_corrected", ""),
                "is_abstained": m_item.get("is_abstained", ""),
                "correction_source": m_item.get("correction_source", ""),
            })

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
