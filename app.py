"""Interactive Web Application Server for Paraphrase Mitigation Framework.

Serves the scientific research frontend and REST APIs for prompt entry,
paraphrase generation, live baseline execution, mitigation routing, side-by-side
comparison metrics, and session exports.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service.export_service import ExportService
from src.service.inference_service import InferenceService
from src.service.model_service import ModelManager
from src.service.paraphrase_service import ParaphraseService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("prompt_sensitivity.frontend.server")


def log_user_test_activity(action: str, payload: Dict[str, Any], result: Dict[str, Any]):
    """Automatically record live user test activity to a persistent JSONL log."""
    try:
        log_dir = PROJECT_ROOT / "outputs" / "session_exports"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "user_live_testing_log.jsonl"
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "session_id": result.get("session_id") or payload.get("session_id"),
            "condition": payload.get("condition"),
            "reference_answer": payload.get("reference_answer"),
            "prompts": payload.get("prompts") or ([payload.get("prompt")] if payload.get("prompt") else []),
            "result": result,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Recorded live test activity '{action}' (session: {entry['session_id']})")
    except Exception as e:
        logger.warning(f"Could not record live user test activity: {e}")


class MitigationAppHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler serving static frontend and JSON REST APIs."""

    def __init__(self, *args, **kwargs):
        self.frontend_dir = PROJECT_ROOT / "frontend"
        super().__init__(*args, directory=str(self.frontend_dir), **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Route static assets and GET API endpoints."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._handle_health()
        elif path == "/api/export/download":
            self._handle_download(parsed.query)
        elif path == "/" or path == "/index.html":
            self._serve_file(self.frontend_dir / "index.html", "text/html")
        elif path in ("/style.css", "/app.js"):
            file_path = self.frontend_dir / path.lstrip("/")
            mime_type = "text/css" if path.endswith(".css") else "application/javascript"
            self._serve_file(file_path, mime_type)
        else:
            # Fall back to standard static file server
            super().do_GET()

    def do_POST(self):
        """Route POST API endpoints."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Read JSON body safely
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len > 5 * 1024 * 1024:  # 5 MB limit
                self._send_json_error("Request body exceeds 5MB limit", HTTPStatus.BAD_REQUEST)
                return

            body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as e:
            self._send_json_error(f"Malformed JSON request: {e}", HTTPStatus.BAD_REQUEST)
            return

        # Route endpoints
        try:
            if path == "/api/model/load":
                self._handle_model_load(payload)
            elif path == "/api/paraphrase/generate":
                self._handle_paraphrase_generate(payload)
            elif path == "/api/inference/baseline":
                self._handle_inference_baseline(payload)
            elif path == "/api/inference/mitigation":
                self._handle_inference_mitigation(payload)
            elif path == "/api/inference/run_both":
                self._handle_inference_run_both(payload)
            elif path == "/api/metrics/compare":
                self._handle_metrics_compare(payload)
            elif path == "/api/export/session":
                self._handle_export_session(payload)
            else:
                self._send_json_error(f"Unknown endpoint '{path}'", HTTPStatus.NOT_FOUND)
        except Exception as e:
            logger.error(f"Error handling {path}: {e}", exc_info=True)
            self._send_json_error(f"Server error: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)

    # -------------------------------------------------------------------------
    # API Handlers
    # -------------------------------------------------------------------------

    def _handle_health(self):
        mm = ModelManager.get_instance()
        status = mm.get_status()
        self._send_json_response({"status": "ok", "model": status})

    def _handle_model_load(self, payload: Dict[str, Any]):
        mm = ModelManager.get_instance()
        force_mock = payload.get("force_mock", False)
        status = mm.load_model(force_mock=force_mock)
        self._send_json_response({"status": "ok", "model": status})

    def _handle_paraphrase_generate(self, payload: Dict[str, Any]):
        prompt = payload.get("prompt", "")
        count = payload.get("count", 5)
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            self._send_json_error("Prompt text is required.", HTTPStatus.BAD_REQUEST)
            return

        res = ParaphraseService.generate_paraphrases(prompt=prompt, target_count=count)
        log_user_test_activity("paraphrase_generate", payload, res)
        self._send_json_response(res)

    def _handle_inference_baseline(self, payload: Dict[str, Any]):
        prompts = payload.get("prompts", [])
        session_id = payload.get("session_id") or f"sess_{int(time.time())}"
        ref_ans = payload.get("reference_answer")

        if not prompts or not isinstance(prompts, list):
            self._send_json_error("A non-empty list of prompts is required.", HTTPStatus.BAD_REQUEST)
            return

        service = InferenceService()
        res = service.run_baseline(session_id=session_id, prompts=prompts, reference_answer=ref_ans)
        log_user_test_activity("inference_baseline", payload, res)
        self._send_json_response(res)

    def _handle_inference_mitigation(self, payload: Dict[str, Any]):
        prompts = payload.get("prompts", [])
        candidates = payload.get("baseline_candidates")
        session_id = payload.get("session_id") or f"sess_{int(time.time())}"
        condition = payload.get("condition", "E4_combined")
        ref_ans = payload.get("reference_answer")

        if not prompts or not isinstance(prompts, list):
            self._send_json_error("A non-empty list of prompts is required.", HTTPStatus.BAD_REQUEST)
            return

        service = InferenceService()
        res = service.run_mitigation(
            session_id=session_id,
            prompts=prompts,
            baseline_candidates=candidates,
            condition=condition,
            reference_answer=ref_ans,
        )
        log_user_test_activity("inference_mitigation", payload, res)
        self._send_json_response(res)

    def _handle_inference_run_both(self, payload: Dict[str, Any]):
        prompts = payload.get("prompts", [])
        session_id = payload.get("session_id") or f"sess_{int(time.time())}"
        condition = payload.get("condition", "E4_combined")
        ref_ans = payload.get("reference_answer")

        if not prompts or not isinstance(prompts, list):
            self._send_json_error("A non-empty list of prompts is required.", HTTPStatus.BAD_REQUEST)
            return

        service = InferenceService()
        res = service.run_both(
            session_id=session_id,
            prompts=prompts,
            condition=condition,
            reference_answer=ref_ans,
        )
        log_user_test_activity("inference_run_both", payload, res)
        self._send_json_response(res)

    def _handle_metrics_compare(self, payload: Dict[str, Any]):
        base_res = payload.get("baseline_results", [])
        mit_res = payload.get("mitigation_results", [])
        ref_ans = payload.get("reference_answer")

        service = InferenceService()
        res = service.compute_comparison_metrics(
            baseline_results=base_res,
            mitigation_results=mit_res,
            reference_answer=ref_ans,
        )
        self._send_json_response(res)

    def _handle_export_session(self, payload: Dict[str, Any]):
        session_data = payload.get("session_data", {})
        fmt = payload.get("format", "json")

        if not session_data:
            self._send_json_error("No session_data provided for export.", HTTPStatus.BAD_REQUEST)
            return

        res = ExportService.export_session(session_data=session_data, export_format=fmt)
        log_user_test_activity("export_session", payload, res)
        self._send_json_response(res)

    def _handle_download(self, query_string: str):
        params = urllib.parse.parse_qs(query_string)
        filename = params.get("filename", [""])[0]
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            self._send_json_error("Invalid filename parameter.", HTTPStatus.BAD_REQUEST)
            return

        target_file = ExportService.EXPORT_DIR / filename
        if not target_file.exists() or not target_file.is_file():
            self._send_json_error("Requested export file not found.", HTTPStatus.NOT_FOUND)
            return

        mime_type = "text/csv" if filename.endswith(".csv") else "application/json"
        self._serve_file(target_file, mime_type, as_attachment=True, filename=filename)

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------

    def _send_json_response(self, data: Any, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_json_error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        self._send_json_response({"error": message, "status": status.value}, status=status)

    def _serve_file(self, file_path: Path, content_type: str, as_attachment: bool = False, filename: Optional[str] = None):
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {file_path.name}")
            return

        with open(file_path, "rb") as f:
            content = f.read()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        if as_attachment:
            disp_name = filename or file_path.name
            self.send_header("Content-Disposition", f'attachment; filename="{disp_name}"')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(content)


def run_server(host: str = "127.0.0.1", port: int = 8000, force_mock: bool = False):
    """Start the interactive web application server."""
    mm = ModelManager.get_instance()
    if force_mock:
        mm.load_model(force_mock=True)
    else:
        logger.info("Initiating background pre-warm of Qwen2.5-7B-Instruct onto GPU...")
        threading.Thread(target=mm.load_model, kwargs={"force_mock": False}, daemon=True).start()

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, MitigationAppHandler)
    url = f"http://{host}:{port}"

    print("\n" + "=" * 70)
    print("      PARAPHRASE SENSITIVITY MITIGATION: RESEARCH WEB DEMO")
    print("=" * 70)
    print(f"Server URL:        {url}")
    print(f"Model:             {mm.model_name}")
    print(f"Mock Mode:         {mm.mock_mode}")
    print(f"CUDA Available:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device:        {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:        {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    print("=" * 70)
    print("Press Ctrl+C to terminate the application server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase Sensitivity Mitigation Web App Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--mock", action="store_true", help="Launch in mock mode without loading GPU weights")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, force_mock=args.mock)
