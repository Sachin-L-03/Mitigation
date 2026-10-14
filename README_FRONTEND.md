# Paraphrase Sensitivity Mitigation — Interactive Web Application

A scientific web application and research demonstration comparing baseline LLM responses with verification-based mitigation across semantically equivalent prompt paraphrases.

---

## 1. Overview & Architecture

This application provides an interactive laboratory for inspecting how language models respond to semantically equivalent prompt variations, demonstrating:
1. **Meaning-Preserving Paraphrase Cohorts:** Generates 4–5 diverse paraphrases along established linguistic axes (Lexical, Syntactic, Pragmatic, and Concise) with the original prompt strictly as index 0.
2. **Heuristic Semantic Validation:** Uses [`PromptEquivalenceChecker`](file:///c:/Users/Student5/Desktop/Mitigation/src/mitigation/factual_risk_detector.py#L366) to check entity, date, quantity, and negation preservation, warning users if an attribute was altered.
3. **Editable User Review:** Allows full inspection and editing of prompt text prior to execution.
4. **Live Baseline Inference:** Calls the unmitigated base model (`Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4) with greedy decoding.
5. **Phase 6 Evidence-Grounded Mitigation:** Routes the approved prompt set through the real Phase 6 multi-path framework:
   - **Path A:** Safe AST and deterministic mathematical verification.
   - **Path B:** Authoritative SQLite FTS5 evidence retrieval.
   - **Path C:** Evidence-support verification, contradiction detection, and conservative abstention.
6. **Side-by-Side Comparison:** Compares answers, latencies, semantic relationships (identical, paraphrase, contradiction), and verification routes side-by-side with expandable evidence/AST traces.
7. **Scientific Metric Dashboard:** Real-time calculation of pairwise Token F1 consistency, factual accuracy against optional reference answers, contradiction counts, and verification pathway distributions.
8. **Session Export:** Exports comparison sessions to JSON or CSV in `outputs/session_exports/` without overwriting historical research artifacts.

---

## 2. Prerequisites & Environment

The application runs inside the project virtual environment on Windows:
- **Python:** 3.11+
- **PyTorch:** 2.5.1 with CUDA 12.1
- **Target GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Quantization:** `bitsandbytes` 4-bit NF4
- **Backend Stack:** Native Python `http.server.ThreadingHTTPServer` (zero external framework dependencies)
- **Frontend Stack:** Semantic HTML5, Vanilla CSS3 (modern glassmorphism dark mode), and Vanilla JavaScript (SPA reactive architecture)

---

## 3. Quick Start & Launch Commands

### A. Live GPU Serving Mode (Default)
Loads `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 onto the local NVIDIA RTX 4090 GPU:
```powershell
.\.venv\Scripts\python.exe app.py --port 8000
```
- Open your browser at: `http://127.0.0.1:8000`
- The server will dynamically load the 4-bit weights into ~7.4 GB of GPU VRAM on first generation or via the status bar.

### B. Lightweight Mock / Demonstration Mode
Runs the entire frontend, paraphrase generator, Phase 6 router, and metric engine using deterministic simulated baseline inference without loading the 14 GB model weights into VRAM:
```powershell
.\.venv\Scripts\python.exe app.py --port 8000 --mock
```

### C. Specifying Host and Port
```powershell
.\.venv\Scripts\python.exe app.py --host 127.0.0.1 --port 8080
```

---

## 4. User Journey & Step-by-Step Usage

1. **Step 1: Enter or Select a Prompt**
   - Click one of the research preset chips (e.g. *Modular Math*, *Signed Math*, *Historical Dates*, *Capital Entity*, *Percentages*, *Periodic Table*, *Ambiguous Fallback*) or type a custom prompt.
   - Optionally enter a ground truth reference answer (e.g. `7`, `Canberra`, `-31`). This reference is strictly used for offline correctness evaluation and is **never** shown to the model or router.
   - Choose whether you want 4 or 5 paraphrases.
   - Select the desired mitigation condition (default: `E4_combined`).
   - Click **Generate Paraphrases**.

2. **Step 2: Review and Edit Paraphrases**
   - The original prompt is always locked or displayed at Index 0.
   - Variants 1–4 are displayed with their linguistic axis tag (*Lexical*, *Syntactic*, *Pragmatic*, *Concise*).
   - If an attribute conflict or quantity omission is detected, an amber warning badge is shown.
   - You can click into any text box to edit or refine the wording.

3. **Step 3: Run Inference**
   - Click **Run Both (Baseline & Mitigation)** to execute both methods sequentially.
   - Alternatively, click **Run Baseline Only** or **Run Mitigation Only**.
   - A real-time progress bar shows the execution stage.

4. **Step 4: Inspect Side-by-Side Results**
   - Each prompt card displays:
     - Baseline answer, latency, and correctness badge.
     - Mitigated answer, latency, verification route badge, and state badge.
     - Semantic difference indicator (`Identical`, `Equivalent Paraphrase`, or `Contradiction Detected`).
     - Latency delta showing algorithmic verification overhead in milliseconds.
     - Expandable **Evidence Drawer**: Click to inspect retrieved passage snippets with source provenance or deterministic AST formal expressions.

5. **Step 5: Review Metrics & Export**
   - **Answer Consistency:** Pairwise Token F1 agreement before vs. after mitigation.
   - **Factual Correctness:** Accuracy before vs. after (with exact $k/N$ numerator and denominator).
   - **Contradictions Resolved:** Count of detected answer conflicts.
   - **Median Latency:** Comparison of baseline vs. mitigated generation latency and average algorithmic overhead.
   - **Pathway Breakdown:** Visual bar showing the proportion of Deterministic AST, Evidence Grounded, Consensus, and Conservative Abstentions.
   - Click **Export JSON** or **Export CSV** to save the session to `outputs/session_exports/` and trigger an immediate browser download.

---

## 5. REST API Reference

| Endpoint | Method | Description | Payload Example |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | Hardware, VRAM, and model status | N/A |
| `/api/model/load` | `POST` | Trigger model weight loading | `{"force_mock": false}` |
| `/api/paraphrase/generate` | `POST` | Generate meaning-preserving paraphrases | `{"prompt": "...", "count": 5}` |
| `/api/inference/baseline` | `POST` | Execute baseline model on prompt cohort | `{"session_id": "...", "prompts": [...], "reference_answer": "7"}` |
| `/api/inference/mitigation` | `POST` | Execute Phase 6 router on prompt cohort | `{"session_id": "...", "prompts": [...], "condition": "E4_combined"}` |
| `/api/inference/run_both` | `POST` | Execute baseline + mitigation + metrics | `{"session_id": "...", "prompts": [...], "condition": "E4_combined"}` |
| `/api/metrics/compare` | `POST` | Recompute comparison metrics | `{"baseline_results": [...], "mitigation_results": [...]}` |
| `/api/export/session` | `POST` | Save session to JSON or CSV | `{"session_data": {...}, "format": "json"}` |
| `/api/export/download` | `GET` | Stream exported session file | Query param: `?filename=session_...json` |

---

## 6. Testing & Validation

The application test suite is located in [`tests/test_frontend_api.py`](file:///c:/Users/Student5/Desktop/Mitigation/tests/test_frontend_api.py) and is fully integrated with pytest:
```powershell
.\.venv\Scripts\pytest.exe tests/test_frontend_api.py -v
```
All 12 targeted unit and integration tests pass cleanly:
- Paraphrase preservation of index 0.
- Paraphrase quantity and attribute validation.
- Identical prompt cohort alignment across baseline and mitigation.
- Phase 6 router formalization and correction.
- Unknown correctness handling when no reference is provided.
- Semantic relation classification (identical, paraphrase, contradiction).
- Metric denominators and consistency calculations.
- Session export persistence in `outputs/session_exports/`.

---

## 7. Research Transparency & Boundaries

- **Live vs. Benchmark:** This application runs live inference on user-entered queries. It is designed for interactive exploration and does not replace the formal 450-cluster audited Phase 6 benchmark.
- **Consistency $\ne$ Correctness:** High answer agreement across paraphrases does not guarantee truth. Systematic false consensus is explicitly highlighted and resolved by Path A deterministic verification.
- **Evidence Corpus Boundary:** The local SQLite FTS5 database contains a curated set of authoritative benchmark records. Open-domain queries outside the corpus fallback safely to consensus or conservative abstention.
- **Artifact Protection:** All exports are strictly isolated to `outputs/session_exports/`. Historical artifacts in `outputs/phase1/` through `outputs/phase6/` are never overwritten.

---

## 8. Troubleshooting

- **Port in Use:** If port 8000 is occupied, run with `--port 8080`.
- **Out of VRAM:** The 4-bit NF4 Qwen2.5-7B model requires ~7.4 GB VRAM. If another process is consuming GPU memory, close competing processes or run in `--mock` mode.
- **Server Shutdown:** Press `Ctrl+C` in the terminal to terminate the server.
