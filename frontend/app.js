/**
 * Paraphrase Sensitivity Mitigation - Scientific Research Demo Frontend
 * Vanilla JavaScript implementation for interactive prompt comparison.
 */

document.addEventListener("DOMContentLoaded", () => {
  // --------------------------------------------------------------------------
  // Application State
  // --------------------------------------------------------------------------
  const state = {
    sessionId: `session_${Date.now()}`,
    currentParaphrases: [],
    baselineResults: null,
    mitigationResults: null,
    metrics: null,
    isExecuting: false,
    systemInfo: null,
  };

  // --------------------------------------------------------------------------
  // Research Benchmark Presets
  // --------------------------------------------------------------------------
  const PRESETS = {
    mod_arith: {
      prompt: "What is the remainder when 7^25 is divided by 13?",
      ref: "7",
      cond: "E4_combined",
    },
    signed_math: {
      prompt: "What is (-3)^3 - (-2)^2?",
      ref: "-31",
      cond: "E4_combined",
    },
    date_duration: {
      prompt: "How many years elapsed between 1776 and 1789?",
      ref: "13",
      cond: "E4_combined",
    },
    capital_city: {
      prompt: "What is the capital city of Australia?",
      ref: "Canberra",
      cond: "E4_combined",
    },
    percentage: {
      prompt: "What is 15% of 240?",
      ref: "36",
      cond: "E4_combined",
    },
    periodic_table: {
      prompt: "What periodic table group is Fluorine in?",
      ref: "Group 17",
      cond: "E4_combined",
    },
    ambiguous_query: {
      prompt: "What is the speed of sound and light?",
      ref: "",
      cond: "E5_conservative",
    },
  };

  // --------------------------------------------------------------------------
  // DOM Elements
  // --------------------------------------------------------------------------
  const promptInput = document.getElementById("promptInput");
  const referenceAnswerInput = document.getElementById("referenceAnswerInput");
  const paraphraseCount = document.getElementById("paraphraseCount");
  const mitigationCondition = document.getElementById("mitigationCondition");
  const btnGenerateParaphrases = document.getElementById("btnGenerateParaphrases");

  const paraphraseCard = document.getElementById("paraphraseCard");
  const paraphraseList = document.getElementById("paraphraseList");
  const paraphraseCountBadge = document.getElementById("paraphraseCountBadge");

  const btnRunBoth = document.getElementById("btnRunBoth");
  const btnRunBaseline = document.getElementById("btnRunBaseline");
  const btnRunMitigation = document.getElementById("btnRunMitigation");
  const btnResetSession = document.getElementById("btnResetSession");

  const progressBarContainer = document.getElementById("progressBarContainer");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");

  const metricsCard = document.getElementById("metricsCard");
  const comparisonCard = document.getElementById("comparisonCard");
  const emptyStateCard = document.getElementById("emptyStateCard");
  const comparisonList = document.getElementById("comparisonList");

  const btnExportJson = document.getElementById("btnExportJson");
  const btnExportCsv = document.getElementById("btnExportCsv");

  const systemStatusDot = document.getElementById("systemStatusDot");
  const systemStatusText = document.getElementById("systemStatusText");
  const modelNameDisplay = document.getElementById("modelNameDisplay");
  const vramDisplay = document.getElementById("vramDisplay");

  const aboutModal = document.getElementById("aboutModal");
  const btnOpenAbout = document.getElementById("btnOpenAbout");
  const btnCloseModal = document.getElementById("btnCloseModal");
  const btnCloseModalBtn = document.getElementById("btnCloseModalBtn");

  // --------------------------------------------------------------------------
  // Initialization & Health Check
  // --------------------------------------------------------------------------
  checkSystemHealth();

  // Preset Chips Listener
  document.getElementById("presetChips").addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) {
      const presetKey = e.target.getAttribute("data-preset");
      const preset = PRESETS[presetKey];
      if (preset) {
        promptInput.value = preset.prompt;
        referenceAnswerInput.value = preset.ref;
        mitigationCondition.value = preset.cond;

        document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
        e.target.classList.add("active");

        // Automatically trigger paraphrase generation on preset click for smooth demo
        generateParaphrases();
      }
    }
  });

  btnGenerateParaphrases.addEventListener("click", generateParaphrases);
  btnRunBoth.addEventListener("click", () => executeInference("both"));
  btnRunBaseline.addEventListener("click", () => executeInference("baseline"));
  btnRunMitigation.addEventListener("click", () => executeInference("mitigation"));
  btnResetSession.addEventListener("click", resetSession);

  btnExportJson.addEventListener("click", () => exportSession("json"));
  btnExportCsv.addEventListener("click", () => exportSession("csv"));

  btnOpenAbout.addEventListener("click", () => (aboutModal.style.display = "flex"));
  btnCloseModal.addEventListener("click", () => (aboutModal.style.display = "none"));
  btnCloseModalBtn.addEventListener("click", () => (aboutModal.style.display = "none"));

  // --------------------------------------------------------------------------
  // System Health
  // --------------------------------------------------------------------------
  async function checkSystemHealth() {
    try {
      const resp = await fetch("/api/health");
      const data = await resp.json();
      state.systemInfo = data;

      if (data.status === "ok" && data.model) {
        const m = data.model;
        modelNameDisplay.textContent = m.model_name.replace("Qwen/", "").replace("-Instruct", "");
        vramDisplay.textContent = `${m.vram_allocated_gb} / ${m.vram_total_gb} GB`;

        if (m.mock_mode) {
          systemStatusDot.className = "status-dot";
          systemStatusText.textContent = "Mock Mode Active";
        } else if (m.is_loaded) {
          systemStatusDot.className = "status-dot online";
          systemStatusText.textContent = "GPU Ready (RTX 4090)";
        } else {
          systemStatusDot.className = "status-dot online";
          systemStatusText.textContent = "Ready (Loads on Run)";
        }
      }
    } catch (err) {
      systemStatusDot.className = "status-dot error";
      systemStatusText.textContent = "Server Offline";
    }
  }

  // --------------------------------------------------------------------------
  // Step 1: Generate Paraphrases
  // --------------------------------------------------------------------------
  async function generateParaphrases() {
    const promptText = promptInput.value.trim();
    if (!promptText) {
      alert("Please enter a canonical prompt first.");
      promptInput.focus();
      return;
    }

    const count = parseInt(paraphraseCount.value, 10) || 5;
    btnGenerateParaphrases.disabled = true;
    btnGenerateParaphrases.innerHTML = `<span class="spinner"></span> Synthesizing Paraphrases...`;

    try {
      const resp = await fetch("/api/paraphrase/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptText, count: count }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || "Failed to generate paraphrases");
      }

      const data = await resp.json();
      state.currentParaphrases = data.paraphrases;
      renderParaphraseList();
    } catch (err) {
      alert(`Error generating paraphrases: ${err.message}`);
    } finally {
      btnGenerateParaphrases.disabled = false;
      btnGenerateParaphrases.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
        Generate Paraphrases
      `;
    }
  }

  function renderParaphraseList() {
    paraphraseList.innerHTML = "";
    paraphraseCountBadge.textContent = `${state.currentParaphrases.length} Variants`;

    state.currentParaphrases.forEach((p, idx) => {
      const item = document.createElement("div");
      item.className = "paraphrase-item";

      const badgeClass = p.is_original ? "axis-badge original" : "axis-badge";
      const valClass = p.is_valid ? "validation-pill valid" : "validation-pill warning";
      const valIcon = p.is_valid ? "✓" : "⚠";

      item.innerHTML = `
        <div class="paraphrase-item-header">
          <span class="${badgeClass}">${escapeHtml(p.axis)}</span>
          <span class="${valClass}" title="${escapeHtml(p.validation_notes)}">
            ${valIcon} ${p.is_valid ? "Valid" : "Equivalence Warning"}
          </span>
        </div>
        <input type="text" class="paraphrase-input" data-index="${idx}" value="${escapeHtml(p.text)}" />
      `;

      paraphraseList.appendChild(item);
    });

    paraphraseCard.style.display = "block";
    paraphraseCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function getApprovedPrompts() {
    const inputs = document.querySelectorAll(".paraphrase-input");
    const approved = [];
    inputs.forEach((input) => {
      const val = input.value.trim();
      if (val) approved.push(val);
    });
    return approved;
  }

  // --------------------------------------------------------------------------
  // Step 2: Execute Inference (Baseline, Mitigation, Both)
  // --------------------------------------------------------------------------
  async function executeInference(mode) {
    const prompts = getApprovedPrompts();
    if (prompts.length === 0) {
      alert("No active prompts found. Please generate or enter at least one prompt.");
      return;
    }

    state.isExecuting = true;
    setExecutionButtonsDisabled(true);
    progressBarContainer.style.display = "block";
    progressFill.style.width = "20%";
    progressText.textContent = `Initializing ${mode.toUpperCase()} inference on ${prompts.length} variants...`;

    const refAns = referenceAnswerInput.value.trim() || null;
    const cond = mitigationCondition.value;

    try {
      if (mode === "both") {
        progressFill.style.width = "40%";
        progressText.textContent = `Executing Baseline & Phase 6 Mitigation sequentially...`;

        const resp = await fetch("/api/inference/run_both", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: state.sessionId,
            prompts: prompts,
            condition: cond,
            reference_answer: refAns,
          }),
        });

        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.error || "Inference execution failed");
        }

        const data = await resp.json();
        state.baselineResults = data.baseline;
        state.mitigationResults = data.mitigation;
        state.metrics = data.metrics;

      } else if (mode === "baseline") {
        progressFill.style.width = "50%";
        progressText.textContent = `Executing Baseline generation across prompts...`;

        const resp = await fetch("/api/inference/baseline", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: state.sessionId,
            prompts: prompts,
            reference_answer: refAns,
          }),
        });

        if (!resp.ok) throw new Error("Baseline execution failed");
        state.baselineResults = await resp.json();

        // If mitigation results exist, re-calculate metrics
        if (state.mitigationResults) {
          await recalculateMetrics(refAns);
        }

      } else if (mode === "mitigation") {
        progressFill.style.width = "50%";
        progressText.textContent = `Executing Phase 6 Multi-Path Mitigation...`;

        const baseCands = state.baselineResults
          ? state.baselineResults.results.map((r) => r.answer)
          : null;

        const resp = await fetch("/api/inference/mitigation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: state.sessionId,
            prompts: prompts,
            baseline_candidates: baseCands,
            condition: cond,
            reference_answer: refAns,
          }),
        });

        if (!resp.ok) throw new Error("Mitigation execution failed");
        state.mitigationResults = await resp.json();

        // If baseline results exist, re-calculate metrics
        if (state.baselineResults) {
          await recalculateMetrics(refAns);
        }
      }

      progressFill.style.width = "100%";
      progressText.textContent = "Inference completed successfully.";
      renderComparisonDashboard();
    } catch (err) {
      alert(`Inference Error: ${err.message}`);
    } finally {
      state.isExecuting = false;
      setExecutionButtonsDisabled(false);
      setTimeout(() => {
        progressBarContainer.style.display = "none";
        progressFill.style.width = "0%";
      }, 1000);
    }
  }

  async function recalculateMetrics(refAns) {
    if (!state.baselineResults || !state.mitigationResults) return;
    try {
      const resp = await fetch("/api/metrics/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline_results: state.baselineResults.results,
          mitigation_results: state.mitigationResults.results,
          reference_answer: refAns,
        }),
      });
      if (resp.ok) {
        state.metrics = await resp.json();
      }
    } catch (e) {
      console.warn("Metrics recalculation failed", e);
    }
  }

  function setExecutionButtonsDisabled(disabled) {
    btnRunBoth.disabled = disabled;
    btnRunBaseline.disabled = disabled;
    btnRunMitigation.disabled = disabled;
    btnGenerateParaphrases.disabled = disabled;
  }

  // --------------------------------------------------------------------------
  // Step 3: Render Comparison Dashboard
  // --------------------------------------------------------------------------
  function renderComparisonDashboard() {
    emptyStateCard.style.display = "none";
    comparisonCard.style.display = "block";
    comparisonList.innerHTML = "";

    const baseResults = state.baselineResults ? state.baselineResults.results : [];
    const mitResults = state.mitigationResults ? state.mitigationResults.results : [];
    const count = Math.max(baseResults.length, mitResults.length);

    for (let i = 0; i < count; i++) {
      const b = baseResults[i] || {};
      const m = mitResults[i] || {};
      const promptText = b.prompt_text || m.prompt_text || `Variant ${i + 1}`;

      const cardItem = document.createElement("div");
      cardItem.className = "comparison-card-item";

      // Header
      const isOriginal = i === 0;
      const tagLabel = isOriginal ? "Original Prompt" : `Paraphrase ${i}`;

      // Correctness Pills
      const bCorrectHtml = renderCorrectnessBadge(b.is_correct);
      const mCorrectHtml = renderCorrectnessBadge(m.is_correct);

      // Semantic relation badge
      const semBadge = renderSemanticRelationBadge(m.semantic_relation);

      // Overhead diff
      const bLat = b.latency_ms || 0;
      const mLat = m.latency_ms || 0;
      const diff = mLat - bLat;
      const diffSign = diff > 0 ? `+${diff.toFixed(1)}` : `${diff.toFixed(1)}`;
      const diffClass = diff > 0 ? "overhead-diff positive" : "overhead-diff neutral";

      // Verification Route Badge
      const routeBadge = m.verification_route
        ? `<span class="badge badge-route">${escapeHtml(m.verification_route)}</span>`
        : "";

      // State badge
      const stateBadge = renderVerificationStateBadge(m.verification_state, m.is_corrected, m.is_abstained);

      // Supporting Passages Accordion
      let evidenceDrawerHtml = "";
      if (m.supporting_passages && m.supporting_passages.length > 0) {
        const passagesHtml = m.supporting_passages
          .map(
            (p) => `
            <div class="passage-snippet">
              <div class="passage-title">${escapeHtml(p.title || "Retrieved Passage")}</div>
              <div>"${escapeHtml(p.passage_text || "")}"</div>
              <div class="provenance-tag">Source: ${escapeHtml(p.provenance || "Local Evidence Index")}</div>
            </div>
          `
          )
          .join("");

        evidenceDrawerHtml = `
          <div class="evidence-drawer">
            <button type="button" class="drawer-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
              <span>View Retrieved Evidence Passages (${m.supporting_passages.length})</span>
              <span>▼</span>
            </button>
            <div class="drawer-content">${passagesHtml}</div>
          </div>
        `;
      } else if (m.deterministic_result && m.deterministic_result.is_eligible) {
        evidenceDrawerHtml = `
          <div class="evidence-drawer">
            <button type="button" class="drawer-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
              <span>Deterministic AST Evaluation Trace</span>
              <span>▼</span>
            </button>
            <div class="drawer-content">
              <div class="passage-snippet">
                <div class="passage-title">Formalized Expression: <code>${escapeHtml(m.deterministic_result.formalized_expression || "")}</code></div>
                <div>Computed Value: <strong>${escapeHtml(m.deterministic_result.computed_value || "")}</strong></div>
                <div class="provenance-tag">Reason: ${escapeHtml(m.deterministic_result.reason || "AST Safe Math Execution")}</div>
              </div>
            </div>
          </div>
        `;
      }

      cardItem.innerHTML = `
        <div class="comp-header">
          <div class="comp-prompt">
            <span class="badge ${isOriginal ? "badge-neutral" : "badge-neutral"}">${tagLabel}</span>
            <span>${escapeHtml(promptText)}</span>
          </div>
          <div>${semBadge}</div>
        </div>

        <div class="comp-columns">
          <!-- Baseline Column -->
          <div class="comp-col">
            <div class="comp-col-header">
              <span class="col-tag baseline">Baseline Model (Qwen2.5-7B)</span>
              <div class="comp-badges">${bCorrectHtml}</div>
            </div>
            <div class="comp-answer">${escapeHtml(b.answer || "(No baseline output)")}</div>
            <div class="comp-meta">
              <span>Latency: <strong>${bLat ? bLat.toFixed(1) : "--"} ms</strong></span>
              <span>Greedy Decoding</span>
            </div>
          </div>

          <!-- Mitigated Column -->
          <div class="comp-col">
            <div class="comp-col-header">
              <span class="col-tag mitigated">Phase 6 Mitigation Framework</span>
              <div class="comp-badges">
                ${stateBadge}
                ${mCorrectHtml}
              </div>
            </div>
            <div class="comp-answer">${escapeHtml(m.answer || "(No mitigation output)")}</div>
            <div class="comp-meta">
              <div class="comp-badges">${routeBadge}</div>
              <div>
                Latency: <strong>${mLat ? mLat.toFixed(1) : "--"} ms</strong>
                <span class="${diffClass}">(${diffSign} ms)</span>
              </div>
            </div>
          </div>
        </div>

        ${evidenceDrawerHtml}
      `;

      comparisonList.appendChild(cardItem);
    }

    // Render Metric Cards
    if (state.metrics) {
      renderMetricsDashboard(state.metrics);
    }
  }

  function renderMetricsDashboard(m) {
    metricsCard.style.display = "block";

    // 1. Consistency
    document.getElementById("metricBaseConsistency").textContent = `${m.consistency.baseline_token_f1_pct}%`;
    document.getElementById("metricMitConsistency").textContent = `${m.consistency.mitigation_token_f1_pct}%`;
    const diffSign = m.consistency.change_pp >= 0 ? `+${m.consistency.change_pp}` : `${m.consistency.change_pp}`;
    document.getElementById("metricConsistencyDiff").textContent = `Pairwise agreement (Δ: ${diffSign} pp)`;

    // 2. Factual Correctness
    if (m.factual_accuracy.evaluated) {
      document.getElementById("metricBaseAcc").textContent = `${m.factual_accuracy.baseline_correct}/${m.factual_accuracy.denominator} (${m.factual_accuracy.baseline_accuracy_pct}%)`;
      document.getElementById("metricMitAcc").textContent = `${m.factual_accuracy.mitigation_correct}/${m.factual_accuracy.denominator} (${m.factual_accuracy.mitigation_accuracy_pct}%)`;
      document.getElementById("metricAccFootnote").textContent = `Reference: "${m.factual_accuracy.reference_answer}"`;
    } else {
      document.getElementById("metricBaseAcc").textContent = "N/A";
      document.getElementById("metricMitAcc").textContent = "N/A";
      document.getElementById("metricAccFootnote").textContent = "No reference answer supplied";
    }

    // 3. Contradictions
    document.getElementById("metricContradictions").textContent = m.relations.contradiction_count;
    document.getElementById("metricContradictionsFootnote").textContent = `${m.relations.same_answer_count} identical, ${m.relations.paraphrase_count} paraphrased, ${m.relations.different_answer_count} differed`;

    // 4. Latency
    document.getElementById("metricBaseLatency").textContent = `${m.latency.median_baseline_ms} ms`;
    document.getElementById("metricMitLatency").textContent = `${m.latency.median_mitigation_ms} ms`;
    document.getElementById("metricOverheadFootnote").textContent = `Avg verification overhead: ${m.latency.mean_verification_overhead_ms} ms`;

    // 5. Pathway Distribution Bar
    const b_det = m.verification_breakdown.deterministic_ast;
    const b_ev = m.verification_breakdown.evidence_retrieval;
    const b_con = m.verification_breakdown.consensus_retained;
    const b_abs = m.verification_breakdown.conservative_abstention;
    const total = m.n_prompts || 1;

    document.getElementById("segDeterministic").style.width = `${(b_det / total) * 100}%`;
    document.getElementById("segEvidence").style.width = `${(b_ev / total) * 100}%`;
    document.getElementById("segConsensus").style.width = `${(b_con / total) * 100}%`;
    document.getElementById("segAbstention").style.width = `${(b_abs / total) * 100}%`;

    document.getElementById("pathwayLegend").innerHTML = `
      <div class="legend-item"><span class="legend-dot seg-deterministic"></span> Deterministic AST: <strong>${b_det}</strong></div>
      <div class="legend-item"><span class="legend-dot seg-evidence"></span> Evidence Grounded: <strong>${b_ev}</strong></div>
      <div class="legend-item"><span class="legend-dot seg-consensus"></span> Consensus Retained: <strong>${b_con}</strong></div>
      <div class="legend-item"><span class="legend-dot seg-abstention"></span> Abstentions: <strong>${b_abs}</strong></div>
    `;

    metricsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderCorrectnessBadge(isCorr) {
    if (isCorr === true) return '<span class="badge badge-success">✓ Correct</span>';
    if (isCorr === false) return '<span class="badge badge-danger">✗ Incorrect</span>';
    return '<span class="badge badge-neutral">Not Evaluated</span>';
  }

  function renderVerificationStateBadge(state, isCorrected, isAbstained) {
    if (isAbstained) return '<span class="badge badge-warning">⚠ Abstained</span>';
    if (isCorrected) return '<span class="badge badge-success">✓ Corrected</span>';
    if (state === "SUPPORTED") return '<span class="badge badge-success">✓ Supported</span>';
    if (state === "CONTRADICTED") return '<span class="badge badge-danger">Contradicted</span>';
    if (state === "AMBIGUOUS_QUESTION") return '<span class="badge badge-warning">Ambiguous</span>';
    return '<span class="badge badge-neutral">Consensus</span>';
  }

  function renderSemanticRelationBadge(rel) {
    if (rel === "identical") return '<span class="badge badge-success">Identical Answers</span>';
    if (rel === "paraphrase") return '<span class="badge badge-route">Equivalent Paraphrase</span>';
    if (rel === "contradiction") return '<span class="badge badge-danger">Contradiction Detected</span>';
    if (rel === "different") return '<span class="badge badge-warning">Answers Differ</span>';
    return "";
  }

  // --------------------------------------------------------------------------
  // Step 4: Export Session
  // --------------------------------------------------------------------------
  async function exportSession(format) {
    if (!state.baselineResults && !state.mitigationResults) {
      alert("No active session data to export. Please run inference first.");
      return;
    }

    const sessionPayload = {
      session_id: state.sessionId,
      prompts: getApprovedPrompts(),
      reference_answer: referenceAnswerInput.value.trim() || null,
      baseline: state.baselineResults,
      mitigation: state.mitigationResults,
      metrics: state.metrics,
    };

    try {
      const resp = await fetch("/api/export/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_data: sessionPayload, format: format }),
      });

      if (!resp.ok) throw new Error("Failed to export session");
      const data = await resp.json();

      // Trigger browser download via download endpoint
      window.location.href = `/api/export/download?filename=${encodeURIComponent(data.filename)}`;
    } catch (err) {
      alert(`Export error: ${err.message}`);
    }
  }

  // --------------------------------------------------------------------------
  // Reset Session
  // --------------------------------------------------------------------------
  function resetSession() {
    state.sessionId = `session_${Date.now()}`;
    state.currentParaphrases = [];
    state.baselineResults = null;
    state.mitigationResults = null;
    state.metrics = null;

    paraphraseCard.style.display = "none";
    metricsCard.style.display = "none";
    comparisonCard.style.display = "none";
    emptyStateCard.style.display = "flex";
    promptInput.value = "";
    referenceAnswerInput.value = "";
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
  }

  // Helper
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
