// PolyHarness Web Studio Application Logic

const SAMPLE_TRAJECTORIES = {
  ecommerce: {
    version: "adp-1.0",
    id: "traj_ecommerce_web_01",
    task: "Find a wireless noise-cancelling headphone under $200 and add the highest rated item to cart.",
    system_prompt: "You are an autonomous shopping assistant. Use available tools to fulfill user requirements.",
    environment: {
      name: "ecommerce_sandbox",
      version: "1.0.0",
      observation_space: "html/axtree/markdown",
      action_space: "function_call",
      harness_id: "browser_agent"
    },
    tools: [
      {
        name: "search_products",
        description: "Query the product catalog with keywords and price limits",
        parameters: {
          query: { name: "query", type: "string", description: "Search query text", required: true },
          max_price: { name: "max_price", type: "number", description: "Maximum price filter in USD", required: false }
        }
      },
      {
        name: "add_to_cart",
        description: "Add a specific product to shopping cart",
        parameters: {
          product_id: { name: "product_id", type: "string", description: "Canonical product ID", required: true },
          quantity: { name: "quantity", type: "integer", description: "Number of units", required: false, default: 1 }
        }
      }
    ],
    steps: [
      {
        step_index: 0,
        thought: "I need to search for wireless noise-cancelling headphones under $200.",
        action_intent: "Search products in catalog",
        tool_calls: [
          {
            id: "call_search_101",
            name: "search_products",
            arguments: { query: "wireless noise-cancelling headphones", max_price: 200 }
          }
        ],
        tool_results: [
          {
            tool_call_id: "call_search_101",
            name: "search_products",
            content: '[{"id": "PROD-882", "name": "AcousticPro ANC", "price": 179.99, "rating": 4.8}]'
          }
        ],
        observation: {
          primary_type: "html",
          raw_content: "<div class='product-list'><div class='card' id='PROD-882'><h3>AcousticPro ANC</h3><span class='price'>$179.99</span><span class='rating'>4.8 stars</span><button id='btn-add-882'>Add to Cart</button></div></div>",
          html_content: "<div class='product-list'><div class='card' id='PROD-882'><h3>AcousticPro ANC</h3><span class='price'>$179.99</span><span class='rating'>4.8 stars</span><button id='btn-add-882'>Add to Cart</button></div></div>",
          markdown: "# Search Results\n- **AcousticPro ANC** ($179.99, Rating: 4.8) [Button: Add to Cart]",
          accessibility_tree: "RootWebArea\n  heading 'Search Results'\n  StaticText 'AcousticPro ANC - $179.99 (4.8 stars)'\n  button 'Add to Cart' [id=btn-add-882]"
        }
      },
      {
        step_index: 1,
        thought: "AcousticPro ANC (PROD-882) is highest rated at 4.8. Adding to cart now.",
        action_intent: "Add top rated item to cart",
        tool_calls: [
          {
            id: "call_cart_102",
            name: "add_to_cart",
            arguments: { product_id: "PROD-882", quantity: 1 }
          }
        ],
        tool_results: [
          {
            tool_call_id: "call_cart_102",
            name: "add_to_cart",
            content: '{"status": "success", "cart_item_count": 1, "total": 179.99}'
          }
        ],
        is_recovery_turn: false
      }
    ],
    outcome: {
      success: true,
      final_answer: "Successfully added AcousticPro ANC ($179.99, 4.8 stars) to cart."
    },
    metadata: {
      source_framework: "polyharness",
      recorder_version: "0.1.0",
      success: true,
      tags: ["ecommerce", "web"]
    }
  },

  sql: {
    version: "adp-1.0",
    id: "traj_sql_data_analyst_02",
    task: "Calculate monthly gross revenue for Q3 2024 from transactions table.",
    system_prompt: "You are a senior data engineer assistant. Verify schemas before running aggregations.",
    environment: {
      name: "analytics_postgres",
      version: "1.0.0",
      observation_space: "json_records",
      action_space: "sql_queries",
      harness_id: "sql_runner"
    },
    tools: [
      {
        name: "describe_table",
        description: "Inspect schema columns and types for a table",
        parameters: {
          table_name: { name: "table_name", type: "string", description: "Target table", required: true }
        }
      },
      {
        name: "execute_sql",
        description: "Run analytical SQL query on PostgreSQL database",
        parameters: {
          query: { name: "query", type: "string", description: "SQL query statement", required: true }
        }
      }
    ],
    steps: [
      {
        step_index: 0,
        thought: "First inspect table schema to verify date and amount column names.",
        action_intent: "Inspect schema",
        tool_calls: [
          {
            id: "call_desc_201",
            name: "describe_table",
            arguments: { table_name: "transactions" }
          }
        ],
        tool_results: [
          {
            tool_call_id: "call_desc_201",
            name: "describe_table",
            content: '{"columns": ["id", "created_at", "amount_cents", "status"]}'
          }
        ]
      },
      {
        step_index: 1,
        thought: "Amounts are in amount_cents. Execute aggregation query grouped by month.",
        action_intent: "Run monthly revenue aggregation",
        tool_calls: [
          {
            id: "call_sql_202",
            name: "execute_sql",
            arguments: {
              query: "SELECT to_char(created_at, 'YYYY-MM') as month, SUM(amount_cents)/100.0 as revenue FROM transactions WHERE created_at >= '2024-07-01' AND created_at < '2024-10-01' AND status = 'settled' GROUP BY 1 ORDER BY 1;"
            }
          }
        ],
        tool_results: [
          {
            tool_call_id: "call_sql_202",
            name: "execute_sql",
            content: '[{"month": "2024-07", "revenue": 142500.0}, {"month": "2024-08", "revenue": 158200.0}, {"month": "2024-09", "revenue": 169800.0}]'
          }
        ]
      }
    ],
    outcome: {
      success: true,
      final_answer: "Q3 2024 Revenue: Jul $142.5k, Aug $158.2k, Sep $169.8k. Total: $470.5k."
    },
    metadata: {
      source_framework: "polyharness",
      recorder_version: "0.1.0",
      success: true,
      tags: ["sql", "analytics"]
    }
  },

  custom: {
    version: "adp-1.0",
    id: "traj_custom_blank",
    task: "Custom task goal description goes here",
    system_prompt: "You are an autonomous agent.",
    environment: { name: "custom_env", harness_id: "generic" },
    tools: [],
    steps: [],
    outcome: { success: true, final_answer: "" },
    metadata: { source_framework: "polyharness", success: true }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const sampleSelect = document.getElementById("sample-select");
  const adpJsonEditor = document.getElementById("adp-json-editor");
  const renderedOutput = document.getElementById("rendered-output-display");
  const harnessTargetSelect = document.getElementById("harness-target-select");
  const btnRenderHarness = document.getElementById("btn-render-harness");
  const btnValidateStudio = document.getElementById("btn-validate-studio");
  const btnCopyAdp = document.getElementById("btn-copy-adp");
  const btnRetestDiag = document.getElementById("btn-retest-diagnostics");
  const btnRunAugment = document.getElementById("btn-run-augment");
  const btnRunEval = document.getElementById("btn-run-eval");
  const evalProfileSelect = document.getElementById("eval-profile-select");
  const btnCompileDataset = document.getElementById("btn-compile-dataset");
  const compilerPreviewCode = document.getElementById("compiler-preview-code");
  const btnDownloadSft = document.getElementById("btn-download-sft");

  // Tab Navigation
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(`panel-${target}`).classList.add("active");

      if (target === "diagnostics") {
        runDiagnosticsAudit();
      } else if (target === "evals") {
        runEvaluationMatrix();
      }
    });
  });

  // Load Initial Sample
  function loadSample(key) {
    const data = SAMPLE_TRAJECTORIES[key] || SAMPLE_TRAJECTORIES.ecommerce;
    adpJsonEditor.value = JSON.stringify(data, null, 2);
    renderTargetHarness();
  }

  sampleSelect.addEventListener("change", (e) => {
    loadSample(e.target.value);
  });

  // Render Target Harness
  async function renderTargetHarness() {
    try {
      const trajectory = JSON.parse(adpJsonEditor.value);
      const harness = harnessTargetSelect.value;
      renderedOutput.innerHTML = "<code>Compiling to " + harness + "...</code>";

      const res = await fetch("/api/trajectories/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trajectory, target_harness: harness })
      });

      if (!res.ok) {
        const err = await res.json();
        renderedOutput.innerHTML = `<code style="color: #f43f5e;">Error: ${err.detail || "Rendering failed"}</code>`;
        return;
      }

      const data = await res.json();
      renderedOutput.innerHTML = `<code>${escapeHtml(JSON.stringify(data.rendered, null, 2))}</code>`;
    } catch (e) {
      renderedOutput.innerHTML = `<code style="color: #f43f5e;">JSON Parse Error: ${e.message}</code>`;
    }
  }

  btnRenderHarness.addEventListener("click", renderTargetHarness);
  harnessTargetSelect.addEventListener("change", renderTargetHarness);

  btnCopyAdp.addEventListener("click", () => {
    navigator.clipboard.writeText(adpJsonEditor.value);
    btnCopyAdp.innerText = "Copied!";
    setTimeout(() => { btnCopyAdp.innerText = "Copy JSON"; }, 2000);
  });

  btnValidateStudio.addEventListener("click", () => {
    document.getElementById("tab-diagnostics-btn").click();
  });

  // Diagnostics Tab Logic
  async function runDiagnosticsAudit() {
    try {
      const trajectory = JSON.parse(adpJsonEditor.value);
      const res = await fetch("/api/trajectories/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trajectory })
      });

      const report = await res.json();

      document.getElementById("diag-quality-score").innerText = report.quality_score;
      document.getElementById("diag-task-name").innerText = report.task || "Generic";
      document.getElementById("diag-steps-count").innerText = `${report.total_steps} steps`;

      const badge = document.getElementById("diag-valid-badge");
      if (report.is_valid) {
        badge.className = "status-badge valid";
        badge.innerText = "Valid Schema";
      } else {
        badge.className = "status-badge error";
        badge.innerText = "Schema Errors";
      }

      // Update 3 Whiteboard Failure Modes
      const risks = report.harness_overfitting_risks;

      // 1. Tool Syntax
      const synScore = risks.tool_syntax_fragility || 0;
      updateRiskCard("syntax", synScore);

      // 2. Observation Format
      const obsScore = risks.observation_format_lockin || 0;
      updateRiskCard("obs", obsScore);

      // 3. Teacher Forcing Cascade
      const casScore = risks.teacher_forcing_cascade_vulnerability || 0;
      updateRiskCard("cascade", casScore);

      // Render Issues
      const list = document.getElementById("diag-issues-list");
      list.innerHTML = "";
      if (report.issues.length === 0) {
        list.innerHTML = "<div style='color: var(--status-emerald); font-size: 12px; font-family: var(--font-mono);'>[PASS] Zero fragility vulnerabilities detected. Trajectory exhibits robust cross-harness properties.</div>";
      } else {
        report.issues.forEach(issue => {
          const div = document.createElement("div");
          div.className = `issue-item ${issue.severity}`;
          div.innerHTML = `
            <div class="issue-title">[${issue.category.toUpperCase()}] ${escapeHtml(issue.message)}</div>
            <div class="issue-rec">Fix: ${escapeHtml(issue.recommendation)}</div>
          `;
          list.appendChild(div);
        });
      }
    } catch (e) {
      console.error(e);
    }
  }

  function updateRiskCard(prefix, score) {
    const fill = document.getElementById(`progress-${prefix}-fill`);
    const tag = document.getElementById(`risk-${prefix}-tag`);
    const pct = Math.round(score * 100);
    fill.style.width = `${pct}%`;

    if (score >= 0.5) {
      tag.className = "risk-badge high";
      tag.innerText = `High Risk (${pct}%)`;
    } else {
      tag.className = "risk-badge low";
      tag.innerText = `Low Risk (${pct}%)`;
    }
  }

  btnRetestDiag.addEventListener("click", runDiagnosticsAudit);

  // Augmentation Tab Logic
  btnRunAugment.addEventListener("click", async () => {
    try {
      const trajectory = JSON.parse(adpJsonEditor.value);
      const container = document.getElementById("variants-container");
      container.innerHTML = "<div class='empty-placeholder'>Synthesizing multi-representation variants...</div>";

      const res = await fetch("/api/trajectories/augment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trajectory,
          enable_syntax_perturbation: document.getElementById("chk-augment-syntax").checked,
          enable_multi_observation: document.getElementById("chk-augment-obs").checked,
          enable_cascade_guard: document.getElementById("chk-augment-cascade").checked
        })
      });

      const data = await res.json();
      container.innerHTML = "";

      Object.entries(data.variants).forEach(([k, variant]) => {
        const card = document.createElement("div");
        card.className = "variant-card";
        const title = k.replace(/_/g, " ").toUpperCase();
        card.innerHTML = `
          <h4>${title}</h4>
          <p style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">${variant.metadata.notes || "Synthetically generated invariant"}</p>
          <div class="variant-preview">${escapeHtml(JSON.stringify(variant, null, 2))}</div>
        `;
        container.appendChild(card);
      });
    } catch (e) {
      alert("Error generating augmentations: " + e.message);
    }
  });

  // Evaluation Matrix Logic
  async function runEvaluationMatrix() {
    try {
      const profile = evalProfileSelect.value;
      const res = await fetch("/api/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_profile: profile,
          native_harness: "openai"
        })
      });

      const report = await res.json();

      document.getElementById("eval-hoc-val").innerText = report.harness_overfitting_coefficient.toFixed(3);
      document.getElementById("eval-native-acc").innerText = `${(report.native_accuracy * 100).toFixed(1)}%`;
      document.getElementById("eval-unseen-acc").innerText = `${(report.unseen_harnesses_average_accuracy * 100).toFixed(1)}%`;
      document.getElementById("eval-chts-val").innerText = report.cross_harness_transfer_score.toFixed(3);
      document.getElementById("eval-cdr-val").innerText = report.cascading_divergence_rate.toFixed(3);

      const hocDesc = document.getElementById("eval-hoc-desc");
      const tag = document.getElementById("eval-verdict-tag");
      const text = document.getElementById("eval-verdict-text");

      if (report.is_production_safe) {
        tag.className = "verdict-stamp pass";
        tag.innerText = "SAFE FOR PRODUCTION (INVARIANT)";
        text.innerText = "Model successfully completed tasks across unseen tool syntaxes and observation formats without cascading errors.";
        hocDesc.innerText = "Negligible transfer penalty across unseen harnesses";
        document.getElementById("eval-hoc-val").style.color = "var(--status-emerald)";
      } else {
        tag.className = "verdict-stamp fail";
        tag.innerText = "FAILED AUDIT: OVERFITTED";
        text.innerText = "Single-harness fine-tuning caused severe trajectory overfitting. The model collapses when tool schemas or observations diverge.";
        hocDesc.innerText = "Catastrophic degradation on unseen harnesses";
        document.getElementById("eval-hoc-val").style.color = "var(--status-rose)";
      }

      // Populate Table
      const tbody = document.getElementById("eval-table-body");
      tbody.innerHTML = "";

      Object.entries(report.detailed_harness_results).forEach(([hId, r]) => {
        const tr = document.createElement("tr");
        const roleStr = r.is_native_training_harness
          ? "<span style='color: var(--text-accent); font-weight: 600; font-family: var(--font-mono); font-size: 11px;'>NATIVE</span>"
          : "<span style='color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;'>UNSEEN EVAL</span>";
        const accPct = (r.accuracy * 100).toFixed(1);
        const statusBadge = r.accuracy >= 0.8
          ? "<span class='risk-badge low'>PASS</span>"
          : "<span class='risk-badge high'>FAIL</span>";

        tr.innerHTML = `
          <td><strong class='monospace'>${hId.toUpperCase()}</strong></td>
          <td>${roleStr}</td>
          <td><span class='monospace'>${accPct}%</span></td>
          <td class='monospace'>${r.malformed_tool_calls}</td>
          <td class='monospace'>${r.unvisited_state_crashes}</td>
          <td class='monospace'>${r.cascading_failure_count}</td>
          <td>${statusBadge}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.error(e);
    }
  }

  btnRunEval.addEventListener("click", runEvaluationMatrix);
  evalProfileSelect.addEventListener("change", runEvaluationMatrix);

  // Compiler Logic
  let lastCompiledJson = "";
  btnCompileDataset.addEventListener("click", async () => {
    try {
      const trajectory = JSON.parse(adpJsonEditor.value);
      const isMixture = document.getElementById("radio-mode-mixture").checked;
      const targetHarness = document.getElementById("export-target-harness").value;
      const format = document.getElementById("export-format").value;

      compilerPreviewCode.innerText = "Compiling SFT records...";

      const res = await fetch("/api/trajectories/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trajectories: [trajectory],
          target_harness: targetHarness,
          is_mixture: isMixture
        })
      });

      const data = await res.json();
      if (format === "jsonl") {
        lastCompiledJson = data.records.map(r => JSON.stringify(r)).join("\n");
      } else {
        lastCompiledJson = JSON.stringify(data.records, null, 2);
      }
      compilerPreviewCode.innerText = lastCompiledJson;
    } catch (e) {
      compilerPreviewCode.innerText = "Compilation error: " + e.message;
    }
  });

  btnDownloadSft.addEventListener("click", () => {
    if (!lastCompiledJson) return;
    const isJsonl = document.getElementById("export-format").value === "jsonl";
    const blob = new Blob([lastCompiledJson], { type: isJsonl ? "application/x-ndjson" : "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = isJsonl ? "polyharness_sft_dataset.jsonl" : "polyharness_sft_dataset.json";
    a.click();
    URL.revokeObjectURL(url);
  });

  // Utility to escape HTML
  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial load
  loadSample("ecommerce");
});
