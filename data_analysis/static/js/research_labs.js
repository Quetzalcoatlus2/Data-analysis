(() => {
  "use strict";

  const LAB_LABELS = {
    forecast: "Forecast",
    anomaly: "Anomaly",
    quality: "Quality",
    "change-points": "Change Points",
    conformal: "Conformal",
    shap: "SHAP",
    multivariate: "Multivariate",
  };

  const LAB_CONFIG = {
    forecast: { controls: ["column", "forecast_pct", "contamination"] },
    anomaly: { controls: ["column", "contamination"] },
    quality: { controls: [] },
    "change-points": { controls: ["column"] },
    conformal: { controls: ["column", "forecast_pct"] },
    shap: { controls: ["column"] },
    multivariate: { controls: [] },
    hub: { controls: [] },
  };

  const fmtNum = (v, digits = 4) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    if (Math.abs(n) >= 100000) return n.toLocaleString();
    return n.toLocaleString(undefined, { maximumFractionDigits: digits });
  };

  const esc = (value) => {
    const s = String(value ?? "");
    return s
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  };

  const qs = (root, selector) => root.querySelector(selector);

  function setStatus(el, text, kind = "info") {
    if (!el) return;
    el.textContent = text || "";
    el.classList.remove("is-error", "is-success");
    if (kind === "error") el.classList.add("is-error");
    if (kind === "success") el.classList.add("is-success");
  }

  function renderSummaryCards(container, items) {
    if (!container) return;
    const safeItems = Array.isArray(items) ? items.filter(Boolean) : [];
    container.innerHTML = safeItems
      .map(
        (item) => `
          <article class="labs-kv">
            <div class="labs-kv-label">${esc(item.label || "Metric")}</div>
            <div class="labs-kv-value">${esc(item.value ?? "—")}</div>
          </article>
        `
      )
      .join("");
  }

  function renderTable(container, rows, preferredColumns) {
    if (!container) return;
    if (!Array.isArray(rows) || rows.length === 0) {
      container.innerHTML = "";
      return;
    }

    const keys = Array.isArray(preferredColumns) && preferredColumns.length
      ? preferredColumns.filter((k) => rows.some((row) => row && Object.prototype.hasOwnProperty.call(row, k)))
      : Array.from(
          rows.reduce((acc, row) => {
            Object.keys(row || {}).forEach((k) => acc.add(k));
            return acc;
          }, new Set())
        );

    const head = keys.map((k) => `<th>${esc(k)}</th>`).join("");
    const body = rows
      .map((row) => {
        const cells = keys
          .map((k) => {
            const raw = row?.[k];
            if (Array.isArray(raw)) {
              return `<td>${raw.map((v) => `<span class="labs-badge">${esc(v)}</span>`).join(" ")}</td>`;
            }
            if (typeof raw === "number") {
              return `<td>${esc(fmtNum(raw, 6))}</td>`;
            }
            return `<td>${esc(raw ?? "")}</td>`;
          })
          .join("");
        return `<tr>${cells}</tr>`;
      })
      .join("");

    container.innerHTML = `
      <table class="labs-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    `;
  }

  function normalizePoints(points) {
    if (!Array.isArray(points)) return [];
    return points
      .map((pt, i) => {
        const y = Number(pt?.y);
        if (!Number.isFinite(y)) return null;
        return {
          x: String(pt?.x ?? i),
          y,
          pos: Number.isFinite(Number(pt?.pos)) ? Number(pt?.pos) : i,
        };
      })
      .filter(Boolean);
  }

  function sparklineSvg(points, options = {}) {
    const normalized = normalizePoints(points);
    if (!normalized.length) {
      return `<div class="labs-kv"><div class="labs-kv-label">No series data</div><div class="labs-kv-value">—</div></div>`;
    }

    const width = 640;
    const height = 140;
    const pad = 12;
    const ys = normalized.map((p) => p.y);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const ySpan = Math.max(yMax - yMin, 1e-9);

    const xAt = (index) => {
      if (normalized.length <= 1) return width / 2;
      return pad + (index / (normalized.length - 1)) * (width - pad * 2);
    };
    const yAt = (value) => height - pad - ((value - yMin) / ySpan) * (height - pad * 2);

    const path = normalized
      .map((point, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(2)} ${yAt(point.y).toFixed(2)}`)
      .join(" ");

    const markers = Array.isArray(options.markerPositions)
      ? options.markerPositions
          .map((position) => {
            const idx = normalized.findIndex((p) => Number(p.pos) >= Number(position));
            if (idx < 0) return "";
            const point = normalized[idx];
            return `<circle class="labs-series-marker" cx="${xAt(idx).toFixed(2)}" cy="${yAt(point.y).toFixed(2)}" r="2.9" />`;
          })
          .join("")
      : "";

    return `
      <svg class="labs-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${esc(options.label || "series")}">
        <path class="labs-series-line ${options.secondary ? "secondary" : ""}" d="${path}" />
        ${markers}
      </svg>
    `;
  }

  function renderCharts(container, charts) {
    if (!container) return;
    const safeCharts = Array.isArray(charts) ? charts.filter(Boolean) : [];
    if (!safeCharts.length) {
      container.innerHTML = "";
      return;
    }
    container.innerHTML = safeCharts
      .map(
        (item) => `
          <article class="labs-chart-card">
            <h3 class="labs-chart-title">${esc(item.title || "Series")}</h3>
            ${item.svg || ""}
          </article>
        `
      )
      .join("");
  }

  function renderWarnings(statusEl, warnings) {
    if (!Array.isArray(warnings) || !warnings.length) return;
    const msg = `Completed with ${warnings.length} warning${warnings.length === 1 ? "" : "s"}: ${warnings.join(" ")}`;
    setStatus(statusEl, msg, "success");
  }

  function buildHubCards(rootData, meta) {
    const cards = [
      {
        key: "forecast",
        description: "Forecast trajectories, backtest metrics, and anomaly overlap.",
      },
      {
        key: "anomaly",
        description: "Outlier detection with severity ranking and score diagnostics.",
      },
      {
        key: "quality",
        description: "Dataset quality scoring, issue flags, and remediation hints.",
      },
      {
        key: "change-points",
        description: "Shift detection for structural breaks and segment transitions.",
      },
      {
        key: "conformal",
        description: "Split-conformal uncertainty bands and empirical coverage.",
      },
      {
        key: "shap",
        description: "Model explainability with SHAP or surrogate fallback.",
      },
      {
        key: "multivariate",
        description: "Correlation, VIF, PCA, and joint anomaly diagnostics.",
      },
    ];

    return cards
      .map((item) => {
        const capability = meta?.capabilities || {};
        let badge = `<span class="labs-badge">Ready</span>`;
        if (item.key === "shap" && !capability.shap) {
          badge = `<span class="labs-badge is-warn">Fallback mode</span>`;
        }
        if (item.key === "change-points" && !capability.ruptures) {
          badge = `<span class="labs-badge is-warn">Baseline detector</span>`;
        }
        const href = `/labs/${encodeURIComponent(rootData.filename)}/${item.key}?display=${encodeURIComponent(rootData.displayName)}&column=${encodeURIComponent(rootData.selectedCol || "")}`;
        return `
          <article class="labs-hub-card">
            <h3>${esc(LAB_LABELS[item.key] || item.key)}</h3>
            <p>${esc(item.description)}</p>
            <div style="margin-top:8px;">${badge}</div>
            <a href="${href}">Open ${esc(LAB_LABELS[item.key] || item.key)} Lab →</a>
          </article>
        `;
      })
      .join("");
  }

  function renderForecast(shell, payload) {
    const summary = qs(shell, "[data-lab-summary]");
    const chart = qs(shell, "[data-lab-chart]");
    const table = qs(shell, "[data-lab-table]");

    const historyStats = payload?.history_stats || {};
    const backtest = payload?.backtest || {};
    renderSummaryCards(summary, [
      { label: "History rows", value: payload?.history_count ?? "—" },
      { label: "Forecast steps", value: payload?.forecast_steps ?? "—" },
      { label: "History mean", value: fmtNum(historyStats.mean) },
      { label: "History std", value: fmtNum(historyStats.std) },
      { label: "Volatility ratio", value: fmtNum(payload?.volatility_ratio) },
      { label: "Backtest MAE", value: fmtNum(backtest?.mae) },
      { label: "Backtest RMSE", value: fmtNum(backtest?.rmse) },
      { label: "Backtest MAPE %", value: fmtNum(backtest?.mape) },
    ]);

    renderCharts(chart, [
      {
        title: "History",
        svg: sparklineSvg(payload?.series?.history, { label: "History series" }),
      },
      {
        title: "Forecast",
        svg: sparklineSvg(payload?.series?.forecast, { label: "Forecast series", secondary: true }),
      },
    ]);

    const anomalyRows = payload?.anomalies?.rows || [];
    renderTable(table, anomalyRows, ["index", "value", "score", "pos"]);
  }

  function renderAnomaly(shell, payload) {
    const summary = qs(shell, "[data-lab-summary]");
    const chart = qs(shell, "[data-lab-chart]");
    const table = qs(shell, "[data-lab-table]");

    renderSummaryCards(summary, [
      { label: "Detected anomalies", value: payload?.count ?? 0 },
      { label: "Contamination", value: fmtNum(payload?.contamination, 4) },
      { label: "Score p90", value: fmtNum(payload?.score_stats?.p90) },
      { label: "Score p95", value: fmtNum(payload?.score_stats?.p95) },
    ]);

    const markerPositions = (payload?.anomalies || []).map((row) => row?.pos).filter((v) => Number.isFinite(Number(v)));
    renderCharts(chart, [
      {
        title: "Series with anomaly markers",
        svg: sparklineSvg(payload?.series, { markerPositions, label: "Anomaly series" }),
      },
    ]);

    renderTable(table, payload?.anomalies || [], ["index", "value", "score", "robust_z", "severity", "direction"]);
  }

  function renderQuality(shell, payload) {
    const summary = qs(shell, "[data-lab-summary]");
    const chart = qs(shell, "[data-lab-chart]");
    const table = qs(shell, "[data-lab-table]");

    const q = Number(payload?.quality_score);
    const barVal = Number.isFinite(q) ? Math.max(0, Math.min(100, q)) : 0;

    renderSummaryCards(summary, [
      { label: "Quality score", value: Number.isFinite(q) ? `${fmtNum(q, 2)} / 100` : "—" },
      { label: "Rows", value: payload?.summary?.rows ?? "—" },
      { label: "Columns", value: payload?.summary?.columns ?? "—" },
      { label: "Missing rate", value: payload?.summary?.missing_rate != null ? `${fmtNum(Number(payload.summary.missing_rate) * 100, 2)}%` : "—" },
      { label: "Duplicate rows", value: payload?.summary?.duplicate_rows ?? "—" },
      { label: "Constant cols", value: payload?.summary?.constant_columns ?? "—" },
    ]);

    renderCharts(chart, [
      {
        title: "Quality score",
        svg: `
          <div class="labs-kv">
            <div class="labs-kv-label">Quality progress</div>
            <div style="height:12px;border-radius:999px;background:#1a273b;border:1px solid #2e4464;overflow:hidden;">
              <div style="height:100%;width:${barVal}%;background:linear-gradient(90deg,#ef4444,#f59e0b,#22c55e);"></div>
            </div>
            <div style="margin-top:8px;color:#b6c6de;font-size:0.86rem;">${esc(payload?.recommendations?.[0] || "")}</div>
          </div>
        `,
      },
    ]);

    renderTable(table, payload?.issue_columns || [], ["column", "dtype", "missing_pct", "non_null", "unique", "issues"]);
  }

  function renderChangePoints(shell, payload) {
    const summary = qs(shell, "[data-lab-summary]");
    const chart = qs(shell, "[data-lab-chart]");
    const table = qs(shell, "[data-lab-table]");

    const points = payload?.change_points || [];
    renderSummaryCards(summary, [
      { label: "Column", value: payload?.column ?? "—" },
      { label: "Window", value: payload?.window ?? "—" },
      { label: "Change points", value: points.length },
      { label: "Segments", value: Array.isArray(payload?.segments) ? payload.segments.length : 0 },
    ]);

    const markerPositions = points.map((row) => row?.pos).filter((v) => Number.isFinite(Number(v)));
    renderCharts(chart, [
      {
        title: "Series with change-point markers",
        svg: sparklineSvg(payload?.series, { markerPositions, label: "Change-point series" }),
      },
    ]);

    const mergedRows = [
      ...(payload?.change_points || []),
      ...(payload?.segments || []).map((segment) => ({
        segment: segment.segment,
        start_pos: segment.start_pos,
        end_pos: segment.end_pos,
        mean: segment.mean,
        std: segment.std,
      })),
    ];
    renderTable(table, mergedRows, ["index", "pos", "value", "score", "sources", "segment", "start_pos", "end_pos", "mean", "std"]);
  }

  function renderConformal(shell, payload) {
    const summary = qs(shell, "[data-lab-summary]");
    const chart = qs(shell, "[data-lab-chart]");
    const table = qs(shell, "[data-lab-table]");

    renderSummaryCards(summary, [
      { label: "Column", value: payload?.column ?? "—" },
      { label: "Calibration rows", value: payload?.calibration_size ?? "—" },
      { label: "MAE (calibration)", value: fmtNum(payload?.residual_stats?.mean_abs_error) },
      { label: "p90 abs error", value: fmtNum(payload?.residual_stats?.p90_abs_error) },
      { label: "Forecast points", value: Array.isArray(payload?.forecast) ? payload.forecast.length : 0 },
    ]);

    const level90 = payload?.bands?.["90"] || [];
    renderCharts(chart, [
      {
        title: "Conformal forecast",
        svg: sparklineSvg(payload?.forecast, { label: "Conformal forecast", secondary: true }),
      },
      {
        title: "90% lower band",
        svg: sparklineSvg(level90.map((p) => ({ ...p, y: p?.lower })), { label: "Lower band" }),
      },
      {
        title: "90% upper band",
        svg: sparklineSvg(level90.map((p) => ({ ...p, y: p?.upper })), { label: "Upper band", secondary: true }),
      },
    ]);

    renderTable(table, payload?.levels || [], ["level", "target", "empirical", "quantile"]);
  }

  function renderShap(shell, payload) {
    const summary = qs(shell, "[data-lab-summary]");
    const chart = qs(shell, "[data-lab-chart]");
    const table = qs(shell, "[data-lab-table]");

    renderSummaryCards(summary, [
      { label: "Target", value: payload?.column ?? "—" },
      { label: "Mode", value: payload?.mode ?? "—" },
      { label: "Rows used", value: payload?.rows_used ?? "—" },
      { label: "Feature rows", value: Array.isArray(payload?.feature_importance) ? payload.feature_importance.length : 0 },
    ]);

    const importanceRows = payload?.feature_importance || [];
    const pseudoSeries = importanceRows.map((row, idx) => ({ x: row.feature || idx, y: Number(row.importance || 0), pos: idx }));
    renderCharts(chart, [
      {
        title: "Importance profile",
        svg: sparklineSvg(pseudoSeries, { label: "Feature importance" }),
      },
    ]);

    renderTable(table, importanceRows, ["feature", "importance", "shap_importance", "permutation_importance", "model_importance"]);
  }

  function renderMultivariate(shell, payload) {
    const summary = qs(shell, "[data-lab-summary]");
    const chart = qs(shell, "[data-lab-chart]");
    const table = qs(shell, "[data-lab-table]");

    renderSummaryCards(summary, [
      { label: "Numeric columns", value: Array.isArray(payload?.numeric_columns) ? payload.numeric_columns.length : 0 },
      { label: "Top correlations", value: Array.isArray(payload?.top_correlations) ? payload.top_correlations.length : 0 },
      { label: "VIF rows", value: Array.isArray(payload?.vif) ? payload.vif.length : 0 },
      { label: "Joint anomalies", value: Array.isArray(payload?.joint_anomalies) ? payload.joint_anomalies.length : 0 },
      { label: "PCA components", value: payload?.pca?.components ?? "—" },
    ]);

    const pcaSeries = (payload?.pca?.explained_variance_ratio || []).map((v, idx) => ({ x: `PC${idx + 1}`, y: Number(v || 0), pos: idx }));
    renderCharts(chart, [
      {
        title: "PCA explained variance ratio",
        svg: sparklineSvg(pcaSeries, { label: "PCA variance" }),
      },
    ]);

    const tableRows = [
      ...(payload?.top_correlations || []).slice(0, 20),
      ...(payload?.vif || []).slice(0, 10),
      ...(payload?.joint_anomalies || []).slice(0, 10),
    ];
    renderTable(table, tableRows, ["feature_a", "feature_b", "pearson", "spearman", "abs_corr", "feature", "vif", "index", "distance"]);
  }

  function renderLabByKey(shell, labKey, payload) {
    const table = qs(shell, "[data-lab-table]");
    if (table) table.innerHTML = "";

    switch (labKey) {
      case "forecast":
        renderForecast(shell, payload);
        break;
      case "anomaly":
        renderAnomaly(shell, payload);
        break;
      case "quality":
        renderQuality(shell, payload);
        break;
      case "change-points":
        renderChangePoints(shell, payload);
        break;
      case "conformal":
        renderConformal(shell, payload);
        break;
      case "shap":
        renderShap(shell, payload);
        break;
      case "multivariate":
        renderMultivariate(shell, payload);
        break;
      default:
        break;
    }
  }

  async function fetchJson(url) {
    const resp = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    });
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok || payload?.ok === false) {
      const msg = payload?.error || payload?.message || `Request failed (${resp.status})`;
      throw new Error(msg);
    }
    return payload;
  }

  function hydrateControls(shell, metaData, defaults, labKey) {
    const config = LAB_CONFIG[labKey] || { controls: [] };
    const allowedControls = new Set(config.controls || []);

    shell.querySelectorAll("[data-control]").forEach((group) => {
      const key = String(group.getAttribute("data-control") || "");
      group.style.display = allowedControls.has(key) ? "" : "none";
    });

    const columnSelect = qs(shell, "[data-lab-control='column']");
    if (columnSelect) {
      const numericCols = Array.isArray(metaData?.numeric_columns) ? metaData.numeric_columns : [];
      columnSelect.innerHTML = numericCols
        .map((column) => `<option value="${esc(column)}">${esc(column)}</option>`)
        .join("");
      const selected = defaults.selectedCol || metaData?.selected_col || numericCols[0] || "";
      if (selected) columnSelect.value = selected;
      if (!numericCols.length) {
        columnSelect.innerHTML = "<option value=''>No numeric columns</option>";
      }
    }

    const forecastInput = qs(shell, "[data-lab-control='forecast_pct']");
    if (forecastInput && defaults.forecastPct) {
      forecastInput.value = defaults.forecastPct;
    }

    const contaminationInput = qs(shell, "[data-lab-control='contamination']");
    if (contaminationInput && defaults.contamination) {
      contaminationInput.value = defaults.contamination;
    }
  }

  function buildRunParams(shell, defaults) {
    const params = new URLSearchParams();
    const column = qs(shell, "[data-lab-control='column']")?.value || defaults.selectedCol || "";
    const forecastPct = qs(shell, "[data-lab-control='forecast_pct']")?.value || defaults.forecastPct || "0.05";
    const contamination = qs(shell, "[data-lab-control='contamination']")?.value || defaults.contamination || "0.02";

    if (column) params.set("column", column);
    if (forecastPct) params.set("forecast_pct", forecastPct);
    if (contamination) params.set("contamination", contamination);
    return params;
  }

  async function runLab(rootData, shell, labKey) {
    const statusEl = qs(shell, "[data-lab-status]");
    setStatus(statusEl, `Running ${LAB_LABELS[labKey] || labKey} diagnostics…`);

    const params = buildRunParams(shell, rootData);
    const endpoint = `/api/labs/${encodeURIComponent(rootData.filename)}/${encodeURIComponent(labKey)}?${params.toString()}`;

    const envelope = await fetchJson(endpoint);
    renderLabByKey(shell, labKey, envelope?.data || {});

    if (Array.isArray(envelope?.warnings) && envelope.warnings.length) {
      renderWarnings(statusEl, envelope.warnings);
    } else {
      setStatus(statusEl, `${LAB_LABELS[labKey] || labKey} diagnostics ready.`, "success");
    }
  }

  function getRootData(root) {
    return {
      filename: root?.dataset?.filename || "",
      displayName: root?.dataset?.displayName || "",
      selectedCol: root?.dataset?.selectedCol || "",
      activeLab: root?.dataset?.activeLab || "hub",
      forecastPct: root?.dataset?.forecastPct || "0.05",
      contamination: root?.dataset?.contamination || "0.02",
    };
  }

  async function initHub(rootData, shell, metaEnvelope) {
    const statusEl = qs(shell, "[data-lab-status]");
    const summaryEl = qs(shell, "[data-lab-summary]");
    const contentEl = qs(shell, "[data-lab-content]");

    const meta = metaEnvelope?.data || {};
    const ds = meta.dataset || {};

    renderSummaryCards(summaryEl, [
      { label: "Rows", value: ds.rows ?? "—" },
      { label: "Columns", value: ds.columns ?? "—" },
      { label: "Numeric columns", value: ds.numeric_columns ?? "—" },
      { label: "Missing rate", value: ds.missing_rate != null ? `${fmtNum(Number(ds.missing_rate) * 100, 2)}%` : "—" },
      { label: "Duplicate rows", value: ds.duplicate_rows ?? "—" },
      { label: "Selected numeric column", value: meta.selected_col || "—" },
    ]);

    if (contentEl) {
      contentEl.innerHTML = buildHubCards(rootData, meta);
    }
    setStatus(statusEl, "Research Labs modules are ready.", "success");
  }

  async function init() {
    const root = document.getElementById("labs-root");
    const shell = document.querySelector(".labs-shell");
    if (!root || !shell) return;

    const rootData = getRootData(root);
    if (!rootData.filename) return;

    const labKey = shell.getAttribute("data-lab-page") || rootData.activeLab || "hub";
    const statusEl = qs(shell, "[data-lab-status]");

    try {
      setStatus(statusEl, "Loading dataset metadata…");
      const metaParams = new URLSearchParams();
      if (rootData.selectedCol) metaParams.set("column", rootData.selectedCol);
      const metaEnvelope = await fetchJson(`/api/labs/${encodeURIComponent(rootData.filename)}/meta?${metaParams.toString()}`);

      if (labKey === "hub") {
        await initHub(rootData, shell, metaEnvelope);
        return;
      }

      hydrateControls(shell, metaEnvelope?.data || {}, rootData, labKey);

      const runBtn = qs(shell, "[data-lab-action='run']");
      if (runBtn) {
        runBtn.addEventListener("click", async () => {
          try {
            await runLab(rootData, shell, labKey);
          } catch (err) {
            setStatus(statusEl, err?.message || "Failed to run lab diagnostics.", "error");
          }
        });
      }

      await runLab(rootData, shell, labKey);
    } catch (err) {
      setStatus(statusEl, err?.message || "Failed to initialize Research Labs.", "error");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
