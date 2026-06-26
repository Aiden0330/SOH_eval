/* sohbench dashboard — data-driven render of the Lab Light design.
   Loads ./data/leaderboard.json + ./data/qc_report.json (+ optional run_config.json).
   Falls back to embedded sample data when fetch is unavailable (file:// preview). */

"use strict";

// ---- embedded fallback (real cross-domain MAE matrix; superseded by live ./data) ----
const FALLBACK = {
  config: { datasets: ["nasa", "calce_cs2", "hust", "oxford"], protocols: ["lodo"], seeds: [0, 1] },
  leaderboard: [
    e("ridge", "full", 0.4791, { rmse: 0.500, mbe: 0.000, picp: 0.262, ci: "conformal" }),
    e("ridge", "voltage", 0.4105, { ci: "conformal" }),
    e("ridge", "ic_dva", 0.1736, { ci: "conformal" }),
    e("gbdt", "full", 0.2042, { rmse: 0.238, mbe: 0.021, picp: 0.079, ci: "conformal" }),
    e("gbdt", "voltage", 0.1963, { ci: "conformal" }),
    e("gbdt", "ic_dva", 0.1168, { ci: "conformal" }),
    e("mlp", "full", 0.5687, { ci: "conformal" }),
    e("mlp", "voltage", 0.2476, { ci: "conformal" }),
    e("mlp", "ic_dva", 0.1762, { ci: "conformal" }),
    e("phys_resid", "full", 0.2789, { ci: "conformal" }),
  ],
  qc: { pass_rate: 0.9981, n_in: 164775, n_out: 318, drop_reasons: { soh_out_of_range: 318 },
        dropped_columns: ["ambient_temp_c"] },
};
function e(estimator, featureset, mae, o = {}) {
  return {
    estimator, featureset, protocol: "lodo", track: "unified", ci_source: o.ci || "conformal",
    metrics: {
      mae: { mean: mae, std: 0 },
      rmse: o.rmse != null ? { mean: o.rmse } : null,
      mean_bias_error: o.mbe != null ? { mean: o.mbe } : null,
      picp: o.picp != null ? { mean: o.picp } : null,
    },
    verdicts: o.verdicts || { generalization: false, calibration: false, physical_consistency: false },
  };
}

// ---- helpers ----
const $ = (id) => document.getElementById(id);
const FS_ORDER = ["full", "voltage", "ic_dva", "partial_window", "capacity_time"];
const num = (v) => (v == null || Number.isNaN(v) ? null : v);
const mget = (entry, name) => num(entry.metrics?.[name]?.mean);
const fmt = (v, d = 3, sign = false) =>
  v == null ? "—" : (sign && v >= 0 ? "+" : "") + v.toFixed(d);
const uniq = (a) => [...new Set(a)];
function fsOrder(sets) {
  return [...sets].sort((a, b) => (FS_ORDER.indexOf(a) + 1 || 99) - (FS_ORDER.indexOf(b) + 1 || 99));
}

// heatmap tint: low MAE -> green, high -> red (light tint, dark text stays readable)
function heat(t) { // t in [0,1], 0=best
  const h = 140 - 140 * Math.max(0, Math.min(1, t)); // 140(green)->0(red)
  return `hsl(${h} 65% 92%)`;
}

let DATA = null;
let activeFs = null;

async function load() {
  try {
    const [lb, qc, cfg] = await Promise.all([
      fetch("./data/leaderboard.json").then((r) => { if (!r.ok) throw 0; return r.json(); }),
      fetch("./data/qc_report.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
      fetch("./data/run_config.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    DATA = { leaderboard: lb, qc, config: cfg || {} };
  } catch (_) {
    DATA = { leaderboard: FALLBACK.leaderboard, qc: FALLBACK.qc, config: FALLBACK.config, fallback: true };
  }
  render();
}

function render() {
  const lb = DATA.leaderboard || [];
  if (!lb.length) { $("leaderboard").innerHTML = '<div class="state">No data.</div>'; return; }
  const sets = fsOrder(uniq(lb.map((x) => x.featureset || "full")));
  if (!activeFs || !sets.includes(activeFs)) activeFs = sets.includes("ic_dva") ? "ic_dva" : sets[0];
  renderRunMeta(); renderKpis(lb); renderMatrix(lb, sets); renderTabs(sets);
  renderLeaderboard(lb); renderPicp(lb); renderQc(); renderFoot();
}

function renderRunMeta() {
  const c = DATA.config || {};
  const parts = [];
  if (c.datasets) parts.push(`datasets <b>${c.datasets.length}</b>`);
  const protos = c.protocols || uniq(DATA.leaderboard.map((x) => x.protocol));
  if (protos?.length) parts.push(`protocol <b>${protos.join(",")}</b>`);
  if (c.seeds) parts.push(`seeds <b>${c.seeds.length}</b>`);
  parts.push(`source <b>${DATA.fallback ? "sample" : "live"}</b>`);
  $("runmeta").innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
}

function renderKpis(lb) {
  const withMae = lb.filter((x) => mget(x, "mae") != null);
  const best = withMae.reduce((a, b) => (mget(b, "mae") < mget(a, "mae") ? b : a), withMae[0]);
  const algos = uniq(lb.map((x) => x.estimator));
  let pass = 0, total = 0;
  lb.forEach((x) => Object.values(x.verdicts || {}).forEach((v) => { total++; if (v) pass++; }));
  const qc = DATA.qc;
  const kpi = (lab, val, hint, color) =>
    `<div class="kpi"><div class="lab">${lab}</div><div class="val"${color ? ` style="color:${color}"` : ""}>${val}</div><div class="hint">${hint}</div></div>`;
  $("kpis").innerHTML =
    kpi("Best cross-domain", best.estimator, `MAE ${fmt(mget(best, "mae"))} · ${best.featureset}`, "var(--primary)") +
    kpi("Algorithms", algos.length, algos.join(" · ")) +
    kpi("Acceptance pass", `${pass}/${total}`, "gen · cal · phys gates", pass ? "var(--warn)" : "var(--fail)") +
    kpi("Data QC pass", qc ? (100 * qc.pass_rate).toFixed(2) + "%" : "—",
        qc ? `${qc.n_out} rows dropped` : "no qc", "var(--pass)");
}

function renderMatrix(lb, sets) {
  const algos = uniq(lb.map((x) => x.estimator));
  const cell = {};
  lb.forEach((x) => { cell[x.estimator + "|" + (x.featureset || "full")] = mget(x, "mae"); });
  const vals = Object.values(cell).filter((v) => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const best = Math.min(...vals);
  let h = "<table class='matrix'><thead><tr><th>algorithm</th>" +
    sets.map((s) => `<th>${s}</th>`).join("") + "</tr></thead><tbody>";
  algos.forEach((a) => {
    h += `<tr><td>${a}</td>`;
    sets.forEach((s) => {
      const v = cell[a + "|" + s];
      if (v == null) { h += `<td class="na" title="not applicable">N/A</td>`; return; }
      const t = hi > lo ? (v - lo) / (hi - lo) : 0;
      const isBest = Math.abs(v - best) < 1e-9;
      h += `<td class="cell${isBest ? " best" : ""}" style="background:${heat(t)}" title="${a} × ${s}: MAE ${v.toFixed(4)}">${v.toFixed(3)}</td>`;
    });
    h += "</tr>";
  });
  h += "</tbody></table>";
  $("matrix").innerHTML = h;
}

function renderTabs(sets) {
  $("fs-tabs").innerHTML = sets.map((s) =>
    `<button class="tab" role="tab" aria-selected="${s === activeFs}" data-fs="${s}">${s}</button>`).join("");
  $("fs-tabs").querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => { activeFs = t.dataset.fs; render(); }));
}

function renderLeaderboard(lb) {
  $("lb-fs").textContent = activeFs;
  const rows = lb.filter((x) => (x.featureset || "full") === activeFs)
    .filter((x) => mget(x, "mae") != null)
    .sort((a, b) => mget(a, "mae") - mget(b, "mae"));
  const maxMae = Math.max(...rows.map((x) => mget(x, "mae")), 0.001);
  const vdot = (ok) => ok == null ? '<span class="dot na">·</span>' : ok ? '<span class="dot p">✓</span>' : '<span class="dot f">✗</span>';
  let h = "<table class='lb'><thead><tr><th>Algorithm</th><th>MAE</th><th>RMSE</th><th>MBE</th><th>PICP</th><th>CI</th><th>Verdicts</th></tr></thead><tbody>";
  rows.forEach((x, i) => {
    const mae = mget(x, "mae"), w = Math.round((mae / maxMae) * 90);
    const v = x.verdicts || {};
    const ci = x.ci_source || "conformal";
    h += `<tr class="${i === 0 ? "best" : ""}">` +
      `<td><span class="rank">${i + 1}</span><span class="algo">${x.estimator}</span></td>` +
      `<td><div class="bar-cell"><span class="bar ${i === rows.length - 1 && rows.length > 1 ? "bad" : ""}" style="width:${w}px"></span><span class="tnum">${fmt(mae)}</span></div></td>` +
      `<td class="tnum">${fmt(mget(x, "rmse"))}</td>` +
      `<td class="tnum">${fmt(mget(x, "mean_bias_error"), 3, true)}</td>` +
      `<td class="tnum">${fmt(mget(x, "picp"))}</td>` +
      `<td><span class="pill ${ci}">${ci}</span></td>` +
      `<td><span class="v">${vdot(v.generalization)}${vdot(v.calibration)}${vdot(v.physical_consistency)}</span></td>` +
      `</tr>`;
  });
  h += "</tbody></table>";
  $("leaderboard").innerHTML = h;
}

function renderPicp(lb) {
  const rows = lb.filter((x) => (x.featureset || "full") === activeFs && mget(x, "picp") != null)
    .sort((a, b) => mget(b, "picp") - mget(a, "picp"));
  if (!rows.length) { $("picp").innerHTML = '<div class="note">No interval coverage recorded for this feature set.</div>'; return; }
  $("picp").innerHTML = rows.map((x) => {
    const p = mget(x, "picp"), low = p < 0.9;
    return `<div class="picp-row"><span class="nm">${x.estimator}</span>` +
      `<div class="picp-track"><div class="picp-fill ${low ? "low" : ""}" style="width:${(p * 100).toFixed(1)}%"></div>` +
      `<div class="nominal" style="left:90%"></div></div><span class="picp-val">${p.toFixed(3)}</span></div>`;
  }).join("") + '<p class="note">Amber line = nominal 90%. <span class="mono">native</span> = model variance; <span class="mono">conformal</span> = split-conformal (under-covers cross-domain).</p>';
}

function renderQc() {
  const qc = DATA.qc;
  if (!qc) { $("qc").innerHTML = '<div class="note">No QC report.</div>'; return; }
  const line = (k, v, color) => `<div class="qc-line"><span>${k}</span><b class="tnum"${color ? ` style="color:${color}"` : ""}>${v}</b></div>`;
  const reasons = qc.drop_reasons || {};
  $("qc").innerHTML =
    line("Rows in", (qc.n_in ?? 0).toLocaleString()) +
    Object.entries(reasons).map(([k, v]) => line("Dropped — " + k, v, "var(--fail)")).join("") +
    line("Dropped columns", (qc.dropped_columns || []).join(", ") || "none") +
    line("Pass rate", (100 * (qc.pass_rate ?? 1)).toFixed(2) + "%", "var(--pass)");
}

function renderFoot() {
  const algos = uniq(DATA.leaderboard.map((x) => x.estimator)).join(" · ");
  $("foot").textContent = `${DATA.fallback ? "Sample data" : "Live sohbench run"} · ${algos} · leave-one-dataset-out`;
}

load();
