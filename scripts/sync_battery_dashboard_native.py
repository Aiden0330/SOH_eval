from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from sohbench.frontend_publish import (
    _publish_directory,
    resolve_published_directory,
    resolve_safe_publication_asset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"D:\battery_SOH\dashboard")
DEFAULT_TARGET = ROOT / "frontend" / "app" / "battery-dashboard"
SAFE_SCENARIO_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
RESERVED_SCENARIO_NAMES = frozenset(
    {".generations", "current", "current.json"}
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_names(manifest: dict) -> list[str]:
    names: list[str] = []
    for item in manifest.get("scenarios") or []:
        raw_name = item.get("mode") or item.get("id")
        name = "" if raw_name is None else str(raw_name)
        if (
            not SAFE_SCENARIO_NAME_RE.fullmatch(name)
            or name.casefold() in RESERVED_SCENARIO_NAMES
        ):
            raise ValueError(f"invalid scenario name: {raw_name!r}")
        names.append(name)
    return names


def _required_assets(scenarios: list[str], *, include_curves: bool, include_detection: bool) -> list[str]:
    required = ["manifest.json", "overview.json"]
    for name in scenarios:
        required.append(f"{name}.json")
        if include_curves:
            required.append(f"{name}_curves.json")
    if include_detection:
        required.extend(["ess_detection.json", "ess_detection_curves.json"])
    return required


def _validate_json_assets(data_dir: Path, filenames: list[str]) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for filename in filenames:
        path = resolve_safe_publication_asset(
            data_dir, filename, label="dashboard asset path"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source dashboard asset: {path}")
        _read_json(path)
        assets[filename] = path
    return assets


LINKED_CONTROL_CSS = """  .linked-select { max-width:220px; }
  #linkcontext { flex-basis:100%; margin-top:-4px; }"""

LINKED_CONTROL_HTML = """  <label>数据集 <select id="dataset" class="linked-select"></select></label>
  <label>场景 <select id="scenario" class="linked-select"></select></label>
  <label>算法 <select id="algorithm" class="linked-select"></select></label>
  <label>特征 <select id="featureset" class="linked-select"></select></label>"""

LINKED_CONTEXT_JS = """const DATASET_ALIASES = Object.freeze({
  calce_cs2: "calce",
  rwth_isea_public: "rwth",
  rwth: "rwth",
  nasa: "nasa",
  hust: "hust",
  oxford: "oxford",
  snl_lfp: "snl_lfp",
  uconn: "uconn",
  xjtu: "xjtu",
  mit: "mit",
});
const linkedContext = {
  dataset: canonicalDataset(params.get("dataset") || ""),
  source: params.get("source") || "",
  model: params.get("model") || "",
  featureset: params.get("featureset") || "",
};"""

LINKED_HELPERS_JS = """function htmlEscape(v){ return String(v ?? "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch])); }
function canonicalDataset(value){
  const key = String(value || "").trim().toLowerCase();
  return DATASET_ALIASES[key] || key;
}
function scenarioForDataset(dataset){
  const canonical = canonicalDataset(dataset);
  const scenarios = (state.manifest && state.manifest.scenarios) || [];
  return scenarios.some(s => s.mode === canonical) ? canonical : "";
}
function datasetForScenario(mode){
  if(!mode || mode === "overview") return "";
  return mode === "calce" ? "calce_cs2" : (mode === "rwth" ? "rwth_isea_public" : mode);
}
function sourceLabel(){
  return state.linkedContext.source === "inference" ? "推理页" :
    (state.linkedContext.source === "training" ? "训练页" : "前序页");
}
function uniqueRows(rows){
  const seen = new Set();
  return rows.filter(row => {
    const value = String(row.value || row.label || "");
    if(!value || seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}
function selectedAlgorithmRows(){
  const rows = [];
  if(state.linkedContext.model) rows.push({ value: state.linkedContext.model, label: `前序${sourceLabel()}：${state.linkedContext.model}` });
  const overviewScenario = state.overview && (state.overview.scenarios || []).find(s => s.mode === state.scenario);
  (overviewScenario?.algorithms || []).forEach(a => rows.push({ value: a.id || a.label, label: a.label || a.id }));
  const nativeModel = state.data?.config_brief?.model || state.data?.config?.model;
  if(nativeModel) rows.push({ value: nativeModel, label: nativeModel });
  return uniqueRows(rows.length ? rows : [{ value: "native_soh_model", label: "原生 SOH 评估模型" }]);
}
function selectedFeatureRows(){
  const rows = [];
  if(state.linkedContext.featureset) rows.push({ value: state.linkedContext.featureset, label: `前序特征：${state.linkedContext.featureset}` });
  const overviewScenario = state.overview && (state.overview.scenarios || []).find(s => s.mode === state.scenario);
  const featureColumns = overviewScenario?.features?.columns || state.data?.feature_columns || [];
  const win = overviewScenario?.features?.window || state.data?.feature_window;
  const label = win ? `原生窗口特征 ${win[0]}-${win[1]} V` : `原生特征列 ${featureColumns.length || 0}`;
  rows.push({ value: "native_features", label });
  return uniqueRows(rows);
}
function renderSelect(id, rows, selectedValue){
  const node = $(id);
  if(!node) return;
  node.innerHTML = rows.map(row =>
    `<option value="${htmlEscape(row.value)}">${htmlEscape(row.label)}</option>`
  ).join("");
  if(rows.some(row => row.value === selectedValue)) node.value = selectedValue;
}
function renderLinkedSelectors(){
  if(!state.manifest) return;
  const scenarios = state.manifest.scenarios || [];
  const datasetRows = [{ value:"", label:"全部数据集" }].concat(
    scenarios.map(s => ({ value: datasetForScenario(s.mode), label: `${datasetForScenario(s.mode)} · ${s.title}` }))
  );
  renderSelect("dataset", datasetRows, datasetForScenario(state.scenario));
  renderSelect("algorithm", selectedAlgorithmRows(), state.linkedContext.model || "");
  renderSelect("featureset", selectedFeatureRows(), state.linkedContext.featureset || "native_features");
  const context = $("linkcontext");
  if(context){
    const parts = [];
    if(state.linkedContext.source) parts.push(`来源：${sourceLabel()}`);
    if(state.linkedContext.dataset) parts.push(`数据集：${state.linkedContext.dataset}`);
    if(state.linkedContext.model) parts.push(`模型：${state.linkedContext.model}`);
    if(state.linkedContext.featureset) parts.push(`特征：${state.linkedContext.featureset}`);
    context.textContent = parts.length
      ? `已关联前序测评组合｜${parts.join("｜")}`
      : "可在看板内切换数据集、场景、算法和特征；算法/特征选择用于对齐前序测评组合。";
  }
}
function applyLinkedContext(){
  const direct = scenarioForDataset(state.linkedContext.dataset);
  if(direct) return direct;
  const text = [state.linkedContext.model, state.linkedContext.featureset]
    .map(v => canonicalDataset(v))
    .join(" ");
  return ((state.manifest && state.manifest.scenarios) || []).find(s => text.includes(s.mode))?.mode || "";
}"""

LINKED_LISTENERS_JS = """$("dataset").addEventListener("change", e => {
  state.linkedContext.dataset = canonicalDataset(e.target.value);
  const next = scenarioForDataset(e.target.value);
  if(!next) return loadOverview();
  $("scenario").value = next;
  return loadScenario(next);
});
$("algorithm").addEventListener("change", e => {
  state.linkedContext.model = e.target.value;
  renderLinkedSelectors();
});
$("featureset").addEventListener("change", e => {
  state.linkedContext.featureset = e.target.value === "native_features" ? "" : e.target.value;
  renderLinkedSelectors();
});"""


def _patch_linked_controls(html: str) -> str:
    if ".linked-select" not in html:
        select_anchor = (
            "  select { background:var(--panel); color:var(--text); border:1px solid var(--line); "
            "border-radius:6px; padding:6px 10px; font-size:14px; max-width:100%; }"
        )
        if select_anchor in html:
            html = html.replace(select_anchor, select_anchor + "\n" + LINKED_CONTROL_CSS, 1)
        elif "</style>" in html:
            html = html.replace("</style>", LINKED_CONTROL_CSS + "\n</style>", 1)

    if 'id="dataset"' not in html:
        scenario_label = '  <label>数据源 <select id="scenario"></select></label>'
        if scenario_label in html:
            html = html.replace(scenario_label, LINKED_CONTROL_HTML, 1)
        elif '<select id="scenario"></select>' in html:
            html = html.replace(
                '<select id="scenario"></select>',
                '<select id="scenario" class="linked-select"></select>',
                1,
            )
            html = html.replace('<label>数据源 ', '<label>场景 ', 1)
            if '<span class="modetabs">' in html:
                html = html.replace('</span>', '</span>\n' + LINKED_CONTROL_HTML, 1)
        elif '<span class="modetabs">' in html:
            html = html.replace('</span>', '</span>\n' + LINKED_CONTROL_HTML, 1)

    if 'id="linkcontext"' not in html:
        exp_anchor = '<span id="expname" class="hint"></span>'
        link = '<span id="linkcontext" class="hint"></span>'
        if exp_anchor in html:
            html = html.replace(exp_anchor, exp_anchor + "\n  " + link, 1)
        elif "</header>" in html:
            html = html.replace("</header>", "  " + link + "\n</header>", 1)

    if 'params.get("dataset")' not in html:
        anchor = 'const embedded = params.get("embed") === "1";'
        if anchor in html:
            html = html.replace(anchor, anchor + "\n" + LINKED_CONTEXT_JS, 1)

    if "linkedContext };" not in html and "linkedContext }" not in html:
        html = html.replace("detCurvesUrl:null };", "detCurvesUrl:null, linkedContext };")

    if "function canonicalDataset" not in html:
        fetch_anchor = 'async function fetchJSON(path){ const r = await fetch(path); if(!r.ok) throw new Error(path+" "+r.status); return r.json(); }'
        if fetch_anchor in html:
            html = html.replace(fetch_anchor, fetch_anchor + "\n" + LINKED_HELPERS_JS, 1)
        elif LINKED_CONTEXT_JS in html:
            html = html.replace(LINKED_CONTEXT_JS, LINKED_CONTEXT_JS + "\n" + LINKED_HELPERS_JS, 1)

    if "renderOverview(); renderLinkedSelectors();" not in html:
        html = html.replace("renderOverview(); }", "renderOverview(); renderLinkedSelectors(); }")
    if "state.detection.title;\n  renderLinkedSelectors();" not in html:
        html = html.replace(
            '$("expname").style.color = ""; $("expname").textContent = state.detection.title;',
            '$("expname").style.color = ""; $("expname").textContent = state.detection.title;\n  renderLinkedSelectors();',
        )
    if "drawIC(null);\n  renderLinkedSelectors();" not in html:
        html = html.replace(
            'drawSOH(); drawCurve("cv"); drawCurve("ci"); drawCurve("ct"); drawIC(null);',
            'drawSOH(); drawCurve("cv"); drawCurve("ci"); drawCurve("ct"); drawIC(null);\n  renderLinkedSelectors();',
        )
    html = html.replace(
        "const def = state.manifest.scenarios[0].mode;",
        "const def = applyLinkedContext() || state.manifest.scenarios[0].mode;",
    )
    if '$("dataset").addEventListener("change"' not in html:
        if '$("scenario").addEventListener("change"' in html:
            html = html.replace(
                '$("scenario").addEventListener("change"',
                LINKED_LISTENERS_JS + '\n$("scenario").addEventListener("change"',
                1,
            )
        elif "</script>" in html:
            html = html.replace("</script>", LINKED_LISTENERS_JS + "\n</script>", 1)
    if "state.linkedContext.dataset = datasetForScenario(e.target.value);" not in html:
        html = html.replace(
            '$("scenario").addEventListener("change", e => {\n',
            '$("scenario").addEventListener("change", e => {\n  state.linkedContext.dataset = datasetForScenario(e.target.value);\n',
        )

    return html


def _patch_embedded_entry(html: str) -> str:
    if "URLSearchParams(location.search)" not in html:
        anchor = "const $ = id => document.getElementById(id);"
        if anchor not in html:
            raise RuntimeError("Could not find dashboard script anchor for URL parameter patch")
        html = html.replace(
            anchor,
            "\n".join(
                [
                    anchor,
                    "const params = new URLSearchParams(location.search);",
                    'const initialMode = params.get("mode") === "detection" ? "detection" : "soh";',
                    'const embedded = params.get("embed") === "1";',
                    'if(embedded) document.body.classList.add("embedded");',
                ]
            ),
            1,
        )

    if "body.embedded .modetabs" not in html:
        style_anchor = "  footer { color:var(--dim); font-size:12px; margin-top:14px; }"
        embedded_css = "\n".join(
            [
                "  body.embedded { padding:0; }",
                "  body.embedded header { padding:12px 16px; margin-bottom:0; }",
                "  body.embedded .modetabs { display:none; }",
                "  body.embedded footer { padding:0 16px 12px; }",
            ]
        )
        if style_anchor in html:
            html = html.replace(style_anchor, embedded_css + "\n" + style_anchor, 1)
        elif "</style>" in html:
            html = html.replace("</style>", embedded_css + "\n</style>", 1)
        else:
            raise RuntimeError("Could not find dashboard style block for embedded CSS patch")

    html = html.replace('await setMode("soh");', "await setMode(initialMode);")
    return _patch_linked_controls(html)


def sync_native_dashboard(
    source: Path = DEFAULT_SOURCE,
    target: Path = DEFAULT_TARGET,
    *,
    include_curves: bool = True,
    include_detection: bool = False,
) -> dict:
    source = Path(source)
    target = Path(target)
    source = resolve_published_directory(source)
    source_data = source / "data"

    if not (source / "index.html").is_file():
        raise FileNotFoundError(f"Missing source dashboard index.html: {source / 'index.html'}")
    if not source_data.is_dir():
        raise FileNotFoundError(f"Missing source dashboard data directory: {source_data}")
    if not (source_data / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing required source dashboard asset: {source_data / 'manifest.json'}")

    manifest = _read_json(source_data / "manifest.json")
    scenarios = _scenario_names(manifest)
    if not scenarios:
        raise RuntimeError("Source manifest has no scenarios")
    html = _patch_embedded_entry((source / "index.html").read_text(encoding="utf-8"))
    required = _required_assets(
        scenarios,
        include_curves=include_curves,
        include_detection=include_detection,
    )
    source_assets = _validate_json_assets(source_data, required)

    manifest_payload = {
        "source": str(source),
        "target": str(target),
        "scenario_count": len(scenarios),
        "include_soh_curves": bool(include_curves),
        "excluded_detection_assets": not include_detection,
        "copied_files": list(required),
    }

    def build(staging: Path) -> None:
        staging_data = staging / "data"
        staging_data.mkdir(parents=True)
        destinations = {
            filename: resolve_safe_publication_asset(
                staging_data, filename, label="staged dashboard asset path"
            )
            for filename in required
        }
        (staging / "index.html").write_text(html, encoding="utf-8", newline="\n")
        for filename in required:
            shutil.copy2(source_assets[filename], destinations[filename])
        (staging / "SYNC_MANIFEST.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        _validate_json_assets(staging_data, required)
        _read_json(staging / "SYNC_MANIFEST.json")

    _publish_directory(target, build)
    return manifest_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync the native battery SOH dashboard into frontend/app.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source battery_SOH/dashboard directory.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Target frontend/app/battery-dashboard directory.")
    parser.add_argument("--no-curves", action="store_true", help="Copy scenario JSON only, without *_curves.json.")
    parser.add_argument("--include-detection", action="store_true", help="Also copy ESS detection data assets.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = sync_native_dashboard(
        args.source,
        args.target,
        include_curves=not args.no_curves,
        include_detection=args.include_detection,
    )
    active_target = resolve_published_directory(args.target)
    total_bytes = sum((active_target / "data" / name).stat().st_size for name in manifest["copied_files"])
    print(
        f"Synced {len(manifest['copied_files'])} files "
        f"({total_bytes / (1024 * 1024):.2f} MiB) to {args.target}"
    )


if __name__ == "__main__":
    main()
