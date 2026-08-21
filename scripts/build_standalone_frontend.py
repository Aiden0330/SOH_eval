from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "frontend" / "app"
DATA_DIR = APP_DIR / "data"
DEFAULT_OUTPUT = ROOT / "dist" / "soh-eval-standalone.html"


IMPORT_RE = re.compile(
    r'^\s*import\s*\{.*?\}\s*from\s*["\']\.\/dashboard-core\.mjs[^"\']*["\'];\s*',
    re.MULTILINE | re.DOTALL,
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def script_safe(text: str) -> str:
    return text.replace("</script", "<\\/script")


def style_safe(text: str) -> str:
    return text.replace("</style", "<\\/style")


def is_detection_asset(path: Path) -> bool:
    return "ess_detection" in path.name


def strip_detection_entries(value):
    if isinstance(value, list):
        return [
            strip_detection_entries(item)
            for item in value
            if not _is_detection_entry(item)
        ]
    if isinstance(value, dict):
        return {
            key: strip_detection_entries(item)
            for key, item in value.items()
            if "ess_detection" not in str(key)
        }
    if isinstance(value, str):
        return _strip_detection_text(value)
    return value


def _is_detection_entry(value) -> bool:
    if isinstance(value, str):
        return "ess_detection" in value
    if not isinstance(value, dict):
        return False
    fields = [
        value.get("dataset"),
        value.get("filename"),
        value.get("source"),
        value.get("id"),
    ]
    return any("ess_detection" in str(field) for field in fields if field is not None)


def _strip_detection_text(value: str) -> str:
    if "ess_detection" not in value:
        return value
    parts = value.split(";")
    if len(parts) > 1:
        return ";".join(part for part in parts if "ess_detection" not in part)
    return ""


def collect_data_payload(app_dir: Path = APP_DIR) -> dict[str, object]:
    app_dir = Path(app_dir)
    data_dir = app_dir / "data"
    payload: dict[str, object] = {}
    for path in sorted(data_dir.rglob("*.json")):
        if is_detection_asset(path):
            continue
        key = "./" + path.relative_to(app_dir).as_posix()
        payload[key] = strip_detection_entries(json.loads(path.read_text(encoding="utf-8")))
    return payload


def strip_module_exports(source: str) -> str:
    source = re.sub(r"^\s*export\s+function\s+", "function ", source, flags=re.MULTILINE)
    source = re.sub(r"^\s*export\s+(const|let|var)\s+", r"\1 ", source, flags=re.MULTILINE)
    source = re.sub(r"^\s*export\s+\{[^}]+\};?\s*$", "", source, flags=re.MULTILINE)
    return source


def strip_app_import(source: str) -> str:
    stripped, count = IMPORT_RE.subn("", source, count=1)
    if count != 1:
        raise RuntimeError("Could not remove dashboard-core import from frontend/app/app.js")
    return stripped


def standalone_bootstrap(data_payload: dict[str, object]) -> str:
    json_payload = json.dumps(data_payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""
const SOH_STANDALONE_DATA = {json_payload};
window.__SOH_STANDALONE__ = true;
window.__SOH_STANDALONE_DATA__ = SOH_STANDALONE_DATA;

function standaloneDataKey(input) {{
  const rawInput = input && input.url ? input.url : input;
  let value = String(rawInput || "").split("#", 1)[0].split("?", 1)[0].replace(/\\\\/g, "/");
  if (/^(file|https?):/i.test(value)) {{
    try {{
      value = new URL(value, window.location.href).pathname.replace(/\\\\/g, "/");
    }} catch (_) {{}}
  }}
  const marker = "/data/";
  const markerIndex = value.lastIndexOf(marker);
  if (markerIndex >= 0) {{
    return "./data/" + value.slice(markerIndex + marker.length);
  }}
  value = value.replace(/^\\.\\//, "").replace(/^\\//, "");
  return value.startsWith("data/") ? "./" + value : value;
}}

window.fetch = async (input, init = {{}}) => {{
  const method = String(init?.method || "GET").toUpperCase();
  const key = standaloneDataKey(input);
  if (method === "GET" && Object.prototype.hasOwnProperty.call(SOH_STANDALONE_DATA, key)) {{
    return new Response(JSON.stringify(SOH_STANDALONE_DATA[key]), {{
      status: 200,
      headers: {{ "Content-Type": "application/json; charset=utf-8" }},
    }});
  }}
  return new Response(JSON.stringify({{
    error: "离线静态包不包含后端接口；上传、启动评测和任务轮询需要打开完整前后端系统。"
  }}), {{
    status: 503,
    headers: {{ "Content-Type": "application/json; charset=utf-8" }},
  }});
}};
""".strip()


def replace_once(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, lambda _: replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Could not inline {label}")
    return updated


def build_standalone_html(app_dir: Path = APP_DIR) -> tuple[str, int]:
    app_dir = Path(app_dir)
    html = read_text(app_dir / "index.html")
    html = html.replace("<span>数据目录</span>", "<span>数据来源</span>")
    html = html.replace("D:\\battery_SOH\\data", "离线内嵌数据")
    if "离线内嵌数据" not in html:
        html = html.replace("<body", '<body data-source-label="离线内嵌数据"', 1)
    css = style_safe(read_text(app_dir / "styles.css"))
    core = strip_module_exports(read_text(app_dir / "dashboard-core.mjs"))
    app = strip_app_import(read_text(app_dir / "app.js"))
    data_payload = collect_data_payload(app_dir)

    bundle = "\n\n".join(
        [
            "/* Standalone data and offline fetch adapter. */",
            standalone_bootstrap(data_payload),
            "/* dashboard-core.mjs */",
            core,
            "/* app.js */",
            app,
        ]
    )

    html = replace_once(
        html,
        r'<link\s+rel="stylesheet"\s+href="styles\.css[^"]*"\s*/>',
        f"<style>\n{css}\n</style>",
        "styles.css",
    )
    html = replace_once(
        html,
        r'<script\s+type="module"\s+src="app\.js[^"]*"></script>',
        f"<script>\n{script_safe(bundle)}\n</script>",
        "app.js",
    )
    banner = (
        "<!-- Standalone SOH evaluation bundle: CSS, JS, and "
        f"{len(data_payload)} JSON artifacts are embedded. -->\n"
    )
    return banner + html, len(data_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable standalone SOH dashboard HTML.")
    parser.add_argument("--app-dir", type=Path, default=APP_DIR, help="Source frontend/app directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML path.")
    args = parser.parse_args()

    html, artifact_count = build_standalone_html(args.app_dir)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.2f} MiB, embedded {artifact_count} JSON artifacts)")


if __name__ == "__main__":
    main()
