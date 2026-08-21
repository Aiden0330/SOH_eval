from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sohbench.evaluation_service import EvaluationService  # noqa: E402


DEFAULT_JSON_OUTPUT = ROOT / "dist" / "soh-eval-delivery-report.json"
DEFAULT_MD_OUTPUT = ROOT / "dist" / "soh-eval-delivery-report.md"


def build_delivery_report(
    *,
    native_dashboard: Path | None = None,
    portable_output: Path | None = None,
    standalone_output: Path | None = None,
) -> dict:
    service = EvaluationService(
        data_dir=ROOT / "data",
        work_dir=ROOT / "work" / "trusted_plugins",
        runs_dir=ROOT / "runs",
        frontend_data_dir=ROOT / "frontend" / "app" / "data",
    )
    report = service.package_status(
        native_dashboard=native_dashboard,
        portable_output=portable_output,
        standalone_output=standalone_output,
    )
    outputs = report.get("outputs", {})
    return {
        "kind": "soh_eval_delivery_report",
        "generated_at": report.get("generated_at"),
        "report": report,
        "artifacts": {
            "native_dashboard": outputs.get("native_dashboard", {}),
            "portable": outputs.get("portable", {}),
            "standalone": outputs.get("standalone", {}),
        },
        "usage": _usage_steps(outputs),
    }


def write_delivery_report(payload: dict, *, json_output: Path, md_output: Path) -> dict:
    json_output = Path(json_output)
    md_output = Path(md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    md_output.write_text(render_markdown_report(payload), encoding="utf-8", newline="\n")
    return {
        "json_output": str(json_output),
        "md_output": str(md_output),
        "ready": bool(payload.get("report", {}).get("summary", {}).get("ready")),
    }


def render_markdown_report(payload: dict) -> str:
    report = payload.get("report", {})
    summary = report.get("summary", {})
    checks = report.get("checks") or []
    artifacts = payload.get("artifacts", {})
    usage = payload.get("usage", {})
    status = "验收通过" if summary.get("ready") else "验收未通过"
    lines = [
        "# SOH Eval 交付包报告",
        "",
        f"- 生成时间：{payload.get('generated_at') or 'unknown'}",
        f"- 验收状态：{status}：{summary.get('passed', 0)}/{summary.get('total', 0)}",
        f"- 失败项：{summary.get('failed', 0)}",
        "",
        "## 交付产物",
        "",
        f"- 目录包：`{_artifact_path(artifacts, 'portable')}`",
        f"- 单 HTML：`{_artifact_path(artifacts, 'standalone')}`",
        f"- SOH 看板同步目录：`{_artifact_path(artifacts, 'native_dashboard')}`",
        "",
        "## 验收检查",
        "",
    ]
    for check in checks:
        mark = "通过" if check.get("passed") else "待处理"
        lines.append(f"- {mark}：{check.get('label', check.get('id', '检查项'))}，{check.get('detail', '')}")
    lines.extend(
        [
            "",
            "## 迁移使用步骤",
            "",
            "### 目录包",
            "",
            *[f"{idx}. {step}" for idx, step in enumerate(usage.get("portable") or [], start=1)],
            "",
            "### 单 HTML",
            "",
            *[f"{idx}. {step}" for idx, step in enumerate(usage.get("standalone") or [], start=1)],
            "",
            "### 验收",
            "",
            *[f"{idx}. {step}" for idx, step in enumerate(usage.get("validation") or [], start=1)],
            "",
        ]
    )
    return "\n".join(lines)


def _usage_steps(outputs: dict) -> dict:
    portable_path = str((outputs.get("portable") or {}).get("path") or "dist/soh-eval-portable")
    standalone_path = str((outputs.get("standalone") or {}).get("path") or "dist/soh-eval-standalone.html")
    return {
        "portable": [
            f"复制整个目录包：{portable_path}",
            f"进入目录包：cd {portable_path}",
            "在该目录执行：python -m http.server 8010",
            "浏览器打开：http://127.0.0.1:8010/",
        ],
        "standalone": [
            f"复制单文件：{standalone_path}",
            "双击 soh-eval-standalone.html，或用浏览器打开该文件。",
        ],
        "validation": [
            "打开全局诊断页，点击“刷新验收”。",
            "确认 SOH 看板同步、SOH 曲线资产、目录包、单 HTML、异常检测资产排除均为通过。",
        ],
    }


def _artifact_path(artifacts: dict, key: str) -> str:
    value = artifacts.get(key) or {}
    return str(value.get("path") or "missing")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SOH Eval delivery package report.")
    parser.add_argument("--native-dashboard", type=Path, default=None, help="Synced native dashboard directory.")
    parser.add_argument("--portable-output", type=Path, default=None, help="Portable directory bundle.")
    parser.add_argument("--standalone-output", type=Path, default=None, help="Standalone HTML bundle.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT, help="JSON report output.")
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT, help="Markdown report output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_delivery_report(
        native_dashboard=args.native_dashboard,
        portable_output=args.portable_output,
        standalone_output=args.standalone_output,
    )
    outputs = write_delivery_report(payload, json_output=args.json_output, md_output=args.md_output)
    print(
        f"Wrote {outputs['json_output']} and {outputs['md_output']} "
        f"(ready={str(outputs['ready']).lower()})"
    )


if __name__ == "__main__":
    main()
