from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


GATE_CONTRACT = "phase311-production-stability/1.0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic phase 3.1.1 production stability gate"
    )
    parser.add_argument(
        "--output-json",
        default="work/trusted_plugins/phase311_production_stability_report.json",
    )
    parser.add_argument(
        "--output-md",
        default="work/trusted_plugins/phase311_production_stability_report.md",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    checks: list[dict] = []
    expected_import = (root / "src" / "sohbench" / "__init__.py").resolve()
    try:
        import sohbench

        resolved_import = Path(sohbench.__file__).resolve()
        import_ok = resolved_import == expected_import
        import_error = "" if import_ok else (
            f"stale checkout import: {resolved_import}; expected {expected_import}"
        )
    except Exception as exc:
        resolved_import = None
        import_ok = False
        import_error = f"{type(exc).__name__}: {exc}"
    checks.append({
        "id": "checkout_import",
        "title": "当前仓库导入边界",
        "passed": import_ok,
        "status": "passed" if import_ok else "failed",
        "duration_seconds": 0.0,
        "details": {
            "expected": str(expected_import),
            "actual": str(resolved_import or ""),
            "error": import_error,
        },
    })

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUTF8"] = "1"
    commands = [
        (
            "python_regression",
            "全量 Python 回归",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
            ],
        ),
        (
            "frontend_core",
            "前端核心状态与跨页组合",
            ["node", "frontend/app/dashboard-core.test.mjs"],
        ),
        (
            "frontend_contract",
            "中文界面与操作收敛契约",
            ["node", "frontend/app/chinese-ui.test.mjs"],
        ),
        (
            "native_evidence",
            "原生证据画布契约",
            ["node", "frontend/app/native-dashboard.test.mjs"],
        ),
        (
            "golden_profile",
            "真实黄金案例契约",
            ["node", "frontend/app/phase22-golden.test.mjs"],
        ),
    ]
    for check_id, title, command in commands:
        checks.append(_run_check(root, environment, check_id, title, command))

    passed_count = sum(1 for item in checks if item["passed"])
    report = {
        "schema_version": GATE_CONTRACT,
        "phase": "3.1.1",
        "gate_id": f"P311-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8].upper()}",
        "status": "passed" if passed_count == len(checks) else "failed",
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "repository": str(root),
        "summary": {
            "total": len(checks),
            "passed": passed_count,
            "failed": len(checks) - passed_count,
        },
        "checks": checks,
        "measured_release_gate": {
            "included": False,
            "reason": "阶段 3.0 实测门禁依赖受控的 D:\\battery_SOH 数据，需在本地或自托管发布环境单独执行。",
            "command": "python scripts/run_phase30_release_gate.py",
        },
    }
    json_output = (root / args.output_json).resolve()
    markdown_output = (root / args.output_md).resolve()
    _write_text_atomic(
        json_output,
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    _write_text_atomic(markdown_output, render_markdown(report))
    print(json.dumps({
        "gate_id": report["gate_id"],
        "status": report["status"],
        "summary": report["summary"],
        "report_json": str(json_output),
        "report_markdown": str(markdown_output),
    }, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


def _run_check(
    root: Path,
    environment: dict[str, str],
    check_id: str,
    title: str,
    command: list[str],
) -> dict:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
        output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        return {
            "id": check_id,
            "title": title,
            "passed": completed.returncode == 0,
            "status": "passed" if completed.returncode == 0 else "failed",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "details": {
                "command": subprocess.list2cmdline(command),
                "exit_code": completed.returncode,
                "output_tail": output[-6000:],
            },
        }
    except Exception as exc:
        return {
            "id": check_id,
            "title": title,
            "passed": False,
            "status": "failed",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "details": {
                "command": subprocess.list2cmdline(command),
                "exit_code": None,
                "output_tail": f"{type(exc).__name__}: {exc}",
            },
        }


def render_markdown(report: dict) -> str:
    lines = [
        "# 阶段 3.1.1 生产稳定化门禁报告",
        "",
        f"- 门禁：`{report['gate_id']}`",
        f"- 结论：**{'通过' if report['status'] == 'passed' else '未通过'}**",
        f"- 检查：{report['summary']['passed']} / {report['summary']['total']} 通过",
        f"- 用时：{report['duration_seconds']} 秒",
        "",
        "| 检查 | 结果 | 用时 |",
        "| --- | --- | ---: |",
    ]
    for item in report.get("checks") or []:
        lines.append(
            f"| {item['title']} | {'通过' if item['passed'] else '未通过'} | {item['duration_seconds']} 秒 |"
        )
    lines.extend([
        "",
        "说明：本门禁可在普通 CI 环境重复执行，不伪造外部实测数据。阶段 3.0 实测发布门禁仍需在本地或自托管发布环境单独运行。",
        "",
    ])
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
