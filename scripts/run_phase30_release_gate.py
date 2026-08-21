from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_CURVE_SOURCE = DEFAULT_DATA_DIR / "curves" / "nasa_curves.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the phase 3.0 measured release regression gate"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--curve-source",
        default=str(DEFAULT_CURVE_SOURCE),
    )
    parser.add_argument("--profile", default="")
    parser.add_argument("--dataset", default="nasa")
    parser.add_argument("--asset", default="B0018")
    parser.add_argument("--estimator", default="ridge")
    parser.add_argument("--featureset", default="full")
    parser.add_argument("--protocol", default="loco")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--criterion-set", default="exploratory")
    parser.add_argument(
        "--output-json",
        default=r"work\trusted_plugins\phase30_release_gate_report.json",
    )
    parser.add_argument(
        "--output-md",
        default=r"work\trusted_plugins\phase30_release_gate_report.md",
    )
    parser.add_argument(
        "--archive-dir",
        default=r"work\trusted_plugins\release_gates\phase30",
    )
    args = parser.parse_args()

    root = ROOT
    import sohbench

    resolved = Path(sohbench.__file__).resolve()
    expected = root / "src" / "sohbench" / "__init__.py"
    if resolved != expected:
        raise SystemExit(
            f"Refusing to run with stale sohbench import: {resolved}; expected {expected}. "
            "Set PYTHONPATH to the active repository src directory."
        )

    from sohbench.evaluation_service import EvaluationService
    from sohbench.phase30_release_gate import (
        Phase30ReleaseGateConfig,
        render_phase30_release_gate_markdown,
        run_phase30_release_gate,
    )

    service = EvaluationService(
        data_dir=Path(args.data_dir),
        work_dir=root / "work" / "trusted_plugins",
        runs_dir=root / "runs",
        frontend_data_dir=root / "frontend" / "app" / "data",
    )
    report = run_phase30_release_gate(
        service,
        Phase30ReleaseGateConfig(
            profile_name=args.profile,
            dataset=args.dataset,
            asset_id=args.asset,
            estimator=args.estimator,
            featureset=args.featureset,
            protocol=args.protocol,
            seed=args.seed,
            curve_source=args.curve_source,
            criterion_set_id=args.criterion_set,
        ),
    )
    digest_payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["report_digest"] = f"sha256:{hashlib.sha256(digest_payload).hexdigest()}"
    report["report_digest_contract"] = "canonical JSON before report_digest fields"
    markdown = render_phase30_release_gate_markdown(report)

    json_output = (root / args.output_json).resolve()
    markdown_output = (root / args.output_md).resolve()
    archive_dir = (root / args.archive_dir).resolve()
    _write_json_atomic(json_output, report)
    _write_text_atomic(markdown_output, markdown)
    _write_json_atomic(archive_dir / f"{report['gate_id']}.json", report)
    _write_text_atomic(archive_dir / f"{report['gate_id']}.md", markdown)

    print(
        json.dumps(
            {
                "gate_id": report["gate_id"],
                "status": report["status"],
                "case_id": report.get("case_id") or "",
                "profile_name": report.get("profile_name") or "",
                "checks": report["summary"],
                "report_json": str(json_output),
                "report_markdown": str(markdown_output),
                "url": report.get("url") or "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


def _write_json_atomic(path: Path, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_text_atomic(path, text)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
