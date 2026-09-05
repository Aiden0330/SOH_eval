"""Write the K12 partial-charge health indicators into the processed tables (from the raw charge curves).

For every dataset with a raw charge-curve source (``sohkernel.raw_windows.CHARGE_CURVE_SOURCES``) this script computes
``sohkernel.health_indicators.RAW_COLUMNS`` per cycle, audits the charge columns under the label-safety DOD rule, and
writes the columns into
``data/processed/<dataset>_features.pkl`` (backup under ``processed/backup_pre_partial_charge_hi/``), updating the
manifest's bytes, digest, column list and a ``health_indicators`` block.  ``--dry-run`` prints the report only.

    .venv-mac/bin/python scripts/rebuild_partial_charge_hi.py --data-dir data [--datasets hust snl_lfp ...] [--dry-run] [--report path.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohkernel import health_indicators as hi  # noqa: E402
from sohkernel.feature_safety import audit_feature_frame  # noqa: E402
from sohkernel.metadata import DATASET_C_RATED_AH, DATASET_CHEMISTRY  # noqa: E402
from sohkernel.raw_windows import CHARGE_CURVE_SOURCES, iter_charge_curves  # noqa: E402


def rebuild_dataset(name: str, data_dir: Path, *, dry_run: bool) -> dict:
    chemistry = DATASET_CHEMISTRY[name]
    processed_path = data_dir / "processed" / f"{name}_features.pkl"
    table = pd.read_pickle(processed_path)
    indicators = hi.hi_from_curves(iter_charge_curves(data_dir, name, table), chemistry)
    table = hi.attach(table, indicators)
    matched = int(pd.to_numeric(table["hi_span_dq_ah"], errors="coerce").notna().sum())
    summary = hi.coverage_summary(table, DATASET_C_RATED_AH.get(name))
    audit = audit_feature_frame(table, list(hi.WINDOW_CAPACITY_COLUMNS))
    report = {
        "dataset": name, "chemistry": chemistry, "windows": [list(w) for w in hi.windows_for(chemistry)],
        "rows": int(len(table)), "span_matched": matched, "span_matched_fraction": round(matched / max(len(table), 1), 4),
        "span_coverage_median": summary["span_coverage_median"], "available": summary["available"],
        "audit_status": audit["status"], "audit_violations": [item["reason"] for item in audit["violations"]],
        "audit_warnings": [item["reason"] for item in audit.get("warnings", [])],
    }
    if audit["violations"]:
        raise SystemExit(f"{name}: indicator charge columns violate label safety — refusing to write\n"
                         + "\n".join(report["audit_violations"]))
    if dry_run:
        return report
    backup_dir = data_dir / "processed" / "backup_pre_partial_charge_hi"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / processed_path.name
    if not backup_path.exists():
        shutil.copy2(processed_path, backup_path)
    table.to_pickle(processed_path)
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("datasets", []):
            if entry.get("name") == name and isinstance(entry.get("processed"), dict):
                blob = processed_path.read_bytes()
                entry["processed"]["bytes"] = len(blob)
                entry["processed"]["sha256"] = hashlib.sha256(blob).hexdigest()
                entry["processed"]["columns"] = list(map(str, table.columns))
                entry["processed"]["health_indicators"] = {
                    "source": "raw_charge_curves", "chemistry": chemistry, "windows": report["windows"],
                    "span_matched_fraction": report["span_matched_fraction"],
                    "span_coverage_median": report["span_coverage_median"],
                    "rebuilt_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
        manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def markdown(reports: list[dict], dry_run: bool) -> str:
    lines = [f"# Partial-charge health indicators — rebuild {'(dry run) ' if dry_run else ''}{datetime.now(timezone.utc).strftime('%Y-%m-%d')}", "",
             "K12 (Wave 6 plan §4): `sohkernel.health_indicators` computed from the raw charge curves and written into the processed tables; "
             "the charge columns are audited under the label-safety DOD rule on every rebuild.", "",
             "| Dataset | Chemistry | Windows (V) | Rows | Span matched | Span coverage (median of rated) | Audit |",
             "|---|---|---|---|---|---|---|"]
    for r in reports:
        windows = ", ".join(f"{lo:.2f}–{hi_:.2f}" for lo, hi_ in r["windows"])
        lines.append(f"| {r['dataset']} | {r['chemistry']} | {windows} | {r['rows']} | {r['span_matched']} ({r['span_matched_fraction']:.1%}) | "
                     f"{r['span_coverage_median'] if r['span_coverage_median'] is not None else '—'} | {r['audit_status']}"
                     f"{' · ' + '; '.join(r['audit_warnings']) if r['audit_warnings'] else ''} |")
    lines += ["", "## Availability per column (fraction of rows with a value)", ""]
    columns = list(hi.RAW_COLUMNS)
    lines.append("| Dataset | " + " | ".join(columns) + " |")
    lines.append("|---|" + "---|" * len(columns))
    for r in reports:
        lines.append(f"| {r['dataset']} | " + " | ".join(f"{r['available'].get(c, float('nan')):.2f}" for c in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=None, help="default: every dataset with a raw charge-curve source")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=None, help="write the markdown report here")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    names = args.datasets or [name for name in DATASET_CHEMISTRY if name in CHARGE_CURVE_SOURCES]
    reports = []
    for name in names:
        if name not in CHARGE_CURVE_SOURCES:
            print(f"{name}: no raw charge-curve source registered in sohkernel.raw_windows", file=sys.stderr)
            continue
        report = rebuild_dataset(name, data_dir, dry_run=args.dry_run)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))
    text = markdown(reports, args.dry_run)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"report written to {args.report}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
