"""D12 (2026-09-05): raw-archive provenance and measured sampling intervals for the datasheets.

For every dataset named on the command line this script (1) lists the raw files the manifest's ``raw_paths`` point at
(plus explicit extra sources such as the RWTH mirror .mat kept under ``work/``), with size and sha256 each, and folds
them into one ``raw_sha256`` (sha256 over the sorted ``path\\tsize\\tsha256`` lines); (2) measures the sampling interval
of the raw charge curves (median of the per-cycle median time step over the first 400 charge segments the kernel's
``iter_charge_curves`` yields, or a direct read of the .mat for nasa and oxford); and (3) writes both into
``data/datasheets/_provenance.json`` (tracked beside the sheets; the shared manifest is left alone) so
``scripts/draft_datasheets.py`` can put them in the sheets. Primary evidence, reproducible.

    .venv-mac/bin/python scripts/measure_raw_provenance.py --data-dir <data root> --out data/datasheets/_provenance.json --datasets hust ...
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXTRA_SOURCES = {   # raw sources that live outside data/raw (recorded relative to the repository root)
    "rwth_isea_public": ["work/public_sources/rwth-battery-degradation/Pre-processing Codes/Degradation_Prediction_Dataset_ISEA.mat"],
}
SKIP_SUFFIXES = (".DS_Store",)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def raw_files(entry: dict, data_dir: Path, repo_root: Path) -> list[dict]:
    files: list[Path] = []
    for rel in entry.get("raw_paths") or []:
        path = data_dir / rel
        if path.is_dir():
            files += [p for p in sorted(path.rglob("*")) if p.is_file() and not p.name.endswith(SKIP_SUFFIXES)]
        elif path.is_file():
            files.append(path)
    for rel in EXTRA_SOURCES.get(entry["name"], []):
        # extra sources live in a checkout's work/ tree: the data directory's parent (the main checkout) or this repository
        for base in (data_dir.parent, repo_root):
            path = base / rel
            if path.is_file():
                files.append(path)
                break
    records = []
    for path in files:
        try:
            shown = f"data/{path.relative_to(data_dir)}"
        except ValueError:
            base = next((b for b in (data_dir.parent, repo_root) if b in path.parents), None)
            shown = str(path.relative_to(base)) if base else str(path)
        records.append({"path": shown, "bytes": path.stat().st_size, "sha256": sha256_of(path)})
    return records


def fold(records: list[dict]) -> str | None:
    if not records:
        return None
    lines = sorted(f"{r['path']}\t{r['bytes']}\t{r['sha256']}" for r in records)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def sampling_interval(name: str, data_dir: Path) -> dict | None:
    """Median time step of the raw charge curves (seconds) with the evidence of how it was measured."""
    from sohkernel.raw_windows import CHARGE_CURVE_SOURCES, iter_charge_curves

    if name in CHARGE_CURVE_SOURCES:
        table = pd.read_pickle(data_dir / "processed" / f"{name}_features.pkl")
        steps = []
        for _cell, _cycle, _v, _q, t in itertools.islice(iter_charge_curves(data_dir, name, table), 0, 400):
            d = np.diff(np.asarray(t, dtype=float))
            d = d[np.isfinite(d) & (d > 0)]
            if len(d):
                steps.append(float(np.median(d)))
        if steps:
            return {"seconds": round(float(np.median(steps)), 3), "p10": round(float(np.percentile(steps, 10)), 3),
                    "p90": round(float(np.percentile(steps, 90)), 3), "segments": len(steps),
                    "method": "median per-cycle time step of the raw charge segments (sohkernel.raw_windows.iter_charge_curves, first 400 segments)"}
        return None
    from scipy.io import loadmat

    if name == "nasa":
        steps = []
        for path in sorted((data_dir / "raw" / "nasa").glob("B00*.mat"))[:4]:
            payload = loadmat(path, simplify_cells=True)
            key = next(k for k in payload if k.startswith("B0"))
            for cycle in payload[key]["cycle"]:
                if cycle.get("type") != "charge":
                    continue
                t = np.asarray(cycle["data"]["Time"], dtype=float)
                d = np.diff(t)
                d = d[np.isfinite(d) & (d > 0)]
                if len(d):
                    steps.append(float(np.median(d)))
        if steps:
            return {"seconds": round(float(np.median(steps)), 3), "p10": round(float(np.percentile(steps, 10)), 3),
                    "p90": round(float(np.percentile(steps, 90)), 3), "segments": len(steps),
                    "method": "median time step of the charge cycles' Time vectors in B0005/B0006/B0007/B0018 (.mat), variable-rate logging"}
    if name == "oxford":
        payload = loadmat(data_dir / "raw" / "oxford" / "Oxford_Battery_Degradation_Dataset_1.mat", simplify_cells=True)
        steps = []
        for cell_key in sorted(k for k in payload if k.startswith("Cell"))[:8]:
            cell = payload[cell_key]
            for cyc in sorted(k for k in cell if k.startswith("cyc"))[:5]:
                t = np.asarray(cell[cyc]["C1ch"]["t"], dtype=float) * 86400.0   # stored as a fraction of a day
                d = np.diff(t)
                d = d[np.isfinite(d) & (d > 0)]
                if len(d):
                    steps.append(float(np.median(d)))
        if steps:
            return {"seconds": round(float(np.median(steps)), 3), "p10": round(float(np.percentile(steps, 10)), 3),
                    "p90": round(float(np.percentile(steps, 90)), 3), "segments": len(steps),
                    "method": "median time step of the 1C characterisation charges (C1ch.t, stored as a fraction of a day) over 8 cells x 5 check-ups"}
    if name == "rwth_isea_public":
        return {"seconds": None, "method": "not applicable: the ingested mirror .mat (Degradation_Prediction_Dataset_ISEA.mat) holds per-cycle "
                                            "capacity histories, not time series; the record's Rawdata.zip time series are not ingested"}
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=str(ROOT / "data" / "datasheets" / "_provenance.json"))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = {e["name"]: e for e in manifest["datasets"]}
    out = Path(args.out)
    provenance = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"schema_version": "1.0.0", "datasets": {}}
    for name in args.datasets:
        entry = entries[name]
        records = raw_files(entry, data_dir, ROOT)
        digest = fold(records)
        interval = sampling_interval(name, data_dir)
        print(json.dumps({"dataset": name, "raw_files": len(records), "raw_bytes": sum(r["bytes"] for r in records),
                          "raw_sha256": digest, "sampling_interval_s": interval}, ensure_ascii=False))
        provenance["datasets"][name] = {"raw_sha256": digest, "raw_files": records, "sampling_interval_s": interval,
                                        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        "data_dir": str(data_dir)}
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"provenance written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
