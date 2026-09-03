"""Minimal kernel CLI (extraction plan Task 11): ``sohkernel load | preflight | score``.

Every command prints JSON so a shell or an agent can consume it; scoring writes the usual run
artifacts (``leaderboard.json`` and friends) and prints the run directory.  Estimators are
registered explicitly through ``sohkernel.estimators.register_all()``.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import api
from .estimators import register_all


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=None, help="data root (default: $SOHBENCH_DATA_DIR)")
    parser.add_argument("--datasets", nargs="+", default=["all"])


def _matrix(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--estimators", nargs="+", default=["ridge"])
    parser.add_argument("--featuresets", nargs="+", default=["full"])
    parser.add_argument("--protocols", nargs="+", default=["loco"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("sohkernel", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    load = sub.add_parser("load", help="load datasets through QC and print the QC report")
    _common(load)
    load.add_argument("--no-qc", action="store_true")
    preflight = sub.add_parser("preflight", help="admission oracle for a data x feature x estimator x protocol matrix")
    _common(preflight)
    _matrix(preflight)
    score = sub.add_parser("score", help="run the matrix and write run artifacts")
    _common(score)
    _matrix(score)
    score.add_argument("--out-dir", default="runs")
    return parser


def _request(args: argparse.Namespace) -> dict:
    return {"datasets": args.datasets, "estimators": args.estimators, "featuresets": args.featuresets,
            "protocols": args.protocols, "seeds": args.seeds}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    register_all()
    if args.command == "load":
        table, meta, qc = api.load(args.datasets, data_dir=args.data_dir, qc=not args.no_qc)
        payload = {"datasets": sorted(map(str, table["dataset"].unique())), "rows": int(len(table)),
                   "cells": int(len(meta)), "qc": qc.__dict__}
    elif args.command == "preflight":
        request = api.normalize_request(_request(args), data_dir=api.resolve_data_dir(args.data_dir))
        payload = api.preflight(request, data_dir=args.data_dir).to_dict()
    else:
        request = api.normalize_request(_request(args), data_dir=api.resolve_data_dir(args.data_dir))
        result = api.preflight(request, data_dir=args.data_dir)
        if not result.ready:
            payload = {"ready": False, "reasons": result.reasons}
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            return 2
        results = api.score(None, result.table, result.meta, request["protocols"], request["estimators"],
                            request["featuresets"], seeds=request["seeds"])
        out = api.write_artifacts(results, result.qc_report, None, {**request, "kernel": "sohkernel"}, args.out_dir)
        payload = {"run_dir": out, "units": int(results.drop_duplicates(["estimator", "featureset", "protocol", "held_out_key", "seed"]).shape[0])}
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
