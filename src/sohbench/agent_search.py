"""Protocol-gated search harness for algorithm x featureset selection.

Design: docs/superpowers/specs/2026-08-26-agent-guided-search-design.md

The loop that makes agent-driven search safe in this benchmark:

- a Policy proposes candidates (``GridPolicy`` ships; an LLM-backed policy
  plugs into the same protocol),
- ``EvaluationService.training_preflight`` is the HARD constraint oracle —
  label-safety violations, window-DOD violations and coverage failures
  eliminate a candidate before anything runs,
- the inner loop scores admitted candidates per inner dataset under LOCO
  and ranks by worst-dataset MAE,
- the outer dataset is never touched during search; the final claim runs
  once, pooled LODO, extracting only the fold that holds the outer dataset
  out.

Usage:
    python -m sohbench.agent_search --inner calce_cs2 snl_lfp --outer hust \
        --estimators ridge extra_trees gbdt \
        --featuresets voltage ic_dva partial_window degradation_kinetics \
        --out runs/agent_search_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from . import registry as R
from .data import DatasetLoader
from .evaluation_service import PROTOCOLS, EvaluationService
from .runner import BenchmarkRunner
from .tasks import SOHEstimationTask


@dataclass(frozen=True)
class Candidate:
    estimator: str
    featureset: str

    def key(self) -> str:
        return f"{self.estimator}×{self.featureset}"


@dataclass
class SearchHistory:
    rounds: list[dict] = field(default_factory=list)

    def scored(self) -> dict[str, dict]:
        merged: dict[str, dict] = {}
        for entry in self.rounds:
            for item in entry.get("scores", []):
                merged[item["candidate"]] = item
        return merged


class Policy(Protocol):
    """Proposes candidates per round; an empty proposal ends the search."""

    def propose(self, history: SearchHistory) -> list[Candidate]: ...


class GridPolicy:
    """One round: the full estimator x featureset grid. Converges after it."""

    def __init__(self, estimators: list[str], featuresets: list[str]):
        self.grid = [
            Candidate(estimator, featureset)
            for estimator in estimators
            for featureset in featuresets
        ]

    def propose(self, history: SearchHistory) -> list[Candidate]:
        return [] if history.rounds else list(self.grid)


def _mae_from_results(results: pd.DataFrame) -> tuple[float, int, int]:
    """Mean fold MAE, fold count, failed-fold count from runner long format."""
    rows = results[results["metric"] == "mae"]
    values = pd.to_numeric(rows["value"], errors="coerce")
    failed = int(values.isna().sum())
    finite = values.dropna()
    mae = float(finite.mean()) if len(finite) else float("nan")
    return mae, int(len(rows)), failed


class AgentSearch:
    def __init__(
        self,
        service: EvaluationService,
        *,
        inner_datasets: list[str],
        outer_dataset: str,
        protocol: str = "loco",
        seeds: list[int] | None = None,
        top_k: int = 3,
        max_rounds: int = 8,
    ):
        if outer_dataset in inner_datasets:
            raise ValueError("outer dataset must not appear in the inner search pool")
        self.service = service
        self.inner_datasets = list(inner_datasets)
        self.outer_dataset = outer_dataset
        self.protocol = protocol
        self.seeds = list(seeds or [0])
        self.top_k = top_k
        self.max_rounds = max_rounds
        self._loader = DatasetLoader(service.data_dir)
        self._table_cache: dict[tuple[str, ...], tuple[pd.DataFrame, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # constraint oracle
    # ------------------------------------------------------------------

    def _admit(self, candidate: Candidate, datasets: list[str]) -> tuple[bool, list[str], list[str]]:
        preflight = self.service.training_preflight(
            {
                "datasets": datasets,
                "estimators": [candidate.estimator],
                "featuresets": [candidate.featureset],
                "protocols": [self.protocol],
                "seeds": self.seeds,
            }
        )
        reasons: list[str] = []
        warnings: list[str] = []
        for combo in preflight.get("combinations", []):
            reasons.extend(combo.get("reasons") or [])
            warnings.extend(combo.get("warnings") or [])
        return bool(preflight.get("ready")), sorted(set(reasons)), sorted(set(warnings))

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _load(self, datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        cache_key = tuple(sorted(datasets))
        if cache_key not in self._table_cache:
            names = self._loader.resolve_datasets(datasets)
            table, meta, _qc = self._loader.load(names)
            self._table_cache[cache_key] = (table, meta)
        return self._table_cache[cache_key]

    def _run(
        self,
        candidate: Candidate,
        datasets: list[str],
        protocol: str,
    ) -> pd.DataFrame:
        table, meta = self._load(datasets)
        table = self.service.store.apply_feature_transforms(
            table, meta, [candidate.featureset]
        )
        runner = BenchmarkRunner(seeds=self.seeds)
        return runner.run(
            SOHEstimationTask(),
            table=table,
            meta=meta,
            protocols={protocol: PROTOCOLS[protocol]()},
            estimators=[R.get_estimator(candidate.estimator)],
            featuresets=[candidate.featureset],
        )

    def _score_inner(self, candidate: Candidate) -> dict:
        per_dataset = {}
        for dataset in self.inner_datasets:
            results = self._run(candidate, [dataset], self.protocol)
            mae, folds, failed = _mae_from_results(results)
            per_dataset[dataset] = {"mae": mae, "folds": folds, "failed_folds": failed}
        maes = [item["mae"] for item in per_dataset.values()]
        finite = [value for value in maes if np.isfinite(value)]
        return {
            "candidate": candidate.key(),
            "estimator": candidate.estimator,
            "featureset": candidate.featureset,
            "per_dataset": per_dataset,
            "mae_mean": float(np.mean(finite)) if finite else float("nan"),
            "mae_worst": float(np.max(finite)) if len(finite) == len(maes) and finite else float("nan"),
        }

    def _outer_claim(self, candidate: Candidate) -> dict:
        pooled = [*self.inner_datasets, self.outer_dataset]
        results = self._run(candidate, pooled, "lodo")
        rows = results[
            (results["metric"] == "mae")
            & (results["held_out_key"] == self.outer_dataset)
        ]
        values = pd.to_numeric(rows["value"], errors="coerce").dropna()
        return {
            "candidate": candidate.key(),
            "outer_dataset": self.outer_dataset,
            "outer_mae": float(values.mean()) if len(values) else float("nan"),
            "folds": int(len(rows)),
        }

    # ------------------------------------------------------------------
    # loop
    # ------------------------------------------------------------------

    def run(self, policy: Policy, *, log=print) -> dict:
        history = SearchHistory()
        excluded: list[dict] = []
        for round_index in range(self.max_rounds):
            proposals = policy.propose(history)
            if not proposals:
                break
            entry = {"round": round_index, "proposed": len(proposals), "scores": []}
            for candidate in proposals:
                admitted, reasons, warnings = self._admit(candidate, self.inner_datasets)
                if not admitted:
                    excluded.append(
                        {"candidate": candidate.key(), "reasons": reasons}
                    )
                    log(f"  [gate] {candidate.key()} excluded: {reasons[:1]}")
                    continue
                score = self._score_inner(candidate)
                score["preflight_warnings"] = warnings
                entry["scores"].append(score)
                log(
                    f"  [inner] {candidate.key()} mae_mean={score['mae_mean']:.4f} "
                    f"mae_worst={score['mae_worst']:.4f}"
                )
            history.rounds.append(entry)

        ranking = sorted(
            history.scored().values(),
            key=lambda item: (
                not np.isfinite(item["mae_worst"]),
                item["mae_worst"],
            ),
        )
        finalists = [
            Candidate(item["estimator"], item["featureset"])
            for item in ranking[: self.top_k]
        ]
        outer_claims = []
        for candidate in finalists:
            log(f"  [outer] {candidate.key()} on held-out {self.outer_dataset} ...")
            outer_claims.append(self._outer_claim(candidate))

        return {
            "schema_version": "1.0.0",
            "inner_datasets": self.inner_datasets,
            "outer_dataset": self.outer_dataset,
            "protocol": {"inner": self.protocol, "outer": "lodo"},
            "seeds": self.seeds,
            "rounds": history.rounds,
            "excluded": excluded,
            "ranking": ranking,
            "outer_claims": outer_claims,
            "integrity": {
                "outer_dataset_visible_during_search": False,
                "outer_evaluations": len(outer_claims),
                "constraint_oracle": "training_preflight (label-safety + window DOD + coverage)",
                "selection_metric": "worst-case inner-dataset LOCO MAE",
            },
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("sohbench-agent-search")
    parser.add_argument("--inner", nargs="+", required=True)
    parser.add_argument("--outer", required=True)
    parser.add_argument("--estimators", nargs="+", default=["ridge", "extra_trees", "gbdt"])
    parser.add_argument(
        "--featuresets",
        nargs="+",
        default=["voltage", "ic_dva", "partial_window", "degradation_kinetics"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--work-dir", default="work/trusted_plugins")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    service = EvaluationService(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        runs_dir="runs",
        frontend_data_dir="frontend/app/data",
    )
    search = AgentSearch(
        service,
        inner_datasets=args.inner,
        outer_dataset=args.outer,
        top_k=args.top_k,
    )
    report = search.run(GridPolicy(args.estimators, args.featuresets))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"report written to {out_path}")
    print("\n=== ranking (worst-case inner MAE) ===")
    for item in report["ranking"]:
        print(f"  {item['candidate']:45s} worst={item['mae_worst']:.4f} mean={item['mae_mean']:.4f}")
    print("=== outer claims (single-shot LODO) ===")
    for item in report["outer_claims"]:
        print(f"  {item['candidate']:45s} outer_mae={item['outer_mae']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
