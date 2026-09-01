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
from .evaluation_service import (
    PROTOCOLS,
    EvaluationService,
    _immutable_model_binding,
    _inference_transform_featuresets,
)
from .runner import BenchmarkRunner
from .tasks import SOHEstimationTask


@dataclass(frozen=True)
class Candidate:
    estimator: str
    featureset: str
    # Tuned-window dimension: when set, q_win/t_win are substituted from the
    # window bank at per-dataset bounds chosen (label-free) so the median
    # q_win / rated-capacity fraction lands on this target.
    window_target: float | None = None

    def key(self) -> str:
        base = f"{self.estimator}×{self.featureset}"
        if self.window_target is not None:
            base += f"@w{int(round(self.window_target * 100))}%"
        return base


class WindowGateError(RuntimeError):
    """A tuned-window candidate failed a hard gate (bank missing or audit)."""


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
    """One round: the full estimator x featureset grid. Converges after it.

    ``window_targets`` adds tuned-window candidates (estimator x
    partial_window_norm at each coverage target) on top of the plain grid.
    """

    def __init__(
        self,
        estimators: list[str],
        featuresets: list[str],
        window_targets: list[float] | None = None,
    ):
        self.grid = [
            Candidate(estimator, featureset)
            for estimator in estimators
            for featureset in featuresets
        ]
        for target in window_targets or []:
            for estimator in estimators:
                self.grid.append(Candidate(estimator, "partial_window_norm", float(target)))

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
        self._bank_cache: dict[str, pd.DataFrame] = {}
        self._bounds_cache: dict[tuple[str, float], dict] = {}
        self.window_provenance: dict[str, dict[str, dict]] = {}

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

    def _window_bounds(self, dataset: str, target: float, meta: pd.DataFrame) -> dict:
        from .window_bank import load_bank, select_bounds

        cache_key = (dataset, round(float(target), 4))
        if cache_key not in self._bounds_cache:
            if dataset not in self._bank_cache:
                try:
                    self._bank_cache[dataset] = load_bank(self.service.data_dir, dataset)
                except FileNotFoundError as exc:
                    raise WindowGateError(str(exc)) from exc
            rated_rows = meta[meta["dataset"].astype(str) == dataset]
            rated_by_cell = {
                str(cell): float(value)
                for cell, value in zip(rated_rows["cell_id"], rated_rows["c_rated_ah"])
                if pd.notna(value)
            }
            if not rated_by_cell:
                raise WindowGateError(f"no rated capacities for dataset {dataset}")
            self._bounds_cache[cache_key] = select_bounds(
                self._bank_cache[dataset], rated_by_cell, float(target)
            )
        return self._bounds_cache[cache_key]

    def _substitute_windows(
        self, candidate: Candidate, table: pd.DataFrame, meta: pd.DataFrame, datasets: list[str]
    ) -> pd.DataFrame:
        from .feature_safety import audit_feature_frame
        from .window_bank import apply_bank_window

        provenance = self.window_provenance.setdefault(candidate.key(), {})
        for dataset in datasets:
            bounds = self._window_bounds(dataset, candidate.window_target, meta)
            table = apply_bank_window(
                table, dataset, self._bank_cache[dataset], bounds["lo_mv"], bounds["hi_mv"]
            )
            provenance[dataset] = dict(bounds)
        # The DOD/label gates stay hard constraints on the ACTUAL substituted
        # column, not just on the processed defaults preflight audited.
        audit = audit_feature_frame(table, ["q_win_ah"])
        if audit["violations"]:
            reasons = "; ".join(item["reason"] for item in audit["violations"])
            raise WindowGateError(f"tuned window failed label-safety audit: {reasons}")
        return table

    def _run(
        self,
        candidate: Candidate,
        datasets: list[str],
        protocol: str,
    ) -> pd.DataFrame:
        table, meta = self._load(datasets)
        if candidate.window_target is not None:
            table = self._substitute_windows(candidate, table, meta, datasets)
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
            "window_target": candidate.window_target,
            "window_bounds": self.window_provenance.get(candidate.key(), {}),
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
            "window_bounds": self.window_provenance.get(candidate.key(), {}),
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
            entry = {"round": round_index, "proposed": len(proposals), "scores": [], "excluded": []}
            for candidate in proposals:
                admitted, reasons, warnings = self._admit(candidate, self.inner_datasets)
                if not admitted:
                    exclusion = {"candidate": candidate.key(), "reasons": reasons}
                    excluded.append(exclusion)
                    entry["excluded"].append(exclusion)
                    log(f"  [gate] {candidate.key()} excluded: {reasons[:1]}")
                    continue
                try:
                    score = self._score_inner(candidate)
                except WindowGateError as exc:
                    exclusion = {"candidate": candidate.key(), "reasons": [str(exc)]}
                    excluded.append(exclusion)
                    entry["excluded"].append(exclusion)
                    log(f"  [gate] {candidate.key()} excluded: {exc}")
                    continue
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
            Candidate(item["estimator"], item["featureset"], item.get("window_target"))
            for item in ranking[: self.top_k]
        ]
        outer_claims = []
        for candidate in finalists:
            log(f"  [outer] {candidate.key()} on held-out {self.outer_dataset} ...")
            try:
                outer_claims.append(self._outer_claim(candidate))
            except WindowGateError as exc:
                outer_claims.append(
                    {"candidate": candidate.key(), "outer_dataset": self.outer_dataset,
                     "outer_mae": float("nan"), "error": str(exc)}
                )

        return {
            "schema_version": "1.0.0",
            "policy": type(policy).__name__,
            "policy_log": list(getattr(policy, "log_entries", []) or []),
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


@dataclass(frozen=True)
class AdaptationCandidate:
    model: str
    operator: str

    def key(self) -> str:
        return f"{self.model} · {self.operator}"


class AdaptationGridPolicy:
    def __init__(self, models: list[str], operators: list[str]):
        self.grid = [
            AdaptationCandidate(model, operator)
            for model in models
            for operator in operators
        ]

    def propose(self, history: SearchHistory) -> list["AdaptationCandidate"]:
        return [] if history.rounds else list(self.grid)


class AdaptationSearch:
    """Inference-phase search: frozen model x adaptation operator on a target.

    The model never changes — candidates pair a trained artifact with a
    deployment-legal adaptation operator (see ``sohbench.adaptation``).
    Every operator's information channel is declared in the report;
    evaluation always runs on the fixed non-calibration subset so budgets
    don't bias comparisons.  The trusted-inference label-safety gate runs
    inside every predict.
    """

    def __init__(
        self,
        service,
        *,
        target_dataset: str,
        calibration_budget: int = 32,
        calibration_split: str = "earliest",
        interval: str = "model",
        top_k: int = 3,
        max_rounds: int = 8,
    ):
        from .adaptation import CALIBRATION_SPLITS, INTERVAL_MODES

        if calibration_split not in CALIBRATION_SPLITS:
            raise ValueError(f"unknown calibration split: {calibration_split}")
        if interval not in INTERVAL_MODES:
            raise ValueError(f"unknown interval mode: {interval}")
        self.service = service
        self.target_dataset = target_dataset
        self.calibration_budget = max(0, int(calibration_budget))
        self.calibration_split = calibration_split
        self.interval = interval
        self.top_k = top_k
        self.max_rounds = max_rounds
        self._loader = DatasetLoader(service.data_dir)
        self._table_cache: dict[tuple[str, ...], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _load(self, datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        cache_key = tuple(sorted(datasets))
        if cache_key not in self._table_cache:
            names = self._loader.resolve_datasets(datasets)
            table, meta, _qc = self._loader.load(names)
            self._table_cache[cache_key] = (table, meta)
        return self._table_cache[cache_key]

    def _prepared_target(self, binding: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        table, meta = self._load([self.target_dataset])
        transforms = _inference_transform_featuresets([binding])
        table = self.service.store.apply_feature_transforms(table, meta, transforms)
        return table, meta

    def _score(self, candidate: AdaptationCandidate) -> dict:
        from .adaptation import (
            OPERATOR_CHANNELS,
            apply_coral,
            apply_monotone,
            apply_recalibration,
            apply_residual,
            calibration_split,
            fit_coral,
            fit_recalibration,
            fit_residual,
        )
        from .adaptation import _feature_matrix  # noqa: PLC2701 - shared helper

        entry = next(
            (item for item in self.service.model_store.catalog() if item["name"] == candidate.model),
            None,
        )
        if entry is None:
            raise ValueError(f"unknown model: {candidate.model}")
        binding = _immutable_model_binding(entry)
        loaded = self.service.model_store.load_artifact(binding["artifact_id"])
        table, meta = self._prepared_target(binding)
        table = table.reset_index(drop=True)
        provenance: dict = {"channel": OPERATOR_CHANNELS[candidate.operator]}

        if "coral" in candidate.operator:
            source_names = [name for name in binding.get("trained_on") or [] if name]
            if not source_names:
                raise ValueError(f"model {candidate.model} declares no training datasets for CORAL")
            source_table, source_meta = self._load(source_names)
            transforms = _inference_transform_featuresets([binding])
            source_table = self.service.store.apply_feature_transforms(
                source_table, source_meta, transforms
            )
            columns = [c for c in binding.get("feature_columns") or [] if c in table.columns]
            coral = fit_coral(
                _feature_matrix(source_table, columns),
                _feature_matrix(table, columns),
                columns,
            )
            table = apply_coral(table, coral)
            provenance["coral_columns"] = columns

        calibration_mask, evaluation_mask = calibration_split(
            table, self.calibration_budget, self.calibration_split
        )
        predictions = loaded.predict(table, meta)
        truth = pd.to_numeric(table["soh"], errors="coerce").to_numpy(dtype=float)

        if "residual" in candidate.operator:
            columns = [c for c in binding.get("feature_columns") or [] if c in table.columns]
            if "residual_wide" in candidate.operator:
                from .adaptation import wide_residual_columns

                columns = wide_residual_columns(table, columns)
            features = _feature_matrix(table, columns)
            residual_model = fit_residual(
                features[calibration_mask],
                predictions[calibration_mask],
                truth[calibration_mask],
                columns,
            )
            predictions = apply_residual(features, predictions, residual_model)
            provenance["residual"] = residual_model
        elif "recalibration" in candidate.operator:
            recal = fit_recalibration(truth[calibration_mask], predictions[calibration_mask])
            predictions = apply_recalibration(predictions, recal)
            provenance["recalibration"] = recal
        if "monotone" in candidate.operator:
            predictions, monotone_info = apply_monotone(table, predictions)
            provenance["monotone"] = monotone_info

        finite = evaluation_mask & np.isfinite(truth) & np.isfinite(predictions)
        if int(finite.sum()) < 8:
            raise ValueError("too few evaluable target samples after the calibration split")
        mae = float(np.mean(np.abs(truth[finite] - predictions[finite])))
        bias = float(np.mean(predictions[finite] - truth[finite]))
        conformal_metrics = {}
        if self.interval == "conformal":
            from .adaptation import fit_conformal_interval

            conformal = fit_conformal_interval(
                truth[calibration_mask], predictions[calibration_mask]
            )
            q = conformal["quantile"]
            covered = np.abs(truth[finite] - predictions[finite]) <= q
            conformal_metrics = {
                "picp": float(np.mean(covered)),
                "interval_width": float(2.0 * q),
            }
            provenance["conformal"] = conformal
        return {
            **conformal_metrics,
            "candidate": candidate.key(),
            "model": candidate.model,
            "operator": candidate.operator,
            "target_mae": mae,
            "target_bias": bias,
            "evaluated_samples": int(finite.sum()),
            "calibration_samples": int(calibration_mask.sum()),
            "calibration_split": self.calibration_split,
            "provenance": provenance,
        }

    def run(self, policy, *, log=print) -> dict:
        history = SearchHistory()
        excluded: list[dict] = []
        for round_index in range(self.max_rounds):
            proposals = policy.propose(history)
            if not proposals:
                break
            entry = {"round": round_index, "proposed": len(proposals), "scores": [], "excluded": []}
            for candidate in proposals:
                try:
                    score = self._score(candidate)
                except Exception as exc:  # noqa: BLE001 - gates + data errors exclude
                    exclusion = {"candidate": candidate.key(), "reasons": [str(exc)]}
                    excluded.append(exclusion)
                    entry["excluded"].append(exclusion)
                    log(f"  [gate] {candidate.key()} excluded: {str(exc)[:140]}")
                    continue
                entry["scores"].append({**score, "mae_worst": score["target_mae"], "mae_mean": score["target_mae"]})
                log(f"  [target] {candidate.key()} mae={score['target_mae']:.4f} bias={score['target_bias']:+.4f}")
            history.rounds.append(entry)

        ranking = sorted(
            history.scored().values(),
            key=lambda item: (not np.isfinite(item["target_mae"]), item["target_mae"]),
        )
        return {
            "schema_version": "1.0.0",
            "mode": "adaptation",
            "policy": type(policy).__name__,
            "policy_log": list(getattr(policy, "log_entries", []) or []),
            "target_dataset": self.target_dataset,
            "calibration_budget": self.calibration_budget,
            "calibration_split": self.calibration_split,
            "interval": self.interval,
            "rounds": history.rounds,
            "excluded": excluded,
            "ranking": ranking,
            "outer_claims": [],
            "integrity": {
                "frozen_models": True,
                "operator_channels_declared": True,
                "evaluation_subset": "non-calibration samples only (identical across operators)",
                "label_budget": self.calibration_budget,
                "constraint_oracle": "trusted-inference label-safety gate (inside every predict)",
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
    parser.add_argument(
        "--window-targets",
        nargs="*",
        type=float,
        default=None,
        help="coverage targets (fractions of rated capacity) for tuned-window candidates; needs window banks",
    )
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
    report = search.run(GridPolicy(args.estimators, args.featuresets, args.window_targets))
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
