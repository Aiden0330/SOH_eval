import hashlib, json, os, subprocess, threading, unicodedata, pandas as pd, numpy as np, pytest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from sohbench.fabric.evidence import verify_evidence_pack
import sohbench.report as report_module
from sohbench.report import _leaderboard_entries, write_artifacts
from sohbench.tasks import SOHEstimationTask

def _results():
    rows=[]
    for seed in range(5):
        for metric,val in [("mae",0.01),("picp",0.92),
                           ("monotonicity_violation_rate",0.0),("value_range_violation",0.0)]:
            rows.append(dict(estimator="ridge",feature_mode="unified",backend="sklearn",
                             protocol="loco",dataset="ds",held_out_key="c1",fold=0,
                             seed=seed,metric=metric,value=val))
    return pd.DataFrame(rows)

def test_write_artifacts(tmp_path):
    out = write_artifacts(_results(), qc_report=None, task=SOHEstimationTask(),
                          config={"datasets":["ds"]}, out_dir=tmp_path)
    lb = json.loads((Path(out)/"leaderboard.json").read_text())
    entry = lb[0]
    assert entry["estimator"]=="ridge" and entry["track"]=="unified"
    assert entry["metrics"]["mae"]["mean"]==0.01
    assert entry["verdicts"]["generalization"] is True
    assert entry["coverage"] == 1.0
    assert entry["execution"] == {
        "outcome": "complete", "attempted": 5, "succeeded": 5, "failed": 0,
    }
    assert entry["comparison_key"] is None
    assert entry["confidence_intervals"] == {
        "mae": {"available": False, "reason": "predictions_not_provided"},
        "rmse": {"available": False, "reason": "predictions_not_provided"},
    }
    assert (Path(out)/"results.parquet").exists()
    assert (Path(out)/"statistics.json").exists()
    assert not (Path(out)/"predictions.parquet").exists()
    statistics = json.loads((Path(out)/"statistics.json").read_text())
    assert statistics["leaderboard"][0]["confidence_intervals"] == entry["confidence_intervals"]
    pack = verify_evidence_pack(Path(out)/"evidence-pack.json")
    assert "statistics.json" in {artifact["path"] for artifact in pack["artifacts"]}
    assert "predictions.parquet" not in {artifact["path"] for artifact in pack["artifacts"]}

def test_leaderboard_matches_schema(tmp_path):
    import jsonschema, json
    out = write_artifacts(_results(), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    lb = json.loads((Path(out)/"leaderboard.json").read_text())
    schema = json.loads((Path(__file__).parents[1]/"src/sohbench/schemas/leaderboard.schema.json").read_text())
    jsonschema.validate(lb, schema)


def test_leaderboard_schema_accepts_legacy_entries_without_task9_fields():
    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    legacy = {
        "estimator": "ridge",
        "track": "unified",
        "protocol": "loco",
        "metrics": {"mae": {"mean": 0.1, "std": 0.01}},
        "verdicts": {},
    }

    jsonschema.validate([legacy], schema)


def test_leaderboard_schema_accepts_task8_coverage_and_execution_without_task9_fields():
    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    task8 = {
        "estimator": "ridge",
        "track": "unified",
        "protocol": "loco",
        "metrics": {"mae": {"mean": 0.1, "std": 0.01}},
        "verdicts": {},
        "coverage": 1.0,
        "execution": {
            "outcome": "complete",
            "attempted": 1,
            "succeeded": 1,
            "failed": 0,
        },
    }

    jsonschema.validate([task8], schema)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comparison_key", "comparison-one"),
        (
            "confidence_intervals",
            {
                "mae": {"available": False, "reason": "predictions_not_provided"},
                "rmse": {"available": False, "reason": "predictions_not_provided"},
            },
        ),
    ],
)
def test_leaderboard_schema_rejects_partial_task9_upgrade_from_task8(field, value):
    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    entry = {
        "estimator": "ridge",
        "track": "unified",
        "protocol": "loco",
        "metrics": {},
        "verdicts": {},
        "coverage": 1.0,
        "execution": {
            "outcome": "complete",
            "attempted": 1,
            "succeeded": 1,
            "failed": 0,
        },
        field: value,
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([entry], schema)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comparison_key", "comparison-one"),
        ("coverage", 1.0),
        (
            "execution",
            {"outcome": "complete", "attempted": 1, "succeeded": 1, "failed": 0},
        ),
        (
            "confidence_intervals",
            {
                "mae": {"available": False, "reason": "predictions_not_provided"},
                "rmse": {"available": False, "reason": "predictions_not_provided"},
            },
        ),
    ],
)
def test_leaderboard_schema_rejects_partial_task9_upgrades(field, value):
    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    entry = {
        "estimator": "ridge",
        "track": "unified",
        "protocol": "loco",
        "metrics": {},
        "verdicts": {},
        field: value,
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([entry], schema)


@pytest.mark.parametrize("candidate_id", ["", " ", 7, None])
def test_leaderboard_schema_validates_candidate_id_when_present(candidate_id):
    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    entry = {
        "candidate_id": candidate_id,
        "estimator": "ridge",
        "track": "unified",
        "protocol": "loco",
        "metrics": {},
        "verdicts": {},
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([entry], schema)


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("comparison_key",), ""),
        (("coverage",), -0.01),
        (("coverage",), 1.01),
        (("coverage",), "1.0"),
        (("execution", "outcome"), "success"),
        (("execution", "attempted"), -1),
        (("execution", "succeeded"), 1.5),
        (("confidence_intervals", "mae", "low"), "0.01"),
        (("confidence_intervals", "mae", "confidence"), 1.0),
        (("confidence_intervals", "mae", "n_assets"), 0),
        (("confidence_intervals", "mae", "unexpected"), 1),
        (("confidence_intervals", "mae"), {"available": False}),
        (
            ("confidence_intervals", "mae"),
            {"available": False, "reason": "unknown_reason"},
        ),
        (("confidence_intervals", "mape"), {}),
    ],
)
def test_leaderboard_schema_rejects_malformed_task9_fields(path, invalid):
    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    entry = {
        "estimator": "ridge",
        "featureset": "full",
        "track": "unified",
        "protocol": "loco",
        "metrics": {"mae": {"mean": 0.1, "std": 0.01}},
        "verdicts": {},
        "comparison_key": "comparison-one",
        "coverage": 1.0,
        "execution": {
            "outcome": "complete",
            "attempted": 1,
            "succeeded": 1,
            "failed": 0,
        },
        "confidence_intervals": {
            "mae": {
                "available": True,
                "mean": 0.1,
                "low": 0.05,
                "high": 0.15,
                "confidence": 0.95,
                "n_assets": 2,
            },
            "rmse": {"available": False, "reason": "predictions_not_provided"},
        },
    }
    target = entry
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([entry], schema)


def test_leaderboard_schema_requires_complete_execution_and_ci_objects():
    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    base = {
        "estimator": "ridge",
        "track": "unified",
        "protocol": "loco",
        "metrics": {},
        "verdicts": {},
        "coverage": 1.0,
        "execution": {"outcome": "complete", "attempted": 1, "succeeded": 1},
        "confidence_intervals": {
            "mae": {
                "available": True,
                "mean": 0.1,
                "low": 0.0,
                "high": 0.2,
                "confidence": 0.95,
            },
            "rmse": {"available": False, "reason": "predictions_not_provided"},
        },
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([base], schema)


def _results_with_featureset():
    """Results frame that includes a featureset column (two featuresets)."""
    rows = []
    for fs in ["full", "voltage"]:
        for seed in range(5):
            for metric, val in [("mae", 0.01), ("picp", 0.92),
                                 ("monotonicity_violation_rate", 0.0),
                                 ("value_range_violation", 0.0)]:
                rows.append(dict(estimator="ridge", feature_mode="unified", backend="sklearn",
                                 protocol="loco", dataset="ds", held_out_key="c1", fold=0,
                                 seed=seed, metric=metric, value=val,
                                 featureset=fs))
    return pd.DataFrame(rows)


def test_leaderboard_featureset_column(tmp_path):
    """When results carry featureset, leaderboard entries include featureset."""
    import json
    out = write_artifacts(_results_with_featureset(), qc_report=None,
                          task=SOHEstimationTask(), config={}, out_dir=tmp_path)
    lb = json.loads((Path(out) / "leaderboard.json").read_text())
    # Both featuresets should appear
    featuresets_in_lb = {e["featureset"] for e in lb}
    assert featuresets_in_lb == {"full", "voltage"}, \
        f"expected both featuresets in leaderboard, got {featuresets_in_lb}"
    # Each entry must have the field
    for entry in lb:
        assert "featureset" in entry, f"entry missing featureset: {entry}"


def _results_mixed_tracks():
    """Results frame with both unified and byo track records."""
    rows = []
    for track, fs in [("unified", "full"), ("byo", "(byo)")]:
        for seed in range(5):
            for metric, val in [("mae", 0.01), ("picp", 0.92),
                                 ("monotonicity_violation_rate", 0.0),
                                 ("value_range_violation", 0.0)]:
                rows.append(dict(
                    estimator="ridge" if track == "unified" else "byo_skb_ridge",
                    feature_mode=track, backend="sklearn",
                    protocol="loco", dataset="ds", held_out_key="c1", fold=0,
                    seed=seed, metric=metric, value=val,
                    featureset=fs, track=track, ci_source="conformal",
                ))
    return pd.DataFrame(rows)


def test_leaderboard_separates_unified_and_byo_tracks(tmp_path):
    """A results frame mixing track='unified' and track='byo' yields separate
    leaderboard entries, each labeled with the correct track.
    Unified and byo entries must never be merged together."""
    import json
    out = write_artifacts(_results_mixed_tracks(), qc_report=None,
                          task=SOHEstimationTask(), config={}, out_dir=tmp_path)
    lb = json.loads((Path(out) / "leaderboard.json").read_text())

    unified_entries = [e for e in lb if e["track"] == "unified"]
    byo_entries = [e for e in lb if e["track"] == "byo"]

    assert len(unified_entries) >= 1, "no unified entries found"
    assert len(byo_entries) >= 1, "no byo entries found"

    # Every entry must declare exactly one track
    for entry in lb:
        assert entry["track"] in ("unified", "byo"), \
            f"unexpected track value: {entry['track']}"

    # Featuresets should match their track
    for e in unified_entries:
        assert e["featureset"] != "(byo)", \
            f"unified entry has byo featureset: {e}"
    for e in byo_entries:
        assert e["featureset"] == "(byo)", \
            f"byo entry has non-byo featureset: {e['featureset']}"


def test_nan_metrics_serialize_as_valid_json_null(tmp_path):
    """Failed/N-A folds produce NaN metrics; the JSON must stay valid (NaN -> null)
    so browsers (frontend) can JSON.parse it."""
    rows = []
    for seed in range(3):  # one estimator with all-NaN metrics (e.g. N/A on a featureset)
        for metric in ("mae", "picp", "monotonicity_violation_rate", "value_range_violation"):
            rows.append(dict(estimator="phys_resid", feature_mode="unified", backend="sklearn",
                             protocol="lodo", dataset="ds", held_out_key="ds", fold=0,
                             seed=seed, metric=metric, value=float("nan"), featureset="voltage"))
    out = write_artifacts(pd.DataFrame(rows), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    raw = (Path(out) / "leaderboard.json").read_text()
    assert "NaN" not in raw  # bare NaN token would break JSON.parse
    lb = json.loads(raw)     # must parse without error
    assert lb[0]["metrics"]["mae"]["mean"] is None
    assert lb[0]["coverage"] == 0.0
    assert lb[0]["execution"] == {
        "outcome": "failed", "attempted": 3, "succeeded": 0, "failed": 3,
    }
    assert lb[0]["verdicts"] == {}


def test_partial_results_report_attempted_execution_coverage(tmp_path):
    rows = []
    for seed, status in [(0, "succeeded"), (1, "failed")]:
        for metric, value in [("mae", 0.01), ("picp", 0.92)]:
            rows.append(
                dict(
                    estimator="ridge", feature_mode="unified", backend="sklearn",
                    protocol="loco", held_out_key="c1", fold=0, seed=seed,
                    metric=metric, value=value if status == "succeeded" else np.nan,
                    featureset="full", unit_status=status,
                )
            )

    out = write_artifacts(pd.DataFrame(rows), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]

    assert entry["coverage"] == 0.5
    assert entry["execution"] == {
        "outcome": "partial", "attempted": 2, "succeeded": 1, "failed": 1,
    }


def test_leaderboard_uses_only_finite_evidence_from_successful_units(tmp_path):
    rows = []
    metric_values = {
        "mae": (0.01, 10.0),
        "picp": (0.92, 0.0),
        "monotonicity_violation_rate": (0.0, 1.0),
        "value_range_violation": (0.0, 1.0),
    }
    for metric, (good, bad) in metric_values.items():
        rows.extend(
            [
                dict(estimator="ridge", feature_mode="unified", backend="sklearn",
                     protocol="loco", held_out_key="c1", fold=0, seed=0,
                     metric=metric, value=good, featureset="full", unit_status="succeeded"),
                dict(estimator="ridge", feature_mode="unified", backend="sklearn",
                     protocol="loco", held_out_key="c1", fold=0, seed=1,
                     metric=metric, value=bad, featureset="full", unit_status="failed"),
                dict(estimator="ridge", feature_mode="unified", backend="sklearn",
                     protocol="loco", held_out_key="c1", fold=0, seed=2,
                     metric=metric, value=np.inf, featureset="full", unit_status="succeeded"),
            ]
        )
    rows.extend(
        [
            dict(estimator="ridge", feature_mode="unified", backend="sklearn",
                 protocol="loco", held_out_key="c1", fold=0, seed=3,
                 metric="mae", value=20.0, featureset="full", unit_status="succeeded"),
            dict(estimator="ridge", feature_mode="unified", backend="sklearn",
                 protocol="loco", held_out_key="c1", fold=0, seed=3,
                 metric="picp", value=0.0, featureset="full", unit_status=None),
        ]
    )

    results = pd.DataFrame(rows, index=[0] * len(rows))
    out = write_artifacts(results, qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]

    assert entry["metrics"]["mae"] == {"mean": 0.01, "std": 0.0}
    assert entry["metrics"]["picp"] == {"mean": 0.92, "std": 0.0}
    assert entry["verdicts"] == {}
    assert entry["coverage"] == 0.5
    assert entry["execution"] == {
        "outcome": "partial", "attempted": 4, "succeeded": 2, "failed": 2,
    }


def test_succeeded_non_finite_evidence_has_no_acceptance_verdicts(tmp_path):
    values = [np.nan, np.inf, -np.inf, np.nan]
    rows = [
        dict(estimator="ridge", feature_mode="unified", backend="sklearn",
             protocol="loco", held_out_key="c1", fold=0, seed=0,
             metric=metric, value=value, featureset="full", unit_status="succeeded")
        for metric, value in zip(
            ("mae", "picp", "monotonicity_violation_rate", "value_range_violation"),
            values,
        )
    ]

    out = write_artifacts(pd.DataFrame(rows), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]

    assert all(metric["mean"] is None for metric in entry["metrics"].values())
    assert entry["verdicts"] == {}
    assert entry["coverage"] == 1.0


def test_partial_finite_metrics_do_not_produce_acceptance_verdicts(tmp_path):
    rows = [
        dict(estimator="ridge", feature_mode="unified", backend="sklearn",
             protocol="loco", held_out_key="c1", fold=0, seed=0,
             metric="mae", value=0.01, featureset="full", unit_status="succeeded"),
        dict(estimator="ridge", feature_mode="unified", backend="sklearn",
             protocol="loco", held_out_key="c1", fold=0, seed=0,
             metric="picp", value=0.92, featureset="full", unit_status="succeeded"),
    ]

    out = write_artifacts(pd.DataFrame(rows), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]

    assert entry["metrics"]["mae"] == {"mean": 0.01, "std": 0.0}
    assert entry["verdicts"] == {}


def test_disjoint_successful_units_cannot_pool_acceptance_completeness(tmp_path):
    rows = []
    unit_metrics = {
        0: [("mae", 0.01), ("picp", 0.92)],
        1: [("monotonicity_violation_rate", 0.0), ("value_range_violation", 0.0)],
    }
    for seed, metrics in unit_metrics.items():
        for metric, value in metrics:
            rows.append(
                dict(
                    estimator="ridge", feature_mode="unified", backend="sklearn",
                    protocol="loco", held_out_key="c1", fold=0, seed=seed,
                    metric=metric, value=value, featureset="full", unit_status="succeeded",
                )
            )

    out = write_artifacts(pd.DataFrame(rows), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]

    assert entry["verdicts"] == {}
    assert entry["coverage"] == 1.0
    assert entry["execution"] == {
        "outcome": "complete", "attempted": 2, "succeeded": 2, "failed": 0,
    }


def test_successful_unit_without_finite_evidence_blocks_verdicts():
    required_values = {
        "mae": 0.01,
        "picp": 0.92,
        "monotonicity_violation_rate": 0.0,
        "value_range_violation": 0.0,
    }
    invalid_values = {
        "mae": np.nan,
        "picp": "0.92",
        "monotonicity_violation_rate": False,
        "value_range_violation": np.complex64(0 + 0j),
    }
    rows = []
    for seed, values in ((0, required_values), (1, invalid_values)):
        rows.extend(
            dict(
                estimator="ridge", feature_mode="unified", backend="sklearn",
                protocol="loco", held_out_key="c1", fold=0, seed=seed,
                metric=metric, value=value, featureset="full", unit_status="succeeded",
            )
            for metric, value in values.items()
        )

    entry = _leaderboard_entries(
        pd.DataFrame(rows), SOHEstimationTask(), ["estimator", "featureset", "protocol"]
    )[0]

    assert entry["verdicts"] == {}
    assert entry["coverage"] == 1.0
    assert entry["execution"] == {
        "outcome": "complete", "attempted": 2, "succeeded": 2, "failed": 0,
    }


@pytest.mark.parametrize(
    ("invalid_metric", "invalid_value"),
    [
        ("mae", False),
        ("picp", np.complex64(0.92 + 0j)),
    ],
)
def test_non_real_acceptance_values_cannot_produce_verdicts(
    invalid_metric, invalid_value,
):
    values = {
        "mae": 0.01,
        "picp": 0.92,
        "monotonicity_violation_rate": 0.0,
        "value_range_violation": 0.0,
    }
    values[invalid_metric] = invalid_value
    rows = [
        dict(
            estimator="ridge", feature_mode="unified", backend="sklearn",
            protocol="loco", held_out_key="c1", fold=0, seed=0,
            metric=metric, value=value, featureset="full", unit_status="succeeded",
        )
        for metric, value in values.items()
    ]

    entry = _leaderboard_entries(
        pd.DataFrame(rows), SOHEstimationTask(), ["estimator", "featureset", "protocol"]
    )[0]

    assert np.isnan(entry["metrics"][invalid_metric]["mean"])
    assert entry["verdicts"] == {}


def test_complete_unit_with_python_and_numpy_real_values_produces_verdicts(tmp_path):
    values = {
        "mae": np.float32(0.01),
        "picp": np.float64(0.92),
        "monotonicity_violation_rate": 0,
        "value_range_violation": np.int64(0),
    }
    rows = [
        dict(
            estimator="ridge", feature_mode="unified", backend="sklearn",
            protocol="loco", held_out_key="c1", fold=0, seed=0,
            metric=metric, value=value, featureset="full", unit_status="succeeded",
        )
        for metric, value in values.items()
    ]

    out = write_artifacts(pd.DataFrame(rows), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]

    assert entry["verdicts"] == {
        "generalization": True,
        "calibration": True,
        "physical_consistency": True,
    }


def test_write_artifacts_emits_dataset_leaderboard(tmp_path):
    rows = []
    for held_out, mae in [("nasa", 0.20), ("hust", 0.05)]:
        for metric, val in [
            ("mae", mae),
            ("picp", 0.91),
            ("monotonicity_violation_rate", 0.0),
            ("value_range_violation", 0.0),
        ]:
            rows.append(dict(
                estimator="ridge", feature_mode="unified", backend="sklearn",
                protocol="lodo", dataset="nasa,hust", held_out_key=held_out,
                fold=0, seed=0, metric=metric, value=val,
                featureset="full", track="unified", ci_source="weighted_conformal",
            ))
    out = write_artifacts(pd.DataFrame(rows), qc_report=None,
                          task=SOHEstimationTask(), config={}, out_dir=tmp_path)

    by_dataset = json.loads((Path(out) / "dataset_leaderboard.json").read_text())
    assert {e["held_out_key"] for e in by_dataset} == {"nasa", "hust"}
    hust = next(e for e in by_dataset if e["held_out_key"] == "hust")
    assert hust["metrics"]["mae"]["mean"] == 0.05
    assert hust["verdicts"]["calibration"] is True


def test_write_inference_artifacts(tmp_path):
    from sohbench.report import write_inference_artifacts

    out = write_inference_artifacts(
        [
            {
                "model": "predict_only",
                "dataset": "new_a",
                "n_samples": 2,
                "mae": 0.01,
                "rmse": 0.02,
                "mean_bias_error": 0.0,
                "predict_ms": 1.2,
                "load_ms": 0.4,
                "ms_per_1000_samples": 600.0,
                "samples_per_second": 1666.7,
                "status": "succeeded",
                "error": "",
            }
        ],
        {"datasets": ["new_a"], "models": ["predict_only"]},
        tmp_path,
    )
    run_dir = Path(out)
    assert (run_dir / "inference_results.csv").exists()
    assert (run_dir / "inference_benchmark.json").exists()
    payload = json.loads((run_dir / "inference_benchmark.json").read_text(encoding="utf-8"))
    assert payload["records"][0]["model"] == "predict_only"


def test_inference_summary_ignores_nonfinite_accuracy_and_speed(tmp_path):
    from sohbench.report import write_inference_artifacts

    records = [
        {
            "model": "first_bad",
            "dataset": "a_unlabeled",
            "mae": float("nan"),
            "ms_per_1000_samples": float("nan"),
            "status": "succeeded",
        },
        {
            "model": "second_good",
            "dataset": "z_labeled",
            "mae": 0.0,
            "ms_per_1000_samples": 12.0,
            "status": "succeeded",
        },
    ]

    out = Path(write_inference_artifacts(records, {}, tmp_path))
    payload = json.loads(
        (out / "inference_benchmark.json").read_text(encoding="utf-8")
    )

    assert payload["summary"]["best_transfer_model"] == "second_good"
    assert payload["summary"]["best_transfer_mae"] == 0.0
    assert payload["summary"]["fastest_model"] == "second_good"
    assert payload["summary"]["fastest_ms_per_1000_samples"] == 12.0


def test_inference_summary_breaks_finite_ties_independently_of_record_order():
    records = [
        {
            "model": "beta",
            "dataset": "z_dataset",
            "featureset": "full",
            "estimator": "ridge",
            "mae": 0.1,
            "ms_per_1000_samples": 10.0,
            "status": "succeeded",
        },
        {
            "model": "alpha",
            "dataset": "a_dataset",
            "featureset": "full",
            "estimator": "ridge",
            "mae": 0.1,
            "ms_per_1000_samples": 10.0,
            "status": "succeeded",
        },
    ]

    forward = report_module._inference_summary(records)
    reverse = report_module._inference_summary(list(reversed(records)))

    assert forward == reverse
    assert forward["best_transfer_model"] == "alpha"
    assert forward["fastest_model"] == "alpha"


def _predictions(run_id="RUN-a"):
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "candidate_id": "ridge",
                "featureset": "full",
                "protocol": "loco",
                "fold": 0,
                "seed": 0,
                "dataset": "ds",
                "cell_id": "c1",
                "cycle_index": 1,
                "sample_id": "ds:c1:1",
                "y_pred": 0.9,
            }
        ]
    )


def _asset_predictions(run_id="RUN-stats"):
    rows = []
    values = {
        "c1": [(1, 1.0, 1.0)],
        "c2": [(1, 1.0, 0.0), (2, 1.0, 0.0), (3, 1.0, 0.0)],
    }
    for cell_id, samples in values.items():
        for cycle_index, y_true, y_pred in samples:
            rows.append(
                {
                    "run_id": run_id,
                    "candidate_id": "ridge",
                    "featureset": "full",
                    "protocol": "loco",
                    "fold": 0,
                    "seed": 0,
                    "dataset": "ds",
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "sample_id": f"ds:{cell_id}:{cycle_index}",
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "held_out_key": "c1",
                }
            )
    return pd.DataFrame(rows)


def _result_group(**overrides):
    identity = {
        "estimator": "ridge",
        "feature_mode": "unified",
        "backend": "sklearn",
        "protocol": "loco",
        "dataset": "ds",
        "fold": 0,
        "seed": 0,
        "featureset": "full",
        "track": "unified",
        "ci_source": "conformal",
    }
    identity.update(overrides)
    return [
        {**identity, "metric": metric, "value": value}
        for metric, value in [
            ("mae", 0.1),
            ("picp", 0.92),
            ("monotonicity_violation_rate", 0.0),
            ("value_range_violation", 0.0),
        ]
    ]


def _prediction_row(
    *,
    run_id,
    candidate_id,
    y_pred,
    dataset="ds",
    cell_id="c1",
    protocol="loco",
    featureset="full",
    fold=0,
    track="unified",
    held_out_key=None,
):
    row = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "featureset": featureset,
        "protocol": protocol,
        "fold": fold,
        "seed": 0,
        "dataset": dataset,
        "cell_id": cell_id,
        "cycle_index": 1,
        "sample_id": f"{dataset}:{cell_id}:1",
        "y_true": 0.0,
        "y_pred": y_pred,
    }
    if track is not None:
        row["track"] = track
    if held_out_key is not None:
        row["held_out_key"] = held_out_key
    return row


def test_write_artifacts_groups_and_matches_distinct_candidate_ids(tmp_path):
    results = pd.DataFrame(
        _result_group(candidate_id="candidate-a", fold=0)
        + _result_group(candidate_id="candidate-b", fold=1)
    )
    predictions = pd.DataFrame(
        [
            _prediction_row(
                run_id="RUN-candidates",
                candidate_id="candidate-a",
                cell_id="a",
                fold=0,
                y_pred=1.0,
            ),
            _prediction_row(
                run_id="RUN-candidates",
                candidate_id="candidate-b",
                cell_id="b",
                fold=1,
                y_pred=3.0,
            ),
        ]
    )

    out = Path(
        write_artifacts(
            results,
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-candidates",
            predictions=predictions,
        )
    )

    leaderboard = json.loads((out / "leaderboard.json").read_text())
    by_candidate = {entry["candidate_id"]: entry for entry in leaderboard}
    assert set(by_candidate) == {"candidate-a", "candidate-b"}
    assert {entry["estimator"] for entry in leaderboard} == {"ridge"}
    assert by_candidate["candidate-a"]["confidence_intervals"]["mae"]["mean"] == 1.0
    assert by_candidate["candidate-b"]["confidence_intervals"]["mae"]["mean"] == 3.0

    statistics = json.loads((out / "statistics.json").read_text())
    assert {entry["candidate_id"] for entry in statistics["leaderboard"]} == {
        "candidate-a",
        "candidate-b",
    }

    import jsonschema

    schema = json.loads(
        (Path(__file__).parents[1] / "src/sohbench/schemas/leaderboard.schema.json").read_text()
    )
    jsonschema.validate(leaderboard, schema)


def test_candidate_id_results_do_not_fall_back_to_estimator_matching(tmp_path):
    results = pd.DataFrame(_result_group(candidate_id="candidate-a"))
    predictions = pd.DataFrame(
        [
            _prediction_row(
                run_id="RUN-no-candidate-fallback",
                candidate_id="ridge",
                y_pred=1.0,
            )
        ]
    )

    out = Path(
        write_artifacts(
            results,
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-no-candidate-fallback",
            predictions=predictions,
        )
    )
    entry = json.loads((out / "leaderboard.json").read_text())[0]

    assert entry["candidate_id"] == "candidate-a"
    assert entry["confidence_intervals"] == {
        "mae": {"available": False, "reason": "prediction_entry_unmatched"},
        "rmse": {"available": False, "reason": "prediction_entry_unmatched"},
    }


def test_track_grouped_results_require_prediction_track_for_unambiguous_ci(tmp_path):
    results = pd.DataFrame(
        _result_group(track="unified", feature_mode="unified")
        + _result_group(track="byo", feature_mode="byo", fold=1)
    )
    predictions = pd.DataFrame(
        [
            _prediction_row(
                run_id="RUN-track-ambiguous",
                candidate_id="ridge",
                track=None,
                y_pred=1.0,
            )
        ]
    )

    out = Path(
        write_artifacts(
            results,
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-track-ambiguous",
            predictions=predictions,
        )
    )
    leaderboard = json.loads((out / "leaderboard.json").read_text())

    unavailable = {
        "mae": {"available": False, "reason": "prediction_entry_unmatched"},
        "rmse": {"available": False, "reason": "prediction_entry_unmatched"},
    }
    assert {entry["track"] for entry in leaderboard} == {"unified", "byo"}
    assert all(entry["confidence_intervals"] == unavailable for entry in leaderboard)


def test_held_out_entries_match_prediction_dataset_without_held_out_key(tmp_path):
    results = pd.DataFrame(
        _result_group(
            protocol="lodo",
            dataset="nasa,hust",
            held_out_key="nasa",
            fold=0,
        )
        + _result_group(
            protocol="lodo",
            dataset="nasa,hust",
            held_out_key="hust",
            fold=1,
        )
    )
    predictions = pd.DataFrame(
        [
            _prediction_row(
                run_id="RUN-held-out-datasets",
                candidate_id="ridge",
                protocol="lodo",
                dataset="nasa",
                cell_id="nasa-cell",
                fold=0,
                y_pred=1.0,
            ),
            _prediction_row(
                run_id="RUN-held-out-datasets",
                candidate_id="ridge",
                protocol="lodo",
                dataset="hust",
                cell_id="hust-cell",
                fold=1,
                y_pred=3.0,
            ),
        ]
    )

    out = Path(
        write_artifacts(
            results,
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-held-out-datasets",
            predictions=predictions,
        )
    )
    dataset_leaderboard = json.loads((out / "dataset_leaderboard.json").read_text())
    by_held_out = {entry["held_out_key"]: entry for entry in dataset_leaderboard}

    assert by_held_out["nasa"]["confidence_intervals"]["mae"]["mean"] == 1.0
    assert by_held_out["hust"]["confidence_intervals"]["mae"]["mean"] == 3.0


def test_held_out_dataset_fallback_is_unavailable_for_lcho(tmp_path):
    results = pd.DataFrame(
        _result_group(
            protocol="lcho",
            dataset="nasa,hust",
            held_out_key="nasa",
        )
    )
    predictions = pd.DataFrame(
        [
            _prediction_row(
                run_id="RUN-lcho-held-out-datasets",
                candidate_id="ridge",
                protocol="lcho",
                dataset="nasa",
                cell_id="nasa-cell",
                y_pred=1.0,
            ),
        ]
    )

    out = Path(
        write_artifacts(
            results,
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-lcho-held-out-datasets",
            predictions=predictions,
        )
    )
    entry = json.loads((out / "dataset_leaderboard.json").read_text())[0]

    assert entry["confidence_intervals"] == {
        "mae": {"available": False, "reason": "prediction_entry_unmatched"},
        "rmse": {"available": False, "reason": "prediction_entry_unmatched"},
    }


@pytest.mark.parametrize("candidate_id", [None, np.nan, 7, "", " "])
def test_write_artifacts_rejects_invalid_result_candidate_id_before_reservation(
    tmp_path, candidate_id,
):
    run_id = "RUN-invalid-result-candidate"
    results = pd.DataFrame(_result_group(candidate_id=candidate_id))

    with pytest.raises(ValueError, match="candidate_id"):
        write_artifacts(
            results, None, SOHEstimationTask(), {}, tmp_path, run_id=run_id,
        )

    assert not (tmp_path / report_module._reservation_name(run_id)).exists()
    assert not (tmp_path / run_id).exists()


def test_write_artifacts_uses_explicit_run_id_and_dual_writes_pack(tmp_path):
    out = Path(
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-a",
            evidence_context={"comparison_key": "comparison-one"},
            predictions=_predictions(),
        )
    )

    assert out.name == "RUN-a"
    assert (out / "results.parquet").exists()
    assert (out / "results.csv").exists()
    assert (out / "leaderboard.json").exists()
    assert (out / "dataset_leaderboard.json").exists()
    assert (out / "run_config.json").exists()
    assert (out / "predictions.parquet").exists()
    assert (out / "statistics.json").exists()
    leaderboard = json.loads((out / "leaderboard.json").read_text())
    statistics = json.loads((out / "statistics.json").read_text())
    assert leaderboard[0]["comparison_key"] == "comparison-one"
    unavailable = {
        "mae": {"available": False, "reason": "prediction_truth_unavailable"},
        "rmse": {"available": False, "reason": "prediction_truth_unavailable"},
    }
    assert leaderboard[0]["confidence_intervals"] == unavailable
    assert statistics["comparison_key"] == "comparison-one"
    assert statistics["leaderboard"][0]["confidence_intervals"] == unavailable
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert pack["run_id"] == "RUN-a"
    assert pack["comparison_key"] == "comparison-one"
    assert [item["path"] for item in pack["artifacts"]] == sorted(
        item["path"] for item in pack["artifacts"]
    )
    assert "evidence-pack.json" not in {item["path"] for item in pack["artifacts"]}
    assert "statistics.json" in {item["path"] for item in pack["artifacts"]}


def test_write_artifacts_uses_asset_level_prediction_metrics_for_intervals(tmp_path):
    out = Path(
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-stats",
            evidence_context={"comparison_key": "comparison-one"},
            predictions=_asset_predictions(),
        )
    )

    leaderboard = json.loads((out / "leaderboard.json").read_text())
    dataset_leaderboard = json.loads((out / "dataset_leaderboard.json").read_text())
    statistics = json.loads((out / "statistics.json").read_text())
    entry = leaderboard[0]

    assert entry["comparison_key"] == "comparison-one"
    assert entry["metrics"]["mae"] == {"mean": 0.01, "std": 0.0}
    assert set(entry["confidence_intervals"]) == {"mae", "rmse"}
    for interval in entry["confidence_intervals"].values():
        assert interval["available"] is True
        assert interval["mean"] == 0.5
        assert interval["n_assets"] == 2
        assert interval["confidence"] == 0.95
        assert interval["low"] <= interval["mean"] <= interval["high"]

    assert dataset_leaderboard[0]["confidence_intervals"] == entry["confidence_intervals"]
    assert statistics == {
        "schema_version": "1.0.0",
        "comparison_key": "comparison-one",
        "bootstrap_unit": "asset",
        "confidence": 0.95,
        "n_resamples": 2000,
        "seed": 0,
        "leaderboard": [
            {
                "estimator": "ridge",
                "featureset": "full",
                "track": "unified",
                "protocol": "loco",
                "confidence_intervals": entry["confidence_intervals"],
            }
        ],
        "dataset_leaderboard": [
            {
                "estimator": "ridge",
                "featureset": "full",
                "track": "unified",
                "protocol": "loco",
                "held_out_key": "c1",
                "confidence_intervals": entry["confidence_intervals"],
            }
        ],
    }
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert "statistics.json" in {artifact["path"] for artifact in pack["artifacts"]}


def _serialized_run_spec_document():
    from sohbench.fabric.specs import RunSpec, SpecRef

    def ref(kind, spec_id):
        return SpecRef(kind=kind, spec_id=spec_id, version="1.0.0", digest="a" * 64)

    return RunSpec(
        run_type="training",
        task=ref("task", "soh"),
        dataset=ref("dataset_snapshot", "snapshot"),
        protocol=ref("protocol", "loco"),
        metric=ref("metric", "core"),
        policy=ref("policy", "fair"),
        candidates=(ref("candidate", "ridge"),),
    ).to_dict()


def test_write_artifacts_derives_comparison_key_from_serialized_run_spec(tmp_path):
    from sohbench.fabric.ids import sha256_json

    run_spec = _serialized_run_spec_document()
    expected_key = sha256_json(
        {
            field: run_spec[field]
            for field in ("task", "dataset", "protocol", "metric", "policy")
        }
    )

    out = Path(
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-serialized-spec",
            evidence_context={"run_spec": run_spec},
        )
    )

    leaderboard = json.loads((out / "leaderboard.json").read_text())
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert leaderboard[0]["comparison_key"] == expected_key
    assert pack["comparison_key"] == expected_key
    assert pack["run_spec"] == run_spec


def test_write_artifacts_validates_serialized_run_spec_before_reservation(tmp_path):
    run_spec = _serialized_run_spec_document()
    run_spec.pop("metric")

    with pytest.raises(ValueError, match="metric|run_spec"):
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-invalid-serialized-spec",
            evidence_context={"run_spec": run_spec},
        )

    assert list(tmp_path.iterdir()) == []


def test_serialized_run_spec_rejects_conflicting_key_before_reservation(tmp_path):
    run_spec = _serialized_run_spec_document()

    with pytest.raises(ValueError, match="comparison_key"):
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-conflicting-serialized-spec",
            evidence_context={"run_spec": run_spec, "comparison_key": "conflict"},
        )

    assert list(tmp_path.iterdir()) == []


def test_write_artifacts_derives_comparison_key_from_run_spec_context(tmp_path):
    from sohbench.fabric.specs import RunSpec, SpecRef

    def ref(kind, spec_id):
        return SpecRef(kind=kind, spec_id=spec_id, version="1.0.0", digest="a" * 64)

    run_spec = RunSpec(
        run_type="training",
        task=ref("task", "soh"),
        dataset=ref("dataset_snapshot", "snapshot"),
        protocol=ref("protocol", "loco"),
        metric=ref("metric", "core"),
        policy=ref("policy", "fair"),
        candidates=(ref("candidate", "ridge"),),
    )

    out = Path(
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-spec",
            evidence_context={"run_spec": run_spec},
        )
    )

    leaderboard = json.loads((out / "leaderboard.json").read_text())
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert leaderboard[0]["comparison_key"] == run_spec.comparison_key
    assert pack["comparison_key"] == run_spec.comparison_key
    assert pack["run_spec"] == run_spec.to_dict()


def test_write_artifacts_rejects_comparison_key_conflicting_with_run_spec(tmp_path):
    from sohbench.fabric.specs import RunSpec, SpecRef

    def ref(kind, spec_id):
        return SpecRef(kind=kind, spec_id=spec_id, version="1.0.0", digest="a" * 64)

    run_spec = RunSpec(
        run_type="training",
        task=ref("task", "soh"),
        dataset=ref("dataset_snapshot", "snapshot"),
        protocol=ref("protocol", "loco"),
        metric=ref("metric", "core"),
        policy=ref("policy", "fair"),
        candidates=(ref("candidate", "ridge"),),
    )

    with pytest.raises(ValueError, match="comparison_key"):
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-spec",
            evidence_context={"run_spec": run_spec, "comparison_key": "conflict"},
        )
    assert not (tmp_path / "RUN-spec").exists()


def test_write_artifacts_marks_unmatched_prediction_entries_unavailable(tmp_path):
    predictions = _asset_predictions().copy()
    predictions["candidate_id"] = "other"

    out = Path(
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-stats",
            predictions=predictions,
        )
    )
    entry = json.loads((out / "leaderboard.json").read_text())[0]

    assert entry["confidence_intervals"] == {
        "mae": {"available": False, "reason": "prediction_entry_unmatched"},
        "rmse": {"available": False, "reason": "prediction_entry_unmatched"},
    }


def test_write_artifacts_uses_distinct_generated_run_ids(tmp_path):
    first = Path(write_artifacts(_results(), None, SOHEstimationTask(), {}, tmp_path))
    second = Path(write_artifacts(_results(), None, SOHEstimationTask(), {}, tmp_path))

    assert first != second
    assert first.name.startswith("RUN-")
    assert second.name.startswith("RUN-")


@pytest.mark.parametrize("occupied", [False, True])
def test_write_artifacts_rejects_every_preexisting_target(tmp_path, occupied):
    run_id = f"RUN-existing-{occupied}"
    target = tmp_path / run_id
    target.mkdir()
    marker = target / "marker.txt"
    if occupied:
        marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="exists|target"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert target.is_dir()
    assert list(target.iterdir()) == ([marker] if occupied else [])
    if occupied:
        assert marker.read_text(encoding="utf-8") == "do not overwrite"
    reservation = (
        tmp_path
        / report_module._reservation_name(run_id)
    )
    assert reservation.is_dir()


@pytest.mark.parametrize(
    "run_id",
    [
        ".",
        "..",
        "nested/RUN",
        r"nested\RUN",
        "RUN:stream",
        "RUN.",
        "RUN ",
        "RUN<bad",
        "RUN>bad",
        'RUN"bad',
        "RUN|bad",
        "RUN?bad",
        "RUN*bad",
        "RUN\u001fbad",
        "CON",
        "con.log",
        "PRN",
        "AUX.txt",
        "NUL",
        "COM1.log",
        "lpt9.txt",
        "COM\u00b9",
        "com\u00b2.txt",
        "CoM\u00b3.log",
        "LPT\u00b9",
        "lpt\u00b2.txt",
        "LpT\u00b3.log",
    ],
)
def test_write_artifacts_rejects_nonportable_run_ids_before_creating_target(
    tmp_path, run_id,
):
    with pytest.raises(ValueError, match="run_id"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "run_id",
    [
        ".sohbench-run-reservation-user",
        ".SOHBENCH-RUN-RESERVATION-" + "a" * 64,
    ],
)
def test_write_artifacts_rejects_reservation_prefix_as_run_id(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id|reserved"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path,
            run_id=run_id,
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("requested", ["RUN-Case", "run-case"])
def test_write_artifacts_rejects_case_equivalent_existing_run_directory(
    tmp_path, requested,
):
    root = tmp_path / requested
    root.mkdir()
    physical = root / "RUN-Case"
    physical.mkdir()

    with pytest.raises(FileExistsError, match="exists|case|target"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, root, run_id=requested
        )

    assert list(physical.iterdir()) == []


def _run_concurrent_writers_at_reservation(tmp_path, monkeypatch, run_ids):
    root = tmp_path / "runs"
    root.mkdir()
    reservation_barrier = threading.Barrier(2)
    original_acquire = report_module._acquire_run_reservation

    def synchronized_acquire(root_path, run_id):
        reservation_barrier.wait(timeout=10)
        return original_acquire(root_path, run_id)

    monkeypatch.setattr(
        report_module, "_acquire_run_reservation", synchronized_acquire
    )

    def run_writer(role, run_id):
        try:
            return Path(
                write_artifacts(
                    _results(),
                    None,
                    SOHEstimationTask(),
                    {"writer": role},
                    root,
                    run_id=run_id,
                )
            )
        except BaseException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_writer, "first", run_ids[0]),
            executor.submit(run_writer, "second", run_ids[1]),
        ]
        outcomes = [future.result(timeout=20) for future in futures]

    return root, list(zip(("first", "second"), run_ids, outcomes))


def _assert_single_reserved_writer(root, outcomes):
    successes = [item for item in outcomes if isinstance(item[2], Path)]
    failures = [item for item in outcomes if isinstance(item[2], BaseException)]
    if len(successes) == 2:
        assert successes[0][2].resolve() != successes[1][2].resolve(), (
            "concurrent writers returned the same physical run directory"
        )
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0][2], (FileExistsError, ValueError))
    assert any(
        word in str(failures[0][2]).lower()
        for word in ("reservation", "exists", "collision", "case")
    )

    role, run_id, run_dir = successes[0]
    assert json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) == {
        "writer": role
    }
    pack = verify_evidence_pack(run_dir / "evidence-pack.json")
    assert pack["run_id"] == run_id
    artifact_paths = {item["path"] for item in pack["artifacts"]}
    assert not any("reservation" in path for path in artifact_paths)
    assert list(run_dir.glob(".*")) == []
    reservation = root / report_module._reservation_name(run_id)
    assert reservation.is_dir()
    assert sorted(entry.name for entry in root.iterdir()) == sorted(
        [run_dir.name, reservation.name]
    )


def test_concurrent_exact_run_ids_allow_only_one_artifact_transaction(
    tmp_path, monkeypatch,
):
    root, outcomes = _run_concurrent_writers_at_reservation(
        tmp_path, monkeypatch, ("RUN-Race", "RUN-Race")
    )

    _assert_single_reserved_writer(root, outcomes)


def test_concurrent_case_equivalent_run_ids_allow_only_one_artifact_transaction(
    tmp_path, monkeypatch,
):
    root, outcomes = _run_concurrent_writers_at_reservation(
        tmp_path, monkeypatch, ("RUN-Race", "run-race")
    )

    _assert_single_reserved_writer(root, outcomes)


def test_concurrent_canonical_equivalent_run_ids_allow_only_one_transaction(
    tmp_path, monkeypatch,
):
    root, outcomes = _run_concurrent_writers_at_reservation(
        tmp_path, monkeypatch, ("RUN-é", "RUN-e\u0301")
    )

    _assert_single_reserved_writer(root, outcomes)


@pytest.mark.parametrize("reservation_kind", ["file", "directory"])
def test_stale_durable_reservation_fails_closed(tmp_path, reservation_kind):
    run_id = "RUN-stale"
    reservation = tmp_path / report_module._reservation_name(run_id)
    if reservation_kind == "file":
        reservation.write_bytes(b"stale reservation")
    else:
        reservation.mkdir()

    with pytest.raises(FileExistsError, match="reserved|reservation"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    if reservation_kind == "file":
        assert reservation.read_bytes() == b"stale reservation"
    else:
        assert reservation.is_dir()
    assert not (tmp_path / run_id).exists()


def test_inference_keeps_reservation_outside_pack_and_final_run(
    tmp_path, monkeypatch,
):
    from sohbench.report import write_inference_artifacts

    original_write_pack = report_module.write_evidence_pack
    observed_reservations = []

    def observe_claim(run_dir, manifest, artifact_paths):
        reservation = (
            Path(run_dir).parent
            / report_module._reservation_name(manifest["run_id"])
        )
        observed_reservations.append(reservation.is_dir())
        return original_write_pack(run_dir, manifest, artifact_paths)

    monkeypatch.setattr(report_module, "write_evidence_pack", observe_claim)
    out = Path(
        write_inference_artifacts(
            [{"model": "predict_only", "dataset": "new_a", "status": "succeeded"}],
            {},
            tmp_path,
            run_id="RUN-inference-claim",
        )
    )

    assert observed_reservations == [True]
    assert not any(
        path.name.startswith(".sohbench-run-reservation-")
        for path in out.iterdir()
    )
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert not any("reservation" in item["path"] for item in pack["artifacts"])


def test_reservation_transaction_uses_no_ephemeral_claim_metadata(
    tmp_path, monkeypatch,
):
    real_lstat = Path.lstat

    def reject_ephemeral_claim_lstat(path):
        if path.name == ".sohbench-run.claim":
            pytest.fail("ephemeral claim metadata was consulted")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", reject_ephemeral_claim_lstat)

    with report_module._run_directory_transaction(
        tmp_path, "RUN-no-ephemeral-claim"
    ) as out:
        assert out.is_dir()

    assert not any(
        path.name == ".sohbench-run.claim" for path in tmp_path.rglob("*")
    )


def _expected_reservation_key(run_id):
    canonical = unicodedata.normalize(
        "NFC", unicodedata.normalize("NFC", run_id).casefold()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _expected_reservation_name(run_id):
    return ".sohbench-run-reservation-" + _expected_reservation_key(run_id)


@pytest.mark.parametrize(
    "first, second",
    [
        ("RUN-Race", "run-race"),
        ("RUN-Straße", "run-strasse"),
        ("RUN-é", "run-e\u0301"),
    ],
)
def test_reservation_key_uses_unicode_casefold_canonical_form(first, second):
    expected = _expected_reservation_key(first)

    assert report_module._reservation_key(first) == expected
    assert report_module._reservation_key(second) == expected
    assert report_module._reservation_name(first) == _expected_reservation_name(first)
    assert report_module._reservation_name(second) == _expected_reservation_name(first)


def test_final_target_creation_race_preserves_replacement_and_reservation(
    tmp_path, monkeypatch,
):
    run_id = "RUN-target-race"
    target = tmp_path / run_id
    marker = target / "replacement.txt"
    real_mkdir = Path.mkdir
    injected = False

    def inject_replacement(path, *args, **kwargs):
        nonlocal injected
        if path == target and not injected:
            injected = True
            real_mkdir(path)
            marker.write_text("replacement survives", encoding="utf-8")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", inject_replacement)
    with pytest.raises(FileExistsError, match="exists|target|collision"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert marker.read_text(encoding="utf-8") == "replacement survives"
    reservation = tmp_path / report_module._reservation_name(run_id)
    assert reservation.is_dir()


def test_canonical_alias_in_final_mkdir_window_fails_closed(
    tmp_path, monkeypatch,
):
    run_id = "RUN-é"
    target = tmp_path / run_id
    alias = tmp_path / "RUN-e\u0301"
    real_mkdir = Path.mkdir
    injected = False

    def inject_canonical_alias(path, *args, **kwargs):
        nonlocal injected
        if path == target and not injected:
            injected = True
            real_mkdir(alias)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", inject_canonical_alias)
    with pytest.raises(FileExistsError, match="canonical|collision|exists"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    reservation = tmp_path / report_module._reservation_name(run_id)
    assert reservation.is_dir()
    assert target.exists()
    assert alias.exists()
    assert not (target / "evidence-pack.json").exists()
    assert not (alias / "evidence-pack.json").exists()


def _replace_with_distinct_empty_directory(path):
    original_identity = (path.lstat().st_dev, path.lstat().st_ino)
    path.rmdir()
    for index in range(16):
        holder = path.with_name(f"{path.name}.identity-holder-{index}")
        holder.mkdir()
        path.mkdir()
        replacement_identity = (path.lstat().st_dev, path.lstat().st_ino)
        if replacement_identity != original_identity:
            return original_identity, replacement_identity
        path.rmdir()
    pytest.fail("filesystem repeatedly reused the original directory identity")


@pytest.mark.parametrize("occupied", [False, True], ids=["empty", "occupied"])
def test_final_handoff_rejects_replaced_ordinary_directory(
    tmp_path, monkeypatch, occupied,
):
    run_id = f"RUN-handoff-{'occupied' if occupied else 'empty'}"
    target = tmp_path / run_id
    marker = target / "replacement.txt"
    original_scan = report_module._require_sole_canonical_entry
    replaced = False
    identities = []

    def replace_after_scan(root, expected, label):
        nonlocal replaced
        original_scan(root, expected, label)
        if label == "target run directory" and not replaced:
            replaced = True
            identities.extend(_replace_with_distinct_empty_directory(expected))
            if occupied:
                marker.write_bytes(b"replacement survives")

    monkeypatch.setattr(
        report_module, "_require_sole_canonical_entry", replace_after_scan
    )
    with pytest.raises(ValueError, match="identity|handoff|empty|replaced"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert replaced
    assert identities[0] != identities[1]
    assert target.is_dir()
    if occupied:
        assert marker.read_bytes() == b"replacement survives"
    else:
        assert list(target.iterdir()) == []
    assert not (target / "results.parquet").exists()
    assert not (target / "evidence-pack.json").exists()


def test_reservation_handoff_rejects_replaced_directory(tmp_path, monkeypatch):
    run_id = "RUN-reservation-handoff"
    reservation = tmp_path / _expected_reservation_name(run_id)
    original_scan = report_module._require_sole_canonical_entry
    replaced = False
    identities = []

    def replace_after_scan(root, expected, label):
        nonlocal replaced
        original_scan(root, expected, label)
        if label == "run reservation" and not replaced:
            replaced = True
            identities.extend(_replace_with_distinct_empty_directory(expected))

    monkeypatch.setattr(
        report_module, "_require_sole_canonical_entry", replace_after_scan
    )
    with pytest.raises(ValueError, match="identity|handoff|replaced"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert replaced
    assert identities[0] != identities[1]
    assert reservation.is_dir()
    assert list(reservation.iterdir()) == []
    assert not (tmp_path / run_id).exists()


@pytest.mark.skipif(os.name != "nt", reason="directory junctions require Windows")
def test_final_handoff_rejects_replacement_junction(tmp_path, monkeypatch):
    run_id = "RUN-handoff-junction"
    target = tmp_path / run_id
    external = tmp_path / "external-handoff-target"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    original_scan = report_module._require_sole_canonical_entry
    replaced = False

    def replace_after_scan(root, expected, label):
        nonlocal replaced
        original_scan(root, expected, label)
        if label == "target run directory" and not replaced:
            replaced = True
            expected.rmdir()
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(expected), str(external)],
                capture_output=True,
                text=True,
            )
            if created.returncode:
                pytest.skip(
                    "directory junction creation unavailable: "
                    f"{created.stdout}{created.stderr}"
                )

    monkeypatch.setattr(
        report_module, "_require_sole_canonical_entry", replace_after_scan
    )
    try:
        with pytest.raises(ValueError, match="handoff|ordinary|reparse|link|identity"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
            )
        assert replaced
        assert target.is_junction()
        assert sentinel.read_bytes() == b"external sentinel"
        assert list(external.iterdir()) == [sentinel]
        assert not (external / "results.parquet").exists()
        assert not (external / "evidence-pack.json").exists()
    finally:
        if target.exists():
            os.rmdir(target)


@pytest.mark.parametrize("directory_kind", ["reservation", "final"])
def test_second_handoff_snapshot_rejects_late_canonical_sibling(
    tmp_path, monkeypatch, directory_kind,
):
    run_id = "RUN-\u00e9"
    target = tmp_path / run_id
    reservation = tmp_path / _expected_reservation_name(run_id)
    expected = reservation if directory_kind == "reservation" else target
    label = "run reservation" if directory_kind == "reservation" else "target run directory"
    alias = (
        tmp_path / expected.name.upper()
        if directory_kind == "reservation"
        else tmp_path / "RUN-e\u0301"
    )
    original_scan = report_module._require_sole_canonical_entry
    injected = False

    def inject_alias_after_first_scan(root, scanned, scanned_label):
        nonlocal injected
        original_scan(root, scanned, scanned_label)
        if scanned_label == label and not injected:
            injected = True
            try:
                alias.mkdir()
            except FileExistsError:
                pytest.skip(
                    "filesystem cannot represent the canonical reservation sibling"
                )

    monkeypatch.setattr(
        report_module, "_require_sole_canonical_entry", inject_alias_after_first_scan
    )
    with pytest.raises(FileExistsError, match="canonical|collision|alias"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert injected
    assert expected.is_dir()
    assert alias.is_dir()
    assert not (target / "evidence-pack.json").exists()
    if directory_kind == "reservation":
        assert not target.exists()


@pytest.mark.parametrize("directory_kind", ["reservation", "final"])
def test_second_handoff_snapshot_rejects_late_foreign_marker(
    tmp_path, monkeypatch, directory_kind,
):
    run_id = f"RUN-late-marker-{directory_kind}"
    target = tmp_path / run_id
    reservation = tmp_path / _expected_reservation_name(run_id)
    selected = reservation if directory_kind == "reservation" else target
    marker = selected / "foreign-marker.txt"
    real_scandir = report_module.os.scandir
    injected = False

    @contextmanager
    def inject_marker_after_first_empty_scan(path):
        nonlocal injected
        with real_scandir(path) as entries:
            yield entries
        if Path(path) == selected and not injected:
            injected = True
            marker.write_bytes(b"foreign data survives")

    monkeypatch.setattr(report_module.os, "scandir", inject_marker_after_first_empty_scan)
    with pytest.raises(ValueError, match="empty|handoff"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert injected
    assert marker.read_bytes() == b"foreign data survives"
    assert not (target / "evidence-pack.json").exists()
    assert not (target / "results.parquet").exists()
    if directory_kind == "reservation":
        assert not target.exists()


def test_partial_artifact_failure_leaves_durable_fail_closed_history(
    tmp_path, monkeypatch,
):
    run_id = "RUN-partial"

    def fail_pack(*args, **kwargs):
        raise RuntimeError("injected artifact failure")

    monkeypatch.setattr(report_module, "write_evidence_pack", fail_pack)
    with pytest.raises(RuntimeError, match="injected artifact failure"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    run_dir = tmp_path / run_id
    reservation = tmp_path / report_module._reservation_name(run_id)
    assert reservation.is_dir()
    assert run_dir.is_dir()
    assert (run_dir / "results.parquet").is_file()
    assert not (run_dir / "evidence-pack.json").exists()
    partial_entries = sorted(path.name for path in run_dir.iterdir())

    with pytest.raises(FileExistsError, match="reserved|reservation"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert sorted(path.name for path in run_dir.iterdir()) == partial_entries
    assert reservation.is_dir()


def test_success_keeps_reservation_outside_final_run_and_pack(tmp_path):
    run_id = "RUN-durable"
    out = Path(
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )
    )
    reservation = tmp_path / report_module._reservation_name(run_id)

    assert reservation.is_dir()
    assert reservation.parent == tmp_path
    assert not (out / reservation.name).exists()
    assert not (out / ".sohbench-run.claim").exists()
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert not any(
        "reservation" in entry["path"] or "claim" in entry["path"]
        for entry in pack["artifacts"]
    )


def test_writer_performs_no_destructive_run_cleanup(tmp_path, monkeypatch):
    def reject_rmdir(path):
        pytest.fail(f"destructive run cleanup attempted for {path}")

    monkeypatch.setattr(Path, "rmdir", reject_rmdir)
    out = Path(
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path,
            run_id="RUN-no-cleanup",
        )
    )

    assert (out / "evidence-pack.json").is_file()


def test_canonical_reservation_alias_fails_closed(tmp_path):
    run_id = "RUN-reservation-alias"
    exact_name = _expected_reservation_name(run_id)
    alias = tmp_path / exact_name.upper()
    alias.mkdir()

    with pytest.raises(FileExistsError, match="reservation|alias|canonical"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert alias.is_dir()
    assert not (tmp_path / run_id).exists()


def test_canonical_reservation_alias_in_mkdir_window_fails_closed(
    tmp_path, monkeypatch,
):
    run_id = "RUN-reservation-window"
    reservation = tmp_path / _expected_reservation_name(run_id)
    alias = tmp_path / reservation.name.upper()
    real_mkdir = Path.mkdir
    injected = False

    def inject_reservation_alias(path, *args, **kwargs):
        nonlocal injected
        if path == reservation and not injected:
            injected = True
            real_mkdir(alias)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", inject_reservation_alias)
    with pytest.raises(FileExistsError, match="reservation|alias|canonical"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert alias.is_dir()
    assert not (tmp_path / run_id).exists()


def test_exact_and_canonical_reservation_alias_coexistence_is_rejected(tmp_path):
    run_id = "RUN-reservation-alias-pair"
    exact = tmp_path / _expected_reservation_name(run_id)
    alias = tmp_path / exact.name.upper()
    exact.mkdir()
    try:
        alias.mkdir()
    except FileExistsError:
        pytest.skip("filesystem cannot represent case-equivalent sibling names")

    with pytest.raises(FileExistsError, match="reservation|alias|canonical"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )

    assert exact.is_dir()
    assert alias.is_dir()
    assert not (tmp_path / run_id).exists()


@pytest.mark.skipif(os.name != "nt", reason="directory junctions require Windows")
def test_direct_reservation_replaced_by_junction_after_mkdir_fails_closed(
    tmp_path, monkeypatch,
):
    run_id = "RUN-reservation-junction"
    reservation = tmp_path / _expected_reservation_name(run_id)
    target = tmp_path / "external-reservation-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    real_mkdir = Path.mkdir
    injected = False

    def replace_with_junction(path, *args, **kwargs):
        nonlocal injected
        result = real_mkdir(path, *args, **kwargs)
        if path == reservation and not injected:
            injected = True
            path.rmdir()
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(path), str(target)],
                capture_output=True,
                text=True,
            )
            if created.returncode:
                pytest.skip(
                    "directory junction creation unavailable: "
                    f"{created.stdout}{created.stderr}"
                )
        return result

    monkeypatch.setattr(Path, "mkdir", replace_with_junction)
    try:
        with pytest.raises(ValueError, match="reservation|ordinary|reparse|link"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
            )
        assert reservation.is_junction()
        assert sentinel.read_bytes() == b"external sentinel"
        assert list(target.iterdir()) == [sentinel]
        assert not (tmp_path / run_id).exists()
    finally:
        if reservation.exists():
            os.rmdir(reservation)


@pytest.mark.skipif(os.name != "nt", reason="directory junctions require Windows")
def test_former_registry_junction_window_cannot_receive_reservation(
    tmp_path, monkeypatch,
):
    run_id = "RUN-no-registry-window"
    former_registry = tmp_path / ".sohbench-run-reservations"
    former_registry.mkdir()
    target = tmp_path / "external-registry-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")

    probe = tmp_path / "junction-probe"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(probe), str(target)],
        capture_output=True,
        text=True,
    )
    if created.returncode:
        pytest.skip(
            "directory junction creation unavailable: "
            f"{created.stdout}{created.stderr}"
        )
    os.rmdir(probe)

    real_mkdir = Path.mkdir
    injected = False

    def inject_former_registry_swap(path, *args, **kwargs):
        nonlocal injected
        if path.parent == former_registry and not injected:
            injected = True
            former_registry.rmdir()
            replaced = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(former_registry), str(target)],
                capture_output=True,
                text=True,
            )
            assert replaced.returncode == 0, replaced.stdout + replaced.stderr
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", inject_former_registry_swap)
    try:
        out = Path(
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
            )
        )
        assert not injected
        assert (tmp_path / _expected_reservation_name(run_id)).is_dir()
        assert (out / "evidence-pack.json").is_file()
        assert sentinel.read_bytes() == b"external sentinel"
        assert list(target.iterdir()) == [sentinel]
    finally:
        if former_registry.exists() and former_registry.is_junction():
            os.rmdir(former_registry)

def test_ephemeral_claim_cleanup_implementation_is_removed():
    obsolete_names = (
        "_acquire_run_claim",
        "_cleanup_run_claim",
        "_portable_delete_claim_directory",
        "_windows_open_claim_for_delete",
        "_windows_mark_claim_for_deletion",
        "_windows_close_claim_handle",
        "_windows_delete_claim_directory",
        "_select_reservation_registry",
        "_validate_reservation_registry",
        "_ensure_reservation_registry",
    )

    assert not any(hasattr(report_module, name) for name in obsolete_names)

@pytest.mark.parametrize("writer_kind", ["training", "inference"])
def test_writers_share_run_directory_transaction_helper(
    tmp_path, monkeypatch, writer_kind,
):
    original_transaction = report_module._run_directory_transaction
    observed = []

    @contextmanager
    def observe_transaction(out_dir, run_id):
        observed.append(("enter", run_id))
        with original_transaction(out_dir, run_id) as run_dir:
            yield run_dir
        observed.append(("exit", run_id))

    monkeypatch.setattr(
        report_module, "_run_directory_transaction", observe_transaction
    )
    run_id = f"RUN-shared-{writer_kind}"
    if writer_kind == "training":
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id=run_id
        )
    else:
        report_module.write_inference_artifacts(
            [{"model": "predict_only", "dataset": "new_a", "status": "succeeded"}],
            {},
            tmp_path,
            run_id=run_id,
        )

    assert observed == [("enter", run_id), ("exit", run_id)]


def test_write_artifacts_rejects_existing_directory_symlink(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    target = tmp_path / "symlink-target"
    target.mkdir()
    link = root / "RUN-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    try:
        with pytest.raises(FileExistsError, match="exists|target"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, root, run_id="RUN-link"
            )
        assert list(target.iterdir()) == []
        assert link.is_symlink()
    finally:
        if link.is_symlink():
            link.unlink()


@pytest.mark.skipif(os.name != "nt", reason="directory junctions require Windows")
def test_write_artifacts_rejects_existing_directory_junction(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    target = tmp_path / "junction-target"
    target.mkdir()
    link = root / "RUN-link"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if created.returncode:
        pytest.skip(
            "directory junction creation unavailable: "
            f"{created.stdout}{created.stderr}"
        )

    try:
        with pytest.raises(FileExistsError, match="exists|target"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, root, run_id="RUN-link"
            )
        assert list(target.iterdir()) == []
        assert link.is_junction()
    finally:
        if link.exists():
            os.rmdir(link)


@pytest.mark.parametrize("reserved", ["schema_version", "artifacts"])
def test_write_artifacts_rejects_reserved_evidence_context(tmp_path, reserved):
    with pytest.raises(ValueError, match=reserved):
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-a",
            evidence_context={reserved: "override"},
        )


def test_write_artifacts_does_not_replace_explicit_run_id_from_context(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        write_artifacts(
            _results(),
            None,
            SOHEstimationTask(),
            {},
            tmp_path,
            run_id="RUN-explicit",
            evidence_context={"run_id": "RUN-context"},
        )
    assert not (tmp_path / "RUN-explicit").exists()
    assert not (tmp_path / "RUN-context").exists()


def test_write_inference_artifacts_uses_run_id_and_dual_writes_pack(tmp_path):
    from sohbench.report import write_inference_artifacts

    out = Path(
        write_inference_artifacts(
            [{"model": "predict_only", "dataset": "new_a", "status": "succeeded"}],
            {},
            tmp_path,
            run_id="RUN-inference",
            evidence_context={"run_type": "inference"},
        )
    )

    assert out.name == "RUN-inference"
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert pack["run_id"] == "RUN-inference"
    assert pack["run_type"] == "inference"
    assert {item["path"] for item in pack["artifacts"]} == {
        "inference_results.parquet",
        "inference_results.csv",
        "inference_benchmark.json",
        "inference_run_config.json",
    }
