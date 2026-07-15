import json, pandas as pd, numpy as np
from pathlib import Path
from sohbench.report import write_artifacts
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
    assert (Path(out)/"results.parquet").exists()

def test_leaderboard_matches_schema(tmp_path):
    import jsonschema, json
    out = write_artifacts(_results(), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    lb = json.loads((Path(out)/"leaderboard.json").read_text())
    schema = json.loads((Path(__file__).parents[1]/"src/sohbench/schemas/leaderboard.schema.json").read_text())
    jsonschema.validate(lb, schema)


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
    assert entry["verdicts"] == {
        "generalization": True,
        "calibration": True,
        "physical_consistency": True,
    }
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
