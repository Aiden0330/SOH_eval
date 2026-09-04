import json, pandas as pd, numpy as np
import pytest
from pathlib import Path
from sohkernel.report import write_artifacts
from sohkernel.tasks import SOHEstimationTask

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
    assert (Path(out)/"results.parquet").exists()

def test_leaderboard_matches_schema(tmp_path):
    import jsonschema, json
    out = write_artifacts(_results(), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    lb = json.loads((Path(out)/"leaderboard.json").read_text())
    import sohkernel.report
    schema_path = Path(sohkernel.report.__file__).with_name("schemas") / "leaderboard.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
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
    from sohkernel.report import write_inference_artifacts

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


# ---------------------------------------------------------------------------
# explicit outcomes: failed units stay out of the evidence (Task 7e)
# ---------------------------------------------------------------------------

def _results_with_failures(failed_seeds, value_for_failed=0.0):
    rows = []
    for seed in range(3):
        failed = seed in failed_seeds
        for metric, val in [("mae", 0.01), ("picp", 0.92),
                            ("monotonicity_violation_rate", 0.0), ("value_range_violation", 0.0)]:
            rows.append(dict(estimator="ridge", feature_mode="unified", backend="sklearn",
                             protocol="loco", dataset="ds", held_out_key="c1", fold=0, seed=seed,
                             status="failed" if failed else "ok",
                             error="boom" if failed else None,
                             error_type="RuntimeError" if failed else None,
                             metric=metric, value=value_for_failed if failed else val))
    return pd.DataFrame(rows)


def test_failed_units_are_excluded_from_aggregates_and_counted(tmp_path):
    out = write_artifacts(_results_with_failures({2}), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]
    # the failed unit carried finite garbage; it must not touch the mean
    assert entry["metrics"]["mae"]["mean"] == 0.01 and entry["metrics"]["picp"]["mean"] == 0.92
    assert entry["units"] == {"attempted": 3, "succeeded": 2, "failed": 1, "outcome": "partial"}
    assert entry["verdicts"]["generalization"] is True


def test_all_units_failed_is_an_explicit_failed_outcome(tmp_path):
    out = write_artifacts(_results_with_failures({0, 1, 2}, value_for_failed=float("nan")), qc_report=None,
                          task=SOHEstimationTask(), config={}, out_dir=tmp_path)
    entry = json.loads((Path(out) / "leaderboard.json").read_text())[0]
    assert entry["units"] == {"attempted": 3, "succeeded": 0, "failed": 3, "outcome": "failed"}
    assert entry["metrics"]["mae"]["mean"] is None          # no evidence, keys still present
    assert entry["verdicts"]["generalization"] is False
    dataset_entry = json.loads((Path(out) / "dataset_leaderboard.json").read_text())[0]
    assert dataset_entry["units"]["outcome"] == "failed"


def test_complete_outcome_and_schema(tmp_path):
    import jsonschema
    import sohkernel.report
    out = write_artifacts(_results(), qc_report=None, task=SOHEstimationTask(), config={}, out_dir=tmp_path)
    lb = json.loads((Path(out) / "leaderboard.json").read_text())
    assert lb[0]["units"] == {"attempted": 5, "succeeded": 5, "failed": 0, "outcome": "complete"}
    schema = json.loads((Path(sohkernel.report.__file__).with_name("schemas") / "leaderboard.schema.json").read_text())
    jsonschema.validate(lb, schema)


# ---------------------------------------------------------------------------
# asset-level bootstrap intervals (Task 7h)
# ---------------------------------------------------------------------------

def _loco_results(cells=("c1", "c2", "c3", "c4"), seeds=(0, 1), protocol="loco"):
    rows = []
    for i, cell in enumerate(cells):
        for seed in seeds:
            for metric, val in [("mae", 0.01 + 0.005 * i + 0.001 * seed), ("picp", 0.9 + 0.02 * (i % 2)),
                                ("monotonicity_violation_rate", 0.0), ("value_range_violation", 0.0)]:
                rows.append(dict(estimator="ridge", feature_mode="unified", backend="sklearn", protocol=protocol,
                                 dataset="ds", held_out_key=f"ds/{cell}", fold=i, seed=seed, status="ok",
                                 metric=metric, value=val))
    return pd.DataFrame(rows)


def test_loco_entries_carry_asset_level_bootstrap_intervals(tmp_path):
    import jsonschema
    import sohkernel.report
    out = write_artifacts(_loco_results(), qc_report=None, task=SOHEstimationTask(), config={}, out_dir=tmp_path)
    lb = json.loads((Path(out) / "leaderboard.json").read_text())
    ci = lb[0]["confidence_intervals"]
    assert ci["status"] == "available" and ci["unit"] == "asset" and ci["assets"] == 4 and ci["confidence"] == 0.9
    mae = ci["intervals"]["mae"]
    assert mae["lower"] <= mae["mean"] <= mae["upper"] and mae["n"] == 4
    assert mae["mean"] == pytest.approx(lb[0]["metrics"]["mae"]["mean"])
    assert "picp" in ci["intervals"]
    schema = json.loads((Path(sohkernel.report.__file__).with_name("schemas") / "leaderboard.schema.json").read_text())
    jsonschema.validate(lb, schema)
    # per-asset (dataset leaderboard) entries have one asset each: explicitly unavailable
    dataset_lb = json.loads((Path(out) / "dataset_leaderboard.json").read_text())
    assert dataset_lb[0]["confidence_intervals"]["status"] == "unavailable"


def test_pooled_folds_report_intervals_as_unavailable(tmp_path):
    results = _loco_results(cells=("hust", "snl_lfp"), protocol="lodo")
    out = write_artifacts(results, qc_report=None, task=SOHEstimationTask(), config={}, out_dir=tmp_path)
    ci = json.loads((Path(out) / "leaderboard.json").read_text())[0]["confidence_intervals"]
    assert ci["status"] == "unavailable" and "lodo" in ci["reason"]


def test_bootstrap_is_deterministic_and_widens_with_spread():
    from sohkernel.statistics import bootstrap_mean_ci
    tight = bootstrap_mean_ci([0.01, 0.011, 0.012, 0.0105], seed=1)
    wide = bootstrap_mean_ci([0.01, 0.05, 0.02, 0.09], seed=1)
    assert bootstrap_mean_ci([0.01, 0.05, 0.02, 0.09], seed=1) == wide
    assert (wide["upper"] - wide["lower"]) > (tight["upper"] - tight["lower"])
    assert bootstrap_mean_ci([0.03])["lower"] == bootstrap_mean_ci([0.03])["upper"] == 0.03
    with pytest.raises(ValueError):
        bootstrap_mean_ci([float("nan")])
