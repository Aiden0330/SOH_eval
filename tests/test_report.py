import json, os, subprocess, pandas as pd, numpy as np, pytest
from pathlib import Path
from sohbench.fabric.evidence import verify_evidence_pack
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
    pack = verify_evidence_pack(out / "evidence-pack.json")
    assert pack["run_id"] == "RUN-a"
    assert pack["comparison_key"] == "comparison-one"
    assert [item["path"] for item in pack["artifacts"]] == sorted(
        item["path"] for item in pack["artifacts"]
    )
    assert "evidence-pack.json" not in {item["path"] for item in pack["artifacts"]}


def test_write_artifacts_uses_distinct_generated_run_ids(tmp_path):
    first = Path(write_artifacts(_results(), None, SOHEstimationTask(), {}, tmp_path))
    second = Path(write_artifacts(_results(), None, SOHEstimationTask(), {}, tmp_path))

    assert first != second
    assert first.name.startswith("RUN-")
    assert second.name.startswith("RUN-")


def test_write_artifacts_accepts_empty_target_and_rejects_nonempty_target(tmp_path):
    empty = tmp_path / "RUN-empty"
    empty.mkdir()
    out = Path(
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id="RUN-empty"
        )
    )
    assert out == empty

    occupied = tmp_path / "RUN-occupied"
    occupied.mkdir()
    marker = occupied / "marker.txt"
    marker.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id="RUN-occupied"
        )
    assert marker.read_text(encoding="utf-8") == "do not overwrite"
    assert list(occupied.iterdir()) == [marker]


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
        with pytest.raises(ValueError, match="link|reparse|outside"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, root, run_id="RUN-link"
            )
        assert list(target.iterdir()) == []
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
        with pytest.raises(ValueError, match="link|reparse|outside"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, root, run_id="RUN-link"
            )
        assert list(target.iterdir()) == []
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
