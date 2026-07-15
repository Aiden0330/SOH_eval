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
        / ".sohbench-run-reservations"
        / report_module._reservation_key(run_id)
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


def test_write_artifacts_rejects_reservation_registry_name_as_run_id(tmp_path):
    with pytest.raises(ValueError, match="run_id|reserved"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path,
            run_id=".SOHBENCH-RUN-RESERVATIONS",
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
    registry = root / ".sohbench-run-reservations"
    assert registry.is_dir()
    assert len(list(registry.iterdir())) == 1
    assert sorted(entry.name for entry in root.iterdir()) == sorted(
        [run_dir.name, registry.name]
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


@pytest.mark.parametrize("reservation_kind", ["file", "directory"])
def test_stale_durable_reservation_fails_closed(tmp_path, reservation_kind):
    run_id = "RUN-stale"
    registry = tmp_path / ".sohbench-run-reservations"
    registry.mkdir()
    reservation = registry / report_module._reservation_key(run_id)
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
        registry = Path(run_dir).parent / ".sohbench-run-reservations"
        reservation = registry / report_module._reservation_key(manifest["run_id"])
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
    assert not (out / ".sohbench-run-reservations").exists()
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
    reservation = (
        tmp_path
        / ".sohbench-run-reservations"
        / report_module._reservation_key(run_id)
    )
    assert reservation.is_dir()


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
    reservation = (
        tmp_path
        / ".sohbench-run-reservations"
        / report_module._reservation_key(run_id)
    )
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
    registry = tmp_path / ".sohbench-run-reservations"
    reservation = registry / report_module._reservation_key(run_id)

    assert reservation.is_dir()
    assert registry.parent == tmp_path
    assert not (out / registry.name).exists()
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


def test_reservation_registry_must_be_an_ordinary_directory(tmp_path):
    registry = tmp_path / ".sohbench-run-reservations"
    registry.write_bytes(b"not a directory")

    with pytest.raises(ValueError, match="reservation registry|ordinary"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path,
            run_id="RUN-registry-file",
        )

    assert registry.read_bytes() == b"not a directory"
    assert not (tmp_path / "RUN-registry-file").exists()


def test_replaced_reservation_registry_symlink_is_not_followed(tmp_path):
    target = tmp_path / "external-registry-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    registry = tmp_path / ".sohbench-run-reservations"
    try:
        registry.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    try:
        with pytest.raises(ValueError, match="reservation registry|link|reparse"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, tmp_path,
                run_id="RUN-registry-link",
            )
        assert registry.is_symlink()
        assert sentinel.read_bytes() == b"external sentinel"
        assert list(target.iterdir()) == [sentinel]
    finally:
        if registry.is_symlink():
            registry.unlink()


@pytest.mark.skipif(os.name != "nt", reason="directory junctions require Windows")
def test_replaced_reservation_registry_junction_is_not_followed(tmp_path):
    target = tmp_path / "external-registry-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    registry = tmp_path / ".sohbench-run-reservations"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(registry), str(target)],
        capture_output=True,
        text=True,
    )
    if created.returncode:
        pytest.skip(
            "directory junction creation unavailable: "
            f"{created.stdout}{created.stderr}"
        )

    try:
        with pytest.raises(ValueError, match="reservation registry|link|reparse"):
            write_artifacts(
                _results(), None, SOHEstimationTask(), {}, tmp_path,
                run_id="RUN-registry-junction",
            )
        assert registry.is_junction()
        assert sentinel.read_bytes() == b"external sentinel"
        assert list(target.iterdir()) == [sentinel]
    finally:
        if registry.exists():
            os.rmdir(registry)


def test_ephemeral_claim_cleanup_implementation_is_removed():
    obsolete_names = (
        "_acquire_run_claim",
        "_cleanup_run_claim",
        "_portable_delete_claim_directory",
        "_windows_open_claim_for_delete",
        "_windows_mark_claim_for_deletion",
        "_windows_close_claim_handle",
        "_windows_delete_claim_directory",
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
