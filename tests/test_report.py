import json, os, subprocess, threading, pandas as pd, numpy as np, pytest
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


@pytest.mark.skipif(os.name != "nt", reason="Windows case-alias regression")
def test_write_artifacts_rejects_case_equivalent_existing_run_directory(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    physical = root / "RUN-Case"
    physical.mkdir()
    requested_alias = root / "run-case"
    if not requested_alias.exists():
        pytest.skip("filesystem proved case-sensitive")

    with pytest.raises(FileExistsError, match="case"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, root, run_id="run-case"
        )

    assert list(physical.iterdir()) == []
    out = Path(
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, root, run_id="RUN-Case"
        )
    )
    assert out == physical


def _run_concurrent_writers_after_shared_selection(
    tmp_path, monkeypatch, run_ids, existing_empty=None,
):
    root = tmp_path / "runs"
    root.mkdir()
    if existing_empty is not None:
        (root / existing_empty).mkdir()
    selection_barrier = threading.Barrier(2)
    first_ready_to_write = threading.Event()
    second_finished_attempt = threading.Event()
    state = threading.local()
    original_select = report_module._select_run_target
    original_to_parquet = pd.DataFrame.to_parquet

    def synchronized_select(root_path, run_id):
        selected = original_select(root_path, run_id)
        if not getattr(state, "initial_selection_complete", False):
            state.initial_selection_complete = True
            selection_barrier.wait(timeout=10)
            if state.role == "second":
                assert first_ready_to_write.wait(timeout=10)
        return selected

    def synchronized_to_parquet(frame, *args, **kwargs):
        if state.role == "first" and not getattr(state, "write_paused", False):
            state.write_paused = True
            first_ready_to_write.set()
            assert second_finished_attempt.wait(timeout=10)
        return original_to_parquet(frame, *args, **kwargs)

    monkeypatch.setattr(report_module, "_select_run_target", synchronized_select)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", synchronized_to_parquet)

    def run_writer(role, run_id):
        state.role = role
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
        finally:
            if role == "second":
                second_finished_attempt.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_writer, "first", run_ids[0]),
            executor.submit(run_writer, "second", run_ids[1]),
        ]
        outcomes = [future.result(timeout=20) for future in futures]

    return root, list(zip(("first", "second"), run_ids, outcomes))


def _assert_single_claimed_writer(root, outcomes):
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
        for word in ("claim", "collision", "case", "not empty")
    )

    role, run_id, run_dir = successes[0]
    assert json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) == {
        "writer": role
    }
    pack = verify_evidence_pack(run_dir / "evidence-pack.json")
    assert pack["run_id"] == run_id
    assert ".sohbench-run.claim" not in {item["path"] for item in pack["artifacts"]}
    assert not (run_dir / ".sohbench-run.claim").exists()
    assert len([entry for entry in root.iterdir() if entry.is_dir()]) == 1


def test_concurrent_exact_run_ids_allow_only_one_artifact_transaction(
    tmp_path, monkeypatch,
):
    root, outcomes = _run_concurrent_writers_after_shared_selection(
        tmp_path,
        monkeypatch,
        ("RUN-Race", "RUN-Race"),
        existing_empty="RUN-Race",
    )

    _assert_single_claimed_writer(root, outcomes)


@pytest.mark.skipif(os.name != "nt", reason="Windows case-alias regression")
def test_concurrent_case_equivalent_run_ids_allow_only_one_artifact_transaction(
    tmp_path, monkeypatch,
):
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    physical_probe = probe_root / "Case-Probe"
    physical_probe.mkdir()
    if not (probe_root / "case-probe").exists():
        pytest.skip("filesystem proved case-sensitive")

    root, outcomes = _run_concurrent_writers_after_shared_selection(
        tmp_path, monkeypatch, ("RUN-Race", "run-race")
    )

    _assert_single_claimed_writer(root, outcomes)


@pytest.mark.parametrize("claim_kind", ["file", "directory"])
def test_stale_run_claim_fails_closed(tmp_path, claim_kind):
    run_dir = tmp_path / "RUN-stale"
    run_dir.mkdir()
    claim = run_dir / ".sohbench-run.claim"
    if claim_kind == "file":
        claim.write_bytes(b"stale")
    else:
        claim.mkdir()

    with pytest.raises(FileExistsError, match="already claimed"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id="RUN-stale"
        )

    if claim_kind == "file":
        assert claim.read_bytes() == b"stale"
    else:
        assert claim.is_dir()
    assert list(run_dir.iterdir()) == [claim]


def test_inference_holds_claim_through_evidence_pack_and_removes_it(
    tmp_path, monkeypatch,
):
    from sohbench.report import write_inference_artifacts

    original_write_pack = report_module.write_evidence_pack
    observed_claims = []

    def observe_claim(run_dir, manifest, artifact_paths):
        claim = Path(run_dir) / ".sohbench-run.claim"
        observed_claims.append(claim.is_dir())
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

    assert observed_claims == [True]
    assert not (out / ".sohbench-run.claim").exists()


def test_claim_transaction_does_not_use_descriptor_lifecycle_apis(
    tmp_path, monkeypatch,
):
    def fail_descriptor_api(*args, **kwargs):
        pytest.fail("run claim used a descriptor lifecycle API")

    monkeypatch.setattr(report_module.os, "open", fail_descriptor_api)
    monkeypatch.setattr(report_module.os, "fstat", fail_descriptor_api)
    monkeypatch.setattr(report_module.os, "close", fail_descriptor_api)

    with report_module._run_directory_transaction(
        tmp_path, "RUN-no-descriptor"
    ) as out:
        assert (out / ".sohbench-run.claim").is_dir()

    assert not (tmp_path / "RUN-no-descriptor" / ".sohbench-run.claim").exists()


def test_claim_acquisition_retries_one_transient_lstat_failure(
    tmp_path, monkeypatch,
):
    real_lstat = Path.lstat
    claim_lstat_calls = 0

    def fail_first_claim_lstat(path):
        nonlocal claim_lstat_calls
        if path.name == ".sohbench-run.claim":
            claim_lstat_calls += 1
            if claim_lstat_calls == 1:
                raise OSError("injected claim lstat failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_first_claim_lstat)
    out = Path(
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id="RUN-lstat"
        )
    )

    assert claim_lstat_calls >= 2
    assert out.is_dir()
    assert not (out / ".sohbench-run.claim").exists()


def test_persistent_acquisition_metadata_failure_leaves_replacement_fail_closed(
    tmp_path, monkeypatch,
):
    real_lstat = Path.lstat
    first_error = OSError("injected persistent acquisition metadata failure")
    replacement_identity = None
    claim_lstat_calls = 0

    def replace_then_fail_claim_lstat(path):
        nonlocal claim_lstat_calls, replacement_identity
        if path.name == ".sohbench-run.claim":
            claim_lstat_calls += 1
            if claim_lstat_calls == 1:
                path.rmdir()
                path.mkdir()
                replacement = real_lstat(path)
                replacement_identity = (replacement.st_dev, replacement.st_ino)
                raise first_error
            raise OSError("injected repeated acquisition metadata failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", replace_then_fail_claim_lstat)
    with pytest.raises(OSError) as exc_info:
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path,
            run_id="RUN-unknown-acquisition",
        )

    monkeypatch.setattr(Path, "lstat", real_lstat)
    claim = tmp_path / "RUN-unknown-acquisition" / ".sohbench-run.claim"
    current = real_lstat(claim)
    assert exc_info.value is first_error
    assert claim_lstat_calls == 2
    assert claim.is_dir()
    assert (current.st_dev, current.st_ino) == replacement_identity
    claim.rmdir()


def test_claim_release_retries_one_transient_lstat_failure(
    tmp_path, monkeypatch,
):
    real_lstat = Path.lstat
    claim_lstat_calls = 0

    def fail_second_claim_lstat(path):
        nonlocal claim_lstat_calls
        if path.name == ".sohbench-run.claim":
            claim_lstat_calls += 1
            if claim_lstat_calls == 2:
                raise OSError("injected transient release lstat failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_second_claim_lstat)
    out = Path(
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id="RUN-retry"
        )
    )

    assert claim_lstat_calls >= 3
    assert not (out / ".sohbench-run.claim").exists()


def _install_persistent_release_lstat_failure(monkeypatch):
    real_lstat = Path.lstat
    claim_lstat_calls = 0

    def fail_release_lstat(path):
        nonlocal claim_lstat_calls
        if path.name == ".sohbench-run.claim":
            claim_lstat_calls += 1
            if claim_lstat_calls >= 2:
                raise OSError("injected persistent release lstat failure")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_release_lstat)


def test_successful_body_surfaces_persistent_claim_metadata_failure(
    tmp_path, monkeypatch,
):
    _install_persistent_release_lstat_failure(monkeypatch)

    with pytest.raises(OSError, match="persistent release lstat failure"):
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id="RUN-metadata"
        )

    assert (tmp_path / "RUN-metadata" / ".sohbench-run.claim").exists()


def test_artifact_error_remains_primary_when_claim_metadata_cleanup_fails(
    tmp_path, monkeypatch,
):
    _install_persistent_release_lstat_failure(monkeypatch)

    def fail_pack(*args, **kwargs):
        raise RuntimeError("injected artifact failure")

    monkeypatch.setattr(report_module, "write_evidence_pack", fail_pack)
    with pytest.raises(RuntimeError, match="injected artifact failure") as exc_info:
        write_artifacts(
            _results(), None, SOHEstimationTask(), {}, tmp_path, run_id="RUN-primary"
        )

    assert any(
        "persistent release lstat failure" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )
    assert (tmp_path / "RUN-primary" / ".sohbench-run.claim").exists()


def test_claim_identity_mismatch_is_reported_and_left_fail_closed(
    tmp_path, monkeypatch,
):
    real_lstat = Path.lstat
    claim = None
    owned_identity = None

    class DifferentIdentity:
        def __init__(self, original):
            self._original = original
            self.st_ino = original.st_ino + 1

        def __getattr__(self, name):
            return getattr(self._original, name)

    def mismatched_claim_lstat(path):
        result = real_lstat(path)
        if path.name == ".sohbench-run.claim":
            return DifferentIdentity(result)
        return result

    try:
        with pytest.raises(RuntimeError, match="identity"):
            with report_module._run_directory_transaction(
                tmp_path, "RUN-identity"
            ) as out:
                claim = out / ".sohbench-run.claim"
                owned = real_lstat(claim)
                owned_identity = (owned.st_dev, owned.st_ino)
                monkeypatch.setattr(Path, "lstat", mismatched_claim_lstat)
        monkeypatch.setattr(Path, "lstat", real_lstat)
        current = real_lstat(claim)
        assert claim.is_dir()
        assert (current.st_dev, current.st_ino) == owned_identity
    finally:
        monkeypatch.setattr(Path, "lstat", real_lstat)
        if claim is not None and claim.exists():
            if claim.is_dir():
                claim.rmdir()
            else:
                claim.unlink()


def test_nonempty_claim_directory_is_reported_and_not_removed(tmp_path):
    claim = None
    marker = None
    try:
        with pytest.raises(RuntimeError, match="empty"):
            with report_module._run_directory_transaction(
                tmp_path, "RUN-nonempty-claim"
            ) as out:
                claim = out / ".sohbench-run.claim"
                marker = claim / "marker.txt"
                marker.write_text("owned content", encoding="utf-8")
        assert claim.is_dir()
        assert marker.read_text(encoding="utf-8") == "owned content"
    finally:
        if marker is not None and marker.exists():
            marker.unlink()
        if claim is not None and claim.exists():
            if claim.is_dir():
                claim.rmdir()
            else:
                claim.unlink()


def test_replaced_claim_symlink_is_not_followed_or_removed(tmp_path):
    target = tmp_path / "external-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    else:
        probe.unlink()

    claim = None
    try:
        with pytest.raises(RuntimeError, match="ordinary|identity|reparse"):
            with report_module._run_directory_transaction(
                tmp_path, "RUN-replaced-symlink"
            ) as out:
                claim = out / ".sohbench-run.claim"
                claim.rmdir()
                claim.symlink_to(target, target_is_directory=True)
        assert claim.is_symlink()
        assert sentinel.read_bytes() == b"external sentinel"
    finally:
        if claim is not None and claim.is_symlink():
            claim.unlink()

    assert sentinel.read_bytes() == b"external sentinel"


@pytest.mark.skipif(os.name != "nt", reason="directory junctions require Windows")
def test_replaced_claim_junction_is_not_followed_or_removed(tmp_path):
    target = tmp_path / "junction-claim-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    claim = None
    try:
        with pytest.raises(RuntimeError, match="ordinary|identity|reparse"):
            with report_module._run_directory_transaction(
                tmp_path, "RUN-replaced-junction"
            ) as out:
                claim = out / ".sohbench-run.claim"
                claim.rmdir()
                created = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(claim), str(target)],
                    capture_output=True,
                    text=True,
                )
                if created.returncode:
                    pytest.skip(
                        "directory junction creation unavailable: "
                        f"{created.stdout}{created.stderr}"
                    )
        assert claim.exists()
        assert claim.is_junction()
        assert sentinel.read_bytes() == b"external sentinel"
    finally:
        if claim is not None and claim.exists():
            os.rmdir(claim)

    assert sentinel.read_bytes() == b"external sentinel"


@pytest.mark.skipif(os.name != "nt", reason="native claim deletion requires Windows")
def test_windows_replacement_before_claim_handle_open_survives(
    tmp_path, monkeypatch,
):
    real_open = report_module._windows_open_claim_for_delete
    replacement_identity = None
    marker = None

    def replace_before_open(claim_path):
        nonlocal replacement_identity, marker
        claim_path.rmdir()
        claim_path.mkdir()
        marker = claim_path / "replacement.txt"
        marker.write_text("replacement survives", encoding="utf-8")
        current = claim_path.lstat()
        replacement_identity = (current.st_dev, current.st_ino)
        return real_open(claim_path)

    monkeypatch.setattr(
        report_module, "_windows_open_claim_for_delete", replace_before_open
    )
    claim = None
    try:
        with pytest.raises(RuntimeError, match="identity"):
            with report_module._run_directory_transaction(
                tmp_path, "RUN-replaced-before-open"
            ) as out:
                claim = out / ".sohbench-run.claim"

        current = claim.lstat()
        assert (current.st_dev, current.st_ino) == replacement_identity
        assert marker.read_text(encoding="utf-8") == "replacement survives"
    finally:
        if marker is not None and marker.exists():
            marker.unlink()
        if claim is not None and claim.exists():
            claim.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="native claim deletion requires Windows")
def test_windows_replacement_in_final_claim_deletion_window_survives(
    tmp_path, monkeypatch,
):
    real_mark = report_module._windows_mark_claim_for_deletion
    detached = tmp_path / "detached-owned-claim"
    replacement_identity = None
    marker = None
    claim = None

    def replace_after_handle_verification(handle):
        nonlocal replacement_identity, marker
        claim.rename(detached)
        claim.mkdir()
        marker = claim / "replacement.txt"
        marker.write_text("replacement survives", encoding="utf-8")
        current = claim.lstat()
        replacement_identity = (current.st_dev, current.st_ino)
        return real_mark(handle)

    monkeypatch.setattr(
        report_module, "_windows_mark_claim_for_deletion",
        replace_after_handle_verification,
    )
    try:
        with pytest.raises(RuntimeError, match="replacement|claim"):
            with report_module._run_directory_transaction(
                tmp_path, "RUN-final-window"
            ) as out:
                claim = out / ".sohbench-run.claim"

        current = claim.lstat()
        assert (current.st_dev, current.st_ino) == replacement_identity
        assert marker.read_text(encoding="utf-8") == "replacement survives"
        assert not detached.exists()
    finally:
        if marker is not None and marker.exists():
            marker.unlink()
        if claim is not None and claim.exists():
            claim.rmdir()
        if detached.exists():
            detached.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="native claim deletion requires Windows")
def test_windows_native_claim_deletion_removes_ordinary_claim(
    tmp_path, monkeypatch,
):
    real_mark = report_module._windows_mark_claim_for_deletion
    marked_handles = []

    def observe_mark(handle):
        marked_handles.append(handle)
        return real_mark(handle)

    monkeypatch.setattr(
        report_module, "_windows_mark_claim_for_deletion", observe_mark
    )
    with report_module._run_directory_transaction(
        tmp_path, "RUN-native-delete"
    ) as out:
        claim = out / ".sohbench-run.claim"
        assert claim.is_dir()

    assert len(marked_handles) == 1
    assert not claim.exists()


@pytest.mark.skipif(os.name != "nt", reason="native claim deletion requires Windows")
def test_windows_claim_handle_close_failure_is_not_retried(
    tmp_path, monkeypatch,
):
    real_close = report_module._CLOSE_HANDLE
    observed_handles = []

    def close_then_report_failure(handle):
        observed_handles.append(handle)
        assert real_close(handle)
        report_module.ctypes.set_last_error(5)
        return 0

    monkeypatch.setattr(
        report_module, "_CLOSE_HANDLE", close_then_report_failure
    )
    claim = tmp_path / "RUN-close-handle" / ".sohbench-run.claim"
    with pytest.raises(OSError):
        with report_module._run_directory_transaction(
            tmp_path, "RUN-close-handle"
        ):
            pass

    assert len(observed_handles) == 1
    assert not claim.exists()


def test_portable_claim_cleanup_revalidation_leaves_replacement(
    tmp_path, monkeypatch,
):
    claim = tmp_path / ".sohbench-run.claim"
    claim.mkdir()
    owned = claim.lstat()
    identity = (owned.st_dev, owned.st_ino)
    real_lstat = Path.lstat
    marker = None

    def replace_before_revalidation(path):
        nonlocal marker
        if path == claim and marker is None:
            path.rmdir()
            path.mkdir()
            marker = path / "replacement.txt"
            marker.write_text("replacement survives", encoding="utf-8")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", replace_before_revalidation)
    with pytest.raises(RuntimeError, match="identity"):
        report_module._portable_delete_claim_directory(claim, identity)

    monkeypatch.setattr(Path, "lstat", real_lstat)
    assert marker.read_text(encoding="utf-8") == "replacement survives"
    marker.unlink()
    claim.rmdir()


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
