import numpy as np
import pandas as pd

from sohbench.fabric import failures
from sohbench.fabric.failures import FailureCategory, classify_exception
from sohbench.fabric.outcomes import RunOutcome


def _result_rows(statuses):
    rows = []
    for index, status in enumerate(statuses):
        for metric in ("mae", "picp"):
            rows.append(
                {
                    "estimator": "demo",
                    "featureset": "full",
                    "protocol": "loco",
                    "held_out_key": f"cell-{index}",
                    "fold": index,
                    "seed": 0,
                    "metric": metric,
                    "value": 0.1 if status == "succeeded" else np.nan,
                    "unit_status": status,
                }
            )
    return pd.DataFrame(rows)


def test_failure_categories_have_exact_stable_values():
    assert [category.value for category in FailureCategory] == [
        "INPUT_CONTRACT",
        "DATA_QC",
        "LEAKAGE",
        "ALGORITHM",
        "RESOURCE",
        "INFRASTRUCTURE",
        "METRIC",
        "POLICY",
    ]


def test_classify_exception_handles_resource_and_algorithm_failures():
    assert classify_exception(MemoryError("out of memory")) is FailureCategory.RESOURCE
    assert classify_exception(MemoryError("leakage detected: allocator exhausted")) is FailureCategory.RESOURCE
    assert classify_exception(TypeError("estimator rejected dtype")) is FailureCategory.ALGORITHM
    assert classify_exception(ValueError("estimator rejected values")) is FailureCategory.ALGORITHM
    assert classify_exception(RuntimeError("estimator fit failed")) is FailureCategory.ALGORITHM


def test_classify_exception_uses_narrow_contextual_failure_types():
    assert hasattr(failures, "InputContractError")
    assert hasattr(failures, "MetricEvaluationError")
    assert (
        classify_exception(failures.InputContractError("invalid feature shape"))
        is FailureCategory.INPUT_CONTRACT
    )
    assert (
        classify_exception(failures.MetricEvaluationError("mae failed"))
        is FailureCategory.METRIC
    )

    resource_error = failures.MetricEvaluationError("metric allocation failed")
    resource_error.__cause__ = MemoryError("out of memory")
    assert classify_exception(resource_error) is FailureCategory.RESOURCE

    leakage_error = failures.InputContractError("feature preparation failed")
    leakage_error.__cause__ = ValueError("feature pipeline references mutable closure state")
    assert classify_exception(leakage_error) is FailureCategory.LEAKAGE


def test_classify_exception_recognizes_specific_leakage_markers_only():
    assert (
        classify_exception(ValueError("feature pipeline references mutable closure state"))
        is FailureCategory.LEAKAGE
    )
    assert classify_exception(RuntimeError("leakage detected: train/test overlap")) is FailureCategory.LEAKAGE
    assert classify_exception(RuntimeError("worker leaked a temporary file")) is FailureCategory.ALGORITHM


def test_all_failed_is_failed():
    outcome = RunOutcome.from_results(_result_rows(["failed", "failed"]))
    assert outcome == RunOutcome(outcome="failed", attempted=2, succeeded=0, failed=2)


def test_mixed_units_are_partial_and_keep_attempted_denominator():
    outcome = RunOutcome.from_results(_result_rows(["succeeded", "failed"]))
    assert outcome == RunOutcome(outcome="partial", attempted=2, succeeded=1, failed=1)


def test_all_succeeded_is_complete():
    outcome = RunOutcome.from_results(_result_rows(["succeeded", "succeeded"]))
    assert outcome == RunOutcome(outcome="complete", attempted=2, succeeded=2, failed=0)


def test_legacy_results_infer_status_and_use_available_unit_columns():
    results = pd.DataFrame(
        [
            {"estimator": "demo", "protocol": "loco", "fold": 0, "seed": 0, "metric": "mae", "value": 0.1},
            {"estimator": "demo", "protocol": "loco", "fold": 0, "seed": 0, "metric": "picp", "value": 0.9},
            {"estimator": "demo", "protocol": "loco", "fold": 1, "seed": 0, "metric": "mae", "value": np.nan},
            {"estimator": "demo", "protocol": "loco", "fold": 1, "seed": 0, "metric": "picp", "value": np.nan},
        ]
    )

    assert RunOutcome.from_results(results) == RunOutcome(
        outcome="partial", attempted=2, succeeded=1, failed=1
    )


def test_unit_status_requires_every_row_to_explicitly_succeed():
    results = _result_rows(["succeeded"])
    results.loc[0, "unit_status"] = None

    assert RunOutcome.from_results(results) == RunOutcome(
        outcome="failed", attempted=1, succeeded=0, failed=1
    )

    results.loc[0, "unit_status"] = "unknown"
    assert RunOutcome.from_results(results) == RunOutcome(
        outcome="failed", attempted=1, succeeded=0, failed=1
    )


def test_nullable_unit_status_missing_row_fails_the_unit():
    results = _result_rows(["succeeded"])
    results["unit_status"] = pd.Series(
        ["succeeded", pd.NA], index=results.index, dtype="string"
    )

    assert RunOutcome.from_results(results) == RunOutcome(
        outcome="failed", attempted=1, succeeded=0, failed=1
    )


def test_legacy_success_requires_finite_numeric_metric_evidence():
    results = pd.DataFrame(
        [
            {"estimator": "demo", "fold": 0, "seed": 0, "metric": "mae", "value": np.inf},
            {"estimator": "demo", "fold": 0, "seed": 0, "metric": "picp", "value": -np.inf},
            {"estimator": "demo", "fold": 1, "seed": 0, "metric": "mae", "value": "bad"},
            {"estimator": "demo", "fold": 1, "seed": 0, "metric": "picp", "value": np.nan},
            {"estimator": "demo", "fold": 2, "seed": 0, "metric": "mae", "value": 0.1},
            {"estimator": "demo", "fold": 2, "seed": 0, "metric": "picp", "value": np.inf},
        ]
    )

    assert RunOutcome.from_results(results) == RunOutcome(
        outcome="partial", attempted=3, succeeded=1, failed=2
    )
