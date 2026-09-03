"""Compatibility shim: the implementation moved to ``sohkernel.report`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.report``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.report import *  # noqa: F401,F403
from sohkernel.report import (  # noqa: F401
    json,
    math,
    time,
    Path,
    pd,
    evaluate_verdicts,
    SOHEstimationTask,
    _json_safe,
    _dump,
    _leaderboard_entries,
    _attach_training_provenance,
    _claim_run_dir,
    write_artifacts,
    write_inference_artifacts,
    _inference_summary,
)
