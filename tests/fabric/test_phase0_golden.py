import json
from pathlib import Path

import pandas as pd

from sohbench.evaluation_service import EvaluationService
from sohbench.fabric.evidence import verify_evidence_pack


def _dataset_csv():
    rows = []
    for cell in ("c1", "c2", "c3"):
        for cycle in range(4):
            rows.append({
                "dataset": "golden_demo",
                "cell_id": cell,
                "cycle_index": cycle,
                "soh": 1.0 - 0.01 * cycle,
                "voltage_mean": 3.5 - 0.01 * cycle,
            })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _stable_summary(job):
    pack = verify_evidence_pack(job["evidence_path"])
    run_spec = pack["run_spec"]
    statistics = json.loads(
        (Path(job["raw_run_dir"]) / "statistics.json").read_text(encoding="utf-8")
    )
    results = pd.read_csv(Path(job["raw_run_dir"]) / "results.csv")
    return {
        "evidence_schema_version": pack["schema_version"],
        "statistics_schema_version": statistics["schema_version"],
        "comparison_key": pack["comparison_key"],
        "run_type": run_spec["run_type"],
        "candidate_count": len(run_spec["candidates"]),
        "seed_count": len(run_spec["seeds"]),
        "artifact_names": [item["path"] for item in pack["artifacts"]],
        "result_count": len(results),
        "failure_categories": sorted(
            category for category in results["error_category"].dropna().unique() if category
        ),
    }


def test_phase0_evidence_golden_summary(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("golden_demo", "golden.csv", _dataset_csv())
    job = service.create_job(
        {
            "datasets": ["golden_demo"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    finished = service.run_job_inline(job["id"])

    expected = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "fabric" / "expected-evidence-summary.json")
        .read_text(encoding="utf-8")
    )
    assert _stable_summary(finished) == expected
