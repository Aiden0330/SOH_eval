import json

from sohbench import phase24_pipeline
from sohbench.phase24_pipeline import Phase24PipelineConfig, run_phase24_pipeline


class _ModelStore:
    def catalog(self):
        return [
            {
                "name": "ridge_full_loco_nasa_B0018_seed0",
                "artifact_id": "model@sha256:phase24",
                "estimator": "ridge",
                "featureset": "full",
                "protocol": "loco",
                "held_out_key": "nasa/B0018",
                "seed": 0,
            }
        ]


class _Service:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.model_store = _ModelStore()
        self.training_payload = None
        self.inference_payload = None

    def training_preflight(self, payload):
        return {
            "ready": True,
            "preflight_id": "preflight@sha256:phase24",
            "summary": {"rows": 2705},
        }

    def create_job(self, payload, *, start=True):
        self.training_payload = payload
        return {"id": "train-24"}

    def run_job_inline(self, job_id, *, publish=True):
        assert publish is False
        return {
            "id": job_id,
            "status": "succeeded",
            "saved_models": ["ridge_full_loco_nasa_B0018_seed0"],
            "model_artifacts": self.model_store.catalog(),
        }

    def create_inference_job(self, payload, *, start=True):
        self.inference_payload = payload
        return {"id": "infer-24"}

    def run_inference_job_inline(self, job_id, *, publish=True):
        assert publish is False
        run_dir = self.tmp_path / "inference"
        run_dir.mkdir()
        (run_dir / "inference_benchmark.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "model": "ridge_full_loco_nasa_B0018_seed0",
                            "dataset": "nasa",
                            "status": "succeeded",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {"id": job_id, "status": "succeeded", "run_dir": str(run_dir)}


def test_phase24_pipeline_threads_exact_jobs_into_case_release(tmp_path, monkeypatch):
    service = _Service(tmp_path)
    captured = {}

    def fake_build(_service, **kwargs):
        captured.update(kwargs)
        return {
            "phase": "2.4",
            "status": "completed",
            "case_id": "EV-24",
            "evidence_pack": {"state": "signed"},
        }

    monkeypatch.setattr(phase24_pipeline, "build_and_sign_phase22_golden", fake_build)
    summary = run_phase24_pipeline(
        service,
        Phase24PipelineConfig(profile_name="phase24-fixture"),
    )

    assert service.training_payload["preflight_id"] == "preflight@sha256:phase24"
    assert service.inference_payload == {
        "datasets": ["nasa"],
        "models": ["ridge_full_loco_nasa_B0018_seed0"],
        "launch_strategy": "compatible_only",
        "selected_datasets": ["nasa"],
        "submitted_datasets": ["nasa"],
        "skipped_datasets": [],
    }
    assert captured["release"]["training_job_id"] == "train-24"
    assert captured["release"]["inference_job_id"] == "infer-24"
    assert captured["release"]["profile_kind"] == "phase24_case_pipeline"
    assert summary["status"] == "completed"
    assert summary["pipeline"] == {
        "preflight_id": "preflight@sha256:phase24",
        "preflight_ready": True,
        "dataset_rows": 2705,
        "training_job_id": "train-24",
        "training_status": "succeeded",
        "training_model_count": 1,
        "selected_model": "ridge_full_loco_nasa_B0018_seed0",
        "selected_model_artifact_id": "model@sha256:phase24",
        "inference_job_id": "infer-24",
        "inference_status": "succeeded",
        "inference_records": 1,
    }


def test_phase24_pipeline_allows_a_release_gate_to_override_labels(tmp_path, monkeypatch):
    service = _Service(tmp_path)
    captured = {}

    def fake_build(_service, **kwargs):
        captured.update(kwargs)
        return {
            "phase": kwargs["release"]["phase"],
            "status": "completed",
            "case_id": "EV-30",
            "evidence_pack": {"state": "signed"},
        }

    monkeypatch.setattr(phase24_pipeline, "build_and_sign_phase22_golden", fake_build)
    summary = run_phase24_pipeline(
        service,
        Phase24PipelineConfig(
            profile_name="phase30-fixture",
            release_overrides={
                "phase": "3.0",
                "profile_kind": "phase30_release_gate",
                "profile_label": "阶段 3.0 · 自动回归门禁基线",
            },
        ),
    )

    assert summary["phase"] == "3.0"
    assert captured["release"]["profile_kind"] == "phase30_release_gate"
    assert captured["release"]["profile_label"] == "阶段 3.0 · 自动回归门禁基线"
    assert captured["release"]["training_job_id"] == "train-24"
    assert captured["release"]["inference_job_id"] == "infer-24"
