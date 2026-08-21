from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .evaluation_service import EvaluationService
from .phase24_pipeline import Phase24PipelineConfig, run_phase24_pipeline
from .review_workflow import ReviewWorkflowError
from .paths import DEFAULT_CURVE_DIR


GATE_CONTRACT = "phase30-release-gate/1.0"
REQUIRED_EVENT_TYPES = {
    "case.created",
    "evidence.pinned",
    "review.approved",
    "retest.completed",
    "reconciliation.match",
    "evidence_pack.signed",
}


@dataclass(frozen=True)
class Phase30ReleaseGateConfig:
    profile_name: str = ""
    dataset: str = "nasa"
    asset_id: str = "B0018"
    estimator: str = "ridge"
    featureset: str = "full"
    protocol: str = "loco"
    seed: int = 0
    curve_source: str | Path = DEFAULT_CURVE_DIR / "nasa_curves.json"
    criterion_set_id: str = "exploratory"
    reviewer: str = "phase30-release-gate"
    version: str = "phase30-regression-gate-20260814a"
    outside_scope_dataset: str = "phase30-outside-scope"


def run_phase30_release_gate(
    service: EvaluationService,
    config: Phase30ReleaseGateConfig,
) -> dict:
    """Run the measured pipeline plus isolated release-blocking regressions."""
    started_at = _now()
    started = time.perf_counter()
    gate_id = _gate_id()
    profile_name = config.profile_name or _profile_name(config.dataset)
    checks: list[dict] = []
    baseline: dict = {}

    try:
        baseline = run_phase24_pipeline(
            service,
            Phase24PipelineConfig(
                profile_name=profile_name,
                dataset=config.dataset,
                asset_id=config.asset_id,
                estimator=config.estimator,
                featureset=config.featureset,
                protocol=config.protocol,
                seed=config.seed,
                curve_source=config.curve_source,
                criterion_set_id=config.criterion_set_id,
                reviewer=config.reviewer,
                version=config.version,
                release_overrides={
                    "phase": "3.0",
                    "profile_label": "阶段 3.0 · 自动回归门禁基线",
                    "profile_kind": "phase30_release_gate",
                    "case_title": (
                        f"阶段 3.0 · {config.dataset.upper()}/{config.asset_id} 自动回归门禁"
                    ),
                    "idempotency_prefix": "phase30",
                    "scenario_title": (
                        f"{config.dataset.upper()}/{config.asset_id} 实测发布门禁轨迹"
                    ),
                    "experiment": "phase30-automated-regression-gate",
                    "evidence_reason": (
                        "phase 3.0 release gate pinned representative evidence from the "
                        "measured training and inference pipeline"
                    ),
                    "review_note": (
                        "已核对真实训练、真实推理、代表证据、同协议复测、KPI 对账、"
                        "跨进程案例恢复及自动发布门禁。"
                    ),
                    "signature_reason": (
                        "阶段 3.0 自动回归门禁基线通过；该签署为工作台本地审计签署，"
                        "不代表法规认证。"
                    ),
                },
            ),
        )
        checks.append(_pipeline_check(baseline))
    except Exception as exc:  # the report must survive a failed release candidate
        checks.append(
            _check(
                "measured_pipeline",
                "真实训练到签署闭环",
                expected="pass",
                actual="error",
                passed=False,
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
        )

    if baseline:
        checks.append(_case_reload_check(service, baseline))
        checks.append(_audit_integrity_check(service, baseline))
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"{gate_id.lower()}-",
                dir=service.work_dir,
            ) as temporary:
                sandbox_root = Path(temporary)
                checks.append(
                    _negative_guard_check(
                        service,
                        baseline,
                        sandbox_root / "incompatible-scope",
                        scenario="incompatible_scope",
                        selected_dataset=config.outside_scope_dataset,
                        criterion_set_id=config.criterion_set_id,
                    )
                )
                checks.append(
                    _negative_guard_check(
                        service,
                        baseline,
                        sandbox_root / "missing-evidence",
                        scenario="missing_evidence",
                        selected_dataset=config.dataset,
                        criterion_set_id=config.criterion_set_id,
                    )
                )
        except Exception as exc:
            for check_id, title in (
                ("incompatible_scope_blocked", "不兼容范围阻断"),
                ("missing_evidence_blocked", "必需证据缺失阻断"),
            ):
                if not any(item["id"] == check_id for item in checks):
                    checks.append(
                        _check(
                            check_id,
                            title,
                            expected="blocked",
                            actual="error",
                            passed=False,
                            details={"error_type": type(exc).__name__, "error": str(exc)},
                        )
                    )
    else:
        checks.extend(
            [
                _not_run("case_reload_stable", "刷新及服务重载保持同一案例", "pass"),
                _not_run("audit_integrity", "EvidencePack 与审计事件完整性", "pass"),
                _not_run("incompatible_scope_blocked", "不兼容范围阻断", "blocked"),
                _not_run("missing_evidence_blocked", "必需证据缺失阻断", "blocked"),
            ]
        )

    passed_count = sum(bool(item["passed"]) for item in checks)
    report = {
        "schema_version": GATE_CONTRACT,
        "phase": "3.0",
        "gate_id": gate_id,
        "status": "passed" if passed_count == len(checks) else "failed",
        "started_at": started_at,
        "completed_at": _now(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "profile_name": baseline.get("profile_name") or profile_name,
        "case_id": baseline.get("case_id") or "",
        "case_version": baseline.get("case_version"),
        "summary": {
            "total": len(checks),
            "passed": passed_count,
            "failed": len(checks) - passed_count,
        },
        "checks": checks,
        "baseline": baseline,
        "url": baseline.get("url") or "",
    }
    return report


def render_phase30_release_gate_markdown(report: dict) -> str:
    lines = [
        "# 阶段 3.0 自动回归门禁报告",
        "",
        f"- 门禁：`{report.get('gate_id') or ''}`",
        f"- 结论：**{'通过' if report.get('status') == 'passed' else '未通过'}**",
        f"- 案例：`{report.get('case_id') or '未生成'}`",
        f"- Profile：`{report.get('profile_name') or ''}`",
        f"- 用时：{report.get('duration_seconds', 0)} 秒",
        "",
        "| 检查 | 预期 | 实际 | 结果 |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.get("checks") or []:
        lines.append(
            "| {title} | `{expected}` | `{actual}` | {result} |".format(
                title=str(item.get("title") or item.get("id") or "").replace("|", "/"),
                expected=item.get("expected") or "",
                actual=item.get("actual") or "",
                result="通过" if item.get("passed") else "未通过",
            )
        )
    if report.get("url"):
        lines.extend(["", f"验收页面：{report['url']}"])
    lines.extend(
        [
            "",
            "说明：负向场景在隔离副本中运行；预期被阻断即表示门禁工作正常。",
            "EvidencePack 当前仍为本地审计摘要签署，不等同于外部证书签名。",
            "",
        ]
    )
    return "\n".join(lines)


def _pipeline_check(summary: dict) -> dict:
    pipeline = summary.get("pipeline") or {}
    readiness = summary.get("signing_readiness") or {}
    evidence = summary.get("evidence_coverage") or {}
    retest = summary.get("retest") or {}
    reconciliation = summary.get("reconciliation") or {}
    conditions = {
        "pipeline_completed": summary.get("status") == "completed",
        "preflight_ready": pipeline.get("preflight_ready") is True,
        "training_succeeded": pipeline.get("training_status") == "succeeded",
        "training_has_models": int(pipeline.get("training_model_count") or 0) > 0,
        "inference_succeeded": pipeline.get("inference_status") == "succeeded",
        "inference_has_records": int(pipeline.get("inference_records") or 0) > 0,
        "prediction_facts_present": int(summary.get("prediction_records") or 0) > 0,
        "evidence_complete": evidence.get("state") == "complete",
        "verdict_pass": (summary.get("verdict") or {}).get("result") == "pass",
        "review_approved": (summary.get("review") or {}).get("state") == "approved",
        "controlled_retest_pass": (
            retest.get("status") == "completed"
            and retest.get("execution_mode") == "model_reinference"
            and retest.get("prediction_recomputed") is True
            and retest.get("verdict") == "pass"
        ),
        "kpi_reconciled": reconciliation.get("status") == "match",
        "evidence_pack_signed": (summary.get("evidence_pack") or {}).get("state") == "signed",
        "release_ready": readiness.get("can_sign") is True,
        "immutable_lineage_present": all(
            bool(summary.get(key))
            for key in (
                "comparison_id",
                "model_artifact_id",
                "dataset_snapshot_id",
                "split_assignment_id",
            )
        ),
    }
    return _check(
        "measured_pipeline",
        "真实训练到签署闭环",
        expected="pass",
        actual="pass" if all(conditions.values()) else "fail",
        passed=all(conditions.values()),
        details={
            "conditions": conditions,
            "training_job_id": summary.get("training_job_id") or "",
            "inference_job_id": summary.get("inference_job_id") or "",
            "model_artifact_id": summary.get("model_artifact_id") or "",
            "selected_sample_id": summary.get("selected_sample_id") or "",
        },
    )


def _case_reload_check(service: EvaluationService, baseline: dict) -> dict:
    reloaded = EvaluationService(
        data_dir=service.data_dir,
        work_dir=service.work_dir,
        runs_dir=service.runs_dir,
        frontend_data_dir=service.frontend_data_dir,
        allow_unsafe_plugins=False,
    )
    case = reloaded.get_evaluation_case(baseline["case_id"])
    resolved = reloaded.resolve_evaluation_case(
        {
            "profile_name": case["profile_name"],
            "idempotency_key": case["idempotency_key"],
        }
    )
    workspace = reloaded.evaluation_case_review_workspace(case["id"])
    signed = next(
        (item for item in workspace.get("evidence_packs") or [] if item.get("state") == "signed"),
        {},
    )
    execution = case.get("execution") or {}
    selection = case.get("evidence_selection") or {}
    conditions = {
        "same_case_id": case.get("id") == baseline.get("case_id"),
        "same_case_version": case.get("version") == baseline.get("case_version"),
        "same_profile": case.get("profile_name") == baseline.get("profile_name"),
        "same_training_job": execution.get("training_job_id") == baseline.get("training_job_id"),
        "same_inference_job": execution.get("inference_job_id") == baseline.get("inference_job_id"),
        "same_sample": selection.get("sample_id") == baseline.get("selected_sample_id"),
        "same_pack": signed.get("id") == (baseline.get("evidence_pack") or {}).get("id"),
        "same_pack_digest": signed.get("digest") == (baseline.get("evidence_pack") or {}).get("digest"),
        "idempotent_resolve": resolved.get("created") is False
        and (resolved.get("case") or {}).get("id") == case.get("id"),
    }
    return _check(
        "case_reload_stable",
        "刷新及服务重载保持同一案例",
        expected="pass",
        actual="pass" if all(conditions.values()) else "fail",
        passed=all(conditions.values()),
        details={"conditions": conditions, "case_id": case.get("id"), "case_version": case.get("version")},
    )


def _audit_integrity_check(service: EvaluationService, baseline: dict) -> dict:
    workspace = service.evaluation_case_review_workspace(baseline["case_id"])
    signed = next(
        (item for item in workspace.get("evidence_packs") or [] if item.get("state") == "signed"),
        {},
    )
    manifest_digest = _digest_json(signed.get("manifest") or {})
    expected_digest = str(signed.get("digest") or "").removeprefix("sha256:")
    pack_cas = _cas_path(service.work_dir / "artifacts", expected_digest)
    events = workspace.get("events") or []
    event_types = {str(item.get("event_type") or "") for item in events}
    event_cas_valid = all(_event_cas_valid(service.work_dir / "artifacts", item) for item in events)
    event_keys = [
        (str(item.get("event_source") or ""), int(item.get("sequence") or 0))
        for item in events
    ]
    tampered = deepcopy(signed.get("manifest") or {})
    tampered["case_id"] = f"{tampered.get('case_id') or 'case'}-tampered"
    conditions = {
        "signed_pack_present": bool(signed),
        "manifest_digest_matches": bool(expected_digest) and manifest_digest == expected_digest,
        "manifest_cas_present": pack_cas.is_file(),
        "manifest_cas_valid": pack_cas.is_file()
        and hashlib.sha256(pack_cas.read_bytes()).hexdigest() == expected_digest,
        "required_events_present": REQUIRED_EVENT_TYPES.issubset(event_types),
        "event_cas_valid": bool(events) and event_cas_valid,
        "event_sequence_keys_unique": len(event_keys) == len(set(event_keys)),
        "tamper_fixture_detected": _digest_json(tampered) != expected_digest,
    }
    return _check(
        "audit_integrity",
        "EvidencePack 与审计事件完整性",
        expected="pass",
        actual="pass" if all(conditions.values()) else "fail",
        passed=all(conditions.values()),
        details={
            "conditions": conditions,
            "evidence_pack_id": signed.get("id") or "",
            "evidence_pack_digest": signed.get("digest") or "",
            "event_count": len(events),
            "event_types": sorted(event_types),
        },
    )


def _negative_guard_check(
    service: EvaluationService,
    baseline: dict,
    sandbox_root: Path,
    *,
    scenario: str,
    selected_dataset: str,
    criterion_set_id: str,
) -> dict:
    profile_name = str(baseline["profile_name"])
    source_profile = service.frontend_data_dir / "profiles" / profile_name
    if not source_profile.is_dir():
        raise ValueError(f"baseline profile is unavailable: {source_profile}")
    frontend_data = sandbox_root / "frontend" / "app" / "data"
    target_profile = frontend_data / "profiles" / profile_name
    target_profile.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_profile, target_profile)
    if scenario == "missing_evidence":
        manifest_path = target_profile / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing_root = sandbox_root / "intentionally-missing"
        manifest["curve_artifacts"] = [
            {
                **item,
                "source": str(missing_root / str(item.get("filename") or "curves.json")),
                "copied": False,
            }
            for item in (manifest.get("curve_artifacts") or [])
        ]
        _write_json(manifest_path, manifest)

    isolated = EvaluationService(
        data_dir=service.data_dir,
        work_dir=sandbox_root / "work",
        runs_dir=sandbox_root / "runs",
        frontend_data_dir=frontend_data,
        allow_unsafe_plugins=False,
    )
    baseline_case = service.get_evaluation_case(baseline["case_id"])
    candidate = next(
        (
            item
            for item in baseline_case.get("candidates") or []
            if item.get("model_artifact_id") == baseline.get("model_artifact_id")
        ),
        (baseline_case.get("candidates") or [{}])[0],
    )
    resolved = isolated.resolve_evaluation_case(
        {
            "profile_name": profile_name,
            "idempotency_key": f"phase30-negative:{scenario}:{uuid.uuid4().hex}",
            "title": f"阶段 3.0 负向门禁 · {scenario}",
            "source_class": "measured",
            "execution": {
                "state": "completed",
                "training_job_id": baseline.get("training_job_id") or "",
                "inference_job_id": baseline.get("inference_job_id") or "",
                "run_id": baseline.get("inference_job_id") or "",
            },
            "candidates": [candidate],
        }
    )
    case = resolved["case"]
    baseline_selection = baseline_case.get("evidence_selection") or {}
    selection = {
        "candidate_id": candidate.get("candidate_id") or "",
        "dataset": selected_dataset,
        "asset_id": baseline_selection.get("asset_id") or "",
        "cycle_id": baseline_selection.get("cycle_id"),
        "evidence_tab": "raw",
    }
    if scenario == "incompatible_scope":
        selection.update({"asset_id": "", "cycle_id": None})
    case = isolated.update_evaluation_case_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "criterion_set_id": criterion_set_id,
            "selection": selection,
            "actor": "phase30-negative-fixture",
            "reason": f"expected release block: {scenario}",
        },
    )
    workspace = isolated.evaluation_case_review_workspace(case["id"])
    pack = isolated.create_case_evidence_pack(
        case["id"],
        {"expected_case_version": case["version"], "actor": "phase30-negative-fixture"},
    )
    sign_error = ""
    try:
        isolated.sign_evidence_pack(
            pack["id"],
            {
                "expected_case_version": case["version"],
                "signer": "phase30-negative-fixture",
                "reason": "this negative fixture must not be signable",
            },
        )
    except ReviewWorkflowError as exc:
        sign_error = str(exc)

    missing_required = sorted(
        item["kind"]
        for item in case["evidence_coverage"].get("items") or []
        if item.get("required") and not item.get("available")
    )
    if scenario == "incompatible_scope":
        scenario_ok = all(
            kind in missing_required
            for kind in ("comparison_scope", "split_assignment", "prediction_record")
        )
        check_id = "incompatible_scope_blocked"
        title = "不兼容范围阻断"
    elif scenario == "missing_evidence":
        scenario_ok = "raw_curve" in missing_required
        check_id = "missing_evidence_blocked"
        title = "必需证据缺失阻断"
    else:
        raise ValueError(f"unknown negative gate scenario: {scenario}")
    conditions = {
        "evidence_incomplete": case["evidence_coverage"].get("state") == "incomplete",
        "release_not_ready": workspace["signing_readiness"].get("can_sign") is False,
        "expected_missing_items": scenario_ok,
        "signing_rejected": "signing is blocked" in sign_error,
    }
    return _check(
        check_id,
        title,
        expected="blocked",
        actual="blocked" if all(conditions.values()) else "not_blocked",
        passed=all(conditions.values()),
        details={
            "conditions": conditions,
            "selected_dataset": selected_dataset,
            "missing_required": missing_required,
            "readiness_blockers": workspace["signing_readiness"].get("blockers") or [],
            "sign_error": sign_error,
        },
    )


def _event_cas_valid(cas_root: Path, event: dict) -> bool:
    digest = str(event.get("payload_digest") or "")
    if not digest.startswith("sha256:"):
        return False
    raw = digest.removeprefix("sha256:")
    path = _cas_path(cas_root, raw)
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == raw


def _cas_path(root: Path, digest: str) -> Path:
    return root / "sha256" / digest[:2] / digest


def _digest_json(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _check(
    check_id: str,
    title: str,
    *,
    expected: str,
    actual: str,
    passed: bool,
    details: dict,
) -> dict:
    return {
        "id": check_id,
        "title": title,
        "expected": expected,
        "actual": actual,
        "passed": bool(passed),
        "details": details,
    }


def _not_run(check_id: str, title: str, expected: str) -> dict:
    return _check(
        check_id,
        title,
        expected=expected,
        actual="not_run",
        passed=False,
        details={"reason": "measured pipeline did not produce a baseline case"},
    )


def _gate_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"GATE-{stamp}-{uuid.uuid4().hex[:8].upper()}"


def _profile_name(dataset: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_dataset = "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(dataset or "dataset")
    ).strip("-")
    return f"phase30-gate-{safe_dataset or 'dataset'}-{stamp}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
