from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator, FormatChecker

from .case_events import (
    CaseEventConflictError,
    CaseEventNotFoundError,
    CaseEventStore,
    CaseEventStoreError,
)


SCHEMA_VERSION = "1.3.0"
CASE_ID_RE = re.compile(r"^EV-[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
SOURCE_CLASSES = {"measured", "simulated", "demo"}
LIFECYCLE_STATES = {"draft", "active", "completed", "failed", "archived"}
EXECUTION_STATES = {"idle", "queued", "running", "completed", "failed"}
EVIDENCE_STATES = {"empty", "incomplete", "complete"}
VERDICT_RESULTS = {"pending", "pass", "fail", "conditional", "not_applicable"}
REVIEW_STATES = {"not_started", "in_review", "approved", "rejected"}
EVIDENCE_TABS = {"overview", "raw", "icdv", "features", "contribution"}
SAVED_SOURCE_STAGES = {"training", "inference"}
VALIDATION_KINDS = {"training_internal", "replay", "cross_dataset"}


class EvaluationCaseError(ValueError):
    pass


class EvaluationCaseNotFoundError(KeyError):
    pass


class EvaluationCaseConflictError(EvaluationCaseError):
    pass


class EvaluationCaseValidationError(EvaluationCaseError):
    pass


class EvaluationCaseStore:
    """Event-sourced evaluation cases with JSON compatibility projections."""

    def __init__(
        self,
        root: str | Path,
        *,
        evidence_deriver: Callable[[dict, dict], dict] | None = None,
        require_context_binding: bool = False,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._evidence_deriver = evidence_deriver
        self._require_context_binding = bool(require_context_binding)
        self.events = CaseEventStore(
            self.root.parent / "evaluation_cases.sqlite3",
            self.root.parent / "artifacts",
        )
        self._migrate_json_projections()
        self._migrate_event_snapshots()

    def create(self, payload: dict | None = None) -> dict:
        with self._lock:
            document = build_evaluation_case(payload or {})
            path = self._path(document["id"])
            if path.exists():
                raise EvaluationCaseConflictError(f"case already exists: {document['id']}")
            try:
                saved = self.events.create(document, actor="local-ui", reason="case created")
            except CaseEventStoreError as exc:
                raise EvaluationCaseConflictError(str(exc)) from exc
            self._write_projection(saved)
            return deepcopy(saved)

    def resolve(self, payload: dict | None = None) -> tuple[dict, bool]:
        payload = dict(payload or {})
        _plain_text(payload.get("profile_name"), "profile_name", required=True, max_length=128)
        idempotency_key = _plain_text(
            payload.get("idempotency_key"), "idempotency_key", required=True, max_length=240
        )
        with self._lock:
            active = self.events.find_by_idempotency_key(idempotency_key)
            if active is None:
                try:
                    return self.create(payload), True
                except EvaluationCaseConflictError:
                    active = self.events.find_by_idempotency_key(idempotency_key)
                    if active is None:
                        raise
                    return deepcopy(active), False
            return deepcopy(active), False

    def get(self, case_id: str) -> dict:
        self._path(case_id)
        try:
            document = self.events.get(case_id)
        except CaseEventNotFoundError as exc:
            raise EvaluationCaseNotFoundError(case_id)
        validate_case_document(document)
        return document

    def update_identity(self, case_id: str, payload: dict | None = None) -> dict:
        """Attach execution and immutable candidate lineage to an existing case."""
        payload = dict(payload or {})
        with self._lock:
            current = self.get(case_id)
            expected_version = payload.get("expected_version")
            if expected_version is None or int(expected_version) != current["version"]:
                raise EvaluationCaseConflictError(
                    f"case version conflict: expected {expected_version}, current {current['version']}"
                )
            self._assert_context_binding(current, payload)
            if current["lifecycle_state"] in {"completed", "archived"}:
                raise EvaluationCaseConflictError(
                    f"case is immutable in lifecycle state: {current['lifecycle_state']}"
                )

            updated = merge_case_identity(current, payload)
            _activate_draft_if_started(updated)
            _refresh_context_binding(updated, current=current)
            if _meaningful_case_state(updated) == _meaningful_case_state(current):
                return deepcopy(current)

            updated["version"] = current["version"] + 1
            updated["updated_at"] = _now()
            validate_case_document(updated)
            try:
                saved = self.events.commit(
                    updated,
                    expected_version=int(expected_version),
                    event_type="case.identity_updated",
                    actor=_plain_text(payload.get("actor") or "local-ui", "actor", max_length=160),
                    reason=_plain_text(
                        payload.get("reason") or "case execution lineage updated",
                        "reason",
                        max_length=500,
                    ),
                )
            except CaseEventConflictError as exc:
                raise EvaluationCaseConflictError(str(exc)) from exc
            self._write_projection(saved)
            return deepcopy(saved)

    def update_saved_combinations(self, case_id: str, payload: dict | None = None) -> dict:
        """Curate completed immutable runs without changing execution completion."""
        payload = dict(payload or {})
        with self._lock:
            current = self.get(case_id)
            expected_version = payload.get("expected_version")
            if expected_version is None or int(expected_version) != current["version"]:
                raise EvaluationCaseConflictError(
                    f"case version conflict: expected {expected_version}, current {current['version']}"
                )
            self._assert_context_binding(current, payload)
            if current["lifecycle_state"] == "archived":
                raise EvaluationCaseConflictError("archived case is immutable")

            action = str(payload.get("action") or "save")
            updated = deepcopy(current)
            if action == "save":
                incoming = _unique_saved_combinations(payload.get("items") or [])
                if not incoming:
                    raise EvaluationCaseValidationError("at least one saved combination is required")
                existing = {
                    item["saved_combination_id"]: item
                    for item in updated.get("saved_combinations") or []
                }
                for item in incoming:
                    existing[item["saved_combination_id"]] = item
                updated["saved_combinations"] = list(existing.values())
                event_type = "combination.saved"
                default_reason = "completed combinations added to evidence workspace"
            elif action == "remove":
                ids = set(_unique_text_list(
                    payload.get("saved_combination_ids") or [],
                    "saved_combination_ids",
                    max_items=512,
                ))
                if not ids:
                    raise EvaluationCaseValidationError("at least one saved combination id is required")
                updated["saved_combinations"] = [
                    item for item in updated.get("saved_combinations") or []
                    if item["saved_combination_id"] not in ids
                ]
                if updated.get("active_saved_combination_id") in ids:
                    updated["active_saved_combination_id"] = ""
                event_type = "combination.removed"
                default_reason = "saved combinations removed from evidence workspace"
            else:
                raise EvaluationCaseValidationError(f"unsupported saved combination action: {action}")

            _refresh_context_binding(updated, current=current)
            if _meaningful_case_state(updated) == _meaningful_case_state(current):
                return deepcopy(current)
            updated["version"] = current["version"] + 1
            updated["updated_at"] = _now()
            validate_case_document(updated)
            try:
                saved = self.events.commit(
                    updated,
                    expected_version=int(expected_version),
                    event_type=event_type,
                    actor=_plain_text(payload.get("actor") or "local-ui", "actor", max_length=160),
                    reason=_plain_text(payload.get("reason") or default_reason, "reason", max_length=500),
                )
            except CaseEventConflictError as exc:
                raise EvaluationCaseConflictError(str(exc)) from exc
            self._write_projection(saved)
            return deepcopy(saved)

    def activate_saved_combination(self, case_id: str, payload: dict | None = None) -> dict:
        """Select one curated result; inference selections restore their evidence lineage."""
        payload = dict(payload or {})
        with self._lock:
            current = self.get(case_id)
            expected_version = payload.get("expected_version")
            if expected_version is None or int(expected_version) != current["version"]:
                raise EvaluationCaseConflictError(
                    f"case version conflict: expected {expected_version}, current {current['version']}"
                )
            self._assert_context_binding(current, payload)
            if current["lifecycle_state"] == "archived":
                raise EvaluationCaseConflictError("archived case is immutable")
            saved_id = _plain_text(
                payload.get("saved_combination_id"),
                "saved_combination_id",
                required=True,
                max_length=160,
            )
            combination = next(
                (
                    item for item in current.get("saved_combinations") or []
                    if item["saved_combination_id"] == saved_id
                ),
                None,
            )
            if combination is None:
                raise EvaluationCaseValidationError(f"unknown saved combination: {saved_id}")

            updated = deepcopy(current)
            updated["active_saved_combination_id"] = saved_id
            candidate = deepcopy(combination["candidate"])
            updated["candidates"] = [candidate]
            if combination["source_stage"] == "inference":
                updated["execution"] = normalize_execution(
                    {
                        "state": "completed",
                        "training_job_id": combination.get("training_job_id") or current["execution"].get("training_job_id"),
                        "inference_job_id": combination["inference_job_id"],
                        "run_id": combination["run_id"] or combination["inference_job_id"],
                        "failure_reason": "",
                    },
                    current=updated["execution"],
                )
                selection = normalize_evidence_selection({
                    "comparison_id": saved_id,
                    "candidate_id": candidate["candidate_id"],
                    "dataset": combination["target_dataset"],
                    "dataset_snapshot_id": combination["dataset_snapshot_id"],
                    "split_assignment_id": "",
                    "slice_id": "",
                    "asset_id": "",
                    "sample_id": "",
                    "cycle_id": None,
                    "evidence_tab": "overview",
                    "compare_cycle_ids": [],
                    "curve_ref": {
                        "record_id": "",
                        "source_uri": "",
                        "signals": [],
                        "availability": "unknown",
                        "missing_reason": "",
                    },
                })
                if self._evidence_deriver is None:
                    updated["evidence_selection"] = selection
                    updated["evidence_coverage"] = normalize_evidence_coverage({})
                    updated["verdict"] = normalize_verdict({
                        "criterion_set_version": combination["criterion_set_version"],
                        "result": combination["verdict_result"],
                        "scope": "candidate",
                        "observed": combination["metrics"],
                    })
                else:
                    derived = self._evidence_deriver(updated, {
                        "selection": selection,
                        "criterion_set_id": combination["criterion_set_version"],
                    })
                    updated["evidence_selection"] = derived["selection"]
                    updated["evidence_coverage"] = derived["coverage"]
                    updated["verdict"] = derived["verdict"]
            else:
                training_dataset = next(iter(combination.get("training_datasets") or []), "")
                updated["execution"] = normalize_execution(
                    {
                        "state": "completed",
                        "training_job_id": combination["training_job_id"],
                        "inference_job_id": "",
                        "run_id": combination["run_id"] or combination["training_job_id"],
                        "failure_reason": "",
                    },
                    current=updated["execution"],
                )
                updated["evidence_selection"] = normalize_evidence_selection({
                    "comparison_id": saved_id,
                    "candidate_id": candidate["candidate_id"],
                    "dataset": training_dataset,
                    "dataset_snapshot_id": combination["dataset_snapshot_id"],
                    "split_assignment_id": "",
                    "slice_id": "",
                    "asset_id": "",
                    "sample_id": "",
                    "cycle_id": None,
                    "evidence_tab": "overview",
                    "compare_cycle_ids": [],
                    "curve_ref": {
                        "record_id": "",
                        "source_uri": "",
                        "signals": [],
                        "availability": "not_generated",
                        "missing_reason": "training_internal selection has no target inference record",
                    },
                })
                updated["evidence_coverage"] = normalize_evidence_coverage({})
                updated["verdict"] = normalize_verdict({
                    "criterion_set_version": combination["criterion_set_version"],
                    "result": combination["verdict_result"],
                    "scope": "training_internal",
                    "observed": combination["metrics"],
                })

            # Activating a combination rewrites candidates, evidence selection
            # and verdict; an approval given for the previous evidence does
            # not carry over (signing_readiness only checks review.state).
            previous_criterion_version = str(
                (current.get("verdict") or {}).get("criterion_set_version") or ""
            )
            if (updated.get("review") or {}).get("state") == "approved" and (
                _evidence_identity(updated) != _evidence_identity(current)
                or str(updated["verdict"].get("criterion_set_version") or "") != previous_criterion_version
            ):
                previous_review = updated.get("review") or {}
                updated["review"] = normalize_review({
                    "state": "in_review",
                    "reviewer": previous_review.get("reviewer") or "",
                    "note": "已切换到其他已保存组合，需要重新批准",
                    "updated_at": _now(),
                })

            _refresh_context_binding(updated, current=current)
            if _meaningful_case_state(updated) == _meaningful_case_state(current):
                return deepcopy(current)
            updated["version"] = current["version"] + 1
            updated["updated_at"] = _now()
            validate_case_document(updated)
            try:
                saved = self.events.commit(
                    updated,
                    expected_version=int(expected_version),
                    event_type="combination.activated",
                    actor=_plain_text(payload.get("actor") or "local-ui", "actor", max_length=160),
                    reason=_plain_text(payload.get("reason") or "saved combination selected", "reason", max_length=500),
                )
            except CaseEventConflictError as exc:
                raise EvaluationCaseConflictError(str(exc)) from exc
            self._write_projection(saved)
            return deepcopy(saved)

    def list(
        self,
        *,
        profile_name: str | None = None,
        lifecycle_state: str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        if lifecycle_state and lifecycle_state not in LIFECYCLE_STATES:
            raise EvaluationCaseValidationError(f"unsupported lifecycle_state: {lifecycle_state}")
        rows = self.events.list(
            profile_name=profile_name,
            lifecycle_state=lifecycle_state,
            include_archived=include_archived,
            limit=limit,
        )
        for item in rows:
            validate_case_document(item)
        return rows

    def update_evidence(self, case_id: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        with self._lock:
            current = self.get(case_id)
            expected_version = payload.get("expected_version")
            if expected_version is None or int(expected_version) != current["version"]:
                raise EvaluationCaseConflictError(
                    f"case version conflict: expected {expected_version}, current {current['version']}"
                )
            self._assert_context_binding(current, payload)
            if current["lifecycle_state"] in {"completed", "archived"}:
                raise EvaluationCaseConflictError(
                    f"case is immutable in lifecycle state: {current['lifecycle_state']}"
                )

            updated = deepcopy(current)
            candidate_payload = payload.get("candidate")
            if candidate_payload:
                if self._require_context_binding:
                    raise EvaluationCaseValidationError(
                        "candidate identity is server-owned; update case identity before selecting evidence"
                    )
                candidate = normalize_candidate(candidate_payload)
                existing = {item["candidate_id"]: item for item in updated["candidates"]}
                existing[candidate["candidate_id"]] = candidate
                updated["candidates"] = list(existing.values())

            if self._evidence_deriver is None:
                if {"coverage", "verdict"}.intersection(payload):
                    raise EvaluationCaseValidationError(
                        "coverage and verdict require a server-side evidence deriver"
                    )
                derived = {
                    "selection": normalize_evidence_selection(
                        payload.get("selection") or {}, current=updated["evidence_selection"]
                    ),
                    "coverage": updated["evidence_coverage"],
                    "verdict": updated["verdict"],
                }
            else:
                derived = self._evidence_deriver(updated, payload)
            updated["evidence_selection"] = derived["selection"]
            updated["evidence_coverage"] = derived["coverage"]
            updated["verdict"] = derived["verdict"]
            previous_criterion_version = str(
                (current.get("verdict") or {}).get("criterion_set_version") or ""
            )
            criterion_changed = bool(previous_criterion_version) and (
                updated["verdict"].get("criterion_set_version") != previous_criterion_version
                or updated["verdict"].get("thresholds")
                != (current.get("verdict") or {}).get("thresholds")
            )
            # A review approval is bound to what was approved: the criterion
            # set AND the evidence identity (candidate × dataset).  Re-pinning
            # the evidence on another candidate or target must send an
            # approved case back to review, otherwise a pack could be signed
            # against an approval that was given for different evidence.
            selection_changed = _evidence_identity(updated) != _evidence_identity(current)
            if (criterion_changed or selection_changed) and (
                (updated.get("review") or {}).get("state") == "approved"
            ):
                previous_review = updated.get("review") or {}
                updated["review"] = normalize_review({
                    "state": "in_review",
                    "reviewer": previous_review.get("reviewer") or "",
                    "note": (
                        "判据或阈值已变更，需要重新批准" if criterion_changed
                        else "证据选择已变更（候选或数据集），需要重新批准"
                    ),
                    "updated_at": _now(),
                })
            _activate_draft_if_started(updated)
            _refresh_context_binding(updated, current=current)

            if _meaningful_case_state(updated) == _meaningful_case_state(current):
                return deepcopy(current)

            updated["version"] = current["version"] + 1
            updated["updated_at"] = _now()
            validate_case_document(updated)
            try:
                event_type = "criterion.updated" if criterion_changed else "evidence.pinned"
                default_reason = (
                    "criterion thresholds updated and verdict recomputed"
                    if criterion_changed else "evidence pinned"
                )
                saved = self.events.commit(
                    updated,
                    expected_version=int(expected_version),
                    event_type=event_type,
                    actor=_plain_text(payload.get("actor") or "local-ui", "actor", max_length=160),
                    reason=_plain_text(payload.get("reason") or default_reason, "reason", max_length=500),
                )
            except CaseEventConflictError as exc:
                raise EvaluationCaseConflictError(str(exc)) from exc
            self._write_projection(saved)
            return deepcopy(saved)

    def transition_lifecycle(self, case_id: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        with self._lock:
            current = self.get(case_id)
            expected_version = payload.get("expected_version")
            if expected_version is None or int(expected_version) != current["version"]:
                raise EvaluationCaseConflictError(
                    f"case version conflict: expected {expected_version}, current {current['version']}"
                )
            self._assert_context_binding(current, payload)
            target = str(payload.get("lifecycle_state") or "")
            allowed = {
                "draft": {"active", "archived"},
                "active": {"completed", "failed", "archived"},
                "failed": {"archived"},
                "completed": {"archived"},
                "archived": set(),
            }
            if target not in allowed.get(current["lifecycle_state"], set()):
                raise EvaluationCaseValidationError(
                    f"invalid lifecycle transition: {current['lifecycle_state']} -> {target}"
                )
            updated = deepcopy(current)
            updated["lifecycle_state"] = target
            _refresh_context_binding(updated, current=current)
            updated["version"] = current["version"] + 1
            updated["updated_at"] = _now()
            validate_case_document(updated)
            try:
                saved = self.events.commit(
                    updated,
                    expected_version=int(expected_version),
                    event_type=f"case.{target}",
                    actor=_plain_text(payload.get("actor") or "local-ui", "actor", max_length=160),
                    reason=_plain_text(payload.get("reason") or "lifecycle transition", "reason", max_length=500),
                )
            except CaseEventConflictError as exc:
                raise EvaluationCaseConflictError(str(exc)) from exc
            self._write_projection(saved)
            return deepcopy(saved)

    def update_review(self, case_id: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        with self._lock:
            current = self.get(case_id)
            expected_version = payload.get("expected_version")
            if expected_version is None or int(expected_version) != current["version"]:
                raise EvaluationCaseConflictError(
                    f"case version conflict: expected {expected_version}, current {current['version']}"
                )
            self._assert_context_binding(current, payload)
            if current["lifecycle_state"] == "archived":
                raise EvaluationCaseConflictError("archived case is immutable")
            target = str(payload.get("state") or "")
            allowed = {
                "not_started": {"in_review"},
                "in_review": {"approved", "rejected"},
                "approved": {"in_review"},
                "rejected": {"in_review"},
            }
            if target not in allowed.get(current["review"]["state"], set()):
                raise EvaluationCaseValidationError(
                    f"invalid review transition: {current['review']['state']} -> {target}"
                )
            reviewer = _plain_text(
                payload.get("reviewer"), "review.reviewer", required=True, max_length=160
            )
            note = _plain_text(payload.get("note"), "review.note", max_length=2000)
            if target in {"approved", "rejected"} and not note:
                raise EvaluationCaseValidationError("review decision requires a note")
            if target == "approved":
                if current["evidence_coverage"]["state"] != "complete":
                    raise EvaluationCaseValidationError("review approval requires complete evidence")
                if current["verdict"]["result"] not in {"pass", "conditional"}:
                    raise EvaluationCaseValidationError(
                        "review approval requires pass or conditional verdict"
                    )
            updated = deepcopy(current)
            updated["review"] = normalize_review(
                {
                    "state": target,
                    "reviewer": reviewer,
                    "note": note,
                    "updated_at": _now(),
                }
            )
            _refresh_context_binding(updated, current=current)
            updated["version"] = current["version"] + 1
            updated["updated_at"] = _now()
            validate_case_document(updated)
            try:
                saved = self.events.commit(
                    updated,
                    expected_version=int(expected_version),
                    event_type=f"review.{target}",
                    actor=reviewer,
                    reason=note or "review transition",
                )
            except CaseEventConflictError as exc:
                raise EvaluationCaseConflictError(str(exc)) from exc
            self._write_projection(saved)
            return deepcopy(saved)

    def list_events(self, case_id: str, *, after_sequence: int = 0) -> list[dict]:
        self.get(case_id)
        return self.events.list_events(case_id, after_sequence=after_sequence)

    def migration_errors(self) -> list[dict]:
        return self.events.list_migration_errors()

    def evidence_workspace(self, case_id: str, overrides: dict | None = None) -> dict:
        case = self.get(case_id)
        context = deepcopy(case["evidence_selection"])
        for key, value in (overrides or {}).items():
            if key in context and value not in (None, ""):
                context[key] = value
        return {
            "schema_version": SCHEMA_VERSION,
            "case": case,
            "context": context,
            "status_axes": {
                "execution": case["execution"]["state"],
                "evidence": case["evidence_coverage"]["state"],
                "verdict": case["verdict"]["result"],
                "review": case["review"]["state"],
            },
        }

    def assert_context_binding(self, case_id: str, binding_id: str | None) -> dict:
        """Return the case only when the caller is bound to its canonical semantic context."""
        case = self.get(case_id)
        expected = str(binding_id or "").strip()
        actual = str((case.get("context_binding") or {}).get("binding_id") or "")
        if not expected:
            raise EvaluationCaseConflictError("expected_context_binding_id is required")
        if expected != actual:
            raise EvaluationCaseConflictError(
                f"case context conflict: expected {expected}, current {actual}"
            )
        return case

    def _assert_context_binding(self, current: dict, payload: dict) -> None:
        if not self._require_context_binding:
            return
        expected = str(payload.get("expected_context_binding_id") or "").strip()
        actual = str((current.get("context_binding") or {}).get("binding_id") or "")
        if not expected:
            raise EvaluationCaseConflictError("expected_context_binding_id is required")
        if expected != actual:
            raise EvaluationCaseConflictError(
                f"case context conflict: expected {expected}, current {actual}"
            )

    def _path(self, case_id: str) -> Path:
        if not CASE_ID_RE.fullmatch(str(case_id or "")):
            raise EvaluationCaseValidationError(f"invalid case id: {case_id!r}")
        return self.root / f"{case_id}.json"

    def _write_projection(self, document: dict) -> None:
        validate_case_document(document)
        target = self._path(document["id"])
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        data = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_retry(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _migrate_json_projections(self) -> None:
        for path in sorted(self.root.glob("EV-*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                document = migrate_case_document(raw)
                validate_case_document(document)
                imported = self.events.import_legacy(document, source=str(path))
                self._write_projection(migrate_case_document(imported))
            except Exception as exc:
                self.events.record_migration_error(str(path), str(exc))

    def _migrate_event_snapshots(self) -> None:
        for raw in self.events.list(include_archived=True):
            try:
                migrated = migrate_case_document(raw)
                if migrated == raw:
                    continue
                migrated["version"] = int(raw.get("version") or 0) + 1
                migrated["updated_at"] = _now()
                validate_case_document(migrated)
                saved = self.events.commit(
                    migrated,
                    expected_version=int(raw.get("version") or 0),
                    event_type="case.schema_migrated",
                    actor="migration",
                    reason=f"evaluation case schema migrated to {SCHEMA_VERSION}",
                )
                self._write_projection(saved)
            except Exception as exc:
                self.events.record_migration_error(f"sqlite:{raw.get('id', 'unknown')}", str(exc))


def build_evaluation_case(payload: dict | None = None) -> dict:
    payload = dict(payload or {})
    timestamp = _now()
    case_id = payload.get("id") or _new_case_id()
    if not CASE_ID_RE.fullmatch(str(case_id)):
        raise EvaluationCaseValidationError(f"invalid case id: {case_id!r}")
    profile_name = _plain_text(payload.get("profile_name"), "profile_name", required=True, max_length=128)
    source_class = str(payload.get("source_class") or "measured")
    if source_class not in SOURCE_CLASSES:
        raise EvaluationCaseValidationError(f"unsupported source_class: {source_class}")
    execution = normalize_execution(payload.get("execution") or {})
    candidates = _unique_candidates(payload.get("candidates") or [])
    saved_combinations = _unique_saved_combinations(payload.get("saved_combinations") or [])
    evidence_selection = normalize_evidence_selection(payload.get("evidence_selection") or {})
    requested_lifecycle = str(payload.get("lifecycle_state") or "").strip()
    lifecycle_state = requested_lifecycle or (
        "active"
        if _case_has_started(execution, candidates, evidence_selection)
        else "draft"
    )
    if lifecycle_state not in LIFECYCLE_STATES:
        raise EvaluationCaseValidationError(f"unsupported lifecycle_state: {lifecycle_state}")

    document = {
        "schema_version": SCHEMA_VERSION,
        "id": str(case_id),
        "title": _plain_text(payload.get("title") or profile_name, "title", required=True, max_length=200),
        "profile_name": profile_name,
        "idempotency_key": _plain_text(
            payload.get("idempotency_key") or f"case:{case_id}",
            "idempotency_key",
            required=True,
            max_length=240,
        ),
        "source_class": source_class,
        "lifecycle_state": lifecycle_state,
        "version": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "execution": execution,
        "candidates": candidates,
        "saved_combinations": saved_combinations,
        "active_saved_combination_id": _plain_text(
            payload.get("active_saved_combination_id"),
            "active_saved_combination_id",
            max_length=160,
        ),
        "evidence_selection": evidence_selection,
        "evidence_coverage": normalize_evidence_coverage(payload.get("evidence_coverage") or {}),
        "verdict": normalize_verdict(payload.get("verdict") or {}),
        "review": normalize_review(payload.get("review") or {}),
    }
    document["context_binding"] = build_context_binding(document)
    selected_candidate = document["evidence_selection"].get("candidate_id")
    if selected_candidate and selected_candidate not in {
        item["candidate_id"] for item in document["candidates"]
    }:
        raise EvaluationCaseValidationError(
            f"evidence selection references unknown candidate: {selected_candidate}"
        )
    if document["active_saved_combination_id"] and document["active_saved_combination_id"] not in {
        item["saved_combination_id"] for item in document["saved_combinations"]
    }:
        raise EvaluationCaseValidationError("active saved combination references an unknown item")
    validate_case_document(document)
    return document


def merge_case_identity(current: dict, payload: dict) -> dict:
    updated = deepcopy(current)
    previous_execution = deepcopy(updated.get("execution") or {})
    for key, limit in (("title", 200),):
        if payload.get(key):
            updated[key] = _plain_text(payload[key], key, required=True, max_length=limit)
    if payload.get("source_class"):
        source_class = str(payload["source_class"])
        if source_class not in SOURCE_CLASSES:
            raise EvaluationCaseValidationError(f"unsupported source_class: {source_class}")
        updated["source_class"] = source_class
    if payload.get("execution"):
        updated["execution"] = normalize_execution(payload["execution"], current=updated["execution"])
    if "candidates" in payload:
        incoming = _unique_candidates(payload.get("candidates") or [])
        if payload.get("replace_candidates") is True:
            updated["candidates"] = incoming
            selected_candidate = str(
                (updated.get("evidence_selection") or {}).get("candidate_id") or ""
            )
            valid_candidate_ids = {item["candidate_id"] for item in incoming}
            if selected_candidate and selected_candidate not in valid_candidate_ids:
                updated["active_saved_combination_id"] = ""
                updated["evidence_selection"] = normalize_evidence_selection({})
                updated["evidence_coverage"] = normalize_evidence_coverage({})
                updated["verdict"] = normalize_verdict({})
                updated["review"] = normalize_review({})
        elif incoming:
            existing = {item["candidate_id"]: item for item in updated["candidates"]}
            for item in incoming:
                existing[item["candidate_id"]] = item
            updated["candidates"] = list(existing.values())
            # The merged pool keeps every candidate, but the evidence view
            # follows the authoritative run being merged in: a selection
            # outside it would 400 against the new run's records.
            selected_candidate = str(
                (updated.get("evidence_selection") or {}).get("candidate_id") or ""
            )
            incoming_ids = {item["candidate_id"] for item in incoming}
            if selected_candidate and selected_candidate not in incoming_ids:
                updated["active_saved_combination_id"] = ""
                updated["evidence_selection"] = normalize_evidence_selection({})
                updated["evidence_coverage"] = normalize_evidence_coverage({})
                updated["verdict"] = normalize_verdict({})
                updated["review"] = normalize_review({})
    next_execution = updated.get("execution") or {}
    training_changed = bool(next_execution.get("training_job_id")) and (
        str(next_execution.get("training_job_id") or "")
        != str(previous_execution.get("training_job_id") or "")
    )
    inference_changed = bool(next_execution.get("inference_job_id")) and (
        str(next_execution.get("inference_job_id") or "")
        != str(previous_execution.get("inference_job_id") or "")
    )
    if training_changed:
        updated["execution"]["inference_job_id"] = ""
        if not ("candidates" in payload and payload.get("candidates")):
            updated["candidates"] = []
        updated["active_saved_combination_id"] = ""
        updated["evidence_selection"] = normalize_evidence_selection({})
        updated["evidence_coverage"] = normalize_evidence_coverage({})
        updated["verdict"] = normalize_verdict({})
        updated["review"] = normalize_review({})
    elif inference_changed:
        updated["active_saved_combination_id"] = ""
        updated["evidence_selection"] = normalize_evidence_selection({})
        updated["evidence_coverage"] = normalize_evidence_coverage({})
        updated["verdict"] = normalize_verdict({})
        updated["review"] = normalize_review({})
    return updated


def migrate_case_document(document: dict) -> dict:
    migrated = deepcopy(document)
    version = str(migrated.get("schema_version") or "")
    if version == SCHEMA_VERSION:
        return migrated
    if version == "1.0.0":
        migrated["idempotency_key"] = migrated.get("idempotency_key") or f"legacy:{migrated.get('id')}"
        version = "1.1.0"
    if version == "1.1.0":
        migrated["saved_combinations"] = migrated.get("saved_combinations") or []
        migrated["active_saved_combination_id"] = migrated.get("active_saved_combination_id") or ""
        version = "1.2.0"
    if version == "1.2.0":
        candidate_ids = {
            str(item.get("candidate_id") or "")
            for item in migrated.get("candidates") or []
        }
        selected_candidate = str(
            (migrated.get("evidence_selection") or {}).get("candidate_id") or ""
        )
        if selected_candidate and selected_candidate not in candidate_ids:
            # Some legacy projections retained a selection after their
            # candidate list had been replaced.  Do not carry a dangling
            # evidence identity into the context-binding contract.
            migrated["active_saved_combination_id"] = ""
            migrated["evidence_selection"] = normalize_evidence_selection({})
            migrated["evidence_coverage"] = normalize_evidence_coverage({})
            migrated["verdict"] = normalize_verdict({})
            migrated["review"] = normalize_review({})
        migrated["schema_version"] = SCHEMA_VERSION
        migrated["context_binding"] = build_context_binding(migrated)
        return migrated
    raise EvaluationCaseValidationError(f"unsupported evaluation case schema_version: {version!r}")


def build_context_binding(document: dict, *, previous: dict | None = None) -> dict:
    """Build the canonical, server-owned semantic binding shared by all workflow pages."""
    execution = document.get("execution") or {}
    selection = document.get("evidence_selection") or {}
    saved_id = str(document.get("active_saved_combination_id") or "")
    saved = next(
        (
            item for item in document.get("saved_combinations") or []
            if str(item.get("saved_combination_id") or "") == saved_id
        ),
        None,
    )
    candidate_id = str(selection.get("candidate_id") or "")
    candidate = next(
        (
            item for item in document.get("candidates") or []
            if str(item.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if saved is not None:
        saved_candidate = saved.get("candidate") or {}
        if not candidate_id or candidate_id == str(saved_candidate.get("candidate_id") or ""):
            candidate = saved_candidate
            candidate_id = str(saved_candidate.get("candidate_id") or "")

    inference_job_id = str(execution.get("inference_job_id") or "")
    training_job_id = str(execution.get("training_job_id") or "")
    if saved is not None:
        training_job_id = str(saved.get("training_job_id") or training_job_id)
        inference_job_id = str(saved.get("inference_job_id") or inference_job_id)
    if inference_job_id:
        stage = "inference"
    elif training_job_id:
        stage = "training"
    else:
        stage = "design"

    training_datasets = list(saved.get("training_datasets") or []) if saved else []
    target_datasets = []
    if saved and saved.get("target_dataset"):
        target_datasets.append(str(saved["target_dataset"]))
    active_dataset = str(selection.get("dataset") or "")
    semantic = {
        "case_id": str(document.get("id") or ""),
        "stage": stage,
        "source": "saved_combination" if saved is not None else "case_execution",
        "training_job_id": training_job_id,
        "inference_job_id": inference_job_id,
        "run_id": str((saved or {}).get("run_id") or execution.get("run_id") or ""),
        "saved_combination_id": saved_id,
        "candidate_id": candidate_id,
        "model": str((candidate or {}).get("model") or ""),
        "featureset": str((candidate or {}).get("featureset") or ""),
        "model_artifact_id": str((candidate or {}).get("model_artifact_id") or ""),
        "training_datasets": training_datasets,
        "target_datasets": target_datasets,
        "active_dataset": active_dataset,
        "dataset_snapshot_id": str(
            selection.get("dataset_snapshot_id")
            or (saved or {}).get("dataset_snapshot_id")
            or ""
        ),
        "protocol": str((saved or {}).get("protocol") or ""),
    }
    encoded = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    binding_id = f"ctx@sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
    previous = previous or {}
    previous_id = str(previous.get("binding_id") or "")
    previous_generation = max(0, int(previous.get("generation") or 0))
    generation = previous_generation if previous_id == binding_id and previous_generation else previous_generation + 1
    return {"binding_id": binding_id, "generation": generation, **semantic}


def _refresh_context_binding(document: dict, *, current: dict | None = None) -> None:
    document["context_binding"] = build_context_binding(
        document,
        previous=(current or {}).get("context_binding") or {},
    )


def _meaningful_case_state(document: dict) -> dict:
    value = deepcopy(document)
    for key in ("version", "updated_at"):
        value.pop(key, None)
    if isinstance(value.get("verdict"), dict):
        value["verdict"].pop("updated_at", None)
    return value


def _activate_draft_if_started(document: dict) -> None:
    if document.get("lifecycle_state") != "draft":
        return
    if _case_has_started(
        document.get("execution") or {},
        document.get("candidates") or [],
        document.get("evidence_selection") or {},
    ):
        document["lifecycle_state"] = "active"


def _case_has_started(execution: dict, candidates: list[dict], selection: dict) -> bool:
    if str(execution.get("state") or "idle") != "idle":
        return True
    if any(execution.get(key) for key in ("training_job_id", "inference_job_id", "run_id")):
        return True
    if candidates:
        return True
    if any(
        selection.get(key) not in (None, "", [])
        for key in (
            "comparison_id", "candidate_id", "dataset", "dataset_snapshot_id",
            "split_assignment_id", "slice_id", "asset_id", "sample_id", "cycle_id",
            "compare_cycle_ids",
        )
    ):
        return True
    curve_ref = selection.get("curve_ref") or {}
    return bool(
        curve_ref.get("record_id")
        or curve_ref.get("source_uri")
        or curve_ref.get("signals")
        or curve_ref.get("availability") not in (None, "", "unknown")
    )


def normalize_execution(payload: dict, *, current: dict | None = None) -> dict:
    base = deepcopy(current) if current else {
        "state": "idle",
        "training_job_id": "",
        "inference_job_id": "",
        "run_id": "",
        "failure_reason": "",
    }
    for key in ("training_job_id", "inference_job_id", "run_id", "failure_reason"):
        if key in payload:
            base[key] = _plain_text(payload.get(key), key, max_length=500)
    if "state" in payload:
        state = str(payload.get("state") or "idle")
        if state not in EXECUTION_STATES:
            raise EvaluationCaseValidationError(f"unsupported execution state: {state}")
        base["state"] = state
    return base


def normalize_candidate(payload: dict) -> dict:
    model = _plain_text(payload.get("model"), "candidate.model", required=True, max_length=160)
    featureset = _plain_text(payload.get("featureset") or "full", "candidate.featureset", required=True, max_length=160)
    candidate_id = payload.get("candidate_id") or f"legacy:{_id_token(model)}:{_id_token(featureset)}"
    if not SAFE_ID_RE.fullmatch(str(candidate_id)):
        raise EvaluationCaseValidationError(f"invalid candidate_id: {candidate_id!r}")
    role = str(payload.get("role") or "candidate")
    if role not in {"candidate", "champion", "baseline"}:
        raise EvaluationCaseValidationError(f"unsupported candidate role: {role}")
    return {
        "candidate_id": str(candidate_id),
        "model": model,
        "featureset": featureset,
        "role": role,
        "model_artifact_id": _plain_text(payload.get("model_artifact_id"), "model_artifact_id", max_length=200),
        "run_id": _plain_text(payload.get("run_id"), "candidate.run_id", max_length=200),
    }


def normalize_saved_combination(payload: dict) -> dict:
    candidate = normalize_candidate(payload.get("candidate") or {})
    if not candidate["model_artifact_id"]:
        raise EvaluationCaseValidationError("saved combination requires an immutable model artifact")
    source_stage = str(payload.get("source_stage") or "")
    if source_stage not in SAVED_SOURCE_STAGES:
        raise EvaluationCaseValidationError(f"unsupported saved combination source stage: {source_stage}")
    validation_kind = str(payload.get("validation_kind") or "")
    if validation_kind not in VALIDATION_KINDS:
        raise EvaluationCaseValidationError(f"unsupported validation kind: {validation_kind}")
    if source_stage == "training" and validation_kind != "training_internal":
        raise EvaluationCaseValidationError("training combinations must use training_internal validation")
    if source_stage == "inference" and validation_kind == "training_internal":
        raise EvaluationCaseValidationError("inference combinations must use replay or cross_dataset validation")
    saved_id = _plain_text(
        payload.get("saved_combination_id"),
        "saved_combination_id",
        required=True,
        max_length=160,
    )
    if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,159}$", saved_id):
        raise EvaluationCaseValidationError(f"invalid saved_combination_id: {saved_id!r}")
    training_job_id = _plain_text(payload.get("training_job_id"), "training_job_id", max_length=500)
    inference_job_id = _plain_text(payload.get("inference_job_id"), "inference_job_id", max_length=500)
    if source_stage == "training" and not training_job_id:
        raise EvaluationCaseValidationError("training saved combination requires training_job_id")
    if source_stage == "inference" and not inference_job_id:
        raise EvaluationCaseValidationError("inference saved combination requires inference_job_id")
    target_dataset = _plain_text(payload.get("target_dataset"), "target_dataset", max_length=500)
    if source_stage == "inference" and not target_dataset:
        raise EvaluationCaseValidationError("inference saved combination requires target_dataset")
    verdict_result = str(payload.get("verdict_result") or "pending")
    if verdict_result not in VERDICT_RESULTS:
        raise EvaluationCaseValidationError(f"unsupported saved combination verdict: {verdict_result}")
    return {
        "saved_combination_id": saved_id,
        "source_stage": source_stage,
        "validation_kind": validation_kind,
        "label": _plain_text(
            payload.get("label") or f"{candidate['model']} × {candidate['featureset']}",
            "saved_combination.label",
            required=True,
            max_length=240,
        ),
        "candidate": candidate,
        "training_job_id": training_job_id,
        "inference_job_id": inference_job_id,
        "run_id": _plain_text(payload.get("run_id"), "saved_combination.run_id", max_length=500),
        "training_datasets": _unique_text_list(
            payload.get("training_datasets") or [],
            "training_datasets",
            max_items=32,
        ),
        "target_dataset": target_dataset,
        "dataset_snapshot_id": _plain_text(
            payload.get("dataset_snapshot_id"),
            "saved_combination.dataset_snapshot_id",
            max_length=500,
        ),
        "protocol": _plain_text(payload.get("protocol"), "saved_combination.protocol", max_length=160),
        "criterion_set_version": _plain_text(
            payload.get("criterion_set_version") or "engineering",
            "saved_combination.criterion_set_version",
            max_length=160,
        ),
        "metrics": _finite_mapping(payload.get("metrics") or {}, "saved_combination.metrics"),
        "verdict_result": verdict_result,
        "saved_at": _plain_text(payload.get("saved_at") or _now(), "saved_combination.saved_at", max_length=80),
    }


def normalize_evidence_selection(payload: dict, *, current: dict | None = None) -> dict:
    base = deepcopy(current) if current else {
        "comparison_id": "",
        "candidate_id": "",
        "dataset": "",
        "dataset_snapshot_id": "",
        "split_assignment_id": "",
        "slice_id": "",
        "asset_id": "",
        "sample_id": "",
        "cycle_id": None,
        "evidence_tab": "overview",
        "compare_cycle_ids": [],
        "curve_ref": {
            "record_id": "",
            "source_uri": "",
            "signals": [],
            "availability": "unknown",
            "missing_reason": "",
        },
    }
    for key in (
        "comparison_id", "candidate_id", "dataset", "dataset_snapshot_id", "split_assignment_id",
        "slice_id", "asset_id", "sample_id",
    ):
        if key in payload:
            base[key] = _plain_text(payload.get(key), key, max_length=500)
    if "cycle_id" in payload:
        base["cycle_id"] = _cycle_value(payload.get("cycle_id"), "cycle_id")
    if "evidence_tab" in payload:
        evidence_tab = str(payload.get("evidence_tab") or "overview")
        if evidence_tab not in EVIDENCE_TABS:
            raise EvaluationCaseValidationError(f"unsupported evidence_tab: {evidence_tab}")
        base["evidence_tab"] = evidence_tab
    if "compare_cycle_ids" in payload:
        values = [_cycle_value(value, "compare_cycle_ids") for value in (payload.get("compare_cycle_ids") or [])]
        values = [value for value in values if value is not None]
        if len(values) > 3:
            raise EvaluationCaseValidationError("compare_cycle_ids supports at most 3 values")
        if len({str(value) for value in values}) != len(values):
            raise EvaluationCaseValidationError("compare_cycle_ids contains duplicates")
        base["compare_cycle_ids"] = values
    if "curve_ref" in payload:
        base["curve_ref"] = normalize_curve_ref(payload.get("curve_ref") or {}, current=base["curve_ref"])
    return base


def normalize_curve_ref(payload: dict, *, current: dict | None = None) -> dict:
    base = deepcopy(current) if current else {
        "record_id": "",
        "source_uri": "",
        "signals": [],
        "availability": "unknown",
        "missing_reason": "",
    }
    for key in ("record_id", "source_uri", "missing_reason"):
        if key in payload:
            base[key] = _plain_text(payload.get(key), f"curve_ref.{key}", max_length=1000)
    if "signals" in payload:
        base["signals"] = _unique_text_list(payload.get("signals") or [], "curve_ref.signals", max_items=16)
    if "availability" in payload:
        availability = str(payload.get("availability") or "unknown")
        if availability not in {"unknown", "available", "missing", "not_generated", "forbidden"}:
            raise EvaluationCaseValidationError(f"unsupported curve availability: {availability}")
        base["availability"] = availability
    return base


def normalize_evidence_coverage(payload: dict) -> dict:
    items = []
    seen = set()
    for raw in payload.get("items") or []:
        kind = _plain_text(raw.get("kind"), "coverage.kind", required=True, max_length=80)
        if kind in seen:
            raise EvaluationCaseValidationError(f"duplicate evidence coverage kind: {kind}")
        seen.add(kind)
        items.append({
            "kind": kind,
            "required": bool(raw.get("required", True)),
            "available": bool(raw.get("available", False)),
            "missing_reason": _plain_text(raw.get("missing_reason"), "coverage.missing_reason", max_length=500),
        })
    required_items = [item for item in items if item["required"]]
    available_count = sum(1 for item in required_items if item["available"])
    if not required_items:
        state = "empty"
    elif available_count == len(required_items):
        state = "complete"
    else:
        state = "incomplete"
    requested_state = payload.get("state")
    if requested_state is not None and str(requested_state) not in EVIDENCE_STATES:
        raise EvaluationCaseValidationError(f"unsupported evidence state: {requested_state}")
    return {
        "state": state,
        "available_count": available_count,
        "required_count": len(required_items),
        "items": items,
    }


def _evidence_identity(case: dict) -> tuple[str, str]:
    selection = case.get("evidence_selection") or {}
    return (
        str(selection.get("candidate_id") or ""),
        str(selection.get("dataset") or ""),
    )


def normalize_verdict(payload: dict) -> dict:
    result = str(payload.get("result") or "pending")
    if result not in VERDICT_RESULTS:
        raise EvaluationCaseValidationError(f"unsupported verdict result: {result}")
    return {
        "criterion_set_version": _plain_text(payload.get("criterion_set_version"), "criterion_set_version", max_length=160),
        "result": result,
        "scope": _plain_text(payload.get("scope") or "case", "verdict.scope", required=True, max_length=160),
        "observed": _finite_mapping(payload.get("observed") or {}, "verdict.observed"),
        "thresholds": _finite_mapping(payload.get("thresholds") or {}, "verdict.thresholds"),
        "reasons": _unique_text_list(payload.get("reasons") or [], "verdict.reasons", max_items=32),
        "updated_at": _plain_text(payload.get("updated_at") or "", "verdict.updated_at", max_length=80),
    }


def normalize_review(payload: dict) -> dict:
    state = str(payload.get("state") or "not_started")
    if state not in REVIEW_STATES:
        raise EvaluationCaseValidationError(f"unsupported review state: {state}")
    return {
        "state": state,
        "reviewer": _plain_text(payload.get("reviewer"), "review.reviewer", max_length=160),
        "note": _plain_text(payload.get("note"), "review.note", max_length=2000),
        "updated_at": _plain_text(payload.get("updated_at") or "", "review.updated_at", max_length=80),
    }


def validate_case_document(document: dict) -> None:
    validator = _case_validator()
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "document"
        raise EvaluationCaseValidationError(f"invalid evaluation case at {location}: {error.message}")
    candidate_ids = {item["candidate_id"] for item in document.get("candidates") or []}
    selected_candidate = str((document.get("evidence_selection") or {}).get("candidate_id") or "")
    if selected_candidate and selected_candidate not in candidate_ids:
        raise EvaluationCaseValidationError(
            f"evidence selection references unknown candidate: {selected_candidate}"
        )
    saved_ids = {
        item["saved_combination_id"] for item in document.get("saved_combinations") or []
    }
    active_saved = str(document.get("active_saved_combination_id") or "")
    if active_saved and active_saved not in saved_ids:
        raise EvaluationCaseValidationError(
            "active saved combination references an unknown item"
        )
    current_binding = document.get("context_binding") or {}
    canonical_binding = build_context_binding(document, previous=current_binding)
    if current_binding != canonical_binding:
        raise EvaluationCaseValidationError(
            "context_binding is not the canonical server-derived case context"
        )


def _case_validator() -> Draft202012Validator:
    schema_path = Path(__file__).with_name("schemas") / "evaluation-case.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _unique_candidates(values) -> list[dict]:
    candidates = []
    seen = set()
    for value in values:
        candidate = normalize_candidate(value)
        if candidate["candidate_id"] in seen:
            raise EvaluationCaseValidationError(f"duplicate candidate_id: {candidate['candidate_id']}")
        seen.add(candidate["candidate_id"])
        candidates.append(candidate)
    return candidates


def _unique_saved_combinations(values) -> list[dict]:
    combinations = []
    seen = set()
    for value in values:
        combination = normalize_saved_combination(value)
        saved_id = combination["saved_combination_id"]
        if saved_id in seen:
            raise EvaluationCaseValidationError(f"duplicate saved_combination_id: {saved_id}")
        seen.add(saved_id)
        combinations.append(combination)
    return combinations


def _unique_text_list(values, field: str, *, max_items: int) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise EvaluationCaseValidationError(f"{field} must be a list")
    output = []
    seen = set()
    for value in values:
        text = _plain_text(value, field, required=True, max_length=500)
        if text not in seen:
            seen.add(text)
            output.append(text)
    if len(output) > max_items:
        raise EvaluationCaseValidationError(f"{field} supports at most {max_items} values")
    return output


def _finite_mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise EvaluationCaseValidationError(f"{field} must be an object")
    encoded = json.dumps(value, allow_nan=False)
    decoded = json.loads(encoded)
    return decoded


def _cycle_value(value, field: str):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise EvaluationCaseValidationError(f"{field} must be an integer or string")
    if isinstance(value, int):
        return value
    return _plain_text(value, field, required=True, max_length=160)


def _plain_text(value, field: str, *, required: bool = False, max_length: int = 500) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise EvaluationCaseValidationError(f"{field} is required")
    if len(text) > max_length:
        raise EvaluationCaseValidationError(f"{field} exceeds {max_length} characters")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        raise EvaluationCaseValidationError(f"{field} contains control characters")
    return text


def _id_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return (token or "unknown")[:48]


def _new_case_id() -> str:
    return f"EV-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 8) -> None:
    """Handle the short Windows sharing window created by concurrent JSON readers."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.01 * (attempt + 1))
