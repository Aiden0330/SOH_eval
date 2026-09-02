from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ISSUE_STATES = {"open", "in_retest", "resolved", "closed"}
ISSUE_SEVERITIES = {"minor", "major", "critical"}
RETEST_STATES = {"requested", "running", "completed", "failed"}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,159}$")


class ReviewWorkflowError(ValueError):
    pass


class ReviewWorkflowNotFoundError(KeyError):
    pass


class ReviewWorkflowConflictError(ReviewWorkflowError):
    pass


class ReviewWorkflowStore:
    """Durable issue -> regression set -> retest -> EvidencePack workflow."""

    def __init__(
        self,
        db_path: str | Path,
        cas_root: str | Path,
        *,
        get_case: Callable[[str], dict],
        list_case_events: Callable[..., list[dict]] | None = None,
        frontend_data_dir: str | Path,
        has_executable_model_artifact: Callable[[str], bool] | None = None,
    ):
        self.db_path = Path(db_path)
        self.cas_root = Path(cas_root)
        self.get_case = get_case
        self.list_case_events = list_case_events
        self.frontend_data_dir = Path(frontend_data_dir)
        self.has_executable_model_artifact = has_executable_model_artifact
        self._lock = threading.RLock()
        self._initialize()

    def workspace(self, case_id: str) -> dict:
        case = self.get_case(case_id)
        issues = self._rows(
            "SELECT * FROM review_issues WHERE case_id = ? ORDER BY created_at DESC", (case_id,)
        )
        regression_sets = self._rows(
            "SELECT * FROM regression_sets WHERE case_id = ? ORDER BY created_at DESC", (case_id,)
        )
        members = self._rows(
            """
            SELECT member.* FROM regression_members member
            JOIN regression_sets set_row ON set_row.id = member.regression_set_id
            WHERE set_row.case_id = ? ORDER BY member.added_at
            """,
            (case_id,),
        )
        by_set: dict[str, list[dict]] = {}
        for member in members:
            by_set.setdefault(member["regression_set_id"], []).append(member)
        for item in regression_sets:
            item["members"] = by_set.get(item["id"], [])
        retests = self._rows(
            "SELECT * FROM retest_requests WHERE case_id = ? ORDER BY created_at DESC", (case_id,)
        )
        reconciliations = self._rows(
            "SELECT * FROM kpi_reconciliations WHERE case_id = ? ORDER BY created_at DESC", (case_id,)
        )
        packs = self._rows(
            "SELECT * FROM evidence_packs WHERE case_id = ? ORDER BY created_at DESC", (case_id,)
        )
        readiness = self.signing_readiness(
            case,
            issues=issues,
            retests=retests,
            reconciliations=reconciliations,
        )
        workflow_events = self._rows(
            "SELECT * FROM workflow_events WHERE case_id = ? ORDER BY sequence DESC LIMIT 100",
            (case_id,),
        )
        for event in workflow_events:
            event["event_source"] = "workflow"
        case_events = []
        if self.list_case_events is not None:
            case_events = [
                {
                    **event,
                    "entity_type": "case",
                    "entity_id": case_id,
                    "payload_digest": f"sha256:{event.get('snapshot_digest', '')}",
                    "event_source": "case",
                }
                for event in self.list_case_events(case_id, after_sequence=0)
            ]
        events = sorted(
            workflow_events + case_events,
            key=lambda item: (str(item.get("created_at") or ""), int(item.get("sequence") or 0)),
            reverse=True,
        )[:100]
        return {
            "case_id": case_id,
            "case_version": case["version"],
            "issues": issues,
            "regression_sets": regression_sets,
            "retests": retests,
            "reconciliations": reconciliations,
            "evidence_packs": packs,
            "signing_readiness": readiness,
            "events": events,
        }

    def create_issue(self, payload: dict) -> dict:
        case_id = _required_text(payload.get("case_id"), "case_id", 80)
        case = self._expected_case(case_id, payload.get("expected_case_version"))
        severity = str(payload.get("severity") or "major")
        if severity not in ISSUE_SEVERITIES:
            raise ReviewWorkflowError(f"unsupported issue severity: {severity}")
        selection = case["evidence_selection"]
        context = {
            "candidate_id": selection.get("candidate_id") or "",
            "dataset": selection.get("dataset") or "",
            "dataset_snapshot_id": selection.get("dataset_snapshot_id") or "",
            "prediction_snapshot_id": _prediction_snapshot_ref(
                self.frontend_data_dir,
                case.get("profile_name") or "",
                selection.get("dataset") or "",
            ),
            "split_assignment_id": selection.get("split_assignment_id") or "",
            "asset_id": selection.get("asset_id") or "",
            "sample_id": selection.get("sample_id") or "",
            "cycle_id": selection.get("cycle_id"),
            "model_artifact_id": next(
                (
                    item.get("model_artifact_id") or ""
                    for item in case.get("candidates") or []
                    if item.get("candidate_id") == selection.get("candidate_id")
                ),
                "",
            ),
            "criterion_set_version": case["verdict"].get("criterion_set_version") or "",
            "verdict": case["verdict"].get("result") or "pending",
            "verdict_observed": dict(case["verdict"].get("observed") or {}),
            "verdict_thresholds": dict(case["verdict"].get("thresholds") or {}),
            "missing_evidence": [
                item["kind"] for item in case["evidence_coverage"].get("items", [])
                if item.get("required") and not item.get("available")
            ],
        }
        issue_id = _new_id("ISS")
        now = _now()
        issue = {
            "id": issue_id,
            "case_id": case_id,
            "case_version": case["version"],
            "status": "open",
            "severity": severity,
            "scope": "sample" if context["sample_id"] else "case",
            "title": _required_text(
                payload.get("title") or _default_issue_title(case), "title", 240
            ),
            "description": _text(payload.get("description"), "description", 2000),
            "context": context,
            "version": 1,
            "created_by": _text(payload.get("actor") or "local-ui", "actor", 160),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO review_issues(
                    id, case_id, case_version, status, severity, scope, title, description,
                    context_json, version, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    issue_id, case_id, case["version"], issue["status"], severity, issue["scope"],
                    issue["title"], issue["description"], _json(context), 1,
                    issue["created_by"], now, now,
                ),
            )
            self._event(connection, case_id, "issue", issue_id, "issue.created", issue["created_by"], issue)
            connection.commit()
        return issue

    def update_issue_status(self, issue_id: str, payload: dict) -> dict:
        issue = self._get_issue(issue_id)
        case = self._expected_case(issue["case_id"], payload.get("expected_case_version"))
        target = str(payload.get("status") or "")
        if target not in ISSUE_STATES:
            raise ReviewWorkflowError(f"unsupported issue status: {target}")
        allowed = {
            "open": {"in_retest", "resolved", "closed"},
            "in_retest": {"resolved", "open"},
            "resolved": {"closed", "open"},
            "closed": set(),
        }
        if target not in allowed[issue["status"]]:
            raise ReviewWorkflowError(f"invalid issue transition: {issue['status']} -> {target}")
        if target in {"resolved", "closed"}:
            reason = _required_text(payload.get("reason"), "reason", 1000)
            if target == "resolved" and not self._has_completed_retest(issue_id):
                raise ReviewWorkflowError("issue resolution requires a completed retest")
        else:
            reason = _text(payload.get("reason"), "reason", 1000)
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE review_issues SET status = ?, version = version + 1, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (target, now, issue_id, issue["version"]),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise ReviewWorkflowConflictError("issue version conflict")
            updated = {**issue, "status": target, "version": issue["version"] + 1, "updated_at": now}
            self._event(
                connection, case["id"], "issue", issue_id, f"issue.{target}",
                _text(payload.get("actor") or "local-ui", "actor", 160),
                {**updated, "reason": reason},
            )
            connection.commit()
        return updated

    def create_regression_set(self, payload: dict) -> dict:
        issue = self._get_issue(_required_text(payload.get("issue_id"), "issue_id", 80))
        case = self._expected_case(issue["case_id"], payload.get("expected_case_version"))
        if issue["status"] == "closed":
            raise ReviewWorkflowError("closed issue cannot create a regression set")
        set_id = _new_id("REG")
        now = _now()
        protocol_ref = case["evidence_selection"].get("split_assignment_id") or ""
        item = {
            "id": set_id,
            "case_id": case["id"],
            "issue_id": issue["id"],
            "name": _required_text(
                payload.get("name") or f"{issue['title']} · 回归集", "name", 240
            ),
            "protocol_ref": protocol_ref,
            "difference_reason": "",
            "created_by": _text(payload.get("actor") or "local-ui", "actor", 160),
            "created_at": now,
            "updated_at": now,
            "members": [],
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO regression_sets(
                    id, case_id, issue_id, name, protocol_ref, difference_reason,
                    created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id, case["id"], issue["id"], item["name"], protocol_ref, "",
                    item["created_by"], now, now,
                ),
            )
            self._event(
                connection, case["id"], "regression_set", set_id,
                "regression_set.created", item["created_by"], item,
            )
            connection.commit()
        return item

    def add_current_evidence_member(self, regression_set_id: str, payload: dict) -> dict:
        regression_set = self._get_regression_set(regression_set_id)
        case = self._expected_case(
            regression_set["case_id"], payload.get("expected_case_version")
        )
        selection = case["evidence_selection"]
        if not selection.get("sample_id") or selection.get("cycle_id") in (None, ""):
            raise ReviewWorkflowError("pin an exact prediction record before adding a regression member")
        radius = _bounded_integer(payload.get("neighbor_radius", 0), "neighbor_radius", 0, 20)
        issue = self._get_issue(regression_set["issue_id"])
        actor = _text(payload.get("actor") or "local-ui", "actor", 160)
        added_at = _now()
        members = self._neighboring_regression_members(
            case,
            issue.get("context") or {},
            radius=radius,
            actor=actor,
            added_at=added_at,
        )
        for member in members:
            member["regression_set_id"] = regression_set_id
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            inserted_members = []
            for member in members:
                inserted = connection.execute(
                    """
                    INSERT OR IGNORE INTO regression_members(
                        regression_set_id, sample_id, dataset, asset_id, cycle_id, added_by, added_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        regression_set_id, member["sample_id"], member["dataset"],
                        member["asset_id"], str(member["cycle_id"]), member["added_by"], member["added_at"],
                    ),
                ).rowcount
                if inserted:
                    inserted_members.append(member)
            if inserted_members:
                event_payload = {
                    "regression_set_id": regression_set_id,
                    "neighbor_radius": radius,
                    "requested_count": len(members),
                    "added_count": len(inserted_members),
                    "members": inserted_members,
                }
                self._event(
                    connection, case["id"], "regression_set", regression_set_id,
                    "regression_set.member_added" if radius == 0 else "regression_set.members_added",
                    actor,
                    inserted_members[0] if radius == 0 else event_payload,
                )
            connection.commit()
        summary = {
            "regression_set_id": regression_set_id,
            "neighbor_radius": radius,
            "requested_count": len(members),
            "added_count": len(inserted_members),
            "existing_count": len(members) - len(inserted_members),
            "members": members,
            "added": bool(inserted_members),
        }
        return {**members[0], **summary} if radius == 0 else summary

    def _neighboring_regression_members(
        self,
        case: dict,
        issue_context: dict,
        *,
        radius: int,
        actor: str,
        added_at: str,
    ) -> list[dict]:
        selection = case["evidence_selection"]
        selected_cycle = _cycle_integer(selection.get("cycle_id"), "cycle_id")
        selected_dataset = str(selection.get("dataset") or "")
        selected_asset = str(selection.get("asset_id") or "")
        records = []
        profile_name = str(case.get("profile_name") or "")
        if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", profile_name):
            profile_dir = self.frontend_data_dir / "profiles" / profile_name
            prediction_payload = _read_json(profile_dir / "prediction_records.json", {})
            prediction_records = (
                prediction_payload.get("records") or []
                if isinstance(prediction_payload, dict) else []
            )
            if prediction_records:
                records = [
                    record for record in prediction_records
                    if _record_matches_issue(
                        record,
                        issue_context,
                        dataset=selected_dataset,
                        asset_id=selected_asset,
                    )
                    and _cycle_within(record.get("cycle_index"), selected_cycle, radius)
                ]
                if not records and radius > 0:
                    raise ReviewWorkflowError(
                        "no neighboring prediction records match the pinned dataset, candidate, artifact and protocol"
                    )
            else:
                scenario_name = "calce" if selected_dataset == "calce_cs2" else selected_dataset
                scenario = _read_json(profile_dir / f"battery_dashboard_{scenario_name}.json", {})
                records = [
                    point for point in (scenario.get("points") or [])
                    if str(point.get("cell_id") or "") == selected_asset
                    and _cycle_within(point.get("cycle_index"), selected_cycle, radius)
                ]

        if not records:
            records = [{
                "sample_id": selection["sample_id"],
                "dataset": selected_dataset,
                "cell_id": selected_asset,
                "cycle_index": selected_cycle,
            }]

        by_sample: dict[str, dict] = {}
        for record in records:
            dataset = str(record.get("dataset") or selected_dataset)
            asset_id = str(record.get("cell_id") or record.get("asset_id") or selected_asset)
            cycle_id = _cycle_integer(record.get("cycle_index", record.get("cycle_id")), "cycle_index")
            sample_id = str(record.get("sample_id") or f"{dataset}:{asset_id}:{cycle_id}")
            by_sample[sample_id] = {
                "regression_set_id": "",
                "sample_id": sample_id,
                "dataset": dataset,
                "asset_id": asset_id,
                "cycle_id": cycle_id,
                "added_by": actor,
                "added_at": added_at,
            }
        members = sorted(by_sample.values(), key=lambda item: (item["dataset"], item["asset_id"], item["cycle_id"]))
        # The caller owns the regression-set id; keep the candidates independent of DB state.
        return members

    def create_retest(self, case_id: str, payload: dict) -> dict:
        case = self._expected_case(case_id, payload.get("expected_case_version"))
        issue = self._get_issue(_required_text(payload.get("issue_id"), "issue_id", 80))
        regression_set = self._get_regression_set(
            _required_text(payload.get("regression_set_id"), "regression_set_id", 80)
        )
        if issue["case_id"] != case_id or regression_set["case_id"] != case_id:
            raise ReviewWorkflowError("issue and regression set must belong to the retest case")
        if regression_set["issue_id"] != issue["id"]:
            raise ReviewWorkflowError("regression set does not belong to the issue")
        if self._has_active_retest(issue["id"]):
            raise ReviewWorkflowConflictError("an active retest already exists for this issue")
        members = self._member_count(regression_set["id"])
        if members < 1:
            raise ReviewWorkflowError("retest requires at least one regression member")
        baseline_protocol = regression_set["protocol_ref"] or case["evidence_selection"].get("split_assignment_id") or ""
        requested_protocol = _text(payload.get("protocol_ref") or baseline_protocol, "protocol_ref", 500)
        if not baseline_protocol or not requested_protocol:
            raise ReviewWorkflowError("retest requires an auditable protocol reference")
        difference_reason = _text(payload.get("difference_reason"), "difference_reason", 1000)
        if requested_protocol != baseline_protocol and not difference_reason:
            raise ReviewWorkflowError("protocol change requires difference_reason")
        retest_id = _new_id("RET")
        now = _now()
        retest = {
            "id": retest_id,
            "case_id": case_id,
            "issue_id": issue["id"],
            "regression_set_id": regression_set["id"],
            "baseline_case_version": case["version"],
            "baseline_protocol_ref": baseline_protocol,
            "requested_protocol_ref": requested_protocol,
            "difference_reason": difference_reason,
            "execution_mode": self._retest_execution_mode(issue),
            "status": "requested",
            "run_id": "",
            "result": {},
            "created_by": _text(payload.get("actor") or "local-ui", "actor", 160),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO retest_requests(
                    id, case_id, issue_id, regression_set_id, baseline_case_version,
                    baseline_protocol_ref, requested_protocol_ref, difference_reason,
                    status, run_id, result_json, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    retest_id, case_id, issue["id"], regression_set["id"], case["version"],
                    baseline_protocol, requested_protocol, difference_reason, "requested", "",
                    _json({"execution_mode": retest["execution_mode"]}),
                    retest["created_by"], now, now,
                ),
            )
            changed = connection.execute(
                """
                UPDATE review_issues SET status = 'in_retest', version = version + 1, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (now, issue["id"], issue["version"]),
            ).rowcount
            if changed != 1:
                # The active-retest check ran outside the lock; a concurrent
                # request already moved the issue on.  Version-guarding the
                # issue update (and rolling back the INSERT) keeps at most
                # one active retest per issue.
                connection.rollback()
                raise ReviewWorkflowConflictError("an active retest already exists for this issue")
            self._event(
                connection, case_id, "retest", retest_id, "retest.requested",
                retest["created_by"], retest,
            )
            connection.commit()
        return retest

    def _retest_execution_mode(self, issue: dict) -> str:
        artifact_id = str((issue.get("context") or {}).get("model_artifact_id") or "")
        if self.has_executable_model_artifact is None:
            return "evidence_replay"
        try:
            return "model_reinference" if self.has_executable_model_artifact(artifact_id) else "evidence_replay"
        except Exception:
            return "evidence_replay"

    def start_retest(self, retest_id: str, payload: dict) -> dict:
        retest = self._get_retest(retest_id)
        self._expected_case(retest["case_id"], payload.get("expected_case_version"))
        return self._start_retest(
            retest_id,
            run_id=_required_text(payload.get("run_id"), "run_id", 240),
            dispatch=payload.get("dispatch") or {},
            actor=_text(payload.get("actor") or "control-plane", "actor", 160),
        )

    def start_retest_from_runner(
        self,
        retest_id: str,
        *,
        run_id: str,
        dispatch: dict,
    ) -> dict:
        """Start an already-authorized retest from its pinned request snapshot."""
        return self._start_retest(
            retest_id,
            run_id=_required_text(run_id, "run_id", 240),
            dispatch=dispatch,
            actor="controlled-runner",
        )

    def _start_retest(
        self,
        retest_id: str,
        *,
        run_id: str,
        dispatch: dict,
        actor: str,
    ) -> dict:
        retest = self._get_retest(retest_id)
        if retest["status"] != "requested":
            raise ReviewWorkflowConflictError(
                f"retest cannot start from status {retest['status']}"
            )
        if not isinstance(dispatch, dict):
            raise ReviewWorkflowError("retest dispatch must be an object")
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE retest_requests SET status = 'running', run_id = ?, result_json = ?, updated_at = ?
                WHERE id = ? AND status = 'requested'
                """,
                (run_id, _json({"dispatch": dispatch}), now, retest_id),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise ReviewWorkflowConflictError("retest start conflict")
            updated = {
                **retest,
                "status": "running",
                "run_id": run_id,
                "result": {"dispatch": dispatch},
                "updated_at": now,
            }
            self._event(
                connection,
                retest["case_id"],
                "retest",
                retest_id,
                "retest.running",
                actor,
                updated,
            )
            connection.commit()
        return updated

    def complete_retest(self, retest_id: str, payload: dict) -> dict:
        """Compatibility hook for an authenticated executor; not a reviewer result form."""
        retest = self._get_retest(retest_id)
        self._expected_case(retest["case_id"], payload.get("expected_case_version"))
        target = str(payload.get("status") or "completed")
        if target not in {"completed", "failed"}:
            raise ReviewWorkflowError("retest completion status must be completed or failed")
        if retest["status"] not in {"requested", "running"}:
            raise ReviewWorkflowConflictError(f"retest is already {retest['status']}")
        run_id = _required_text(payload.get("run_id"), "run_id", 240)
        result = payload.get("result") or {}
        if not isinstance(result, dict):
            raise ReviewWorkflowError("retest result must be an object")
        if target == "completed" and result.get("prediction_recomputed") is not True:
            raise ReviewWorkflowError("completed retest requires recomputed model predictions")
        return self._finish_retest(
            retest_id,
            target=target,
            run_id=run_id,
            result=result,
            actor=_text(payload.get("actor") or "runner", "actor", 160),
        )

    def complete_retest_from_runner(
        self,
        retest_id: str,
        *,
        run_id: str,
        status: str,
        result: dict,
    ) -> dict:
        """Idempotently attach a control-plane run result, even if the case advanced."""
        if status not in {"completed", "failed"}:
            raise ReviewWorkflowError("runner retest status must be completed or failed")
        return self._finish_retest(
            retest_id,
            target=status,
            run_id=_required_text(run_id, "run_id", 240),
            result=result,
            actor="controlled-runner",
        )

    def retest_execution_context(self, retest_id: str) -> dict:
        """Return the immutable review links needed by the controlled runner."""
        retest = self._get_retest(retest_id)
        issue = self._get_issue(retest["issue_id"])
        regression_set = self._get_regression_set(retest["regression_set_id"])
        if issue["case_id"] != retest["case_id"] or regression_set["case_id"] != retest["case_id"]:
            raise ReviewWorkflowError("retest workflow links are inconsistent")
        members = self._rows(
            """
            SELECT * FROM regression_members
            WHERE regression_set_id = ? ORDER BY added_at, sample_id
            """,
            (regression_set["id"],),
        )
        return {
            "case": self.get_case(retest["case_id"]),
            "issue": issue,
            "regression_set": {**regression_set, "members": members},
            "retest": retest,
        }

    def _finish_retest(
        self,
        retest_id: str,
        *,
        target: str,
        run_id: str,
        result: dict,
        actor: str,
    ) -> dict:
        if not isinstance(result, dict):
            raise ReviewWorkflowError("retest result must be an object")
        with self._lock:
            retest = self._get_retest(retest_id)
            if retest["status"] in {"completed", "failed"}:
                if retest["run_id"] == run_id and retest["status"] == target:
                    return retest
                raise ReviewWorkflowConflictError(f"retest is already {retest['status']}")
            if retest["status"] not in {"requested", "running"}:
                raise ReviewWorkflowConflictError(
                    f"retest cannot finish from status {retest['status']}"
                )
            if retest["run_id"] and retest["run_id"] != run_id:
                raise ReviewWorkflowConflictError("retest run id does not match controlled run")
            now = _now()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                changed = connection.execute(
                    """
                    UPDATE retest_requests SET status = ?, run_id = ?, result_json = ?, updated_at = ?
                    WHERE id = ? AND status IN ('requested', 'running')
                    """,
                    (target, run_id, _json(result), now, retest_id),
                ).rowcount
                if changed != 1:
                    connection.rollback()
                    latest = self._get_retest(retest_id)
                    if latest["run_id"] == run_id and latest["status"] == target:
                        return latest
                    raise ReviewWorkflowConflictError("retest completion conflict")
                updated = {
                    **retest,
                    "status": target,
                    "run_id": run_id,
                    "result": result,
                    "updated_at": now,
                }
                self._event(
                    connection,
                    retest["case_id"],
                    "retest",
                    retest_id,
                    f"retest.{target}",
                    actor,
                    updated,
                )
                connection.commit()
            return updated

    def reconcile(self, case_id: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        case = self._expected_case(case_id, payload.get("expected_case_version"))
        tolerance = float(payload.get("tolerance", 1e-6))
        if tolerance <= 0 or tolerance > 1:
            raise ReviewWorkflowError("tolerance must be in (0, 1]")
        result = self._derive_reconciliation(case, tolerance)
        record_id = _new_id("REC")
        now = _now()
        record = {
            "id": record_id,
            "case_id": case_id,
            "case_version": case["version"],
            "status": result["status"],
            "tolerance": tolerance,
            "canonical": result["canonical"],
            "legacy": result["legacy"],
            "deltas": result["deltas"],
            "reasons": result["reasons"],
            "created_by": _text(payload.get("actor") or "local-ui", "actor", 160),
            "created_at": now,
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO kpi_reconciliations(
                    id, case_id, case_version, status, tolerance, canonical_json,
                    legacy_json, deltas_json, reasons_json, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id, case_id, case["version"], record["status"], tolerance,
                    _json(record["canonical"]), _json(record["legacy"]),
                    _json(record["deltas"]), _json(record["reasons"]),
                    record["created_by"], now,
                ),
            )
            self._event(
                connection, case_id, "reconciliation", record_id,
                f"reconciliation.{record['status']}", record["created_by"], record,
            )
            connection.commit()
        return record

    def create_evidence_pack(self, case_id: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        case = self._expected_case(case_id, payload.get("expected_case_version"))
        workspace = self.workspace(case_id)
        pack_id = _new_id("EPK")
        now = _now()
        artifact_refs = self._evidence_artifact_refs(case)
        manifest = {
            "schema_version": "1.1.0",
            "pack_id": pack_id,
            "case_id": case_id,
            "case_version": case["version"],
            "candidate_id": case["evidence_selection"].get("candidate_id") or "",
            "dataset_snapshot_id": case["evidence_selection"].get("dataset_snapshot_id") or "",
            "split_assignment_id": case["evidence_selection"].get("split_assignment_id") or "",
            "sample_id": case["evidence_selection"].get("sample_id") or "",
            "criterion_set_version": case["verdict"].get("criterion_set_version") or "",
            "issue_ids": [item["id"] for item in workspace["issues"]],
            "regression_set_ids": [item["id"] for item in workspace["regression_sets"]],
            "retest_ids": [item["id"] for item in workspace["retests"]],
            "reconciliation_id": (
                workspace["reconciliations"][0]["id"] if workspace["reconciliations"] else ""
            ),
            "artifact_refs": artifact_refs,
            "signing_readiness": workspace["signing_readiness"],
            "created_at": now,
        }
        manifest_json = _json(manifest)
        digest = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        self._write_cas(digest, manifest_json.encode("utf-8"))
        pack = {
            "id": pack_id,
            "case_id": case_id,
            "case_version": case["version"],
            "state": "draft",
            "manifest": manifest,
            "digest": f"sha256:{digest}",
            "signer": "",
            "signature_reason": "",
            "created_by": _text(payload.get("actor") or "local-ui", "actor", 160),
            "created_at": now,
            "signed_at": "",
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO evidence_packs(
                    id, case_id, case_version, state, manifest_json, digest, signer,
                    signature_reason, created_by, created_at, signed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id, case_id, case["version"], "draft", manifest_json, pack["digest"],
                    "", "", pack["created_by"], now, "",
                ),
            )
            self._event(
                connection, case_id, "evidence_pack", pack_id,
                "evidence_pack.created", pack["created_by"], pack,
            )
            connection.commit()
        return pack

    def _evidence_artifact_refs(self, case: dict) -> list[dict]:
        profile_dir = self.frontend_data_dir / "profiles" / str(case.get("profile_name") or "")
        benchmark_dir = profile_dir if (profile_dir / "run_config.json").is_file() else self.frontend_data_dir
        required = {
            "run_spec": benchmark_dir / "run_config.json",
            "aggregate_metrics": benchmark_dir / "leaderboard.json",
            "prediction_records": profile_dir / "prediction_records.json",
            "metric_contributions": profile_dir / "metric_contributions.json",
            "curve_segment_refs": profile_dir / "curve_segment_refs.json",
        }
        refs = []
        for kind, path in required.items():
            if not path.is_file():
                continue
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            self._write_cas(digest, payload)
            refs.append(
                {
                    "kind": kind,
                    "filename": path.name,
                    "digest": f"sha256:{digest}",
                    "bytes": len(payload),
                }
            )
        selection = case.get("evidence_selection") or {}
        curve_ref = selection.get("curve_ref") or {}
        if curve_ref.get("source_uri"):
            refs.append(
                {
                    "kind": "raw_curve_source",
                    "record_id": curve_ref.get("record_id") or "",
                    "digest": curve_ref["source_uri"],
                    "signals": list(curve_ref.get("signals") or []),
                }
            )
        candidate = next(
            (
                item for item in case.get("candidates") or []
                if item.get("candidate_id") == selection.get("candidate_id")
            ),
            {},
        )
        if candidate.get("model_artifact_id"):
            refs.append(
                {
                    "kind": "model_artifact",
                    "candidate_id": candidate.get("candidate_id") or "",
                    "digest": candidate["model_artifact_id"],
                }
            )
        return refs

    def sign_evidence_pack(self, pack_id: str, payload: dict) -> dict:
        pack = self._get_pack(pack_id)
        if pack["state"] != "draft":
            raise ReviewWorkflowConflictError("EvidencePack is immutable after signing")
        case = self._expected_case(pack["case_id"], payload.get("expected_case_version"))
        if case["version"] != pack["case_version"]:
            raise ReviewWorkflowConflictError(
                "EvidencePack case version is stale; generate a new draft"
            )
        workspace = self.workspace(case["id"])
        readiness = workspace["signing_readiness"]
        if not readiness["can_sign"]:
            raise ReviewWorkflowError(
                "EvidencePack signing is blocked: " + "; ".join(readiness["blockers"])
            )
        signer = _required_text(payload.get("signer"), "signer", 160)
        reason = _required_text(payload.get("reason"), "reason", 1000)
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE evidence_packs
                SET state = 'signed', signer = ?, signature_reason = ?, signed_at = ?
                WHERE id = ? AND state = 'draft'
                """,
                (signer, reason, now, pack_id),
            )
            if cursor.rowcount != 1:
                # The draft check above ran outside the lock: a concurrent
                # signer (double-click, second reviewer) already won.  Without
                # this guard a second "signed" event with a different signer
                # would be appended to an already-signed pack.
                connection.rollback()
                raise ReviewWorkflowConflictError("EvidencePack is immutable after signing")
            updated = {
                **pack,
                "state": "signed",
                "signer": signer,
                "signature_reason": reason,
                "signed_at": now,
            }
            self._event(
                connection, case["id"], "evidence_pack", pack_id,
                "evidence_pack.signed", signer, updated,
            )
            connection.commit()
        return updated

    def signing_readiness(
        self,
        case: dict,
        *,
        issues: list[dict] | None = None,
        retests: list[dict] | None = None,
        reconciliations: list[dict] | None = None,
    ) -> dict:
        issues = issues if issues is not None else self._rows(
            "SELECT * FROM review_issues WHERE case_id = ?", (case["id"],)
        )
        retests = retests if retests is not None else self._rows(
            "SELECT * FROM retest_requests WHERE case_id = ?", (case["id"],)
        )
        reconciliations = reconciliations if reconciliations is not None else self._rows(
            "SELECT * FROM kpi_reconciliations WHERE case_id = ? ORDER BY created_at DESC",
            (case["id"],),
        )
        blockers = []
        if case["source_class"] != "measured":
            blockers.append("only measured cases can be signed")
        if case["execution"]["state"] != "completed":
            blockers.append("execution is not completed")
        if case["evidence_coverage"]["state"] != "complete":
            blockers.append("required evidence is incomplete")
        if case["verdict"]["result"] not in {"pass", "conditional"}:
            blockers.append("verdict is not pass or conditional")
        if case["review"]["state"] != "approved":
            blockers.append("review is not approved")
        if any(item["status"] not in {"resolved", "closed"} for item in issues):
            blockers.append("open review issues remain")
        if any(item["status"] in {"requested", "running"} for item in retests):
            blockers.append("retest is not completed")
        latest_reconciliation = reconciliations[0] if reconciliations else None
        reconciliation_current = bool(
            latest_reconciliation
            and latest_reconciliation["status"] == "match"
            and int(latest_reconciliation["case_version"]) == int(case["version"])
        )
        if not reconciliation_current:
            blockers.append("canonical and legacy KPI reconciliation has not matched the current case version")
        gates = {
            "execution": case["execution"]["state"] == "completed",
            "evidence": case["evidence_coverage"]["state"] == "complete",
            "verdict": case["verdict"]["result"] in {"pass", "conditional"},
            "review": case["review"]["state"] == "approved",
            "issues": not any(item["status"] not in {"resolved", "closed"} for item in issues),
            "reconciliation": reconciliation_current,
        }
        return {"can_sign": not blockers, "gates": gates, "blockers": blockers}

    def _derive_reconciliation(self, case: dict, tolerance: float) -> dict:
        selection = case["evidence_selection"]
        candidate = next(
            (
                item for item in case["candidates"]
                if item["candidate_id"] == selection.get("candidate_id")
            ),
            None,
        )
        canonical = dict(case["verdict"].get("observed") or {})
        dataset = str(selection.get("dataset") or "")
        scenario_path = (
            self.frontend_data_dir / "profiles" / case["profile_name"]
            / f"battery_dashboard_{'calce' if dataset == 'calce_cs2' else dataset}.json"
        )
        scenario = _read_json(scenario_path, {})
        scenario_metrics = scenario.get("metrics") or {}
        legacy = {
            key: value for key, value in {
                "mae": scenario_metrics.get("mae"),
                "picp": scenario_metrics.get("coverage"),
                "monotonicity": scenario_metrics.get("monotonicity_violation_rate"),
            }.items() if isinstance(value, (int, float))
        }
        reasons = []
        model_label = str((scenario.get("config_brief") or {}).get("model") or "")
        if not scenario_path.is_file():
            reasons.append("legacy dashboard scenario is unavailable")
        if not candidate:
            reasons.append("selected candidate is unavailable")
        elif model_label and candidate["model"].lower() not in model_label.lower():
            reasons.append("legacy dashboard model identity does not match selected candidate")
        if not canonical:
            reasons.append("canonical verdict has no observed KPI values")
        common = sorted(set(canonical).intersection(legacy))
        if not common:
            reasons.append("canonical and legacy sources have no comparable KPI fields")
        deltas = {
            key: abs(float(canonical[key]) - float(legacy[key])) for key in common
        }
        status = "unavailable" if reasons else (
            "match" if all(delta <= tolerance for delta in deltas.values()) else "mismatch"
        )
        if status == "mismatch":
            reasons.append("one or more KPI deltas exceed tolerance")
        return {
            "status": status,
            "canonical": {key: canonical[key] for key in common},
            "legacy": {key: legacy[key] for key in common},
            "deltas": deltas,
            "reasons": reasons,
        }

    def _expected_case(self, case_id: str, expected_version) -> dict:
        case = self.get_case(case_id)
        if expected_version is None or int(expected_version) != int(case["version"]):
            raise ReviewWorkflowConflictError(
                f"case version conflict: expected {expected_version}, current {case['version']}"
            )
        return case

    def _get_issue(self, issue_id: str) -> dict:
        return self._row("SELECT * FROM review_issues WHERE id = ?", (issue_id,), "issue")

    def _get_regression_set(self, set_id: str) -> dict:
        return self._row("SELECT * FROM regression_sets WHERE id = ?", (set_id,), "regression set")

    def _get_retest(self, retest_id: str) -> dict:
        return self._row("SELECT * FROM retest_requests WHERE id = ?", (retest_id,), "retest")

    def _get_pack(self, pack_id: str) -> dict:
        return self._row("SELECT * FROM evidence_packs WHERE id = ?", (pack_id,), "EvidencePack")

    def _member_count(self, set_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM regression_members WHERE regression_set_id = ?",
                (set_id,),
            ).fetchone()
        return int(row["count"])

    def _has_completed_retest(self, issue_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM retest_requests WHERE issue_id = ? AND status = 'completed' LIMIT 1",
                (issue_id,),
            ).fetchone()
        return row is not None

    def _has_active_retest(self, issue_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM retest_requests
                WHERE issue_id = ? AND status IN ('requested', 'running') LIMIT 1
                """,
                (issue_id,),
            ).fetchone()
        return row is not None

    def _row(self, sql: str, values: tuple, label: str) -> dict:
        rows = self._rows(sql, values)
        if not rows:
            raise ReviewWorkflowNotFoundError(label)
        return rows[0]

    def _rows(self, sql: str, values: tuple = ()) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(sql, values).fetchall()
        return [_decode_row(dict(row)) for row in rows]

    def _event(
        self,
        connection: sqlite3.Connection,
        case_id: str,
        entity_type: str,
        entity_id: str,
        event_type: str,
        actor: str,
        payload: dict,
    ) -> None:
        payload_json = _json(payload)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        self._write_cas(digest, payload_json.encode("utf-8"))
        connection.execute(
            """
            INSERT INTO workflow_events(
                case_id, entity_type, entity_id, event_type, actor,
                payload_json, payload_digest, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id, entity_type, entity_id, event_type, actor or "local-ui",
                payload_json, f"sha256:{digest}", _now(),
            ),
        )

    def _write_cas(self, digest: str, payload: bytes) -> None:
        target = self.cas_root / "sha256" / digest[:2] / digest
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_retry(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_issues(
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    case_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_review_issues_case ON review_issues(case_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS regression_sets(
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    issue_id TEXT NOT NULL REFERENCES review_issues(id),
                    name TEXT NOT NULL,
                    protocol_ref TEXT NOT NULL,
                    difference_reason TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS regression_members(
                    regression_set_id TEXT NOT NULL REFERENCES regression_sets(id),
                    sample_id TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    cycle_id TEXT NOT NULL,
                    added_by TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY(regression_set_id, sample_id)
                );
                CREATE TABLE IF NOT EXISTS retest_requests(
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    issue_id TEXT NOT NULL REFERENCES review_issues(id),
                    regression_set_id TEXT NOT NULL REFERENCES regression_sets(id),
                    baseline_case_version INTEGER NOT NULL,
                    baseline_protocol_ref TEXT NOT NULL,
                    requested_protocol_ref TEXT NOT NULL,
                    difference_reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kpi_reconciliations(
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    case_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    tolerance REAL NOT NULL,
                    canonical_json TEXT NOT NULL,
                    legacy_json TEXT NOT NULL,
                    deltas_json TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence_packs(
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    case_version INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    signer TEXT NOT NULL,
                    signature_reason TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    signed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workflow_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_workflow_events_case ON workflow_events(case_id, sequence DESC);
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            yield connection
        finally:
            connection.close()


def _decode_row(row: dict) -> dict:
    mapping = {
        "context_json": "context",
        "result_json": "result",
        "canonical_json": "canonical",
        "legacy_json": "legacy",
        "deltas_json": "deltas",
        "reasons_json": "reasons",
        "manifest_json": "manifest",
        "payload_json": "payload",
    }
    for source, target in mapping.items():
        if source in row:
            raw = row.pop(source)
            try:
                row[target] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                row[target] = {} if target not in {"reasons"} else []
    if "cycle_id" in row:
        raw_cycle = row["cycle_id"]
        if isinstance(raw_cycle, str) and raw_cycle.isdigit():
            row["cycle_id"] = int(raw_cycle)
    return row


def _default_issue_title(case: dict) -> str:
    verdict = case["verdict"].get("result") or "pending"
    if verdict == "fail":
        return "判定未通过，需要加入回归闭环"
    if case["evidence_coverage"].get("state") != "complete":
        return "必需证据不完整，阻止发布"
    return "评测结论需要人工复核"


def _required_text(value, field: str, max_length: int) -> str:
    text = _text(value, field, max_length)
    if not text:
        raise ReviewWorkflowError(f"{field} is required")
    return text


def _text(value, field: str, max_length: int) -> str:
    text = str(value or "").strip()
    if len(text) > max_length:
        raise ReviewWorkflowError(f"{field} exceeds {max_length} characters")
    return text


def _new_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8].upper()}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(path: Path, fallback):
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _bounded_integer(value, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value).strip()):
        raise ReviewWorkflowError(f"{field} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ReviewWorkflowError(f"{field} must be between {minimum} and {maximum}")
    return number


def _cycle_integer(value, field: str) -> int:
    if isinstance(value, bool):
        raise ReviewWorkflowError(f"{field} must be an integer cycle")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewWorkflowError(f"{field} must be an integer cycle") from exc
    if not number.is_integer():
        raise ReviewWorkflowError(f"{field} must be an integer cycle")
    return int(number)


def _cycle_within(value, center: int, radius: int) -> bool:
    try:
        return abs(_cycle_integer(value, "cycle_index") - center) <= radius
    except ReviewWorkflowError:
        return False


def _record_matches_issue(
    record: dict,
    issue_context: dict,
    *,
    dataset: str,
    asset_id: str,
) -> bool:
    if str(record.get("dataset") or "") != dataset:
        return False
    if str(record.get("cell_id") or record.get("asset_id") or "") != asset_id:
        return False
    pinned_fields = (
        "candidate_id",
        "model_artifact_id",
        "dataset_snapshot_id",
        "split_assignment_id",
    )
    return all(
        not issue_context.get(field)
        or str(record.get(field) or "") == str(issue_context.get(field) or "")
        for field in pinned_fields
    )


def _prediction_snapshot_ref(frontend_data_dir: Path, profile_name: str, dataset: str) -> str:
    if not profile_name or not dataset:
        return ""
    scenario_name = "calce" if dataset == "calce_cs2" else dataset
    path = frontend_data_dir / "profiles" / profile_name / f"battery_dashboard_{scenario_name}.json"
    if not path.is_file():
        return ""
    return f"prediction-snapshot@sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.01 * (attempt + 1))
