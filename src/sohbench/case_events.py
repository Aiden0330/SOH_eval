from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class CaseEventStoreError(RuntimeError):
    pass


class CaseEventNotFoundError(KeyError):
    pass


class CaseEventConflictError(CaseEventStoreError):
    def __init__(self, expected_version: int | None, current_version: int):
        self.expected_version = expected_version
        self.current_version = current_version
        super().__init__(
            f"case version conflict: expected {expected_version}, current {current_version}"
        )


class CaseEventStore:
    """SQLite/WAL source of truth with append-only case events and CAS snapshots."""

    def __init__(self, db_path: str | Path, cas_root: str | Path):
        self.db_path = Path(db_path)
        self.cas_root = Path(cas_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cas_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def create(self, document: dict, *, actor: str, reason: str) -> dict:
        snapshot, digest = self._snapshot(document)
        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO evaluation_cases(
                        id, profile_name, idempotency_key, lifecycle_state, version,
                        snapshot_json, snapshot_digest, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document["id"], document["profile_name"], document["idempotency_key"],
                        document["lifecycle_state"], document["version"], snapshot, digest,
                        document["created_at"], document["updated_at"],
                    ),
                )
                self._insert_event(
                    connection, document, "case.created", actor, reason, snapshot, digest
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise CaseEventStoreError(str(exc)) from exc
        return json.loads(snapshot)

    def import_legacy(self, document: dict, *, source: str) -> dict:
        existing = self.find_by_idempotency_key(document["idempotency_key"])
        if existing:
            return existing
        snapshot, digest = self._snapshot(document)
        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                present = connection.execute(
                    "SELECT snapshot_json FROM evaluation_cases WHERE id = ?",
                    (document["id"],),
                ).fetchone()
                if present:
                    connection.rollback()
                    return json.loads(present["snapshot_json"])
                connection.execute(
                    """
                    INSERT INTO evaluation_cases(
                        id, profile_name, idempotency_key, lifecycle_state, version,
                        snapshot_json, snapshot_digest, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document["id"], document["profile_name"], document["idempotency_key"],
                        document["lifecycle_state"], document["version"], snapshot, digest,
                        document["created_at"], document["updated_at"],
                    ),
                )
                self._insert_event(
                    connection, document, "case.legacy_migrated", "migration", source,
                    snapshot, digest,
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise CaseEventStoreError(str(exc)) from exc
        return json.loads(snapshot)

    def commit(
        self,
        document: dict,
        *,
        expected_version: int | None,
        event_type: str,
        actor: str,
        reason: str,
    ) -> dict:
        if expected_version is None:
            raise CaseEventConflictError(None, self.get(document["id"])["version"])
        snapshot, digest = self._snapshot(document)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT version, snapshot_digest FROM evaluation_cases WHERE id = ?",
                (document["id"],),
            ).fetchone()
            if current is None:
                connection.rollback()
                raise CaseEventNotFoundError(document["id"])
            if int(current["version"]) != int(expected_version):
                connection.rollback()
                raise CaseEventConflictError(expected_version, int(current["version"]))
            if current["snapshot_digest"] == digest:
                connection.rollback()
                return self.get(document["id"])
            changed = connection.execute(
                """
                UPDATE evaluation_cases
                   SET profile_name = ?, idempotency_key = ?, lifecycle_state = ?, version = ?,
                       snapshot_json = ?, snapshot_digest = ?, updated_at = ?
                 WHERE id = ? AND version = ?
                """,
                (
                    document["profile_name"], document["idempotency_key"],
                    document["lifecycle_state"], document["version"], snapshot, digest,
                    document["updated_at"], document["id"], expected_version,
                ),
            ).rowcount
            if changed != 1:
                latest = connection.execute(
                    "SELECT version FROM evaluation_cases WHERE id = ?", (document["id"],)
                ).fetchone()
                connection.rollback()
                raise CaseEventConflictError(
                    expected_version, int(latest["version"]) if latest else -1
                )
            self._insert_event(
                connection, document, event_type, actor, reason, snapshot, digest
            )
            connection.commit()
        return json.loads(snapshot)

    def get(self, case_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM evaluation_cases WHERE id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise CaseEventNotFoundError(case_id)
        return json.loads(row["snapshot_json"])

    def find_by_idempotency_key(self, key: str) -> dict | None:
        if not key:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM evaluation_cases WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return json.loads(row["snapshot_json"]) if row else None

    def list(
        self,
        *,
        profile_name: str | None = None,
        lifecycle_state: str | None = None,
        include_archived: bool = True,
        limit: int | None = None,
    ) -> list[dict]:
        clauses = []
        values: list[object] = []
        if profile_name is not None:
            clauses.append("profile_name = ?")
            values.append(profile_name)
        if lifecycle_state is not None:
            clauses.append("lifecycle_state = ?")
            values.append(lifecycle_state)
        elif not include_archived:
            clauses.append("lifecycle_state <> ?")
            values.append("archived")
        sql = "SELECT snapshot_json FROM evaluation_cases"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            values.append(max(0, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(sql, values).fetchall()
        return [json.loads(row["snapshot_json"]) for row in rows]

    def list_events(self, case_id: str, *, after_sequence: int = 0) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, case_id, case_version, event_type, actor, reason,
                       snapshot_digest, created_at
                  FROM case_events
                 WHERE case_id = ? AND sequence > ?
                 ORDER BY sequence
                """,
                (case_id, max(0, int(after_sequence))),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_migration_error(self, source: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO case_migration_errors(source, error, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET error = excluded.error, created_at = excluded.created_at
                """,
                (source, error, _now()),
            )

    def list_migration_errors(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source, error, created_at FROM case_migration_errors ORDER BY source"
            ).fetchall()
        return [dict(row) for row in rows]

    def _snapshot(self, document: dict) -> tuple[str, str]:
        snapshot = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        payload = snapshot.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        target = self.cas_root / "sha256" / digest[:2] / digest
        if not target.exists():
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
        return snapshot, digest

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        document: dict,
        event_type: str,
        actor: str,
        reason: str,
        snapshot: str,
        digest: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO case_events(
                case_id, case_version, event_type, actor, reason, payload_json,
                snapshot_digest, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document["id"], document["version"], event_type, actor or "local-ui",
                reason or "", snapshot, digest, document["updated_at"],
            ),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_cases(
                    id TEXT PRIMARY KEY,
                    profile_name TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    lifecycle_state TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cases_profile_updated
                    ON evaluation_cases(profile_name, updated_at DESC);
                CREATE TABLE IF NOT EXISTS case_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES evaluation_cases(id),
                    case_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(case_id, case_version)
                );
                CREATE TABLE IF NOT EXISTS case_migration_errors(
                    source TEXT PRIMARY KEY,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
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


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.01 * (attempt + 1))


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
