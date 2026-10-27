"""SQLite persistence for processing runs and an append-only audit event log.

MVP audit design, not a formally compliant or tamper-proof system:
* ``audit_events`` rejects UPDATE and DELETE via triggers; changes are new events.
* Each event stores the hash of the previous event (a hash chain), so after-the-fact
  edits made by bypassing the triggers are detectable with ``verify_chain``. Anyone with
  write access to the file can still rewrite the whole chain.
* Payloads hold identifiers, hashes and results, not raw document text or secrets.
"""

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.case import CaseResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    document_hashes TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    outcome TEXT NOT NULL,
    source TEXT NOT NULL,
    evaluation_run_id TEXT,
    duration_ms INTEGER NOT NULL,
    errors TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_case ON processing_runs(case_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_fingerprint ON processing_runs(case_id, input_fingerprint);

CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL,
    run_id TEXT,
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_case ON audit_events(case_id, seq);

CREATE TRIGGER IF NOT EXISTS audit_events_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS processing_runs_no_update BEFORE UPDATE ON processing_runs
BEGIN SELECT RAISE(ABORT, 'processing_runs are immutable; record a new run instead'); END;
CREATE TRIGGER IF NOT EXISTS processing_runs_no_delete BEFORE DELETE ON processing_runs
BEGIN SELECT RAISE(ABORT, 'processing_runs are immutable; record a new run instead'); END;
"""

GENESIS_HASH = "0" * 64


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _event_hash(prev_hash: str, event_id: str, case_id: str, run_id: str | None, event_type: str, created_at: str, payload: str) -> str:
    material = json.dumps([prev_hash, event_id, case_id, run_id, event_type, created_at, payload], separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


class AuditStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._memory_conn or sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._memory_conn is None:
                conn.close()

    # ------------------------------------------------------------------- events

    def append_event(self, case_id: str, event_type: str, payload: dict[str, Any], run_id: str | None = None) -> str:
        event_id = str(uuid.uuid4())
        created_at = _now()
        body = json.dumps(payload, sort_keys=True, default=str)
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT event_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["event_hash"] if row else GENESIS_HASH
            digest = _event_hash(prev, event_id, case_id, run_id, event_type, created_at, body)
            conn.execute(
                "INSERT INTO audit_events (event_id, case_id, run_id, event_type, created_at, payload, prev_hash, event_hash) VALUES (?,?,?,?,?,?,?,?)",
                (event_id, case_id, run_id, event_type, created_at, body, prev, digest),
            )
        return event_id

    def events_for_case(self, case_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM audit_events WHERE case_id = ? ORDER BY seq", (case_id,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def verify_chain(self) -> tuple[bool, int | None]:
        """Return (ok, first_bad_seq)."""
        prev = GENESIS_HASH
        with self._conn() as conn:
            for r in conn.execute("SELECT * FROM audit_events ORDER BY seq"):
                expected = _event_hash(prev, r["event_id"], r["case_id"], r["run_id"], r["event_type"], r["created_at"], r["payload"])
                if r["prev_hash"] != prev or r["event_hash"] != expected:
                    return False, r["seq"]
                prev = r["event_hash"]
        return True, None

    # --------------------------------------------------------------------- runs

    def save_run(self, result: CaseResult) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO processing_runs (run_id, case_id, created_at, input_fingerprint, document_hashes, provider, model,
                   prompt_version, schema_version, policy_version, outcome, source, evaluation_run_id, duration_ms, errors, result_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.run_id,
                    result.case_id,
                    result.processed_at.isoformat(),
                    result.input_fingerprint,
                    json.dumps([d.sha256 for d in result.documents]),
                    result.metadata.provider,
                    result.metadata.model,
                    result.metadata.prompt_version,
                    result.metadata.schema_version,
                    result.metadata.policy_version,
                    result.decision.outcome.value,
                    result.source,
                    result.evaluation_run_id,
                    result.duration_ms,
                    json.dumps(result.metadata.errors),
                    result.model_dump_json(),
                ),
            )

    def latest_run(self, case_id: str) -> CaseResult | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT result_json FROM processing_runs WHERE case_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1", (case_id,)
            ).fetchone()
        return CaseResult.model_validate_json(row["result_json"]) if row else None

    def find_run_by_fingerprint(self, case_id: str, fingerprint: str) -> CaseResult | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT result_json FROM processing_runs WHERE case_id = ? AND input_fingerprint = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (case_id, fingerprint),
            ).fetchone()
        return CaseResult.model_validate_json(row["result_json"]) if row else None

    def runs_for_case(self, case_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT run_id, created_at, outcome, provider, model, prompt_version, schema_version, policy_version, source,
                   evaluation_run_id, duration_ms, errors, input_fingerprint FROM processing_runs WHERE case_id = ? ORDER BY created_at, rowid""",
                (case_id,),
            ).fetchall()
        return [{**dict(r), "errors": json.loads(r["errors"])} for r in rows]

    def list_cases(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT case_id, MAX(created_at) AS last_processed, COUNT(*) AS runs FROM processing_runs
                   GROUP BY case_id ORDER BY last_processed DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
