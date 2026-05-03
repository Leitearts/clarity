from __future__ import annotations
import asyncio
import logging
import sqlite3
import threading
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Iterator
from app.config import settings
from app.models.patient import PatientCase
from app.models.response import AnalysisResponse, AuditRecord

logger = logging.getLogger(__name__)

# Maximum number of records held in the in-memory cache for fast recent lookups.
_RECENT_CACHE_SIZE = 500


class AuditLogger:
    """Concurrent-safe audit logger backed by SQLite (WAL mode).

    Writes are non-blocking — they are dispatched to a thread-pool via
    ``asyncio.to_thread`` so the event loop is never stalled.  SQLite's
    Write-Ahead Log (WAL) mode guarantees atomic, process-safe appends.

    Reads use indexed queries (O(log n)) instead of full-file scans.
    ``get_recent`` is served entirely from an in-memory deque (O(1)).
    """

    def __init__(self, log_path: str | None = None):
        path = Path(log_path or settings.audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Derive the SQLite database path from the configured log path.
        self._db_path = str(path.with_suffix(".db"))
        self._recent: deque[AuditRecord] = deque(maxlen=_RECENT_CACHE_SIZE)
        self._cache_lock = threading.Lock()
        self._setup_db()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_db(self) -> None:
        """Create the table, indexes, and enable WAL mode; seed the cache."""
        with closing(sqlite3.connect(self._db_path)) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    audit_id    TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    case_id     TEXT NOT NULL,
                    risk_level  TEXT NOT NULL,
                    data        TEXT NOT NULL
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis ON audit_records(analysis_id)"
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_case ON audit_records(case_id)")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_level ON audit_records(risk_level)"
            )
            con.commit()
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Seed the in-memory cache from the most recent SQLite rows."""
        with closing(sqlite3.connect(self._db_path)) as con:
            rows = con.execute(
                "SELECT data FROM audit_records ORDER BY rowid DESC LIMIT ?",
                (_RECENT_CACHE_SIZE,),
            ).fetchall()
        records: list[AuditRecord] = []
        for (data,) in reversed(rows):
            try:
                records.append(AuditRecord.model_validate_json(data))
            except Exception as exc:
                logger.warning("audit.cache_warm_parse_error: %s", exc)
        with self._cache_lock:
            self._recent = deque(records, maxlen=_RECENT_CACHE_SIZE)

    # ── Write (async, non-blocking) ───────────────────────────────────────────

    async def log(
        self,
        response: AnalysisResponse,
        case: PatientCase,
        had_errors: bool = False,
        error_details: str | None = None,
    ) -> str:
        """Persist an audit record without blocking the event loop."""
        record = _build_record(response, case, had_errors, error_details)
        try:
            await asyncio.to_thread(self._sync_write, record)
            logger.debug("audit.write analysis_id=%s", record.audit_id)
        except Exception as exc:
            logger.error("audit.write_failed: %s", exc)
        return record.audit_id

    def _sync_write(self, record: AuditRecord) -> None:
        """Atomic SQLite INSERT (called in a worker thread)."""
        with closing(sqlite3.connect(self._db_path)) as con:
            con.execute(
                "INSERT INTO audit_records "
                "(audit_id, analysis_id, case_id, risk_level, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record.audit_id,
                    record.analysis_id,
                    record.case_id,
                    record.risk_level,
                    record.model_dump_json(),
                ),
            )
            con.commit()
        with self._cache_lock:
            self._recent.append(record)

    # ── Reads (async, non-blocking, indexed) ──────────────────────────────────

    async def get(self, analysis_id: str) -> AuditRecord | None:
        """Return a record by analysis_id; checks the in-memory cache first."""
        with self._cache_lock:
            for record in reversed(list(self._recent)):
                if record.analysis_id == analysis_id:
                    return record
        return await asyncio.to_thread(self._sync_get, analysis_id)

    def _sync_get(self, analysis_id: str) -> AuditRecord | None:
        with closing(sqlite3.connect(self._db_path)) as con:
            row = con.execute(
                "SELECT data FROM audit_records WHERE analysis_id = ? LIMIT 1",
                (analysis_id,),
            ).fetchone()
        if row:
            try:
                return AuditRecord.model_validate_json(row[0])
            except Exception as exc:
                logger.warning("audit.parse_error: %s", exc)
        return None

    async def get_by_case(self, case_id: str) -> list[AuditRecord]:
        return await asyncio.to_thread(self._sync_get_by_case, case_id)

    def _sync_get_by_case(self, case_id: str) -> list[AuditRecord]:
        with closing(sqlite3.connect(self._db_path)) as con:
            rows = con.execute(
                "SELECT data FROM audit_records WHERE case_id = ?",
                (case_id,),
            ).fetchall()
        return _parse_rows(rows)

    async def get_recent(self, limit: int = 50) -> list[AuditRecord]:
        """Return the most recent records from the in-memory cache (no I/O)."""
        with self._cache_lock:
            records = list(self._recent)
        return records[-limit:]

    async def get_by_level(self, level: str) -> list[AuditRecord]:
        return await asyncio.to_thread(self._sync_get_by_level, level)

    def _sync_get_by_level(self, level: str) -> list[AuditRecord]:
        with closing(sqlite3.connect(self._db_path)) as con:
            rows = con.execute(
                "SELECT data FROM audit_records WHERE risk_level = ?",
                (level,),
            ).fetchall()
        return _parse_rows(rows)

    def _iter_records(self) -> Iterator[AuditRecord]:
        """Yield records from the in-memory cache; used by the impact endpoint."""
        with self._cache_lock:
            yield from list(self._recent)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_rows(rows: list[tuple[str, ...]]) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for (data,) in rows:
        try:
            records.append(AuditRecord.model_validate_json(data))
        except Exception as exc:
            logger.warning("audit.parse_error: %s", exc)
    return records


def _build_record(
    response: AnalysisResponse,
    case: PatientCase,
    had_errors: bool,
    error_details: str | None,
) -> AuditRecord:
    return AuditRecord(
        analysis_id=response.analysis_id,
        case_id=case.case_id,
        patient_id=case.patient_id,
        timestamp=response.timestamp,
        input_diagnosis_count=len(case.diagnoses),
        input_medication_count=len(case.medications),
        input_lab_count=len(case.lab_results),
        input_vitals_count=len(case.vitals_results),
        diagnosis_score=response.risk.diagnosis_score,
        medication_score=response.risk.medication_score,
        lab_score=response.risk.lab_score,
        vitals_score=response.risk.vitals_score,
        context_multiplier=response.risk.context_multiplier,
        total_risk_score=response.risk.total_score,
        risk_level=response.risk.level.value,
        agent_reasoning={
            r.agent_type.value: r.reasoning for r in response.agent_responses
        },
        model_version=response.model_version,
        processing_time_ms=response.processing_time_ms,
        had_errors=had_errors,
        error_details=error_details,
    )
