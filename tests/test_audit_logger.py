"""Unit tests for the SQLite-backed AuditLogger.

Covers:
- Atomic writes (no corruption on repeated inserts)
- Concurrent writes from multiple asyncio tasks
- Indexed reads (get, get_by_case, get_by_level)
- In-memory cache correctness (get_recent served without I/O)
- Record isolation between AuditLogger instances sharing the same DB
"""
from __future__ import annotations

import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.audit.logger import AuditLogger
from app.models.response import AuditRecord


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_record(
    *,
    analysis_id: str | None = None,
    case_id: str = "CASE-TEST",
    risk_level: str = "medium",
) -> AuditRecord:
    return AuditRecord(
        audit_id=str(uuid.uuid4()),
        analysis_id=analysis_id or str(uuid.uuid4()),
        case_id=case_id,
        patient_id="P-TEST",
        timestamp=datetime.now(timezone.utc),
        input_diagnosis_count=1,
        input_medication_count=1,
        input_lab_count=1,
        input_vitals_count=0,
        diagnosis_score=0.5,
        medication_score=0.5,
        lab_score=0.5,
        vitals_score=0.0,
        context_multiplier=1.0,
        total_risk_score=0.5,
        risk_level=risk_level,
        agent_reasoning={"diagnosis": "mock"},
        model_version="clarity-v1.0",
        processing_time_ms=100,
        had_errors=False,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestAuditLoggerWrites:

    def test_sync_write_persists_to_sqlite(self, tmp_path: Path):
        """A record written via _sync_write can be retrieved from SQLite."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        rec = _make_record()
        al._sync_write(rec)

        retrieved = al._sync_get(rec.analysis_id)
        assert retrieved is not None
        assert retrieved.analysis_id == rec.analysis_id
        assert retrieved.risk_level == rec.risk_level

    @pytest.mark.asyncio
    async def test_async_log_persists_record(self, tmp_path: Path):
        """log() dispatches a write via asyncio.to_thread and updates the cache."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from app.models.agent import AgentResponse, AgentType
        from app.models.risk import RiskScore
        from app.models.response import AnalysisResponse
        from app.models.patient import PatientCase

        # Build the minimal objects that _build_record requires.
        risk = RiskScore(
            diagnosis_score=0.5, medication_score=0.5,
            lab_score=0.5, vitals_score=0.5,
        )
        response = MagicMock(spec=AnalysisResponse)
        response.analysis_id = str(uuid.uuid4())
        response.timestamp = datetime.now(timezone.utc)
        response.risk = risk
        response.agent_responses = []
        response.model_version = "clarity-v1.0"
        response.processing_time_ms = 50

        case = MagicMock(spec=PatientCase)
        case.case_id = "CASE-ASYNC"
        case.patient_id = "P-ASYNC"
        case.diagnoses = []
        case.medications = []
        case.lab_results = []
        case.vitals_results = []

        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        audit_id = await al.log(response=response, case=case)

        assert audit_id  # non-empty string returned
        result = await al.get(response.analysis_id)
        assert result is not None
        assert result.analysis_id == response.analysis_id

    @pytest.mark.asyncio
    async def test_concurrent_writes_no_corruption(self, tmp_path: Path):
        """20 concurrent writes must all be persisted and parseable."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        n = 20
        records = [_make_record() for _ in range(n)]

        await asyncio.gather(
            *[asyncio.to_thread(al._sync_write, r) for r in records]
        )

        written_ids = {r.analysis_id for r in records}
        recent = await al.get_recent(limit=n + 5)
        retrieved_ids = {r.analysis_id for r in recent}
        assert written_ids == retrieved_ids, (
            f"Missing records after concurrent writes: "
            f"{written_ids - retrieved_ids}"
        )

    def test_duplicate_audit_id_does_not_crash(self, tmp_path: Path):
        """Inserting a record with the same audit_id should be rejected by PRIMARY KEY."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        rec = _make_record()
        al._sync_write(rec)

        # Second insert with same audit_id must not corrupt DB.
        con = sqlite3.connect(al._db_path)
        try:
            con.execute(
                "INSERT INTO audit_records (audit_id, analysis_id, case_id, risk_level, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (rec.audit_id, rec.analysis_id, rec.case_id, rec.risk_level, rec.model_dump_json()),
            )
            con.commit()
        except sqlite3.IntegrityError:
            pass  # expected
        finally:
            con.close()

        rows = al._sync_get_by_case(rec.case_id)
        assert len(rows) == 1, "Duplicate insert should be rejected, not duplicated"


class TestAuditLoggerReads:

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown_id(self, tmp_path: Path):
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        result = await al.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_case_returns_matching_records(self, tmp_path: Path):
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        case_a = [_make_record(case_id="CASE-A") for _ in range(3)]
        case_b = [_make_record(case_id="CASE-B") for _ in range(2)]
        for r in case_a + case_b:
            al._sync_write(r)

        results = await al.get_by_case("CASE-A")
        assert len(results) == 3
        assert all(r.case_id == "CASE-A" for r in results)

    @pytest.mark.asyncio
    async def test_get_by_level_returns_matching_records(self, tmp_path: Path):
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        criticals = [_make_record(risk_level="critical") for _ in range(4)]
        lows = [_make_record(risk_level="low") for _ in range(2)]
        for r in criticals + lows:
            al._sync_write(r)

        results = await al.get_by_level("critical")
        assert len(results) == 4
        assert all(r.risk_level == "critical" for r in results)

    @pytest.mark.asyncio
    async def test_get_recent_served_from_cache(self, tmp_path: Path):
        """get_recent must return records from the in-memory deque (no DB I/O)."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        records = [_make_record() for _ in range(5)]
        for r in records:
            al._sync_write(r)

        recent = await al.get_recent(limit=10)
        assert len(recent) == 5
        inserted_ids = {r.analysis_id for r in records}
        returned_ids = {r.analysis_id for r in recent}
        assert inserted_ids == returned_ids

    @pytest.mark.asyncio
    async def test_get_recent_respects_limit(self, tmp_path: Path):
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        for _ in range(10):
            al._sync_write(_make_record())

        recent = await al.get_recent(limit=3)
        assert len(recent) == 3

    def test_iter_records_yields_cached_records(self, tmp_path: Path):
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        records = [_make_record() for _ in range(4)]
        for r in records:
            al._sync_write(r)

        yielded = list(al._iter_records())
        assert len(yielded) == 4


class TestAuditLoggerCacheWarm:

    @pytest.mark.asyncio
    async def test_new_instance_warms_cache_from_existing_db(self, tmp_path: Path):
        """A second AuditLogger on the same DB must reload existing records."""
        path = str(tmp_path / "audit.jsonl")
        al1 = AuditLogger(log_path=path)
        records = [_make_record() for _ in range(5)]
        for r in records:
            al1._sync_write(r)

        # Fresh instance — must reload from SQLite
        al2 = AuditLogger(log_path=path)
        recent = await al2.get_recent(limit=10)
        assert len(recent) == 5
