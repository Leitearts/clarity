from __future__ import annotations
import asyncio
import gzip
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from app.config import settings
from app.models.patient import PatientCase
from app.models.response import AnalysisResponse, AuditRecord

logger = logging.getLogger(__name__)
MAX_LOG_SIZE_BYTES = 50 * 1024 * 1024


class AuditLogger:
    def __init__(self, log_path: str | None = None):
        self._path = Path(log_path or settings.audit_log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, response: AnalysisResponse, case: PatientCase,
                  had_errors: bool = False, error_details: str | None = None) -> str:
        record = _build_record(response, case, had_errors, error_details)
        async with self._lock:
            try:
                await _rotate_if_needed(self._path)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")
                logger.debug("audit.write analysis_id=%s", record.audit_id)
            except Exception as exc:
                logger.error("audit.write_failed: %s", exc)
        return record.audit_id

    def get(self, analysis_id: str) -> AuditRecord | None:
        for record in self._iter_records():
            if record.analysis_id == analysis_id:
                return record
        return None

    def get_by_case(self, case_id: str) -> list[AuditRecord]:
        return [r for r in self._iter_records() if r.case_id == case_id]

    def get_recent(self, limit: int = 50) -> list[AuditRecord]:
        return list(self._iter_records())[-limit:]

    def get_by_level(self, level: str) -> list[AuditRecord]:
        return [r for r in self._iter_records() if r.risk_level == level]

    def _iter_records(self) -> Iterator[AuditRecord]:
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield AuditRecord.model_validate_json(line)
                except Exception as exc:
                    logger.warning("audit.parse_error: %s", exc)


def _build_record(response: AnalysisResponse, case: PatientCase,
                  had_errors: bool, error_details: str | None) -> AuditRecord:
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
        agent_reasoning={r.agent_type.value: r.reasoning for r in response.agent_responses},
        model_version=response.model_version,
        processing_time_ms=response.processing_time_ms,
        had_errors=had_errors,
        error_details=error_details,
    )


async def _rotate_if_needed(path: Path) -> None:
    if not path.exists() or path.stat().st_size < MAX_LOG_SIZE_BYTES:
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rotated = path.with_suffix(f".{ts}.jsonl")
    shutil.move(str(path), str(rotated))
    asyncio.create_task(_gzip_file(rotated))
    logger.info("audit.rotated %s", rotated.name)


async def _gzip_file(path: Path) -> None:
    gz_path = path.with_suffix(".jsonl.gz")
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    path.unlink()
