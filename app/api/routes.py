from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request, Query
from app.models.patient import PatientCase
from app.models.response import AnalysisResponse, AuditRecord

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze a patient case for clinical risk",
    tags=["analysis"],
)
async def analyze(case: PatientCase, request: Request) -> AnalysisResponse:
    """
    Submit a patient case for multi-agent risk analysis.
    Returns a structured risk score with findings and explanation.
    All data must be synthetic or de-identified — no PHI.
    """
    orchestrator = request.app.state.orchestrator
    try:
        return await orchestrator.analyze(case)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@router.get("/health", tags=["system"], summary="Service health check")
async def health():
    return {"status": "ok", "service": "CLARITY", "version": "1.0"}


@router.get(
    "/audit/{analysis_id}",
    response_model=AuditRecord,
    summary="Retrieve audit record by analysis ID",
    tags=["audit"],
)
async def get_audit(analysis_id: str, request: Request):
    record = request.app.state.audit_logger.get(analysis_id)
    if not record:
        raise HTTPException(
            status_code=404, detail=f"Audit record not found: {analysis_id}"
        )
    return record


@router.get(
    "/audit/case/{case_id}",
    response_model=list[AuditRecord],
    summary="All audit records for a case",
    tags=["audit"],
)
async def get_case_audit(case_id: str, request: Request):
    return request.app.state.audit_logger.get_by_case(case_id)


@router.get(
    "/audit",
    response_model=list[AuditRecord],
    summary="List recent audit records",
    tags=["audit"],
)
async def list_audit(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
    level: str | None = Query(default=None, pattern="^(low|medium|high|critical)$"),
):
    logger = request.app.state.audit_logger
    if level:
        return logger.get_by_level(level)[-limit:]
    return logger.get_recent(limit)


@router.get(
    "/reasoning/{analysis_id}",
    summary="Full LLM reasoning trace for an analysis",
    tags=["explainability"],
)
async def get_reasoning(analysis_id: str, request: Request):
    record = request.app.state.audit_logger.get(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {
        "analysis_id": analysis_id,
        "agent_reasoning": record.agent_reasoning,
        "formula_trace": (
            f"R = ({record.diagnosis_score:.4f}×w1 + "
            f"{record.medication_score:.4f}×w2 + "
            f"{record.lab_score:.4f}×w3) × "
            f"{record.context_multiplier:.2f} = {record.total_risk_score:.4f}"
        ),
        "risk_level": record.risk_level,
        "model_version": record.model_version,
        "note": "Every reasoning string is preserved verbatim from agent output.",
    }
