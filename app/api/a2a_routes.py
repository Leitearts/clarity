from __future__ import annotations
import time
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from app.a2a.models import A2ARequest, A2AResponse, AgentCard
from app.api.auth import require_api_key
from app.api.limiter import limiter
from app.config import settings

router = APIRouter(tags=["a2a"])

_AGENT_CARD = AgentCard()


@router.get(
    "/agent-card",
    response_model=AgentCard,
    summary="CLARITY agent discovery card",
)
async def get_agent_card() -> AgentCard:
    """
    Returns the CLARITY AgentCard.
    No authentication required — public discovery endpoint.
    Prompt Opinion and other orchestrators poll this to learn how to invoke CLARITY.
    """
    return _AGENT_CARD


@router.post(
    "/a2a/invoke",
    response_model=A2AResponse,
    summary="Invoke CLARITY via A2A protocol",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(lambda: settings.rate_limit_a2a)
async def a2a_invoke(
    request_body: A2ARequest,
    request: Request,
    include_full: bool = Query(
        default=False,
        description="Include complete AnalysisResponse in full_result field",
    ),
) -> A2AResponse:
    """
    Standard A2A invocation endpoint.
    Accepts an A2ARequest envelope with a PatientCase payload.
    Returns an A2AResponse with risk score, findings, and explanation.
    Trace IDs are passed through for distributed tracing.
    """
    adapter = request.app.state.a2a_adapter
    start = time.monotonic()
    response = await adapter.invoke(request_body, include_full=include_full)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    if response.processing_time_ms is None:
        response.processing_time_ms = elapsed_ms
    return response


@router.get("/a2a/schema", summary="JSON Schema for A2A payload")
async def get_schema():
    """Returns the JSON Schema for PatientCase — the expected payload structure."""
    from app.models.patient import PatientCase

    return JSONResponse(content=PatientCase.model_json_schema())


@router.get("/a2a/capabilities", summary="List supported A2A capabilities")
async def get_capabilities():
    return {
        "agent_id": _AGENT_CARD.agent_id,
        "capabilities": [c.model_dump() for c in _AGENT_CARD.capabilities],
        "schema_version": _AGENT_CARD.schema_version,
    }


@router.get("/a2a/health", summary="Detailed health check with agent status")
async def a2a_health():
    from app.config import settings

    return {
        "status": "ok",
        "agent_id": _AGENT_CARD.agent_id,
        "version": _AGENT_CARD.version,
        "llm_available": bool(settings.anthropic_api_key),
        "llm_model": settings.llm_model if settings.anthropic_api_key else None,
        "agents": {
            "diagnosis": {"status": "ok", "fallback_available": True},
            "drug_interaction": {"status": "ok", "fallback_available": True},
            "lab_analysis": {"status": "ok", "fallback_available": True},
            "patient_context": {"status": "ok", "fallback_available": True},
        },
        "risk_presets": list(
            __import__(
                "app.risk.engine", fromlist=["WEIGHT_PRESETS"]
            ).WEIGHT_PRESETS.keys()
        ),
    }
