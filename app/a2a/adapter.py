from __future__ import annotations
import logging
from pydantic import ValidationError
from app.a2a.models import A2ARequest, A2AResponse, A2AStatus, AgentContributionSummary
from app.models.patient import PatientCase
from app.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class A2AAdapter:
    def __init__(self, orchestrator: Orchestrator):
        self._orchestrator = orchestrator

    async def invoke(
        self, request: A2ARequest, include_full: bool = False
    ) -> A2AResponse:
        logger.info(
            "a2a.invoke trace_id=%s caller=%s capability=%s",
            request.trace_id,
            request.caller_id,
            request.capability,
        )
        handler = self._capability_router(request.capability)
        if handler is None:
            return A2AResponse(
                trace_id=request.trace_id,
                status=A2AStatus.REJECTED,
                analysis_id="",
                case_id="",
                error_code="UNKNOWN_CAPABILITY",
                error_message=f"Unknown capability: '{request.capability}'.",
            )
        return await handler(request, include_full)

    def _capability_router(self, capability: str):
        return {
            "clinical_risk_analysis": self._handle_full_analysis,
            "drug_interaction_check": self._handle_drug_check,
            "lab_interpretation": self._handle_lab_check,
        }.get(capability)

    async def _handle_full_analysis(
        self, request: A2ARequest, include_full: bool
    ) -> A2AResponse:
        try:
            case = PatientCase(**request.payload)
        except (ValidationError, TypeError) as exc:
            return A2AResponse(
                trace_id=request.trace_id,
                status=A2AStatus.REJECTED,
                analysis_id="",
                case_id=request.payload.get("case_id", ""),
                error_code="INVALID_PAYLOAD",
                error_message=f"Validation failed: {exc}",
            )
        try:
            analysis = await self._orchestrator.analyze(case)
        except Exception as exc:
            logger.exception("a2a.analysis_error trace_id=%s", request.trace_id)
            return A2AResponse(
                trace_id=request.trace_id,
                status=A2AStatus.ERROR,
                analysis_id="",
                case_id=case.case_id,
                error_code="ANALYSIS_FAILED",
                error_message=str(exc),
            )

        had_errors = any(r.error is not None for r in analysis.agent_responses)
        status = A2AStatus.PARTIAL if had_errors else A2AStatus.SUCCESS
        weight_map = {
            "diagnosis": analysis.risk.w_diagnosis,
            "drug_interaction": analysis.risk.w_medication,
            "lab_analysis": analysis.risk.w_lab,
            "vitals": analysis.risk.w_vitals,
            "patient_context": 0.0,
        }
        ws = max(
            analysis.risk.w_diagnosis * analysis.risk.diagnosis_score
            + analysis.risk.w_medication * analysis.risk.medication_score
            + analysis.risk.w_lab * analysis.risk.lab_score
            + analysis.risk.w_vitals * analysis.risk.vitals_score,
            1e-9,
        )
        agent_contributions = [
            AgentContributionSummary(
                agent=r.agent_type.value,
                risk_score=r.risk_score,
                weight=weight_map.get(r.agent_type.value, 0.0),
                contribution_pct=round(
                    (weight_map.get(r.agent_type.value, 0.0) * r.risk_score / ws) * 100,
                    1,
                ),
                top_finding=r.findings[0].description if r.findings else None,
                had_error=r.error is not None,
            )
            for r in analysis.agent_responses
        ]
        formula_trace = (analysis.explanation_structured or {}).get("formula_trace")
        return A2AResponse(
            trace_id=request.trace_id,
            status=status,
            analysis_id=analysis.analysis_id,
            case_id=analysis.case_id,
            risk_score=analysis.risk.total_score,
            risk_level=analysis.risk.level.value,
            narrative=analysis.explanation,
            formula_trace=formula_trace,
            agent_contributions=agent_contributions,
            critical_findings=analysis.critical_findings,
            recommended_actions=analysis.recommended_actions,
            full_result=analysis.model_dump() if include_full else None,
            processing_time_ms=analysis.processing_time_ms,
        )

    async def _handle_drug_check(
        self, request: A2ARequest, include_full: bool
    ) -> A2AResponse:
        from app.agents.drug_interaction import DrugInteractionAgent
        from app.models.agent import AgentRequest, AgentType

        try:
            meds = request.payload.get("medications", [])
            allergies = request.payload.get("allergies", [])
            case_id = request.payload.get("case_id", "drug-check")
            patient_id = request.payload.get("patient_id", "unknown")
        except Exception as exc:
            return A2AResponse(
                trace_id=request.trace_id,
                status=A2AStatus.REJECTED,
                analysis_id="",
                case_id="",
                error_code="INVALID_PAYLOAD",
                error_message=str(exc),
            )
        agent = DrugInteractionAgent()
        agent_request = AgentRequest(
            agent_type=AgentType.DRUG_INTERACTION,
            case_id=case_id,
            patient_id=patient_id,
            payload={"medications": meds, "allergies": allergies, "diagnoses": []},
        )
        try:
            result = await agent.run(agent_request)
        except Exception as exc:
            return A2AResponse(
                trace_id=request.trace_id,
                status=A2AStatus.ERROR,
                analysis_id="",
                case_id=case_id,
                error_code="AGENT_FAILED",
                error_message=str(exc),
            )
        return A2AResponse(
            trace_id=request.trace_id,
            status=A2AStatus.SUCCESS,
            analysis_id=f"drug-{request.trace_id}",
            case_id=case_id,
            risk_score=result.risk_score,
            risk_level=_score_to_level(result.risk_score),
            narrative=result.reasoning,
            critical_findings=[
                f.description
                for f in result.findings
                if f.severity.value in ("critical", "high")
            ],
            full_result=result.model_dump() if include_full else None,
        )

    async def _handle_lab_check(
        self, request: A2ARequest, include_full: bool
    ) -> A2AResponse:
        from app.agents.lab_analysis import LabAnalysisAgent
        from app.models.agent import AgentRequest, AgentType
        from app.models.patient import PatientContext

        try:
            labs = request.payload.get("lab_results", [])
            age = request.payload.get("patient_age", 40)
            sex = request.payload.get("patient_sex", "other")
            case_id = request.payload.get("case_id", "lab-check")
            patient_id = request.payload.get("patient_id", "unknown")
            context = PatientContext(age=age, sex=sex)
        except Exception as exc:
            return A2AResponse(
                trace_id=request.trace_id,
                status=A2AStatus.REJECTED,
                analysis_id="",
                case_id="",
                error_code="INVALID_PAYLOAD",
                error_message=str(exc),
            )
        agent = LabAnalysisAgent()
        agent_request = AgentRequest(
            agent_type=AgentType.LAB_ANALYSIS,
            case_id=case_id,
            patient_id=patient_id,
            payload={"lab_results": labs, "context": context.model_dump()},
        )
        try:
            result = await agent.run(agent_request)
        except Exception as exc:
            return A2AResponse(
                trace_id=request.trace_id,
                status=A2AStatus.ERROR,
                analysis_id="",
                case_id=case_id,
                error_code="AGENT_FAILED",
                error_message=str(exc),
            )
        return A2AResponse(
            trace_id=request.trace_id,
            status=A2AStatus.SUCCESS,
            analysis_id=f"lab-{request.trace_id}",
            case_id=case_id,
            risk_score=result.risk_score,
            risk_level=_score_to_level(result.risk_score),
            narrative=result.reasoning,
            critical_findings=[
                f.description
                for f in result.findings
                if f.severity.value in ("critical", "high")
            ],
            full_result=result.model_dump() if include_full else None,
        )


def _score_to_level(score: float) -> str:
    from app.config import settings

    if score >= settings.threshold_critical:
        return "critical"
    if score >= settings.threshold_high:
        return "high"
    if score >= settings.threshold_low:
        return "medium"
    return "low"
