from __future__ import annotations
import asyncio
import logging
import time
from typing import Any
from app.agents.diagnosis import DiagnosisAgent
from app.agents.drug_interaction import DrugInteractionAgent
from app.agents.lab_analysis import LabAnalysisAgent
from app.agents.patient_context import PatientContextAgent
from app.audit.explainer import ExplanationBuilder
from app.audit.logger import AuditLogger
from app.config import settings
from app.models.agent import AgentRequest, AgentResponse, AgentType
from app.models.patient import PatientCase
from app.models.response import AnalysisResponse
from app.models.risk import RiskScore
from app.risk.engine import RiskEngine, WEIGHT_PRESETS
from app.safety.guardrails import validate_input, sanitize_output, append_disclaimer

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        preset = settings.risk_weight_preset
        self._risk_engine = (
            RiskEngine(preset=preset) if preset in WEIGHT_PRESETS else RiskEngine()
        )
        self._context_agent = PatientContextAgent(
            timeout_seconds=settings.agent_timeout_seconds
        )
        self._diagnosis_agent = DiagnosisAgent(
            timeout_seconds=settings.agent_timeout_seconds
        )
        self._drug_agent = DrugInteractionAgent(
            timeout_seconds=settings.agent_timeout_seconds
        )
        self._lab_agent = LabAnalysisAgent(
            timeout_seconds=settings.agent_timeout_seconds
        )
        self._explainer = ExplanationBuilder()
        self._audit_logger = AuditLogger()

    async def analyze(self, case: PatientCase) -> AnalysisResponse:
        validate_input(case)
        start = time.monotonic()
        logger.info(
            "orchestrator.start case=%s patient=%s", case.case_id, case.patient_id
        )

        context_req = self._make_request(
            case, AgentType.PATIENT_CONTEXT, {"context": case.context.model_dump()}
        )
        diagnosis_req = self._make_request(
            case,
            AgentType.DIAGNOSIS,
            {"diagnoses": [d.model_dump() for d in case.diagnoses]},
        )
        drug_req = self._make_request(
            case,
            AgentType.DRUG_INTERACTION,
            {
                "medications": [m.model_dump() for m in case.medications],
                "allergies": case.context.allergies,
                "diagnoses": [d.model_dump() for d in case.diagnoses],
            },
        )
        lab_req = self._make_request(
            case,
            AgentType.LAB_ANALYSIS,
            {
                "lab_results": [l.model_dump() for l in case.lab_results],
                "context": case.context.model_dump(),
            },
        )

        results: list[AgentResponse] = await asyncio.gather(
            self._context_agent.run(context_req),
            self._diagnosis_agent.run(diagnosis_req),
            self._drug_agent.run(drug_req),
            self._lab_agent.run(lab_req),
        )
        context_resp, diagnosis_resp, drug_resp, lab_resp = results

        context_multiplier = context_resp.metadata.get("context_multiplier", 1.0)
        risk_score = self._risk_engine.score(
            diagnosis_score=diagnosis_resp.risk_score,
            medication_score=drug_resp.risk_score,
            lab_score=lab_resp.risk_score,
            context_multiplier=context_multiplier,
        )

        recommended_actions = self._recommend_actions(risk_score, results)
        import uuid

        analysis_id = str(uuid.uuid4())

        explanation_payload = self._explainer.build(
            analysis_id=analysis_id,
            risk=risk_score,
            agent_responses=results,
            recommended_actions=recommended_actions,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        response = AnalysisResponse(
            analysis_id=analysis_id,
            case_id=case.case_id,
            patient_id=case.patient_id,
            risk=risk_score,
            agent_responses=results,
            explanation=explanation_payload.narrative,
            explanation_structured=explanation_payload.to_dict(),
            recommended_actions=recommended_actions,
            processing_time_ms=elapsed_ms,
        )

        response = sanitize_output(response)
        response = append_disclaimer(response)

        asyncio.create_task(
            self._audit_logger.log(
                response=response,
                case=case,
                had_errors=any(r.error is not None for r in results),
            )
        )

        logger.info(
            "orchestrator.done case=%s risk=%s score=%.3f ms=%d",
            case.case_id,
            risk_score.level.value,
            risk_score.total_score,
            elapsed_ms,
        )
        return response

    @staticmethod
    def _make_request(
        case: PatientCase, agent_type: AgentType, payload: dict[str, Any]
    ) -> AgentRequest:
        return AgentRequest(
            agent_type=agent_type,
            case_id=case.case_id,
            patient_id=case.patient_id,
            payload=payload,
        )

    @staticmethod
    def _recommend_actions(
        risk: RiskScore, responses: list[AgentResponse]
    ) -> list[str]:
        from app.models.risk import RiskLevel
        from app.models.agent import FindingSeverity

        actions = []
        if risk.level == RiskLevel.CRITICAL:
            actions.append("Immediate clinical review required — CRITICAL risk level.")
        elif risk.level == RiskLevel.HIGH:
            actions.append("Urgent clinical review recommended within 1 hour.")
        elif risk.level == RiskLevel.MEDIUM:
            actions.append("Schedule clinical review within 24 hours.")
        else:
            actions.append("Routine monitoring. No immediate intervention indicated.")
        for resp in responses:
            for finding in resp.findings:
                if finding.severity == FindingSeverity.CRITICAL:
                    actions.append(f"Address critical finding: {finding.description}")
        return list(dict.fromkeys(actions))
