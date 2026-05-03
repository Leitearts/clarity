from __future__ import annotations
import logging
from app.agents.base import BaseAgent
from app.errors import log_error, CATEGORY_DETECTION, CATEGORY_INIT
from app.models.agent import AgentRequest, AgentResponse, AgentType, Finding, FindingSeverity
from app.models.patient import VitalSign, PatientContext

logger = logging.getLogger(__name__)


class VitalsAgent(BaseAgent):
    """Analyzes vital signs for hemodynamic stability and acute risk."""

    agent_type = AgentType.VITALS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from app.llm.service import LLMService
            self._llm = LLMService()
        except Exception:
            log_error(logger, CATEGORY_INIT, "VitalsAgent.__init__",
                      "LLM service unavailable; agent will use rule-based analysis only")
            self._llm = None

    async def _analyze(self, request: AgentRequest) -> AgentResponse:
        raw_vitals = request.payload.get("vitals_results", [])
        vitals = [VitalSign(**v) for v in raw_vitals]
        context = PatientContext(**request.payload.get("context", {
            "age": 40, "sex": "other", "care_setting": "inpatient"
        }))

        if not vitals:
            return self._build_response(
                request=request, risk_score=0.05, findings=[],
                reasoning="No vital signs provided.",
            )

        if self._llm and request.payload.get("use_llm", True):
            try:
                from app.llm.service import LLMServiceError
                from app.llm.prompts import VITALS_SYSTEM, build_vitals_prompt
                result = await self._llm.complete(
                    system_prompt=VITALS_SYSTEM,
                    user_prompt=build_vitals_prompt(vitals, context),
                    required_keys=["risk_score", "reasoning", "critical_vitals"],
                )
                findings: list[Finding] = []
                for cv in result.get("critical_vitals", []):
                    sign = cv.get("sign", "unknown")
                    direction = cv.get("direction", "abnormal")
                    value = cv.get("value", "?")
                    unit = cv.get("unit", "")
                    significance = cv.get("clinical_significance", "")
                    findings.append(Finding(
                        code=f"CRITICAL_{sign.upper().replace(' ', '_')}",
                        description=f"CRITICAL: {sign} is {direction} at {value} {unit}. {significance}".strip(),
                        severity=FindingSeverity.CRITICAL,
                        affected_items=[sign],
                    ))
                for vital_name in result.get("abnormal_vitals", []):
                    findings.append(Finding(
                        code=f"ABNORMAL_{vital_name.upper().replace(' ', '_')}",
                        description=f"{vital_name} is outside reference range.",
                        severity=FindingSeverity.MEDIUM,
                        affected_items=[vital_name],
                    ))
                trend = result.get("trend_concerns")
                if trend:
                    findings.append(Finding(
                        code="VITALS_TREND_CONCERN",
                        description=f"Trend concern: {trend}",
                        severity=FindingSeverity.HIGH,
                    ))
                hemodynamic_risk = result.get("hemodynamic_risk", "stable")
                if hemodynamic_risk == "unstable":
                    findings.append(Finding(
                        code="HEMODYNAMIC_INSTABILITY",
                        description="Patient appears hemodynamically unstable based on vital signs.",
                        severity=FindingSeverity.CRITICAL,
                    ))
                return self._build_response(
                    request=request,
                    risk_score=result.get("risk_score", 0.5),
                    findings=findings,
                    reasoning=result.get("reasoning", "LLM vital assessment completed."),
                    confidence=result.get("confidence", 0.9),
                    metadata={"llm_raw": result.data},
                )
            except Exception:
                log_error(logger, CATEGORY_DETECTION, "VitalsAgent._analyze",
                          "LLM vital-signs analysis failed; falling back to rule-based scoring")

        from app.agents._vitals_rules import vitals_rules
        return vitals_rules(request, vitals, self)
