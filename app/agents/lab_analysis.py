from __future__ import annotations
import logging
from app.agents.base import BaseAgent
from app.errors import log_error, CATEGORY_DETECTION, CATEGORY_INIT
from app.models.agent import (
    AgentRequest,
    AgentResponse,
    AgentType,
    Finding,
    FindingSeverity,
)
from app.models.patient import LabResult, PatientContext

logger = logging.getLogger(__name__)


class LabAnalysisAgent(BaseAgent):
    """Evaluates lab results against reference ranges and critical thresholds."""

    agent_type = AgentType.LAB_ANALYSIS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from app.llm.service import LLMService

            self._llm = LLMService()
        except Exception:
            log_error(
                logger,
                CATEGORY_INIT,
                "LabAnalysisAgent.__init__",
                "LLM service unavailable; agent will use rule-based analysis only",
            )
            self._llm = None

    async def _analyze(self, request: AgentRequest) -> AgentResponse:
        labs = [LabResult(**l) for l in request.payload.get("lab_results", [])]
        context = PatientContext(
            **request.payload.get(
                "context", {"age": 40, "sex": "other", "care_setting": "inpatient"}
            )
        )

        if not labs:
            return self._build_response(
                request=request,
                risk_score=0.05,
                findings=[],
                reasoning="No lab results provided.",
            )

        if self._llm and request.payload.get("use_llm", True):
            try:
                from app.llm.prompts import LAB_SYSTEM, build_lab_prompt

                result = await self._llm.complete(
                    system_prompt=LAB_SYSTEM,
                    user_prompt=build_lab_prompt(labs, context),
                    required_keys=["risk_score", "reasoning", "critical_values"],
                )
                findings: list[Finding] = []
                for cv in result.get("critical_values", []):
                    test = cv.get("test", "unknown")
                    direction = cv.get("direction", "abnormal")
                    value = cv.get("value", "?")
                    unit = cv.get("unit", "")
                    significance = cv.get("clinical_significance", "")
                    findings.append(
                        Finding(
                            code=f"CRITICAL_{test.upper().replace(' ', '_')}",
                            description=f"CRITICAL: {test} is {direction} at {value} {unit}. {significance}".strip(),
                            severity=FindingSeverity.CRITICAL,
                            affected_items=[test],
                        )
                    )
                for test_name in result.get("abnormal_values", []):
                    findings.append(
                        Finding(
                            code=f"ABNORMAL_{test_name.upper().replace(' ', '_')}",
                            description=f"{test_name} is outside reference range.",
                            severity=FindingSeverity.MEDIUM,
                            affected_items=[test_name],
                        )
                    )
                pattern = result.get("pattern_concerns")
                if pattern:
                    findings.append(
                        Finding(
                            code="LAB_PATTERN_CONCERN",
                            description=f"Pattern concern: {pattern}",
                            severity=FindingSeverity.HIGH,
                        )
                    )
                return self._build_response(
                    request=request,
                    risk_score=result.get("risk_score", 0.5),
                    findings=findings,
                    reasoning=result.get("reasoning", "LLM lab assessment completed."),
                    confidence=result.get("confidence", 0.9),
                    metadata={"llm_raw": result.data},
                )
            except Exception:
                log_error(
                    logger,
                    CATEGORY_DETECTION,
                    "LabAnalysisAgent._analyze",
                    "LLM lab-analysis failed; falling back to rule-based scoring",
                )

        from app.agents._lab_rules import lab_rules

        return lab_rules(request, labs, self)
