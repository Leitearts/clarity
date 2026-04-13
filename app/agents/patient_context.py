from __future__ import annotations
from app.agents.base import BaseAgent
from app.models.agent import (
    AgentRequest,
    AgentResponse,
    AgentType,
    Finding,
    FindingSeverity,
)
from app.models.patient import PatientContext


class PatientContextAgent(BaseAgent):
    """Evaluates patient demographics and comorbidities to produce a context multiplier."""

    agent_type = AgentType.PATIENT_CONTEXT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from app.llm.service import LLMService

            self._llm = LLMService()
        except Exception:
            self._llm = None

    async def _analyze(self, request: AgentRequest) -> AgentResponse:
        context = PatientContext(**request.payload.get("context", {}))

        if self._llm and request.payload.get("use_llm", True):
            try:
                from app.llm.prompts import CONTEXT_SYSTEM, build_context_prompt

                result = await self._llm.complete(
                    system_prompt=CONTEXT_SYSTEM,
                    user_prompt=build_context_prompt(context),
                    required_keys=["context_multiplier", "reasoning"],
                )
                multiplier = float(result.get("context_multiplier", 1.0))
                multiplier = round(min(1.5, max(0.5, multiplier)), 2)
                findings: list[Finding] = []
                if result.get("age_risk_flag"):
                    findings.append(
                        Finding(
                            code="AGE_RISK",
                            description=f"Age {context.age} flagged as elevated risk.",
                            severity=FindingSeverity.LOW,
                        )
                    )
                burden = result.get("comorbidity_burden", "none")
                if burden in ("moderate", "high"):
                    severity = (
                        FindingSeverity.HIGH
                        if burden == "high"
                        else FindingSeverity.MEDIUM
                    )
                    high_risk = result.get("high_risk_comorbidities", [])
                    findings.append(
                        Finding(
                            code=f"COMORBIDITY_{burden.upper()}",
                            description=(
                                f"{burden.capitalize()} comorbidity burden. "
                                + (
                                    f"High-risk: {', '.join(high_risk)}."
                                    if high_risk
                                    else ""
                                )
                            ),
                            severity=severity,
                            affected_items=high_risk,
                        )
                    )
                acuity = result.get("care_setting_acuity", "routine")
                if acuity in ("elevated", "high"):
                    findings.append(
                        Finding(
                            code=f"SETTING_ACUITY_{acuity.upper()}",
                            description=f"Care setting acuity: {acuity} ({context.care_setting}).",
                            severity=FindingSeverity.MEDIUM,
                        )
                    )
                return self._build_response(
                    request=request,
                    risk_score=0.0,
                    findings=findings,
                    reasoning=result.get("reasoning", "Context assessment completed."),
                    confidence=result.get("confidence", 0.9),
                    metadata={"context_multiplier": multiplier, "llm_raw": result.data},
                )
            except Exception:
                pass

        from app.agents._context_rules import context_rules

        return context_rules(request, context, self)
