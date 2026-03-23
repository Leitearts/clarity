from __future__ import annotations
from app.agents.base import BaseAgent
from app.models.agent import AgentRequest, AgentResponse, AgentType, Finding, FindingSeverity
from app.models.patient import Diagnosis


class DiagnosisAgent(BaseAgent):
    """Validates ICD-10 diagnoses and identifies high-acuity clinical conditions."""

    agent_type = AgentType.DIAGNOSIS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from app.llm.service import LLMService
            self._llm = LLMService()
        except Exception:
            self._llm = None

    async def _analyze(self, request: AgentRequest) -> AgentResponse:
        raw_diagnoses = request.payload.get("diagnoses", [])
        diagnoses = [Diagnosis(**d) for d in raw_diagnoses]

        if self._llm and request.payload.get("use_llm", True):
            try:
                from app.llm.service import LLMServiceError
                from app.llm.prompts import DIAGNOSIS_SYSTEM, build_diagnosis_prompt
                result = await self._llm.complete(
                    system_prompt=DIAGNOSIS_SYSTEM,
                    user_prompt=build_diagnosis_prompt(diagnoses),
                    required_keys=["risk_score", "reasoning", "confidence"],
                )
                findings: list[Finding] = []
                for code in result.get("high_acuity_conditions", []):
                    findings.append(Finding(
                        code=f"HIGH_ACUITY_{code}",
                        description=f"High-acuity condition: {code}.",
                        severity=FindingSeverity.HIGH,
                        affected_items=[code],
                        evidence="LLM clinical reasoning.",
                    ))
                if result.get("conflict_detected"):
                    findings.append(Finding(
                        code="CONFLICTING_DIAGNOSES",
                        description=result.get("conflict_description", "Conflicting diagnoses detected."),
                        severity=FindingSeverity.MEDIUM,
                    ))
                if result.get("rapid_onset_detected"):
                    findings.append(Finding(
                        code="RAPID_ONSET",
                        description="Rapid onset diagnosis detected.",
                        severity=FindingSeverity.MEDIUM,
                    ))
                return self._build_response(
                    request=request,
                    risk_score=result.get("risk_score", 0.5),
                    findings=findings,
                    reasoning=result.get("reasoning", "LLM assessment completed."),
                    confidence=result.get("confidence", 0.9),
                    metadata={"llm_raw": result.data},
                )
            except Exception:
                pass

        # Rule-based fallback
        from app.agents._diagnosis_rules import diagnose_rules
        return diagnose_rules(request, diagnoses, self)
