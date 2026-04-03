from __future__ import annotations
from app.agents.base import BaseAgent
from app.models.agent import (
    AgentRequest,
    AgentResponse,
    AgentType,
    Finding,
    FindingSeverity,
)
from app.models.patient import Medication, Diagnosis

_SEVERITY_MAP = {
    "critical": FindingSeverity.CRITICAL,
    "high": FindingSeverity.HIGH,
    "medium": FindingSeverity.MEDIUM,
    "low": FindingSeverity.LOW,
}


class DrugInteractionAgent(BaseAgent):
    """Screens medications for DDIs, allergy conflicts, and polypharmacy."""

    agent_type = AgentType.DRUG_INTERACTION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from app.llm.service import LLMService

            self._llm = LLMService()
        except Exception:
            self._llm = None

    async def _analyze(self, request: AgentRequest) -> AgentResponse:
        medications = [Medication(**m) for m in request.payload.get("medications", [])]
        allergies = request.payload.get("allergies", [])
        diagnoses = [Diagnosis(**d) for d in request.payload.get("diagnoses", [])]

        if not medications:
            return self._build_response(
                request=request,
                risk_score=0.05,
                findings=[],
                reasoning="No medications listed. Drug interaction risk is minimal.",
            )

        if self._llm and request.payload.get("use_llm", True):
            try:
                from app.llm.prompts import DRUG_SYSTEM, build_drug_prompt

                result = await self._llm.complete(
                    system_prompt=DRUG_SYSTEM,
                    user_prompt=build_drug_prompt(medications, allergies, diagnoses),
                    required_keys=["risk_score", "reasoning", "interactions"],
                )
                findings: list[Finding] = []
                for interaction in result.get("interactions", []):
                    drugs = interaction.get("drugs", [])
                    severity = _SEVERITY_MAP.get(
                        interaction.get("severity", "medium").lower(),
                        FindingSeverity.MEDIUM,
                    )
                    findings.append(
                        Finding(
                            code="DDI_" + "_".join(d.upper() for d in drugs[:2]),
                            description=interaction.get(
                                "description", f"Interaction: {', '.join(drugs)}"
                            ),
                            severity=severity,
                            affected_items=drugs,
                            evidence="LLM pharmacology reasoning.",
                        )
                    )
                for drug in result.get("allergy_conflicts", []):
                    findings.append(
                        Finding(
                            code=f"ALLERGY_{drug.upper()}",
                            description=f"Allergy conflict: {drug}.",
                            severity=FindingSeverity.CRITICAL,
                            affected_items=[drug],
                        )
                    )
                for drug_class in result.get("duplicate_classes", []):
                    findings.append(
                        Finding(
                            code=f"DUPLICATE_{drug_class.upper()}",
                            description=f"Duplicate drug class: {drug_class}.",
                            severity=FindingSeverity.HIGH,
                        )
                    )
                if result.get("polypharmacy"):
                    findings.append(
                        Finding(
                            code="POLYPHARMACY",
                            description=f"{len(medications)} concurrent medications.",
                            severity=FindingSeverity.MEDIUM,
                        )
                    )
                return self._build_response(
                    request=request,
                    risk_score=result.get("risk_score", 0.5),
                    findings=findings,
                    reasoning=result.get(
                        "reasoning", "LLM drug safety assessment completed."
                    ),
                    confidence=result.get("confidence", 0.9),
                    metadata={"llm_raw": result.data},
                )
            except Exception:
                pass

        from app.agents._drug_rules import drug_rules

        return drug_rules(request, medications, allergies, self)
