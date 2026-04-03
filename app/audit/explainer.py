from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from app.models.agent import AgentResponse, AgentType, FindingSeverity
from app.models.risk import RiskLevel, RiskScore


@dataclass
class AgentExplanation:
    agent_name: str
    risk_score: float
    weight: float
    weighted_contribution: float
    contribution_pct: float
    findings_summary: list[str]
    critical_findings: list[str]
    reasoning: str
    had_error: bool = False


@dataclass
class ExplanationPayload:
    narrative: str
    risk_level: str
    risk_score: float
    dominant_factor: str
    agent_explanations: list[AgentExplanation]
    formula_trace: str
    all_critical_findings: list[str]
    recommended_actions: list[str]
    model_version: str
    analysis_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "narrative": self.narrative,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "dominant_factor": self.dominant_factor,
            "formula_trace": self.formula_trace,
            "all_critical_findings": self.all_critical_findings,
            "recommended_actions": self.recommended_actions,
            "agent_breakdown": [
                {
                    "agent": ae.agent_name,
                    "risk_score": ae.risk_score,
                    "weight": ae.weight,
                    "weighted_contribution": ae.weighted_contribution,
                    "contribution_pct": ae.contribution_pct,
                    "findings_summary": ae.findings_summary,
                    "reasoning": ae.reasoning,
                    "had_error": ae.had_error,
                }
                for ae in self.agent_explanations
            ],
            "model_version": self.model_version,
            "analysis_id": self.analysis_id,
        }


class ExplanationBuilder:
    def build(
        self,
        analysis_id: str,
        risk: RiskScore,
        agent_responses: list[AgentResponse],
        recommended_actions: list[str],
        model_version: str = "clarity-v1.0",
    ) -> ExplanationPayload:
        weight_map = {
            AgentType.DIAGNOSIS: risk.w_diagnosis,
            AgentType.DRUG_INTERACTION: risk.w_medication,
            AgentType.LAB_ANALYSIS: risk.w_lab,
            AgentType.PATIENT_CONTEXT: 0.0,
        }
        weighted_sum = max(
            risk.w_diagnosis * risk.diagnosis_score
            + risk.w_medication * risk.medication_score
            + risk.w_lab * risk.lab_score,
            1e-9,
        )
        agent_explanations: list[AgentExplanation] = []
        for resp in agent_responses:
            weight = weight_map.get(resp.agent_type, 0.0)
            wc = round(weight * resp.risk_score, 4)
            pct = round((wc / weighted_sum) * 100, 1) if weight > 0 else 0.0
            agent_explanations.append(
                AgentExplanation(
                    agent_name=_display_name(resp.agent_type),
                    risk_score=resp.risk_score,
                    weight=weight,
                    weighted_contribution=wc,
                    contribution_pct=pct,
                    findings_summary=[f.description for f in resp.findings],
                    critical_findings=[
                        f.description
                        for f in resp.findings
                        if f.severity
                        in (FindingSeverity.CRITICAL, FindingSeverity.HIGH)
                    ],
                    reasoning=resp.reasoning,
                    had_error=resp.error is not None,
                )
            )
        scored = [ae for ae in agent_explanations if ae.weight > 0]
        dominant = max(scored, key=lambda ae: ae.weighted_contribution, default=None)
        dominant_name = dominant.agent_name if dominant else "unknown"
        all_critical = [
            f"[{ae.agent_name.upper()}] {f}"
            for ae in agent_explanations
            for f in ae.critical_findings
        ]
        formula_trace = (
            f"R = ({risk.w_diagnosis:.2f} × {risk.diagnosis_score:.4f}"
            f" + {risk.w_medication:.2f} × {risk.medication_score:.4f}"
            f" + {risk.w_lab:.2f} × {risk.lab_score:.4f})"
            f" × {risk.context_multiplier:.2f}"
            f" = {risk.total_score:.4f}"
        )
        narrative = _build_narrative(
            risk, agent_explanations, dominant_name, all_critical, recommended_actions
        )
        return ExplanationPayload(
            narrative=narrative,
            risk_level=risk.level.value,
            risk_score=risk.total_score,
            dominant_factor=dominant_name,
            agent_explanations=agent_explanations,
            formula_trace=formula_trace,
            all_critical_findings=all_critical,
            recommended_actions=recommended_actions,
            model_version=model_version,
            analysis_id=analysis_id,
        )


def _display_name(agent_type: AgentType) -> str:
    return {
        AgentType.DIAGNOSIS: "diagnosis",
        AgentType.DRUG_INTERACTION: "drug interaction",
        AgentType.LAB_ANALYSIS: "lab analysis",
        AgentType.PATIENT_CONTEXT: "patient context",
    }.get(agent_type, agent_type.value)


def _build_narrative(
    risk: RiskScore,
    agent_explanations: list[AgentExplanation],
    dominant_name: str,
    all_critical: list[str],
    recommended_actions: list[str],
) -> str:
    level_phrases = {
        RiskLevel.LOW: f"CLARITY assessed this case as LOW risk (score {risk.total_score:.2f}).",
        RiskLevel.MEDIUM: f"CLARITY assessed this case as MEDIUM risk (score {risk.total_score:.2f}), warranting clinical review within 24 hours.",
        RiskLevel.HIGH: f"CLARITY assessed this case as HIGH risk (score {risk.total_score:.2f}). Urgent review recommended within 1 hour.",
        RiskLevel.CRITICAL: f"CLARITY assessed this case as CRITICAL risk (score {risk.total_score:.2f}). Immediate clinical intervention is indicated.",
    }
    parts = [level_phrases.get(risk.level, f"Risk: {risk.level.value}"), ""]
    for ae in agent_explanations:
        if ae.weight == 0:
            continue
        if ae.had_error:
            parts.append(
                f"The {ae.agent_name} agent was unavailable (fallback score: {ae.risk_score:.2f})."
            )
        elif ae.risk_score >= 0.65:
            parts.append(
                f"The {ae.agent_name} agent returned a high score of {ae.risk_score:.2f} ({ae.contribution_pct:.0f}% of total): {ae.reasoning}"
            )
        elif ae.risk_score >= 0.35:
            parts.append(
                f"The {ae.agent_name} agent flagged moderate risk ({ae.risk_score:.2f}): {ae.reasoning}"
            )
        else:
            parts.append(
                f"The {ae.agent_name} agent found minimal concerns (score {ae.risk_score:.2f})."
            )
    if risk.context_multiplier != 1.0:
        direction = "increased" if risk.context_multiplier > 1.0 else "reduced"
        ctx = next(
            (ae for ae in agent_explanations if ae.agent_name == "patient context"),
            None,
        )
        ctx_reasoning = ctx.reasoning if ctx else ""
        parts.extend(
            [
                "",
                f"Patient context {direction} the score by ×{risk.context_multiplier:.2f}. {ctx_reasoning}",
            ]
        )
    if all_critical:
        count = len(all_critical)
        parts.extend(
            [
                "",
                f"{count} critical/high finding{'s' if count > 1 else ''}: "
                + "; ".join(all_critical[:3])
                + (" [and more]" if count > 3 else "."),
            ]
        )
    parts.extend(["", f"Primary driver: {dominant_name} assessment."])
    if recommended_actions:
        parts.append(f"Recommended action: {recommended_actions[0]}")
    return "\n".join(parts)
