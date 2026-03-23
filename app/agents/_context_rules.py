from __future__ import annotations
from app.models.agent import AgentRequest, AgentResponse, Finding, FindingSeverity
from app.models.patient import PatientContext

HIGH_RISK_COMORBIDITIES = {
    "diabetes", "hypertension", "ckd", "chronic kidney disease",
    "heart failure", "copd", "immunocompromised", "cirrhosis",
    "malignancy", "cancer", "hiv", "aids",
}
HIGH_ACUITY_SETTINGS = {"icu", "emergency"}


def context_rules(request: AgentRequest, context: PatientContext, agent) -> AgentResponse:
    findings = []
    multiplier = 1.0

    if context.age >= 75:
        multiplier += 0.15
        findings.append(Finding(code="AGE_ELDERLY",
                                description=f"Patient age {context.age}: elevated baseline risk.",
                                severity=FindingSeverity.LOW))
    elif context.age <= 2:
        multiplier += 0.10
        findings.append(Finding(code="AGE_INFANT",
                                description=f"Patient age {context.age}: pediatric dosing applies.",
                                severity=FindingSeverity.MEDIUM))

    active = [c for c in context.comorbidities
              if any(h in c.lower() for h in HIGH_RISK_COMORBIDITIES)]
    if len(active) >= 3:
        multiplier += 0.20
        findings.append(Finding(code="COMORBIDITY_HIGH_BURDEN",
                                description=f"High comorbidity burden: {', '.join(active[:3])}.",
                                severity=FindingSeverity.MEDIUM, affected_items=active))
    elif active:
        multiplier += 0.10
        findings.append(Finding(code="COMORBIDITY_PRESENT",
                                description=f"Relevant comorbidities: {', '.join(active)}.",
                                severity=FindingSeverity.LOW, affected_items=active))

    if context.care_setting in HIGH_ACUITY_SETTINGS:
        multiplier += 0.10
        findings.append(Finding(code="HIGH_ACUITY_SETTING",
                                description=f"Care setting: {context.care_setting.upper()}.",
                                severity=FindingSeverity.LOW))

    multiplier = round(min(1.5, max(0.5, multiplier)), 2)
    reasoning = (f"Patient {context.age}y {context.sex}, {context.care_setting}. "
                 f"{len(active)} high-risk comorbidities. Multiplier: {multiplier}x.")

    response = agent._build_response(request=request, risk_score=0.0,
                                      findings=findings, reasoning=reasoning,
                                      metadata={"context_multiplier": multiplier, "fallback": True})
    return response
