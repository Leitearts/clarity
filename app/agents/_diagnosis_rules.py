from __future__ import annotations
from app.models.agent import AgentRequest, AgentResponse, Finding, FindingSeverity
from app.models.patient import Diagnosis

HIGH_ACUITY_ICD_PREFIXES = {
    "I21": "Acute myocardial infarction",
    "I22": "Subsequent MI",
    "I63": "Cerebral infarction",
    "A41": "Sepsis",
    "J96": "Respiratory failure",
    "N17": "Acute kidney injury",
    "K92": "GI hemorrhage",
    "I50": "Heart failure",
    "J18": "Pneumonia",
}

CONFLICTING_PAIRS = [("E11", "E10")]


def diagnose_rules(
    request: AgentRequest, diagnoses: list[Diagnosis], agent
) -> AgentResponse:
    findings = []
    risk_score = 0.0
    acuity_hits = []

    for diag in diagnoses:
        prefix = diag.code[:3]
        if prefix in HIGH_ACUITY_ICD_PREFIXES:
            acuity_hits.append(diag)
            findings.append(
                Finding(
                    code=f"HIGH_ACUITY_{prefix}",
                    description=f"{HIGH_ACUITY_ICD_PREFIXES[prefix]} ({diag.code}).",
                    severity=FindingSeverity.HIGH,
                    affected_items=[diag.code],
                    evidence="Rule-based: ICD-10 high-acuity prefix match.",
                )
            )

    codes_present = {d.code[:3] for d in diagnoses}
    for pair in CONFLICTING_PAIRS:
        if pair[0] in codes_present and pair[1] in codes_present:
            findings.append(
                Finding(
                    code="CONFLICTING_DIAGNOSES",
                    description=f"Conflicting: {pair[0]} and {pair[1]}.",
                    severity=FindingSeverity.MEDIUM,
                    affected_items=list(pair),
                )
            )
            risk_score += 0.15

    risk_score += 0.75 if len(acuity_hits) >= 2 else (0.55 if acuity_hits else 0.10)

    rapid = [d for d in diagnoses if d.onset_days is not None and d.onset_days <= 3]
    if rapid:
        risk_score = min(1.0, risk_score + 0.10)
        findings.append(
            Finding(
                code="RAPID_ONSET",
                description=f"{len(rapid)} diagnosis/es with onset <=3 days.",
                severity=FindingSeverity.MEDIUM,
                affected_items=[d.code for d in rapid],
            )
        )

    primary = next((d for d in diagnoses if d.is_primary), diagnoses[0])
    reasoning = (
        f"Reviewed {len(diagnoses)} diagnosis/es. Primary: {primary.code}. "
        f"{len(acuity_hits)} high-acuity condition(s). Score: {round(min(1.0, risk_score), 4)}."
    )
    return agent._build_response(
        request=request,
        risk_score=round(min(1.0, risk_score), 4),
        findings=findings,
        reasoning=reasoning,
        confidence=0.7,
        metadata={"fallback": True},
    )
