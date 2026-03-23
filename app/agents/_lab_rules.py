from __future__ import annotations
from app.models.agent import AgentRequest, AgentResponse, Finding, FindingSeverity
from app.models.patient import LabResult

CRITICAL_THRESHOLDS: dict[str, tuple] = {
    "potassium":  (2.5, 6.5, "mEq/L"),
    "sodium":     (120.0, 160.0, "mEq/L"),
    "glucose":    (40.0, 500.0, "mg/dL"),
    "hemoglobin": (7.0, None, "g/dL"),
    "hgb":        (7.0, None, "g/dL"),
    "platelet":   (50.0, None, "K/uL"),
    "creatinine": (None, 10.0, "mg/dL"),
    "troponin":   (None, 0.04, "ng/mL"),
    "inr":        (None, 4.0, "ratio"),
    "ph":         (7.2, 7.6, "units"),
    "lactate":    (None, 4.0, "mmol/L"),
    "bilirubin":  (None, 15.0, "mg/dL"),
}


def _match(test_name: str):
    name_lower = test_name.lower()
    for key, thresholds in CRITICAL_THRESHOLDS.items():
        if key in name_lower:
            return thresholds
    return None


def lab_rules(request: AgentRequest, labs: list[LabResult], agent) -> AgentResponse:
    findings = []
    risk_score = 0.0
    critical_count = 0
    abnormal_count = 0

    for lab in labs:
        threshold = _match(lab.test_name)
        if threshold:
            crit_low, crit_high, _ = threshold
            direction = ""
            if crit_low is not None and lab.value < crit_low:
                direction = f"critically low ({lab.value} < {crit_low})"
            elif crit_high is not None and lab.value > crit_high:
                direction = f"critically high ({lab.value} > {crit_high})"
            if direction:
                critical_count += 1
                risk_score = min(1.0, risk_score + 0.40)
                findings.append(Finding(
                    code=f"CRITICAL_{lab.test_name.upper().replace(' ', '_')}",
                    description=f"CRITICAL: {lab.test_name} is {direction} {lab.unit}.",
                    severity=FindingSeverity.CRITICAL,
                    affected_items=[lab.test_name]))
                continue

        if lab.is_abnormal:
            abnormal_count += 1
            direction = "low" if (lab.reference_low and lab.value < lab.reference_low) else "high"
            risk_score = min(1.0, risk_score + 0.10)
            findings.append(Finding(
                code=f"ABNORMAL_{lab.test_name.upper().replace(' ', '_')}",
                description=f"{lab.test_name}: {lab.value} {lab.unit} is {direction}.",
                severity=FindingSeverity.MEDIUM,
                affected_items=[lab.test_name]))

    if critical_count >= 2:
        risk_score = min(1.0, risk_score + 0.15)
        findings.append(Finding(code="MULTI_CRITICAL_LABS",
                                description=f"{critical_count} critical values — possible multi-organ involvement.",
                                severity=FindingSeverity.CRITICAL))

    if not findings:
        risk_score = 0.05

    return agent._build_response(request=request, risk_score=round(risk_score, 4),
                                  findings=findings,
                                  reasoning=f"Reviewed {len(labs)} labs. Critical: {critical_count}. Abnormal: {abnormal_count}.",
                                  confidence=0.7, metadata={"fallback": True})
