from __future__ import annotations
from app.models.agent import AgentRequest, AgentResponse, Finding, FindingSeverity
from app.models.patient import VitalSign

# Clinical reference ranges for common vital signs
CRITICAL_RANGES: dict[str, tuple] = {
    "heart_rate": (40, 130, "bpm"),
    "hr": (40, 130, "bpm"),
    "systolic_bp": (70, 180, "mmHg"),
    "sbp": (70, 180, "mmHg"),
    "diastolic_bp": (40, 110, "mmHg"),
    "dbp": (40, 110, "mmHg"),
    "temperature": (35.0, 39.0, "°C"),
    "temp": (35.0, 39.0, "°C"),
    "spo2": (94, 100, "%"),
    "oxygen_saturation": (94, 100, "%"),
    "respiratory_rate": (12, 20, "/min"),
    "rr": (12, 20, "/min"),
    "mean_arterial_pressure": (65, 100, "mmHg"),
    "map": (65, 100, "mmHg"),
}


def _match(vital_name: str):
    name_lower = vital_name.lower()
    for key, thresholds in CRITICAL_RANGES.items():
        if key in name_lower:
            return thresholds
    return None


def vitals_rules(
    request: AgentRequest, vitals: list[VitalSign], agent
) -> AgentResponse:
    """Deterministic fallback logic for vital sign assessment."""
    findings = []
    risk_score = 0.0
    critical_count = 0
    abnormal_count = 0
    hemodynamic_risk = "stable"

    for vital in vitals:
        threshold = _match(vital.sign_name)
        if threshold:
            crit_low, crit_high, unit = threshold
            direction = ""
            if crit_low is not None and vital.value < crit_low:
                direction = f"critically low ({vital.value} < {crit_low})"
            elif crit_high is not None and vital.value > crit_high:
                direction = f"critically high ({vital.value} > {crit_high})"
            if direction:
                critical_count += 1
                risk_score = min(1.0, risk_score + 0.40)
                findings.append(
                    Finding(
                        code=f"CRITICAL_{vital.sign_name.upper().replace(' ', '_')}",
                        description=f"CRITICAL: {vital.sign_name} is {direction} {unit}.",
                        severity=FindingSeverity.CRITICAL,
                        affected_items=[vital.sign_name],
                    )
                )
                hemodynamic_risk = "unstable" if critical_count >= 2 else "at_risk"
                continue

        if vital.is_abnormal:
            abnormal_count += 1
            direction = (
                "low"
                if (vital.reference_low and vital.value < vital.reference_low)
                else "high"
            )
            risk_score = min(1.0, risk_score + 0.15)
            findings.append(
                Finding(
                    code=f"ABNORMAL_{vital.sign_name.upper().replace(' ', '_')}",
                    description=f"{vital.sign_name}: {vital.value} {vital.unit} is {direction}.",
                    severity=FindingSeverity.MEDIUM,
                    affected_items=[vital.sign_name],
                )
            )
            if critical_count == 0:
                hemodynamic_risk = "at_risk"

    if critical_count >= 2:
        risk_score = min(1.0, risk_score + 0.20)
        findings.append(
            Finding(
                code="MULTI_CRITICAL_VITALS",
                description=f"{critical_count} critical vital signs — hemodynamic compromise likely.",
                severity=FindingSeverity.CRITICAL,
            )
        )

    if not findings:
        risk_score = 0.05

    return agent._build_response(
        request=request,
        risk_score=round(risk_score, 4),
        findings=findings,
        reasoning=f"Reviewed {len(vitals)} vital signs. Critical: {critical_count}. Abnormal: {abnormal_count}. Status: {hemodynamic_risk}.",
        confidence=0.7,
        metadata={"fallback": True, "hemodynamic_risk": hemodynamic_risk},
    )
