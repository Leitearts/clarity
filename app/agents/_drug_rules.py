from __future__ import annotations
from itertools import combinations
from app.models.agent import AgentRequest, AgentResponse, Finding, FindingSeverity
from app.models.patient import Medication

KNOWN_DDI: dict[frozenset, tuple] = {
    frozenset({"warfarin", "aspirin"}): (
        FindingSeverity.HIGH,
        "Warfarin + Aspirin: increased bleeding risk.",
    ),
    frozenset({"warfarin", "ibuprofen"}): (
        FindingSeverity.HIGH,
        "Warfarin + Ibuprofen: elevated bleed risk.",
    ),
    frozenset({"metformin", "contrast"}): (
        FindingSeverity.HIGH,
        "Metformin + Contrast: lactic acidosis risk.",
    ),
    frozenset({"ssri", "tramadol"}): (
        FindingSeverity.CRITICAL,
        "SSRI + Tramadol: serotonin syndrome risk.",
    ),
    frozenset({"fluoxetine", "tramadol"}): (
        FindingSeverity.CRITICAL,
        "SSRI + Tramadol: serotonin syndrome risk.",
    ),
    frozenset({"sertraline", "tramadol"}): (
        FindingSeverity.CRITICAL,
        "SSRI + Tramadol: serotonin syndrome risk.",
    ),
    frozenset({"lisinopril", "potassium"}): (
        FindingSeverity.MEDIUM,
        "ACE inhibitor + K+: hyperkalemia risk.",
    ),
    frozenset({"amiodarone", "warfarin"}): (
        FindingSeverity.CRITICAL,
        "Amiodarone + Warfarin: INR instability.",
    ),
    frozenset({"simvastatin", "amiodarone"}): (
        FindingSeverity.HIGH,
        "Simvastatin + Amiodarone: myopathy risk.",
    ),
    frozenset({"clopidogrel", "omeprazole"}): (
        FindingSeverity.MEDIUM,
        "Clopidogrel + Omeprazole: reduced antiplatelet efficacy.",
    ),
    frozenset({"methotrexate", "nsaid"}): (
        FindingSeverity.HIGH,
        "Methotrexate + NSAID: toxicity risk.",
    ),
}

DRUG_CLASSES = {
    "anticoagulant": ["warfarin", "heparin", "rivaroxaban", "apixaban", "dabigatran"],
    "nsaid": ["ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac"],
    "ssri": ["fluoxetine", "sertraline", "paroxetine", "escitalopram", "citalopram"],
    "opioid": [
        "morphine",
        "oxycodone",
        "fentanyl",
        "tramadol",
        "hydrocodone",
        "codeine",
    ],
    "benzodiazepine": [
        "diazepam",
        "lorazepam",
        "alprazolam",
        "clonazepam",
        "midazolam",
    ],
}


def drug_rules(
    request: AgentRequest, medications: list[Medication], allergies: list[str], agent
) -> AgentResponse:
    findings = []
    risk_score = 0.0
    med_names = [m.name.lower().strip() for m in medications]

    for med_a, med_b in combinations(med_names, 2):
        key = frozenset({med_a, med_b})
        if key in KNOWN_DDI:
            severity, description = KNOWN_DDI[key]
            findings.append(
                Finding(
                    code=f"DDI_{med_a.upper()}_{med_b.upper()}",
                    description=description,
                    severity=severity,
                    affected_items=[med_a, med_b],
                    evidence="Rule-based DDI match.",
                )
            )
            risk_score = min(
                1.0,
                risk_score
                + (
                    0.40
                    if severity == FindingSeverity.CRITICAL
                    else 0.25
                    if severity == FindingSeverity.HIGH
                    else 0.10
                ),
            )

    for drug_class, members in DRUG_CLASSES.items():
        active = [m for m in med_names if any(mem in m for mem in members)]
        if len(active) >= 2:
            findings.append(
                Finding(
                    code=f"DUPLICATE_CLASS_{drug_class.upper()}",
                    description=f"Multiple {drug_class}s: {', '.join(active)}.",
                    severity=FindingSeverity.HIGH,
                    affected_items=active,
                )
            )
            risk_score = min(1.0, risk_score + 0.20)

    norm_allergies = [a.lower().strip() for a in allergies]
    for med_name in med_names:
        for allergy in norm_allergies:
            if allergy in med_name or med_name in allergy:
                findings.append(
                    Finding(
                        code=f"ALLERGY_{med_name.upper()}",
                        description=f"Allergy conflict: {med_name} vs '{allergy}'.",
                        severity=FindingSeverity.CRITICAL,
                        affected_items=[med_name],
                    )
                )
                risk_score = min(1.0, risk_score + 0.50)

    if len(medications) >= 5:
        findings.append(
            Finding(
                code="POLYPHARMACY",
                description=f"{len(medications)} concurrent medications.",
                severity=FindingSeverity.MEDIUM,
            )
        )
        risk_score = min(1.0, risk_score + 0.10)

    if not findings:
        risk_score = 0.05

    return agent._build_response(
        request=request,
        risk_score=round(risk_score, 4),
        findings=findings,
        reasoning=f"Screened {len(medications)} medications. {len(findings)} findings.",
        confidence=0.7,
        metadata={"fallback": True},
    )
