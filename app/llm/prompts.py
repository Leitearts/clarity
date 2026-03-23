from __future__ import annotations
from app.models.patient import Diagnosis, Medication, LabResult, PatientContext

DIAGNOSIS_SYSTEM = """\
You are a clinical decision support AI specialized in diagnosis risk assessment.
You analyze ICD-10 coded diagnoses and return structured JSON risk assessments.

CRITICAL RULES:
- Output ONLY a valid JSON object. No preamble, no explanation, no markdown.
- Never provide treatment recommendations or prescriptive clinical advice.
- Base assessments only on the data provided.
- risk_score must be a float between 0.0 and 1.0.
- reasoning must be 1-3 sentences maximum, factual and neutral.
- This is a decision-support tool. Final clinical decisions rest with human clinicians.
"""

DRUG_SYSTEM = """\
You are a clinical pharmacology AI specialized in drug safety analysis.
You identify drug-drug interactions, contraindications, and dosage concerns.

CRITICAL RULES:
- Output ONLY a valid JSON object. No preamble, no explanation, no markdown.
- Never recommend specific drug changes or dosage adjustments.
- Flag concerns only — do not prescribe solutions.
- risk_score must be a float between 0.0 and 1.0.
- reasoning must be 1-3 sentences maximum.
"""

LAB_SYSTEM = """\
You are a clinical laboratory AI specialized in lab result interpretation.
You compare values against reference ranges and flag abnormalities.

CRITICAL RULES:
- Output ONLY a valid JSON object. No preamble, no explanation, no markdown.
- Never diagnose based on lab results alone.
- risk_score must be a float between 0.0 and 1.0.
- reasoning must be 1-3 sentences maximum.
"""

CONTEXT_SYSTEM = """\
You are a clinical risk stratification AI that evaluates patient demographics
and comorbidity burden to produce a risk context multiplier.

CRITICAL RULES:
- Output ONLY a valid JSON object. No preamble, no explanation, no markdown.
- context_multiplier must be a float between 0.5 and 1.5.
- reasoning must be 1-3 sentences maximum.
- Do not speculate about diagnoses or treatments.
"""


def build_diagnosis_prompt(diagnoses: list[Diagnosis]) -> str:
    diag_list = "\n".join(
        f"  - [{d.code}] {d.description} "
        f"{'(PRIMARY)' if d.is_primary else ''} "
        f"{f'onset {d.onset_days}d ago' if d.onset_days is not None else ''}"
        for d in diagnoses
    )
    return f"""\
Analyze the following clinical diagnoses for risk assessment.

DIAGNOSES:
{diag_list}

Return a JSON object with EXACTLY this structure:
{{
  "risk_score": <float 0.0-1.0>,
  "high_acuity_conditions": [<list of high-acuity diagnosis codes found>],
  "conflict_detected": <true|false>,
  "conflict_description": <string or null>,
  "rapid_onset_detected": <true|false>,
  "reasoning": "<1-3 sentences>",
  "confidence": <float 0.0-1.0>
}}"""


def build_drug_prompt(
    medications: list[Medication],
    allergies: list[str],
    diagnoses: list[Diagnosis],
) -> str:
    med_list = "\n".join(
        f"  - {m.name} {m.dose_mg}mg {m.frequency} ({m.route})"
        for m in medications
    )
    allergy_str = ", ".join(allergies) if allergies else "None documented"
    diag_codes = ", ".join(d.code for d in diagnoses)
    return f"""\
Analyze the following medication list for safety concerns.

MEDICATIONS:
{med_list}

DOCUMENTED ALLERGIES: {allergy_str}
ACTIVE DIAGNOSES (ICD-10): {diag_codes}

Return a JSON object with EXACTLY this structure:
{{
  "risk_score": <float 0.0-1.0>,
  "interactions": [
    {{"drugs": [<drug_a>, <drug_b>], "severity": <"low"|"medium"|"high"|"critical">, "description": "<brief>"}}
  ],
  "allergy_conflicts": [<drug names conflicting with documented allergies>],
  "duplicate_classes": [<drug class names where duplicates detected>],
  "polypharmacy": <true|false>,
  "reasoning": "<1-3 sentences>",
  "confidence": <float 0.0-1.0>
}}"""


def build_lab_prompt(lab_results: list[LabResult], context: PatientContext) -> str:
    lab_list = "\n".join(
        f"  - {l.test_name}: {l.value} {l.unit} (ref: {l.reference_low}-{l.reference_high})"
        for l in lab_results
    ) or "  None provided"
    return f"""\
Analyze the following lab results for clinical significance.

LAB RESULTS:
{lab_list}

PATIENT CONTEXT: Age {context.age}, Sex {context.sex}

Return a JSON object with EXACTLY this structure:
{{
  "risk_score": <float 0.0-1.0>,
  "critical_values": [
    {{"test": "<name>", "value": <number>, "unit": "<unit>", "direction": <"high"|"low">, "clinical_significance": "<note>"}}
  ],
  "abnormal_values": [<list of test names that are abnormal but not critical>],
  "pattern_concerns": "<brief or null>",
  "reasoning": "<1-3 sentences>",
  "confidence": <float 0.0-1.0>
}}"""


def build_context_prompt(context: PatientContext) -> str:
    comorbidity_str = ", ".join(context.comorbidities) if context.comorbidities else "None"
    allergy_str = ", ".join(context.allergies) if context.allergies else "None"
    return f"""\
Evaluate patient context for risk stratification.

PATIENT DEMOGRAPHICS:
  Age: {context.age}
  Sex: {context.sex}
  Weight: {context.weight_kg or 'unknown'} kg
  Care setting: {context.care_setting}
  Comorbidities: {comorbidity_str}
  Known allergies: {allergy_str}

Return a JSON object with EXACTLY this structure:
{{
  "context_multiplier": <float 0.5-1.5>,
  "age_risk_flag": <true|false>,
  "comorbidity_burden": <"none"|"low"|"moderate"|"high">,
  "high_risk_comorbidities": [<list>],
  "care_setting_acuity": <"routine"|"elevated"|"high">,
  "reasoning": "<1-3 sentences>",
  "confidence": <float 0.0-1.0>
}}"""
