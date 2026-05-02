from __future__ import annotations
import logging
import re
from app.models.patient import PatientCase
from app.models.response import AnalysisResponse

logger = logging.getLogger(__name__)

PROHIBITED_OUTPUT_PATTERNS = [
    r"\bprescribe\b", r"\badminister\b", r"\bdose of\b",
    r"\btake \d", r"\bstop taking\b", r"\bdiscontinue\b",
    r"\bshould take\b", r"\bmust take\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_OUTPUT_PATTERNS]

MAX_MEDICATIONS = 100
MAX_DIAGNOSES = 50
MAX_LAB_RESULTS = 200

DISCLAIMER = (
    "\n\n⚠ CLARITY is a clinical decision-support tool. "
    "All findings are informational only. "
    "Final clinical decisions must be made by qualified healthcare professionals."
)


class SafetyError(Exception):
    pass


def validate_input(case: PatientCase) -> None:
    """Validate structural limits on a patient case.

    Prompt-injection sanitization is enforced earlier, at Pydantic model
    validation time (see ``app/safety/sanitize.py`` and the validators in
    ``app/models/patient.py``).  This function checks list-size limits that
    fall outside Pydantic's per-field scope.
    """
    if len(case.medications) > MAX_MEDICATIONS:
        raise SafetyError(f"Medication count {len(case.medications)} exceeds maximum {MAX_MEDICATIONS}.")
    if len(case.diagnoses) > MAX_DIAGNOSES:
        raise SafetyError(f"Diagnosis count {len(case.diagnoses)} exceeds maximum {MAX_DIAGNOSES}.")
    if len(case.lab_results) > MAX_LAB_RESULTS:
        raise SafetyError(f"Lab count {len(case.lab_results)} exceeds maximum {MAX_LAB_RESULTS}.")


def sanitize_output(response: AnalysisResponse) -> AnalysisResponse:
    flagged: list[str] = []

    def _check(text: str, field: str) -> str:
        for pattern in _COMPILED:
            if pattern.search(text):
                flagged.append(f"Prohibited pattern in {field}")
                text = pattern.sub("[REDACTED]", text)
        return text

    clean_explanation = _check(response.explanation, "explanation")
    clean_responses = []
    for ar in response.agent_responses:
        clean_r = _check(ar.reasoning, f"{ar.agent_type.value}.reasoning")
        clean_responses.append(ar.model_copy(update={"reasoning": clean_r}))
    if flagged:
        logger.warning("safety.output_flagged analysis_id=%s flags=%s",
                       response.analysis_id, flagged)
    return response.model_copy(update={
        "explanation": clean_explanation,
        "agent_responses": clean_responses,
    })


def append_disclaimer(response: AnalysisResponse) -> AnalysisResponse:
    return response.model_copy(update={"explanation": response.explanation + DISCLAIMER})
