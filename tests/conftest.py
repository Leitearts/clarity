from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.patient import PatientCase

CASES_PATH = Path(__file__).parent.parent / "data" / "synthetic_cases.json"
_RAW_CASES = json.loads(CASES_PATH.read_text())


def _load_case(case_id: str) -> PatientCase:
    for raw in _RAW_CASES:
        if raw["case_id"] == case_id:
            data = {k: v for k, v in raw.items() if not k.startswith("_")}
            return PatientCase(**data)
    raise KeyError(f"Case not found: {case_id}")


@pytest.fixture
def case_critical() -> PatientCase:
    return _load_case("CASE-CRITICAL-001")


@pytest.fixture
def case_high() -> PatientCase:
    return _load_case("CASE-HIGH-001")


@pytest.fixture
def case_medium() -> PatientCase:
    return _load_case("CASE-MEDIUM-001")


@pytest.fixture
def case_low() -> PatientCase:
    return _load_case("CASE-LOW-001")


@pytest.fixture
def case_allergy() -> PatientCase:
    return _load_case("CASE-ALLERGY-001")


@pytest.fixture
def mock_llm():
    """Patches LLMService.complete — all tests run without a real API call."""
    with patch("app.llm.service.LLMService.complete", new_callable=AsyncMock) as mock:
        async def side_effect(system_prompt, user_prompt, required_keys, **kwargs):
            from app.llm.service import LLMResponse
            
            # Parse the prompt to determine what to return
            prompt_text = (system_prompt or "") + (user_prompt or "")
            
            # Base response
            data = {
                "risk_score": 0.5,
                "reasoning": "Mock LLM response for testing.",
                "confidence": 0.9,
                "high_acuity_conditions": [],
                "conflict_detected": False,
                "conflict_description": None,
                "rapid_onset_detected": False,
                "interactions": [],
                "allergy_conflicts": [],
                "duplicate_classes": [],
                "polypharmacy": False,
                "critical_values": [],
                "abnormal_values": [],
                "pattern_concerns": None,
                "context_multiplier": 1.0,
                "age_risk_flag": False,
                "comorbidity_burden": "none",
                "high_risk_comorbidities": [],
                "care_setting_acuity": "routine",
            }
            
            # Determine agent type from system prompt keywords
            is_diagnosis = "diagnosis risk assessment" in system_prompt.lower()
            is_drug = "drug safety analysis" in system_prompt.lower()
            is_lab = "lab result interpretation" in system_prompt.lower()
            is_context = "risk stratification" in system_prompt.lower()
            
            # DIAGNOSIS AGENT
            if is_diagnosis:
                if "I21" in prompt_text or "I21.9" in prompt_text:  # Acute MI
                    data["risk_score"] = 0.80
                    data["high_acuity_conditions"] = ["I21"]
                elif "A41" in prompt_text:  # Sepsis
                    data["risk_score"] = 0.85
                    data["high_acuity_conditions"] = ["A41"]
                    data["conflict_detected"] = True
                    data["conflict_description"] = "Sepsis detected"
                elif "J06.9" in prompt_text:  # Upper respiratory infection
                    data["risk_score"] = 0.15
                elif "E11" in prompt_text:  # Diabetes
                    data["risk_score"] = 0.3
                elif "J18.9" in prompt_text:  # Pneumonia - serious, especially with allergy conflict
                    data["risk_score"] = 0.80
            
            # DRUG INTERACTION AGENT
            elif is_drug:
                if "warfarin" in prompt_text.lower() and "amiodarone" in prompt_text.lower():
                    data["risk_score"] = 0.95
                    data["interactions"] = [{
                        "drugs": ["warfarin", "amiodarone"],
                        "severity": "critical",
                        "description": "Warfarin-amiodarone interaction: significant bleeding risk."
                    }]
                elif "amoxicillin" in prompt_text.lower() and "penicillin" in prompt_text.lower():
                    # Allergy conflict: amoxicillin is a penicillin-based antibiotic
                    data["risk_score"] = 0.95
                    data["allergy_conflicts"] = ["amoxicillin"]
                elif "fluoxetine" in prompt_text.lower() and "tramadol" in prompt_text.lower():
                    # SSRI + tramadol: serotonin syndrome risk
                    data["risk_score"] = 0.85
                    data["interactions"] = [{
                        "drugs": ["fluoxetine", "tramadol"],
                        "severity": "high",
                        "description": "Serotonin syndrome risk: SSRI + tramadol combination."
                    }]
                elif prompt_text.count("drug_") >= 5 or prompt_text.count("\"name\"") >= 5:  # Polypharmacy
                    data["risk_score"] = 0.65
                    data["polypharmacy"] = True
            
            # LAB ANALYSIS AGENT
            elif is_lab:
                # For normal labs combination, keep score very low
                if "hemoglobin" in prompt_text.lower() and "14.5" in prompt_text and "potassium" in prompt_text.lower() and "4.1" in prompt_text:
                    data["risk_score"] = 0.05
                # Detect specific abnormal values
                elif "14.2" in prompt_text:
                    # Elevated WBC (allergy case) - significant abnormality
                    data["risk_score"] = 0.80
                    data["abnormal_values"] = ["WBC"]
                elif "troponin" in prompt_text.lower() and ("2.1" in prompt_text or "high" in prompt_text.lower()):
                    # Critical troponin
                    data["risk_score"] = 0.90
                    data["critical_values"] = [{"test": "troponin", "value": 2.1, "unit": "ng/mL"}]
                elif "potassium" in prompt_text.lower() and ("6.2" in prompt_text or "2.3" in prompt_text or "2.5" in prompt_text or "6.8" in prompt_text):
                    # Critical potassium
                    data["risk_score"] = 0.85
                    critical_val = 2.3 if "2.3" in prompt_text else 2.5 if "2.5" in prompt_text else 6.2 if "6.2" in prompt_text else 6.8
                    data["critical_values"] = [{"test": "potassium", "value": critical_val, "unit": "mEq/L"}]
                elif "3.8" in prompt_text and "lactate" in prompt_text.lower():
                    # Elevated lactate
                    data["risk_score"] = 0.80
                elif "18.5" in prompt_text and "wbc" in prompt_text.lower():
                    # Very high WBC (sepsis case)
                    data["risk_score"] = 0.85
                elif "11.2" in prompt_text:
                    # Slightly elevated WBC/hemoglobin - mildly abnormal
                    data["risk_score"] = 0.25
                    data["abnormal_values"] = ["WBC"] if "wbc" in prompt_text.lower() else ["hemoglobin"]
                elif "6.8" in prompt_text and "hemoglobin" in prompt_text.lower():
                    # Abnormal hemoglobin
                    data["risk_score"] = 0.75
                elif "wbc" in prompt_text.lower():
                    # WBC present - use moderate/lower risk for generic WBC
                    data["risk_score"] = 0.3
                    data["abnormal_values"] = ["WBC"]
            
            # PATIENT CONTEXT AGENT
            elif is_context:
                if "78" in prompt_text and "icu" in prompt_text.lower():
                    # Elderly + ICU
                    data["context_multiplier"] = 1.25
                    data["age_risk_flag"] = True
                elif "age" in prompt_text.lower() and ("icu" in prompt_text.lower() or "emergency" in prompt_text.lower()):
                    # Handle other elderly/high-acuity settings
                    if any(age_str in prompt_text for age_str in ["71", "64", "65", "78"]):
                        data["context_multiplier"] = 1.15
                        data["age_risk_flag"] = True
            
            resp = LLMResponse(data=data, raw_text=json.dumps(data), token_usage={})
            return resp
        
        mock.side_effect = side_effect
        yield mock


@pytest_asyncio.fixture
async def async_client():
    from app.a2a.adapter import A2AAdapter
    from app.audit.logger import AuditLogger
    from app.orchestrator.orchestrator import Orchestrator
    
    # Manually initialize app state (lifespan may not trigger with ASGITransport)
    audit_logger = AuditLogger()
    orchestrator = Orchestrator()
    orchestrator._audit_logger = audit_logger
    a2a_adapter = A2AAdapter(orchestrator)
    
    app.state.orchestrator = orchestrator
    app.state.audit_logger = audit_logger
    app.state.a2a_adapter = a2a_adapter
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
