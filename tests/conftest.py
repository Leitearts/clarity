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
            return LLMResponse(data=data, raw_text=json.dumps(data), token_usage={})
        mock.side_effect = side_effect
        yield mock


@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
