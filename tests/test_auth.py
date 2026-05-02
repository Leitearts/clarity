"""Tests for API key authentication.

Covers:
- Protected endpoints return 403 when a key is configured and missing/wrong.
- Protected endpoints return 200 with the correct key.
- Public endpoints remain accessible without any key.
- Auth is a no-op when CLARITY_API_KEY is not configured (backward compat).
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock

from app.main import app
from app.config import settings


# Intentionally simple value for test assertions only.
# Production keys must be cryptographically secure random strings (e.g. `openssl rand -hex 32`).
_TEST_KEY = "test-secret-key-12345"

# ---------------------------------------------------------------------------
# Minimal valid PatientCase payload
# ---------------------------------------------------------------------------
_ANALYZE_PAYLOAD = {
    "patient_id": "auth-test-patient",
    "case_id": "auth-test-case-001",
    "context": {
        "age": 45,
        "sex": "male",
        "weight_kg": 75.0,
        "care_setting": "outpatient",
        "comorbidities": [],
        "allergies": [],
    },
    "diagnoses": [
        {"code": "J06.9", "description": "Upper respiratory infection", "is_primary": True},
    ],
    "medications": [],
    "lab_results": [],
}

_A2A_PAYLOAD = {
    "trace_id": "auth-test-001",
    "caller_id": "pytest-auth",
    "capability": "clinical_risk_analysis",
    "payload": _ANALYZE_PAYLOAD,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def authed_client():
    """Client wired to an app instance where CLARITY_API_KEY is set."""
    from app.a2a.adapter import A2AAdapter
    from app.audit.logger import AuditLogger
    from app.orchestrator.orchestrator import Orchestrator

    audit_logger = AuditLogger()
    orchestrator = Orchestrator()
    orchestrator._audit_logger = audit_logger
    a2a_adapter = A2AAdapter(orchestrator)

    app.state.orchestrator = orchestrator
    app.state.audit_logger = audit_logger
    app.state.a2a_adapter = a2a_adapter

    with patch.object(settings, "api_key", _TEST_KEY):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"X-API-Key": _TEST_KEY},
        ) as client:
            yield client


@pytest_asyncio.fixture
async def unauthed_client():
    """Client wired to an app instance where CLARITY_API_KEY is set, but sends no key."""
    from app.a2a.adapter import A2AAdapter
    from app.audit.logger import AuditLogger
    from app.orchestrator.orchestrator import Orchestrator

    audit_logger = AuditLogger()
    orchestrator = Orchestrator()
    orchestrator._audit_logger = audit_logger
    a2a_adapter = A2AAdapter(orchestrator)

    app.state.orchestrator = orchestrator
    app.state.audit_logger = audit_logger
    app.state.a2a_adapter = a2a_adapter

    with patch.object(settings, "api_key", _TEST_KEY):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client


@pytest_asyncio.fixture
async def unconfigured_client():
    """Client where CLARITY_API_KEY is empty — auth is a no-op."""
    from app.a2a.adapter import A2AAdapter
    from app.audit.logger import AuditLogger
    from app.orchestrator.orchestrator import Orchestrator

    audit_logger = AuditLogger()
    orchestrator = Orchestrator()
    orchestrator._audit_logger = audit_logger
    a2a_adapter = A2AAdapter(orchestrator)

    app.state.orchestrator = orchestrator
    app.state.audit_logger = audit_logger
    app.state.a2a_adapter = a2a_adapter

    with patch.object(settings, "api_key", ""):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client


# ---------------------------------------------------------------------------
# Helper to mock LLM so we don't need a real Anthropic key
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    with patch("app.llm.service.LLMService.complete", new_callable=AsyncMock) as mock:
        import json
        from app.llm.service import LLMResponse

        async def _side_effect(system_prompt, user_prompt, required_keys, **kwargs):
            data = {
                "risk_score": 0.1,
                "reasoning": "mock",
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
                "critical_vitals": [],
                "abnormal_vitals": [],
                "trend_concerns": None,
                "hemodynamic_risk": "stable",
            }
            return LLMResponse(data=data, raw_text=json.dumps(data), token_usage={})

        mock.side_effect = _side_effect
        yield mock


# ---------------------------------------------------------------------------
# 403 when key is configured and request provides no key
# ---------------------------------------------------------------------------

class TestUnauthorizedRequests:

    @pytest.mark.asyncio
    async def test_analyze_no_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.post("/api/v1/analyze", json=_ANALYZE_PAYLOAD)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_audit_list_no_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/audit")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_audit_by_id_no_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/audit/some-id")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_audit_by_case_no_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/audit/case/some-case")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_reasoning_no_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/reasoning/some-id")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_a2a_invoke_no_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.post("/api/v1/a2a/invoke", json=_A2A_PAYLOAD)
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_impact_no_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/impact")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_wrong_key_returns_403(self, unauthed_client):
        resp = await unauthed_client.get(
            "/api/v1/audit",
            headers={"X-API-Key": "totally-wrong-key"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 200 when key is configured and request provides the correct key
# ---------------------------------------------------------------------------

class TestAuthorizedRequests:

    @pytest.mark.asyncio
    async def test_analyze_with_key_succeeds(self, authed_client, mock_llm):
        resp = await authed_client.post("/api/v1/analyze", json=_ANALYZE_PAYLOAD)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_audit_list_with_key_succeeds(self, authed_client):
        resp = await authed_client.get("/api/v1/audit")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_a2a_invoke_with_key_succeeds(self, authed_client, mock_llm):
        resp = await authed_client.post("/api/v1/a2a/invoke", json=_A2A_PAYLOAD)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_impact_with_key_succeeds(self, authed_client):
        resp = await authed_client.get("/api/v1/impact")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Public endpoints remain accessible without any key (even when key is set)
# ---------------------------------------------------------------------------

class TestPublicEndpoints:

    @pytest.mark.asyncio
    async def test_health_no_key(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_agent_card_no_key(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/agent-card")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_a2a_schema_no_key(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/a2a/schema")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_a2a_capabilities_no_key(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/a2a/capabilities")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_a2a_health_no_key(self, unauthed_client):
        resp = await unauthed_client.get("/api/v1/a2a/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Backward compatibility: no key configured → all requests pass through
# ---------------------------------------------------------------------------

class TestUnconfiguredAuth:

    @pytest.mark.asyncio
    async def test_analyze_no_config_passes(self, unconfigured_client, mock_llm):
        resp = await unconfigured_client.post("/api/v1/analyze", json=_ANALYZE_PAYLOAD)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_audit_no_config_passes(self, unconfigured_client):
        resp = await unconfigured_client.get("/api/v1/audit")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_a2a_invoke_no_config_passes(self, unconfigured_client, mock_llm):
        resp = await unconfigured_client.post("/api/v1/a2a/invoke", json=_A2A_PAYLOAD)
        assert resp.status_code == 200
