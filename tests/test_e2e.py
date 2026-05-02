from __future__ import annotations
import json
import pytest
from pathlib import Path

CASES_PATH = Path(__file__).parent.parent / "data" / "synthetic_cases.json"
_RAW = json.loads(CASES_PATH.read_text())
_META = {c["case_id"]: c["_meta"] for c in _RAW}


def _payload(case_id: str) -> dict:
    for c in _RAW:
        if c["case_id"] == case_id:
            return {k: v for k, v in c.items() if not k.startswith("_")}
    raise KeyError(case_id)


# ── Full pipeline via POST /analyze ──────────────────────────────────────────

class TestFullPipeline:

    @pytest.mark.asyncio
    async def test_critical_case_returns_critical_level(self, async_client, mock_llm):
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-CRITICAL-001"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk"]["level"] == "critical"
        assert data["risk"]["total_score"] >= _META["CASE-CRITICAL-001"]["expected_score_min"]

    @pytest.mark.asyncio
    async def test_low_case_returns_low_level(self, async_client, mock_llm):
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-LOW-001"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk"]["level"] == "low"
        assert data["risk"]["total_score"] < 0.35

    @pytest.mark.asyncio
    async def test_allergy_conflict_always_critical(self, async_client, mock_llm):
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-ALLERGY-001"))
        assert resp.status_code == 200
        assert resp.json()["risk"]["level"] == "critical"

    @pytest.mark.asyncio
    async def test_response_contains_all_required_fields(self, async_client, mock_llm):
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-MEDIUM-001"))
        assert resp.status_code == 200
        data = resp.json()
        for field in ("analysis_id", "risk", "agent_responses", "explanation",
                      "recommended_actions", "processing_time_ms"):
            assert field in data, f"Missing field: {field}"
        assert len(data["agent_responses"]) == 5

    @pytest.mark.asyncio
    async def test_all_agents_respond(self, async_client, mock_llm):
        from app.models.agent import AgentType
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-CRITICAL-001"))
        agent_types = {r["agent_type"] for r in resp.json()["agent_responses"]}
        assert agent_types == {t.value for t in AgentType}

    @pytest.mark.asyncio
    async def test_risk_scores_bounded(self, async_client, mock_llm):
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-HIGH-001"))
        for ar in resp.json()["agent_responses"]:
            assert 0.0 <= ar["risk_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, async_client, mock_llm):
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-LOW-001"))
        ms = resp.json().get("processing_time_ms")
        assert ms is not None and ms >= 0

    @pytest.mark.asyncio
    async def test_explanation_contains_disclaimer(self, async_client, mock_llm):
        resp = await async_client.post("/api/v1/analyze", json=_payload("CASE-MEDIUM-001"))
        assert "decision-support" in resp.json()["explanation"].lower()

    @pytest.mark.asyncio
    async def test_invalid_icd_code_returns_422(self, async_client):
        bad = _payload("CASE-LOW-001")
        bad["diagnoses"][0]["code"] = "NOT-VALID"
        resp = await async_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422


# ── A2A endpoint ──────────────────────────────────────────────────────────────

class TestA2AEndpoint:

    @pytest.mark.asyncio
    async def test_a2a_invoke_critical_case(self, async_client, mock_llm):
        a2a = {"trace_id": "test-001", "caller_id": "pytest",
               "capability": "clinical_risk_analysis",
               "payload": _payload("CASE-CRITICAL-001")}
        resp = await async_client.post("/api/v1/a2a/invoke", json=a2a)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["risk_level"] == "critical"
        assert data["trace_id"] == "test-001"

    @pytest.mark.asyncio
    async def test_a2a_unknown_capability_returns_rejected(self, async_client):
        a2a = {"trace_id": "test-002", "caller_id": "pytest",
               "capability": "nonexistent_capability", "payload": {}}
        resp = await async_client.post("/api/v1/a2a/invoke", json=a2a)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert data["error_code"] == "UNKNOWN_CAPABILITY"

    @pytest.mark.asyncio
    async def test_agent_card_correct_schema(self, async_client):
        resp = await async_client.get("/api/v1/agent-card")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "clarity-clinical-risk-v1"
        assert data["phi_accepted"] is False
        assert len(data["capabilities"]) >= 3

    @pytest.mark.asyncio
    async def test_a2a_drug_check_capability(self, async_client, mock_llm):
        a2a = {"trace_id": "test-003", "caller_id": "pytest",
               "capability": "drug_interaction_check",
               "payload": {
                   "case_id": "dc-test", "patient_id": "p-test",
                   "medications": [
                       {"name": "warfarin",   "dose_mg": 5.0,   "frequency": "daily", "route": "oral"},
                       {"name": "amiodarone", "dose_mg": 200.0, "frequency": "daily", "route": "oral"},
                   ],
                   "allergies": [],
               }}
        resp = await async_client.post("/api/v1/a2a/invoke", json=a2a)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"


# ── System endpoints ──────────────────────────────────────────────────────────

class TestSystemEndpoints:

    @pytest.mark.asyncio
    async def test_health_ok(self, async_client):
        resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_a2a_health_lists_agents(self, async_client):
        resp = await async_client.get("/api/v1/a2a/health")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        for name in ("diagnosis", "drug_interaction", "lab_analysis", "patient_context"):
            assert name in agents

    @pytest.mark.asyncio
    async def test_impact_endpoint_responds(self, async_client):
        resp = await async_client.get("/api/v1/impact")
        assert resp.status_code == 200
        assert "real_world_context" in resp.json()
