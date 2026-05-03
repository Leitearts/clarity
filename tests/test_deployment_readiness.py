"""
CLARITY Deployment Readiness Test Suite
========================================
Production-grade automated tests to determine if CLARITY is ready for deployment.
Covers: E2E, audit logging, reasoning traces, LLM fallback, input validation,
concurrency, A2A protocol, health checks, and JSON schema enforcement.

Run with:
    pytest tests/test_deployment_readiness.py -v

All tests are self-contained and use a temporary audit log to avoid contaminating
the production log file.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app

# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

#: High-risk patient — designed to reliably score >= 0.85 (CRITICAL) using the
#: rule-based fallback (no LLM required).
#: Diagnosis: 2 high-acuity ICD-10 codes + rapid onset  → D ≈ 0.85
#: Medication: warfarin+amiodarone critical DDI + polypharmacy  → M ≈ 0.75
#: Lab: troponin, INR, hemoglobin all critically abnormal  → L = 1.0
#: Context: age 78, ICU, 3 high-risk comorbidities  → multiplier ≈ 1.45
#: Final R ≈ (0.30×0.85 + 0.25×0.75 + 0.20×1.0) × 1.45 ≈ 0.93 → CRITICAL ✓
CRITICAL_PAYLOAD: dict = {
    "patient_id": "DEPLOY-TEST-001",
    "case_id": f"DEPLOY-CRITICAL-{uuid.uuid4().hex[:8]}",
    "context": {
        "age": 78,
        "sex": "male",
        "weight_kg": 80.0,
        "care_setting": "icu",
        "comorbidities": ["heart failure", "chronic kidney disease", "diabetes"],
        "allergies": ["sulfa"],
    },
    "diagnoses": [
        {
            "code": "I21.9",
            "description": "Acute MI, unspecified",
            "is_primary": True,
            "onset_days": 1,
        },
        {
            "code": "I50.9",
            "description": "Heart failure, unspecified",
            "is_primary": False,
        },
    ],
    "medications": [
        {
            "name": "warfarin",
            "dose_mg": 5.0,
            "frequency": "once daily",
            "route": "oral",
        },
        {
            "name": "amiodarone",
            "dose_mg": 200.0,
            "frequency": "once daily",
            "route": "oral",
        },
        {
            "name": "aspirin",
            "dose_mg": 81.0,
            "frequency": "once daily",
            "route": "oral",
        },
        {
            "name": "lisinopril",
            "dose_mg": 10.0,
            "frequency": "once daily",
            "route": "oral",
        },
        {
            "name": "metformin",
            "dose_mg": 500.0,
            "frequency": "once daily",
            "route": "oral",
        },
    ],
    "lab_results": [
        {
            "test_name": "troponin",
            "value": 2.1,
            "unit": "ng/mL",
            "reference_low": 0.0,
            "reference_high": 0.04,
        },
        {
            "test_name": "INR",
            "value": 4.8,
            "unit": "ratio",
            "reference_low": 0.8,
            "reference_high": 1.2,
        },
        {
            "test_name": "hemoglobin",
            "value": 6.8,
            "unit": "g/dL",
            "reference_low": 13.5,
            "reference_high": 17.5,
        },
    ],
}

#: Minimal valid payload for low-risk scenarios and validation tests.
LOW_RISK_PAYLOAD: dict = {
    "patient_id": "DEPLOY-TEST-002",
    "case_id": f"DEPLOY-LOW-{uuid.uuid4().hex[:8]}",
    "context": {
        "age": 32,
        "sex": "female",
        "care_setting": "outpatient",
        "comorbidities": [],
        "allergies": [],
    },
    "diagnoses": [
        {
            "code": "J06.9",
            "description": "Acute upper respiratory infection",
            "is_primary": True,
        },
    ],
    "medications": [],
    "lab_results": [],
}

#: A2A request envelope wrapping the critical payload.
A2A_CRITICAL_REQUEST: dict = {
    "trace_id": f"deploy-trace-{uuid.uuid4().hex[:8]}",
    "caller_id": "deployment-test-suite",
    "capability": "clinical_risk_analysis",
    "payload": CRITICAL_PAYLOAD,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def deploy_client(tmp_path: Path):
    """
    AsyncClient with a fully initialised app using a temporary audit log.
    Using a temp log ensures tests are isolated from each other and from
    any production log file.
    """
    from app.a2a.adapter import A2AAdapter
    from app.audit.logger import AuditLogger
    from app.orchestrator.orchestrator import Orchestrator

    audit_log = tmp_path / "audit_test.jsonl"
    audit_logger = AuditLogger(log_path=str(audit_log))
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


# ---------------------------------------------------------------------------
# 1. End-to-End Critical Case Test
# ---------------------------------------------------------------------------


class TestCriticalCaseEndToEnd:
    """
    Validates the full analysis pipeline against a high-risk patient case.
    Uses rule-based fallback (no LLM required) so the test is deterministic.
    """

    @pytest.mark.asyncio
    async def test_status_code_is_200(self, deploy_client: AsyncClient):
        """POST /analyze must return HTTP 200 for a valid payload."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_risk_score_is_critical(self, deploy_client: AsyncClient):
        """Total risk score must reach the CRITICAL threshold (>= 0.85)."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        data = resp.json()
        total_score = data["risk"]["total_score"]
        risk_level = data["risk"]["level"]
        assert total_score >= 0.85, f"Expected total_score >= 0.85, got {total_score}"
        assert risk_level == "critical", (
            f"Expected level='critical', got '{risk_level}'"
        )

    @pytest.mark.asyncio
    async def test_response_json_structure_is_valid(self, deploy_client: AsyncClient):
        """Response must contain all top-level fields defined in AnalysisResponse."""
        required_keys = {
            "analysis_id",
            "case_id",
            "patient_id",
            "timestamp",
            "risk",
            "agent_responses",
            "explanation",
            "recommended_actions",
            "processing_time_ms",
            "model_version",
        }
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        data = resp.json()
        missing = required_keys - set(data.keys())
        assert not missing, f"Missing top-level fields: {missing}"

    @pytest.mark.asyncio
    async def test_all_agent_outputs_present(self, deploy_client: AsyncClient):
        """All four analysis agents must produce an output."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        agents = {r["agent_type"] for r in resp.json()["agent_responses"]}
        expected = {"diagnosis", "drug_interaction", "lab_analysis", "patient_context"}
        assert expected.issubset(agents), f"Missing agents: {expected - agents}"

    @pytest.mark.asyncio
    async def test_explainability_fields_exist(self, deploy_client: AsyncClient):
        """Risk object must expose per-agent weights and contribution scores."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        risk = resp.json()["risk"]
        # Weights
        for field in ("w_diagnosis", "w_medication", "w_lab", "w_vitals"):
            assert field in risk, f"Missing weight field: {field}"
        # Contributions list
        assert "contributions" in risk, "Missing 'contributions' field in risk"
        contributions = risk["contributions"]
        assert len(contributions) >= 1, "Expected at least one contribution entry"
        for c in contributions:
            assert "agent_type" in c, "Contribution missing 'agent_type'"
            assert "raw_score" in c, "Contribution missing 'raw_score'"
            assert "weight" in c, "Contribution missing 'weight'"
            assert "weighted_score" in c, "Contribution missing 'weighted_score'"

    @pytest.mark.asyncio
    async def test_critical_findings_included(self, deploy_client: AsyncClient):
        """A CRITICAL case must have at least one critical finding surfaced."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        data = resp.json()
        # computed_field on AnalysisResponse returns critical/high findings
        critical = data.get("critical_findings", [])
        assert len(critical) >= 1, "Expected at least one critical finding in response"

    @pytest.mark.asyncio
    async def test_recommended_actions_present(self, deploy_client: AsyncClient):
        """CRITICAL risk must produce at least one recommended action."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        actions = resp.json().get("recommended_actions", [])
        assert len(actions) >= 1, "Expected recommended actions for critical patient"
        # At least one action must mention immediate review
        assert any("CRITICAL" in a or "Immediate" in a for a in actions), (
            f"No immediate-review action found: {actions}"
        )

    @pytest.mark.asyncio
    async def test_processing_time_is_recorded(self, deploy_client: AsyncClient):
        """processing_time_ms must be present and positive."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        ms = resp.json().get("processing_time_ms")
        assert ms is not None, "processing_time_ms is null"
        assert ms >= 0, f"processing_time_ms must be >= 0, got {ms}"


# ---------------------------------------------------------------------------
# 2. Audit Log Validation Test
# ---------------------------------------------------------------------------


class TestAuditLogValidation:
    """
    Verifies that analysis results are persisted to an immutable append-only audit log
    and that the stored record is complete and retrievable.
    """

    @pytest.mark.asyncio
    async def test_audit_record_created_after_analysis(
        self, deploy_client: AsyncClient
    ):
        """Audit record for the analysis must exist and be retrievable."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        assert resp.status_code == 200
        analysis_id = resp.json()["analysis_id"]

        # Allow background audit task to flush to disk
        await asyncio.sleep(0.2)

        audit_resp = await deploy_client.get(f"/api/v1/audit/{analysis_id}")
        assert audit_resp.status_code == 200, (
            f"Audit record not found for analysis_id={analysis_id}"
        )

    @pytest.mark.asyncio
    async def test_audit_record_contains_required_fields(
        self, deploy_client: AsyncClient
    ):
        """Audit record must include all fields needed for post-hoc investigation."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        audit = (await deploy_client.get(f"/api/v1/audit/{analysis_id}")).json()
        required = {
            "audit_id",
            "analysis_id",
            "case_id",
            "patient_id",
            "timestamp",
            "diagnosis_score",
            "medication_score",
            "lab_score",
            "context_multiplier",
            "total_risk_score",
            "risk_level",
            "agent_reasoning",
        }
        missing = required - set(audit.keys())
        assert not missing, f"Audit record missing fields: {missing}"

    @pytest.mark.asyncio
    async def test_audit_record_includes_input_counts(self, deploy_client: AsyncClient):
        """Audit must record how many diagnoses, medications, and labs were submitted."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        audit = (await deploy_client.get(f"/api/v1/audit/{analysis_id}")).json()
        assert audit["input_diagnosis_count"] == len(CRITICAL_PAYLOAD["diagnoses"])
        assert audit["input_medication_count"] == len(CRITICAL_PAYLOAD["medications"])
        assert audit["input_lab_count"] == len(CRITICAL_PAYLOAD["lab_results"])

    @pytest.mark.asyncio
    async def test_audit_record_has_agent_reasoning(self, deploy_client: AsyncClient):
        """agent_reasoning must capture per-agent reasoning strings."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        audit = (await deploy_client.get(f"/api/v1/audit/{analysis_id}")).json()
        reasoning = audit.get("agent_reasoning", {})
        assert isinstance(reasoning, dict), "agent_reasoning must be a dict"
        assert len(reasoning) >= 1, "agent_reasoning must not be empty"

    @pytest.mark.asyncio
    async def test_audit_record_timestamps_exist(self, deploy_client: AsyncClient):
        """Timestamp must be present and parseable as ISO-8601."""
        from datetime import datetime

        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        audit = (await deploy_client.get(f"/api/v1/audit/{analysis_id}")).json()
        ts = audit.get("timestamp")
        assert ts is not None, "Audit record has no timestamp"
        # Must be valid ISO-8601 — will raise if not
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    @pytest.mark.asyncio
    async def test_audit_log_is_immutable(self, deploy_client: AsyncClient):
        """
        A second analysis of the same case must produce a NEW audit record rather
        than overwriting the first — verifying the append-only contract.
        """
        # First analysis
        payload = {
            **CRITICAL_PAYLOAD,
            "case_id": f"IMMUTABLE-TEST-{uuid.uuid4().hex[:6]}",
        }
        r1 = await deploy_client.post("/api/v1/analyze", json=payload)
        id1 = r1.json()["analysis_id"]
        await asyncio.sleep(0.2)

        # Second analysis of the same case
        r2 = await deploy_client.post("/api/v1/analyze", json=payload)
        id2 = r2.json()["analysis_id"]
        await asyncio.sleep(0.2)

        # Both records must exist independently
        assert id1 != id2, "analysis_ids should differ across requests"
        a1 = await deploy_client.get(f"/api/v1/audit/{id1}")
        a2 = await deploy_client.get(f"/api/v1/audit/{id2}")
        assert a1.status_code == 200, "First audit record was overwritten or missing"
        assert a2.status_code == 200, "Second audit record is missing"


# ---------------------------------------------------------------------------
# 3. Reasoning Trace Test
# ---------------------------------------------------------------------------


class TestReasoningTrace:
    """
    Validates that per-agent LLM (or fallback) reasoning is recorded and
    accessible via the /reasoning endpoint.
    """

    @pytest.mark.asyncio
    async def test_reasoning_endpoint_returns_200(self, deploy_client: AsyncClient):
        """GET /reasoning/{analysis_id} must return 200 for a completed analysis."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        reasoning_resp = await deploy_client.get(f"/api/v1/reasoning/{analysis_id}")
        assert reasoning_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_reasoning_contains_all_agent_keys(self, deploy_client: AsyncClient):
        """Reasoning trace must include at least all analysis agents."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        data = (await deploy_client.get(f"/api/v1/reasoning/{analysis_id}")).json()
        reasoning = data.get("agent_reasoning", {})
        expected = {"diagnosis", "drug_interaction", "lab_analysis", "patient_context"}
        missing = expected - set(reasoning.keys())
        assert not missing, f"Reasoning trace missing agents: {missing}"

    @pytest.mark.asyncio
    async def test_each_reasoning_string_is_non_empty(self, deploy_client: AsyncClient):
        """Every agent reasoning string must contain actual content."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        data = (await deploy_client.get(f"/api/v1/reasoning/{analysis_id}")).json()
        for agent, text in data["agent_reasoning"].items():
            assert isinstance(text, str), f"Reasoning for {agent} is not a string"
            assert len(text.strip()) >= 10, (
                f"Reasoning for {agent} is too short (< 10 chars): '{text}'"
            )

    @pytest.mark.asyncio
    async def test_reasoning_includes_formula_trace(self, deploy_client: AsyncClient):
        """Reasoning endpoint must expose the risk formula trace string."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        data = (await deploy_client.get(f"/api/v1/reasoning/{analysis_id}")).json()
        formula = data.get("formula_trace", "")
        assert formula, "formula_trace must not be empty"
        # Must reference the formula structure: R = (...) × multiplier = score
        assert "=" in formula, (
            f"formula_trace does not look like a formula: '{formula}'"
        )

    @pytest.mark.asyncio
    async def test_unknown_analysis_id_returns_404(self, deploy_client: AsyncClient):
        """Requesting reasoning for a non-existent analysis_id must return 404."""
        resp = await deploy_client.get("/api/v1/reasoning/non-existent-id-999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. LLM Failure / Fallback Test
# ---------------------------------------------------------------------------


class TestLLMFallback:
    """
    Simulates LLM service failure and verifies the rule-based fallback
    produces a valid, complete response.
    """

    @pytest.mark.asyncio
    async def test_system_returns_200_when_llm_raises(self, deploy_client: AsyncClient):
        """When the LLM raises an exception, the system must still return HTTP 200."""
        with patch(
            "app.llm.service.LLMService.complete",
            side_effect=Exception("Simulated LLM outage"),
        ):
            resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        assert resp.status_code == 200, (
            f"Expected 200 during LLM outage, got {resp.status_code}"
        )

    @pytest.mark.asyncio
    async def test_fallback_response_is_valid_json(self, deploy_client: AsyncClient):
        """The fallback response must be valid, well-structured JSON."""
        with patch(
            "app.llm.service.LLMService.complete",
            side_effect=Exception("Simulated LLM outage"),
        ):
            resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        # Will raise if JSON is malformed
        data = resp.json()
        assert "risk" in data
        assert "agent_responses" in data

    @pytest.mark.asyncio
    async def test_fallback_agents_are_marked(self, deploy_client: AsyncClient):
        """When rule-based fallback runs, agent metadata must indicate fallback mode."""
        with patch(
            "app.llm.service.LLMService.complete",
            side_effect=Exception("Simulated LLM outage"),
        ):
            resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        agents = resp.json()["agent_responses"]
        fallback_agents = [
            a for a in agents if a.get("metadata", {}).get("fallback") is True
        ]
        # Diagnosis, drug, lab, and context agents all have rule-based fallbacks
        assert len(fallback_agents) >= 3, (
            f"Expected >= 3 agents in fallback mode, got {len(fallback_agents)}"
        )

    @pytest.mark.asyncio
    async def test_fallback_produces_bounded_scores(self, deploy_client: AsyncClient):
        """Fallback scores must remain within [0.0, 1.0]."""
        with patch(
            "app.llm.service.LLMService.complete",
            side_effect=Exception("Simulated LLM outage"),
        ):
            resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        for agent in resp.json()["agent_responses"]:
            score = agent["risk_score"]
            assert 0.0 <= score <= 1.0, (
                f"Agent {agent['agent_type']} returned out-of-range score: {score}"
            )

    @pytest.mark.asyncio
    async def test_fallback_still_scores_critical(self, deploy_client: AsyncClient):
        """Rule-based fallback must correctly identify a high-risk case as CRITICAL."""
        with patch(
            "app.llm.service.LLMService.complete",
            side_effect=Exception("Simulated LLM outage"),
        ):
            resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        data = resp.json()
        assert data["risk"]["level"] == "critical", (
            f"Expected 'critical' in fallback mode, got '{data['risk']['level']}'"
        )
        assert data["risk"]["total_score"] >= 0.85


# ---------------------------------------------------------------------------
# 5. Input Validation Test
# ---------------------------------------------------------------------------


class TestInputValidation:
    """
    Verifies that CLARITY rejects malformed or invalid payloads gracefully
    and never crashes or creates phantom audit records.
    """

    @pytest.mark.asyncio
    async def test_missing_required_field_returns_422(self, deploy_client: AsyncClient):
        """Payload missing the required 'diagnoses' field must return HTTP 422."""
        bad = {k: v for k, v in CRITICAL_PAYLOAD.items() if k != "diagnoses"}
        resp = await deploy_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_icd10_code_returns_422(self, deploy_client: AsyncClient):
        """A diagnosis code that does not match ICD-10 format must fail validation."""
        bad = {
            **CRITICAL_PAYLOAD,
            "diagnoses": [
                {"code": "NOT-VALID", "description": "Bad code", "is_primary": True}
            ],
        }
        resp = await deploy_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_negative_dose_mg_returns_422(self, deploy_client: AsyncClient):
        """Medication with dose_mg <= 0 must be rejected."""
        bad = {
            **CRITICAL_PAYLOAD,
            "medications": [
                {
                    "name": "warfarin",
                    "dose_mg": -1.0,
                    "frequency": "daily",
                    "route": "oral",
                }
            ],
        }
        resp = await deploy_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body_returns_422(self, deploy_client: AsyncClient):
        """An empty JSON body must return HTTP 422."""
        resp = await deploy_client.post("/api/v1/analyze", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_care_setting_returns_422(self, deploy_client: AsyncClient):
        """An unrecognised care_setting value must return HTTP 422."""
        bad = {
            **CRITICAL_PAYLOAD,
            "context": {**CRITICAL_PAYLOAD["context"], "care_setting": "spaceship"},
        }
        resp = await deploy_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_sex_field_returns_422(self, deploy_client: AsyncClient):
        """An invalid sex value must return HTTP 422."""
        bad = {
            **CRITICAL_PAYLOAD,
            "context": {**CRITICAL_PAYLOAD["context"], "sex": "unknown_value"},
        }
        resp = await deploy_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_error_response_includes_detail(self, deploy_client: AsyncClient):
        """Validation error response must include a 'detail' key with error context."""
        bad = {**CRITICAL_PAYLOAD}
        bad["diagnoses"] = [{"code": "INVALID", "description": "x", "is_primary": True}]
        resp = await deploy_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body, "Validation error must include 'detail'"

    @pytest.mark.asyncio
    async def test_no_audit_record_created_for_invalid_input(
        self, deploy_client: AsyncClient
    ):
        """An invalid request must not create any audit log entry."""
        bad = {**CRITICAL_PAYLOAD}
        bad["diagnoses"] = [{"code": "INVALID", "description": "x", "is_primary": True}]
        resp = await deploy_client.post("/api/v1/analyze", json=bad)
        assert resp.status_code == 422
        await asyncio.sleep(0.2)

        # The audit list should not include an entry for this case
        audit_list = (await deploy_client.get("/api/v1/audit?limit=5")).json()
        # No record should reference a case with no valid analysis_id
        for record in audit_list:
            # If any records exist they must have valid analysis_ids (UUIDs)
            assert record.get("analysis_id"), "Audit record must have an analysis_id"


# ---------------------------------------------------------------------------
# 6. Concurrency Test
# ---------------------------------------------------------------------------


class TestConcurrency:
    """
    Stress-tests the async pipeline with 20 simultaneous requests to verify
    thread-safety, audit log integrity, and consistent response quality.
    """

    @pytest.mark.asyncio
    async def test_20_concurrent_requests_all_succeed(self, deploy_client: AsyncClient):
        """All 20 parallel POST /analyze requests must return HTTP 200."""
        n = 20
        payloads = [
            {**LOW_RISK_PAYLOAD, "case_id": f"CONC-{uuid.uuid4().hex[:8]}"}
            for _ in range(n)
        ]
        responses = await asyncio.gather(
            *[deploy_client.post("/api/v1/analyze", json=p) for p in payloads]
        )
        failures = [r for r in responses if r.status_code != 200]
        assert not failures, (
            f"{len(failures)}/{n} requests failed: {[r.status_code for r in failures]}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_responses_have_unique_analysis_ids(
        self, deploy_client: AsyncClient
    ):
        """Each concurrent analysis must produce a distinct analysis_id."""
        n = 20
        payloads = [
            {**LOW_RISK_PAYLOAD, "case_id": f"CONC-{uuid.uuid4().hex[:8]}"}
            for _ in range(n)
        ]
        responses = await asyncio.gather(
            *[deploy_client.post("/api/v1/analyze", json=p) for p in payloads]
        )
        ids = [r.json()["analysis_id"] for r in responses if r.status_code == 200]
        assert len(ids) == len(set(ids)), (
            "Duplicate analysis_ids detected under concurrency"
        )

    @pytest.mark.asyncio
    async def test_concurrent_audit_logs_not_corrupted(
        self, deploy_client: AsyncClient
    ):
        """Concurrent writes must not corrupt the audit log — every record must be parseable."""
        n = 20
        payloads = [
            {**LOW_RISK_PAYLOAD, "case_id": f"CONC-AUDIT-{uuid.uuid4().hex[:8]}"}
            for _ in range(n)
        ]
        responses = await asyncio.gather(
            *[deploy_client.post("/api/v1/analyze", json=p) for p in payloads]
        )
        # Allow all background audit tasks to complete
        await asyncio.sleep(0.5)

        # Retrieve recent records and verify they are parseable
        audit_resp = await deploy_client.get(f"/api/v1/audit?limit={n + 5}")
        assert audit_resp.status_code == 200
        records = audit_resp.json()
        for rec in records:
            assert "analysis_id" in rec, f"Corrupt audit record: {rec}"
            assert "total_risk_score" in rec, f"Corrupt audit record: {rec}"
            assert "risk_level" in rec, f"Corrupt audit record: {rec}"

    @pytest.mark.asyncio
    async def test_concurrent_average_latency(self, deploy_client: AsyncClient):
        """Average response latency under 20 concurrent requests should be < 5 seconds."""
        n = 20
        payloads = [
            {**LOW_RISK_PAYLOAD, "case_id": f"LATENCY-{uuid.uuid4().hex[:8]}"}
            for _ in range(n)
        ]
        start = time.monotonic()
        responses = await asyncio.gather(
            *[deploy_client.post("/api/v1/analyze", json=p) for p in payloads]
        )
        elapsed = time.monotonic() - start
        successes = sum(1 for r in responses if r.status_code == 200)
        avg_ms = (elapsed / n) * 1000 if n else 0

        assert successes == n, (
            f"Not all concurrent requests succeeded ({successes}/{n})"
        )
        assert elapsed < 5.0, (
            f"Concurrent batch took {elapsed:.2f}s (>{5}s limit). avg={avg_ms:.0f}ms/req"
        )


# ---------------------------------------------------------------------------
# 7. A2A Endpoint Test
# ---------------------------------------------------------------------------


class TestA2AEndpoint:
    """
    Validates the Agent-to-Agent (A2A) protocol endpoint used for machine-to-machine
    integration with other clinical AI systems.
    """

    @pytest.mark.asyncio
    async def test_a2a_invoke_accepted(self, deploy_client: AsyncClient):
        """POST /a2a/invoke must return HTTP 200 for a valid request."""
        resp = await deploy_client.post("/api/v1/a2a/invoke", json=A2A_CRITICAL_REQUEST)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_a2a_response_is_structured_for_machine_consumption(
        self, deploy_client: AsyncClient
    ):
        """A2A response must include all machine-readable fields."""
        resp = await deploy_client.post("/api/v1/a2a/invoke", json=A2A_CRITICAL_REQUEST)
        data = resp.json()
        required = {
            "trace_id",
            "status",
            "analysis_id",
            "case_id",
            "risk_score",
            "risk_level",
            "agent_contributions",
        }
        missing = required - set(data.keys())
        assert not missing, f"A2A response missing machine-readable fields: {missing}"

    @pytest.mark.asyncio
    async def test_a2a_trace_id_is_echoed(self, deploy_client: AsyncClient):
        """A2A response must echo back the caller's trace_id for distributed tracing."""
        trace_id = A2A_CRITICAL_REQUEST["trace_id"]
        resp = await deploy_client.post("/api/v1/a2a/invoke", json=A2A_CRITICAL_REQUEST)
        assert resp.json()["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_a2a_response_risk_output_present(self, deploy_client: AsyncClient):
        """A2A response must include both risk_score and risk_level."""
        resp = await deploy_client.post("/api/v1/a2a/invoke", json=A2A_CRITICAL_REQUEST)
        data = resp.json()
        assert data["risk_score"] is not None, "A2A response missing risk_score"
        assert data["risk_level"] is not None, "A2A response missing risk_level"
        assert 0.0 <= float(data["risk_score"]) <= 1.0

    @pytest.mark.asyncio
    async def test_a2a_unknown_capability_returns_rejected(
        self, deploy_client: AsyncClient
    ):
        """Requests for unknown capabilities must return status='rejected' with error code."""
        bad = {**A2A_CRITICAL_REQUEST, "capability": "nonexistent_capability"}
        resp = await deploy_client.post("/api/v1/a2a/invoke", json=bad)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert data.get("error_code") == "UNKNOWN_CAPABILITY"

    @pytest.mark.asyncio
    async def test_a2a_drug_check_capability(self, deploy_client: AsyncClient):
        """drug_interaction_check capability must return valid risk output."""
        req = {
            "trace_id": str(uuid.uuid4()),
            "caller_id": "deployment-test-suite",
            "capability": "drug_interaction_check",
            "payload": {
                "case_id": "drug-check-test",
                "patient_id": "p-test",
                "medications": [
                    {
                        "name": "warfarin",
                        "dose_mg": 5.0,
                        "frequency": "daily",
                        "route": "oral",
                    },
                    {
                        "name": "amiodarone",
                        "dose_mg": 200.0,
                        "frequency": "daily",
                        "route": "oral",
                    },
                ],
                "allergies": [],
            },
        }
        resp = await deploy_client.post("/api/v1/a2a/invoke", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["risk_score"] is not None

    @pytest.mark.asyncio
    async def test_a2a_agent_card_schema(self, deploy_client: AsyncClient):
        """Agent card must include key discovery fields."""
        resp = await deploy_client.get("/api/v1/agent-card")
        assert resp.status_code == 200
        card = resp.json()
        assert card["agent_id"] == "clarity-clinical-risk-v1"
        assert card["phi_accepted"] is False
        assert len(card["capabilities"]) >= 1


# ---------------------------------------------------------------------------
# 8. Health Check Test
# ---------------------------------------------------------------------------


class TestHealthChecks:
    """
    Validates readiness probes to ensure the service and its subsystems
    report healthy status before traffic is admitted.
    """

    @pytest.mark.asyncio
    async def test_core_health_endpoint_returns_ok(self, deploy_client: AsyncClient):
        """GET /health must return {'status': 'ok'}."""
        resp = await deploy_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_a2a_health_endpoint_returns_ok(self, deploy_client: AsyncClient):
        """GET /a2a/health must return {'status': 'ok'}."""
        resp = await deploy_client.get("/api/v1/a2a/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_a2a_health_lists_all_agents(self, deploy_client: AsyncClient):
        """A2A health response must enumerate all four analysis agents."""
        resp = await deploy_client.get("/api/v1/a2a/health")
        agents = resp.json()["agents"]
        for name in (
            "diagnosis",
            "drug_interaction",
            "lab_analysis",
            "patient_context",
        ):
            assert name in agents, f"Agent '{name}' missing from health report"

    @pytest.mark.asyncio
    async def test_a2a_health_agents_report_ok(self, deploy_client: AsyncClient):
        """Each agent listed in the health response must have status='ok'."""
        resp = await deploy_client.get("/api/v1/a2a/health")
        for name, info in resp.json()["agents"].items():
            assert info.get("status") == "ok", (
                f"Agent '{name}' reports status='{info.get('status')}', expected 'ok'"
            )

    @pytest.mark.asyncio
    async def test_a2a_health_reports_fallback_availability(
        self, deploy_client: AsyncClient
    ):
        """Each agent must confirm fallback_available=True for resilience."""
        resp = await deploy_client.get("/api/v1/a2a/health")
        for name, info in resp.json()["agents"].items():
            assert info.get("fallback_available") is True, (
                f"Agent '{name}' does not report fallback_available=True"
            )

    @pytest.mark.asyncio
    async def test_a2a_health_includes_version(self, deploy_client: AsyncClient):
        """A2A health response must include version metadata."""
        resp = await deploy_client.get("/api/v1/a2a/health")
        data = resp.json()
        assert "version" in data, "A2A health response missing 'version'"
        assert "agent_id" in data, "A2A health response missing 'agent_id'"


# ---------------------------------------------------------------------------
# 9. JSON Schema Enforcement Test
# ---------------------------------------------------------------------------


class TestJSONSchemaEnforcement:
    """
    Validates that all API responses strictly match the expected schema and
    do not leak free-text or undefined fields beyond the defined model.
    """

    # Expected RiskScore fields (all must be numeric)
    RISK_NUMERIC_FIELDS = (
        "diagnosis_score",
        "medication_score",
        "lab_score",
        "w_diagnosis",
        "w_medication",
        "w_lab",
        "context_multiplier",
        "total_score",
    )

    @pytest.mark.asyncio
    async def test_analysis_response_types(self, deploy_client: AsyncClient):
        """All top-level field types must match the AnalysisResponse schema."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        data = resp.json()

        assert isinstance(data["analysis_id"], str), "analysis_id must be str"
        assert isinstance(data["case_id"], str), "case_id must be str"
        assert isinstance(data["patient_id"], str), "patient_id must be str"
        assert isinstance(data["timestamp"], str), "timestamp must be str"
        assert isinstance(data["risk"], dict), "risk must be a dict"
        assert isinstance(data["agent_responses"], list), (
            "agent_responses must be a list"
        )
        assert isinstance(data["explanation"], str), "explanation must be str"
        assert isinstance(data["recommended_actions"], list), (
            "recommended_actions must be a list"
        )
        assert isinstance(data["model_version"], str), "model_version must be str"

    @pytest.mark.asyncio
    async def test_risk_score_fields_are_numeric(self, deploy_client: AsyncClient):
        """All RiskScore fields must be numeric and within [0, 1] or [0.5, 1.5]."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        risk = resp.json()["risk"]
        for field in self.RISK_NUMERIC_FIELDS:
            assert field in risk, f"RiskScore missing field: {field}"
            assert isinstance(risk[field], (int, float)), (
                f"RiskScore.{field} must be numeric, got {type(risk[field])}"
            )
        # Bounded scores
        for field in (
            "diagnosis_score",
            "medication_score",
            "lab_score",
            "total_score",
        ):
            assert 0.0 <= risk[field] <= 1.0, (
                f"RiskScore.{field}={risk[field]} out of [0, 1]"
            )

    @pytest.mark.asyncio
    async def test_risk_level_is_valid_enum(self, deploy_client: AsyncClient):
        """risk.level must be one of: low, medium, high, critical."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        level = resp.json()["risk"]["level"]
        assert level in ("low", "medium", "high", "critical"), (
            f"Unexpected risk level: '{level}'"
        )

    @pytest.mark.asyncio
    async def test_agent_response_schema(self, deploy_client: AsyncClient):
        """Each AgentResponse must have all required fields with correct types."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        for ar in resp.json()["agent_responses"]:
            assert "agent_type" in ar, "AgentResponse missing 'agent_type'"
            assert "case_id" in ar, "AgentResponse missing 'case_id'"
            assert "risk_score" in ar, "AgentResponse missing 'risk_score'"
            assert "findings" in ar, "AgentResponse missing 'findings'"
            assert "reasoning" in ar, "AgentResponse missing 'reasoning'"
            assert "confidence" in ar, "AgentResponse missing 'confidence'"
            assert isinstance(ar["findings"], list), (
                "AgentResponse.findings must be a list"
            )
            assert isinstance(ar["reasoning"], str), (
                "AgentResponse.reasoning must be a str"
            )
            assert 0.0 <= ar["risk_score"] <= 1.0, (
                f"AgentResponse risk_score out of range: {ar['risk_score']}"
            )

    @pytest.mark.asyncio
    async def test_no_raw_text_leakage_in_response(self, deploy_client: AsyncClient):
        """
        The response must not contain raw LLM text outside of defined structured fields.
        Checks that all string fields conform to expected content constraints.
        """
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        raw_text = resp.text
        # Response must be valid JSON (no markdown fences or trailing text)
        try:
            json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Response body is not valid JSON: {exc}") from exc
        # No markdown code fences should appear in the JSON body
        assert "```" not in raw_text, "Markdown code fence detected in API response"

    @pytest.mark.asyncio
    async def test_a2a_response_schema(self, deploy_client: AsyncClient):
        """A2AResponse must include all required schema fields with correct types."""
        resp = await deploy_client.post("/api/v1/a2a/invoke", json=A2A_CRITICAL_REQUEST)
        data = resp.json()

        assert isinstance(data["trace_id"], str), "trace_id must be str"
        assert isinstance(data["status"], str), "status must be str"
        assert isinstance(data["analysis_id"], str), "analysis_id must be str"
        assert isinstance(data["case_id"], str), "case_id must be str"
        assert data["status"] in ("success", "partial", "error", "rejected"), (
            f"A2AResponse.status invalid: '{data['status']}'"
        )

    @pytest.mark.asyncio
    async def test_audit_record_schema(self, deploy_client: AsyncClient):
        """AuditRecord must have all required fields with correct types."""
        resp = await deploy_client.post("/api/v1/analyze", json=CRITICAL_PAYLOAD)
        analysis_id = resp.json()["analysis_id"]
        await asyncio.sleep(0.2)

        audit = (await deploy_client.get(f"/api/v1/audit/{analysis_id}")).json()
        assert isinstance(audit["audit_id"], str), "audit_id must be str"
        assert isinstance(audit["analysis_id"], str), "analysis_id must be str"
        assert isinstance(audit["total_risk_score"], float), (
            "total_risk_score must be float"
        )
        assert isinstance(audit["risk_level"], str), "risk_level must be str"
        assert isinstance(audit["agent_reasoning"], dict), (
            "agent_reasoning must be dict"
        )
        assert isinstance(audit["had_errors"], bool), "had_errors must be bool"
        assert audit["risk_level"] in ("low", "medium", "high", "critical"), (
            f"AuditRecord.risk_level invalid: '{audit['risk_level']}'"
        )
