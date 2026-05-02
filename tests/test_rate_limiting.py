"""Tests for API rate limiting.

Rate limiting is disabled in the shared ``disable_rate_limiting`` autouse
fixture (conftest.py).  Tests in this file selectively re-enable it via the
``rl_client`` fixture factory, which sets a tight limit, clears the in-memory
storage, yields a test client, then restores all settings.

The limiter uses an in-memory backend (``MemoryStorage``), cleared via
``limiter._limiter.storage.reset()`` before and after each test so counters
from one test never bleed into another.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.config import settings
from app.main import app

CASES_PATH = Path(__file__).parent.parent / "data" / "synthetic_cases.json"
_RAW = json.loads(CASES_PATH.read_text())


def _payload(case_id: str) -> dict:
    for c in _RAW:
        if c["case_id"] == case_id:
            return {k: v for k, v in c.items() if not k.startswith("_")}
    raise KeyError(case_id)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset_limiter() -> None:
    """Reset all in-memory rate limit counters."""
    from app.api.limiter import limiter
    limiter._limiter.storage.reset()


def _make_app_state() -> None:
    """Populate app.state (mirrors the conftest async_client fixture)."""
    from app.a2a.adapter import A2AAdapter
    from app.audit.logger import AuditLogger
    from app.orchestrator.orchestrator import Orchestrator

    audit_logger = AuditLogger()
    orchestrator = Orchestrator()
    orchestrator._audit_logger = audit_logger

    app.state.orchestrator = orchestrator
    app.state.audit_logger = audit_logger
    app.state.a2a_adapter = A2AAdapter(orchestrator)


@pytest_asyncio.fixture
async def rl_analyze(mock_llm):
    """Client with rate limiting enabled and a "2/minute" analyze limit."""
    orig_enabled = settings.rate_limit_enabled
    orig_limit = settings.rate_limit_analyze
    settings.rate_limit_enabled = True
    settings.rate_limit_analyze = "2/minute"
    _reset_limiter()
    _make_app_state()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        settings.rate_limit_enabled = orig_enabled
        settings.rate_limit_analyze = orig_limit
        _reset_limiter()


@pytest_asyncio.fixture
async def rl_analyze_1(mock_llm):
    """Client with a "1/minute" analyze limit."""
    orig_enabled = settings.rate_limit_enabled
    orig_limit = settings.rate_limit_analyze
    settings.rate_limit_enabled = True
    settings.rate_limit_analyze = "1/minute"
    _reset_limiter()
    _make_app_state()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        settings.rate_limit_enabled = orig_enabled
        settings.rate_limit_analyze = orig_limit
        _reset_limiter()


@pytest_asyncio.fixture
async def rl_audit(mock_llm):
    """Client with a "2/minute" audit limit."""
    orig_enabled = settings.rate_limit_enabled
    orig_limit = settings.rate_limit_audit
    settings.rate_limit_enabled = True
    settings.rate_limit_audit = "2/minute"
    _reset_limiter()
    _make_app_state()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        settings.rate_limit_enabled = orig_enabled
        settings.rate_limit_audit = orig_limit
        _reset_limiter()


@pytest_asyncio.fixture
async def rl_a2a(mock_llm):
    """Client with a "1/minute" A2A limit."""
    orig_enabled = settings.rate_limit_enabled
    orig_limit = settings.rate_limit_a2a
    settings.rate_limit_enabled = True
    settings.rate_limit_a2a = "1/minute"
    _reset_limiter()
    _make_app_state()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        settings.rate_limit_enabled = orig_enabled
        settings.rate_limit_a2a = orig_limit
        _reset_limiter()


# ─────────────────────────────────────────────────────────────────────────────
# /analyze  — strict limit
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeRateLimit:

    @pytest.mark.asyncio
    async def test_requests_within_limit_succeed(self, rl_analyze):
        """Two requests within a "2/minute" limit should both return 200."""
        for _ in range(2):
            resp = await rl_analyze.post(
                "/api/v1/analyze", json=_payload("CASE-LOW-001")
            )
            assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_exceeding_limit_returns_429(self, rl_analyze):
        """The (n+1)th request over a "2/minute" limit must return 429."""
        for _ in range(2):
            resp = await rl_analyze.post(
                "/api/v1/analyze", json=_payload("CASE-LOW-001")
            )
            assert resp.status_code == 200

        resp = await rl_analyze.post(
            "/api/v1/analyze", json=_payload("CASE-LOW-001")
        )
        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_429_response_is_json(self, rl_analyze_1):
        """The 429 error body must be a JSON object with an 'error' key."""
        await rl_analyze_1.post("/api/v1/analyze", json=_payload("CASE-LOW-001"))
        resp = await rl_analyze_1.post(
            "/api/v1/analyze", json=_payload("CASE-LOW-001")
        )
        assert resp.status_code == 429
        body = resp.json()
        assert "error" in body


# ─────────────────────────────────────────────────────────────────────────────
# /audit  — moderate limit
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditRateLimit:

    @pytest.mark.asyncio
    async def test_audit_list_within_limit(self, rl_audit):
        """Two requests within a "2/minute" audit limit should succeed."""
        for _ in range(2):
            resp = await rl_audit.get("/api/v1/audit")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_audit_list_exceeds_limit_returns_429(self, rl_audit):
        """Exceeding the audit limit returns 429."""
        for _ in range(2):
            await rl_audit.get("/api/v1/audit")
        resp = await rl_audit.get("/api/v1/audit")
        assert resp.status_code == 429


# ─────────────────────────────────────────────────────────────────────────────
# /a2a/invoke  — configurable limit
# ─────────────────────────────────────────────────────────────────────────────

class TestA2ARateLimit:

    def _a2a_payload(self) -> dict:
        return {
            "trace_id": "rl-test-001",
            "caller_id": "pytest",
            "capability": "clinical_risk_analysis",
            "payload": _payload("CASE-LOW-001"),
        }

    @pytest.mark.asyncio
    async def test_a2a_within_limit_succeeds(self, rl_a2a):
        """First request within a "1/minute" limit should succeed."""
        resp = await rl_a2a.post("/api/v1/a2a/invoke", json=self._a2a_payload())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_a2a_exceeds_limit_returns_429(self, rl_a2a):
        """Exceeding the A2A limit returns 429."""
        await rl_a2a.post("/api/v1/a2a/invoke", json=self._a2a_payload())
        resp = await rl_a2a.post("/api/v1/a2a/invoke", json=self._a2a_payload())
        assert resp.status_code == 429


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimitConfiguration:

    def test_defaults_are_present(self):
        """Config must expose all rate limit fields."""
        assert hasattr(settings, "rate_limit_enabled")
        assert hasattr(settings, "rate_limit_analyze")
        assert hasattr(settings, "rate_limit_audit")
        assert hasattr(settings, "rate_limit_a2a")

    def test_default_values(self):
        """Verify the production default limit strings."""
        default_fields = settings.model_fields
        assert default_fields["rate_limit_analyze"].default == "10/minute"
        assert default_fields["rate_limit_audit"].default == "30/minute"
        assert default_fields["rate_limit_a2a"].default == "20/minute"

    def test_rate_limit_enabled_default_is_true(self):
        default = settings.model_fields["rate_limit_enabled"].default
        assert default is True

    @pytest.mark.asyncio
    async def test_disabled_rate_limiting_never_returns_429(self, mock_llm):
        """When rate_limit_enabled=False, no number of requests triggers 429.

        The autouse fixture already sets rate_limit_enabled=False.  We set a
        very tight limit string to make sure the key function truly bypasses
        any counter tracking.
        """
        orig_limit = settings.rate_limit_analyze
        settings.rate_limit_analyze = "1/minute"
        _reset_limiter()
        _make_app_state()
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                for _ in range(5):
                    resp = await client.post(
                        "/api/v1/analyze", json=_payload("CASE-LOW-001")
                    )
                    assert resp.status_code == 200, (
                        f"Unexpected {resp.status_code} on request (rate limiting "
                        "should be disabled)"
                    )
        finally:
            settings.rate_limit_analyze = orig_limit
            _reset_limiter()

