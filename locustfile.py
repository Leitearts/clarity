"""
CLARITY Load Test — Locust
===========================
Simulates sustained production traffic against the CLARITY API to measure:
  - Response time (target p95 < 3 000 ms)
  - Failure rate  (target < 1 %)
  - Throughput    (target ≥ 5 RPS with 10 concurrent users)

Usage
-----
Install Locust:
    pip install locust

Run interactively (opens the web UI at http://localhost:8089):
    locust -f locustfile.py --host http://localhost:8000

Run headless (10 users, spawn rate 2/s, run for 60 s):
    locust -f locustfile.py --host http://localhost:8000 \
           --headless -u 10 -r 2 --run-time 60s \
           --html load_report.html

CSV output:
    locust -f locustfile.py --host http://localhost:8000 \
           --headless -u 10 -r 2 --run-time 60s \
           --csv clarity_load

Notes
-----
- All payloads use synthetic patient data only. No PHI.
- The high-risk payload exercises the full multi-agent pipeline.
- The drug-check payload exercises the lightweight A2A capability.
- Think times are randomised between 0.5 s and 2 s to simulate realistic pacing.
"""

from __future__ import annotations

import json
import random
import uuid

from locust import HttpUser, between, task


# ---------------------------------------------------------------------------
# Reusable synthetic payloads
# ---------------------------------------------------------------------------

def _critical_payload() -> dict:
    """High-risk patient — exercises all four agents + audit log."""
    return {
        "patient_id": f"LOAD-{uuid.uuid4().hex[:8]}",
        "case_id":    f"LOAD-CRITICAL-{uuid.uuid4().hex[:8]}",
        "context": {
            "age": 78,
            "sex": "male",
            "weight_kg": 80.0,
            "care_setting": "icu",
            "comorbidities": ["heart failure", "chronic kidney disease", "diabetes"],
            "allergies": ["sulfa"],
        },
        "diagnoses": [
            {"code": "I21.9", "description": "Acute MI, unspecified",    "is_primary": True,  "onset_days": 1},
            {"code": "I50.9", "description": "Heart failure, unspecified", "is_primary": False},
        ],
        "medications": [
            {"name": "warfarin",   "dose_mg": 5.0,   "frequency": "once daily", "route": "oral"},
            {"name": "amiodarone", "dose_mg": 200.0, "frequency": "once daily", "route": "oral"},
            {"name": "aspirin",    "dose_mg": 81.0,  "frequency": "once daily", "route": "oral"},
            {"name": "lisinopril", "dose_mg": 10.0,  "frequency": "once daily", "route": "oral"},
            {"name": "metformin",  "dose_mg": 500.0, "frequency": "once daily", "route": "oral"},
        ],
        "lab_results": [
            {"test_name": "troponin",   "value": 2.1, "unit": "ng/mL",  "reference_low": 0.0,  "reference_high": 0.04},
            {"test_name": "INR",        "value": 4.8, "unit": "ratio",  "reference_low": 0.8,  "reference_high": 1.2},
            {"test_name": "hemoglobin", "value": 6.8, "unit": "g/dL",   "reference_low": 13.5, "reference_high": 17.5},
        ],
    }


def _low_risk_payload() -> dict:
    """Low-risk patient — lightweight request for baseline throughput measurement."""
    return {
        "patient_id": f"LOAD-LOW-{uuid.uuid4().hex[:8]}",
        "case_id":    f"LOAD-LOW-{uuid.uuid4().hex[:8]}",
        "context": {
            "age": random.randint(20, 60),
            "sex": random.choice(["male", "female"]),
            "care_setting": "outpatient",
            "comorbidities": [],
            "allergies": [],
        },
        "diagnoses": [
            {"code": "J06.9", "description": "Acute upper respiratory infection", "is_primary": True},
        ],
        "medications": [],
        "lab_results": [],
    }


def _a2a_drug_check_payload() -> dict:
    """A2A drug-interaction check — exercises lightweight A2A code path."""
    return {
        "trace_id":   str(uuid.uuid4()),
        "caller_id":  "locust-load-test",
        "capability": "drug_interaction_check",
        "payload": {
            "case_id":    f"dc-{uuid.uuid4().hex[:8]}",
            "patient_id": f"pt-{uuid.uuid4().hex[:8]}",
            "medications": [
                {"name": "warfarin",   "dose_mg": 5.0,   "frequency": "daily", "route": "oral"},
                {"name": "amiodarone", "dose_mg": 200.0, "frequency": "daily", "route": "oral"},
            ],
            "allergies": [],
        },
    }


# ---------------------------------------------------------------------------
# Locust user classes
# ---------------------------------------------------------------------------

class ClarityUser(HttpUser):
    """
    Simulates a clinical workstation submitting patient cases for risk analysis.

    Task weights:
      - 60 % full analysis (high-risk)        → tests full pipeline under load
      - 25 % full analysis (low-risk)         → tests lightweight throughput
      - 15 % A2A drug-interaction check       → tests machine-to-machine path
    """

    wait_time = between(0.5, 2.0)   # seconds between tasks per virtual user

    # ── Health probe ─────────────────────────────────────────────────────────

    def on_start(self) -> None:
        """Verify the service is healthy before the user starts sending requests."""
        with self.client.get("/api/v1/health", catch_response=True, name="/health (on_start)") as resp:
            if resp.status_code != 200 or resp.json().get("status") != "ok":
                resp.failure(f"Health check failed: {resp.status_code} {resp.text}")

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task(6)
    def analyze_critical_patient(self) -> None:
        """POST /api/v1/analyze with a high-risk patient payload."""
        payload = _critical_payload()
        with self.client.post(
            "/api/v1/analyze",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/analyze (critical)",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
                return
            data = resp.json()
            if "risk" not in data or "total_score" not in data.get("risk", {}):
                resp.failure("Response missing risk.total_score")
                return
            if data["risk"]["total_score"] < 0.0 or data["risk"]["total_score"] > 1.0:
                resp.failure(f"risk.total_score out of bounds: {data['risk']['total_score']}")

    @task(2)
    def analyze_low_risk_patient(self) -> None:
        """POST /api/v1/analyze with a low-risk patient payload."""
        payload = _low_risk_payload()
        with self.client.post(
            "/api/v1/analyze",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/analyze (low-risk)",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")

    @task(2)
    def a2a_drug_check(self) -> None:
        """POST /api/v1/a2a/invoke with drug_interaction_check capability."""
        payload = _a2a_drug_check_payload()
        with self.client.post(
            "/api/v1/a2a/invoke",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/a2a/invoke (drug_check)",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
                return
            data = resp.json()
            if data.get("status") not in ("success", "partial"):
                resp.failure(f"Unexpected A2A status: {data.get('status')}")

    @task(1)
    def health_check(self) -> None:
        """GET /api/v1/health — lightweight probe included in traffic mix."""
        with self.client.get(
            "/api/v1/health",
            name="/health",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check returned {resp.status_code}")

    @task(1)
    def a2a_health_check(self) -> None:
        """GET /api/v1/a2a/health — verify agent subsystems are reported healthy."""
        with self.client.get(
            "/api/v1/a2a/health",
            name="/a2a/health",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"A2A health returned {resp.status_code}")
                return
            agents = resp.json().get("agents", {})
            for name, info in agents.items():
                if info.get("status") != "ok":
                    resp.failure(f"Agent '{name}' not healthy: {info}")
                    return
