from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid
from pydantic import BaseModel, Field


class A2AStatus(str, Enum):
    SUCCESS  = "success"
    PARTIAL  = "partial"
    ERROR    = "error"
    REJECTED = "rejected"


class A2ACapability(BaseModel):
    name: str
    description: str
    input_fields: list[str]
    output_fields: list[str]


class AgentCard(BaseModel):
    agent_id: str = "clarity-clinical-risk-v1"
    name: str = "CLARITY"
    version: str = "1.0.0"
    description: str = (
        "Multi-agent clinical risk detection system. "
        "Analyzes diagnoses, medications, and lab results. "
        "Synthetic/de-identified data only — no PHI."
    )
    author: str = "CLARITY Team"
    license: str = "MIT"
    invocation_endpoint: str = "/api/v1/a2a/invoke"
    invocation_method: str = "POST"
    health_endpoint: str = "/api/v1/health"
    schema_version: str = "1.0"
    capabilities: list[A2ACapability] = Field(default_factory=lambda: [
        A2ACapability(
            name="clinical_risk_analysis",
            description="Full multi-agent risk analysis: diagnosis, drug, lab, context.",
            input_fields=["patient_id", "case_id", "context", "diagnoses", "medications", "lab_results"],
            output_fields=["risk_score", "risk_level", "findings", "explanation", "agent_breakdown"],
        ),
        A2ACapability(
            name="drug_interaction_check",
            description="Drug-drug interaction and allergy conflict screening.",
            input_fields=["medications", "allergies"],
            output_fields=["interactions", "allergy_conflicts", "risk_score"],
        ),
        A2ACapability(
            name="lab_interpretation",
            description="Lab result interpretation against reference ranges.",
            input_fields=["lab_results", "patient_age", "patient_sex"],
            output_fields=["critical_values", "abnormal_values", "risk_score"],
        ),
    ])
    max_diagnoses: int = 20
    max_medications: int = 50
    max_lab_results: int = 100
    supported_icd_versions: list[str] = ["ICD-10"]
    data_policy: str = "synthetic_only"
    phi_accepted: bool = False
    expected_latency_ms: int = 3000
    timeout_ms: int = 30000
    tags: list[str] = ["healthcare", "clinical-decision-support", "risk-scoring",
                       "drug-safety", "lab-analysis", "explainable-ai"]


class A2ARequest(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    caller_id: str = Field(..., min_length=1, max_length=128)
    schema_version: str = Field(default="1.0")
    capability: str = Field(default="clinical_risk_analysis")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    idempotency_key: Optional[str] = None
    payload: dict[str, Any] = Field(...)


class AgentContributionSummary(BaseModel):
    agent: str
    risk_score: float
    weight: float
    contribution_pct: float
    top_finding: Optional[str] = None
    had_error: bool = False


class A2AResponse(BaseModel):
    trace_id: str
    schema_version: str = "1.0"
    status: A2AStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_id: str
    case_id: str
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    narrative: Optional[str] = None
    formula_trace: Optional[str] = None
    agent_contributions: list[AgentContributionSummary] = Field(default_factory=list)
    critical_findings: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    full_result: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None
