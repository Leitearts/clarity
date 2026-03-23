from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
import uuid
from pydantic import BaseModel, Field, computed_field
from app.models.agent import AgentResponse, FindingSeverity
from app.models.risk import RiskScore


class AnalysisResponse(BaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    risk: RiskScore
    agent_responses: list[AgentResponse]
    explanation: str = Field(...)
    explanation_structured: Optional[dict[str, Any]] = None
    recommended_actions: list[str] = Field(default_factory=list)
    processing_time_ms: Optional[int] = None
    model_version: str = Field(default="clarity-v1.0")

    @computed_field
    @property
    def critical_findings(self) -> list[str]:
        results = []
        for ar in self.agent_responses:
            for f in ar.findings:
                if f.severity.value in ("critical", "high"):
                    results.append(f"[{ar.agent_type.value.upper()}] {f.description}")
        return results


class AuditRecord(BaseModel):
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_id: str
    case_id: str
    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_diagnosis_count: int
    input_medication_count: int
    input_lab_count: int
    diagnosis_score: float
    medication_score: float
    lab_score: float
    context_multiplier: float
    total_risk_score: float
    risk_level: str
    agent_reasoning: dict[str, str]
    model_version: str
    processing_time_ms: Optional[int] = None
    had_errors: bool = Field(default=False)
    error_details: Optional[str] = None
