from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class AgentType(str, Enum):
    DIAGNOSIS = "diagnosis"
    DRUG_INTERACTION = "drug_interaction"
    LAB_ANALYSIS = "lab_analysis"
    PATIENT_CONTEXT = "patient_context"


class FindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Finding(BaseModel):
    code: str = Field(...)
    description: str = Field(..., min_length=5, max_length=500)
    severity: FindingSeverity
    evidence: Optional[str] = Field(default=None, max_length=300)
    affected_items: list[str] = Field(default_factory=list)


class AgentRequest(BaseModel):
    agent_type: AgentType
    case_id: str
    patient_id: str
    payload: dict[str, Any] = Field(...)
    context_modifiers: dict[str, float] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    agent_type: AgentType
    case_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    findings: list[Finding] = Field(default_factory=list)
    reasoning: str = Field(..., min_length=10, max_length=1000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)

    @field_validator("risk_score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))
