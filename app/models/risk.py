from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, computed_field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentContribution(BaseModel):
    agent_type: str
    raw_score: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    weighted_score: float = Field(..., ge=0.0, le=1.0)


class RiskScore(BaseModel):
    diagnosis_score: float = Field(..., ge=0.0, le=1.0)
    medication_score: float = Field(..., ge=0.0, le=1.0)
    lab_score: float = Field(..., ge=0.0, le=1.0)
    vitals_score: float = Field(..., ge=0.0, le=1.0)
    w_diagnosis: float = Field(default=0.30)
    w_medication: float = Field(default=0.25)
    w_lab: float = Field(default=0.20)
    w_vitals: float = Field(default=0.25)
    context_multiplier: float = Field(default=1.0, ge=0.5, le=1.5)

    @computed_field
    @property
    def total_score(self) -> float:
        raw = (
            self.w_diagnosis * self.diagnosis_score
            + self.w_medication * self.medication_score
            + self.w_lab * self.lab_score
            + self.w_vitals * self.vitals_score
        )
        return round(min(1.0, raw * self.context_multiplier), 4)

    @computed_field
    @property
    def level(self) -> RiskLevel:
        from app.config import settings
        s = self.total_score
        if s >= settings.threshold_critical:
            return RiskLevel.CRITICAL
        if s >= settings.threshold_high:
            return RiskLevel.HIGH
        if s >= settings.threshold_low:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @computed_field
    @property
    def contributions(self) -> list[AgentContribution]:
        return [
            AgentContribution(agent_type="diagnosis", raw_score=self.diagnosis_score,
                              weight=self.w_diagnosis,
                              weighted_score=round(self.w_diagnosis * self.diagnosis_score, 4)),
            AgentContribution(agent_type="drug_interaction", raw_score=self.medication_score,
                              weight=self.w_medication,
                              weighted_score=round(self.w_medication * self.medication_score, 4)),
            AgentContribution(agent_type="lab_analysis", raw_score=self.lab_score,
                              weight=self.w_lab,
                              weighted_score=round(self.w_lab * self.lab_score, 4)),
        ]
