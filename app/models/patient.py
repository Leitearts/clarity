from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_config
import re


class Diagnosis(BaseModel):
    code: str = Field(..., description="ICD-10 code, e.g. 'E11.9'")
    description: str = Field(..., min_length=2, max_length=200)
    is_primary: bool = Field(default=False)
    onset_days: Optional[int] = Field(default=None, ge=0)

    @field_validator("code")
    @classmethod
    def validate_icd10(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.match(r"^[A-Z]\d{2}(\.\d{1,4})?$", v):
            raise ValueError(f"Invalid ICD-10 code: '{v}'. Expected format: A00 or A00.0")
        return v


class Medication(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    dose_mg: float = Field(..., gt=0)
    frequency: str = Field(...)
    route: str = Field(default="oral")
    duration_days: Optional[int] = Field(default=None, ge=0)


class LabResult(BaseModel):
    test_name: str = Field(..., min_length=1, max_length=100)
    value: float
    unit: str = Field(..., min_length=1, max_length=30)
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    collected_hours_ago: Optional[int] = Field(default=None, ge=0)

    @property
    def is_abnormal(self) -> bool:
        if self.reference_low is not None and self.value < self.reference_low:
            return True
        if self.reference_high is not None and self.value > self.reference_high:
            return True
        return False


class PatientContext(BaseModel):
    age: int = Field(..., ge=0, le=130)
    sex: str = Field(..., pattern="^(male|female|other)$")
    weight_kg: Optional[float] = Field(default=None, gt=0, le=500)
    allergies: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    care_setting: str = Field(
        default="inpatient",
        pattern="^(inpatient|outpatient|icu|emergency|primary_care)$"
    )


class PatientCase(BaseModel):
    model_config = model_config(extra="ignore")

    patient_id: str = Field(..., min_length=1, max_length=64)
    case_id: str = Field(..., min_length=1, max_length=64)
    context: PatientContext
    diagnoses: list[Diagnosis] = Field(..., min_length=1)
    medications: list[Medication] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    clinical_notes: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("diagnoses")
    @classmethod
    def require_primary_diagnosis(cls, v: list[Diagnosis]) -> list[Diagnosis]:
        if not any(d.is_primary for d in v):
            v[0] = v[0].model_copy(update={"is_primary": True})
        return v
