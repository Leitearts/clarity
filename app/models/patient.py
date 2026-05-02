from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re

from app.safety.sanitize import sanitize_list, sanitize_text


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

    @field_validator("description")
    @classmethod
    def sanitize_description(cls, v: str) -> str:
        return sanitize_text(v, "diagnosis.description")


class Medication(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    dose_mg: float = Field(..., gt=0)
    frequency: str = Field(..., max_length=100)
    route: str = Field(default="oral", max_length=50)
    duration_days: Optional[int] = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        return sanitize_text(v, "medication.name")

    @field_validator("frequency")
    @classmethod
    def sanitize_frequency(cls, v: str) -> str:
        return sanitize_text(v, "medication.frequency")

    @field_validator("route")
    @classmethod
    def sanitize_route(cls, v: str) -> str:
        return sanitize_text(v, "medication.route")


class LabResult(BaseModel):
    test_name: str = Field(..., min_length=1, max_length=100)
    value: float
    unit: str = Field(..., min_length=1, max_length=30)
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    collected_hours_ago: Optional[int] = Field(default=None, ge=0)

    @field_validator("test_name")
    @classmethod
    def sanitize_test_name(cls, v: str) -> str:
        return sanitize_text(v, "lab_result.test_name")

    @field_validator("unit")
    @classmethod
    def sanitize_unit(cls, v: str) -> str:
        return sanitize_text(v, "lab_result.unit")

    @property
    def is_abnormal(self) -> bool:
        if self.reference_low is not None and self.value < self.reference_low:
            return True
        if self.reference_high is not None and self.value > self.reference_high:
            return True
        return False


class VitalSign(BaseModel):
    sign_name: str = Field(..., min_length=1, max_length=50)
    value: float
    unit: str = Field(..., min_length=1, max_length=30)
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    measured_minutes_ago: Optional[int] = Field(default=None, ge=0)

    @field_validator("sign_name")
    @classmethod
    def sanitize_sign_name(cls, v: str) -> str:
        return sanitize_text(v, "vital_sign.sign_name")

    @field_validator("unit")
    @classmethod
    def sanitize_unit(cls, v: str) -> str:
        return sanitize_text(v, "vital_sign.unit")

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

    @field_validator("allergies")
    @classmethod
    def sanitize_allergies(cls, v: list[str]) -> list[str]:
        return sanitize_list(v, "context.allergies")

    @field_validator("comorbidities")
    @classmethod
    def sanitize_comorbidities(cls, v: list[str]) -> list[str]:
        return sanitize_list(v, "context.comorbidities")


class PatientCase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient_id: str = Field(..., min_length=1, max_length=64)
    case_id: str = Field(..., min_length=1, max_length=64)
    context: PatientContext
    diagnoses: list[Diagnosis] = Field(..., min_length=1)
    medications: list[Medication] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    vitals_results: list[VitalSign] = Field(default_factory=list)
    clinical_notes: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("diagnoses")
    @classmethod
    def require_primary_diagnosis(cls, v: list[Diagnosis]) -> list[Diagnosis]:
        if not any(d.is_primary for d in v):
            v[0] = v[0].model_copy(update={"is_primary": True})
        return v

    @field_validator("clinical_notes")
    @classmethod
    def sanitize_clinical_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return sanitize_text(v, "clinical_notes")
