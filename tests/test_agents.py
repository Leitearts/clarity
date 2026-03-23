from __future__ import annotations
import pytest
from app.agents.diagnosis import DiagnosisAgent
from app.agents.drug_interaction import DrugInteractionAgent
from app.agents.lab_analysis import LabAnalysisAgent
from app.agents.patient_context import PatientContextAgent
from app.models.agent import AgentRequest, AgentType, FindingSeverity
from app.models.patient import Diagnosis


def make_request(agent_type: AgentType, payload: dict) -> AgentRequest:
    return AgentRequest(agent_type=agent_type, case_id="TEST-001",
                        patient_id="SYNTH-TEST", payload=payload)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagnosisAgent:

    @pytest.mark.asyncio
    async def test_high_acuity_ami_scores_above_0_5(self, mock_llm):
        agent = DiagnosisAgent()
        request = make_request(AgentType.DIAGNOSIS, {"diagnoses": [
            {"code": "I21.9", "description": "Acute MI", "is_primary": True, "onset_days": 1}
        ]})
        response = await agent.run(request)
        assert response.risk_score > 0.5
        assert response.error is None

    @pytest.mark.asyncio
    async def test_sepsis_produces_high_acuity_finding(self, mock_llm):
        agent = DiagnosisAgent()
        request = make_request(AgentType.DIAGNOSIS, {"diagnoses": [
            {"code": "A41.9", "description": "Sepsis, unspecified", "is_primary": True}
        ]})
        response = await agent.run(request)
        high_or_critical = [f for f in response.findings
                            if f.severity in (FindingSeverity.HIGH, FindingSeverity.CRITICAL)]
        assert len(high_or_critical) >= 1

    @pytest.mark.asyncio
    async def test_routine_diagnosis_scores_low(self, mock_llm):
        agent = DiagnosisAgent()
        request = make_request(AgentType.DIAGNOSIS, {"diagnoses": [
            {"code": "J06.9", "description": "Upper respiratory infection", "is_primary": True}
        ]})
        response = await agent.run(request)
        assert response.risk_score < 0.35

    @pytest.mark.asyncio
    async def test_rapid_onset_increases_score(self, mock_llm):
        agent = DiagnosisAgent()
        slow = make_request(AgentType.DIAGNOSIS, {"diagnoses": [
            {"code": "I21.9", "description": "Acute MI", "is_primary": True, "onset_days": 30}
        ]})
        fast = make_request(AgentType.DIAGNOSIS, {"diagnoses": [
            {"code": "I21.9", "description": "Acute MI", "is_primary": True, "onset_days": 1}
        ]})
        r_slow = await agent.run(slow)
        r_fast = await agent.run(fast)
        assert r_fast.risk_score >= r_slow.risk_score

    @pytest.mark.asyncio
    async def test_invalid_icd_code_is_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Diagnosis(code="INVALID", description="Bad code", is_primary=True)

    @pytest.mark.asyncio
    async def test_agent_returns_valid_response_structure(self, mock_llm):
        agent = DiagnosisAgent()
        request = make_request(AgentType.DIAGNOSIS, {"diagnoses": [
            {"code": "E11.9", "description": "Type 2 diabetes", "is_primary": True}
        ]})
        response = await agent.run(request)
        assert 0.0 <= response.risk_score <= 1.0
        assert response.agent_type == AgentType.DIAGNOSIS
        assert response.case_id == "TEST-001"
        assert isinstance(response.findings, list)
        assert len(response.reasoning) >= 10


# ══════════════════════════════════════════════════════════════════════════════
# DRUG INTERACTION AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestDrugInteractionAgent:

    @pytest.mark.asyncio
    async def test_warfarin_amiodarone_is_critical(self, mock_llm):
        agent = DrugInteractionAgent()
        request = make_request(AgentType.DRUG_INTERACTION, {
            "medications": [
                {"name": "warfarin",   "dose_mg": 5.0,   "frequency": "once daily", "route": "oral"},
                {"name": "amiodarone", "dose_mg": 200.0, "frequency": "once daily", "route": "oral"},
            ],
            "allergies": [], "diagnoses": [],
        })
        response = await agent.run(request)
        critical = [f for f in response.findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1
        assert response.risk_score > 0.5

    @pytest.mark.asyncio
    async def test_allergy_conflict_is_critical(self, mock_llm):
        agent = DrugInteractionAgent()
        request = make_request(AgentType.DRUG_INTERACTION, {
            "medications": [
                {"name": "amoxicillin", "dose_mg": 875.0, "frequency": "twice daily", "route": "oral"}
            ],
            "allergies": ["penicillin", "amoxicillin"], "diagnoses": [],
        })
        response = await agent.run(request)
        allergy_findings = [f for f in response.findings if "allergy" in f.code.lower()]
        assert len(allergy_findings) >= 1
        assert allergy_findings[0].severity == FindingSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_polypharmacy_flagged_at_5_plus_drugs(self, mock_llm):
        agent = DrugInteractionAgent()
        meds = [{"name": f"drug_{i}", "dose_mg": 10.0, "frequency": "once daily", "route": "oral"}
                for i in range(5)]
        request = make_request(AgentType.DRUG_INTERACTION,
                               {"medications": meds, "allergies": [], "diagnoses": []})
        response = await agent.run(request)
        poly = [f for f in response.findings if "POLYPHARMACY" in f.code]
        assert len(poly) >= 1

    @pytest.mark.asyncio
    async def test_no_medications_returns_minimal_risk(self, mock_llm):
        agent = DrugInteractionAgent()
        request = make_request(AgentType.DRUG_INTERACTION,
                               {"medications": [], "allergies": [], "diagnoses": []})
        response = await agent.run(request)
        assert response.risk_score < 0.15
        assert response.error is None

    @pytest.mark.asyncio
    async def test_ssri_tramadol_serotonin_risk(self, mock_llm):
        agent = DrugInteractionAgent()
        request = make_request(AgentType.DRUG_INTERACTION, {
            "medications": [
                {"name": "fluoxetine", "dose_mg": 20.0, "frequency": "once daily", "route": "oral"},
                {"name": "tramadol",   "dose_mg": 50.0, "frequency": "PRN",        "route": "oral"},
            ],
            "allergies": [], "diagnoses": [],
        })
        response = await agent.run(request)
        serotonin = [f for f in response.findings
                     if "serotonin" in f.description.lower() or f.severity == FindingSeverity.CRITICAL]
        assert len(serotonin) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# LAB ANALYSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestLabAnalysisAgent:

    @pytest.mark.asyncio
    async def test_critical_troponin_scores_high(self, mock_llm):
        agent = LabAnalysisAgent()
        request = make_request(AgentType.LAB_ANALYSIS, {
            "lab_results": [
                {"test_name": "troponin", "value": 2.1, "unit": "ng/mL",
                 "reference_low": 0.0, "reference_high": 0.04}
            ],
            "context": {"age": 65, "sex": "male", "care_setting": "icu",
                        "comorbidities": [], "allergies": []},
        })
        response = await agent.run(request)
        critical = [f for f in response.findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1
        assert response.risk_score > 0.5

    @pytest.mark.asyncio
    async def test_critical_potassium_low_flagged(self, mock_llm):
        agent = LabAnalysisAgent()
        request = make_request(AgentType.LAB_ANALYSIS, {
            "lab_results": [
                {"test_name": "potassium", "value": 2.3, "unit": "mEq/L",
                 "reference_low": 3.5, "reference_high": 5.0}
            ],
            "context": {"age": 50, "sex": "female", "care_setting": "inpatient",
                        "comorbidities": [], "allergies": []},
        })
        response = await agent.run(request)
        critical = [f for f in response.findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    @pytest.mark.asyncio
    async def test_normal_labs_score_near_zero(self, mock_llm):
        agent = LabAnalysisAgent()
        request = make_request(AgentType.LAB_ANALYSIS, {
            "lab_results": [
                {"test_name": "hemoglobin", "value": 14.5, "unit": "g/dL",
                 "reference_low": 13.5, "reference_high": 17.5},
                {"test_name": "potassium",  "value": 4.1,  "unit": "mEq/L",
                 "reference_low": 3.5,  "reference_high": 5.0},
            ],
            "context": {"age": 35, "sex": "male", "care_setting": "outpatient",
                        "comorbidities": [], "allergies": []},
        })
        response = await agent.run(request)
        assert response.risk_score < 0.20

    @pytest.mark.asyncio
    async def test_multiple_critical_labs_compound_score(self, mock_llm):
        agent = LabAnalysisAgent()
        ctx = {"age": 65, "sex": "male", "care_setting": "icu", "comorbidities": [], "allergies": []}
        single = make_request(AgentType.LAB_ANALYSIS, {
            "lab_results": [
                {"test_name": "troponin", "value": 2.1, "unit": "ng/mL",
                 "reference_low": 0.0, "reference_high": 0.04}
            ], "context": ctx,
        })
        double = make_request(AgentType.LAB_ANALYSIS, {
            "lab_results": [
                {"test_name": "troponin",  "value": 2.1, "unit": "ng/mL",
                 "reference_low": 0.0, "reference_high": 0.04},
                {"test_name": "potassium", "value": 6.8, "unit": "mEq/L",
                 "reference_low": 3.5, "reference_high": 5.0},
            ], "context": ctx,
        })
        r_single = await agent.run(single)
        r_double = await agent.run(double)
        assert r_double.risk_score >= r_single.risk_score

    @pytest.mark.asyncio
    async def test_empty_labs_returns_minimal_risk(self, mock_llm):
        agent = LabAnalysisAgent()
        request = make_request(AgentType.LAB_ANALYSIS, {
            "lab_results": [],
            "context": {"age": 40, "sex": "male", "care_setting": "outpatient",
                        "comorbidities": [], "allergies": []},
        })
        response = await agent.run(request)
        assert response.risk_score < 0.15
        assert response.error is None


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT CONTEXT AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestPatientContextAgent:

    @pytest.mark.asyncio
    async def test_elderly_icu_multiplier_above_1(self, mock_llm):
        agent = PatientContextAgent()
        request = make_request(AgentType.PATIENT_CONTEXT, {"context": {
            "age": 78, "sex": "male", "care_setting": "icu",
            "comorbidities": ["heart failure", "chronic kidney disease", "diabetes"],
            "allergies": [],
        }})
        response = await agent.run(request)
        multiplier = response.metadata.get("context_multiplier", 1.0)
        assert multiplier > 1.0

    @pytest.mark.asyncio
    async def test_young_healthy_multiplier_is_1(self, mock_llm):
        agent = PatientContextAgent()
        request = make_request(AgentType.PATIENT_CONTEXT, {"context": {
            "age": 28, "sex": "female", "care_setting": "primary_care",
            "comorbidities": [], "allergies": [],
        }})
        response = await agent.run(request)
        multiplier = response.metadata.get("context_multiplier", 1.0)
        assert multiplier == 1.0

    @pytest.mark.asyncio
    async def test_multiplier_clamped_to_1_5(self, mock_llm):
        agent = PatientContextAgent()
        request = make_request(AgentType.PATIENT_CONTEXT, {"context": {
            "age": 95, "sex": "male", "care_setting": "icu",
            "comorbidities": ["heart failure", "ckd", "diabetes", "malignancy",
                              "cirrhosis", "hiv", "copd"],
            "allergies": [],
        }})
        response = await agent.run(request)
        multiplier = response.metadata.get("context_multiplier", 1.0)
        assert multiplier <= 1.5

    @pytest.mark.asyncio
    async def test_context_agent_risk_score_is_zero(self, mock_llm):
        agent = PatientContextAgent()
        request = make_request(AgentType.PATIENT_CONTEXT, {"context": {
            "age": 60, "sex": "male", "care_setting": "inpatient",
            "comorbidities": ["hypertension"], "allergies": [],
        }})
        response = await agent.run(request)
        assert response.risk_score == 0.0
