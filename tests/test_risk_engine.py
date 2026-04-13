import pytest
from app.risk.engine import RiskEngine, WEIGHT_PRESETS
from app.models.risk import RiskLevel


class TestRiskEngine:
    def test_formula_correctness(self):
        engine = RiskEngine()
        score = engine.score(0.8, 0.6, 0.4, 1.0)
        expected = round(0.4 * 0.8 + 0.35 * 0.6 + 0.25 * 0.4, 4)
        assert abs(score.total_score - expected) < 0.0001

    def test_context_multiplier_applied(self):
        engine = RiskEngine()
        base = engine.score(0.5, 0.5, 0.5, 1.0)
        scaled = engine.score(0.5, 0.5, 0.5, 1.2)
        assert abs(scaled.total_score - min(1.0, base.total_score * 1.2)) < 0.001

    def test_score_never_exceeds_1(self):
        engine = RiskEngine()
        score = engine.score(1.0, 1.0, 1.0, 1.5)
        assert score.total_score <= 1.0

    def test_score_never_below_0(self):
        engine = RiskEngine()
        score = engine.score(0.0, 0.0, 0.0, 0.5)
        assert score.total_score >= 0.0

    def test_critical_threshold(self):
        engine = RiskEngine()
        score = engine.score(1.0, 1.0, 1.0, 1.0)
        assert score.level == RiskLevel.CRITICAL

    def test_low_threshold(self):
        engine = RiskEngine()
        score = engine.score(0.1, 0.1, 0.1, 1.0)
        assert score.level == RiskLevel.LOW

    def test_weights_must_sum_to_1(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            RiskEngine(w_diagnosis=0.5, w_medication=0.5, w_lab=0.5)

    def test_all_presets_valid(self):
        for name in WEIGHT_PRESETS:
            engine = RiskEngine(preset=name)
            score = engine.score(0.5, 0.5, 0.5, 1.0)
            assert 0.0 <= score.total_score <= 1.0

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            RiskEngine(preset="nonexistent")

    def test_sensitivity_dominant_factor(self):
        engine = RiskEngine()
        analysis = engine.sensitivity_analysis(0.9, 0.1, 0.1, 1.0)
        assert analysis["dominant_factor"] == "diagnosis"

    def test_contributions_sum_to_weighted_sum(self):
        engine = RiskEngine()
        _, breakdown = engine.score_with_breakdown(0.7, 0.5, 0.4, 1.1)
        contrib_sum = round(
            breakdown.diagnosis_contribution
            + breakdown.medication_contribution
            + breakdown.lab_contribution,
            4,
        )
        assert abs(contrib_sum - breakdown.weighted_sum) < 0.0001

    def test_simulate_scenarios_returns_all_presets(self):
        engine = RiskEngine()
        results = engine.simulate_scenarios(0.7, 0.5, 0.4)
        preset_names = {r["preset"] for r in results}
        assert preset_names == set(WEIGHT_PRESETS.keys())
