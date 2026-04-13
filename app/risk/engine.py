from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import NamedTuple
from app.config import settings
from app.models.risk import RiskLevel, RiskScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightPreset:
    name: str
    w_diagnosis: float
    w_medication: float
    w_lab: float
    w_vitals: float
    description: str


WEIGHT_PRESETS: dict[str, WeightPreset] = {
    "default": WeightPreset(
        "default", 0.30, 0.25, 0.20, 0.25, "Balanced weights for general inpatient use."
    ),
    "icu": WeightPreset(
        "icu", 0.25, 0.20, 0.30, 0.25, "ICU: lab results weighted higher."
    ),
    "outpatient": WeightPreset(
        "outpatient",
        0.40,
        0.30,
        0.15,
        0.15,
        "Outpatient: diagnosis dominant, labs often stale.",
    ),
    "pharmacy_review": WeightPreset(
        "pharmacy_review",
        0.20,
        0.50,
        0.15,
        0.15,
        "Pharmacy-focused: medication safety dominant.",
    ),
    "emergency": WeightPreset(
        "emergency",
        0.25,
        0.20,
        0.30,
        0.25,
        "ED: acute labs most time-sensitive. Vitals critical.",
    ),
}


class ScoreBreakdown(NamedTuple):
    diagnosis_score: float
    medication_score: float
    lab_score: float
    vitals_score: float
    context_multiplier: float
    w_diagnosis: float
    w_medication: float
    w_lab: float
    w_vitals: float
    diagnosis_contribution: float
    medication_contribution: float
    lab_contribution: float
    vitals_contribution: float
    weighted_sum: float
    raw_total: float
    final_score: float
    level: RiskLevel
    preset_name: str
    weight_sum_valid: bool


class RiskEngine:
    def __init__(
        self,
        preset: str | None = None,
        w_diagnosis: float | None = None,
        w_medication: float | None = None,
        w_lab: float | None = None,
        w_vitals: float | None = None,
    ):
        if preset is not None:
            if preset not in WEIGHT_PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. Available: {list(WEIGHT_PRESETS)}"
                )
            p = WEIGHT_PRESETS[preset]
            self._w_d, self._w_m, self._w_l, self._w_v = (
                p.w_diagnosis,
                p.w_medication,
                p.w_lab,
                p.w_vitals,
            )
            self._preset_name = preset
        else:
            self._w_d = (
                w_diagnosis if w_diagnosis is not None else settings.weight_diagnosis
            )
            self._w_m = (
                w_medication if w_medication is not None else settings.weight_medication
            )
            self._w_l = w_lab if w_lab is not None else settings.weight_lab
            self._w_v = w_vitals if w_vitals is not None else 0.25
            self._preset_name = "custom"
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = self._w_d + self._w_m + self._w_l
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Risk weights must sum to 1.0. Got: {self._w_d}+{self._w_m}+{self._w_l}={total:.4f}"
            )

    def score(
        self,
        diagnosis_score: float,
        medication_score: float,
        lab_score: float,
        vitals_score: float,
        context_multiplier: float = 1.0,
    ) -> RiskScore:
        bd = self._compute(
            diagnosis_score,
            medication_score,
            lab_score,
            vitals_score,
            context_multiplier,
        )
        logger.debug(
            "risk preset=%s D=%.3f M=%.3f L=%.3f V=%.3f C=%.2f -> R=%.4f [%s]",
            self._preset_name,
            diagnosis_score,
            medication_score,
            lab_score,
            vitals_score,
            context_multiplier,
            bd.final_score,
            bd.level.value,
        )
        return RiskScore(
            diagnosis_score=bd.diagnosis_score,
            medication_score=bd.medication_score,
            lab_score=bd.lab_score,
            vitals_score=bd.vitals_score,
            w_diagnosis=bd.w_diagnosis,
            w_medication=bd.w_medication,
            w_lab=bd.w_lab,
            w_vitals=bd.w_vitals,
            context_multiplier=bd.context_multiplier,
        )

    def score_with_breakdown(
        self,
        diagnosis_score: float,
        medication_score: float,
        lab_score: float,
        vitals_score: float,
        context_multiplier: float = 1.0,
    ) -> tuple[RiskScore, ScoreBreakdown]:
        bd = self._compute(
            diagnosis_score,
            medication_score,
            lab_score,
            vitals_score,
            context_multiplier,
        )
        risk = RiskScore(
            diagnosis_score=bd.diagnosis_score,
            medication_score=bd.medication_score,
            lab_score=bd.lab_score,
            vitals_score=bd.vitals_score,
            w_diagnosis=bd.w_diagnosis,
            w_medication=bd.w_medication,
            w_lab=bd.w_lab,
            w_vitals=bd.w_vitals,
            context_multiplier=bd.context_multiplier,
        )
        return risk, bd

    def _compute(
        self,
        diagnosis_score: float,
        medication_score: float,
        lab_score: float,
        context_multiplier: float,
    ) -> ScoreBreakdown:
        d = max(0.0, min(1.0, float(diagnosis_score)))
        m = max(0.0, min(1.0, float(medication_score)))
        l = max(0.0, min(1.0, float(lab_score)))
        c = max(0.5, min(1.5, float(context_multiplier)))
        d_c = round(self._w_d * d, 6)
        m_c = round(self._w_m * m, 6)
        l_c = round(self._w_l * l, 6)
        ws = round(d_c + m_c + l_c, 6)
        raw = round(ws * c, 6)
        final = round(max(0.0, min(1.0, raw)), 4)
        level = _classify(final)
        return ScoreBreakdown(
            d,
            m,
            l,
            c,
            self._w_d,
            self._w_m,
            self._w_l,
            d_c,
            m_c,
            l_c,
            ws,
            raw,
            final,
            level,
            self._preset_name,
            True,
        )

    def sensitivity_analysis(
        self,
        diagnosis_score: float,
        medication_score: float,
        lab_score: float,
        context_multiplier: float = 1.0,
    ) -> dict:
        base = self._compute(
            diagnosis_score, medication_score, lab_score, context_multiplier
        )

        def delta(d_b=0, m_b=0, l_b=0):
            a = self._compute(
                diagnosis_score + d_b,
                medication_score + m_b,
                lab_score + l_b,
                context_multiplier,
            )
            return round(a.final_score - base.final_score, 4)

        return {
            "base_score": base.final_score,
            "base_level": base.level.value,
            "delta_if_diagnosis_plus_0.1": delta(d_b=0.1),
            "delta_if_medication_plus_0.1": delta(m_b=0.1),
            "delta_if_lab_plus_0.1": delta(l_b=0.1),
            "dominant_factor": _dominant_factor(base),
        }

    def simulate_scenarios(
        self, base_diagnosis: float, base_medication: float, base_lab: float
    ) -> list[dict]:
        results = []
        for name, preset in WEIGHT_PRESETS.items():
            engine = RiskEngine(preset=name)
            bd = engine._compute(base_diagnosis, base_medication, base_lab, 1.0)
            results.append(
                {
                    "preset": name,
                    "description": preset.description,
                    "final_score": bd.final_score,
                    "level": bd.level.value,
                    "weights": {
                        "diagnosis": preset.w_diagnosis,
                        "medication": preset.w_medication,
                        "lab": preset.w_lab,
                    },
                }
            )
        return results


def _classify(score: float) -> RiskLevel:
    if score >= settings.threshold_critical:
        return RiskLevel.CRITICAL
    if score >= settings.threshold_high:
        return RiskLevel.HIGH
    if score >= settings.threshold_low:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _dominant_factor(bd: ScoreBreakdown) -> str:
    contribs = {
        "diagnosis": bd.diagnosis_contribution,
        "medication": bd.medication_contribution,
        "lab": bd.lab_contribution,
    }
    return max(contribs, key=contribs.get)
