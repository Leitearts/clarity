from __future__ import annotations
from dataclasses import dataclass
from app.models.risk import RiskLevel


@dataclass(frozen=True)
class ThresholdConfig:
    low_max: float
    medium_max: float
    high_max: float

    def classify(self, score: float) -> RiskLevel:
        if score >= self.high_max:
            return RiskLevel.CRITICAL
        if score >= self.medium_max:
            return RiskLevel.HIGH
        if score >= self.low_max:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def as_dict(self) -> dict:
        return {
            "low": f"[0.00, {self.low_max})",
            "medium": f"[{self.low_max}, {self.medium_max})",
            "high": f"[{self.medium_max}, {self.high_max})",
            "critical": f"[{self.high_max}, 1.00]",
        }


THRESHOLD_PRESETS: dict[str, ThresholdConfig] = {
    "default": ThresholdConfig(0.35, 0.65, 0.85),
    "conservative": ThresholdConfig(0.25, 0.50, 0.75),
    "liberal": ThresholdConfig(0.40, 0.70, 0.90),
}

DEFAULT_THRESHOLDS = THRESHOLD_PRESETS["default"]

CLINICAL_ACTIONS: dict[RiskLevel, dict] = {
    RiskLevel.LOW: {
        "urgency": "routine",
        "review_window": "next scheduled visit",
        "notify": [],
        "escalate": False,
        "description": "No immediate action required. Standard monitoring.",
    },
    RiskLevel.MEDIUM: {
        "urgency": "within 24 hours",
        "review_window": "24h",
        "notify": ["attending physician"],
        "escalate": False,
        "description": "Schedule clinical review. Flag for attending.",
    },
    RiskLevel.HIGH: {
        "urgency": "within 1 hour",
        "review_window": "1h",
        "notify": ["attending physician", "charge nurse"],
        "escalate": True,
        "description": "Urgent review required. Consider intervention.",
    },
    RiskLevel.CRITICAL: {
        "urgency": "immediate",
        "review_window": "now",
        "notify": ["attending physician", "charge nurse", "rapid response team"],
        "escalate": True,
        "description": "Immediate bedside assessment. Activate rapid response if needed.",
    },
}
