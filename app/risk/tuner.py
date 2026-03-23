"""
Offline weight tuner for the CLARITY Risk Engine.

Usage:
    python -m app.risk.tuner --cases data/labeled_cases.json

Labeled case format (extend synthetic_cases.json with true_level):
    {
      "diagnosis_score": 0.7,
      "medication_score": 0.4,
      "lab_score": 0.6,
      "context_multiplier": 1.2,
      "true_level": "high"
    }

Outputs the weight combination with the highest classification accuracy
across the labeled dataset, ready to paste into .env.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import NamedTuple


class TuningResult(NamedTuple):
    w_diagnosis: float
    w_medication: float
    w_lab: float
    accuracy: float
    confusion: dict[str, int]


def tune(cases_path: str, step: float = 0.05) -> TuningResult:
    """Grid-search over all weight combinations that sum to 1.0."""
    cases = json.loads(Path(cases_path).read_text())
    candidates = _weight_grid(step)
    print(f"Evaluating {len(candidates)} weight combinations on {len(cases)} cases...")

    best: TuningResult | None = None

    for w_d, w_m, w_l in candidates:
        from app.risk.engine import RiskEngine
        engine = RiskEngine(w_diagnosis=w_d, w_medication=w_m, w_lab=w_l)
        correct = 0
        confusion: dict[str, int] = {}

        for case in cases:
            predicted = engine.score(
                diagnosis_score=case["diagnosis_score"],
                medication_score=case["medication_score"],
                lab_score=case["lab_score"],
                context_multiplier=case.get("context_multiplier", 1.0),
            ).level.value
            true_level = case["true_level"]
            key = f"{true_level}->{predicted}"
            confusion[key] = confusion.get(key, 0) + 1
            if predicted == true_level:
                correct += 1

        accuracy = correct / len(cases)
        result = TuningResult(w_d, w_m, w_l, accuracy, confusion)
        if best is None or accuracy > best.accuracy:
            best = result

    return best  # type: ignore[return-value]


def _weight_grid(step: float) -> list[tuple[float, float, float]]:
    values = [round(i * step, 2) for i in range(int(1.0 / step) + 1)]
    return [
        (w_d, w_m, w_l)
        for w_d, w_m, w_l in itertools.product(values, repeat=3)
        if abs(w_d + w_m + w_l - 1.0) < 0.001 and all(v > 0 for v in (w_d, w_m, w_l))
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLARITY weight tuner")
    parser.add_argument("--cases", required=True, help="Path to labeled cases JSON")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Grid step size (default: 0.05)")
    args = parser.parse_args()

    result = tune(args.cases, step=args.step)

    print(f"\nBest weights found:")
    print(f"  w_diagnosis  = {result.w_diagnosis}")
    print(f"  w_medication = {result.w_medication}")
    print(f"  w_lab        = {result.w_lab}")
    print(f"  Accuracy     = {result.accuracy:.2%}")
    print(f"\nConfusion matrix (true->predicted):")
    for k, v in sorted(result.confusion.items()):
        print(f"  {k}: {v}")
    print(f"\nPaste into .env:")
    print(f"  CLARITY_WEIGHT_DIAGNOSIS={result.w_diagnosis}")
    print(f"  CLARITY_WEIGHT_MEDICATION={result.w_medication}")
    print(f"  CLARITY_WEIGHT_LAB={result.w_lab}")
