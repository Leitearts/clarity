from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class DetectionResult:
    verdict: str
    action: str
    confidence: float
    file_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, str):
            raise TypeError("verdict must be a string")
        if not self.verdict.strip():
            raise ValueError("verdict must be a non-empty str")
        if not isinstance(self.action, str):
            raise TypeError("action must be a string")
        if not self.action.strip():
            raise ValueError("action must be a non-empty str")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a numeric value")
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.file_path is not None and not isinstance(self.file_path, str):
            raise TypeError("file_path must be a string or None")


def evaluate_file(file_path: str) -> DetectionResult:
    """Return a simple detection result for a file path."""
    if file_path.endswith(".exe"):
        return DetectionResult(
            verdict="malicious",
            action="quarantine",
            confidence=0.95,
            file_path=file_path,
        )
    return DetectionResult(
        verdict="clean",
        action="allow",
        confidence=0.10,
        file_path=file_path,
    )
