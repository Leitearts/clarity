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
        if not isinstance(self.verdict, str) or not self.verdict.strip():
            raise ValueError("verdict must be a non-empty str")
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("action must be a non-empty str")
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a float")
        self.confidence = float(self.confidence)
        if self.file_path is not None and not isinstance(self.file_path, str):
            raise ValueError("file_path must be Optional[str]")


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
