from __future__ import annotations

from detection_system import DetectionResult


def quarantine_file(file_path: str) -> None:
    """Quarantine a file."""
    _ = file_path


def send_alert(result: DetectionResult) -> None:
    """Send an alert for a detection result."""
    _ = result


def handle_threat(result: DetectionResult) -> None:
    """Handle a threat using a dataclass-based detection result."""
    if result.action == "quarantine" and result.file_path:
        quarantine_file(result.file_path)
    if result.verdict.lower() != "clean":
        send_alert(result)
