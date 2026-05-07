from __future__ import annotations

from detection_system import DetectionResult
from response_handler import handle_threat


def test_handle_threat_triggers_quarantine_and_alert_for_malicious_verdict(monkeypatch):
    calls = {"quarantine": 0, "alert": 0}

    def fake_quarantine(_: str) -> None:
        calls["quarantine"] += 1

    def fake_alert(_: DetectionResult) -> None:
        calls["alert"] += 1

    monkeypatch.setattr("response_handler.quarantine_file", fake_quarantine)
    monkeypatch.setattr("response_handler.send_alert", fake_alert)

    result = DetectionResult(
        verdict="malicious",
        action="quarantine",
        confidence=0.99,
        file_path="/tmp/eicar.exe",
    )

    handle_threat(result)

    assert calls["quarantine"] == 1
    assert calls["alert"] == 1
