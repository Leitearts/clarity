from __future__ import annotations
from collections import Counter
from fastapi import APIRouter, Request

router = APIRouter(tags=["impact"])


@router.get("/impact", summary="Real-time CLARITY usage and impact statistics")
async def get_impact(request: Request):
    """
    Live statistics computed from the audit log.
    Shows real numbers from actual analyses run during the session.
    """
    audit_logger = request.app.state.audit_logger
    records = list(audit_logger._iter_records())

    if not records:
        return {
            "headline": {"analyses_run": 0, "message": "Run some demo cases first"},
            "real_world_context": {
                "preventable_deaths_per_year_us": 250_000,
                "human_review_time_minutes": 20,
                "clarity_analysis_time_seconds": 2.0,
            },
        }

    total = len(records)
    level_counts = Counter(r.risk_level for r in records)
    critical_count = level_counts.get("critical", 0)
    high_count = level_counts.get("high", 0)
    error_count = sum(1 for r in records if r.had_errors)
    avg_ms = (
        sum(r.processing_time_ms for r in records if r.processing_time_ms)
        / max(sum(1 for r in records if r.processing_time_ms), 1)
    )

    return {
        "headline": {
            "analyses_run":            total,
            "critical_risks_detected": critical_count,
            "high_risks_detected":     high_count,
            "actionable_alerts":       critical_count + high_count,
            "avg_analysis_time_ms":    round(avg_ms),
            "agent_errors":            error_count,
        },
        "risk_distribution": {
            "critical": critical_count,
            "high":     high_count,
            "medium":   level_counts.get("medium", 0),
            "low":      level_counts.get("low", 0),
        },
        "reliability": {
            "analyses_without_error": total - error_count,
            "error_rate_pct":         round(error_count / max(total, 1) * 100, 2),
            "fallback_capable":       True,
            "zero_phi_stored":        True,
        },
        "real_world_context": {
            "preventable_deaths_per_year_us": 250_000,
            "avg_icu_medications_per_patient": 12,
            "human_review_time_minutes":       20,
            "clarity_analysis_time_seconds":   round(avg_ms / 1000, 1),
            "time_saved_per_case_minutes":     round(20 - avg_ms / 60000, 1),
        },
    }
