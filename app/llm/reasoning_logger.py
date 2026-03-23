"""
Captures the full LLM reasoning chain for each agent.
Attached to the analysis and retrievable via GET /reasoning/{analysis_id}.
Makes the AI decision process completely transparent for auditors and judges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ReasoningStep:
    """One agent's contribution to the reasoning trace."""
    agent: str
    user_prompt_preview: str        # first 200 chars of what the agent was asked
    llm_response_preview: str       # first 200 chars of raw LLM output
    parsed_score: float
    parsed_reasoning: str           # verbatim reasoning string from JSON
    confidence: float
    used_fallback: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReasoningTrace:
    """Collects reasoning steps across all agents for one analysis."""

    def __init__(self, analysis_id: str):
        self.analysis_id = analysis_id
        self.steps: list[ReasoningStep] = []

    def add(self, step: ReasoningStep) -> None:
        self.steps.append(step)

    def summary(self) -> dict[str, Any]:
        return {
            "analysis_id":   self.analysis_id,
            "agent_count":   len(self.steps),
            "fallback_count": sum(1 for s in self.steps if s.used_fallback),
            "avg_confidence": round(
                sum(s.confidence for s in self.steps) / max(len(self.steps), 1), 3
            ),
            "steps": [
                {
                    "agent":            s.agent,
                    "score":            s.parsed_score,
                    "reasoning":        s.parsed_reasoning,
                    "confidence":       s.confidence,
                    "used_fallback":    s.used_fallback,
                    "prompt_preview":   s.user_prompt_preview[:200],
                    "response_preview": s.llm_response_preview[:200],
                    "timestamp":        s.timestamp.isoformat(),
                }
                for s in self.steps
            ],
        }
