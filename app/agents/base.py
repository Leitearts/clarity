from __future__ import annotations
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any
from app.models.agent import (
    AgentRequest,
    AgentResponse,
    AgentType,
    Finding,
    FindingSeverity,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    agent_type: AgentType

    def __init__(self, timeout_seconds: int = 20):
        self.timeout_seconds = timeout_seconds

    async def run(self, request: AgentRequest) -> AgentResponse:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._analyze(request), timeout=self.timeout_seconds
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            response.metadata["processing_ms"] = elapsed_ms
            logger.info(
                "agent=%s case=%s score=%.3f ms=%d",
                self.agent_type.value,
                request.case_id,
                response.risk_score,
                elapsed_ms,
            )
            return response
        except asyncio.TimeoutError:
            logger.warning(
                "agent=%s TIMEOUT case=%s", self.agent_type.value, request.case_id
            )
            return self._fallback_response(request, error="Agent timed out")
        except Exception as exc:
            logger.exception(
                "agent=%s ERROR case=%s: %s",
                self.agent_type.value,
                request.case_id,
                exc,
            )
            return self._fallback_response(request, error=str(exc))

    @abstractmethod
    async def _analyze(self, request: AgentRequest) -> AgentResponse: ...

    def _fallback_response(self, request: AgentRequest, error: str) -> AgentResponse:
        return AgentResponse(
            agent_type=self.agent_type,
            case_id=request.case_id,
            risk_score=0.5,
            findings=[
                Finding(
                    code="AGENT_ERROR",
                    description=f"Agent unavailable: {error}",
                    severity=FindingSeverity.INFO,
                )
            ],
            reasoning=f"Agent failed: {error}. Using neutral fallback score of 0.5.",
            confidence=0.0,
            error=error,
        )

    def _build_response(
        self,
        request: AgentRequest,
        risk_score: float,
        findings: list[Finding],
        reasoning: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            agent_type=self.agent_type,
            case_id=request.case_id,
            risk_score=risk_score,
            findings=findings,
            reasoning=reasoning,
            confidence=confidence,
            metadata=metadata or {},
        )
