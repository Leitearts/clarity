from __future__ import annotations
import json
import logging
import re
from typing import Any
from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


class LLMResponse:
    def __init__(self, data: dict[str, Any], raw_text: str, token_usage: dict):
        self.data = data
        self.raw_text = raw_text
        self.token_usage = token_usage

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class LLMService:
    MAX_RETRIES = 2

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: list[str],
        temperature: float = 0.1,
    ) -> LLMResponse:
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                raw = await self._call_api(system_prompt, user_prompt, temperature)
                parsed = self._extract_json(raw)
                self._validate_keys(parsed, required_keys)
                return LLMResponse(data=parsed, raw_text=raw, token_usage={})
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "llm.parse_fail attempt=%d/%d error=%s",
                    attempt + 1,
                    self.MAX_RETRIES + 1,
                    exc,
                )
                if attempt < self.MAX_RETRIES:
                    user_prompt = (
                        user_prompt + f"\n\nPREVIOUS ATTEMPT FAILED: {exc}. "
                        "Return ONLY a valid JSON object. No markdown, no explanation."
                    )
        raise LLMServiceError(
            f"LLM failed after {self.MAX_RETRIES + 1} attempts. Last: {last_error}"
        )

    async def _call_api(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        client = get_client()
        message = await client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
        return json.loads(text)

    @staticmethod
    def _validate_keys(data: dict, required_keys: list[str]) -> None:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise KeyError(f"LLM response missing required keys: {missing}")


class LLMServiceError(Exception):
    pass
