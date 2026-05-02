from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLARITY_", env_file=".env")

    # LLM
    anthropic_api_key: str = Field(default="")
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 30

    # Risk weights
    weight_diagnosis: float = 0.40
    weight_medication: float = 0.35
    weight_lab: float = 0.25

    # Risk thresholds
    threshold_low: float = 0.35
    threshold_high: float = 0.65
    threshold_critical: float = 0.85

    # Agent timeouts
    agent_timeout_seconds: int = 20

    # Audit
    audit_log_path: str = "logs/audit.jsonl"

    # Risk presets
    risk_weight_preset: str = "default"
    risk_threshold_preset: str = "default"

    # API
    api_version: str = "v1"
    debug: bool = False

    # Auth — set CLARITY_API_KEY in the environment to enforce authentication.
    # When empty (default), authentication is disabled (development mode).
    api_key: str = Field(default="")

    # Rate limiting — set CLARITY_RATE_LIMIT_ENABLED=false to disable (e.g. in tests).
    # Limits are expressed in slowapi format: "<count>/<period>" (e.g. "10/minute").
    rate_limit_enabled: bool = True
    rate_limit_analyze: str = "10/minute"
    rate_limit_audit: str = "30/minute"
    rate_limit_a2a: str = "20/minute"


settings = Settings()
