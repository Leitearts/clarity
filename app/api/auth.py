"""API key authentication dependency.

Usage
-----
Apply ``Depends(require_api_key)`` to any route that must be protected:

    @router.post("/analyze", dependencies=[Depends(require_api_key)])
    async def analyze(...): ...

Configuration
-------------
Set the environment variable ``CLARITY_API_KEY`` (or add it to ``.env``).
When the variable is empty the check is skipped so local/CI environments
continue to work without configuration changes.

Security notes
--------------
* The header value is compared with :func:`secrets.compare_digest` to prevent
  timing-based attacks.
* A missing *or* wrong key both return **403 Forbidden** — this avoids leaking
  whether authentication is enabled at all.
* Public endpoints (/health, /agent-card, /a2a/schema, /a2a/capabilities,
  /a2a/health) intentionally do *not* use this dependency.
"""
from __future__ import annotations

import logging
import secrets

from fastapi import Depends, Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)


async def require_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    """FastAPI dependency that enforces X-API-Key authentication.

    Raises HTTP 403 when:
    - ``CLARITY_API_KEY`` is configured **and**
    - the request either omits the ``X-API-Key`` header or supplies a wrong value.

    When ``CLARITY_API_KEY`` is empty (default) the check is a no-op, preserving
    backward-compatible behaviour in development and CI.
    """
    configured_key: str = settings.api_key
    if not configured_key:
        # Auth not configured — allow request through (dev / CI mode).
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
