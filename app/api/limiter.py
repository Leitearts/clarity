"""Rate limiting setup using slowapi.

The ``Limiter`` singleton defined here is imported by route modules to apply
per-endpoint rate limits.  It is also attached to ``app.state.limiter`` in
``app/main.py`` so that SlowAPIMiddleware can locate it.

Key function
------------
When ``settings.rate_limit_enabled`` is ``True``, clients are identified by
their IP address (``X-Forwarded-For`` → ``client.host`` fallback).

When ``settings.rate_limit_enabled`` is ``False`` (e.g. in test environments),
a fresh UUID is returned for every request.  Since each request has its own
unique bucket, no counter ever reaches a configured limit — effectively
disabling enforcement without changing any decorator signatures.
"""
from __future__ import annotations

import uuid

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.utils.network import is_internal_ip


def _key_func(request: Request) -> str:
    """Return the rate-limit bucket key for a request.

    Uses the real remote address when rate limiting is enabled; returns a
    random UUID per request when disabled so limits are never reached.

    Internal (RFC-1918) source addresses share a single ``internal`` bucket
    so that trusted intra-cluster callers are not throttled individually.
    """
    if not settings.rate_limit_enabled:
        return str(uuid.uuid4())
    remote = get_remote_address(request)
    if is_internal_ip(remote):
        return "internal"
    return remote


limiter = Limiter(key_func=_key_func)
