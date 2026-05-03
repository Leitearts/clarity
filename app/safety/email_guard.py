"""Trusted-sender e-mail validation.

Provides :func:`_is_trusted_sender` which checks whether the domain of an
e-mail address exactly matches one of the configured trusted domains.

Security notes
--------------
* Uses :func:`email.utils.parseaddr` to extract the addr-spec, preventing
  display-name tricks such as ``"evil@trusted.com <real@attacker.com>"``.
* Compares the domain extracted from the addr-spec **exactly** (not via
  substring matching) to prevent spoofing like ``evil@trusted.com.attacker.com``.
* Normalises all domains to lowercase before comparison.
* Returns ``False`` — never raises — for any malformed or empty input.
"""

from __future__ import annotations

import email.utils
import logging

logger = logging.getLogger(__name__)


def _get_trusted_domains() -> frozenset[str]:
    """Return the set of configured trusted domains (normalised to lowercase).

    Reads ``settings.trusted_sender_domains`` which is a comma-separated
    string (e.g. ``"company.com,partner.org"``).  An empty or whitespace-only
    value produces an empty set, meaning no sender is trusted.
    """
    from app.config import settings

    raw = settings.trusted_sender_domains
    return frozenset(part.lower().strip() for part in raw.split(",") if part.strip())


def _is_trusted_sender(email_address: str) -> bool:
    """Return ``True`` if *email_address* belongs to a configured trusted domain.

    The check uses :func:`email.utils.parseaddr` to extract the addr-spec
    (stripping any RFC 5322 display name), then splits on ``@`` to obtain the
    domain and compares it **exactly** (case-insensitive) against the configured
    trusted-domain list.

    Parameters
    ----------
    email_address:
        Raw e-mail string supplied by an external caller.  May include a
        display name (e.g. ``"Alice <alice@company.com>"``).

    Returns
    -------
    bool
        ``True`` only when the addr-spec domain exactly matches a configured
        trusted domain.  ``False`` for every other case, including malformed
        input and missing ``@``.  This function **never raises**.
    """
    try:
        if not isinstance(email_address, str):
            return False

        # parseaddr returns ("", "") when it cannot parse a valid address.
        _display, addr_spec = email.utils.parseaddr(email_address)

        if not addr_spec or "@" not in addr_spec:
            return False

        # Split on the *last* '@' to be robust against unusual local-parts.
        local, _, domain = addr_spec.rpartition("@")

        if not local or not domain:
            return False

        domain_lower = domain.lower()
        trusted = _get_trusted_domains()

        if not trusted:
            # No trusted domains configured — no sender is trusted.
            return False

        return domain_lower in trusted

    except Exception:  # pragma: no cover — belt-and-suspenders safety net
        logger.warning(
            "email_guard._is_trusted_sender raised unexpectedly", exc_info=True
        )
        return False
