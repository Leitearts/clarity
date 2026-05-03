"""Network utility helpers."""
from __future__ import annotations

import ipaddress


def is_internal_ip(ip: str) -> bool:
    """Return ``True`` if *ip* is a private (RFC-1918) address, ``False`` otherwise.

    Uses :func:`ipaddress.ip_address` and its ``is_private`` property for
    accurate classification of the full private ranges:

    * ``10.0.0.0/8``
    * ``172.16.0.0/12``  (covers 172.16.x.x – 172.31.x.x)
    * ``192.168.0.0/16``

    Invalid or empty input is handled gracefully — ``False`` is returned and
    no exception is propagated to the caller.
    """
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False
