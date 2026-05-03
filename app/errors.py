"""Centralized error logging utility with optional Counter-based failure metrics.

Usage::

    from app.errors import log_error, CATEGORY_DETECTION, CATEGORY_PARSING

    try:
        result = await self._llm.complete(...)
    except Exception:
        log_error(logger, CATEGORY_DETECTION, "vitals._analyze",
                  "LLM analysis failed; falling back to rule-based scoring")
        # fall through to rule-based analysis
"""

from __future__ import annotations

import logging
from collections import Counter

# ── Error categories ──────────────────────────────────────────────────────────

CATEGORY_NETWORK = "network"
CATEGORY_PARSING = "parsing"
CATEGORY_DETECTION = "detection"
CATEGORY_INIT = "init"

# ── Module-level failure counter ──────────────────────────────────────────────

_failure_counts: Counter = Counter()

_logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────


def log_error(
    source_logger: logging.Logger,
    category: str,
    location: str,
    message: str,
    exc_info: bool = True,
) -> None:
    """Log a categorized error and increment the in-memory failure counter.

    Args:
        source_logger: Logger from the calling module (``logging.getLogger(__name__)``).
        category: Error category — use one of the ``CATEGORY_*`` constants
            (``"network"``, ``"parsing"``, ``"detection"``, ``"init"``).
        location: Short call-site identifier, e.g. ``"vitals._analyze"``.
        message: Human-readable error context (what was happening when it failed).
        exc_info: When *True* (default) the current exception traceback is
            included in the log record, identical to ``logger.error(..., exc_info=True)``.
    """
    _failure_counts[category] += 1
    _failure_counts[f"{category}:{location}"] += 1
    source_logger.error(
        "error.category=%s location=%s %s",
        category,
        location,
        message,
        exc_info=exc_info,
    )


def get_failure_counts() -> dict[str, int]:
    """Return a snapshot of current failure counts keyed by category and location.

    The returned dictionary contains two kinds of keys:

    * ``"<category>"`` — total errors for that category across all locations.
    * ``"<category>:<location>"`` — errors for a specific category + location pair.

    Returns:
        A plain ``dict`` copy of the internal counter (safe to mutate).
    """
    return dict(_failure_counts)


def reset_failure_counts() -> None:
    """Reset all failure counts to zero.

    Intended for use in unit tests that inspect counter state.  Call this in
    test setup/teardown to avoid cross-test contamination.
    """
    _failure_counts.clear()
