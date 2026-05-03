"""Centralized logging setup for CLARITY.

Call ``setup_logging()`` exactly once at application startup (before any other
module emits log records).  Subsequent calls are no-ops thanks to the guard on
the root logger's handler list.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_LOG_FILE = Path("logs/clarity.log")
_QUARANTINE_DIR = Path("quarantine")

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5


def setup_logging(level: int | None = None) -> None:
    """Configure root logger with a console handler and a rotating file handler.

    This function is idempotent: if the root logger already has handlers
    attached (i.e. ``setup_logging()`` was called earlier, or the test harness
    pre-configured logging), it returns immediately without adding duplicates.

    Args:
        level: Optional log level override (e.g. ``logging.DEBUG``).  Defaults
               to ``logging.DEBUG`` when the ``CLARITY_DEBUG`` env-var is
               truthy, otherwise ``logging.INFO``.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    # Resolve level lazily so we don't import settings at module load time
    # (avoids circular-import issues if settings itself logs).
    if level is None:
        try:
            from app.config import settings  # noqa: PLC0415

            level = logging.DEBUG if settings.debug else logging.INFO
        except Exception:  # pragma: no cover
            level = logging.INFO

    # Ensure required directories exist *before* opening any file handlers.
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
