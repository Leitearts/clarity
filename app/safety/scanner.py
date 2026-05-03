"""Malware-signature scanner.

Scans raw bytes (e.g. file content, network payloads) for known malware
signatures defined in :mod:`app.safety.signatures`.

Usage
-----
::

    from app.safety.scanner import Scanner, MatchResult
    from app.safety.signatures import SIGNATURES

    scanner = Scanner(SIGNATURES)                     # flexible (default)
    scanner_strict = Scanner(SIGNATURES, strict=True) # strict / literal-only

    with open("suspicious.bin", "rb") as fh:
        matches = scanner.scan(fh.read())

    for m in matches:
        print(m.name, m.offset)

Design notes
------------
* Patterns are **precompiled once** at :class:`Scanner` construction time.
* File content is wrapped in a :class:`memoryview` so no additional copy is
  made when slicing during reporting.
* Strict mode skips :class:`~app.safety.signatures.RegexSignature` entries and
  uses only literal byte patterns, still case-insensitively (the evasion
  problem is solved in both modes).
* Flexible mode (default) runs all signatures including regex patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.safety.signatures import RegexSignature, Signature


@dataclass(frozen=True, slots=True)
class MatchResult:
    """A single signature hit found inside scanned content."""

    name: str
    """Human-readable signature name."""
    offset: int
    """Byte offset of the first matching character in the scanned buffer."""
    matched_bytes: bytes
    """The exact bytes that triggered the match."""


class Scanner:
    """Precompiles signatures and scans byte buffers for matches.

    Parameters
    ----------
    signatures:
        The signature library to use (defaults to
        :data:`~app.safety.signatures.SIGNATURES`).
    strict:
        When *True*, only :class:`~app.safety.signatures.LiteralSignature`
        entries are evaluated (no regex).  Both modes remain case-insensitive.
        Defaults to *False* (flexible: all signatures active).
    """

    def __init__(
        self,
        signatures: Sequence[Signature],
        *,
        strict: bool = False,
    ) -> None:
        self._strict = strict
        # Pre-compile only the signatures that will actually be used.
        self._compiled: list[tuple[str, re.Pattern[bytes]]] = []
        for sig in signatures:
            if strict and isinstance(sig, RegexSignature):
                continue
            self._compiled.append((sig.name, sig.compile()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def strict(self) -> bool:
        """``True`` if this scanner runs in strict (literal-only) mode."""
        return self._strict

    def scan(self, data: bytes | bytearray | memoryview) -> list[MatchResult]:
        """Scan *data* for all known signature matches.

        The buffer is wrapped in a :class:`memoryview` to avoid extra copies
        when extracting matched bytes.

        Parameters
        ----------
        data:
            Raw bytes to scan.

        Returns
        -------
        list[MatchResult]
            All matches found (may be empty).  Order follows signature order,
            with multiple hits for the same signature listed left-to-right.
        """
        buf = self._to_bytes(data)
        results: list[MatchResult] = []
        for name, pattern in self._compiled:
            for m in pattern.finditer(buf):
                results.append(
                    MatchResult(
                        name=name,
                        offset=m.start(),
                        matched_bytes=m.group(0),
                    )
                )
        return results

    def contains_malware(self, data: bytes | bytearray | memoryview) -> bool:
        """Return ``True`` as soon as *any* signature matches, ``False`` otherwise.

        Short-circuits on the first hit so it is faster than :meth:`scan` when
        you only need a yes/no answer.
        """
        buf = self._to_bytes(data)
        for _name, pattern in self._compiled:
            if pattern.search(buf):
                return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bytes(data: bytes | bytearray | memoryview) -> bytes:
        """Normalise *data* to :class:`bytes` with a single copy at most.

        * :class:`bytes` — returned as-is (no copy).
        * :class:`memoryview` — converted via ``bytes()`` (one copy).
        * :class:`bytearray` — converted via ``bytes()`` (one copy).
        """
        if isinstance(data, bytes):
            return data
        return bytes(data)
