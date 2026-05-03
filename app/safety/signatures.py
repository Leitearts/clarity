"""Malware signature definitions.

Each entry is either a :class:`LiteralSignature` (plain byte string, matched
case-insensitively) or a :class:`RegexSignature` (compiled against raw bytes,
always case-insensitive).

Add new signatures by appending to :data:`SIGNATURES`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True, slots=True)
class LiteralSignature:
    """A plain-text signature matched as a case-insensitive byte substring."""

    name: str
    pattern: str  # human-readable; lowercased bytes are used at scan time

    def compile(self) -> re.Pattern[bytes]:
        """Return a compiled bytes pattern for this literal signature."""
        return re.compile(re.escape(self.pattern.encode()), re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RegexSignature:
    """A signature expressed as a regular expression matched against raw bytes."""

    name: str
    pattern: str  # raw regex string (ASCII / bytes-compatible)
    _compiled: re.Pattern[bytes] = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        # Bypass frozen to cache the compiled pattern once at construction time.
        object.__setattr__(
            self,
            "_compiled",
            re.compile(self.pattern.encode(), re.IGNORECASE),
        )

    def compile(self) -> re.Pattern[bytes]:
        """Return the pre-compiled bytes pattern."""
        return self._compiled


Signature = Union[LiteralSignature, RegexSignature]

# ---------------------------------------------------------------------------
# Signature library
# ---------------------------------------------------------------------------

SIGNATURES: list[Signature] = [
    # ── Ransomware families ──────────────────────────────────────────────────
    LiteralSignature("WannaCry",     "WannaCry"),
    LiteralSignature("WannaCry_msg", "Wanna Decryptor"),
    LiteralSignature("Locky",        "Locky"),
    LiteralSignature("Petya",        "Petya"),
    LiteralSignature("NotPetya",     "GoldenEye"),
    LiteralSignature("REvil",        "REvil"),
    LiteralSignature("Ryuk",         "Ryuk"),
    LiteralSignature("Conti",        "Conti"),
    # ── Remote access / backdoors ────────────────────────────────────────────
    LiteralSignature("Cobalt_Strike", "cobaltstrike"),
    LiteralSignature("Mimikatz",      "mimikatz"),
    LiteralSignature("Metasploit",    "meterpreter"),
    # ── Suspicious string patterns (regex) ──────────────────────────────────
    RegexSignature(
        "Base64_PE_header",
        r"TVqQAA[A-Za-z0-9+/]{6,}",  # base64-encoded MZ/PE header
    ),
    RegexSignature(
        "PowerShell_encoded_command",
        r"-[Ee](?:nc(?:odedcommand)?)?(?:\s+|=)[A-Za-z0-9+/]{20,}={0,2}",
    ),
    RegexSignature(
        "WannaCry_extension",
        r"\.WNCRY\b",
    ),
]
