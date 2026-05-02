"""Prompt-injection sanitization utilities.

Two protection layers are provided:

1. **Model-level (reject)** — ``sanitize_text`` / ``sanitize_list`` are called
   from Pydantic ``field_validator`` hooks.  Malicious input is *rejected* with a
   ``ValueError`` so the caller receives a 422 Unprocessable Entity before any
   prompt is constructed.

2. **Prompt-level (strip, defense-in-depth)** — ``safe_embed`` is called inside
   every prompt-builder function just before a user-supplied string is
   interpolated into an f-string.  If somehow bad data slipped through
   validation (e.g. via a code path that bypasses Pydantic), the pattern is
   silently stripped from the string and a warning is logged so the LLM call
   can still proceed safely.

Injection patterns
------------------
The patterns focus on structural attacks that try to hijack the LLM's role or
inject new instructions:

* Conversation-role prefixes  (``system:``, ``assistant:``, ``user:``, ``human:``)
* Meta-instruction phrases    (``ignore previous``, ``disregard``, ``forget all``)
* Jailbreak openers           (``you are now``, ``act as``, ``pretend you are``)
* Special control tokens      (``<|``, ``[INST]``, ``[/INST]``, XML role tags)
* New-instruction anchors     (``new instruction:``, ``override instruction``)

The patterns are intentionally conservative to avoid rejecting legitimate
clinical text such as "patient overrides previous treatment preference".
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injection-pattern definitions
# ---------------------------------------------------------------------------

# Each tuple: (human-readable name, compiled regex)
_RAW_PATTERNS: list[tuple[str, str]] = [
    # Conversation-role hijack attempts
    ("role_prefix_system",    r"(?:^|\n)\s*system\s*:"),
    ("role_prefix_assistant", r"(?:^|\n)\s*assistant\s*:"),
    ("role_prefix_user",      r"(?:^|\n)\s*user\s*:"),
    ("role_prefix_human",     r"(?:^|\n)\s*human\s*:"),
    # Meta-instruction phrases
    ("ignore_previous",       r"\bignore\s+(all\s+)?previous\b"),
    ("disregard",             r"\bdisregard\s+(all\s+|previous\s+|your\s+)"),
    ("forget_instructions",   r"\bforget\s+(everything|all|previous|your\s+instructions)\b"),
    # Jailbreak openers
    ("you_are_now",           r"\byou\s+are\s+now\b"),
    ("act_as",                r"\bact\s+as\s+(a|an)\b"),
    ("pretend_you_are",       r"\bpretend\s+(you\s+are|to\s+be)\b"),
    # Special LLM control tokens / XML injection
    ("im_tokens",             r"<\|(?:im_start|im_end|endoftext|pad)\|>"),
    ("inst_tokens",           r"\[/?INST\]"),
    ("xml_role_tags",         r"</?(?:system|assistant|user|prompt|instruction)>"),
    # New-instruction anchors
    ("new_instruction",       r"\bnew\s+instruction\s*:"),
    ("override_instruction",  r"\boverride\s+(all\s+|previous\s+)?instructions?\b"),
]

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for name, pattern in _RAW_PATTERNS
]

# Combined pattern used for bulk stripping in safe_embed
_COMBINED_STRIP = re.compile(
    "|".join(f"(?P<p{i}>{p})" for i, (_, p) in enumerate(_RAW_PATTERNS)),
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

# Maximum per-item length for free-text list entries (allergies, comorbidities)
MAX_LIST_ITEM_LENGTH = 100


def strip_control_chars(text: str) -> str:
    """Remove ASCII control characters (except newline/tab) and Unicode controls.

    Keeps ``\\n`` and ``\\t`` because they appear in clinical notes.
    Strips everything in Unicode category ``Cc`` (control) that isn't those two.
    """
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or unicodedata.category(ch) != "Cc"
    )


def _find_injection(text: str) -> str | None:
    """Return the name of the first injection pattern found, or ``None``."""
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return name
    return None


def sanitize_text(
    text: str,
    field_name: str,
    *,
    max_length: int | None = None,
) -> str:
    """Clean *text* and raise ``ValueError`` if injection patterns are detected.

    Steps:
    1. Strip control characters.
    2. Enforce ``max_length`` (if provided).
    3. Detect injection patterns — raise ``ValueError`` if any are found.

    Parameters
    ----------
    text:
        The raw user-supplied string.
    field_name:
        Human-readable field name used in error messages and log lines.
    max_length:
        Optional hard limit on the string length *after* control-char stripping.
        Pydantic ``max_length`` constraints are usually sufficient; this is a
        fallback for contexts where no Pydantic schema is in play.

    Returns
    -------
    str
        The cleaned string (control chars removed).

    Raises
    ------
    ValueError
        If an injection pattern is detected or the cleaned length exceeds
        ``max_length``.
    """
    cleaned = strip_control_chars(text)

    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(
            f"Field '{field_name}' exceeds maximum length of {max_length} characters."
        )

    pattern_name = _find_injection(cleaned)
    if pattern_name is not None:
        logger.warning(
            "safety.input_injection_detected field=%s pattern=%s",
            field_name,
            pattern_name,
        )
        raise ValueError(
            f"Field '{field_name}' contains a disallowed pattern ({pattern_name}). "
            "Input rejected to prevent prompt injection."
        )

    return cleaned


def sanitize_list(
    items: list[str],
    field_name: str,
    *,
    max_item_length: int = MAX_LIST_ITEM_LENGTH,
) -> list[str]:
    """Sanitize every string in *items*, enforcing per-item length and injection checks.

    Parameters
    ----------
    items:
        List of raw user-supplied strings (e.g. allergies, comorbidities).
    field_name:
        Human-readable field name used in error messages.
    max_item_length:
        Maximum character length allowed per item after control-char stripping.

    Returns
    -------
    list[str]
        The cleaned list.

    Raises
    ------
    ValueError
        If any item exceeds ``max_item_length`` or contains an injection pattern.
    """
    return [
        sanitize_text(item, f"{field_name}[{i}]", max_length=max_item_length)
        for i, item in enumerate(items)
    ]


def safe_embed(text: str, field_name: str = "unknown") -> str:
    """Defense-in-depth: strip injection patterns just before prompt interpolation.

    Unlike ``sanitize_text``, this function *never raises* — it strips the
    offending text and logs a warning.  Use this inside prompt-builder functions
    as a second line of defense in case input somehow bypassed model validation.

    Parameters
    ----------
    text:
        The string about to be embedded in an LLM prompt.
    field_name:
        Human-readable field name for log messages.

    Returns
    -------
    str
        The text with injection patterns replaced by ``[REMOVED]`` and
        control characters stripped.
    """
    cleaned = strip_control_chars(text)

    def _replacer(m: re.Match[str]) -> str:
        logger.warning(
            "safety.prompt_injection_stripped field=%s match=%r",
            field_name,
            m.group(0),
        )
        return "[REMOVED]"

    return _COMBINED_STRIP.sub(_replacer, cleaned)
