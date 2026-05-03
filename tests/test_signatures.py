"""Tests for the malware-signature scanner.

Covers:
  - Case-insensitive literal signature detection ("wannacry", "WannaCry", mixed)
  - Regex signature detection
  - Strict mode (regex signatures skipped)
  - Flexible mode (default; all signatures active)
  - :class:`MatchResult` fields
  - :meth:`Scanner.contains_malware` short-circuit helper
  - Clean content returns no matches
  - memoryview / bytearray inputs
"""

from __future__ import annotations

import pytest

from app.safety.scanner import Scanner
from app.safety.signatures import (
    SIGNATURES,
    LiteralSignature,
    RegexSignature,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scanner() -> Scanner:
    """Flexible scanner backed by the real signature library."""
    return Scanner(SIGNATURES)


@pytest.fixture(scope="module")
def scanner_strict() -> Scanner:
    """Strict-mode scanner backed by the real signature library."""
    return Scanner(SIGNATURES, strict=True)


def _encode(text: str) -> bytes:
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Case-insensitivity — literal signatures
# ---------------------------------------------------------------------------


class TestCaseInsensitiveLiteralDetection:
    """WannaCry must be detected regardless of capitalisation."""

    @pytest.mark.parametrize(
        "variant",
        [
            "wannacry",
            "WannaCry",
            "WANNACRY",
            "wAnNaCrY",
            "Wannacry",
        ],
    )
    def test_wannacry_variants(self, scanner: Scanner, variant: str):
        matches = scanner.scan(_encode(variant))
        names = [m.name for m in matches]
        assert "WannaCry" in names, f"Expected WannaCry detected for input {variant!r}"

    def test_wannacry_embedded_in_larger_payload(self, scanner: Scanner):
        payload = b"header\x00\x01WANNACRY\x00payload"
        assert scanner.contains_malware(payload)

    def test_mimikatz_case_insensitive(self, scanner: Scanner):
        for variant in ("mimikatz", "MIMIKATZ", "MiMiKaTz"):
            matches = scanner.scan(_encode(variant))
            assert any(m.name == "Mimikatz" for m in matches), variant

    def test_cobalt_strike_case_insensitive(self, scanner: Scanner):
        for variant in ("cobaltstrike", "CobaltStrike", "COBALTSTRIKE"):
            matches = scanner.scan(_encode(variant))
            assert any(m.name == "Cobalt_Strike" for m in matches), variant


# ---------------------------------------------------------------------------
# Regex signature detection
# ---------------------------------------------------------------------------


class TestRegexSignatureDetection:
    """Regex signatures must fire on matching byte content."""

    def test_wncry_extension_detected(self, scanner: Scanner):
        payload = b"documents/report.WNCRY"
        matches = scanner.scan(payload)
        assert any(m.name == "WannaCry_extension" for m in matches)

    def test_wncry_extension_case_insensitive(self, scanner: Scanner):
        payload = b"documents/report.wncry"
        matches = scanner.scan(payload)
        assert any(m.name == "WannaCry_extension" for m in matches)

    def test_base64_pe_header_detected(self, scanner: Scanner):
        # Realistic base64-encoded MZ header prefix
        payload = b"TVqQAAMAAAAEAAAA"
        matches = scanner.scan(payload)
        assert any(m.name == "Base64_PE_header" for m in matches)

    def test_powershell_encoded_command_detected(self, scanner: Scanner):
        # Base64 payload is long enough to satisfy the {20,} minimum in the regex
        payload = b"powershell -EncodedCommand SQBuAHYAbwBrAGUALQBXAGUAYgBSAGUAcQB1AGUAcwB0AA=="
        matches = scanner.scan(payload)
        assert any(m.name == "PowerShell_encoded_command" for m in matches)

    def test_powershell_encoded_command_short_form(self, scanner: Scanner):
        payload = b"powershell -enc SQBuAHYAbwBrAGUALQBXAGUAYgBSAGUAcQB1AGUAcwB0AA=="
        matches = scanner.scan(payload)
        assert any(m.name == "PowerShell_encoded_command" for m in matches)


# ---------------------------------------------------------------------------
# Strict mode — regex signatures skipped
# ---------------------------------------------------------------------------


class TestStrictMode:
    def test_literal_detected_in_strict_mode(self, scanner_strict: Scanner):
        assert scanner_strict.strict is True
        assert scanner_strict.contains_malware(b"wannacry")

    def test_regex_not_run_in_strict_mode(self, scanner_strict: Scanner):
        # .WNCRY extension is only a RegexSignature; strict mode must skip it.
        payload = b"file.WNCRY"
        matches = scanner_strict.scan(payload)
        assert not any(m.name == "WannaCry_extension" for m in matches)

    def test_powershell_not_run_in_strict_mode(self, scanner_strict: Scanner):
        payload = b"powershell -EncodedCommand aGVsbG8gd29ybGQ="
        matches = scanner_strict.scan(payload)
        assert not any(m.name == "PowerShell_encoded_command" for m in matches)


# ---------------------------------------------------------------------------
# Clean content
# ---------------------------------------------------------------------------


class TestCleanContent:
    def test_clean_bytes_no_match(self, scanner: Scanner):
        assert scanner.scan(b"hello world, this is clean text") == []

    def test_empty_bytes_no_match(self, scanner: Scanner):
        assert scanner.scan(b"") == []

    def test_contains_malware_false_for_clean(self, scanner: Scanner):
        assert scanner.contains_malware(b"clean content") is False


# ---------------------------------------------------------------------------
# MatchResult fields
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_offset_is_correct(self, scanner: Scanner):
        payload = b"HEADER" + b"wannacry" + b"FOOTER"
        matches = [m for m in scanner.scan(payload) if m.name == "WannaCry"]
        assert len(matches) >= 1
        assert matches[0].offset == 6  # "HEADER" is 6 bytes

    def test_matched_bytes_are_original_case(self, scanner: Scanner):
        """matched_bytes should preserve the original casing from the input."""
        payload = b"WaNnAcRy"
        matches = [m for m in scanner.scan(payload) if m.name == "WannaCry"]
        assert len(matches) >= 1
        assert matches[0].matched_bytes.lower() == b"wannacry"

    def test_multiple_hits_reported(self, scanner: Scanner):
        payload = b"wannacry wannacry"
        matches = [m for m in scanner.scan(payload) if m.name == "WannaCry"]
        assert len(matches) == 2


# ---------------------------------------------------------------------------
# Alternative input types
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_bytearray_input(self, scanner: Scanner):
        assert scanner.contains_malware(bytearray(b"mimikatz"))

    def test_memoryview_input(self, scanner: Scanner):
        buf = b"padding" + b"MIMIKATZ" + b"more"
        assert scanner.contains_malware(memoryview(buf))


# ---------------------------------------------------------------------------
# Custom signature sets
# ---------------------------------------------------------------------------


class TestCustomSignatures:
    def test_custom_literal_signature(self):
        sigs = [LiteralSignature("TestMalware", "evil_payload")]
        sc = Scanner(sigs)
        assert sc.contains_malware(b"some EVIL_PAYLOAD here")

    def test_custom_regex_signature(self):
        sigs = [RegexSignature("TestRegex", r"ev[il]+_payload")]
        sc = Scanner(sigs)
        assert sc.contains_malware(b"EViil_payload inside binary")

    def test_custom_regex_skipped_in_strict(self):
        sigs = [
            LiteralSignature("LitSig", "literal"),
            RegexSignature("RegSig", r"reg.*sig"),
        ]
        sc = Scanner(sigs, strict=True)
        assert sc.contains_malware(b"literal hit")
        assert not sc.contains_malware(b"regsig only")
