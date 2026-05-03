"""Unit tests for ``app.safety.email_guard._is_trusted_sender``.

Covers:
- Valid sender whose domain exactly matches a configured trusted domain.
- Spoof attempt where a trusted domain appears as a *subdomain* of an
  attacker-controlled domain (e.g. ``evil@company.com.attacker.com``).
- Various malformed and edge-case inputs that must return ``False`` without
  raising any exceptions.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.safety.email_guard import _is_trusted_sender


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Valid senders
# ---------------------------------------------------------------------------


class TestTrustedSenders:
    def test_plain_address_accepted(self):
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com"]),
        ):
            assert _is_trusted_sender("user@company.com") is True

    def test_mixed_case_address_accepted(self):
        """Domain comparison must be case-insensitive."""
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com"]),
        ):
            assert _is_trusted_sender("User@Company.COM") is True

    def test_display_name_address_accepted(self):
        """RFC 5322 display-name form — addr-spec domain must still match."""
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com"]),
        ):
            assert _is_trusted_sender("Alice Smith <alice@company.com>") is True

    def test_multiple_trusted_domains(self):
        """A sender from the second configured domain must be accepted."""
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com", "partner.org"]),
        ):
            assert _is_trusted_sender("contact@partner.org") is True


# ---------------------------------------------------------------------------
# Spoofing / untrusted senders
# ---------------------------------------------------------------------------


class TestUntrustedSenders:
    def test_subdomain_spoof_rejected(self):
        """evil@company.com.attacker.com must NOT be trusted."""
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com"]),
        ):
            assert _is_trusted_sender("evil@company.com.attacker.com") is False

    def test_prefix_spoof_rejected(self):
        """evil@notcompany.com must NOT be trusted."""
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com"]),
        ):
            assert _is_trusted_sender("evil@notcompany.com") is False

    def test_subdomain_of_trusted_rejected(self):
        """sub.company.com is not the same as company.com — must be rejected."""
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com"]),
        ):
            assert _is_trusted_sender("user@sub.company.com") is False

    def test_completely_different_domain_rejected(self):
        with patch(
            "app.safety.email_guard._get_trusted_domains",
            return_value=frozenset(["company.com"]),
        ):
            assert _is_trusted_sender("user@attacker.com") is False

    def test_no_trusted_domains_configured(self):
        """When no domains are configured nobody is trusted."""
        with patch(
            "app.safety.email_guard._get_trusted_domains", return_value=frozenset()
        ):
            assert _is_trusted_sender("user@company.com") is False


# ---------------------------------------------------------------------------
# Malformed inputs — must return False, never raise
# ---------------------------------------------------------------------------

MALFORMED_INPUTS = [
    "",  # empty string
    "   ",  # whitespace only
    "notanemail",  # no @ at all
    "@nodomain",  # missing local part
    "nolocal@",  # missing domain
    "@",  # just @
    "user@@double.com",  # double @
    "user@[invalid",  # malformed domain
    "<>",  # empty angle-bracket address
    "display only",  # display name, no addr-spec
    "user@" + "a" * 1000,  # very long domain (not in trusted list)
]


@pytest.mark.parametrize("bad_input", MALFORMED_INPUTS)
def test_malformed_input_returns_false_without_exception(bad_input: str):
    """_is_trusted_sender must never raise for any input."""
    with patch(
        "app.safety.email_guard._get_trusted_domains",
        return_value=frozenset(["company.com"]),
    ):
        result = _is_trusted_sender(bad_input)
    assert result is False


def test_non_string_input_returns_false():
    """Passing a non-string value must return False, not raise."""
    with patch(
        "app.safety.email_guard._get_trusted_domains",
        return_value=frozenset(["company.com"]),
    ):
        assert _is_trusted_sender(None) is False  # type: ignore[arg-type]
        assert _is_trusted_sender(123) is False  # type: ignore[arg-type]
        assert _is_trusted_sender([]) is False  # type: ignore[arg-type]
