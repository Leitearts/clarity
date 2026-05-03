"""Unit tests for app.utils.network.is_internal_ip."""
from __future__ import annotations

import pytest

from app.utils.network import is_internal_ip


class TestIsInternalIp:
    # ── RFC-1918 private ranges ──────────────────────────────────────────────

    def test_192_168_is_internal(self):
        assert is_internal_ip("192.168.1.1") is True

    def test_10_x_is_internal(self):
        assert is_internal_ip("10.0.0.1") is True

    def test_172_20_is_internal(self):
        """172.20.x.x is inside the 172.16.0.0/12 range and must be internal."""
        assert is_internal_ip("172.20.0.1") is True

    def test_172_16_is_internal(self):
        assert is_internal_ip("172.16.0.1") is True

    def test_172_31_is_internal(self):
        """172.31.x.x is the last block of 172.16.0.0/12."""
        assert is_internal_ip("172.31.255.255") is True

    # ── Public addresses ─────────────────────────────────────────────────────

    def test_google_dns_is_not_internal(self):
        assert is_internal_ip("8.8.8.8") is False

    def test_172_32_is_not_internal(self):
        """172.32.x.x is just outside the 172.16.0.0/12 private range."""
        assert is_internal_ip("172.32.0.1") is False

    def test_public_address_is_not_internal(self):
        assert is_internal_ip("1.1.1.1") is False

    # ── Invalid / malformed inputs ────────────────────────────────────────────

    def test_not_an_ip_returns_false(self):
        assert is_internal_ip("not-an-ip") is False

    def test_empty_string_returns_false(self):
        assert is_internal_ip("") is False

    def test_invalid_ip_does_not_raise(self):
        """Malformed input must never raise an unhandled exception."""
        result = is_internal_ip("999.999.999.999")
        assert result is False

    def test_none_like_string_returns_false(self):
        assert is_internal_ip("None") is False

    def test_hostname_returns_false(self):
        assert is_internal_ip("localhost") is False
