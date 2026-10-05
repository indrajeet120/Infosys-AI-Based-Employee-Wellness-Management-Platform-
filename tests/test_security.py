"""
Security & Data Privacy Validation Suite (Milestone 4 Task 8).
Verifies input sanitization, user scoping authorization, and secret redaction.
"""

import pytest
from services.security import SecurityValidator


def test_sanitize_html_and_script_injection():
    malicious_input = "<script>alert('xss')</script> Hello world! <a href='http://bad.com'>Click</a>"
    clean_text = SecurityValidator.sanitize_text_input(malicious_input)

    assert "<script>" not in clean_text
    assert "alert('xss')" not in clean_text
    assert "Hello world!" in clean_text


def test_sanitize_max_length_clamping():
    long_input = "A" * 6000
    clean_text = SecurityValidator.sanitize_text_input(long_input, max_length=5000)

    assert len(clean_text) == 5000


def test_user_scoping_authorization():
    assert SecurityValidator.validate_user_authorization(requesting_user="emp_01", target_user="emp_01") is True
    assert SecurityValidator.validate_user_authorization(requesting_user="admin", target_user="emp_01", is_admin=True) is True
    assert SecurityValidator.validate_user_authorization(requesting_user="emp_02", target_user="emp_01") is False


def test_secret_and_pii_redaction():
    log_line = "User emp_01 auth failed with api_key=sk-proj-1234567890abcdef and password=SuperSecret123!"
    redacted = SecurityValidator.redact_sensitive_information(log_line)

    assert "sk-proj-1234567890abcdef" not in redacted
    assert "SuperSecret123!" not in redacted
    assert "[REDACTED_API_KEY]" in redacted or "[REDACTED_PASSWORD]" in redacted
