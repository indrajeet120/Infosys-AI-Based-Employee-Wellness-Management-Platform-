"""
Task 8: Security & Data Privacy Validation Service.
Provides input validation, sanitization, user-scoping authorization, secret redaction, and privacy checks.
"""

import re
from typing import Dict, List, Any, Optional, Tuple


class SecurityValidator:
    """Provides security, data protection, and privacy verification methods."""

    @staticmethod
    def validate_and_sanitize_input(text: Optional[str], max_length: int = 2000) -> Tuple[bool, str, Optional[str]]:
        """
        Validates text input for length limits, nulls, whitespace, and strips dangerous script/HTML tags.
        Returns: (is_valid, sanitized_text, error_message)
        """
        if text is None:
            return False, "", "Input text cannot be null."
        
        s_text = str(text).strip()
        if not s_text:
            return False, "", "Input text cannot be empty or only whitespace."

        if len(s_text) > max_length:
            return False, s_text[:max_length], f"Input text exceeds maximum allowed length of {max_length} characters."

        # Strip dangerous HTML/script tags
        clean_text = re.sub(r"<script.*?>.*?</script>", "", s_text, flags=re.IGNORECASE | re.DOTALL)
        clean_text = re.sub(r"<[^>]*>", "", clean_text)
        clean_text = clean_text.strip()

        if not clean_text:
            return False, "", "Input text contains only invalid HTML or script tags."

        return True, clean_text, None

    @staticmethod
    def authorize_user_access(requesting_user_id: str, target_user_id: str) -> bool:
        """
        Verifies user scoping: ensures a requesting user can only access their own private records.
        """
        if not requesting_user_id or not target_user_id:
            return False
        return requesting_user_id.strip().lower() == target_user_id.strip().lower()

    @staticmethod
    def redact_secrets(data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redacts sensitive fields (passwords, tokens, api_keys, secrets) from dictionaries before logging/export.
        """
        sensitive_keys = {"password", "secret", "token", "api_key", "auth", "bearer", "private_key"}
        sanitized = {}

        for key, value in data_dict.items():
            if any(s_key in str(key).lower() for s_key in sensitive_keys):
                sanitized[key] = "********"
            elif isinstance(value, dict):
                sanitized[key] = SecurityValidator.redact_secrets(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    SecurityValidator.redact_secrets(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 5000) -> str:
        """Convenience method returning sanitized text string."""
        _, clean, _ = SecurityValidator.validate_and_sanitize_input(text, max_length=max_length)
        return clean

    @staticmethod
    def validate_user_authorization(requesting_user: str, target_user: str, is_admin: bool = False) -> bool:
        """Convenience method checking user scoping authorization."""
        if is_admin:
            return True
        return SecurityValidator.authorize_user_access(requesting_user, target_user)

    @staticmethod
    def redact_sensitive_information(text_or_dict: Any) -> Any:
        """Redacts sensitive strings or dict fields."""
        if isinstance(text_or_dict, dict):
            return SecurityValidator.redact_secrets(text_or_dict)
        s = str(text_or_dict)
        s = re.sub(r"sk-[a-zA-Z0-9\-_]{10,}", "[REDACTED_API_KEY]", s)
        s = re.sub(r"password=\S+", "password=[REDACTED_PASSWORD]", s)
        return s


# Import Tuple for type annotations
from typing import Tuple
