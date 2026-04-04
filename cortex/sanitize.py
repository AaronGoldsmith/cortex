"""Secret detection and redaction for text sent to external LLM APIs."""

import re

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Private keys
    (re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC )?PRIVATE KEY-----"), "private_key"),
    # Specific API key prefixes
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "anthropic_key"),
    (re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"), "openai_project_key"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{32,}"), "api_key"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "github_pat"),
    (re.compile(r"gho_[A-Za-z0-9]{36,}"), "github_oauth"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github_pat"),
    (re.compile(r"glpat-[A-Za-z0-9\-]{20,}"), "gitlab_pat"),
    (re.compile(r"xox[bp]-[A-Za-z0-9\-]{20,}"), "slack_token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key"),
    # Connection strings with credentials
    (re.compile(r"(?:mongodb\+srv|postgres|postgresql|mysql|redis)://[^\s:]+:[^\s@]+@[^\s]+"), "connection_string"),
    (re.compile(r"://[^\s:]+:[^\s@]+@[^\s]+"), "credential_url"),
    # Env var assignments with secret-looking values
    (re.compile(r"(?:PASSWORD|SECRET|TOKEN|API_KEY|APIKEY|API_SECRET)\s*=\s*['\"]?[^\s'\"]{8,}['\"]?", re.IGNORECASE), "env_secret"),
    # Generic long tokens (32+ hex or base64-ish chars, bounded by non-alnum)
    (re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9_])"), "token"),
]


def sanitize(text: str) -> str:
    """Scan text for secret patterns and return sanitized version."""
    for pattern, label in _PATTERNS:
        text = pattern.sub(f"[REDACTED:{label}]", text)
    return text


def has_secrets(text: str) -> bool:
    """Quick check whether text contains any secret patterns."""
    return any(pattern.search(text) for pattern, _ in _PATTERNS)
