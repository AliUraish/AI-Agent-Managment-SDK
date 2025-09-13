"""Global privacy/config settings for MYSDK.

Use `configure` to set:
- capture_content: bool (default True) — whether to record raw prompt/completion content
- redactor: Optional[Callable[[str], str]] — redact text before storing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class _Settings:
    capture_content: bool = True
    redactor: Optional[Callable[[str], str]] = None


settings = _Settings()


def configure(*, capture_content: Optional[bool] = None, redactor: Optional[Callable[[str], str]] = None) -> None:
    if capture_content is not None:
        settings.capture_content = bool(capture_content)
    if redactor is not None:
        settings.redactor = redactor


def maybe_redact(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    if not settings.capture_content:
        return None
    if settings.redactor is not None:
        try:
            return settings.redactor(text)
        except Exception:
            return text
    return text

