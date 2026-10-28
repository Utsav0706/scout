"""Mask personal data and secrets before they reach logs or the audit table.

Questions, search queries and page excerpts can contain emails, phone numbers and the
like; logs are kept longer and seen by more people than the data itself, so they get
masked versions. Secrets (API keys, tokens) are masked too, in case one is ever pasted
into a question or shows up in an error message.

Rule-based, so: no cost, same answer every time, testable. It won't catch everything
(names and addresses need a trained model); it covers the formats that are easy to
recognise and most harmful to leak.
"""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# UK numbers: +44 or a leading 0, then 9-10 more digits with optional spaces.
_PHONE = re.compile(r"(?<![\w.])(?:\+44\s?|0)(?:\d[\s-]?){8,9}\d(?!\d)")
# National Insurance number, e.g. AB 12 34 56 C (case-sensitive: they're upper case).
_NI = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")
# 13-19 digits, optionally grouped: card-number shaped. Confirmed with the Luhn check.
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SECRETS = [
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}"),  # Anthropic API key
    re.compile(r"\btvly-[A-Za-z0-9_-]{10,}"),  # Tavily API key
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),  # OpenAI-style key
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"),  # JWT
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{10,}=*"),
]


def _luhn_ok(digits: str) -> bool:
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        double = not double
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    # Only real card numbers: long digit runs in statistics shouldn't be masked.
    return "[card]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


def redact(text: str) -> str:
    for pattern in _SECRETS:
        text = pattern.sub("[secret]", text)
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub(_card, text)  # before phone: a card number also looks like digits
    text = _NI.sub("[ni-number]", text)
    return _PHONE.sub("[phone]", text)


def redact_value(value: Any) -> Any:
    """redact() applied to every string inside dicts/lists/tuples (log fields, tool args)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(redact_value(v) for v in value)
    return value
