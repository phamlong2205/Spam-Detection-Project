# src/spam_detection/data_filters.py
import re
from typing import Iterable

NEWSWIRE_HINTS: Iterable[str] = (
    "reuters", "news service", "copyright", "all rights reserved",
    "for immediate release", "press release"
)

PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")
URL_RE   = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

def looks_like_newswire(text: str) -> bool:
    t = text.lower()
    hits = sum(h in t for h in NEWSWIRE_HINTS)
    return hits >= 1

def has_many_phones(text: str, min_hits: int = 3) -> bool:
    return len(PHONE_RE.findall(text)) >= min_hits

def has_many_urls(text: str, min_hits: int = 2) -> bool:
    return len(URL_RE.findall(text)) >= min_hits

def has_many_emails(text: str, min_hits: int = 2) -> bool:
    return len(EMAIL_RE.findall(text)) >= min_hits

def special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    specials = sum(ch in r"!@#$%^&*()[]{}<>|\/~`+=:_;\"'.,?-–—" for ch in text)
    return specials / max(1, len(text))

def is_dirty_row(row) -> bool:
    """
    Heuristics to drop extremely noisy/non-message-like rows.
    Works with your existing feature columns if present, but
    also falls back to raw text safely.
    """
    message = (row.get("cleaned_message") or row.get("message") or "").strip()
    if not message:
        return True

    # 1) Length-based outliers (very long news text / dumps)
    msg_len = len(message)
    if msg_len > 1600:  # conservative hard cap
        return True

    # 2) If features exist, leverage them
    digit_ratio = row.get("digit_ratio", None)
    cap_ratio   = row.get("capital_ratio", None)
    spec_count  = row.get("special_char_count", None)

    # A) too numeric and too many specials (e.g., phone dumps)
    if digit_ratio is not None and spec_count is not None:
        if digit_ratio > 0.18 and spec_count > 20:
            return True

    # B) excessive punctuation vs length
    if special_char_ratio(message) > 0.12:
        return True

    # 3) Pattern-based: repeated phones/urls/emails
    if has_many_phones(message) or has_many_urls(message) or has_many_emails(message):
        return True

    # 4) Newswire/press style text
    if looks_like_newswire(message):
        return True

    return False
