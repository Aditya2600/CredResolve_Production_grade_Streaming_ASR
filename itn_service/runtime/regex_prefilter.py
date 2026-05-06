"""Deterministic regex prefilter.

Cheap, conservative pre-tagging for classes the WFST classifier would
otherwise have to derive from scratch. The intent is to *tag and
locate*, not normalise — every prefilter span carries `canonical ==
raw` and `conf == 1.0` because the regex match is itself the evidence;
downstream stages own the rewrite.

Patterns implemented (rule_id format: ``prefilter.<class>.v1``):

* ``URL``           — ``http(s)://...`` and bare ``www.<host>...``
* ``EMAIL``         — practical RFC subset
* ``IFSC``          — Indian bank IFSC code (``[A-Z]{4}0[A-Z0-9]{6}``)
* ``PAN``           — Indian PAN (``[A-Z]{5}\\d{4}[A-Z]``)
* ``AADHAAR``       — 12-digit UID, optional spaces / dashes
* ``PHONE_LATN``    — Indian mobile, optional ``+91`` / leading 0
* ``DATE_NUMERIC``  — ``dd/mm/yy(yy)`` and ``-`` / ``.`` separators
* ``TIME_NUMERIC``  — ``hh:mm[:ss][ AM|PM]``
* ``AMOUNT_LATN``   — leading currency symbol or ISO code + digits
* ``PERCENT_LATN``  — ``\\d+(\\.\\d+)? %``

Overlap resolution is priority-first, greedy: URL > EMAIL > IFSC > PAN
> AADHAAR > PHONE > DATE > TIME > AMOUNT > PERCENT. That ordering is
chosen so identifiers (which carry concrete structural rules) win over
their numeric look-alikes — e.g. a 12-digit Aadhaar dominates a phone-
length suffix; a URL eats any phone-shaped digit run inside its path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contract import Span


@dataclass(frozen=True)
class _Spec:
    cls: str
    rule_id: str
    priority: int  # lower wins in overlap resolution
    pattern: re.Pattern[str]


# --- patterns -----------------------------------------------------------------

# URLs first — they swallow phone-shaped or amount-shaped substrings.
_URL_RE = re.compile(
    r"(?:(?:https?|ftp)://[^\s<>\"']+"
    r"|\bwww\.[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+(?:/[^\s<>\"']*)?)",
    re.IGNORECASE,
)

# Email — practical subset; not full RFC 5322.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+\b"
)

# IFSC: 4 letters, fixed '0', then 6 alphanumeric. Case-insensitive at
# match time but downstream uses raw text. The fixed '0' at position 5
# is what reliably distinguishes IFSC from generic alphanumerics.
_IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)

# PAN: 5 letters + 4 digits + 1 letter.
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)

# Aadhaar: UIDAI rule says first digit is in 2-9; total 12 digits with
# optional ' ' or '-' separators every 4 digits. The leading negative
# lookbehind blocks matches sitting immediately after a '+' — that
# "+91 + 10 digits" shape is unambiguously a phone number with country
# code, and the phone pattern should win.
_AADHAAR_RE = re.compile(r"(?<!\+)\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b")

# Indian mobile: optional '+91'/'0' prefix, then 10 digits starting 6-9.
# Lookbehind / lookahead block adjacent letters or digits so a 10-digit
# substring of a 15-digit alphanumeric run is not falsely tagged.
_PHONE_RE = re.compile(
    r"(?<![A-Za-z\d])"
    r"(?:(?:\+|00)?91[\s\-\.]?|0(?=[6-9]))?"
    r"[6-9]\d{9}"
    r"(?![A-Za-z\d])"
)

# Numeric date: dd[sep]mm[sep]yy(yy) with /, -, or . separators. Leap-
# year and 30/31-day validation belongs in the WFST, not here.
_DATE_RE = re.compile(
    r"\b(?:0?[1-9]|[12]\d|3[01])[\/\-\.](?:0?[1-9]|1[0-2])[\/\-\.]\d{2}(?:\d{2})?\b"
)

# 12 / 24-hour time. AM/PM is optional.
_TIME_RE = re.compile(
    r"(?<![A-Za-z\d:])"
    r"(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?"
    r"(?:\s?(?:AM|PM|am|pm|a\.m\.|p\.m\.))?"
    r"(?![A-Za-z\d:])"
)

# Currency-prefixed amount. Indian grouping (1,25,000) is permitted by
# allowing 2-or-3-digit comma groups; full validation is the WFST's job.
_AMOUNT_RE = re.compile(
    r"(?:₹|Rs\.?|INR|US\$|\$|USD|£|GBP|€|EUR)"
    r"\s?\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?",
    re.IGNORECASE,
)

# Percentage: a decimal then a literal %.
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")


_SPECS: tuple[_Spec, ...] = (
    _Spec("url", "prefilter.url.v1", 1, _URL_RE),
    _Spec("email", "prefilter.email.v1", 2, _EMAIL_RE),
    _Spec("ifsc", "prefilter.ifsc.v1", 3, _IFSC_RE),
    _Spec("pan", "prefilter.pan.v1", 4, _PAN_RE),
    _Spec("aadhaar", "prefilter.aadhaar.v1", 5, _AADHAAR_RE),
    _Spec("phone", "prefilter.phone.v1", 6, _PHONE_RE),
    _Spec("date", "prefilter.date.v1", 7, _DATE_RE),
    _Spec("time", "prefilter.time.v1", 8, _TIME_RE),
    _Spec("amount", "prefilter.amount.v1", 9, _AMOUNT_RE),
    _Spec("percent", "prefilter.percent.v1", 10, _PERCENT_RE),
)


def prefilter(text: str) -> list[Span]:
    """Return non-overlapping prefilter spans, sorted by start offset.

    Each `Span` has ``canonical == raw``; the prefilter does not
    rewrite content. Spans carry ``start`` and ``end`` codepoint
    offsets into `text` so downstream stages can splice them back in.
    """
    if not text:
        return []

    # Collect all candidates from all patterns.
    candidates: list[tuple[int, int, _Spec]] = []
    for spec in _SPECS:
        for m in spec.pattern.finditer(text):
            start, end = m.start(), m.end()
            if end > start:
                candidates.append((start, end, spec))
    if not candidates:
        return []

    # Resolve overlaps: walk in (priority, start) order; accept a
    # candidate only if it doesn't overlap with one already accepted.
    candidates.sort(key=lambda c: (c[2].priority, c[0]))
    selected: list[tuple[int, int, _Spec]] = []
    for start, end, spec in candidates:
        if any(s < end and start < e for s, e, _ in selected):
            continue
        selected.append((start, end, spec))

    selected.sort(key=lambda c: c[0])
    return [
        Span(
            cls=spec.cls,
            raw=text[s:e],
            canonical=text[s:e],
            rule_id=spec.rule_id,
            conf=1.0,
            ambiguous=False,
            start=s,
            end=e,
        )
        for s, e, spec in selected
    ]


__all__ = ["prefilter"]
