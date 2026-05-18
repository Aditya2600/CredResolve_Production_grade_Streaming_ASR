"""Identifier formatters: PAN, Aadhaar, IFSC, and cue-gated generic IDs.

All four parsers are deterministic regex + structural validators. They
never fuzzy-match, never auto-correct, and never accept partial input.
The contract is the stage-2 deliverable's: "if in doubt, emit raw".

* :func:`parse_pan`      — PAN, strict ``^[A-Z]{5}\\d{4}[A-Z]$``.
* :func:`parse_aadhaar`  — 12-digit UID with optional spacing; the
                            Verhoeff checksum is validated by default
                            (see ``validate_checksum``).
* :func:`parse_ifsc`     — IFSC, strict ``^[A-Z]{4}0[A-Z0-9]{6}$``.
* :func:`parse_generic_id` — alphanumeric ID, only when an explicit cue
                              word (account/policy/loan/customer/
                              reference + Hindi variants) appears in the
                              5-token left context.

Each function returns the canonical form on success or ``None`` when
the input fails its structural rule. No partial repair, no "did you
mean" — the caller is expected to emit raw for ``None`` returns.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Strict regexes.
# ---------------------------------------------------------------------------

_PAN_RE: re.Pattern[str] = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_IFSC_RE: re.Pattern[str] = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
_AADHAAR_OUTER_RE: re.Pattern[str] = re.compile(r"^[\d \-]+$")
_AADHAAR_DIGITS_RE: re.Pattern[str] = re.compile(r"^\d{12}$")
_GENERIC_ID_RE: re.Pattern[str] = re.compile(r"^[A-Z0-9]{4,20}$")
_GENERIC_ID_SEPARATOR_RE: re.Pattern[str] = re.compile(r"[ \-\./]")

# Cue words for generic alphanumeric IDs. Lower-case for matching;
# punctuation is stripped from each context token before lookup.
_ID_CUES: frozenset[str] = frozenset({
    # English
    "account", "acct", "a/c", "ac",
    "policy",
    "loan",
    "customer", "cust",
    "reference", "ref",
    "id",
    "number", "no",
    # Hindi
    "खाता", "खाते",
    "अकाउंट", "अकाउण्ट",
    "पॉलिसी", "पालिसी",
    "लोन", "ऋण",
    "ग्राहक", "कस्टमर",
    "रेफरेंस", "रेफ़रेंस", "संदर्भ",
    "आईडी", "नंबर",
})

def _clean_cue_token(token: str) -> str:
    """Strip punctuation while preserving Indic combining marks."""
    return "".join(
        ch
        for ch in token
        if ch == "/" or not unicodedata.category(ch).startswith("P")
    ).lower()


# ---------------------------------------------------------------------------
# Verhoeff checksum (UIDAI Aadhaar).
#
# Tables per the Verhoeff (1969) standard; validation walks the digit
# string from rightmost to leftmost. A valid 12-digit Aadhaar yields
# ``c == 0`` after the full walk. See e.g. UIDAI's "Verhoeff algorithm
# for Aadhaar number" reference and Wikipedia's Verhoeff entry for a
# review of the constants.
# ---------------------------------------------------------------------------

_VERHOEFF_D: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_VERHOEFF_P: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

_VERHOEFF_INV: tuple[int, ...] = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_validate(num: str) -> bool:
    """Return ``True`` iff ``num`` (a digit string) passes Verhoeff."""
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_compute(num_without_check: str) -> str:
    """Compute the Verhoeff check digit for ``num_without_check``.

    Useful for generating valid synthetic Aadhaar numbers in tests; the
    runtime path only needs :func:`verhoeff_validate`.
    """
    c = 0
    for i, ch in enumerate(reversed(num_without_check)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


# ---------------------------------------------------------------------------
# Public parsers.
# ---------------------------------------------------------------------------

def parse_pan(text: str) -> str | None:
    """Strict PAN parser. Uppercases input and validates regex shape.

    Returns the canonical 10-character PAN, or ``None``.
    """
    if not text:
        return None
    candidate = text.strip().upper()
    if _PAN_RE.match(candidate):
        return candidate
    return None


def parse_aadhaar(text: str, *, validate_checksum: bool = True) -> str | None:
    """Aadhaar parser.

    Args:
        text: candidate span (12 digits, optionally separated by single
            spaces or hyphens).
        validate_checksum: when ``True`` (default), the Verhoeff check
            digit must be valid. The deliverable spec marks this as
            optional-but-recommended; we default to ON because rejecting
            checksum-invalid spans is free precision.

    Returns ``"XXXX XXXX XXXX"`` (canonical 4-4-4 grouping) or ``None``.
    """
    if not text:
        return None
    candidate = text.strip()
    if not _AADHAAR_OUTER_RE.match(candidate):
        return None
    digits = candidate.replace(" ", "").replace("-", "")
    if not _AADHAAR_DIGITS_RE.match(digits):
        return None
    if validate_checksum and not verhoeff_validate(digits):
        return None
    return f"{digits[:4]} {digits[4:8]} {digits[8:]}"


def parse_ifsc(text: str) -> str | None:
    """Strict IFSC parser. Uppercases input and validates regex shape.

    Returns the canonical 11-character IFSC, or ``None``.
    """
    if not text:
        return None
    candidate = text.strip().upper()
    if _IFSC_RE.match(candidate):
        return candidate
    return None


def parse_generic_id(
    text: str,
    context_tokens: list[str] | tuple[str, ...],
    *,
    window: int = 5,
) -> str | None:
    """Cue-gated generic alphanumeric ID parser.

    Per the stage-2 spec: the ID is only normalised when an explicit
    cue word (``account`` / ``policy`` / ``loan`` / ``customer`` /
    ``reference`` and Hindi variants) appears in the ``window`` tokens
    immediately preceding the candidate. Otherwise ``None``.

    Args:
        text: candidate alphanumeric span. Internal ``-``/``/``/``.``/
            spaces are stripped; the result is uppercased.
        context_tokens: tokens that precede the candidate, oldest first
            (most-recent at the end of the list).
        window: maximum number of preceding tokens to scan for a cue.

    Returns the canonical (uppercased, separator-free) ID, or ``None``
    when no cue is present, when the candidate is not alphanumeric, or
    when the canonical length is outside [4, 20].
    """
    if not text:
        return None
    if window < 1:
        return None
    recent = context_tokens[-window:] if len(context_tokens) >= window else context_tokens
    if not any(_clean_cue_token(t) in _ID_CUES for t in recent):
        return None
    compact = _GENERIC_ID_SEPARATOR_RE.sub("", text.strip()).upper()
    if not _GENERIC_ID_RE.match(compact):
        return None
    return compact


__all__ = [
    "parse_pan",
    "parse_aadhaar",
    "parse_ifsc",
    "parse_generic_id",
    "verhoeff_validate",
    "verhoeff_compute",
]
