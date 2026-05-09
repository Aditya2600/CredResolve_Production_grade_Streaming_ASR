"""Indian mobile phone formatter (digit-by-digit spoken or Latin -> ``+91 XXXXX XXXXX``).

Pure regex + structural validator. Accepts:

* 10-digit Indian mobile (leading [6-9]).
* Optionally prefixed with ``+91``, ``91``, or a leading ``0``.
* Digits given as Latin (0-9), Devanagari (०-९), or as Hindi digit
  words (``शून्य``/``जीरो``, ``एक``, ``दो``, ..., ``नौ``).
* Common visual separators (single spaces, ``-``, ``.``, ``(``, ``)``).

Rejects everything else. **Never** guesses missing digits — if the input
contains 9 or 11 digit positions, returns ``None`` rather than infer a
plausible mobile number. This is the stage-2 deliverable contract: the
default policy on any ID-class uncertainty is to emit raw, not to
"helpfully" repair.

Returns the canonical form ``+91 XXXXX XXXXX`` on success.
"""

from __future__ import annotations

import re

# Hindi spoken digit words -> Latin digit char.
_SPOKEN_TO_DIGIT: dict[str, str] = {
    "शून्य": "0", "जीरो": "0", "ज़ीरो": "0",
    "एक": "1",
    "दो": "2",
    "तीन": "3",
    "चार": "4",
    "पाँच": "5", "पांच": "5",
    "छह": "6", "छः": "6",
    "सात": "7",
    "आठ": "8",
    "नौ": "9",
}

# Devanagari digit glyphs -> Latin (single char).
_NATIVE_TRANS: dict[int, str] = {
    0x0966: "0", 0x0967: "1", 0x0968: "2", 0x0969: "3", 0x096A: "4",
    0x096B: "5", 0x096C: "6", 0x096D: "7", 0x096E: "8", 0x096F: "9",
}

# Allowed visual separators inside / between digit blocks.
_SEPARATORS_RE: re.Pattern[str] = re.compile(r"[ \-\.\(\)]+")

# Final-shape canonical regex: optional +91/91/0 prefix, then 10 digits leading [6-9].
_CANONICAL_RE: re.Pattern[str] = re.compile(r"^(?:\+91|91|0)?([6-9]\d{9})$")


def _digitify_token(t: str) -> str | None:
    """Map a single whitespace-separated token to its digit chunk.

    Returns:
        * ``""`` if the token is purely separator characters.
        * a digit string (optionally prefixed by a single ``+``) if the
          token is a recognised digit-word, native digit glyph, Latin
          digit run, or compact ``+91`` / ``91`` prefix.
        * ``None`` if the token is not recognised.
    """
    if not t:
        return ""
    if t in _SPOKEN_TO_DIGIT:
        return _SPOKEN_TO_DIGIT[t]
    t = t.translate(_NATIVE_TRANS)
    cleaned = _SEPARATORS_RE.sub("", t)
    if not cleaned:
        return ""
    if cleaned.startswith("+"):
        body = cleaned[1:]
        if not body or not body.isdigit():
            return None
        return "+" + body
    if cleaned.isdigit():
        return cleaned
    return None


def parse_indian_mobile(text: str) -> str | None:
    """Return canonical ``"+91 XXXXX XXXXX"`` or ``None`` for invalid input.

    Strict by construction:
        * No fuzzy matching (lookalike / OCR substitutions are rejected).
        * No digit-count guessing (9 or 11 digits -> ``None``).
        * No prefix invention (a span with ``+44`` is not relabelled +91).
    """
    if not text or not text.strip():
        return None
    parts: list[str] = []
    for tok in text.strip().split():
        chunk = _digitify_token(tok)
        if chunk is None:
            return None
        parts.append(chunk)
    joined = "".join(parts)
    # Drop any separators that survived per-token cleanup (e.g. embedded
    # inside a single token like "9876-543-210").
    joined = _SEPARATORS_RE.sub("", joined)
    if not joined:
        return None
    m = _CANONICAL_RE.match(joined)
    if not m:
        return None
    d = m.group(1)
    return f"+91 {d[:5]} {d[5:]}"


__all__ = ["parse_indian_mobile"]
