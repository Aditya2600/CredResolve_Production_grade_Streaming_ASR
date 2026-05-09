"""Strict dateparser fallback — gated, last-resort date normaliser.

The WFST date grammar (``grammars/hi/date.py``) is the primary path. It
covers every form the implementation blueprint mandates and refuses to
fire on ambiguous numeric dates outside a known DMY tenant. This module
runs **only** when:

    1. The WFST returned no parse (``DATE_MONTHWORD`` and, for DMY
       tenants, ``DATE_NUMERIC`` both failed), AND
    2. A date cue word is present in the surrounding context (so we
       don't dateparser-randomly-rewrite arbitrary digits), AND
    3. The classifier and ASR confidences both clear the WFST date
       gate (``classifier_min`` / ``asr_min`` from ``thresholds.yaml``,
       defaulting to 0.85 here per the deliverable spec).

The library itself is configured in strict mode with the tenant's
date order and Hindi + English language hints. Strict mode means
``dateparser`` returns ``None`` rather than guess on partial / fuzzy
input; we want that behaviour.

Returns ``DateParseResult`` with either a canonical ``DD/MM/YYYY``
string or a structured rejection reason. Callers wire the rejection
reason into ``Span.fallback_reason``.

This module is import-safe even when ``dateparser`` is not installed —
the import is lazy and the loader records a clear error when invoked.
That keeps the WFST request path free of an extra dependency unless
the operator opts in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final


# ---------------------------------------------------------------------------
# Cue-word gate.
#
# Only attempt dateparser when the surrounding text already says "this
# is a date". The set is deliberately broad (Hindi + English + Hinglish
# month abbreviations) but never includes naked digit clusters.
# ---------------------------------------------------------------------------

_DATE_CUE_WORDS: Final[frozenset[str]] = frozenset({
    # Hindi month names + variants
    "जनवरी", "फरवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "सितम्बर", "अक्टूबर",
    "नवंबर", "नवम्बर", "दिसंबर", "दिसम्बर",
    # Hindi date / weekday cues
    "तारीख", "तारीख़", "दिनांक", "साल", "वर्ष", "महीना", "माह",
    "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार",
    "शनिवार", "रविवार",
    # English month names / abbreviations (Hinglish input is common)
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug",
    "Sep", "Sept", "Oct", "Nov", "Dec",
    # English date cues
    "date", "dated", "day", "month", "year",
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
})

_LOWER_CUES: Final[frozenset[str]] = frozenset(w.lower() for w in _DATE_CUE_WORDS)

# Quick check for purely numeric "<d>[sep]<d>[sep]<d>" — when this is
# the entire surface and no cue word is in context, we still treat the
# span as "not a date" to satisfy the cue requirement.
_NUMERIC_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*\d{1,4}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{1,4}\s*$"
)


def has_date_cue(text: str) -> bool:
    """Return ``True`` iff a date cue word appears in ``text``.

    Comparison is whitespace-tokenised and case-insensitive on the
    English-script side. Hindi cues match exactly (Devanagari has no
    case folding).
    """
    if not text:
        return False
    for tok in text.split():
        # Strip punctuation that commonly attaches to month names in
        # ASR output ("May,", "June.", "(May)" etc.).
        clean = tok.strip(",.;:!?()[]\"'")
        if not clean:
            continue
        if clean in _DATE_CUE_WORDS:
            return True
        if clean.lower() in _LOWER_CUES:
            return True
    return False


# ---------------------------------------------------------------------------
# Confidence gate.
#
# The deliverable spec ("Wrapped in a confidence gate (>= 0.85)") sets
# the floor; the values below are kept independently editable so the
# WFST date thresholds in thresholds.yaml can drift without forcing a
# code change here. If a caller wants to use the YAML thresholds,
# pass them through ``classifier_min`` / ``asr_min`` explicitly.
# ---------------------------------------------------------------------------

DEFAULT_MIN_CONFIDENCE: Final[float] = 0.85


# ---------------------------------------------------------------------------
# Result type.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DateParseResult:
    """Outcome of a strict dateparser fallback call."""

    canonical: str | None         # "DD/MM/YYYY" on success, else None
    fallback_reason: str | None   # populated when canonical is None


# ---------------------------------------------------------------------------
# Library loader.
# ---------------------------------------------------------------------------

_DATEPARSER_MODULE: Any | None = None
_DATEPARSER_IMPORT_ERROR: Exception | None = None


def _load_dateparser() -> Any:
    """Lazily import ``dateparser`` and cache the module / error.

    Kept lazy because ``dateparser`` is a heavy import and the WFST
    grammar covers the vast majority of production traffic; we don't
    want to pay the import cost just because the runtime module is
    referenced in tests.
    """
    global _DATEPARSER_MODULE, _DATEPARSER_IMPORT_ERROR
    if _DATEPARSER_MODULE is not None:
        return _DATEPARSER_MODULE
    if _DATEPARSER_IMPORT_ERROR is not None:
        raise _DATEPARSER_IMPORT_ERROR
    try:
        import dateparser  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - depends on env
        _DATEPARSER_IMPORT_ERROR = e
        raise
    _DATEPARSER_MODULE = dateparser
    return dateparser


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def try_dateparser_fallback(
    raw: str,
    *,
    locale_date_order: str,
    classifier_conf: float,
    asr_conf: float,
    context_text: str | None = None,
    classifier_min: float = DEFAULT_MIN_CONFIDENCE,
    asr_min: float = DEFAULT_MIN_CONFIDENCE,
    languages: tuple[str, ...] = ("hi", "en"),
) -> DateParseResult:
    """Strict last-resort date parse.

    Args:
        raw: The candidate span text (already classified as a date by
            an upstream signal — usually because the WFST punted).
        locale_date_order: ``"DMY"``, ``"MDY"``, or ``"YMD"``. Passed
            to dateparser as the ``DATE_ORDER`` setting.
        classifier_conf: Span classifier confidence on this span.
        asr_conf: Aggregated ASR confidence over the span tokens.
        context_text: Optional surrounding text used for the cue-word
            gate. When ``None`` we check ``raw`` itself.
        classifier_min: Floor for ``classifier_conf`` (default 0.85).
        asr_min: Floor for ``asr_conf`` (default 0.85).
        languages: dateparser language hints. Default Hindi + English
            covers our Indic + Hinglish surface forms.

    Returns:
        ``DateParseResult``. On rejection ``canonical`` is ``None`` and
        ``fallback_reason`` is one of: ``"missing_date_cue"``,
        ``"low_confidence"``, ``"dateparser_no_parse"``,
        ``"dateparser_unavailable"``.
    """
    # 1. Confidence gate. Fail fast — no point invoking dateparser if
    # the span isn't trusted enough to auto-rewrite anyway.
    if classifier_conf < classifier_min or asr_conf < asr_min:
        return DateParseResult(canonical=None, fallback_reason="low_confidence")

    # 2. Cue-word gate. dateparser will gleefully turn "1 2 3" into
    # something date-shaped; we don't let it run on bare digits.
    cue_text = context_text if context_text is not None else raw
    if not has_date_cue(cue_text):
        # Allow purely numeric DD/MM/YYYY shapes through the gate when
        # the locale is unambiguously DMY — that case is the documented
        # numeric-date branch and dateparser will respect DATE_ORDER.
        # Otherwise refuse.
        if not (
            locale_date_order == "DMY"
            and _NUMERIC_DATE_RE.match(raw)
        ):
            return DateParseResult(
                canonical=None, fallback_reason="missing_date_cue",
            )

    # 3. Library availability.
    try:
        dateparser = _load_dateparser()
    except Exception:
        return DateParseResult(
            canonical=None, fallback_reason="dateparser_unavailable",
        )

    # 4. Strict parse.
    settings: dict[str, Any] = {
        "DATE_ORDER": locale_date_order,
        "STRICT_PARSING": True,
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DAY_OF_MONTH": "first",
    }
    parsed = dateparser.parse(
        raw, languages=list(languages), settings=settings,
    )
    if parsed is None:
        return DateParseResult(
            canonical=None, fallback_reason="dateparser_no_parse",
        )

    canonical = f"{parsed.day:02d}/{parsed.month:02d}/{parsed.year:04d}"
    return DateParseResult(canonical=canonical, fallback_reason=None)


__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "DateParseResult",
    "has_date_cue",
    "try_dateparser_fallback",
]
