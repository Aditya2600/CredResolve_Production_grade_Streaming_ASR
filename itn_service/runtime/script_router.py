"""Script detection and language routing.

`detect_script` runs an ICU UScript histogram over the working-copy
text and returns the dominant script. Codepoints whose script is
`Common` (ASCII digits, punctuation, spaces) or `Inherited` (combining
marks, ZWJ / ZWNJ) are excluded from the histogram so that, e.g., a
Devanagari sentence with embedded punctuation is correctly identified
as Devanagari. If no real script characters are present at all (a
purely numeric / punctuation segment), the function returns `Common`.

`route_language` chooses the grammar pack and digit map. Priority
order, per the plan's "Script and language router" row:

    (a) ASR language hint, if present and trusted
    (b) Script majority -> language map
    (c) IndicLID fallback (long final spans / romanised spans only) —
        this stage is *flagged* but not *invoked* yet; the stub raises
        NotImplementedError so accidental hot-path use is loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from icu import Script

# The set of long-name scripts we report. Anything else returned by
# ICU collapses to its closest neighbour (e.g. ArabicCommon presentation
# forms still report as "Arabic" via getScript on the base codepoint).
_SCRIPTS_OF_INTEREST: frozenset[str] = frozenset(
    {
        "Devanagari",
        "Bengali",
        "Gurmukhi",
        "Gujarati",
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
        "Arabic",
        "Latin",
    }
)

# Stable tie-break order. When two scripts have identical histogram
# counts we prefer the more specific Indic script over Latin (Latin
# tokens commonly appear inside otherwise-Indic transcripts as
# loanwords or partial romanisation), and otherwise sort
# alphabetically. This keeps `detect_script` deterministic.
_TIE_BREAK_PRIORITY: dict[str, int] = {
    "Devanagari": 0,
    "Bengali": 0,
    "Gurmukhi": 0,
    "Gujarati": 0,
    "Tamil": 0,
    "Telugu": 0,
    "Kannada": 0,
    "Malayalam": 0,
    "Arabic": 0,
    "Latin": 1,
}

# ASR language hints we trust. Mirrors the locales declared in
# configs/locales.yaml plus 'en' for romanised English-Indic mixed
# input.
_TRUSTED_ASR_LANGS: frozenset[str] = frozenset(
    {"hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "ur", "en"}
)

# Script -> default language. For Devanagari we default to Hindi (the
# overwhelmingly dominant traffic share); Marathi must come in via
# `asr_hint`. This is the same disambiguation policy CLDR uses when no
# locale tag is supplied.
_SCRIPT_TO_LANG: dict[str, str] = {
    "Devanagari": "hi",
    "Bengali": "bn",
    "Gurmukhi": "pa",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Arabic": "ur",
    "Latin": "en",
}


@lru_cache(maxsize=4096)
def _script_name(cp: int) -> str:
    """Cached ICU getScript -> long name. Hot path on 200-char text."""
    return Script.getName(Script.getScript(cp))


def detect_script(text: str) -> str:
    """Return the dominant script of `text`.

    One of: Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu,
    Kannada, Malayalam, Arabic, Latin, Common.
    """
    if not text:
        return "Common"
    counts: dict[str, int] = {}
    for ch in text:
        name = _script_name(ord(ch))
        if name in _SCRIPTS_OF_INTEREST:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "Common"
    # Sort by (-count, tie_break_priority, name) for full determinism.
    return min(
        counts.items(),
        key=lambda kv: (-kv[1], _TIE_BREAK_PRIORITY[kv[0]], kv[0]),
    )[0]


@dataclass(frozen=True)
class RouteResult:
    """Outcome of `route_language`.

    `lang`              chosen BCP-47-ish 2-letter code, or ``"und"``
                        when nothing matches.
    `script`            the detected script (same vocabulary as
                        `detect_script`).
    `source`            which rule fired: ``asr_hint`` |
                        ``romanized_hint`` | ``script_majority`` |
                        ``unknown``.
    `needs_indiclid`    True when the priority chain fell through to
                        the IndicLID fallback. Callers that have
                        IndicLID wired in should branch on this; for
                        now it simply propagates so the pipeline knows
                        the routing was uncertain.
    """

    lang: str
    script: str
    source: str
    needs_indiclid: bool = False


def route_language(
    text: str,
    asr_hint: str | None = None,
    romanized_hint: str | None = None,
) -> RouteResult:
    """Pick a language for `text` using the priority cascade.

    Args:
      text: working-copy text (already cleaned by `working_copy`).
      asr_hint: BCP-47-ish 2-letter language code from the ASR gateway.
        Trusted when present and in the supported set.
      romanized_hint: optional override for Latin-script segments that
        upstream has already classified as romanised Indic (e.g. via a
        wakeword model or a CRM-side language tag). Only consulted when
        the script is Latin.
    """
    script = detect_script(text)

    if asr_hint and asr_hint in _TRUSTED_ASR_LANGS:
        return RouteResult(lang=asr_hint, script=script, source="asr_hint")

    if script == "Latin":
        if romanized_hint and romanized_hint in _TRUSTED_ASR_LANGS:
            return RouteResult(
                lang=romanized_hint,
                script=script,
                source="romanized_hint",
            )
        # Romanised Indic without an upstream hint is exactly the case
        # IndicLID is meant for. Flag it; do not call the model yet.
        return RouteResult(
            lang="en",
            script=script,
            source="script_majority",
            needs_indiclid=True,
        )

    if script in _SCRIPT_TO_LANG:
        return RouteResult(
            lang=_SCRIPT_TO_LANG[script],
            script=script,
            source="script_majority",
        )

    # Common / no script characters at all. Nothing to anchor on.
    return RouteResult(
        lang="und",
        script=script,
        source="unknown",
        needs_indiclid=True,
    )


def indiclid_predict(text: str) -> str:
    """IndicLID fallback. Stub.

    Will be wired to the IndicLID model in a later stage. Kept here so
    `route_language` has a single import path to switch over once the
    model is loaded into the FAR / model cache.
    """
    raise NotImplementedError(
        "IndicLID is not wired in yet; see runtime/script_router.py."
    )


__all__ = [
    "RouteResult",
    "detect_script",
    "indiclid_predict",
    "route_language",
]
