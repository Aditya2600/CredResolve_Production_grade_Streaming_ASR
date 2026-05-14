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
    (b) Script majority -> language map, with a Marathi-vs-Hindi
        keyword score for Devanagari (Marathi and Hindi share the
        script, so a lexical tiebreak is required)
    (c) IndicLID fallback (long final spans / romanised spans only) —
        this stage is *flagged* but not *invoked* yet; the stub raises
        NotImplementedError so accidental hot-path use is loud.

Marathi / Hindi disambiguation (priority-step b):

Both languages use the Devanagari script. The default for Devanagari
is Hindi (the overwhelmingly dominant traffic share). The script
router upgrades the routing to Marathi when the working copy carries
clear Marathi-only lexical markers: time cues (``वाजता``, ``वाजून``,
``मिनिटे``), the Marathi hundred-compound shape (``पाचशे``,
``दोनशे``, ...), the Marathi half/quarter words (``दीड``, ``अडीच``,
``पावणे``), the Marathi percent cue ``टक्के`` / ``टक्का``, the
Marathi numerals that diverge sharply from Hindi (``दोन``, ``सहा``,
``नऊ``, ``दहा``, ``अकरा``, ``वीस``), and the Marathi month names
(``जानेवारी``, ``फेब्रुवारी``, ``जुलै``, ``ऑगस्ट``, ...).

A small additive keyword score is used — see :func:`_score_mr_vs_hi`.
No model is loaded; lookup is a hot-path-friendly substring scan
backed by an :class:`lru_cache`-bounded pre-compiled token list. This
mirrors the cheap-feature route discussed in the plan's "Script and
language router" row, which deliberately defers IndicLID to a much
later stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

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

# Script -> default language. For Devanagari the default is Hindi
# (the overwhelmingly dominant traffic share); Marathi is selected
# via the lexical keyword score in :func:`_score_mr_vs_hi`, not by
# script alone.
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


# ---------------------------------------------------------------------------
# Marathi vs Hindi keyword scoring (Devanagari only).
#
# Both languages share the Devanagari script, so the script histogram
# alone cannot pick between them. We add a small additive score over
# high-signal Marathi-only and Hindi-only tokens. The model is
# deliberately simple — a single weight per keyword (defaulting to
# 1.0) — and tuned so that:
#
#   * A bare Devanagari sentence with no Marathi cues stays Hindi
#     (the production default; this is the case the existing
#     ``test_script_majority_devanagari_defaults_to_hindi`` test pins).
#   * A single distinctive Marathi cue (``वाजता``, ``पाचशे``,
#     ``टक्के``, ``दीड``, ``अडीच``, ``जानेवारी``, ...) flips the
#     routing to Marathi.
#   * Mixed inputs (some Hindi cues, some Marathi cues) prefer
#     Marathi only when its score exceeds Hindi's by more than the
#     tie-break margin.
#
# The scan is a single regex pass over the input text. Tokens with
# Devanagari script characters do not bind to ASCII word boundaries
# (``\b``), so we match by substring; the keyword list is intentionally
# kept to surface forms that are not substrings of common unrelated
# words (e.g. we do not put ``एक`` — too short, appears everywhere — in
# the list).
# ---------------------------------------------------------------------------

# Marathi-only lexical cues. Strong signals: never appear in Hindi.
#
# Each entry's weight is 1.0 unless otherwise noted. Weights > 1.0 are
# used for high-confidence cues whose presence alone should flip the
# routing (these are the cues with zero false-positive rate against
# Hindi).
_MR_CUES: Final[tuple[tuple[str, float], ...]] = (
    # Time cues (highest signal — entirely absent from Hindi).
    ("वाजता", 2.0),
    ("वाजून", 2.0),
    ("मिनिटे", 1.5),
    ("मिनिटं", 1.5),
    # Time-of-day modifiers.
    ("सकाळी", 1.5),
    ("दुपारी", 1.5),
    ("संध्याकाळी", 2.0),
    ("रात्री", 1.5),
    # Half/quarter compounds. ``अडीच`` is especially distinctive
    # (Hindi's ``ढाई`` is a totally different stem).
    ("दीड", 2.0),
    ("अडीच", 2.0),
    ("साडे", 1.5),
    ("पावणे", 2.0),
    ("सव्वा", 1.0),  # also occurs in Hindi as ``सवा`` (different spelling)
    # Percent cue. ``टक्के`` is everyday Marathi; ``टक्का`` likewise.
    ("टक्के", 2.0),
    ("टक्का", 1.5),
    # Crore-equivalent. ``कोटी`` is the Marathi form (Hindi: ``करोड़``).
    ("कोटी", 1.5),
    # Cardinal numerals that diverge sharply from Hindi.
    ("दोन", 1.0),    # Hindi: दो
    ("सहा", 1.0),    # Hindi: छह / छः
    ("नऊ", 1.0),     # Hindi: नौ
    ("दहा", 1.0),    # Hindi: दस
    ("अकरा", 1.0),   # Hindi: ग्यारह
    ("बारा", 1.0),   # Hindi: बारह
    ("तेरा", 1.0),   # Hindi: तेरह
    ("चौदा", 1.0),   # Hindi: चौदह
    ("पंधरा", 1.0),  # Hindi: पंद्रह
    ("सोळा", 1.5),   # Hindi: सोलह (uses Marathi-specific ळ)
    ("सतरा", 1.0),   # Hindi: सत्रह
    ("अठरा", 1.0),   # Hindi: अट्ठारह / अठारह
    # Hundreds compound — the most distinctive structural marker.
    ("शंभर", 1.5),   # standalone 100
    ("पाचशे", 2.0),
    ("दोनशे", 2.0),
    ("तीनशे", 2.0),
    ("चारशे", 2.0),
    ("सहाशे", 2.0),
    ("सातशे", 2.0),
    ("आठशे", 2.0),
    ("नऊशे", 2.0),
    ("एकशे", 2.0),
    # Decimal marker.
    ("दशांश", 1.5),  # Hindi: दशमलव (also used in Marathi but rarer)
    # Month names.
    ("जानेवारी", 2.0),
    ("फेब्रुवारी", 2.0),
    ("एप्रिल", 1.5),
    ("जुलै", 2.0),
    ("ऑगस्ट", 2.0),
    ("सप्टेंबर", 2.0),
    ("ऑक्टोबर", 2.0),
    ("नोव्हेंबर", 2.0),
    ("डिसेंबर", 2.0),
)

# Hindi-only lexical cues. Used to suppress accidental Marathi flips
# when a Hindi-specific token clearly anchors the language.
_HI_CUES: Final[tuple[tuple[str, float], ...]] = (
    # Time cues.
    ("बजे", 2.0),
    ("बजकर", 2.0),
    # Half/quarter words (Hindi forms with nukta or distinct stems).
    ("डेढ़", 2.0),
    ("ढाई", 2.0),
    ("साढ़े", 1.5),
    ("पौने", 2.0),
    # Percent cue. ``फीसदी`` is Hindi/Urdu; absent from Marathi.
    ("फीसदी", 2.0),
    ("फ़ीसदी", 2.0),
    # Crore. ``करोड़`` is Hindi; Marathi uses ``कोटी``.
    ("करोड़", 1.5),
    # Cardinal forms that distinguish Hindi from Marathi.
    ("ग्यारह", 1.0),
    ("बारह", 1.0),
    ("तेरह", 1.0),
    ("चौदह", 1.0),
    ("पंद्रह", 1.0),
    ("सोलह", 1.0),
    ("सत्रह", 1.0),
    ("उन्नीस", 1.0),
    # Month names.
    ("जनवरी", 2.0),
    ("फरवरी", 2.0),
    ("फ़रवरी", 2.0),
    ("अप्रैल", 2.0),
    ("जुलाई", 2.0),
    ("अगस्त", 2.0),
    ("सितंबर", 2.0),
    ("सितम्बर", 2.0),
    ("अक्टूबर", 2.0),
    ("नवंबर", 2.0),
    ("नवम्बर", 2.0),
    ("दिसंबर", 2.0),
    ("दिसम्बर", 2.0),
)


# Minimum lead Marathi must hold over Hindi to win. A small positive
# margin keeps the default (Hindi) sticky when evidence is ambiguous —
# the production policy is that Devanagari without strong Marathi cues
# routes to Hindi.
_MR_WIN_MARGIN: Final[float] = 0.5


def _compile_keyword_regex(
    cues: tuple[tuple[str, float], ...],
) -> tuple[re.Pattern[str], dict[str, float]]:
    """Compile one alternation regex over the keyword list.

    Returns (pattern, weight_map). The alternation is sorted by
    length descending so that longer surface forms win over their
    substrings (e.g. ``पाचशे`` over ``पाच`` if both were present).
    """
    by_len = sorted({k for k, _ in cues}, key=len, reverse=True)
    weights = {k: w for k, w in cues}
    pattern = re.compile("|".join(re.escape(k) for k in by_len))
    return pattern, weights


_MR_PATTERN, _MR_WEIGHTS = _compile_keyword_regex(_MR_CUES)
_HI_PATTERN, _HI_WEIGHTS = _compile_keyword_regex(_HI_CUES)


def _score_mr_vs_hi(text: str) -> tuple[float, float]:
    """Return (marathi_score, hindi_score) over the keyword tables.

    Each match contributes its weight to the corresponding score.
    Overlapping matches are not specially handled: the regex engine
    walks the input left-to-right, consuming the longest alternation
    at each position, which gives the substring-precedence behaviour
    we want.
    """
    mr = sum(_MR_WEIGHTS[m.group(0)] for m in _MR_PATTERN.finditer(text))
    hi = sum(_HI_WEIGHTS[m.group(0)] for m in _HI_PATTERN.finditer(text))
    return mr, hi


def _disambiguate_devanagari(text: str) -> str:
    """Pick 'hi' or 'mr' for Devanagari text using the keyword score.

    Default is ``"hi"``; flip to ``"mr"`` only when the Marathi score
    exceeds the Hindi score by more than :data:`_MR_WIN_MARGIN`. This
    keeps bare Devanagari sentences with no cues on Hindi (the
    production traffic share) and only routes to Marathi when there is
    positive lexical evidence for it.
    """
    mr_score, hi_score = _score_mr_vs_hi(text)
    if mr_score - hi_score > _MR_WIN_MARGIN:
        return "mr"
    return "hi"


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
                        ``script_majority_mr_keywords`` (Marathi
                        upgrade fired on a Devanagari segment) |
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

    if script == "Devanagari":
        # Hindi vs Marathi disambiguation via lexical keyword score.
        # Default is Hindi; flip to Marathi when distinctive Marathi
        # cues outweigh Hindi cues by the configured margin.
        lang = _disambiguate_devanagari(text)
        source = (
            "script_majority_mr_keywords" if lang == "mr" else "script_majority"
        )
        return RouteResult(lang=lang, script=script, source=source)

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
