"""Hindi time grammar (cue-required, spoken / Latin -> ``H:MM[ AM|PM]`` or ``HH:MM``).

A cue is **required**. At least one of these must be present:

    बजे     (o'clock)
    बजकर   (literal "having struck", introduces minutes)
    मिनट    (minutes — used with बजकर)
    सुबह    (morning, AM)
    दोपहर   (noon / afternoon, PM)
    शाम    (evening, PM)
    रात    (night, PM)
    AM / PM (and case / dot variants)

Bare numeric spans **never** normalise to time — see the
"Confidence gating and fallback" / Time row of the implementation
blueprint and ``configs/thresholds.yaml`` (``time.require_lex_cue: true``).

Surface forms supported (12-hour Hindi conversational):

    पाँच बजे                          -> 5:00
    साढ़े पाँच बजे                    -> 5:30
    साढ़े पाँच बजे शाम                -> 5:30 PM
    शाम साढ़े पाँच बजे                -> 5:30 PM
    सवा छह बजे                        -> 6:15
    पौने सात बजे                      -> 6:45
    डेढ़ बजे                          -> 1:30
    ढाई बजे                           -> 2:30
    पाँच बजकर तीस मिनट                -> 5:30
    पाँच बजकर तीस मिनट शाम           -> 5:30 PM
    सुबह सात बजे                      -> 7:00 AM
    रात ग्यारह बजे                    -> 11:00 PM

Mixed Hinglish 12-hour (cue = AM/PM):

    5:30 AM                            -> 5:30 AM
    5:30 PM                            -> 5:30 PM
    12:00 AM                           -> 12:00 AM
    08:45 PM                           -> 8:45 PM    (leading-zero hour stripped)

24-hour numeric (the canonical 2-digit-hour form is the cue):

    14:30                              -> 14:30
    05:00                              -> 05:00
    23:59                              -> 23:59

Output:
    * 12-hour forms emit unpadded hour ("5:30", not "05:30").
    * 24-hour forms preserve the 2-digit hour ("14:30").
    * AM / PM is appended with a leading space when a time-of-day
      cue is present.

Per ``CONTRIBUTING.md`` invariants the FST is built at module import
and the canonical storage form uses ASCII colons + Latin digits.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS
from itn_service.grammars.hi.cardinal import _NUM_0_99, _SP


# ---------------------------------------------------------------------------
# Hours 1..12 spoken / Latin -> bare hour string ("1".."12"). 12-hour
# canonical output uses unpadded hour ("5:30", not "05:30") so callers
# don't need to know whether the speaker said "पाँच" or "05".
# ---------------------------------------------------------------------------

def _build_hour_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for n in range(1, 13):
        for w in _NUM_0_99[n]:
            pairs.add((w, str(n)))
        pairs.add((str(n), str(n)))
        pairs.add((str(n).rjust(2, "0"), str(n)))  # "05" -> "5"
    return sorted(pairs)


_HOUR_12: Final[pynini.Fst] = pynini.string_map(_build_hour_pairs()).optimize()


# ---------------------------------------------------------------------------
# Minutes 0..59 spoken / Latin -> 2-digit zero-padded ("MM").
# ---------------------------------------------------------------------------

def _build_minute_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for n in range(0, 60):
        canon = str(n).rjust(2, "0")
        for w in _NUM_0_99[n]:
            pairs.add((w, canon))
        pairs.add((str(n), canon))
        pairs.add((canon, canon))
    return sorted(pairs)


_MINUTE: Final[pynini.Fst] = pynini.string_map(_build_minute_pairs()).optimize()


# ---------------------------------------------------------------------------
# Hour expressions.
#
#   _HOUR_EXPR_PLAIN     : "<H>"            -> "H:00"  (combined with बजे)
#   _HOUR_EXPR_HALF      : "साढ़े <H>"      -> "H:30"  (3..12 only — डेढ़/ढाई)
#   _HOUR_EXPR_QUARTER_PAST : "सवा <H>"     -> "H:15"
#   _HOUR_EXPR_QUARTER_TO   : "पौने <H>"    -> "(H-1):45" (पौने एक = 12:45)
#   _HOUR_EXPR_DEDH      : "डेढ़"            -> "1:30"
#   _HOUR_EXPR_DHAII     : "ढाई"             -> "2:30"
# ---------------------------------------------------------------------------

_SAADHE: Final[pynini.Fst] = pynini.union("साढ़े", "साढे")
_SAVAA: Final[pynini.Fst] = pynini.accep("सवा")
_PAUNE: Final[pynini.Fst] = pynini.accep("पौने")
_DEDH: Final[pynini.Fst] = pynini.union("डेढ़", "डेढ")
_DHAII: Final[pynini.Fst] = pynini.accep("ढाई")


def _build_compound_hour_pairs(
    offset_minutes: int, hour_shift: int = 0,
) -> list[tuple[str, str]]:
    """For ``"<spelling>" -> "<n + hour_shift>:<offset_minutes:02d>"``.

    ``hour_shift`` lets पौने N output (N-1):45 (with the special case
    पौने एक = 12:45 wrapping around)."""
    pairs: list[tuple[str, str]] = []
    for n in range(1, 13):
        target = n + hour_shift
        if target < 1:
            target = 12  # पौने एक wraps to 12:45
        if target > 12:
            target = target - 12
        for w in _NUM_0_99[n]:
            pairs.append((w, f"{target}:{offset_minutes:02d}"))
    return pairs


# साढ़े + <hour>: the literal compound 1:30 = डेढ़, 2:30 = ढाई have
# their own bare words, so साढ़े एक / साढ़े दो are non-idiomatic. The
# grammar supports साढ़े N for N in 3..12.
_SAADHE_HOUR_PAIRS: Final[list[tuple[str, str]]] = [
    (w, f"{n}:30")
    for n in range(3, 13)
    for w in _NUM_0_99[n]
]
_SAADHE_HOUR_BODY: Final[pynini.Fst] = pynini.string_map(
    _SAADHE_HOUR_PAIRS
).optimize()
_HOUR_EXPR_HALF: Final[pynini.Fst] = (
    pynutil.delete(_SAADHE) + pynutil.delete(_SP) + _SAADHE_HOUR_BODY
).optimize()

_HOUR_EXPR_QUARTER_PAST: Final[pynini.Fst] = (
    pynutil.delete(_SAVAA)
    + pynutil.delete(_SP)
    + pynini.string_map(_build_compound_hour_pairs(15)).optimize()
).optimize()

_HOUR_EXPR_QUARTER_TO: Final[pynini.Fst] = (
    pynutil.delete(_PAUNE)
    + pynutil.delete(_SP)
    + pynini.string_map(
        _build_compound_hour_pairs(45, hour_shift=-1)
    ).optimize()
).optimize()

_HOUR_EXPR_DEDH: Final[pynini.Fst] = (
    pynutil.delete(_DEDH) + pynutil.insert("1:30")
).optimize()

_HOUR_EXPR_DHAII: Final[pynini.Fst] = (
    pynutil.delete(_DHAII) + pynutil.insert("2:30")
).optimize()

# Plain hour (used with बजे): "<H>" -> "H:00"
_HOUR_EXPR_PLAIN: Final[pynini.Fst] = (
    _HOUR_12 + pynutil.insert(":00")
).optimize()


# ---------------------------------------------------------------------------
# Cue-bearing "core" forms (no AM/PM yet).
#
#   Form A: <hour-plain> + " बजे"
#   Form B: <fractional-hour> + " बजे"
#   Form C: डेढ़/ढाई + " बजे"
#   Form D: <hour-plain-stem> + " बजकर " + <minute> + " मिनट"
# ---------------------------------------------------------------------------

_BAJE: Final[pynini.Fst] = pynini.accep("बजे")
_BAJKAR: Final[pynini.Fst] = pynini.accep("बजकर")
_MINUTE_CUE: Final[pynini.Fst] = pynini.accep("मिनट")

_FORM_A: Final[pynini.Fst] = (
    _HOUR_EXPR_PLAIN
    + pynutil.delete(_SP)
    + pynutil.delete(_BAJE)
).optimize()

_FORM_B: Final[pynini.Fst] = (
    pynini.union(
        _HOUR_EXPR_HALF,
        _HOUR_EXPR_QUARTER_PAST,
        _HOUR_EXPR_QUARTER_TO,
    )
    + pynutil.delete(_SP)
    + pynutil.delete(_BAJE)
).optimize()

_FORM_C: Final[pynini.Fst] = (
    pynini.union(_HOUR_EXPR_DEDH, _HOUR_EXPR_DHAII)
    + pynutil.delete(_SP)
    + pynutil.delete(_BAJE)
).optimize()

_FORM_D: Final[pynini.Fst] = (
    _HOUR_12
    + pynutil.insert(":")
    + pynutil.delete(_SP)
    + pynutil.delete(_BAJKAR)
    + pynutil.delete(_SP)
    + _MINUTE
    + pynutil.delete(_SP)
    + pynutil.delete(_MINUTE_CUE)
).optimize()


_TIME_12H_CORE: Final[pynini.Fst] = pynini.union(
    _FORM_A, _FORM_B, _FORM_C, _FORM_D,
).optimize()


# ---------------------------------------------------------------------------
# Time-of-day modifiers.
#
# सुबह, दोपहर, शाम, रात, plus AM / PM literal cues. Modifier may appear
# before or after the time core. When present, AM or PM is appended.
# When absent, the bare core stands (the बजे / बजकर cue is sufficient).
# ---------------------------------------------------------------------------

_AM_WORDS: Final[pynini.Fst] = pynini.union(
    "सुबह",
    "AM", "am", "A.M.", "a.m.",
)
_PM_WORDS: Final[pynini.Fst] = pynini.union(
    "दोपहर", "शाम", "रात",
    "PM", "pm", "P.M.", "p.m.",
)


def _wrap_with_tod(
    core: pynini.Fst, tod: pynini.Fst, suffix: str,
) -> pynini.Fst:
    """Build "core + tod" and "tod + core", both -> "core <suffix>"."""
    after = (
        core
        + pynutil.delete(_SP)
        + pynutil.delete(tod)
        + pynutil.insert(suffix)
    )
    before = (
        pynutil.delete(tod)
        + pynutil.delete(_SP)
        + core
        + pynutil.insert(suffix)
    )
    return pynini.union(after, before).optimize()


_TIME_12H_AM: Final[pynini.Fst] = _wrap_with_tod(
    _TIME_12H_CORE, _AM_WORDS, " AM"
)
_TIME_12H_PM: Final[pynini.Fst] = _wrap_with_tod(
    _TIME_12H_CORE, _PM_WORDS, " PM"
)


# ---------------------------------------------------------------------------
# Latin 12-hour form: "H[:|.]MM <AM|PM>" -> "H:MM AM|PM".
# Cue is the explicit AM/PM word.
# ---------------------------------------------------------------------------

_HH_LATIN_12_PAIRS: list[tuple[str, str]] = []
for h in range(1, 13):
    _HH_LATIN_12_PAIRS.append((str(h), str(h)))
    _HH_LATIN_12_PAIRS.append((str(h).rjust(2, "0"), str(h)))
_HH_LATIN_12: Final[pynini.Fst] = pynini.string_map(
    sorted(set(_HH_LATIN_12_PAIRS))
).optimize()

_MM_LATIN_PAIRS: Final[list[tuple[str, str]]] = [
    (str(m).rjust(2, "0"), str(m).rjust(2, "0"))
    for m in range(0, 60)
]
_MM_LATIN: Final[pynini.Fst] = pynini.string_map(_MM_LATIN_PAIRS).optimize()

_LATIN_12H_CORE: Final[pynini.Fst] = (
    _HH_LATIN_12 + pynini.cross(":", ":") + _MM_LATIN
).optimize()

_LATIN_12H_AM: Final[pynini.Fst] = (
    _LATIN_12H_CORE
    + pynutil.delete(_SP)
    + pynutil.delete(_AM_WORDS)
    + pynutil.insert(" AM")
).optimize()
_LATIN_12H_PM: Final[pynini.Fst] = (
    _LATIN_12H_CORE
    + pynutil.delete(_SP)
    + pynutil.delete(_PM_WORDS)
    + pynutil.insert(" PM")
).optimize()


# ---------------------------------------------------------------------------
# 24-hour numeric: "HH:MM" with HH in 00..23, MM in 00..59. Preserved
# as-is — the 2-digit-hour shape is the cue. Single-digit hours like
# "5:30" are deliberately rejected here; without a cue word they are
# ambiguous between 5 AM and 5 PM and must be deferred.
# ---------------------------------------------------------------------------

_HH_LATIN_24_PAIRS: Final[list[tuple[str, str]]] = [
    (str(h).rjust(2, "0"), str(h).rjust(2, "0"))
    for h in range(0, 24)
]
_HH_LATIN_24: Final[pynini.Fst] = pynini.string_map(_HH_LATIN_24_PAIRS).optimize()

_TIME_24H: Final[pynini.Fst] = (
    _HH_LATIN_24 + pynini.cross(":", ":") + _MM_LATIN
).optimize()


# ---------------------------------------------------------------------------
# Public surfaces.
# ---------------------------------------------------------------------------

TIME: Final[pynini.Fst] = pynini.union(
    _TIME_12H_CORE,
    _TIME_12H_AM,
    _TIME_12H_PM,
    _LATIN_12H_AM,
    _LATIN_12H_PM,
    _TIME_24H,
).optimize()

TIME_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('time { value: "') + TIME + pynutil.insert('" }')
).optimize()


__all__ = ["TIME", "TIME_CLASSIFIER"]
