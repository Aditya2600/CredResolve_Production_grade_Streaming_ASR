"""Marathi time grammar (cue-required, spoken / Latin -> ``H:MM[ AM|PM]`` or ``HH:MM``).

Mirror of ``grammars/hi/time.py`` with only the cue-word lexicon
swapped. The Marathi time cues are the **most distinctive** lexical
difference from Hindi — they appear nowhere in Hindi:

    वाजता    (o'clock)                     [Hindi: बजे]
    वाजून    (literal "having struck")     [Hindi: बजकर]
    मिनिटे   (minutes)                     [Hindi: मिनट]
    सकाळी    (morning, AM)                 [Hindi: सुबह]
    दुपारी   (noon / afternoon, PM)        [Hindi: दोपहर]
    संध्याकाळी (evening, PM)               [Hindi: शाम]
    रात्री    (night, PM)                  [Hindi: रात]
    AM / PM (and case / dot variants)      [shared]

Half / quarter compound stems come from the Marathi cardinal lexicon
(``cardinal.py``), so ``साडे पाच वाजता`` -> ``5:30``, ``सव्वा सहा
वाजता`` -> ``6:15``, ``पावणे सात वाजता`` -> ``6:45``, ``दीड वाजता``
-> ``1:30``, ``अडीच वाजता`` -> ``2:30``.

Bare numeric spans **never** normalise to time. The 2-digit-hour form
``14:30`` and the explicit AM/PM token are the only zero-cue surfaces
(both are unambiguous on their own).
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS
from itn_service.grammars.mr.cardinal import _NUM_0_99, _SP


# ---------------------------------------------------------------------------
# Hours 1..12 spoken / Latin -> bare hour string.
# ---------------------------------------------------------------------------

def _build_hour_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for n in range(1, 13):
        for w in _NUM_0_99[n]:
            pairs.add((w, str(n)))
        pairs.add((str(n), str(n)))
        pairs.add((str(n).rjust(2, "0"), str(n)))
    return sorted(pairs)


_HOUR_12: Final[pynini.Fst] = pynini.string_map(_build_hour_pairs()).optimize()


# ---------------------------------------------------------------------------
# Minutes 0..59 spoken / Latin -> 2-digit zero-padded.
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
# Half / quarter compounds — Marathi forms.
# ---------------------------------------------------------------------------

_SAADE: Final[pynini.Fst] = pynini.union("साडे", "साढे")
_SAVVA: Final[pynini.Fst] = pynini.union("सव्वा", "सवा")
_PAAVNE: Final[pynini.Fst] = pynini.union("पावणे", "पाउणे", "पाउने")
_DEED: Final[pynini.Fst] = pynini.union("दीड", "डीड")
_ADEECH: Final[pynini.Fst] = pynini.union("अडीच", "अडिच")


def _build_compound_hour_pairs(
    offset_minutes: int, hour_shift: int = 0,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for n in range(1, 13):
        target = n + hour_shift
        if target < 1:
            target = 12  # पावणे एक wraps to 12:45
        if target > 12:
            target = target - 12
        for w in _NUM_0_99[n]:
            pairs.append((w, f"{target}:{offset_minutes:02d}"))
    return pairs


# साडे + <hour>: 1:30 = दीड, 2:30 = अडीच have their own bare words,
# so साडे एक / साडे दोन are non-idiomatic. The grammar supports
# साडे N for N in 3..12.
_SAADE_HOUR_PAIRS: Final[list[tuple[str, str]]] = [
    (w, f"{n}:30")
    for n in range(3, 13)
    for w in _NUM_0_99[n]
]
_SAADE_HOUR_BODY: Final[pynini.Fst] = pynini.string_map(
    _SAADE_HOUR_PAIRS
).optimize()
_HOUR_EXPR_HALF: Final[pynini.Fst] = (
    pynutil.delete(_SAADE) + pynutil.delete(_SP) + _SAADE_HOUR_BODY
).optimize()

_HOUR_EXPR_QUARTER_PAST: Final[pynini.Fst] = (
    pynutil.delete(_SAVVA)
    + pynutil.delete(_SP)
    + pynini.string_map(_build_compound_hour_pairs(15)).optimize()
).optimize()

_HOUR_EXPR_QUARTER_TO: Final[pynini.Fst] = (
    pynutil.delete(_PAAVNE)
    + pynutil.delete(_SP)
    + pynini.string_map(
        _build_compound_hour_pairs(45, hour_shift=-1)
    ).optimize()
).optimize()

_HOUR_EXPR_DEED: Final[pynini.Fst] = (
    pynutil.delete(_DEED) + pynutil.insert("1:30")
).optimize()

_HOUR_EXPR_ADEECH: Final[pynini.Fst] = (
    pynutil.delete(_ADEECH) + pynutil.insert("2:30")
).optimize()

_HOUR_EXPR_PLAIN: Final[pynini.Fst] = (
    _HOUR_12 + pynutil.insert(":00")
).optimize()


# ---------------------------------------------------------------------------
# Cue-bearing "core" forms.
#
#   Form A: <hour-plain> + " वाजता"
#   Form B: <fractional-hour> + " वाजता"
#   Form C: दीड/अडीच + " वाजता"
#   Form D: <hour-plain-stem> + " वाजून " + <minute> + " मिनिटे"
# ---------------------------------------------------------------------------

_VAJTA: Final[pynini.Fst] = pynini.accep("वाजता")
_VAJUN: Final[pynini.Fst] = pynini.accep("वाजून")
_MINUTE_CUE: Final[pynini.Fst] = pynini.union("मिनिटे", "मिनिट", "मिनिटं")

_FORM_A: Final[pynini.Fst] = (
    _HOUR_EXPR_PLAIN
    + pynutil.delete(_SP)
    + pynutil.delete(_VAJTA)
).optimize()

_FORM_B: Final[pynini.Fst] = (
    pynini.union(
        _HOUR_EXPR_HALF,
        _HOUR_EXPR_QUARTER_PAST,
        _HOUR_EXPR_QUARTER_TO,
    )
    + pynutil.delete(_SP)
    + pynutil.delete(_VAJTA)
).optimize()

_FORM_C: Final[pynini.Fst] = (
    pynini.union(_HOUR_EXPR_DEED, _HOUR_EXPR_ADEECH)
    + pynutil.delete(_SP)
    + pynutil.delete(_VAJTA)
).optimize()

_FORM_D: Final[pynini.Fst] = (
    _HOUR_12
    + pynutil.insert(":")
    + pynutil.delete(_SP)
    + pynutil.delete(_VAJUN)
    + pynutil.delete(_SP)
    + _MINUTE
    + pynutil.delete(_SP)
    + pynutil.delete(_MINUTE_CUE)
).optimize()


_TIME_12H_CORE: Final[pynini.Fst] = pynini.union(
    _FORM_A, _FORM_B, _FORM_C, _FORM_D,
).optimize()


# ---------------------------------------------------------------------------
# Time-of-day modifiers — Marathi forms.
# ---------------------------------------------------------------------------

_AM_WORDS: Final[pynini.Fst] = pynini.union(
    "सकाळी",
    "AM", "am", "A.M.", "a.m.",
)
_PM_WORDS: Final[pynini.Fst] = pynini.union(
    "दुपारी", "संध्याकाळी", "रात्री",
    "PM", "pm", "P.M.", "p.m.",
)


def _wrap_with_tod(
    core: pynini.Fst, tod: pynini.Fst, suffix: str,
) -> pynini.Fst:
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
# Latin 12-hour form. Identical to Hindi.
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
# 24-hour numeric. Identical to Hindi.
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
