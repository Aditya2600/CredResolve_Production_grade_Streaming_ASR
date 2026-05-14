"""Marathi date grammar (spoken / numeric -> ``DD/MM[/YYYY]``).

Mirror of ``grammars/hi/date.py`` with only the lexicon swapped:

* Marathi month names:
  ``जानेवारी`` (vs Hindi ``जनवरी``),
  ``फेब्रुवारी`` (vs ``फरवरी``),
  ``एप्रिल``    (vs ``अप्रैल``),
  ``मे``        (vs ``मई``),
  ``जुलै``      (vs ``जुलाई``),
  ``ऑगस्ट``     (vs ``अगस्त``),
  ``सप्टेंबर``  (vs ``सितंबर``/``सितम्बर``),
  ``ऑक्टोबर``   (vs ``अक्टूबर``),
  ``नोव्हेंबर`` (vs ``नवंबर``/``नवम्बर``),
  ``डिसेंबर``   (vs ``दिसंबर``/``दिसम्बर``).
  ``मार्च`` and ``जून`` are shared.
* Day / year parsing inherits the Marathi cardinal grammar's
  ``CARDINAL`` and the shared ``_NUM_0_99`` table.

The two surface families and their safety contract are unchanged:

1. ``DATE_MONTHWORD`` — always safe; the month name disambiguates.
2. ``DATE_NUMERIC`` — DMY-only; the runtime ``locale_policy`` decides
   whether to compose this for a given tenant.

Output is canonical ``DD/MM/YYYY`` (or ``DD/MM`` when year is absent).
The grammar never invents a year.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS
from itn_service.grammars.mr.cardinal import CARDINAL, _NUM_0_99, _SP


# ---------------------------------------------------------------------------
# Day: 1..31, spoken Marathi or Latin -> "DD".
# ---------------------------------------------------------------------------

def _build_day_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for n in range(1, 32):
        canon = str(n).rjust(2, "0")
        for w in _NUM_0_99[n]:
            pairs.add((w, canon))
        pairs.add((str(n), canon))
        pairs.add((canon, canon))
    return sorted(pairs)


_DAY: Final[pynini.Fst] = pynini.string_map(_build_day_pairs()).optimize()

_DAY_LATIN: Final[pynini.Fst] = pynini.string_map(
    sorted(
        {(s, c) for s, c in _build_day_pairs() if all(ch.isascii() for ch in s)}
    )
).optimize()


# ---------------------------------------------------------------------------
# Month names. Marathi spellings first, then English / Hinglish
# abbreviations (shared across all Indian-English locales).
# ---------------------------------------------------------------------------

_MONTH_NAMES: Final[dict[int, list[str]]] = {
    1: ["जानेवारी", "January", "JANUARY", "january", "Jan", "JAN", "jan"],
    2: [
        "फेब्रुवारी", "फेब्रवारी",
        "February", "FEBRUARY", "february", "Feb", "FEB", "feb",
    ],
    3: ["मार्च", "March", "MARCH", "march", "Mar", "MAR", "mar"],
    4: ["एप्रिल", "April", "APRIL", "april", "Apr", "APR", "apr"],
    5: ["मे", "May", "MAY", "may"],
    6: ["जून", "June", "JUNE", "june", "Jun", "JUN", "jun"],
    7: ["जुलै", "जुलाई", "July", "JULY", "july", "Jul", "JUL", "jul"],
    8: ["ऑगस्ट", "ऑगस्ट", "August", "AUGUST", "august", "Aug", "AUG", "aug"],
    9: [
        "सप्टेंबर", "सप्टेम्बर",
        "September", "SEPTEMBER", "september",
        "Sep", "SEP", "sep", "Sept", "SEPT", "sept",
    ],
    10: ["ऑक्टोबर", "ऑक्टोंबर", "October", "OCTOBER", "october", "Oct", "OCT", "oct"],
    11: [
        "नोव्हेंबर", "नोव्हेम्बर",
        "November", "NOVEMBER", "november", "Nov", "NOV", "nov",
    ],
    12: [
        "डिसेंबर", "डिसेम्बर",
        "December", "DECEMBER", "december", "Dec", "DEC", "dec",
    ],
}


def _build_month_name_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m, names in _MONTH_NAMES.items():
        canon = str(m).rjust(2, "0")
        # ``dict.fromkeys`` preserves insertion order and dedupes the
        # accidental repeated ``ऑगस्ट`` typo etc., without changing the
        # set of accepted surface forms.
        pairs.extend((n, canon) for n in dict.fromkeys(names))
    return pairs


_MONTH_BY_NAME: Final[pynini.Fst] = pynini.string_map(
    _build_month_name_pairs()
).optimize()


def _build_month_num_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for m in range(1, 13):
        canon = str(m).rjust(2, "0")
        pairs.add((str(m), canon))
        pairs.add((canon, canon))
    return sorted(pairs)


_MONTH_BY_NUM: Final[pynini.Fst] = pynini.string_map(
    _build_month_num_pairs()
).optimize()


# ---------------------------------------------------------------------------
# Year handling.
# ---------------------------------------------------------------------------

_FOUR_LATIN_DIGITS: Final[pynini.Fst] = (
    LATIN_DIGITS + LATIN_DIGITS + LATIN_DIGITS + LATIN_DIGITS
).optimize()

_YEAR_4: Final[pynini.Fst] = (CARDINAL @ _FOUR_LATIN_DIGITS).optimize()


def _build_year2_pairs() -> list[tuple[str, str]]:
    return [
        (str(yy).rjust(2, "0"), f"20{str(yy).rjust(2, '0')}")
        for yy in range(0, 100)
    ]


_YEAR_2_TO_4: Final[pynini.Fst] = pynini.string_map(_build_year2_pairs()).optimize()

_NUM_YEAR: Final[pynini.Fst] = pynini.union(
    _FOUR_LATIN_DIGITS, _YEAR_2_TO_4
).optimize()


# ---------------------------------------------------------------------------
# Month-word date (always safe).
# ---------------------------------------------------------------------------

_DM_NO_YEAR: Final[pynini.Fst] = (
    _DAY
    + pynutil.delete(_SP)
    + pynutil.insert("/")
    + _MONTH_BY_NAME
).optimize()

_DM_WITH_YEAR: Final[pynini.Fst] = (
    _DM_NO_YEAR
    + pynutil.delete(_SP)
    + pynutil.insert("/")
    + _YEAR_4
).optimize()

DATE_MONTHWORD: Final[pynini.Fst] = pynini.union(
    _DM_WITH_YEAR, _DM_NO_YEAR
).optimize()


# ---------------------------------------------------------------------------
# Numeric date (DMY-only — runtime gates this by tenant policy).
# ---------------------------------------------------------------------------

_NUM_SEP: Final[pynini.Fst] = pynini.union("/", "-", ".")
_REPLACE_SEP: Final[pynini.Fst] = pynutil.delete(_NUM_SEP) + pynutil.insert("/")

DATE_NUMERIC: Final[pynini.Fst] = (
    _DAY_LATIN
    + _REPLACE_SEP
    + _MONTH_BY_NUM
    + _REPLACE_SEP
    + _NUM_YEAR
).optimize()


# ---------------------------------------------------------------------------
# Public surfaces.
# ---------------------------------------------------------------------------

DATE: Final[pynini.Fst] = pynini.union(DATE_MONTHWORD, DATE_NUMERIC).optimize()

DATE_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('date { value: "') + DATE + pynutil.insert('" }')
).optimize()


__all__ = [
    "DATE",
    "DATE_CLASSIFIER",
    "DATE_MONTHWORD",
    "DATE_NUMERIC",
]
