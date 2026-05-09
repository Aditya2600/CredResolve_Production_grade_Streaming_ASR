"""Hindi date grammar (spoken / numeric -> ``DD/MM[/YYYY]``).

Two surface families with deliberately different safety properties:

1. **Month-word dates** (``DATE_MONTHWORD``). The month is named, so
   day/month is unambiguous. Always safe to fire::

       बारह मई दो हज़ार छब्बीस      -> 12/05/2026
       बारह मई                       -> 12/05         (year omitted)
       12 May 2026                    -> 12/05/2026   (Hinglish, Latin day)
       बारह May                       -> 12/05         (mixed Devanagari + English)
       बारह May 2026                  -> 12/05/2026
       2 जुलाई                        -> 02/07
       31 दिसम्बर 1999                -> 31/12/1999

2. **Purely numeric dates** (``DATE_NUMERIC``). ``DD-MM-YY``, ``DD/MM/YYYY``,
   ``DD.MM.YYYY``. Day/month order is **ambiguous** without locale context.
   This FST encodes the day-first reading and is therefore only safe to
   fire under a tenant whose ``locale_policy.date_order == "DMY"``::

       12/05/2026   -> 12/05/2026
       12-05-26     -> 12/05/2026   (2-digit year expanded to 20YY)
       5/6/2024     -> 05/06/2024   (single-digit padded)
       05.06.2024   -> 05/06/2024

   The runtime gate that decides whether to compose ``DATE_NUMERIC`` for
   a given tenant lives in :mod:`itn_service.runtime.locale_policy` and
   :mod:`itn_service.runtime.normalizer`. When the gate refuses, the
   span is emitted with ``fallback_reason="ambiguous_numeric_date"``.

3. ``DATE = DATE_MONTHWORD | DATE_NUMERIC`` — the build-time union. The
   pipeline uses this for DMY tenants; for other tenants it falls back
   to ``DATE_MONTHWORD`` only.

Output is always canonical ``DD/MM/YYYY`` (or ``DD/MM`` when year is
absent in the spoken/written form). ISO ``YYYY-MM-DD`` rendering and
locale-specific date display are deliberately handled in the display
layer, not here — see ``runtime/display_renderer.py``.

Constraint per the implementation blueprint: this grammar **never**
invents a year. When the speaker omits the year, the canonical output
omits it too and downstream decides what to do.

Per ``CONTRIBUTING.md`` invariants the FST is built at module import
(= grammar build time) and the canonical storage form uses Latin
digits + ICU-canonical separators.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS
from itn_service.grammars.hi.cardinal import CARDINAL, _NUM_0_99, _SP


# ---------------------------------------------------------------------------
# Day: 1..31, spoken Hindi or Latin (with / without leading zero) -> "DD".
# ---------------------------------------------------------------------------

def _build_day_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for n in range(1, 32):
        canon = str(n).rjust(2, "0")
        # Spoken Hindi (all spelling alternates from cardinal lexicon).
        for w in _NUM_0_99[n]:
            pairs.add((w, canon))
        # Latin: "1" / "01" / "12" both -> "01" / "12".
        pairs.add((str(n), canon))
        pairs.add((canon, canon))
    return sorted(pairs)


_DAY: Final[pynini.Fst] = pynini.string_map(_build_day_pairs()).optimize()

# Latin-only day acceptor for the numeric-date branch (no spoken forms).
_DAY_LATIN: Final[pynini.Fst] = pynini.string_map(
    sorted(
        {(s, c) for s, c in _build_day_pairs() if all(ch.isascii() for ch in s)}
    )
).optimize()


# ---------------------------------------------------------------------------
# Month names (Hindi + English + Hinglish abbreviations) -> "MM".
# ---------------------------------------------------------------------------

# Spelling alternates we accept on input. The canonical output for each
# month is the 2-digit Latin form ("01".."12") so the canonical storage
# stays locale-neutral.
_MONTH_NAMES: Final[dict[int, list[str]]] = {
    1: ["जनवरी", "January", "JANUARY", "january", "Jan", "JAN", "jan"],
    2: [
        "फरवरी", "फ़रवरी",
        "February", "FEBRUARY", "february", "Feb", "FEB", "feb",
    ],
    3: ["मार्च", "March", "MARCH", "march", "Mar", "MAR", "mar"],
    4: ["अप्रैल", "April", "APRIL", "april", "Apr", "APR", "apr"],
    5: ["मई", "May", "MAY", "may"],
    6: ["जून", "June", "JUNE", "june", "Jun", "JUN", "jun"],
    7: ["जुलाई", "July", "JULY", "july", "Jul", "JUL", "jul"],
    8: ["अगस्त", "August", "AUGUST", "august", "Aug", "AUG", "aug"],
    9: [
        "सितंबर", "सितम्बर",
        "September", "SEPTEMBER", "september",
        "Sep", "SEP", "sep", "Sept", "SEPT", "sept",
    ],
    10: ["अक्टूबर", "October", "OCTOBER", "october", "Oct", "OCT", "oct"],
    11: [
        "नवंबर", "नवम्बर",
        "November", "NOVEMBER", "november", "Nov", "NOV", "nov",
    ],
    12: [
        "दिसंबर", "दिसम्बर",
        "December", "DECEMBER", "december", "Dec", "DEC", "dec",
    ],
}


def _build_month_name_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m, names in _MONTH_NAMES.items():
        canon = str(m).rjust(2, "0")
        pairs.extend((n, canon) for n in names)
    return pairs


_MONTH_BY_NAME: Final[pynini.Fst] = pynini.string_map(
    _build_month_name_pairs()
).optimize()


# Numeric-date month: 1..12 with optional leading zero -> "MM".
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

# 4-digit Latin acceptor (used both as a constraint over CARDINAL output
# and as direct passthrough for already-Latin years).
_FOUR_LATIN_DIGITS: Final[pynini.Fst] = (
    LATIN_DIGITS + LATIN_DIGITS + LATIN_DIGITS + LATIN_DIGITS
).optimize()

# Spoken / Latin year -> 4-digit Latin string. The cardinal grammar
# already accepts forms like "दो हज़ार छब्बीस" (= 2026), "उन्नीस सौ नब्बे"
# (= 1990), and Latin "2026" / "1990" passthrough. Composition with the
# 4-digit acceptor filters to outputs of exactly 4 digits — i.e. years.
_YEAR_4: Final[pynini.Fst] = (CARDINAL @ _FOUR_LATIN_DIGITS).optimize()

# 2-digit Latin year -> 20YY. Used only on the numeric-date branch (a
# spoken speaker who omits the year is handled by emitting DD/MM with
# no year part — never invent it). Capping at the current century is
# a deliberate locale-policy choice; CLDR allows 2-digit-year pivot
# rules, but we pick "20YY" deterministically and document it.
def _build_year2_pairs() -> list[tuple[str, str]]:
    return [
        (str(yy).rjust(2, "0"), f"20{str(yy).rjust(2, '0')}")
        for yy in range(0, 100)
    ]


_YEAR_2_TO_4: Final[pynini.Fst] = pynini.string_map(_build_year2_pairs()).optimize()

# Numeric-date year accepts either a 4-digit Latin year (passthrough) or
# a 2-digit Latin year (expanded to 20YY).
_NUM_YEAR: Final[pynini.Fst] = pynini.union(
    _FOUR_LATIN_DIGITS, _YEAR_2_TO_4
).optimize()


# ---------------------------------------------------------------------------
# Month-word date (always safe).
#
#   <day> <space> <month-name> [<space> <year>]
#
# Year is optional. When omitted, output is "DD/MM"; never inferred.
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
# Numeric date (DMY-only — the runtime gates this by tenant policy).
#
#   <day-latin> <sep> <month-num> <sep> <year-latin>
#
# Output separator is always "/" regardless of input separator.
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
#
#   ``DATE_MONTHWORD``    : always-safe (month named, unambiguous).
#   ``DATE_NUMERIC``      : DMY-only (caller must gate by tenant policy).
#   ``DATE``              : union; pipeline uses this for DMY tenants and
#                           ``DATE_MONTHWORD`` only for non-DMY tenants.
#   ``DATE_CLASSIFIER``   : NeMo-tagged wrapper around the union.
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
