"""Hindi cardinal grammar (spoken -> Latin integer).

Covers spoken Hindi 0..1 अरब (10⁹ inclusive) plus a small set of half /
quarter compounds (सवा, डेढ़, ढाई, साढ़े, पौने) when they combine with
a scale word (सौ / हज़ार / लाख / करोड़) so that the result is an
integer. Bare compounds (no scale, e.g. ``सवा बारह`` = 12.25) produce
a fractional value and are therefore handled in ``decimal.py``; this
file deliberately refuses them.

Output is wrapped in NeMo-style ``cardinal { value: "<latin_int>" }``
via :data:`CARDINAL_CLASSIFIER`. The undecorated normaliser
:data:`CARDINAL` is also exposed for compose-style use (e.g. inside
``money``, ``percent``, etc.).

Per `CONTRIBUTING.md` invariants:
    * The FST is built at module import (= grammar build time).
    * Output is plain Latin digits with no leading zeros (except for
      the literal ``"0"``); locale-specific Indian thousands grouping
      is applied later by :func:`indian_grouping` in
      ``runtime`` / ``display_renderer``.

References:
    * Plan: ``docs/implementation_bluprint_INR.md`` § "Implementation
      blueprint", Pynini sketch and "Indic-specific handling".
    * NeMo TN/ITN classify -> verbalise pattern.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS

# ---------------------------------------------------------------------------
# Spoken 0..99 lexicon, with the alternates the plan calls out explicitly
# (छह / छः, पाँच / पांच, हज़ार / हजार) plus a few of the most common
# orthographic variants seen in real ASR output (अट्ठारह / अठारह,
# डेढ़ / डेढ, साढ़े / साढे). The single-digit and scale lists also
# accept a narrow Hindi-scoped set of Devanagari-English code-switch
# aliases. Latin-script English remains out of scope.
# ---------------------------------------------------------------------------

_NUM_0_99: Final[dict[int, list[str]]] = {
    0: ["शून्य"],
    1: ["एक", "वन", "वान"],
    2: ["दो", "टू", "टु"],
    3: ["तीन", "थ्री"],
    4: ["चार", "फोर", "फॉर"],
    5: ["पाँच", "पांच", "फाइव", "फाईव"],
    6: ["छह", "छः", "सिक्स"],
    7: ["सात", "सेवन"],
    8: ["आठ", "एट", "ऐट"],
    9: ["नौ", "नाइन"],
    10: ["दस", "टेन"],
    11: ["ग्यारह"],
    12: ["बारह"],
    13: ["तेरह"],
    14: ["चौदह"],
    15: ["पंद्रह", "पन्द्रह"],
    16: ["सोलह"],
    17: ["सत्रह"],
    18: ["अट्ठारह", "अठारह"],
    19: ["उन्नीस"],
    20: ["बीस"],
    21: ["इक्कीस"],
    22: ["बाईस"],
    23: ["तेईस"],
    24: ["चौबीस"],
    25: ["पच्चीस"],
    26: ["छब्बीस"],
    27: ["सत्ताईस"],
    28: ["अट्ठाईस"],
    29: ["उनतीस"],
    30: ["तीस"],
    31: ["इकतीस"],
    32: ["बत्तीस"],
    33: ["तैंतीस"],
    34: ["चौंतीस"],
    35: ["पैंतीस"],
    36: ["छत्तीस"],
    37: ["सैंतीस"],
    38: ["अड़तीस"],
    39: ["उनतालीस"],
    40: ["चालीस"],
    41: ["इकतालीस"],
    42: ["बयालीस"],
    43: ["तैंतालीस"],
    44: ["चौंतालीस"],
    45: ["पैंतालीस"],
    46: ["छियालीस"],
    47: ["सैंतालीस"],
    48: ["अड़तालीस"],
    49: ["उनचास"],
    50: ["पचास"],
    51: ["इक्यावन"],
    52: ["बावन"],
    53: ["तिरेपन"],
    54: ["चौवन"],
    55: ["पचपन"],
    56: ["छप्पन"],
    57: ["सत्तावन"],
    58: ["अट्ठावन"],
    59: ["उनसठ"],
    60: ["साठ"],
    61: ["इकसठ"],
    62: ["बासठ"],
    63: ["तिरेसठ"],
    64: ["चौंसठ"],
    65: ["पैंसठ"],
    66: ["छयासठ"],
    67: ["सरसठ"],
    68: ["अड़सठ"],
    69: ["उनहत्तर"],
    70: ["सत्तर"],
    71: ["इकहत्तर"],
    72: ["बहत्तर"],
    73: ["तिहत्तर"],
    74: ["चौहत्तर"],
    75: ["पचहत्तर"],
    76: ["छिहत्तर"],
    77: ["सतहत्तर"],
    78: ["अठहत्तर"],
    79: ["उन्यासी"],
    80: ["अस्सी"],
    81: ["इक्यासी"],
    82: ["बयासी"],
    83: ["तिरासी"],
    84: ["चौरासी"],
    85: ["पचासी"],
    86: ["छियासी"],
    87: ["सत्तासी"],
    88: ["अठासी"],
    89: ["नवासी"],
    90: ["नब्बे"],
    91: ["इक्यानवे"],
    92: ["बानवे"],
    93: ["तिरानवे"],
    94: ["चौरानवे"],
    95: ["पचानवे"],
    96: ["छियानवे"],
    97: ["सत्तानवे"],
    98: ["अट्ठानवे"],
    99: ["निन्यानवे"],
}


def _flatten(width: int, *, exclude_zero: bool = False) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for n, words in sorted(_NUM_0_99.items()):
        if exclude_zero and n == 0:
            continue
        latin = str(n).rjust(width, "0")
        pairs.extend((w, latin) for w in words)
    return pairs


# ---------------------------------------------------------------------------
# Atomic FSTs.
# ---------------------------------------------------------------------------

# Spoken 0..99 -> 2-digit zero-padded Latin string.
_ZERO_99: Final[pynini.Fst] = pynini.string_map(_flatten(2)).optimize()
# Spoken 1..99 -> 2-digit zero-padded (used as "head" of larger scales).
_ONE_99: Final[pynini.Fst] = pynini.string_map(
    _flatten(2, exclude_zero=True)
).optimize()
# Spoken 1..9 -> 1-digit (hundreds-place head).
_ONE_9: Final[pynini.Fst] = pynini.string_map(
    [(w, str(n)) for n in range(1, 10) for w in _NUM_0_99[n]]
).optimize()

# Scale words and their spelling alternates. Per the plan:
# हज़ार / हजार (with / without nukta) and the canonical (with-nukta)
# forms for ddha-nukta / dda-nukta scales are accepted.
_SAU: Final[pynini.Fst] = pynini.union("सौ", "हंड्रेड", "हन्ड्रेड")
_HAZAR: Final[pynini.Fst] = pynini.union(
    "हज़ार",
    "हजार",
    "थाउजेंड",
    "थाउजंड",
    "थाउज़ेंड",
    "थाउज़ंड",
)
_LAKH: Final[pynini.Fst] = pynini.accep("लाख")
_KAROD: Final[pynini.Fst] = pynini.union("करोड़", "करोड")
_ARAB: Final[pynini.Fst] = pynini.accep("अरब")

_SP: Final[pynini.Fst] = pynini.accep(" ")  # word separator


# ---------------------------------------------------------------------------
# Recursive padded layers.
#
# Each ``_ZERO_<N>`` accepts a spoken Hindi span and emits a Latin
# integer string of fixed width ``ceil(log10(N+1))``. Padding propagates
# upward; leading zeros are stripped once at the end (`STRIP_LEADING`).
#
# Two branches per layer:
#   (a) lower scale, prefixed with extra "0"s so the width matches.
#   (b) "<head> SCALE [<sub>]"; ``<sub>`` defaults to all-zero when
#       absent.
# ---------------------------------------------------------------------------

# 0..999 -> 3 chars.
_ZERO_999: Final[pynini.Fst] = (
    pynutil.insert("0") + _ZERO_99
    | _ONE_9
    + pynutil.delete(_SP)
    + pynutil.delete(_SAU)
    + (pynutil.insert("00") | (pynutil.delete(_SP) + _ZERO_99))
).optimize()

# 0..99,999 -> 5 chars.
_ZERO_99999: Final[pynini.Fst] = (
    pynutil.insert("00") + _ZERO_999
    | _ONE_99
    + pynutil.delete(_SP)
    + pynutil.delete(_HAZAR)
    + (pynutil.insert("000") | (pynutil.delete(_SP) + _ZERO_999))
).optimize()

# 0..99,99,999 -> 7 chars.
_ZERO_9999999: Final[pynini.Fst] = (
    pynutil.insert("00") + _ZERO_99999
    | _ONE_99
    + pynutil.delete(_SP)
    + pynutil.delete(_LAKH)
    + (pynutil.insert("00000") | (pynutil.delete(_SP) + _ZERO_99999))
).optimize()

# 0..99,99,99,999 -> 9 chars (i.e. one less than 1 अरब).
_ZERO_999999999: Final[pynini.Fst] = (
    pynutil.insert("00") + _ZERO_9999999
    | _ONE_99
    + pynutil.delete(_SP)
    + pynutil.delete(_KAROD)
    + (pynutil.insert("0000000") | (pynutil.delete(_SP) + _ZERO_9999999))
).optimize()

# 0..1 अरब (10⁹ inclusive). The अरब head is restricted to "एक"
# because the plan caps coverage at exactly 1 अरब; larger spans like
# "दो अरब" are deferred until a future "billion+" extension.
_ARAB_HEAD: Final[pynini.Fst] = pynini.cross("एक", "1").optimize()

_ZERO_1_ARAB_PADDED: Final[pynini.Fst] = (
    pynutil.insert("0") + _ZERO_999999999
    | _ARAB_HEAD
    + pynutil.delete(_SP)
    + pynutil.delete(_ARAB)
    + (pynutil.insert("000000000") | (pynutil.delete(_SP) + _ZERO_999999999))
).optimize()


# ---------------------------------------------------------------------------
# Half / quarter compounds.
#
# Enumerated explicitly because the prefix changes the *value* of the
# scale, not just the count. Only the integer-producing combinations
# go here — bare ``डेढ़``/``ढाई``/``साढ़े N``/``पौने N``/``सवा X`` (no
# scale, or scale that yields a fractional result) are routed to
# ``decimal.py``.
# ---------------------------------------------------------------------------

_SAVAA: Final[pynini.Fst] = pynini.accep("सवा")
_DEDH: Final[pynini.Fst] = pynini.union("डेढ़", "डेढ")
_DHAII: Final[pynini.Fst] = pynini.accep("ढाई")
_SAADHE: Final[pynini.Fst] = pynini.union("साढ़े", "साढे")
_PAUNE: Final[pynini.Fst] = pynini.accep("पौने")


def _build_compound_fsts() -> list[pynini.Fst]:
    """Build per-quantity FSTs for each (prefix, scale) combination.

    Yielding an FST per quantity (rather than one giant string_map of
    spoken -> latin) means we can union spelling alternates of N — the
    ``_NUM_0_99`` table — without enumerating the cross-product of N
    spellings × scale spellings × prefix spellings.
    """
    fsts: list[pynini.Fst] = []
    scales: list[tuple[pynini.Fst, int, int]] = [
        # (scale word, scale value, max N for "<N> SCALE" pattern)
        (_SAU, 100, 9),
        (_HAZAR, 1_000, 99),
        (_LAKH, 100_000, 99),
        (_KAROD, 10_000_000, 99),
    ]
    for sc_word, sc, max_n in scales:
        # सवा SCALE -> SCALE * 1.25
        fsts.append(
            pynutil.delete(_SAVAA)
            + pynutil.delete(_SP)
            + pynutil.delete(sc_word)
            + pynutil.insert(str(sc + sc // 4))
        )
        # डेढ़ SCALE -> SCALE * 1.5
        fsts.append(
            pynutil.delete(_DEDH)
            + pynutil.delete(_SP)
            + pynutil.delete(sc_word)
            + pynutil.insert(str(sc + sc // 2))
        )
        # ढाई SCALE -> SCALE * 2.5
        fsts.append(
            pynutil.delete(_DHAII)
            + pynutil.delete(_SP)
            + pynutil.delete(sc_word)
            + pynutil.insert(str(2 * sc + sc // 2))
        )
        # साढ़े N SCALE for N >= 3 -> (N + 0.5) * SCALE
        for n in range(3, max_n + 1):
            value = n * sc + sc // 2
            for sp in _NUM_0_99[n]:
                fsts.append(
                    pynutil.delete(_SAADHE)
                    + pynutil.delete(_SP)
                    + pynutil.delete(pynini.accep(sp))
                    + pynutil.delete(_SP)
                    + pynutil.delete(sc_word)
                    + pynutil.insert(str(value))
                )
        # पौने N SCALE for N >= 1 -> (N - 0.25) * SCALE
        for n in range(1, max_n + 1):
            value = (n - 1) * sc + 3 * sc // 4
            for sp in _NUM_0_99[n]:
                fsts.append(
                    pynutil.delete(_PAUNE)
                    + pynutil.delete(_SP)
                    + pynutil.delete(pynini.accep(sp))
                    + pynutil.delete(_SP)
                    + pynutil.delete(sc_word)
                    + pynutil.insert(str(value))
                )
    return fsts


_COMPOUNDS: Final[pynini.Fst] = pynini.union(*_build_compound_fsts()).optimize()


# ---------------------------------------------------------------------------
# Latin passthrough.
#
# ASR mixed-script output sometimes already contains Latin digits (e.g.
# "मुझे 125 रुपये दो"); the cardinal grammar must accept those forms
# without reformatting.
# ---------------------------------------------------------------------------

_DIGIT: Final[pynini.Fst] = LATIN_DIGITS  # acceptor of one Latin digit
_NONZERO: Final[pynini.Fst] = pynini.string_map(
    [(str(d), str(d)) for d in range(1, 10)]
).optimize()

LATIN_INTEGER: Final[pynini.Fst] = (
    pynini.accep("0") | (_NONZERO + pynini.closure(_DIGIT))
).optimize()


# ---------------------------------------------------------------------------
# Strip leading zeros (the recursive padded chain emits fixed-width
# strings; compounds and Latin passthrough already canonical).
# ---------------------------------------------------------------------------

_ALL_ZEROS_TO_ZERO: pynini.Fst = pynini.cross(
    pynini.closure(pynini.accep("0"), 1), "0"
)

_STRIP_LEADING: Final[pynini.Fst] = (
    _ALL_ZEROS_TO_ZERO
    | (
        pynutil.delete(pynini.closure(pynini.accep("0")))
        + _NONZERO
        + pynini.closure(_DIGIT)
    )
).optimize()


# ---------------------------------------------------------------------------
# Public surfaces.
#
#   ``CARDINAL``            : spoken / Latin -> bare Latin integer
#                             (no leading zeros, no decoration).
#   ``CARDINAL_CLASSIFIER`` : the same, wrapped in NeMo-style
#                             ``cardinal { value: "<int>" }``.
# ---------------------------------------------------------------------------

CARDINAL: Final[pynini.Fst] = (
    ((_ZERO_1_ARAB_PADDED | _COMPOUNDS) @ _STRIP_LEADING) | LATIN_INTEGER
).optimize()

CARDINAL_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('cardinal { value: "') + CARDINAL + pynutil.insert('" }')
).optimize()


__all__ = ["CARDINAL", "CARDINAL_CLASSIFIER"]
