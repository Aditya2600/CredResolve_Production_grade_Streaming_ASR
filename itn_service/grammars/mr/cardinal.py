"""Marathi cardinal grammar (spoken -> Latin integer).

Mirror of ``grammars/hi/cardinal.py`` with only the *lexicon* swapped —
the FST topology (0..99 atom, 0..999 recursion, 0..1 अब्ज cap, half /
quarter compound layer, Latin passthrough, strip-leading-zeros) is
identical. The intent is to validate that the per-language template
established by Hindi is re-usable; structural drift between this file
and ``hi/cardinal.py`` is therefore a bug.

Marathi-specific lexicon swaps:

* **0..99 number words.** Marathi cardinals have entirely different
  surface forms than Hindi for most quantities (e.g. ``दोन`` vs
  ``दो``, ``सहा`` vs ``छह``, ``नऊ`` vs ``नौ``, ``दहा`` vs ``दस``,
  ``अकरा`` vs ``ग्यारह``). Every Marathi 0..99 entry is enumerated
  explicitly in :data:`_NUM_0_99`.

* **Hundreds compound.** Marathi attaches the hundred marker as a
  suffix: ``एकशे`` (= 100), ``दोनशे`` (= 200) ... ``नऊशे`` (= 900).
  Hindi treats ``सौ`` as a separate word (``एक सौ``, ``दो सौ`` ...).
  We additionally accept ``शंभर`` as a standalone surface for 100.

* **Crore.** Marathi prefers ``कोटी``; we also accept ``करोड`` /
  ``करोड़`` to absorb code-switched Hindi.

* **Billion (10⁹) cap.** Hindi: ``अरब``. Marathi: ``अब्ज`` is the
  native term; ``अरब`` is also accepted for code-switched callers.

* **Half / quarter compounds.** Marathi forms (with the dialectal
  alternates the plan calls out):

      सव्वा / सवा       quarter-past   (Hindi: सवा)
      दीड / डीड         one-and-a-half (Hindi: डेढ़)
      अडीच / अडिच       two-and-a-half (Hindi: ढाई — completely different)
      साडे / साढे       N-and-a-half   (Hindi: साढ़े)
      पावणे / पाउणे     quarter-to     (Hindi: पौने)

Output: NeMo-style ``cardinal { value: "<latin_int>" }`` via
:data:`CARDINAL_CLASSIFIER`. The undecorated :data:`CARDINAL` is also
exposed for compose-style use inside money / percent / decimal.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS

# ---------------------------------------------------------------------------
# Spoken 0..99 lexicon. Marathi-specific.
#
# Many quantities have multiple accepted surface forms in everyday
# Marathi (e.g. ``सव्वीस`` / ``सब्बीस``, ``त्र्याऐंशी`` /
# ``त्रेऐंशी``); we enumerate the most-frequent alternates here and
# leave less-common variants for the native-speaker review. Items
# marked "REVIEW" must be confirmed before the grammar ships.
# ---------------------------------------------------------------------------

_NUM_0_99: Final[dict[int, list[str]]] = {
    0: ["शून्य"],
    1: ["एक"],
    2: ["दोन"],
    3: ["तीन"],
    4: ["चार"],
    5: ["पाच"],
    6: ["सहा"],
    7: ["सात"],
    8: ["आठ"],
    9: ["नऊ", "नउ"],
    10: ["दहा"],
    11: ["अकरा"],
    12: ["बारा"],
    13: ["तेरा"],
    14: ["चौदा"],
    15: ["पंधरा"],
    16: ["सोळा"],
    17: ["सतरा"],
    18: ["अठरा"],
    19: ["एकोणीस", "एकोणिस"],
    20: ["वीस", "विस"],
    21: ["एकवीस", "एकविस"],
    22: ["बावीस", "बाविस"],
    23: ["तेवीस", "तेविस"],
    24: ["चोवीस", "चोविस"],
    25: ["पंचवीस", "पंचविस"],
    26: ["सव्वीस", "सब्बीस"],
    27: ["सत्तावीस", "सत्ताविस"],
    28: ["अठ्ठावीस", "अठ्ठाविस"],
    29: ["एकोणतीस"],
    30: ["तीस"],
    31: ["एकतीस", "एकत्तीस"],
    32: ["बत्तीस"],
    33: ["तेहतीस", "तेहेतीस"],
    34: ["चौतीस"],
    35: ["पस्तीस", "पंचतीस"],
    36: ["छत्तीस"],
    37: ["सदतीस", "सदोतीस"],
    38: ["अडतीस"],
    39: ["एकोणचाळीस"],
    40: ["चाळीस"],
    41: ["एक्केचाळीस", "एकेचाळीस"],
    42: ["बेचाळीस"],
    43: ["त्रेचाळीस"],
    44: ["चव्वेचाळीस", "चौवेचाळीस"],
    45: ["पंचेचाळीस"],
    46: ["सेहेचाळीस", "शेचाळीस"],
    47: ["सत्तेचाळीस"],
    48: ["अठ्ठेचाळीस"],
    49: ["एकोणपन्नास"],
    50: ["पन्नास"],
    51: ["एक्कावन्न"],
    52: ["बावन्न"],
    53: ["त्रेपन्न"],
    54: ["चोपन्न", "चौपन्न"],
    55: ["पंचावन्न"],
    56: ["छप्पन्न"],
    57: ["सत्तावन्न"],
    58: ["अठ्ठावन्न"],
    59: ["एकोणसाठ"],
    60: ["साठ"],
    61: ["एकसष्ठ", "एक्साष्ठ"],
    62: ["बासष्ठ"],
    63: ["त्रेसष्ठ"],
    64: ["चौसष्ठ"],
    65: ["पासष्ठ"],
    66: ["सहासष्ठ"],
    67: ["सदुसष्ठ"],
    68: ["अडुसष्ठ"],
    69: ["एकोणसत्तर"],
    70: ["सत्तर"],
    71: ["एकाहत्तर"],
    72: ["बाहत्तर"],
    73: ["त्र्याहत्तर", "त्र्याह्त्तर"],
    74: ["चौर्‍याहत्तर", "चौऱ्याहत्तर"],
    75: ["पंचाहत्तर"],
    76: ["शाहत्तर"],
    77: ["सत्त्याहत्तर"],
    78: ["अठ्ठ्याहत्तर"],
    79: ["एकोण्यांशी", "एकोणऐंशी"],
    80: ["ऐंशी", "एंशी"],
    81: ["एक्क्याऐंशी", "एक्याऐंशी"],
    82: ["ब्याऐंशी"],
    83: ["त्र्याऐंशी"],
    84: ["चौऱ्याऐंशी", "चौर्‍याऐंशी"],
    85: ["पंच्याऐंशी", "पंचाऐंशी"],
    86: ["शहाऐंशी"],
    87: ["सत्त्याऐंशी"],
    88: ["अठ्ठ्याऐंशी"],
    89: ["एकोणनव्वद"],
    90: ["नव्वद"],
    91: ["एक्क्याण्णव", "एक्याण्णव"],
    92: ["ब्याण्णव"],
    93: ["त्र्याण्णव"],
    94: ["चौऱ्याण्णव", "चौर्‍याण्णव"],
    95: ["पंच्याण्णव", "पंचाण्णव"],
    96: ["शहाण्णव"],
    97: ["सत्त्याण्णव"],
    98: ["अठ्ठ्याण्णव"],
    99: ["नव्व्याण्णव", "नव्याण्णव"],
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

_ZERO_99: Final[pynini.Fst] = pynini.string_map(_flatten(2)).optimize()
_ONE_99: Final[pynini.Fst] = pynini.string_map(
    _flatten(2, exclude_zero=True)
).optimize()


# ---------------------------------------------------------------------------
# Hundreds compound — the Marathi-specific lexical structure.
#
# Marathi attaches शे as a suffix to the units head:
#   एकशे = 100, दोनशे = 200, ..., नऊशे = 900.
# Plus the standalone form शंभर = 100. This replaces Hindi's
# "<head> <space> सौ" pattern. The FST emits a 1-digit head ("1".."9"),
# matching the shape Hindi's ``_ONE_9 + delete(SP) + delete(SAU)`` chain
# produces, so the layer above (``_ZERO_999``) is unchanged.
# ---------------------------------------------------------------------------

def _build_hundred_head_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    # N-शे compound for N in 1..9.
    for n in range(1, 10):
        for w in _NUM_0_99[n]:
            pairs.append((f"{w}शे", str(n)))
    # शंभर is a synonym for एकशे (= 100).
    pairs.append(("शंभर", "1"))
    return pairs


_HUNDRED_HEAD: Final[pynini.Fst] = pynini.string_map(
    _build_hundred_head_pairs()
).optimize()

# Scale words. हजार with / without nukta absorbs the Hindi spelling for
# code-switched callers; कोटी / करोड / करोड़ likewise.
_HAZAR: Final[pynini.Fst] = pynini.union("हजार", "हज़ार")
_LAKH: Final[pynini.Fst] = pynini.accep("लाख")
_KOTI: Final[pynini.Fst] = pynini.union("कोटी", "करोड", "करोड़")
_ABJA: Final[pynini.Fst] = pynini.union("अब्ज", "अरब")

_SP: Final[pynini.Fst] = pynini.accep(" ")


# ---------------------------------------------------------------------------
# Recursive padded layers — identical topology to Hindi.
# ---------------------------------------------------------------------------

# 0..999 -> 3 chars. Note: no _SAU + delete(SP) chain here because
# Marathi's hundred is encoded as a *single token* (``Nशे`` / ``शंभर``).
_ZERO_999: Final[pynini.Fst] = (
    pynutil.insert("0") + _ZERO_99
    | _HUNDRED_HEAD
    + (pynutil.insert("00") | (pynutil.delete(_SP) + _ZERO_99))
).optimize()

_ZERO_99999: Final[pynini.Fst] = (
    pynutil.insert("00") + _ZERO_999
    | _ONE_99
    + pynutil.delete(_SP)
    + pynutil.delete(_HAZAR)
    + (pynutil.insert("000") | (pynutil.delete(_SP) + _ZERO_999))
).optimize()

_ZERO_9999999: Final[pynini.Fst] = (
    pynutil.insert("00") + _ZERO_99999
    | _ONE_99
    + pynutil.delete(_SP)
    + pynutil.delete(_LAKH)
    + (pynutil.insert("00000") | (pynutil.delete(_SP) + _ZERO_99999))
).optimize()

_ZERO_999999999: Final[pynini.Fst] = (
    pynutil.insert("00") + _ZERO_9999999
    | _ONE_99
    + pynutil.delete(_SP)
    + pynutil.delete(_KOTI)
    + (pynutil.insert("0000000") | (pynutil.delete(_SP) + _ZERO_9999999))
).optimize()

# 0..1 अब्ज (10⁹ inclusive). Head pinned to "एक" — coverage caps at 1 अब्ज.
_ABJA_HEAD: Final[pynini.Fst] = pynini.cross("एक", "1").optimize()

_ZERO_1_ABJA_PADDED: Final[pynini.Fst] = (
    pynutil.insert("0") + _ZERO_999999999
    | _ABJA_HEAD
    + pynutil.delete(_SP)
    + pynutil.delete(_ABJA)
    + (pynutil.insert("000000000") | (pynutil.delete(_SP) + _ZERO_999999999))
).optimize()


# ---------------------------------------------------------------------------
# Half / quarter compounds.
# ---------------------------------------------------------------------------

_SAVVA: Final[pynini.Fst] = pynini.union("सव्वा", "सवा")
_DEED: Final[pynini.Fst] = pynini.union("दीड", "डीड")
_ADEECH: Final[pynini.Fst] = pynini.union("अडीच", "अडिच")
_SAADE: Final[pynini.Fst] = pynini.union("साडे", "साढे")
_PAAVNE: Final[pynini.Fst] = pynini.union("पावणे", "पाउणे", "पाउने")


def _scale_words_with_hundred_token() -> list[tuple[pynini.Fst, int, int, bool]]:
    """Return list of (scale_acceptor, scale_value, max_N, is_hundred).

    For Marathi the "hundred" scale appears as a *standalone* word
    ``शंभर`` in compound contexts (``सव्वा शंभर`` = 125, ``दीड शंभर``
    = 150). This is the surface speakers actually use when prefixing
    a half/quarter to a hundred; ``सव्वाशे`` is non-idiomatic.
    """
    return [
        (pynini.accep("शंभर"), 100, 9, True),
        (_HAZAR, 1_000, 99, False),
        (_LAKH, 100_000, 99, False),
        (_KOTI, 10_000_000, 99, False),
    ]


def _build_compound_fsts() -> list[pynini.Fst]:
    fsts: list[pynini.Fst] = []
    for sc_word, sc, max_n, _is_hundred in _scale_words_with_hundred_token():
        fsts.append(
            pynutil.delete(_SAVVA)
            + pynutil.delete(_SP)
            + pynutil.delete(sc_word)
            + pynutil.insert(str(sc + sc // 4))
        )
        fsts.append(
            pynutil.delete(_DEED)
            + pynutil.delete(_SP)
            + pynutil.delete(sc_word)
            + pynutil.insert(str(sc + sc // 2))
        )
        fsts.append(
            pynutil.delete(_ADEECH)
            + pynutil.delete(_SP)
            + pynutil.delete(sc_word)
            + pynutil.insert(str(2 * sc + sc // 2))
        )
        for n in range(3, max_n + 1):
            value = n * sc + sc // 2
            for sp in _NUM_0_99[n]:
                fsts.append(
                    pynutil.delete(_SAADE)
                    + pynutil.delete(_SP)
                    + pynutil.delete(pynini.accep(sp))
                    + pynutil.delete(_SP)
                    + pynutil.delete(sc_word)
                    + pynutil.insert(str(value))
                )
        for n in range(1, max_n + 1):
            value = (n - 1) * sc + 3 * sc // 4
            for sp in _NUM_0_99[n]:
                fsts.append(
                    pynutil.delete(_PAAVNE)
                    + pynutil.delete(_SP)
                    + pynutil.delete(pynini.accep(sp))
                    + pynutil.delete(_SP)
                    + pynutil.delete(sc_word)
                    + pynutil.insert(str(value))
                )
    return fsts


_COMPOUNDS: Final[pynini.Fst] = pynini.union(*_build_compound_fsts()).optimize()


# ---------------------------------------------------------------------------
# Latin passthrough. Identical to Hindi.
# ---------------------------------------------------------------------------

_DIGIT: Final[pynini.Fst] = LATIN_DIGITS
_NONZERO: Final[pynini.Fst] = pynini.string_map(
    [(str(d), str(d)) for d in range(1, 10)]
).optimize()

LATIN_INTEGER: Final[pynini.Fst] = (
    pynini.accep("0") | (_NONZERO + pynini.closure(_DIGIT))
).optimize()


# ---------------------------------------------------------------------------
# Strip leading zeros. Identical to Hindi.
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
# ---------------------------------------------------------------------------

CARDINAL: Final[pynini.Fst] = (
    ((_ZERO_1_ABJA_PADDED | _COMPOUNDS) @ _STRIP_LEADING) | LATIN_INTEGER
).optimize()

CARDINAL_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('cardinal { value: "') + CARDINAL + pynutil.insert('" }')
).optimize()


__all__ = ["CARDINAL", "CARDINAL_CLASSIFIER"]
