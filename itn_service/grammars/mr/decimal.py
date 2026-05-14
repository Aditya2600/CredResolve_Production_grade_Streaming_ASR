"""Marathi decimal grammar (spoken / Latin -> Latin decimal).

Mirror of ``grammars/hi/decimal.py`` with only the lexicon swapped:

* Decimal marker: Marathi ``दशांश`` (primary) and ``दशमलव`` (used in
  formal Marathi and by code-switched Hindi-Marathi speakers).
* Spoken 0..9 digits and half/quarter compound stems come from the
  Marathi cardinal lexicon (``cardinal.py``).

Three accepted surface families (identical structure to Hindi):

1. ``<int> दशांश <digits>``  — ``बारा दशांश पाच``  -> ``12.5``
2. Bare half/quarter compounds — ``दीड`` -> ``1.5``, ``अडीच`` ->
   ``2.5``, ``साडे पाच`` -> ``5.5``, ``पावणे चार`` -> ``3.75``,
   ``सव्वा बारा`` -> ``12.25``.
3. Latin passthrough — ``12.5`` -> ``12.5``.

Output wrapped in NeMo-style ``decimal { value: "<latin>" }`` via
:data:`DECIMAL_CLASSIFIER`.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS
from itn_service.grammars.mr.cardinal import (
    CARDINAL,
    LATIN_INTEGER,
    _ADEECH,
    _DEED,
    _NUM_0_99,
    _PAAVNE,
    _SAADE,
    _SAVVA,
    _SP,
)


# ---------------------------------------------------------------------------
# Spoken digit lexicon (0..9) for the fractional part. Marathi convention,
# like Hindi, reads the digits *after* the decimal marker one-by-one.
# ---------------------------------------------------------------------------

_SPOKEN_DIGIT_TO_LATN: Final[pynini.Fst] = pynini.string_map(
    [(w, str(n)) for n in range(10) for w in _NUM_0_99[n]]
).optimize()

_SPOKEN_FRAC_DIGITS: Final[pynini.Fst] = (
    _SPOKEN_DIGIT_TO_LATN
    + pynini.closure(pynutil.delete(_SP) + _SPOKEN_DIGIT_TO_LATN)
).optimize()

_LATIN_FRAC_DIGITS: Final[pynini.Fst] = LATIN_DIGITS.closure(1).optimize()

_FRAC_DIGITS: Final[pynini.Fst] = (
    _SPOKEN_FRAC_DIGITS | _LATIN_FRAC_DIGITS
).optimize()


# ---------------------------------------------------------------------------
# Spoken decimal: "<int> दशांश <digits>" -> "<int>.<digits>".
#
# ``दशांश`` is the Marathi-specific decimal marker; ``दशमलव`` is
# also accepted because formal Marathi uses it and ASR for code-
# switched Hindi/Marathi speech often emits it.
# ---------------------------------------------------------------------------

_DECIMAL_MARKER: Final[pynini.Fst] = pynini.union("दशांश", "दशमलव")

_DECIMAL_SPOKEN: Final[pynini.Fst] = (
    CARDINAL
    + pynutil.delete(_SP)
    + pynutil.delete(_DECIMAL_MARKER)
    + pynutil.insert(".")
    + pynutil.delete(_SP)
    + _FRAC_DIGITS
).optimize()


# ---------------------------------------------------------------------------
# Bare half / quarter compounds (Marathi forms).
# ---------------------------------------------------------------------------

def _build_compound_decimal_fsts() -> list[pynini.Fst]:
    fsts: list[pynini.Fst] = []

    # दीड -> "1.5"  (literal compound, no head)
    fsts.append(pynutil.delete(_DEED) + pynutil.insert("1.5"))
    # अडीच -> "2.5"  (literal compound, no head)
    fsts.append(pynutil.delete(_ADEECH) + pynutil.insert("2.5"))

    # सव्वा N -> "N.25" for N >= 1.
    for n in range(1, 100):
        for sp in _NUM_0_99[n]:
            fsts.append(
                pynutil.delete(_SAVVA)
                + pynutil.delete(_SP)
                + pynutil.delete(pynini.accep(sp))
                + pynutil.insert(f"{n}.25")
            )

    # साडे N -> "N.5" for N >= 3 (1.5 = दीड, 2.5 = अडीच have their own
    # bare words and never appear as साडे एक / साडे दोन).
    for n in range(3, 100):
        for sp in _NUM_0_99[n]:
            fsts.append(
                pynutil.delete(_SAADE)
                + pynutil.delete(_SP)
                + pynutil.delete(pynini.accep(sp))
                + pynutil.insert(f"{n}.5")
            )

    # पावणे N -> "(N-1).75" for N >= 1.
    for n in range(1, 100):
        for sp in _NUM_0_99[n]:
            fsts.append(
                pynutil.delete(_PAAVNE)
                + pynutil.delete(_SP)
                + pynutil.delete(pynini.accep(sp))
                + pynutil.insert(f"{n - 1}.75")
            )
    return fsts


_COMPOUND_DECIMAL: Final[pynini.Fst] = pynini.union(
    *_build_compound_decimal_fsts()
).optimize()


# ---------------------------------------------------------------------------
# Latin passthrough.
# ---------------------------------------------------------------------------

_LATIN_DECIMAL: Final[pynini.Fst] = (
    LATIN_INTEGER + pynini.accep(".") + _LATIN_FRAC_DIGITS
).optimize()


# ---------------------------------------------------------------------------
# Public surfaces.
# ---------------------------------------------------------------------------

DECIMAL: Final[pynini.Fst] = (
    _DECIMAL_SPOKEN | _COMPOUND_DECIMAL | _LATIN_DECIMAL
).optimize()

DECIMAL_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('decimal { value: "') + DECIMAL + pynutil.insert('" }')
).optimize()


__all__ = ["DECIMAL", "DECIMAL_CLASSIFIER"]
