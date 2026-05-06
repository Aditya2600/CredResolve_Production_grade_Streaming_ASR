"""Hindi decimal grammar (spoken / Latin -> Latin decimal).

Three surface forms are accepted:

1. **Spoken decimal with explicit ``दशमलव`` marker.** The integer part
   is any value the cardinal grammar can parse (0..1 अरब, including
   integer-producing half/quarter compounds), the decimal marker is
   the word ``दशमलव``, and the fractional part is read digit-by-digit
   in space-separated Hindi digit words::

       बारह दशमलव पाँच       -> 12.5
       शून्य दशमलव एक दो पाँच -> 0.125

2. **Bare half / quarter compounds** that yield a fractional value
   (``डेढ़``, ``ढाई``, ``साढ़े N``, ``पौने N``, ``सवा N`` — i.e. all
   the cases the cardinal grammar declines to accept)::

       डेढ़           -> 1.5
       ढाई            -> 2.5
       साढ़े पाँच      -> 5.5
       पौने चार        -> 3.75
       सवा बारह       -> 12.25

3. **Latin passthrough** for ASR output that already carries a decimal
   surface form (``12.5``, ``0.125``, etc.).

Output is wrapped in NeMo-style ``decimal { value: "<latin>" }`` via
:data:`DECIMAL_CLASSIFIER`. Per ``CONTRIBUTING.md`` invariant 4 the
canonical decimal separator is ``.`` (Latin full stop), independent of
locale.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS

# Re-use the cardinal grammar's integer parser and the spoken 0-99 table.
# Importing private names is intentional: cardinal.py is the single
# source of truth for spoken Hindi number lexica.
from itn_service.grammars.hi.cardinal import (
    CARDINAL,
    LATIN_INTEGER,
    _DEDH,
    _DHAII,
    _NUM_0_99,
    _PAUNE,
    _SAADHE,
    _SAVAA,
    _SP,
)


# ---------------------------------------------------------------------------
# Spoken digit lexicon (0..9) for the fractional part.
#
# Hindi convention reads the digits *after* दशमलव one-by-one rather
# than as a single number. We therefore want a strict 0..9 lookup that
# explicitly does not match longer 10..99 forms.
# ---------------------------------------------------------------------------

_SPOKEN_DIGIT_TO_LATN: Final[pynini.Fst] = pynini.string_map(
    [(w, str(n)) for n in range(10) for w in _NUM_0_99[n]]
).optimize()

# Repeated digit-by-digit fractional reading: at least one digit, with
# space-separated continuations.
_SPOKEN_FRAC_DIGITS: Final[pynini.Fst] = (
    _SPOKEN_DIGIT_TO_LATN
    + pynini.closure(pynutil.delete(_SP) + _SPOKEN_DIGIT_TO_LATN)
).optimize()

# Latin fractional part: a run of ASCII digits (no separators).
_LATIN_FRAC_DIGITS: Final[pynini.Fst] = LATIN_DIGITS.closure(1).optimize()

_FRAC_DIGITS: Final[pynini.Fst] = (
    _SPOKEN_FRAC_DIGITS | _LATIN_FRAC_DIGITS
).optimize()


# ---------------------------------------------------------------------------
# Spoken decimal: "<int> दशमलव <digits>" -> "<int>.<digits>"
# ---------------------------------------------------------------------------

_DASHMALAV: Final[pynini.Fst] = pynini.accep("दशमलव")

_DECIMAL_SPOKEN: Final[pynini.Fst] = (
    CARDINAL
    + pynutil.delete(_SP)
    + pynutil.delete(_DASHMALAV)
    + pynutil.insert(".")
    + pynutil.delete(_SP)
    + _FRAC_DIGITS
).optimize()


# ---------------------------------------------------------------------------
# Bare half / quarter compounds (no scale word -> fractional result).
# ---------------------------------------------------------------------------

def _build_compound_decimal_fsts() -> list[pynini.Fst]:
    """One FST per (prefix, head N) producing ``N + offset`` as a Latin
    decimal string."""
    fsts: list[pynini.Fst] = []

    # डेढ़ -> "1.5"  (literal compound, no head)
    fsts.append(pynutil.delete(_DEDH) + pynutil.insert("1.5"))
    # ढाई -> "2.5"  (literal compound, no head)
    fsts.append(pynutil.delete(_DHAII) + pynutil.insert("2.5"))

    # सवा N -> "N.25" for N >= 1.
    for n in range(1, 100):
        for sp in _NUM_0_99[n]:
            fsts.append(
                pynutil.delete(_SAVAA)
                + pynutil.delete(_SP)
                + pynutil.delete(pynini.accep(sp))
                + pynutil.insert(f"{n}.25")
            )

    # साढ़े N -> "N.5" for N >= 3 (1.5 = डेढ़, 2.5 = ढाई have their
    # own bare words and never appear as साढ़े एक / साढ़े दो).
    for n in range(3, 100):
        for sp in _NUM_0_99[n]:
            fsts.append(
                pynutil.delete(_SAADHE)
                + pynutil.delete(_SP)
                + pynutil.delete(pynini.accep(sp))
                + pynutil.insert(f"{n}.5")
            )

    # पौने N -> "(N-1).75" for N >= 1.
    # पौने एक = 0.75, पौने दो = 1.75, ..., पौने निन्यानवे = 98.75.
    for n in range(1, 100):
        for sp in _NUM_0_99[n]:
            fsts.append(
                pynutil.delete(_PAUNE)
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
