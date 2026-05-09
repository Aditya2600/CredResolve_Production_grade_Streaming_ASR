"""Hindi percent grammar (cue-required, spoken/Latin -> ``<num>%``).

The cue is **required**: at least one of ``प्रतिशत``, ``फीसदी``,
``फ़ीसदी`` (with-nukta variant), or the literal ``%`` symbol must be
present. Bare numeric spans never normalise to percent — see the
"Confidence gating and fallback" / Decimal-percentage row of the
implementation blueprint.

Surface forms supported (cue follows the number, per Hindi word order):

    बारह प्रतिशत                  -> 12%
    पाँच फीसदी                    -> 5%
    सौ प्रतिशत                    -> 100%
    बारह दशमलव पाँच प्रतिशत      -> 12.5%
    शून्य दशमलव एक प्रतिशत       -> 0.1%
    सवा बारह प्रतिशत              -> 12.25%   (bare half/quarter compound)
    साढ़े पाँच प्रतिशत             -> 5.5%
    12 प्रतिशत                    -> 12%      (Latin passthrough + cue)
    12.5%                          -> 12.5%    (compact symbol form)
    12.5 %                         -> 12.5%

Output is the bare number followed by ``%``. Percent values are not
Indian-grouped (they are typically <= 1000); the integer part is
emitted as the bare CARDINAL/DECIMAL output.

NeMo classifier wrapper :data:`PERCENT_CLASSIFIER` emits
``percent { value: "<num>" }`` (without the ``%``) so downstream
verbalisers can choose the locale's percent sign. Per CONTRIBUTING.md
invariants the FST is built at module import.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.hi.cardinal import CARDINAL, _SP
from itn_service.grammars.hi.decimal import DECIMAL


# ---------------------------------------------------------------------------
# Number side: cardinal | decimal. Disjoint surfaces (decimal requires a
# दशमलव marker, a half/quarter compound, or a Latin "X.Y").
# ---------------------------------------------------------------------------

_NUMBER: Final[pynini.Fst] = (CARDINAL | DECIMAL).optimize()


# ---------------------------------------------------------------------------
# Cues. Hindi word cues require a separating space; the ``%`` symbol
# may attach with or without a space.
# ---------------------------------------------------------------------------

_PERCENT_HINDI: Final[pynini.Fst] = pynini.union("प्रतिशत", "फीसदी", "फ़ीसदी")
_PERCENT_PCT: Final[pynini.Fst] = pynini.accep("%")


_VALUE_HINDI_CUE: Final[pynini.Fst] = (
    _NUMBER + pynutil.delete(_SP) + pynutil.delete(_PERCENT_HINDI)
).optimize()

_VALUE_PCT_NOSP: Final[pynini.Fst] = (
    _NUMBER + pynutil.delete(_PERCENT_PCT)
).optimize()

_VALUE_PCT_SP: Final[pynini.Fst] = (
    _NUMBER + pynutil.delete(_SP) + pynutil.delete(_PERCENT_PCT)
).optimize()


_PERCENT_VALUE: Final[pynini.Fst] = pynini.union(
    _VALUE_HINDI_CUE,
    _VALUE_PCT_NOSP,
    _VALUE_PCT_SP,
).optimize()


# ---------------------------------------------------------------------------
# Public surfaces.
# ---------------------------------------------------------------------------

PERCENT: Final[pynini.Fst] = (_PERCENT_VALUE + pynutil.insert("%")).optimize()

PERCENT_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('percent { value: "')
    + _PERCENT_VALUE
    + pynutil.insert('" }')
).optimize()


__all__ = ["PERCENT", "PERCENT_CLASSIFIER"]
