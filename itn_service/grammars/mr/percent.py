"""Marathi percent grammar (cue-required, spoken/Latin -> ``<num>%``).

Mirror of ``grammars/hi/percent.py`` with only the cue-word lexicon
swapped:

* Marathi-specific percent cues: ``टक्के`` (the everyday Marathi
  surface form), ``टक्का`` (singular), and the formal Sanskritic
  ``प्रतिशत`` (shared with Hindi — kept for code-switched callers).
* The literal ``%`` symbol is always accepted (locale-neutral).

Surface forms supported (cue follows the number, per Marathi/Hindi
word order):

    बारा टक्के                     -> 12%
    पाच टक्के                      -> 5%
    एकशे टक्के                     -> 100%
    बारा दशांश पाच टक्के          -> 12.5%
    शून्य दशांश एक टक्के          -> 0.1%
    सव्वा बारा टक्के              -> 12.25%   (bare half/quarter compound)
    साडे पाच टक्के                 -> 5.5%
    12 टक्के                       -> 12%      (Latin passthrough + cue)
    12.5%                           -> 12.5%
    12.5 %                          -> 12.5%
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.mr.cardinal import CARDINAL, _SP
from itn_service.grammars.mr.decimal import DECIMAL


# ---------------------------------------------------------------------------
# Number side: cardinal | decimal.
# ---------------------------------------------------------------------------

_NUMBER: Final[pynini.Fst] = (CARDINAL | DECIMAL).optimize()


# ---------------------------------------------------------------------------
# Marathi percent cues. ``टक्के`` / ``टक्का`` are the surface forms
# that dominate everyday Marathi (and are absent from Hindi); the
# formal Sanskritic ``प्रतिशत`` is accepted because both languages use
# it in writing and code-switched ASR output.
# ---------------------------------------------------------------------------

_PERCENT_MARATHI: Final[pynini.Fst] = pynini.union(
    "टक्के", "टक्का", "प्रतिशत",
)
_PERCENT_PCT: Final[pynini.Fst] = pynini.accep("%")


_VALUE_MR_CUE: Final[pynini.Fst] = (
    _NUMBER + pynutil.delete(_SP) + pynutil.delete(_PERCENT_MARATHI)
).optimize()

_VALUE_PCT_NOSP: Final[pynini.Fst] = (
    _NUMBER + pynutil.delete(_PERCENT_PCT)
).optimize()

_VALUE_PCT_SP: Final[pynini.Fst] = (
    _NUMBER + pynutil.delete(_SP) + pynutil.delete(_PERCENT_PCT)
).optimize()


_PERCENT_VALUE: Final[pynini.Fst] = pynini.union(
    _VALUE_MR_CUE,
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
