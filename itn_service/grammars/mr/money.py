"""Marathi money grammar (cue-required, spoken/Latin -> ₹<grouped>[.<paise>]).

Mirror of ``grammars/hi/money.py`` with only the lexicon swapped:

* Currency cue words: ``रुपये`` / ``रुपया`` / ``रू`` (shared with
  Hindi — the surface form of the rupee word is identical across
  Hindi and Marathi), plus the rupee symbol ``₹`` and the romanised
  ``Rs`` / ``Rs.`` / ``INR``.
* Paise cue: ``पैसे`` / ``पैसा`` (shared).
* Integer / decimal sides come from the Marathi cardinal / decimal
  grammars, so the number-word parsing reflects Marathi forms
  (``एकशे पंचवीस रुपये`` -> ``₹125``).

Indian thousands grouping (1,25,000-style) is applied to every output
integer regardless of locale — CLDR records the Indian grouping rule
for ``mr`` as well as ``hi``.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS
from itn_service.grammars.mr.cardinal import CARDINAL, _NUM_0_99, _SP
from itn_service.grammars.mr.decimal import DECIMAL


# ---------------------------------------------------------------------------
# Indian thousands grouping. Topology copied verbatim from
# ``grammars/hi/money.py``; no language-specific logic here.
# ---------------------------------------------------------------------------

_NONZERO_LATIN_DIGIT: Final[pynini.Fst] = pynini.string_map(
    [(str(d), str(d)) for d in range(1, 10)]
).optimize()


def _grouping_for_length(n: int) -> pynini.Fst:
    if n < 1:
        raise ValueError(n)
    if n == 1:
        return LATIN_DIGITS
    parts: list[pynini.Fst] = [_NONZERO_LATIN_DIGIT]
    for p in range(1, n):
        if n >= 4 and (p == n - 3 or (p < n - 3 and (n - 3 - p) % 2 == 0)):
            parts.append(pynutil.insert(","))
        parts.append(LATIN_DIGITS)
    out = parts[0]
    for x in parts[1:]:
        out = out + x
    return out.optimize()


_GROUPED_INT: Final[pynini.Fst] = pynini.union(
    *[_grouping_for_length(n) for n in range(1, 11)]
).optimize()


# ---------------------------------------------------------------------------
# Currency / paise lexical cues. Marathi-specific note: although the
# surface form ``रुपये`` is shared with Hindi, retaining it here as a
# per-language constant lets a future Marathi-only synonym (e.g. a
# regional / archaic term) be added without touching the Hindi grammar.
# ---------------------------------------------------------------------------

_RUPEE_WORD: Final[pynini.Fst] = pynini.union("रुपये", "रुपया", "रू")

_RUPEE_SYMBOL: Final[pynini.Fst] = pynini.union(
    "₹", "Rs.", "Rs", "rs.", "rs", "RS.", "RS", "INR", "inr",
)

_PAISE_WORD: Final[pynini.Fst] = pynini.union("पैसे", "पैसा")


# Spoken paise 1..99 -> 2-digit zero-padded ("पाच" -> "05").
_PAISE_SPOKEN: Final[pynini.Fst] = pynini.string_map(
    [
        (w, str(n).rjust(2, "0"))
        for n in range(1, 100)
        for w in _NUM_0_99[n]
    ]
).optimize()

_PAISE_LATIN: Final[pynini.Fst] = (LATIN_DIGITS + LATIN_DIGITS).optimize()

_PAISE_INT: Final[pynini.Fst] = (_PAISE_SPOKEN | _PAISE_LATIN).optimize()


# ---------------------------------------------------------------------------
# Cardinal / decimal lifted into grouped form.
# ---------------------------------------------------------------------------

_INT_GROUPED: Final[pynini.Fst] = (CARDINAL @ _GROUPED_INT).optimize()

_DOT_AND_FRAC: Final[pynini.Fst] = (
    pynini.accep(".") + LATIN_DIGITS.closure(1)
).optimize()
_DECIMAL_SPLITTER: Final[pynini.Fst] = (_GROUPED_INT + _DOT_AND_FRAC).optimize()
_DECIMAL_GROUPED: Final[pynini.Fst] = (DECIMAL @ _DECIMAL_SPLITTER).optimize()


# ---------------------------------------------------------------------------
# Already-grouped Latin integer (strip commas, then re-group canonically).
# ---------------------------------------------------------------------------

_DIGIT_OR_COMMA_SIGMA: Final[pynini.Fst] = pynini.union(
    *"0123456789", ",",
).closure().optimize()

_STRIP_COMMAS: Final[pynini.Fst] = pynini.cdrewrite(
    pynutil.delete(","),
    "",
    "",
    _DIGIT_OR_COMMA_SIGMA,
).optimize()

_MAYBE_GROUPED_LATIN_INT: Final[pynini.Fst] = pynini.union(
    LATIN_DIGITS, pynini.accep(","),
).closure(1).optimize()

_REGROUPED_LATIN_INT: Final[pynini.Fst] = (
    _MAYBE_GROUPED_LATIN_INT @ _STRIP_COMMAS @ _GROUPED_INT
).optimize()

_REGROUPED_LATIN_DECIMAL: Final[pynini.Fst] = (
    _REGROUPED_LATIN_INT + _DOT_AND_FRAC
).optimize()


# ---------------------------------------------------------------------------
# Surface forms.
# ---------------------------------------------------------------------------

_OPT_DEL_SP: Final[pynini.Fst] = pynutil.delete(_SP).ques

_AMOUNT_INT_POSTCUE: Final[pynini.Fst] = (
    _INT_GROUPED
    + pynutil.delete(_SP)
    + pynutil.delete(_RUPEE_WORD)
).optimize()

_AMOUNT_INT_PAISE: Final[pynini.Fst] = (
    _INT_GROUPED
    + pynutil.delete(_SP)
    + pynutil.delete(_RUPEE_WORD)
    + pynutil.delete(_SP)
    + pynutil.insert(".")
    + _PAISE_INT
    + pynutil.delete(_SP)
    + pynutil.delete(_PAISE_WORD)
).optimize()

_AMOUNT_DECIMAL_POSTCUE: Final[pynini.Fst] = (
    _DECIMAL_GROUPED
    + pynutil.delete(_SP)
    + pynutil.delete(_RUPEE_WORD)
).optimize()

_AMOUNT_INT_PRECUE: Final[pynini.Fst] = (
    pynutil.delete(_RUPEE_WORD)
    + pynutil.delete(_SP)
    + _INT_GROUPED
).optimize()

_AMOUNT_DECIMAL_PRECUE: Final[pynini.Fst] = (
    pynutil.delete(_RUPEE_WORD)
    + pynutil.delete(_SP)
    + _DECIMAL_GROUPED
).optimize()

_AMOUNT_SYMBOL_INT: Final[pynini.Fst] = (
    pynutil.delete(_RUPEE_SYMBOL)
    + _OPT_DEL_SP
    + _REGROUPED_LATIN_INT
).optimize()

_AMOUNT_SYMBOL_DECIMAL: Final[pynini.Fst] = (
    pynutil.delete(_RUPEE_SYMBOL)
    + _OPT_DEL_SP
    + _REGROUPED_LATIN_DECIMAL
).optimize()


_MONEY_AMOUNT: Final[pynini.Fst] = pynini.union(
    _AMOUNT_INT_POSTCUE,
    _AMOUNT_INT_PAISE,
    _AMOUNT_DECIMAL_POSTCUE,
    _AMOUNT_INT_PRECUE,
    _AMOUNT_DECIMAL_PRECUE,
    _AMOUNT_SYMBOL_INT,
    _AMOUNT_SYMBOL_DECIMAL,
).optimize()


# ---------------------------------------------------------------------------
# Public surfaces.
# ---------------------------------------------------------------------------

MONEY: Final[pynini.Fst] = (pynutil.insert("₹") + _MONEY_AMOUNT).optimize()

MONEY_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('money { currency: "INR" amount: "')
    + _MONEY_AMOUNT
    + pynutil.insert('" }')
).optimize()


__all__ = ["MONEY", "MONEY_CLASSIFIER"]
