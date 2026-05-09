"""Hindi money grammar (cue-required, spoken/Latin -> ₹<grouped>[.<paise>]).

The grammar normalises Indian rupee amounts only when the span carries
an explicit currency cue: the Hindi rupee words ``रुपये``, ``रुपया``,
``रू``, the rupee symbol ``₹``, or the romanised ``Rs`` / ``Rs.`` /
``INR``. Bare numeric spans **never** normalise to money — the plan's
"Confidence gating and fallback" table requires a lexical cue for the
currency class (see ``docs/implementation_bluprint_INR.md``).

Surface forms supported:

1. Spoken integer + post-cue::

       एक हज़ार रुपये                 -> ₹1,000
       एक सौ पच्चीस रुपये             -> ₹125
       ढाई लाख रुपये                  -> ₹2,50,000
       साढ़े पाँच हज़ार रुपये         -> ₹5,500

2. Spoken integer + post-cue + spoken paise tail::

       एक हज़ार रुपये पच्चीस पैसे     -> ₹1,000.25
       सौ रुपये पाँच पैसे             -> ₹100.05    (single-digit paise -> 2-digit)

3. Spoken decimal + post-cue (fractional rupees)::

       एक दशमलव पाँच रुपये            -> ₹1.5
       सवा दो रुपये                  -> ₹2.25       (bare half/quarter compound)
       साढ़े तीन रुपये                -> ₹3.5

4. Pre-cue (rupee word first)::

       रुपये एक हज़ार                 -> ₹1,000
       रुपये एक दशमलव पाँच            -> ₹1.5

5. Symbol prefix on Latin amount::

       ₹1000        -> ₹1,000
       ₹1,25,000    -> ₹1,25,000
       Rs. 250      -> ₹250
       Rs 1500.50   -> ₹1,500.50
       INR 12345    -> ₹12,345

The integer side is always Indian-grouped on output (1,25,000, not
125,000); CLDR records this for ``hi``/``mr``/``gu``/``ur`` etc. Paise
from an explicit ``पैसे`` tail are zero-padded to 2 digits (``5 पैसे``
becomes ``.05``); fractional rupees from a spoken decimal are emitted
literally (``साढ़े तीन रुपये`` -> ``₹3.5``) since the speaker did not
disambiguate between paise and arbitrary fractions.

Output is wrapped in NeMo-style ``money { currency: "INR" amount: "..." }``
via :data:`MONEY_CLASSIFIER`. The user-facing surface :data:`MONEY`
prepends ``₹`` to the amount. Per ``CONTRIBUTING.md`` invariants the
FST is built at module import (= grammar build time) and the canonical
storage form uses Latin digits + ICU-canonical separators.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import LATIN_DIGITS
from itn_service.grammars.hi.cardinal import CARDINAL, _NUM_0_99, _SP
from itn_service.grammars.hi.decimal import DECIMAL


# ---------------------------------------------------------------------------
# Indian thousands grouping for Latin integers up to 10 digits (1 अरब).
#
# Rule: insert a comma before position p (1 <= p < n) iff
#   p == n - 3                                  (split last 3 from rest), or
#   p <  n - 3 and (n - 3 - p) is even          (every 2 digits in the lead).
#
# n=4  -> d,ddd          n=5  -> dd,ddd
# n=6  -> d,dd,ddd       n=7  -> dd,dd,ddd
# n=8  -> d,dd,dd,ddd    n=9  -> dd,dd,dd,ddd
# n=10 -> d,dd,dd,dd,ddd                          (1 अरब = 10 digits)
# ---------------------------------------------------------------------------

_NONZERO_LATIN_DIGIT: Final[pynini.Fst] = pynini.string_map(
    [(str(d), str(d)) for d in range(1, 10)]
).optimize()


def _grouping_for_length(n: int) -> pynini.Fst:
    """FST mapping a length-``n`` Latin integer to its Indian-grouped form."""
    if n < 1:
        raise ValueError(n)
    if n == 1:
        return LATIN_DIGITS  # permit "0"
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
# Currency / paise lexical cues. All are required — the grammar refuses
# to fire on bare numbers (per "Confidence gating and fallback" /
# Currency row).
# ---------------------------------------------------------------------------

_RUPEE_WORD: Final[pynini.Fst] = pynini.union("रुपये", "रुपया", "रू")

_RUPEE_SYMBOL: Final[pynini.Fst] = pynini.union(
    "₹", "Rs.", "Rs", "rs.", "rs", "RS.", "RS", "INR", "inr",
)

_PAISE_WORD: Final[pynini.Fst] = pynini.union("पैसे", "पैसा")


# Spoken paise 1..99 -> 2-digit zero-padded ("पाँच" -> "05").
_PAISE_SPOKEN: Final[pynini.Fst] = pynini.string_map(
    [
        (w, str(n).rjust(2, "0"))
        for n in range(1, 100)
        for w in _NUM_0_99[n]
    ]
).optimize()

# Strict 2-digit Latin paise — single-digit Latin paise is rejected to
# avoid ambiguity (was "5" meant "05" or "50"?).
_PAISE_LATIN: Final[pynini.Fst] = (LATIN_DIGITS + LATIN_DIGITS).optimize()

_PAISE_INT: Final[pynini.Fst] = (_PAISE_SPOKEN | _PAISE_LATIN).optimize()


# ---------------------------------------------------------------------------
# Cardinal / decimal lifted into grouped form.
# ---------------------------------------------------------------------------

# Spoken/Latin integer -> Indian-grouped Latin integer.
_INT_GROUPED: Final[pynini.Fst] = (CARDINAL @ _GROUPED_INT).optimize()

# Spoken/Latin decimal -> grouped-int + "." + raw fractional digits.
_DOT_AND_FRAC: Final[pynini.Fst] = (
    pynini.accep(".") + LATIN_DIGITS.closure(1)
).optimize()
_DECIMAL_SPLITTER: Final[pynini.Fst] = (_GROUPED_INT + _DOT_AND_FRAC).optimize()
_DECIMAL_GROUPED: Final[pynini.Fst] = (DECIMAL @ _DECIMAL_SPLITTER).optimize()


# ---------------------------------------------------------------------------
# Already-grouped Latin integer ("₹1,25,000", "Rs. 1,25,000"): strip
# commas, then re-group canonically. Robust to malformed input
# ("1,2" -> "12") because the canonical Indian-grouped form is the
# source of truth, not the speaker's typing.
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
# Surface forms — each emits the bare amount (no ₹ prefix); MONEY adds it.
# ---------------------------------------------------------------------------

_OPT_DEL_SP: Final[pynini.Fst] = pynutil.delete(_SP).ques

# Form 1: <spoken/latin int> <space> <rupee-word>
_AMOUNT_INT_POSTCUE: Final[pynini.Fst] = (
    _INT_GROUPED
    + pynutil.delete(_SP)
    + pynutil.delete(_RUPEE_WORD)
).optimize()

# Form 2: <spoken/latin int> <space> <rupee-word> <space> <paise> <space> पैसे
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

# Form 3: <spoken/latin decimal> <space> <rupee-word>
_AMOUNT_DECIMAL_POSTCUE: Final[pynini.Fst] = (
    _DECIMAL_GROUPED
    + pynutil.delete(_SP)
    + pynutil.delete(_RUPEE_WORD)
).optimize()

# Form 4: <rupee-word> <space> <spoken/latin int>
_AMOUNT_INT_PRECUE: Final[pynini.Fst] = (
    pynutil.delete(_RUPEE_WORD)
    + pynutil.delete(_SP)
    + _INT_GROUPED
).optimize()

# Form 5: <rupee-word> <space> <spoken/latin decimal>
_AMOUNT_DECIMAL_PRECUE: Final[pynini.Fst] = (
    pynutil.delete(_RUPEE_WORD)
    + pynutil.delete(_SP)
    + _DECIMAL_GROUPED
).optimize()

# Form 6: <symbol/Rs.> [<space>] <Latin maybe-grouped int>
_AMOUNT_SYMBOL_INT: Final[pynini.Fst] = (
    pynutil.delete(_RUPEE_SYMBOL)
    + _OPT_DEL_SP
    + _REGROUPED_LATIN_INT
).optimize()

# Form 7: <symbol/Rs.> [<space>] <Latin decimal>
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
