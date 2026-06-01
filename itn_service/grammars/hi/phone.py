"""Hindi phone grammar (Indian mobile -> canonical Latin form).

Accepts strict Indian mobile numbers only:

* 10 mobile digits with first digit in ``6..9``.
* Optional India prefix: ``+91`` / ``91``.
* Optional leading trunk ``0``.
* Digits may be Latin, Devanagari, or Hindi digit words.
* Common visual separators are accepted for written forms.

The public :data:`PHONE` surface emits ``+91 XXXXX XXXXX``. The
classifier wrapper emits ``phone { digits: "<10 digits>" }`` to match
the structured classify sketch in the implementation blueprint.

This grammar deliberately does not repair malformed numbers. Nine,
eleven, or wrong-leading-digit bodies simply produce no path; the
runtime then preserves the raw span.
"""

from __future__ import annotations

from typing import Final

import pynini
from pynini.lib import pynutil

from itn_service.grammars.common.digit_maps import DEVANAGARI_DIGITS, LATIN_DIGITS
from itn_service.grammars.hi.cardinal import _NUM_0_99, _SP


def _concat(parts: list[pynini.Fst]) -> pynini.Fst:
    """Concatenate a non-empty list of FSTs."""
    if not parts:
        raise ValueError("cannot concatenate an empty FST list")
    out = parts[0]
    for part in parts[1:]:
        out = out + part
    return out.optimize()


# ---------------------------------------------------------------------------
# Written digit forms.
# ---------------------------------------------------------------------------

_DIGIT_GLYPH: Final[pynini.Fst] = pynini.union(
    LATIN_DIGITS,
    DEVANAGARI_DIGITS,
).optimize()

_FIRST_MOBILE_GLYPH: Final[pynini.Fst] = pynini.string_map(
    [
        *[(str(n), str(n)) for n in range(6, 10)],
        *[(chr(0x0966 + n), str(n)) for n in range(6, 10)],
    ]
).optimize()

# Written phone spans often arrive as 98765 43210, 987-654-3210,
# 987.654.3210, or (987) 654-3210. Keep the separator set narrow and
# delete it only inside a digit run / after a prefix.
_VISUAL_SEP: Final[pynini.Fst] = pynini.union(" ", "-", ".", "(", ")").closure(1).optimize()
_DEL_VISUAL_SEP: Final[pynini.Fst] = pynutil.delete(_VISUAL_SEP)
_OPT_DEL_VISUAL_SEP: Final[pynini.Fst] = _DEL_VISUAL_SEP.ques
_OPT_DEL_OPEN_PAREN: Final[pynini.Fst] = pynutil.delete("(").ques
_OPT_DEL_CLOSE_PAREN: Final[pynini.Fst] = pynutil.delete(")").ques


def _written_ten_digits() -> pynini.Fst:
    parts: list[pynini.Fst] = [_OPT_DEL_OPEN_PAREN, _FIRST_MOBILE_GLYPH]
    for _ in range(9):
        parts.extend([_OPT_DEL_VISUAL_SEP, _DIGIT_GLYPH])
    parts.append(_OPT_DEL_CLOSE_PAREN)
    return _concat(parts)


_WRITTEN_TEN_DIGITS: Final[pynini.Fst] = _written_ten_digits()

_WRITTEN_PREFIX: Final[pynini.Fst] = pynutil.delete(
    pynini.union("+91", "91", "+९१", "९१", "0", "०")
).optimize()


# ---------------------------------------------------------------------------
# Spoken digit-word forms.
# ---------------------------------------------------------------------------

_EXTRA_SPOKEN_DIGITS: Final[dict[int, tuple[str, ...]]] = {
    0: ("जीरो", "ज़ीरो"),
}


def _spoken_digit_pairs(digits: range) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for n in digits:
        words = [*_NUM_0_99[n], *_EXTRA_SPOKEN_DIGITS.get(n, ())]
        pairs.extend((word, str(n)) for word in words)
    return pairs


_SPOKEN_DIGIT: Final[pynini.Fst] = pynini.string_map(
    _spoken_digit_pairs(range(10))
).optimize()
_FIRST_MOBILE_SPOKEN: Final[pynini.Fst] = pynini.string_map(
    _spoken_digit_pairs(range(6, 10))
).optimize()

_SPOKEN_ZERO: Final[pynini.Fst] = pynini.union(
    *[word for word, _ in _spoken_digit_pairs(range(1))],
).optimize()
_SPOKEN_ONE: Final[pynini.Fst] = pynini.union(
    *[word for word, _ in _spoken_digit_pairs(range(1, 2))],
).optimize()
_SPOKEN_NINE: Final[pynini.Fst] = pynini.union(
    *[word for word, _ in _spoken_digit_pairs(range(9, 10))],
).optimize()


def _spoken_ten_digits() -> pynini.Fst:
    parts: list[pynini.Fst] = [_FIRST_MOBILE_SPOKEN]
    for _ in range(9):
        parts.extend([pynutil.delete(_SP), _SPOKEN_DIGIT])
    return _concat(parts)


_SPOKEN_TEN_DIGITS: Final[pynini.Fst] = _spoken_ten_digits()

_SPOKEN_COUNTRY_PREFIX: Final[pynini.Fst] = (
    pynutil.delete(_SPOKEN_NINE)
    + pynutil.delete(_SP)
    + pynutil.delete(_SPOKEN_ONE)
).optimize()

_SPOKEN_PREFIX: Final[pynini.Fst] = pynini.union(
    _SPOKEN_COUNTRY_PREFIX,
    pynutil.delete(_SPOKEN_ZERO),
).optimize()


# ---------------------------------------------------------------------------
# Raw 10-digit output and canonical grouping.
# ---------------------------------------------------------------------------

PHONE_DIGITS: Final[pynini.Fst] = pynini.union(
    # Bare written / spoken bodies.
    _WRITTEN_TEN_DIGITS,
    _SPOKEN_TEN_DIGITS,
    # Written prefix with written or spoken body:
    #   +91 9876543210
    #   +91 नौ आठ ...
    _WRITTEN_PREFIX + _OPT_DEL_VISUAL_SEP + _WRITTEN_TEN_DIGITS,
    _WRITTEN_PREFIX + _DEL_VISUAL_SEP + _SPOKEN_TEN_DIGITS,
    # Spoken prefix with spoken or written body:
    #   नौ एक नौ आठ ...
    #   शून्य 9876543210
    _SPOKEN_PREFIX + pynutil.delete(_SP) + _SPOKEN_TEN_DIGITS,
    _SPOKEN_PREFIX + pynutil.delete(_SP) + _WRITTEN_TEN_DIGITS,
).optimize()

_GROUPED_PHONE_DIGITS: Final[pynini.Fst] = _concat(
    [
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        pynutil.insert(" "),
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
        LATIN_DIGITS,
    ]
)


# ---------------------------------------------------------------------------
# Public surfaces.
# ---------------------------------------------------------------------------

PHONE: Final[pynini.Fst] = (
    pynutil.insert("+91 ") + (PHONE_DIGITS @ _GROUPED_PHONE_DIGITS)
).optimize()

PHONE_CLASSIFIER: Final[pynini.Fst] = (
    pynutil.insert('phone { digits: "')
    + PHONE_DIGITS
    + pynutil.insert('" }')
).optimize()


__all__ = ["PHONE", "PHONE_CLASSIFIER", "PHONE_DIGITS"]
