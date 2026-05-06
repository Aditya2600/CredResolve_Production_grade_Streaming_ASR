"""Shared digit-glyph FSTs for all ten supported scripts.

Each map is a single-character FST that rewrites one *native digit
glyph* to its Latin (ASCII 0-9) counterpart. They are used wherever the
working-copy text already carries native digits in it (e.g. an ASR
hypothesis like "मुझे १२५ रुपये दो") and we want a single, locale-
neutral form before the cardinal / decimal / money grammars run.

Mappings come from the Unicode code charts (the "digit zero / digit
one / ... / digit nine" rows for each script block):

* Devanagari      U+0966 - U+096F
* Bengali         U+09E6 - U+09EF
* Gurmukhi        U+0A66 - U+0A6F
* Gujarati        U+0AE6 - U+0AEF
* Tamil           U+0BE6 - U+0BEF   (modern decimal digits, not Tamil
                                     traditional numerals U+0BF0..)
* Telugu          U+0C66 - U+0C6F
* Kannada         U+0CE6 - U+0CEF
* Malayalam       U+0D66 - U+0D6F
* Arabic-Indic    U+0660 - U+0669   (Eastern Arabic; CLDR `arab`)
* Extended Arabic-Indic U+06F0 - U+06F9 (Urdu / Persian; CLDR `arabext`)

Latin 0-9 also rewrites to itself so the union FST is idempotent on
already-canonical input.

Per `CONTRIBUTING.md` invariant 3, the FSTs are constructed at module
import (= grammar build time) and are exposed as immutable singletons.
"""

from __future__ import annotations

from typing import Final

import pynini


def _digit_map(zero_codepoint: int) -> pynini.Fst:
    """Build a `string_map` for a contiguous block of ten digit glyphs."""
    return pynini.string_map(
        [(chr(zero_codepoint + i), str(i)) for i in range(10)]
    ).optimize()


# --- per-script digit maps ----------------------------------------------------

DEVANAGARI_DIGITS: Final[pynini.Fst] = _digit_map(0x0966)
BENGALI_DIGITS: Final[pynini.Fst] = _digit_map(0x09E6)
GURMUKHI_DIGITS: Final[pynini.Fst] = _digit_map(0x0A66)
GUJARATI_DIGITS: Final[pynini.Fst] = _digit_map(0x0AE6)
TAMIL_DIGITS: Final[pynini.Fst] = _digit_map(0x0BE6)
TELUGU_DIGITS: Final[pynini.Fst] = _digit_map(0x0C66)
KANNADA_DIGITS: Final[pynini.Fst] = _digit_map(0x0CE6)
MALAYALAM_DIGITS: Final[pynini.Fst] = _digit_map(0x0D66)
ARABIC_INDIC_DIGITS: Final[pynini.Fst] = _digit_map(0x0660)
EXTENDED_ARABIC_INDIC_DIGITS: Final[pynini.Fst] = _digit_map(0x06F0)
LATIN_DIGITS: Final[pynini.Fst] = pynini.string_map(
    [(str(i), str(i)) for i in range(10)]
).optimize()


# --- combined unions ----------------------------------------------------------

# Single-glyph union covering every supported script. Composing with
# this on a one-character input yields the Latin digit; on a multi-
# character input use ``ANY_DIGIT_TO_LATN.closure(1)``.
ANY_DIGIT_TO_LATN: Final[pynini.Fst] = pynini.union(
    DEVANAGARI_DIGITS,
    BENGALI_DIGITS,
    GURMUKHI_DIGITS,
    GUJARATI_DIGITS,
    TAMIL_DIGITS,
    TELUGU_DIGITS,
    KANNADA_DIGITS,
    MALAYALAM_DIGITS,
    ARABIC_INDIC_DIGITS,
    EXTENDED_ARABIC_INDIC_DIGITS,
    LATIN_DIGITS,
).optimize()

# Multi-character variant — accepts a run of one or more digits (in any
# of the supported scripts) and emits the Latin equivalent. Useful when
# composing with cardinal / decimal grammars.
ANY_DIGITS_TO_LATN: Final[pynini.Fst] = ANY_DIGIT_TO_LATN.closure(1).optimize()


__all__ = [
    "ANY_DIGITS_TO_LATN",
    "ANY_DIGIT_TO_LATN",
    "ARABIC_INDIC_DIGITS",
    "BENGALI_DIGITS",
    "DEVANAGARI_DIGITS",
    "EXTENDED_ARABIC_INDIC_DIGITS",
    "GUJARATI_DIGITS",
    "GURMUKHI_DIGITS",
    "KANNADA_DIGITS",
    "LATIN_DIGITS",
    "MALAYALAM_DIGITS",
    "TAMIL_DIGITS",
    "TELUGU_DIGITS",
]
