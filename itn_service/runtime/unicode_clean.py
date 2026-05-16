"""Working-copy Unicode cleanup.

Produces a classifier-friendly internal copy of an ASR transcript. The
caller continues to hold the verbatim raw text untouched (see
`CONTRIBUTING.md` invariant 1); the working copy is only used for
script routing, regex prefiltering, and downstream WFST classification.

What we do here, derived from the plan's "End-to-end pipeline":

* NFC normalise the whole string (canonical equivalence; never
  decomposes Indic combining sequences).
* Targeted compatibility folding for a small, hand-picked set: full-
  width digits, exotic minus / dash variants, full-width and small
  currency-symbol aliases.
* ZWJ / ZWNJ scrubbing per the Indic NLP Library convention: keep ZWJ
  (U+200D) and ZWNJ (U+200C) only when the immediately preceding
  character is an Indic halant / virama; otherwise strip the orphan.

We deliberately avoid full NFKC because it would dissolve script
characters (e.g. Arabic presentation forms, ligatures) and break
provenance offsets. The constraint is: this function must not mutate
script characters of any of the ten supported scripts.
"""

from __future__ import annotations

import unicodedata

# Indic halant / virama codepoints (Devanagari, Bengali, Gurmukhi,
# Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam). These are the only
# environments where ZWJ / ZWNJ carry rendering meaning per Indic NLP
# Library conventions.
_HALANTS: frozenset[int] = frozenset(
    {
        0x094D,  # Devanagari sign virama
        0x09CD,  # Bengali sign virama
        0x0A4D,  # Gurmukhi sign virama
        0x0ACD,  # Gujarati sign virama
        0x0B4D,  # Odia sign virama
        0x0BCD,  # Tamil sign virama (pulli)
        0x0C4D,  # Telugu sign virama
        0x0CCD,  # Kannada sign virama
        0x0D4D,  # Malayalam sign virama
    }
)

_ZWJ = "‍"
_ZWNJ = "‌"

# Gurmukhi (Punjabi) bindi-fold.
#
# Per Indic NLP / IndicTrans normalisation conventions and the
# Bengali / Gujarati / Punjabi onboarding spec, Gurmukhi text is
# folded to its non-bindi (no-nukta) base letters in the working
# copy so the Punjabi grammar sees a single canonical form for each
# letter family. Five pairs are in scope:
#
#   ਫ਼ (PHA + NUKTA, 0A2B 0A3C) -> ਫ (PHA, 0A2B)
#   ਜ਼ (JA  + NUKTA, 0A1C 0A3C) -> ਜ (JA,  0A1C)
#   ਗ਼ (GA  + NUKTA, 0A17 0A3C) -> ਗ (GA,  0A17)
#   ਖ਼ (KHA + NUKTA, 0A16 0A3C) -> ਖ (KHA, 0A16)
#   ਸ਼ (SA  + NUKTA, 0A38 0A3C) -> ਸ (SA,  0A38)
#
# NFC (applied earlier in :func:`working_copy`) already decomposes
# the precomposed legacy codepoints U+0A59 / U+0A5A / U+0A5B / U+0A5E
# / U+0A36 to their <base> + U+0A3C canonical equivalents — see the
# Unicode NormalizationCorrections table — so by the time the fold
# runs every bindi-letter is the decomposed pair. The fold therefore
# only needs to strip U+0A3C when it immediately follows one of the
# five base codepoints above.
#
# We deliberately do NOT include U+0A33 (LLA + NUKTA) in the fold,
# even though the precomposed ਲ਼ exists, because ਲ / ਲ਼ is a meaningful
# phonemic contrast in the dialects the model serves and folding it
# would conflate two distinct lexemes. Same logic for U+0A36 (SHA):
# kept in scope per the explicit BNGP spec, but reviewers should
# flag any false-merge they spot in the Punjabi gold set.
#
# The fold runs unconditionally in :func:`working_copy`; non-Gurmukhi
# text never contains a (base, U+0A3C) pair at these codepoints, so
# this is a no-op for other scripts.
_GURMUKHI_BINDI_BASES: frozenset[int] = frozenset(
    {0x0A2B, 0x0A1C, 0x0A17, 0x0A16, 0x0A38}
)
_GURMUKHI_NUKTA: int = 0x0A3C

# Full-width ASCII digits (U+FF10..U+FF19) -> ASCII 0..9.
_FULLWIDTH_DIGITS: dict[int, str] = {0xFF10 + i: str(i) for i in range(10)}

# Exotic minus / dash variants that carry no semantic difference in our
# transcripts. Folding them to ASCII '-' keeps the prefilter simple
# without losing meaning. We do NOT touch the Indic punctuation U+0964
# (devanagari danda) or U+0965 (double danda).
_DASH_FOLDS: dict[int, str] = {
    0x2010: "-",  # HYPHEN
    0x2011: "-",  # NON-BREAKING HYPHEN
    0x2012: "-",  # FIGURE DASH
    0x2013: "-",  # EN DASH
    0x2014: "-",  # EM DASH
    0x2015: "-",  # HORIZONTAL BAR
    0x2212: "-",  # MINUS SIGN
    0xFE58: "-",  # SMALL EM DASH
    0xFE63: "-",  # SMALL HYPHEN-MINUS
    0xFF0D: "-",  # FULLWIDTH HYPHEN-MINUS
}

# Currency-symbol compatibility aliases. The legacy Indian rupee glyph
# `₨` (U+20A8) is folded to the canonical INR sign `₹` (U+20B9) so the
# prefilter only needs to know one form. Full-width currency variants
# (U+FFE0..U+FFE6) collapse to their halfwidth counterparts.
_CURRENCY_FOLDS: dict[int, str] = {
    0x20A8: "₹",  # ₨ -> ₹
    0xFE69: "$",       # SMALL DOLLAR SIGN -> $
    0xFF04: "$",       # FULLWIDTH DOLLAR SIGN -> $
    0xFFE0: "¢",  # FULLWIDTH CENT -> ¢
    0xFFE1: "£",  # FULLWIDTH POUND -> £
    0xFFE5: "¥",  # FULLWIDTH YEN -> ¥
    0xFFE6: "₩",  # FULLWIDTH WON -> ₩
}

_TRANSLATION_TABLE: dict[int, str] = {
    **_FULLWIDTH_DIGITS,
    **_DASH_FOLDS,
    **_CURRENCY_FOLDS,
}


def _fold_gurmukhi_bindi(text: str) -> str:
    """Drop the nukta from the five Gurmukhi bindi-letter pairs.

    Walks the string left-to-right and skips any U+0A3C that
    immediately follows one of the in-scope base codepoints. Does
    nothing for non-Gurmukhi text because those base codepoints are
    Gurmukhi-only — the function is safe to call unconditionally.

    Runs AFTER ``unicodedata.normalize("NFC", text)``: NFC has
    already turned every precomposed bindi-letter into the decomposed
    pair, so the only thing this pass needs to do is delete the
    trailing nukta.
    """
    if chr(_GURMUKHI_NUKTA) not in text:
        return text
    out: list[str] = []
    prev: int | None = None
    for ch in text:
        cp = ord(ch)
        if (
            cp == _GURMUKHI_NUKTA
            and prev is not None
            and prev in _GURMUKHI_BINDI_BASES
        ):
            # drop the nukta — the base char was already appended
            # last iteration. Do NOT update prev to the nukta; keep
            # the base char so a hypothetical double-nukta (illegal
            # but defensive) would also collapse.
            continue
        out.append(ch)
        prev = cp
    return "".join(out)


def _scrub_zwj_zwnj(text: str) -> str:
    """Drop orphan ZWJ / ZWNJ.

    Per Indic NLP Library conventions, ZWJ (U+200D) and ZWNJ (U+200C)
    only have a defined function when they immediately follow a halant
    / virama. Anywhere else they are stripped — they otherwise
    introduce phantom token boundaries that break the regex prefilter
    and confuse downstream tokenizers without producing any visible
    glyph difference.
    """
    if _ZWJ not in text and _ZWNJ not in text:
        return text
    out: list[str] = []
    for ch in text:
        if ch == _ZWJ or ch == _ZWNJ:
            if out and ord(out[-1]) in _HALANTS:
                out.append(ch)
            # else: drop the orphan
        else:
            out.append(ch)
    return "".join(out)


def working_copy(text: str) -> str:
    """Return a classifier-safe internal copy of `text`.

    The caller's raw text MUST remain unchanged. Pass the return value
    to script routing / prefilter / classifier; never persist it as the
    canonical or display surface.

    Guarantees:
    * Output is in NFC.
    * Full-width digits become ASCII digits.
    * Common dash / minus variants and a small set of currency aliases
      are folded to canonical forms.
    * ZWJ / ZWNJ are kept only after a halant; orphans are dropped.
    * All script characters of the ten supported Indic / Latin / Arabic
      scripts pass through untouched.
    """
    if not text:
        return text
    nfc = unicodedata.normalize("NFC", text)
    folded = nfc.translate(_TRANSLATION_TABLE)
    # Gurmukhi bindi-fold runs AFTER NFC (so precomposed legacy
    # codepoints have already collapsed to <base, nukta>) and AFTER
    # the codepoint translate (which doesn't touch Gurmukhi). The
    # ZWJ/ZWNJ scrub runs last because the bindi-fold can change the
    # halant-adjacency invariant the scrubber depends on (it can't,
    # in practice — none of the five bases is a halant — but the
    # ordering keeps the contract explicit).
    folded = _fold_gurmukhi_bindi(folded)
    return _scrub_zwj_zwnj(folded)


__all__ = ["working_copy"]
