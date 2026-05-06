"""Unit tests for ``grammars.common.digit_maps``.

Each script's digit block is a 10-entry FST mapping the native glyph
to its ASCII counterpart. These tests assert correctness on every
glyph in every supported script and verify the multi-script union FST
collapses mixed runs.
"""

from __future__ import annotations

import pytest
import pynini

from itn_service.grammars.common.digit_maps import (
    ANY_DIGITS_TO_LATN,
    ANY_DIGIT_TO_LATN,
    ARABIC_INDIC_DIGITS,
    BENGALI_DIGITS,
    DEVANAGARI_DIGITS,
    EXTENDED_ARABIC_INDIC_DIGITS,
    GUJARATI_DIGITS,
    GURMUKHI_DIGITS,
    KANNADA_DIGITS,
    LATIN_DIGITS,
    MALAYALAM_DIGITS,
    TAMIL_DIGITS,
    TELUGU_DIGITS,
)


# --- per-script blocks -------------------------------------------------------


@pytest.mark.parametrize(
    "fst,zero_codepoint,name",
    [
        (DEVANAGARI_DIGITS, 0x0966, "Devanagari"),
        (BENGALI_DIGITS, 0x09E6, "Bengali"),
        (GURMUKHI_DIGITS, 0x0A66, "Gurmukhi"),
        (GUJARATI_DIGITS, 0x0AE6, "Gujarati"),
        (TAMIL_DIGITS, 0x0BE6, "Tamil"),
        (TELUGU_DIGITS, 0x0C66, "Telugu"),
        (KANNADA_DIGITS, 0x0CE6, "Kannada"),
        (MALAYALAM_DIGITS, 0x0D66, "Malayalam"),
        (ARABIC_INDIC_DIGITS, 0x0660, "Arabic-Indic"),
        (EXTENDED_ARABIC_INDIC_DIGITS, 0x06F0, "Extended Arabic-Indic"),
    ],
)
def test_each_script_block(
    fst: pynini.Fst, zero_codepoint: int, name: str
) -> None:
    for i in range(10):
        glyph = chr(zero_codepoint + i)
        out = (pynini.accep(glyph) @ fst).string()
        assert out == str(i), f"{name} {i}: {glyph!r} -> {out!r}"


def test_latin_passthrough() -> None:
    for i in range(10):
        out = (pynini.accep(str(i)) @ LATIN_DIGITS).string()
        assert out == str(i)


# --- union FSTs --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("०", "0"),       # Devanagari zero
        ("९", "9"),       # Devanagari nine
        ("৭", "7"),        # Bengali seven
        ("੪", "4"),        # Gurmukhi four
        ("૩", "3"),        # Gujarati three
        ("௨", "2"),        # Tamil two
        ("౫", "5"),        # Telugu five
        ("೬", "6"),        # Kannada six
        ("൦", "0"),       # Malayalam zero
        ("٩", "9"),        # Arabic-Indic nine
        ("۹", "9"),        # Extended Arabic-Indic nine
        ("0", "0"),        # Latin zero
    ],
)
def test_any_digit_single(raw: str, expected: str) -> None:
    assert (pynini.accep(raw) @ ANY_DIGIT_TO_LATN).string() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("१२५", "125"),                   # Devanagari run
        ("৭৮৯", "789"),                    # Bengali run
        ("9876543210", "9876543210"),     # Latin passthrough
        ("١٢٣", "123"),                    # Arabic-Indic
        ("۱۲۳۴۵", "12345"),                # Urdu (extended Arabic-Indic)
        # Mixed-script run still collapses cleanly.
        ("१" + "2" + "৩", "123"),
    ],
)
def test_any_digits_run(raw: str, expected: str) -> None:
    assert (pynini.accep(raw) @ ANY_DIGITS_TO_LATN).string() == expected
