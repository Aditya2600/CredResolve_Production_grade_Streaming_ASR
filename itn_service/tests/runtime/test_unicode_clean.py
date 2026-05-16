"""Unit tests for `runtime.unicode_clean.working_copy`."""

from __future__ import annotations

import unicodedata

import pytest

from itn_service.runtime.unicode_clean import working_copy


# --- pass-through and NFC -----------------------------------------------------


def test_empty_string_passes_through() -> None:
    assert working_copy("") == ""


def test_plain_ascii_unchanged() -> None:
    assert working_copy("Hello world") == "Hello world"


def test_devanagari_passes_through_untouched() -> None:
    text = "एक हजार रुपये"
    out = working_copy(text)
    assert out == unicodedata.normalize("NFC", text)
    # No script char dropped.
    assert all(ch in out for ch in text)


def test_already_nfc_unchanged() -> None:
    text = unicodedata.normalize("NFC", "नमस्ते")
    assert working_copy(text) == text


def test_decomposed_input_renormalised_to_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "हिन्दी")
    out = working_copy(decomposed)
    assert out == unicodedata.normalize("NFC", "हिन्दी")


# --- digit folding ------------------------------------------------------------


def test_fullwidth_digits_become_ascii() -> None:
    assert working_copy("Total ５００") == "Total 500"


def test_all_fullwidth_digit_codepoints_fold() -> None:
    fw = "".join(chr(0xFF10 + i) for i in range(10))
    assert working_copy(fw) == "0123456789"


def test_native_indic_digits_are_preserved() -> None:
    # Devanagari digits are *script* characters; storage is Latin per
    # policy.yaml but the cleaner must not touch them — the locale
    # renderer / WFST is responsible for digit shaping.
    text = "१२३४५"
    assert working_copy(text) == text


# --- minus / dash folding -----------------------------------------------------


@pytest.mark.parametrize(
    "exotic",
    ["‐", "‑", "‒", "–", "—", "―", "−",
     "﹣", "－"],
)
def test_dash_variants_fold_to_ascii_hyphen(exotic: str) -> None:
    assert working_copy(f"3{exotic}5") == "3-5"


def test_existing_ascii_hyphen_unchanged() -> None:
    assert working_copy("a-b") == "a-b"


# --- currency aliases ---------------------------------------------------------


def test_legacy_rupee_sign_folds_to_inr_sign() -> None:
    assert working_copy("₨1250") == "₹1250"


def test_fullwidth_dollar_folds() -> None:
    assert working_copy("＄10") == "$10"


def test_small_dollar_folds() -> None:
    assert working_copy("﹩20") == "$20"


def test_canonical_inr_sign_unchanged() -> None:
    assert working_copy("₹1250") == "₹1250"


# --- ZWJ / ZWNJ scrubbing -----------------------------------------------------


def test_zwj_after_halant_kept() -> None:
    # Devanagari KA + virama + ZWJ -> form retained.
    text = "क्‍त"
    assert working_copy(text) == text


def test_zwj_after_odia_virama_kept() -> None:
    text = "କ୍‍ତ"
    assert working_copy(text) == text


def test_zwnj_after_halant_kept() -> None:
    text = "क्‌त"
    assert working_copy(text) == text


def test_orphan_zwj_dropped() -> None:
    text = "नमस्ते‍"
    assert working_copy(text).rstrip().endswith("नमस्ते") or working_copy(text) == "नमस्ते"


def test_orphan_zwnj_between_words_dropped() -> None:
    text = "एक‌हजार"
    out = working_copy(text)
    assert "‌" not in out
    assert out == "एकहजार"


def test_zwj_at_string_start_dropped() -> None:
    text = "‍नमस्ते"
    out = working_copy(text)
    assert "‍" not in out
    assert out == "नमस्ते"


# --- script-character invariant ----------------------------------------------


@pytest.mark.parametrize(
    "sample",
    [
        "नमस्ते",      # Devanagari
        "নমস্কার",     # Bengali
        "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",  # Gurmukhi
        "નમસ્તે",      # Gujarati
        "வணக்கம்",     # Tamil
        "నమస్కారం",   # Telugu
        "ನಮಸ್ಕಾರ",     # Kannada
        "നമസ്കാരം",    # Malayalam
        "السلام",      # Arabic
    ],
)
def test_script_characters_preserved_byte_for_byte_when_already_nfc(
    sample: str,
) -> None:
    nfc = unicodedata.normalize("NFC", sample)
    out = working_copy(nfc)
    assert out == nfc


# --- Gurmukhi bindi-fold ------------------------------------------------------
#
# Added with bn/gu/pa support: Punjabi (Gurmukhi) text is folded to its
# non-bindi (no-nukta) base letters in the working copy. Five pairs are
# in scope per the spec:
#
#   ਫ਼ -> ਫ      ਜ਼ -> ਜ      ਗ਼ -> ਗ      ਖ਼ -> ਖ      ਸ਼ -> ਸ
#
# NFC (which runs first in `working_copy`) decomposes the precomposed
# legacy codepoints into <base, U+0A3C>, so the fold only has to drop
# the trailing nukta. Pre-composed inputs and decomposed inputs must
# both fold identically.


@pytest.mark.parametrize(
    "bindi,plain",
    [
        ("ਫ਼", "ਫ"),
        ("ਜ਼", "ਜ"),
        ("ਗ਼", "ਗ"),
        ("ਖ਼", "ਖ"),
        ("ਸ਼", "ਸ"),
    ],
)
def test_gurmukhi_bindi_letters_fold_to_non_bindi(
    bindi: str, plain: str,
) -> None:
    """Every in-scope bindi-letter folds to its non-bindi base.

    Verifies both the decomposed input form (NFC turns precomposed
    glyphs into <base, U+0A3C> first) and the codepoint identity of
    the result (single base codepoint, no nukta).
    """
    assert working_copy(bindi) == plain
    assert "਼" not in working_copy(bindi)


def test_gurmukhi_precomposed_legacy_codepoints_fold() -> None:
    """The legacy precomposed codepoints (U+0A59 / U+0A5A / U+0A5B /
    U+0A5E / U+0A36) must fold even when emitted directly by an ASR
    that does not honour NFC. The working-copy NFC pass decomposes
    them to <base, U+0A3C> first, then the bindi-fold strips the
    nukta."""
    precomposed = "".join(
        chr(cp) for cp in (0x0A5E, 0x0A5B, 0x0A5A, 0x0A59, 0x0A36)
    )
    out = working_copy(precomposed)
    assert out == "ਫਜਗਖਸ"


def test_gurmukhi_la_with_nukta_is_NOT_folded() -> None:
    """U+0A33 (LLA + NUKTA -> ਲ਼) is deliberately OUT of scope: ਲ vs
    ਲ਼ is a meaningful phonemic contrast in the dialects we serve and
    folding it would conflate two distinct lexemes. This test pins
    the explicit exclusion so a future contributor can't widen the
    fold set without seeing this rationale."""
    lla = unicodedata.normalize("NFC", "ਲ਼")  # decomposes to <0x0A32, 0x0A3C>
    out = working_copy(lla)
    assert "਼" in out  # nukta NOT stripped
    assert out == lla


def test_bindi_fold_is_noop_for_non_gurmukhi_text() -> None:
    """Other scripts must pass through untouched even when they
    happen to contain a U+0A3C-shaped sequence (they cannot, in
    practice — the fold targets pairs at specific Gurmukhi base
    codepoints — but the safety property matters)."""
    for s in (
        "नमस्ते",         # Devanagari
        "নমস্কাৰ",        # Bengali (note: Bengali has its own nukta U+09BC)
        "السلام عليكم",    # Arabic
        "hello world",     # ASCII
    ):
        assert working_copy(s) == unicodedata.normalize("NFC", s)


def test_bindi_fold_inside_a_word() -> None:
    """The fold must work in the middle of a word, not just on bare
    letters — the typical case is currency or month vocabulary that
    embeds a bindi letter (e.g. ``ਪੈਸ਼ੇ``, ``ਖ਼ਾਸ``)."""
    # ਖ਼ਾਸ ("khaas") -> ਖਾਸ after fold.
    assert working_copy("ਖ਼ਾਸ") == "ਖਾਸ"
    # Mixed word with multiple folds.
    assert working_copy("ਖ਼ਜ਼ਾਨਾ") == "ਖਜਾਨਾ"


def test_orphan_gurmukhi_nukta_outside_letter_pair_kept() -> None:
    """A U+0A3C that does NOT follow one of the five in-scope base
    codepoints is left in place. This is defensive: malformed input
    shouldn't get silently mutated beyond the targeted fold. The
    most likely case in practice is a leading nukta or one after
    another script's letter, both of which are noise the prefilter
    will handle separately."""
    # Leading nukta — no preceding base.
    assert working_copy("਼ਕ") == "਼ਕ"
    # Nukta after ਕ (U+0A15) which is NOT in the fold set.
    assert working_copy("ਕ਼") == "ਕ਼"


# --- combined case ------------------------------------------------------------


def test_combined_real_world_segment() -> None:
    raw = "Call ＋91-9876543210 — pay ₨1,250 today"
    out = working_copy(raw)
    assert "₨" not in out
    assert "₹" in out
    # Em dash folded to hyphen.
    assert "—" not in out
    # Full-width '+' (U+FF0B) is NOT in our fold list; only digits and
    # dashes / currency symbols are. This documents the contract.
    assert "9876543210" in out
