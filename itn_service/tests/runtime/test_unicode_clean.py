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
