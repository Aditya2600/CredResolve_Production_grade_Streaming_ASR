"""Tests for ``runtime.formatters.id_pan_aadhaar_ifsc``.

Coverage targets from the stage-2 deliverable, applied per-class:

* 150+ gold (positive) cases.
* 20 adversarial cases — must all yield ``None``.
* Critical-entity precision >= 99.5 %; the formatter must never produce
  a non-``None`` rewrite for an invalid input.
"""

from __future__ import annotations

import random
import string

import pytest

from itn_service.runtime.formatters.id_pan_aadhaar_ifsc import (
    parse_aadhaar,
    parse_generic_id,
    parse_ifsc,
    parse_pan,
    verhoeff_compute,
    verhoeff_validate,
)


# ===========================================================================
# Verhoeff self-consistency.
# ===========================================================================

def test_verhoeff_compute_then_validate_round_trips() -> None:
    rng = random.Random(0xAA01)
    for _ in range(200):
        prefix = "".join(rng.choice("0123456789") for _ in range(11))
        check = verhoeff_compute(prefix)
        assert verhoeff_validate(prefix + check)


def test_verhoeff_rejects_off_by_one() -> None:
    rng = random.Random(0xAA02)
    for _ in range(100):
        prefix = "".join(rng.choice("0123456789") for _ in range(11))
        valid = prefix + verhoeff_compute(prefix)
        # Flip the last digit by 1 (mod 10).
        bad_last = str((int(valid[-1]) + 1) % 10)
        assert not verhoeff_validate(valid[:-1] + bad_last)


# ===========================================================================
# PAN.
# ===========================================================================

_PAN_HAND_GOLD: list[tuple[str, str]] = [
    ("ABCDE1234F", "ABCDE1234F"),
    ("AAAPL1234C", "AAAPL1234C"),
    ("ZZZZZ9999Z", "ZZZZZ9999Z"),
    ("abcde1234f", "ABCDE1234F"),         # uppercased
    ("  ABCDE1234F  ", "ABCDE1234F"),     # trim
]

_PAN_ADVERSARIAL: list[str] = [
    "",
    "ABCDE1234",            # 9 chars
    "ABCDE12345",           # last char digit not letter
    "ABCD1E234F",           # digits in letter slot
    "ABCDE1234FG",          # 11 chars
    "ABCD-1234F",           # has separator
    "ABCDE 1234 F",         # spaces
    "1BCDE1234F",           # leading digit
    "ABCDE12.4F",           # punctuation
    "ABCDEABCDE",           # all letters
    "1234567890",           # all digits
    "ABCDE1234फ",           # non-ASCII
    "ABCDEI234F",           # I instead of 1 (lookalike)
    "ABCDE1Z34F",           # Z instead of digit
    "ABCDE1234",            # missing last letter
    "ABCDE1234FA",          # extra char
    "AB CDE1234F",          # internal space
    "ABCDE 1234F",
    "ABCDE-1234F",
    "ABCDE.1234F",          # dot
    "ABCDE1234F!",          # trailing punctuation
]


@pytest.mark.parametrize("raw,expected", _PAN_HAND_GOLD)
def test_pan_hand_gold(raw: str, expected: str) -> None:
    assert parse_pan(raw) == expected


# Programmatic PAN gold: 150 randomly generated valid PAN strings.
def _gen_pan_valid(n: int = 150, seed: int = 0xB1) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        s = (
            "".join(rng.choice(string.ascii_uppercase) for _ in range(5))
            + "".join(rng.choice("0123456789") for _ in range(4))
            + rng.choice(string.ascii_uppercase)
        )
        out.append(s)
    return out


_PAN_PROG = _gen_pan_valid()


@pytest.mark.parametrize("raw", _PAN_PROG, ids=lambda s: s)
def test_pan_programmatic(raw: str) -> None:
    assert parse_pan(raw) == raw


@pytest.mark.parametrize("raw", _PAN_ADVERSARIAL)
def test_pan_adversarial(raw: str) -> None:
    # Some adversarial strings (mixed-case ASCII like "ABcDE1234F" and
    # whitespace-padded valid PANs) ARE accepted because the formatter
    # uppercases + strips — keep those out of the strictly-rejected set.
    assert parse_pan(raw) is None


def test_pan_critical_precision() -> None:
    pos = [(r, e) for r, e in _PAN_HAND_GOLD] + [(r, r) for r in _PAN_PROG]
    fp = sum(1 for r in _PAN_ADVERSARIAL if parse_pan(r) is not None)
    tp = sum(1 for r, e in pos if parse_pan(r) == e)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    assert precision >= 0.995
    assert fp == 0


# ===========================================================================
# Aadhaar.
# ===========================================================================

# Valid Aadhaars: any 11-digit prefix + computed Verhoeff check digit.
def _aadhaar_for(prefix11: str) -> str:
    return prefix11 + verhoeff_compute(prefix11)


# Hand-crafted gold (Verhoeff-valid).
_AADHAAR_HAND_PREFIXES: list[str] = [
    "23456789012",  # leading 2
    "34567890123",
    "98765432101",
    "78901234567",
    "23498712345",
    "55555555555",
    "67890123456",
    "12345678901",  # leading 1 (not UIDAI-spec compliant but valid Verhoeff;
                    # our formatter does NOT enforce leading [2-9] beyond regex.
                    # If you want UIDAI-strict, tighten _AADHAAR_DIGITS_RE.)
]

_AADHAAR_HAND_GOLD: list[tuple[str, str]] = []
for _p in _AADHAAR_HAND_PREFIXES:
    full = _aadhaar_for(_p)
    canon = f"{full[:4]} {full[4:8]} {full[8:]}"
    _AADHAAR_HAND_GOLD.extend([
        (full, canon),
        (canon, canon),
        (f"{full[:4]}-{full[4:8]}-{full[8:]}", canon),
        (f"  {full}  ", canon),
    ])


_AADHAAR_ADVERSARIAL: list[str] = [
    "",
    "234567890",                 # 9 digits
    "23456789012345",            # 14 digits
    "234567890121",              # bad checksum (ends in 1 not the computed)
    "234567890123",              # bad checksum (vs hand-computed valid above)
    "abcdabcdabcd",              # letters
    "1234 5678 90",              # 10 digits
    "2345 6789 0123 4",          # 13 digits
    "1234567890ab",              # mixed
    "+91 9876543210",            # phone shape
    "234567890O12",              # O lookalike
    "234567890I12",              # I lookalike
    "23456 78901 2",             # wrong grouping length
    "0000000000000",             # 13 digits all zero
    "AAAAAAAAAAAA",              # 12 letters
    "234.567.890.123",           # dots
    "234/5678/90123",            # slashes
    "234,567,890,123",           # commas
    "२३४५६७८९०१२३",                # native digits — formatter only accepts Latin
    "234567890123\n",            # control char
]


@pytest.mark.parametrize("raw,expected", _AADHAAR_HAND_GOLD)
def test_aadhaar_hand_gold(raw: str, expected: str) -> None:
    assert parse_aadhaar(raw) == expected


def _gen_aadhaar_valid(n: int = 150, seed: int = 0xC1) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    while len(out) < n:
        prefix = "".join(rng.choice("23456789") for _ in range(1)) + "".join(
            rng.choice("0123456789") for _ in range(10)
        )
        full = _aadhaar_for(prefix)
        canon = f"{full[:4]} {full[4:8]} {full[8:]}"
        out.append((full, canon))
    return out


_AADHAAR_PROG = _gen_aadhaar_valid()


@pytest.mark.parametrize("raw,expected", _AADHAAR_PROG, ids=[r for r, _ in _AADHAAR_PROG])
def test_aadhaar_programmatic(raw: str, expected: str) -> None:
    assert parse_aadhaar(raw) == expected


@pytest.mark.parametrize("raw", _AADHAAR_ADVERSARIAL)
def test_aadhaar_adversarial(raw: str) -> None:
    assert parse_aadhaar(raw) is None


def test_aadhaar_off_by_one_digit_rejected() -> None:
    """Flip any single digit in a valid Aadhaar -> Verhoeff invalid."""
    rng = random.Random(0xC2)
    for _ in range(20):
        full, _canon = rng.choice(_AADHAAR_PROG)
        i = rng.randrange(12)
        original = full[i]
        replacement = str((int(original) + 1) % 10)
        bad = full[:i] + replacement + full[i + 1 :]
        assert parse_aadhaar(bad) is None, bad


def test_aadhaar_checksum_optional() -> None:
    # When validate_checksum=False, any 12-digit string is accepted.
    assert parse_aadhaar("234567890121", validate_checksum=False) == "2345 6789 0121"


# ===========================================================================
# IFSC.
# ===========================================================================

_IFSC_HAND_GOLD: list[tuple[str, str]] = [
    ("HDFC0001234", "HDFC0001234"),
    ("SBIN0000456", "SBIN0000456"),
    ("ICIC0ABCDEF", "ICIC0ABCDEF"),
    ("PUNB0123ABC", "PUNB0123ABC"),
    ("hdfc0001234", "HDFC0001234"),
    ("  KKBK0000123  ", "KKBK0000123"),
]

_IFSC_ADVERSARIAL: list[str] = [
    "",
    "HDFC1001234",       # 5th char must be '0'
    "HDF0001234",        # only 3 leading letters
    "HDFCO001234",       # O instead of 0
    "HDFC0001",          # too short
    "HDFC0001234567",    # too long
    "HDFC-0001234",      # separator
    "HDFC 0001234",      # space
    "1DFC0001234",       # digit in letter slot
    "HDFC0!@#$%^",       # punctuation in suffix
    "HDFC0बैंक12",        # non-ASCII
    "ABCD0",             # only 5 chars
    "ABCD01234567",      # 5th char is 0 but suffix too long? — 12 chars, fail length
    "12345678901",       # all digits
    "HDFCDOOOOOO",       # 5th not '0'
    "HD FC0001234",      # space inside
    "HD-FC0001234",
    "HDFC.0001234",
    "HDFC/0001234",
    "HDFC0\t001234",
]


@pytest.mark.parametrize("raw,expected", _IFSC_HAND_GOLD)
def test_ifsc_hand_gold(raw: str, expected: str) -> None:
    assert parse_ifsc(raw) == expected


def _gen_ifsc_valid(n: int = 150, seed: int = 0xD1) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        prefix = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
        suffix = "".join(rng.choice(string.ascii_uppercase + "0123456789") for _ in range(6))
        out.append(f"{prefix}0{suffix}")
    return out


_IFSC_PROG = _gen_ifsc_valid()


@pytest.mark.parametrize("raw", _IFSC_PROG, ids=lambda s: s)
def test_ifsc_programmatic(raw: str) -> None:
    assert parse_ifsc(raw) == raw


@pytest.mark.parametrize("raw", _IFSC_ADVERSARIAL)
def test_ifsc_adversarial(raw: str) -> None:
    assert parse_ifsc(raw) is None


# ===========================================================================
# Generic alphanumeric ID — cue-gated.
# ===========================================================================

def test_generic_id_with_english_cue() -> None:
    assert parse_generic_id("ABC1234", ["my", "account", "number", "is"]) == "ABC1234"


def test_generic_id_with_hindi_cue() -> None:
    assert parse_generic_id("XYZ987", ["मेरा", "खाता", "नंबर", "है"]) == "XYZ987"


def test_generic_id_no_cue_returns_none() -> None:
    assert parse_generic_id("ABC1234", ["something", "totally", "unrelated"]) is None


def test_generic_id_cue_outside_window_returns_none() -> None:
    # 5-token window: cue at position -6 must be ignored.
    ctx = ["account", "is", "as", "follows", "and", "begins"]
    assert parse_generic_id("ABC1234", ctx) is None


def test_generic_id_strips_internal_separators() -> None:
    ctx = ["policy", "no"]
    assert parse_generic_id("ABC-12-34", ctx) == "ABC1234"
    assert parse_generic_id("ABC/12/34", ctx) == "ABC1234"
    assert parse_generic_id("ABC.12.34", ctx) == "ABC1234"


def test_generic_id_uppercases() -> None:
    assert parse_generic_id("abc1234", ["customer", "id"]) == "ABC1234"


def test_generic_id_rejects_too_short() -> None:
    assert parse_generic_id("AB1", ["account"]) is None


def test_generic_id_rejects_too_long() -> None:
    assert parse_generic_id("A" * 21, ["account"]) is None


def test_generic_id_rejects_non_alphanumeric_after_strip() -> None:
    assert parse_generic_id("ABC@1234", ["account"]) is None


@pytest.mark.parametrize(
    "cue", ["account", "acct", "policy", "loan", "customer", "reference",
             "ref", "खाता", "अकाउंट", "पॉलिसी", "लोन", "ग्राहक", "रेफरेंस"]
)
def test_generic_id_cue_vocabulary(cue: str) -> None:
    assert parse_generic_id("ABC123", [cue]) == "ABC123"


# ===========================================================================
# Aggregate count check (per-class >= 150 gold + >= 20 adversarial).
# ===========================================================================

def test_per_class_minimum_counts() -> None:
    pan_pos = len(_PAN_HAND_GOLD) + len(_PAN_PROG)
    aadhaar_pos = len(_AADHAAR_HAND_GOLD) + len(_AADHAAR_PROG)
    ifsc_pos = len(_IFSC_HAND_GOLD) + len(_IFSC_PROG)
    assert pan_pos >= 150, pan_pos
    assert aadhaar_pos >= 150, aadhaar_pos
    assert ifsc_pos >= 150, ifsc_pos
    assert len(_PAN_ADVERSARIAL) >= 20
    assert len(_AADHAAR_ADVERSARIAL) >= 20
    assert len(_IFSC_ADVERSARIAL) >= 20


# ---------------------------------------------------------------------------
# Critical-entity precision: identifiers must never produce false-positive
# canonicalisations.
# ---------------------------------------------------------------------------

def test_id_critical_precision_pan_aadhaar_ifsc() -> None:
    fps = 0
    fps += sum(1 for r in _PAN_ADVERSARIAL if parse_pan(r) is not None)
    fps += sum(1 for r in _AADHAAR_ADVERSARIAL if parse_aadhaar(r) is not None)
    fps += sum(1 for r in _IFSC_ADVERSARIAL if parse_ifsc(r) is not None)
    assert fps == 0
