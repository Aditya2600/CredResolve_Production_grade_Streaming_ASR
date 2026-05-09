"""Tests for ``runtime.formatters.phone_in.parse_indian_mobile``.

Coverage targets from the stage-2 deliverable:

* 150+ gold cases (positive: input -> canonical "+91 XXXXX XXXXX").
* 20 adversarial cases (each must yield ``None``: bad length, leading
  digit out of [6-9], lookalike OCR/ASR errors, mid-utterance
  corrections, etc.).
* Critical-entity precision >= 99.5 % — i.e. the formatter must never
  emit a non-``None`` answer for an invalid input.
"""

from __future__ import annotations

import random

import pytest

from itn_service.runtime.formatters.phone_in import parse_indian_mobile


# Hand-curated spoken / Latin gold cases.
_HAND_GOLD: list[tuple[str, str]] = [
    # Bare 10-digit Latin.
    ("9876543210", "+91 98765 43210"),
    ("6000000000", "+91 60000 00000"),
    ("7012345678", "+91 70123 45678"),
    ("8888888888", "+91 88888 88888"),
    ("9999999999", "+91 99999 99999"),
    # +91 prefix variants.
    ("+919876543210", "+91 98765 43210"),
    ("+91 9876543210", "+91 98765 43210"),
    ("+91-9876543210", "+91 98765 43210"),
    ("+91 98765 43210", "+91 98765 43210"),
    ("+91 9876 543 210", "+91 98765 43210"),
    # 91 prefix without +.
    ("919876543210", "+91 98765 43210"),
    ("91 9876543210", "+91 98765 43210"),
    ("91-98765-43210", "+91 98765 43210"),
    # Leading-0 prefix.
    ("09876543210", "+91 98765 43210"),
    ("0 98765 43210", "+91 98765 43210"),
    ("0-9876543210", "+91 98765 43210"),
    # Devanagari digit glyphs.
    ("९८७६५४३२१०", "+91 98765 43210"),
    ("+९१ ९८७६५४३२१०", "+91 98765 43210"),
    # Spoken Hindi digit-by-digit.
    ("नौ आठ सात छह पाँच चार तीन दो एक शून्य", "+91 98765 43210"),
    ("नौ आठ सात छः पाँच चार तीन दो एक शून्य", "+91 98765 43210"),  # छः variant
    ("नौ आठ सात छह पांच चार तीन दो एक जीरो", "+91 98765 43210"),  # पांच + जीरो
    ("छह सात आठ नौ शून्य एक दो तीन चार पाँच", "+91 67890 12345"),
    ("सात आठ नौ छह पाँच चार तीन दो एक शून्य", "+91 78965 43210"),
    # Mixed Hindi spoken + Latin.
    ("+91 नौ आठ सात छह पाँच चार तीन दो एक शून्य", "+91 98765 43210"),
    # Extra inner spacing (single-digit chunks).
    ("9 8 7 6 5 4 3 2 1 0", "+91 98765 43210"),
    ("(987) 654-3210", "+91 98765 43210"),
    ("987.654.3210", "+91 98765 43210"),
]

# Adversarial cases — every entry MUST yield None (the formatter must
# never "fix" a malformed phone number).
_ADVERSARIAL: list[str] = [
    # Wrong length.
    "987654321",        # 9 digits
    "98765432101",      # 11 digits (no recognised prefix)
    "987654",           # 6 digits
    "",                 # empty
    "   ",              # whitespace only
    # Leading digit out of [6-9].
    "1234567890",
    "5876543210",
    "0876543210",       # leading 0 + 9 more (9 digits after 0)
    # Lookalike characters that aren't real digits.
    "9876S43210",       # S instead of 5 (OCR-style)
    "98765O3210",       # O instead of 0
    "9876543L10",       # L instead of 1
    # Wrong country code.
    "+449876543210",
    "+11234567890",
    # Bad prefix combos.
    "+91 5876543210",   # +91 + leading 5
    "0091 9876543210",  # double-prefix
    # Stray garbage.
    "9876543210 abc",
    "abc 9876543210",
    "9876543210x",
    # Mid-utterance correction (shouldn't be normalised here — separate stage).
    "9876543210 no sorry 9876543211",
    # Non-digit Hindi tokens mixed in.
    "नौ आठ सात नमस्ते छह पाँच चार तीन दो एक शून्य",
]


@pytest.mark.parametrize("raw,expected", _HAND_GOLD)
def test_hand_gold(raw: str, expected: str) -> None:
    assert parse_indian_mobile(raw) == expected


@pytest.mark.parametrize("raw", _ADVERSARIAL)
def test_adversarial_returns_none(raw: str) -> None:
    assert parse_indian_mobile(raw) is None


# ---------------------------------------------------------------------------
# Programmatic gold: 150 random valid 10-digit numbers, generated with a
# fixed seed so the test is reproducible.
# ---------------------------------------------------------------------------

def _gen_valid_numbers(n: int = 150, seed: int = 0xA9D)->list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        first = rng.choice("6789")
        rest = "".join(rng.choice("0123456789") for _ in range(9))
        out.append(first + rest)
    return out


_PROG_NUMS = _gen_valid_numbers()


@pytest.mark.parametrize("num", _PROG_NUMS, ids=lambda n: n)
def test_programmatic_bare(num: str) -> None:
    assert parse_indian_mobile(num) == f"+91 {num[:5]} {num[5:]}"


@pytest.mark.parametrize("num", _PROG_NUMS[:50], ids=lambda n: f"+91-{n}")
def test_programmatic_plus91(num: str) -> None:
    assert parse_indian_mobile(f"+91 {num}") == f"+91 {num[:5]} {num[5:]}"


@pytest.mark.parametrize("num", _PROG_NUMS[:50], ids=lambda n: f"0-{n}")
def test_programmatic_leading_zero(num: str) -> None:
    assert parse_indian_mobile(f"0{num}") == f"+91 {num[:5]} {num[5:]}"


# ---------------------------------------------------------------------------
# Critical-entity precision check: feed positive + adversarial; precision
# must be >= 99.5 %. By construction this is 100 %, but we assert the
# bar so any future regression (e.g. accidentally accepting a 9-digit
# input) trips the build.
# ---------------------------------------------------------------------------

def test_critical_entity_precision() -> None:
    cases: list[tuple[str, str | None]] = [
        *((raw, exp) for raw, exp in _HAND_GOLD),
        *((num, f"+91 {num[:5]} {num[5:]}") for num in _PROG_NUMS),
        *((raw, None) for raw in _ADVERSARIAL),
    ]
    tp = 0
    fp = 0
    fn = 0
    for raw, expected in cases:
        got = parse_indian_mobile(raw)
        if expected is None:
            if got is not None:
                fp += 1
        else:
            if got is None:
                fn += 1
            elif got == expected:
                tp += 1
            else:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    assert precision >= 0.995, f"precision {precision:.4f} below 99.5 % bar"
    assert fp == 0, f"phone formatter must never produce false positives ({fp} found)"


def test_total_case_count_meets_minimum() -> None:
    # 150+ gold positives, 20+ adversarial — per stage-2 spec.
    total_positive = len(_HAND_GOLD) + len(_PROG_NUMS)
    assert total_positive >= 150, total_positive
    assert len(_ADVERSARIAL) >= 20, len(_ADVERSARIAL)
