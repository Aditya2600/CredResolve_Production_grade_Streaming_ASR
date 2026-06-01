"""Tests for Hindi PAN and Aadhaar ID grammars."""

from __future__ import annotations

import pynini
import pytest

from itn_service.grammars.hi.id import AADHAAR, AADHAAR_CLASSIFIER, PAN, PAN_CLASSIFIER


def _compose(raw: str, fst: pynini.Fst) -> str | None:
    composed = pynini.accep(raw) @ fst
    if composed.start() == pynini.NO_STATE_ID:
        return None
    try:
        return pynini.shortestpath(composed).string()
    except pynini.FstOpError:
        return None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ABCDE1234F", "ABCDE1234F"),
        ("abcde1234f", "ABCDE1234F"),
        ("ए बी सी डी ई एक दो तीन चार एफ", "ABCDE1234F"),
        ("के एल एम एन ओ नौ आठ सात छह पी", "KLMNO9876P"),
    ],
)
def test_pan_normalises(raw: str, expected: str) -> None:
    assert _compose(raw, PAN) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("234567890124", "2345 6789 0124"),
        ("2345 6789 0124", "2345 6789 0124"),
        ("२३४५-६७८९-०१२४", "2345 6789 0124"),
        (
            "दो तीन चार पाँच छह सात आठ नौ शून्य एक दो चार",
            "2345 6789 0124",
        ),
    ],
)
def test_aadhaar_normalises(raw: str, expected: str) -> None:
    assert _compose(raw, AADHAAR) == expected


@pytest.mark.parametrize(
    ("raw", "fst"),
    [
        ("ABCD1234F", PAN),
        ("ए बी सी डी एक दो तीन चार एफ", PAN),
        ("123456789012", AADHAAR),
        ("दो तीन चार पाँच", AADHAAR),
    ],
)
def test_id_rejects_malformed_inputs(raw: str, fst: pynini.Fst) -> None:
    assert _compose(raw, fst) is None


def test_id_classifiers_emit_tags() -> None:
    assert _compose("ए बी सी डी ई एक दो तीन चार एफ", PAN_CLASSIFIER) == (
        'pan { value: "ABCDE1234F" }'
    )
    assert _compose("234567890124", AADHAAR_CLASSIFIER) == (
        'aadhaar { value: "2345 6789 0124" }'
    )
