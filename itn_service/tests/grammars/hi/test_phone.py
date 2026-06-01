"""Tests for the Hindi phone grammar."""

from __future__ import annotations

import pynini
import pytest

from itn_service.grammars.hi.phone import PHONE, PHONE_CLASSIFIER, PHONE_DIGITS


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
        ("9876543210", "+91 98765 43210"),
        ("+919876543210", "+91 98765 43210"),
        ("+91 9876543210", "+91 98765 43210"),
        ("+91 98765 43210", "+91 98765 43210"),
        ("91-98765-43210", "+91 98765 43210"),
        ("09876543210", "+91 98765 43210"),
        ("९८७६५४३२१०", "+91 98765 43210"),
        ("+९१ ९८७६५४३२१०", "+91 98765 43210"),
        ("987.654.3210", "+91 98765 43210"),
        ("(987) 654-3210", "+91 98765 43210"),
        ("नौ आठ सात छह पाँच चार तीन दो एक शून्य", "+91 98765 43210"),
        ("नौ आठ सात छः पांच चार तीन दो एक जीरो", "+91 98765 43210"),
        ("+91 नौ आठ सात छह पाँच चार तीन दो एक शून्य", "+91 98765 43210"),
        ("नौ एक नौ आठ सात छह पाँच चार तीन दो एक शून्य", "+91 98765 43210"),
        ("शून्य नौ आठ सात छह पाँच चार तीन दो एक शून्य", "+91 98765 43210"),
    ],
)
def test_phone_normalises(raw: str, expected: str) -> None:
    assert _compose(raw, PHONE) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "987654321",
        "98765432101",
        "5876543210",
        "+44 9876543210",
        "0091 9876543210",
        "नौ आठ सात नमस्ते छह पाँच चार तीन दो एक शून्य",
    ],
)
def test_phone_rejects_malformed_inputs(raw: str) -> None:
    assert _compose(raw, PHONE) is None


def test_phone_digits_surface_is_ungrouped() -> None:
    assert _compose("९८७६५४३२१०", PHONE_DIGITS) == "9876543210"


def test_phone_classifier_emits_structured_digits() -> None:
    assert (
        _compose("नौ आठ सात छह पाँच चार तीन दो एक शून्य", PHONE_CLASSIFIER)
        == 'phone { digits: "9876543210" }'
    )
