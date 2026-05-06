"""Tests for the Hindi decimal grammar.

Covers the three surface forms documented in
``itn_service.grammars.hi.decimal``:
  1. spoken decimal with explicit ``दशमलव`` marker,
  2. bare half / quarter compounds (fractional only),
  3. Latin passthrough.
"""

from __future__ import annotations

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


# --- spoken दशमलव form ------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("बारह दशमलव पाँच", "12.5"),
        ("बारह दशमलव पांच", "12.5"),       # alt spelling of पाँच
        ("शून्य दशमलव एक", "0.1"),
        ("शून्य दशमलव एक दो पाँच", "0.125"),
        ("एक दशमलव चार", "1.4"),
        ("एक सौ दशमलव पाँच", "100.5"),
        ("एक हज़ार दशमलव शून्य पाँच", "1000.05"),
        ("निन्यानवे दशमलव नौ नौ", "99.99"),
    ],
)
def test_spoken_decimal(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    assert pipeline.normalize_span(raw, "decimal") == expected


# --- bare half / quarter compounds ------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("डेढ़", "1.5"),
        ("डेढ", "1.5"),              # alt spelling
        ("ढाई", "2.5"),
        # साढ़े N for N >= 3
        ("साढ़े तीन", "3.5"),
        ("साढ़े पाँच", "5.5"),
        ("साढ़े दस", "10.5"),
        ("साढ़े पच्चीस", "25.5"),
        ("साढ़े निन्यानवे", "99.5"),
        ("साढे पाँच", "5.5"),       # alt prefix spelling
        # पौने N for N >= 1
        ("पौने एक", "0.75"),
        ("पौने दो", "1.75"),
        ("पौने चार", "3.75"),
        ("पौने पच्चीस", "24.75"),
        ("पौने निन्यानवे", "98.75"),
        # सवा N for N >= 1
        ("सवा एक", "1.25"),
        ("सवा बारह", "12.25"),
        ("सवा पच्चीस", "25.25"),
        ("सवा निन्यानवे", "99.25"),
    ],
)
def test_bare_compound(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    assert pipeline.normalize_span(raw, "decimal") == expected


# --- Latin passthrough ------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["12.5", "0.125", "100.5", "0.0", "999.999"],
)
def test_latin_passthrough(pipeline: WFSTPipeline, raw: str) -> None:
    assert pipeline.normalize_span(raw, "decimal") == raw


# --- rejection (must NOT decimal-parse) -------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        # साढ़े N for N < 3 is grammatical-but-unidiomatic; the spec
        # explicitly restricts to N >= 3 (1.5 = डेढ़, 2.5 = ढाई).
        "साढ़े एक",
        "साढ़े दो",
        # Bare cardinal — not a decimal even though parseable as cardinal.
        "एक सौ पच्चीस",
        # Garbage.
        "नमस्ते",
        # Empty.
        "",
    ],
)
def test_decimal_rejects(pipeline: WFSTPipeline, raw: str) -> None:
    assert pipeline.normalize_span(raw, "decimal") is None


# --- classifier wrapper -----------------------------------------------------


def test_decimal_classifier_emits_nemo_tag(pipeline: WFSTPipeline) -> None:
    out = pipeline.classify_span("बारह दशमलव पाँच", "decimal")
    assert out == 'decimal { value: "12.5" }'


def test_cardinal_classifier_emits_nemo_tag(pipeline: WFSTPipeline) -> None:
    out = pipeline.classify_span("एक सौ पच्चीस", "cardinal")
    assert out == 'cardinal { value: "125" }'
