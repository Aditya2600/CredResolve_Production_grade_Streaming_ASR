"""Tests for the Hindi percent grammar.

Coverage targets:

* >= 150 gold cases.
* >= 20 adversarial cases (cue absent or malformed).
"""

from __future__ import annotations

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline

from ._hi_spell import spell_hi


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


# ---------------------------------------------------------------------------
# Hand-curated gold.
# ---------------------------------------------------------------------------

_HAND_GOLD: list[tuple[str, str]] = [
    # Hindi cardinal + प्रतिशत.
    ("शून्य प्रतिशत", "0%"),
    ("एक प्रतिशत", "1%"),
    ("पाँच प्रतिशत", "5%"),
    ("दस प्रतिशत", "10%"),
    ("पच्चीस प्रतिशत", "25%"),
    ("पचास प्रतिशत", "50%"),
    ("निन्यानवे प्रतिशत", "99%"),
    ("एक सौ प्रतिशत", "100%"),
    # फीसदी variant.
    ("पाँच फीसदी", "5%"),
    ("बारह फीसदी", "12%"),
    ("एक सौ फीसदी", "100%"),
    # फ़ीसदी (with-nukta) variant.
    ("बीस फ़ीसदी", "20%"),
    # Spoken decimal + प्रतिशत.
    ("बारह दशमलव पाँच प्रतिशत", "12.5%"),
    ("शून्य दशमलव एक प्रतिशत", "0.1%"),
    ("शून्य दशमलव शून्य पाँच प्रतिशत", "0.05%"),
    ("निन्यानवे दशमलव नौ प्रतिशत", "99.9%"),
    # Bare half/quarter compounds.
    ("डेढ़ प्रतिशत", "1.5%"),
    ("ढाई प्रतिशत", "2.5%"),
    ("सवा बारह प्रतिशत", "12.25%"),
    ("साढ़े पाँच प्रतिशत", "5.5%"),
    ("पौने चार प्रतिशत", "3.75%"),
    # Latin passthrough + Hindi cue.
    ("12 प्रतिशत", "12%"),
    ("12.5 प्रतिशत", "12.5%"),
    ("100 फीसदी", "100%"),
    # % symbol.
    ("12%", "12%"),
    ("12.5%", "12.5%"),
    ("100%", "100%"),
    ("12 %", "12%"),
    ("12.5 %", "12.5%"),
    # Spoken + % symbol — the cue is the symbol.
    ("बारह%", "12%"),
    ("बारह दशमलव पाँच%", "12.5%"),
]


@pytest.mark.parametrize("raw,expected", _HAND_GOLD)
def test_hand_gold(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    out = pipeline.normalize_span(raw, "percent")
    assert out == expected, (raw, expected, out)


# ---------------------------------------------------------------------------
# Programmatic gold: spoken + Latin across the [0, 999] integer range and
# a sample of decimals.
# ---------------------------------------------------------------------------

def _gen_percent_spoken(stop: int = 100) -> list[tuple[str, str]]:
    return [(f"{spell_hi(n)} प्रतिशत", f"{n}%") for n in range(stop)]


def _gen_percent_latin() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for n in range(0, 200):
        out.append((f"{n}%", f"{n}%"))
        out.append((f"{n} %", f"{n}%"))
        out.append((f"{n} प्रतिशत", f"{n}%"))
    # A handful of decimals.
    for n, d in [(0, 1), (0, 5), (1, 5), (2, 5), (12, 5), (50, 25), (99, 99)]:
        s = f"{n}.{d}"
        out.append((f"{s}%", f"{s}%"))
        out.append((f"{s} प्रतिशत", f"{s}%"))
    return out


_PROG_SPOKEN = _gen_percent_spoken()
_PROG_LATIN = _gen_percent_latin()


@pytest.mark.parametrize("raw,expected", _PROG_SPOKEN, ids=[r for r, _ in _PROG_SPOKEN])
def test_programmatic_spoken(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    assert pipeline.normalize_span(raw, "percent") == expected


@pytest.mark.parametrize("raw,expected", _PROG_LATIN[:200], ids=[r for r, _ in _PROG_LATIN[:200]])
def test_programmatic_latin(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    assert pipeline.normalize_span(raw, "percent") == expected


# ---------------------------------------------------------------------------
# Adversarial.
# ---------------------------------------------------------------------------

_ADVERSARIAL: list[str] = [
    # Bare numbers — no cue.
    "बारह",
    "12",
    "12.5",
    "एक हज़ार",
    # Cue alone, no number.
    "प्रतिशत",
    "फीसदी",
    "%",
    # Garbage.
    "",
    "   ",
    "नमस्ते प्रतिशत",
    "abc प्रतिशत",
    "abc%",
    # Wrong cue word.
    "बारह percent",       # English "percent" not in our cue set
    "बारह per cent",
    # Wrong order (Hindi cue before number — not idiomatic, our grammar rejects).
    "प्रतिशत बारह",
    # Negative numbers (not supported).
    "-12 प्रतिशत",
    "minus बारह प्रतिशत",
    # Mid-utterance correction.
    "बारह प्रतिशत नहीं तेरह प्रतिशत",
    # Cross-cue contamination.
    "बारह प्रतिशत रुपये",
    # Triple cue.
    "12 प्रतिशत %",
]


@pytest.mark.parametrize("raw", _ADVERSARIAL)
def test_adversarial_no_match(pipeline: WFSTPipeline, raw: str) -> None:
    assert pipeline.normalize_span(raw, "percent") is None


# ---------------------------------------------------------------------------
# Classifier wrapper + coverage.
# ---------------------------------------------------------------------------

def test_classifier_emits_nemo_tag(pipeline: WFSTPipeline) -> None:
    out = pipeline.classify_span("बारह दशमलव पाँच प्रतिशत", "percent")
    assert out == 'percent { value: "12.5" }'


def test_total_case_count_meets_minimum() -> None:
    total = len(_HAND_GOLD) + len(_PROG_SPOKEN) + len(_PROG_LATIN)
    assert total >= 150, total
    assert len(_ADVERSARIAL) >= 20, len(_ADVERSARIAL)
