"""Tests for the Hindi time grammar.

The grammar lives in ``itn_service/grammars/hi/time.py``. A cue is
**required** (बजे / बजकर / मिनट / सुबह / दोपहर / शाम / रात / AM / PM
or the canonical 24-hour numeric shape). Bare numeric spans never
normalise to time.

Coverage targets:

* >= 100 gold cases including all half / quarter forms (साढ़े, सवा,
  पौने, डेढ़, ढाई).
* Adversarial cases: cue absent, malformed times, mid-utterance
  corrections, etc.

Compile the FAR first::

    python -m itn_service.compile --lang hi
"""

from __future__ import annotations

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


# ---------------------------------------------------------------------------
# Hand-curated gold.
#
# Sections mirror the surface forms in the grammar's module docstring.
# ---------------------------------------------------------------------------

_HAND_GOLD: list[tuple[str, str]] = [
    # बजे — plain hour (12-hour, no AM/PM cue, बजे is enough).
    ("एक बजे", "1:00"),
    ("दो बजे", "2:00"),
    ("तीन बजे", "3:00"),
    ("चार बजे", "4:00"),
    ("पाँच बजे", "5:00"),
    ("पांच बजे", "5:00"),       # alternate spelling
    ("छह बजे", "6:00"),
    ("छः बजे", "6:00"),
    ("सात बजे", "7:00"),
    ("आठ बजे", "8:00"),
    ("नौ बजे", "9:00"),
    ("दस बजे", "10:00"),
    ("ग्यारह बजे", "11:00"),
    ("बारह बजे", "12:00"),
    # साढ़े N बजे — N:30 for N in 3..12.
    ("साढ़े तीन बजे", "3:30"),
    ("साढ़े चार बजे", "4:30"),
    ("साढ़े पाँच बजे", "5:30"),
    ("साढ़े छह बजे", "6:30"),
    ("साढ़े सात बजे", "7:30"),
    ("साढ़े आठ बजे", "8:30"),
    ("साढ़े नौ बजे", "9:30"),
    ("साढ़े दस बजे", "10:30"),
    ("साढ़े ग्यारह बजे", "11:30"),
    ("साढ़े बारह बजे", "12:30"),
    ("साढे पाँच बजे", "5:30"),                # without-nukta variant
    # सवा N बजे — N:15.
    ("सवा एक बजे", "1:15"),
    ("सवा दो बजे", "2:15"),
    ("सवा तीन बजे", "3:15"),
    ("सवा चार बजे", "4:15"),
    ("सवा पाँच बजे", "5:15"),
    ("सवा छह बजे", "6:15"),
    ("सवा सात बजे", "7:15"),
    ("सवा आठ बजे", "8:15"),
    ("सवा नौ बजे", "9:15"),
    ("सवा दस बजे", "10:15"),
    ("सवा ग्यारह बजे", "11:15"),
    ("सवा बारह बजे", "12:15"),
    # पौने N बजे — (N-1):45; पौने एक wraps to 12:45.
    ("पौने एक बजे", "12:45"),
    ("पौने दो बजे", "1:45"),
    ("पौने तीन बजे", "2:45"),
    ("पौने चार बजे", "3:45"),
    ("पौने पाँच बजे", "4:45"),
    ("पौने छह बजे", "5:45"),
    ("पौने सात बजे", "6:45"),
    ("पौने आठ बजे", "7:45"),
    ("पौने नौ बजे", "8:45"),
    ("पौने दस बजे", "9:45"),
    ("पौने ग्यारह बजे", "10:45"),
    ("पौने बारह बजे", "11:45"),
    # डेढ़ / ढाई — literal compounds.
    ("डेढ़ बजे", "1:30"),
    ("डेढ बजे", "1:30"),                       # without-nukta variant
    ("ढाई बजे", "2:30"),
    # बजकर / मिनट form — explicit minute count.
    ("पाँच बजकर तीस मिनट", "5:30"),
    ("एक बजकर पाँच मिनट", "1:05"),
    ("दो बजकर दस मिनट", "2:10"),
    ("तीन बजकर पंद्रह मिनट", "3:15"),
    ("चार बजकर बीस मिनट", "4:20"),
    ("छह बजकर पच्चीस मिनट", "6:25"),
    ("सात बजकर तीस मिनट", "7:30"),
    ("आठ बजकर पैंतीस मिनट", "8:35"),
    ("नौ बजकर चालीस मिनट", "9:40"),
    ("दस बजकर पैंतालीस मिनट", "10:45"),
    ("ग्यारह बजकर पचास मिनट", "11:50"),
    ("बारह बजकर पचपन मिनट", "12:55"),
    ("बारह बजकर एक मिनट", "12:01"),
    ("बारह बजकर शून्य मिनट", "12:00"),
    # AM modifier (after the time).
    ("सात बजे सुबह", "7:00 AM"),
    ("आठ बजे सुबह", "8:00 AM"),
    ("साढ़े सात बजे सुबह", "7:30 AM"),
    ("सवा छह बजे सुबह", "6:15 AM"),
    ("पौने आठ बजे सुबह", "7:45 AM"),
    # AM modifier (before the time).
    ("सुबह सात बजे", "7:00 AM"),
    ("सुबह साढ़े सात बजे", "7:30 AM"),
    ("सुबह सवा छह बजे", "6:15 AM"),
    # PM modifier — शाम / रात / दोपहर (after).
    ("पाँच बजे शाम", "5:00 PM"),
    ("साढ़े पाँच बजे शाम", "5:30 PM"),
    ("सवा छह बजे शाम", "6:15 PM"),
    ("पौने सात बजे शाम", "6:45 PM"),
    ("ग्यारह बजे रात", "11:00 PM"),
    ("बारह बजे दोपहर", "12:00 PM"),
    ("दो बजे दोपहर", "2:00 PM"),
    # PM modifier (before).
    ("शाम पाँच बजे", "5:00 PM"),
    ("शाम साढ़े पाँच बजे", "5:30 PM"),
    ("रात ग्यारह बजे", "11:00 PM"),
    ("दोपहर बारह बजे", "12:00 PM"),
    ("दोपहर दो बजे", "2:00 PM"),
    # बजकर / मिनट form with PM modifier.
    ("पाँच बजकर तीस मिनट शाम", "5:30 PM"),
    ("शाम पाँच बजकर तीस मिनट", "5:30 PM"),
    ("रात ग्यारह बजकर पैंतालीस मिनट", "11:45 PM"),
    # Hinglish 12-hour Latin with AM/PM cue.
    ("5:30 AM", "5:30 AM"),
    ("5:30 PM", "5:30 PM"),
    ("12:00 AM", "12:00 AM"),
    ("12:00 PM", "12:00 PM"),
    ("9:15 AM", "9:15 AM"),
    ("11:45 PM", "11:45 PM"),
    ("8:00 am", "8:00 AM"),
    ("8:00 pm", "8:00 PM"),
    ("8:00 a.m.", "8:00 AM"),
    ("8:00 p.m.", "8:00 PM"),
    ("08:00 AM", "8:00 AM"),                  # leading-zero hour stripped on output
    ("08:45 PM", "8:45 PM"),
    # 24-hour numeric — preserved.
    ("00:00", "00:00"),
    ("00:30", "00:30"),
    ("01:00", "01:00"),
    ("05:00", "05:00"),
    ("09:30", "09:30"),
    ("12:00", "12:00"),
    ("13:00", "13:00"),
    ("13:45", "13:45"),
    ("14:30", "14:30"),
    ("17:15", "17:15"),
    ("19:59", "19:59"),
    ("21:00", "21:00"),
    ("23:00", "23:00"),
    ("23:59", "23:59"),
]


@pytest.mark.parametrize("raw,expected", _HAND_GOLD)
def test_hand_gold(pipeline: WFSTPipeline, raw: str, expected: str) -> None:
    out = pipeline.normalize_span(raw, "time")
    assert out == expected, (raw, expected, out)


# ---------------------------------------------------------------------------
# Programmatic gold: every plain-hour form in 1..12 + every साढ़े H form
# in 3..12 + every सवा / पौने H in 1..12 — guarantees the half / quarter
# coverage requirement.
# ---------------------------------------------------------------------------

_HOUR_WORDS: list[tuple[int, str]] = [
    (1, "एक"), (2, "दो"), (3, "तीन"), (4, "चार"), (5, "पाँच"),
    (6, "छह"), (7, "सात"), (8, "आठ"), (9, "नौ"), (10, "दस"),
    (11, "ग्यारह"), (12, "बारह"),
]


def _gen_plain_hours() -> list[tuple[str, str]]:
    return [(f"{w} बजे", f"{n}:00") for n, w in _HOUR_WORDS]


def _gen_saadhe() -> list[tuple[str, str]]:
    # साढ़े only valid for N in 3..12 (1.5 = डेढ़, 2.5 = ढाई).
    return [(f"साढ़े {w} बजे", f"{n}:30") for n, w in _HOUR_WORDS if n >= 3]


def _gen_savaa() -> list[tuple[str, str]]:
    return [(f"सवा {w} बजे", f"{n}:15") for n, w in _HOUR_WORDS]


def _gen_paune() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for n, w in _HOUR_WORDS:
        prev = n - 1 if n > 1 else 12
        out.append((f"पौने {w} बजे", f"{prev}:45"))
    return out


_PROG_GOLD: list[tuple[str, str]] = (
    _gen_plain_hours() + _gen_saadhe() + _gen_savaa() + _gen_paune()
)


@pytest.mark.parametrize("raw,expected", _PROG_GOLD)
def test_programmatic_hours(
    pipeline: WFSTPipeline, raw: str, expected: str,
) -> None:
    assert pipeline.normalize_span(raw, "time") == expected, raw


# ---------------------------------------------------------------------------
# Adversarial — cue absent or malformed; must not normalise.
# ---------------------------------------------------------------------------

_ADVERSARIAL: list[str] = [
    # Bare numbers — no cue.
    "पाँच",
    "5",
    "5:30",                  # ambiguous AM/PM, no cue
    "9:15",
    "12:00",
    # Cue alone, no number.
    "बजे",
    "बजकर",
    "मिनट",
    "AM",
    "PM",
    # Half/quarter compound without बजे.
    "साढ़े पाँच",
    "सवा छह",
    "पौने सात",
    "डेढ़",
    "ढाई",
    # Time-of-day modifier alone.
    "सुबह",
    "शाम",
    "रात",
    "दोपहर",
    # Out-of-range hour (Hindi 12-hour clock).
    "तेरह बजे",              # 13 — not in lexicon
    "0 बजे",                 # 0 — not in 1..12 lexicon
    "13 बजे",                # 13 Latin — not in 1..12 lexicon
    # Out-of-range minute.
    "पाँच बजकर साठ मिनट",   # 60 — not in 0..59 lexicon
    "5 बजकर 60 मिनट",
    # Garbage / empty.
    "",
    "   ",
    "नमस्ते",
    "abc",
    "abc बजे",
    # Mid-utterance correction.
    "पाँच बजे नहीं छह बजे",
    # Cross-cue contamination.
    "पाँच बजे शाम रुपये",
    # 24-hour out of range.
    "24:00",
    "25:30",
    "12:60",
    # Single-digit hour without cue (rejected — ambiguous).
    "1:30",
    "9:00",
    # 12-hour Latin without AM/PM cue.
    "12:30",                 # in 12-hour clock this is ambiguous — and falls under 24-hour-numeric? "12:30" is HH=12 which is in 24-hour range, so should match.
]


@pytest.mark.parametrize("raw", _ADVERSARIAL)
def test_adversarial_rejected_or_24h(
    pipeline: WFSTPipeline, raw: str,
) -> None:
    """Most of these should return ``None``. The few that are valid
    24-hour-numeric shapes (e.g. ``12:30``) are accepted by the
    24-hour branch and verified separately via ``_HAND_GOLD``; the
    intent of this test is "no cue == no time" for the rest.
    """
    out = pipeline.normalize_span(raw, "time")
    if raw == "12:30":
        # 24-hour-numeric shape: HH=12 in 0..23, MM=30 in 0..59.
        assert out == "12:30"
    else:
        assert out is None, (raw, out)


# ---------------------------------------------------------------------------
# Classifier wrapper.
# ---------------------------------------------------------------------------

def test_classifier_emits_nemo_tag(pipeline: WFSTPipeline) -> None:
    out = pipeline.classify_span("साढ़े पाँच बजे शाम", "time")
    assert out == 'time { value: "5:30 PM" }'


# ---------------------------------------------------------------------------
# Coverage budget.
# ---------------------------------------------------------------------------

def test_total_case_count_meets_minimum() -> None:
    total = len(_HAND_GOLD) + len(_PROG_GOLD)
    assert total >= 100, total
    # Half / quarter forms specifically.
    half_quarter_count = (
        len(_gen_saadhe()) + len(_gen_savaa()) + len(_gen_paune())
        + sum(1 for r, _ in _HAND_GOLD if any(p in r for p in (
            "साढ़े", "सवा", "पौने", "डेढ़", "ढाई",
        )))
    )
    assert half_quarter_count >= 30, half_quarter_count
