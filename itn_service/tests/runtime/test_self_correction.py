"""Tests for ``runtime.self_correction.detect_self_corrections``."""

from __future__ import annotations

import pytest

from itn_service.runtime.contract import Span
from itn_service.runtime.self_correction import detect_self_corrections


def _span(text: str, raw: str, *, cls: str, start: int) -> Span:
    end = start + len(raw)
    assert text[start:end] == raw, (text[start:end], raw)
    return Span(
        cls=cls,
        raw=raw,
        canonical=raw,
        rule_id=f"test.{cls}",
        conf=0.99,
        start=start,
        end=end,
    )


def test_returns_empty_when_no_correction_marker() -> None:
    text = "the amount is 5000 and 6000"
    spans = [
        _span(text, "5000", cls="cardinal", start=14),
        _span(text, "6000", cls="cardinal", start=23),
    ]
    assert detect_self_corrections(text, spans) == set()


def test_detects_english_no_sorry() -> None:
    text = "9876543210 no sorry 9876543211"
    spans = [
        _span(text, "9876543210", cls="phone", start=0),
        _span(text, "9876543211", cls="phone", start=20),
    ]
    assert detect_self_corrections(text, spans) == {0, 1}


def test_detects_hindi_matlab() -> None:
    text = "ABC123 गलत मतलब ABC1234"
    spans = [
        _span(text, "ABC123", cls="id", start=0),
        _span(text, "ABC1234", cls="id", start=text.index("ABC1234")),
    ]
    assert detect_self_corrections(text, spans) == {0, 1}


def test_detects_actually_marker() -> None:
    text = "5000 actually 6000"
    spans = [
        _span(text, "5000", cls="money", start=0),
        _span(text, "6000", cls="money", start=text.index("6000")),
    ]
    assert detect_self_corrections(text, spans) == {0, 1}


def test_does_not_pair_different_classes() -> None:
    text = "9876543210 no 5000"
    spans = [
        _span(text, "9876543210", cls="phone", start=0),
        _span(text, "5000", cls="money", start=14),
    ]
    assert detect_self_corrections(text, spans) == set()


def test_window_too_long_means_no_pairing() -> None:
    text = "9876543210 sorry one two three four five six seven eight nine 9876543211"
    spans = [
        _span(text, "9876543210", cls="phone", start=0),
        _span(text, "9876543211", cls="phone", start=text.index("9876543211")),
    ]
    # Gap is much longer than default window=6 — should NOT trigger.
    assert detect_self_corrections(text, spans) == set()


def test_window_parameter_respected() -> None:
    text = "5000 a b c d sorry 6000"
    spans = [
        _span(text, "5000", cls="money", start=0),
        _span(text, "6000", cls="money", start=text.index("6000")),
    ]
    # Default window=6: the gap "a b c d sorry" is 5 tokens — pair.
    assert detect_self_corrections(text, spans) == {0, 1}
    # window=4: gap is 5 tokens → exceeds → no pair.
    assert detect_self_corrections(text, spans, window=4) == set()


def test_detects_chained_corrections() -> None:
    text = "5000 no 6000 sorry 7000"
    p2 = text.index("6000")
    p3 = text.index("7000")
    spans = [
        _span(text, "5000", cls="money", start=0),
        _span(text, "6000", cls="money", start=p2),
        _span(text, "7000", cls="money", start=p3),
    ]
    # Each adjacent pair has a marker -> all three flagged.
    assert detect_self_corrections(text, spans) == {0, 1, 2}


def test_no_marker_in_gap() -> None:
    text = "5000 and 6000"
    spans = [
        _span(text, "5000", cls="money", start=0),
        _span(text, "6000", cls="money", start=9),
    ]
    assert detect_self_corrections(text, spans) == set()


def test_spans_without_offsets_are_skipped() -> None:
    spans = [
        Span(cls="phone", raw="x", canonical="x", rule_id="r", conf=1.0),
        Span(cls="phone", raw="y", canonical="y", rule_id="r", conf=1.0),
    ]
    assert detect_self_corrections("x no y", spans) == set()


def test_overlapping_spans_skipped() -> None:
    text = "ABCDEF"
    spans = [
        _span(text, "ABCDEF", cls="id", start=0),
        _span(text, "ABCDEF", cls="id", start=0),  # overlaps fully
    ]
    assert detect_self_corrections(text, spans) == set()


def test_single_span_returns_empty() -> None:
    text = "5000"
    spans = [_span(text, "5000", cls="money", start=0)]
    assert detect_self_corrections(text, spans) == set()


def test_multi_token_phrase_marker() -> None:
    text = "5000 i mean 6000"
    spans = [
        _span(text, "5000", cls="money", start=0),
        _span(text, "6000", cls="money", start=text.index("6000")),
    ]
    assert detect_self_corrections(text, spans) == {0, 1}


@pytest.mark.parametrize(
    "marker",
    ["no", "sorry", "actually", "wait", "oops", "मतलब", "गलत", "नहीं", "क्षमा"],
)
def test_marker_vocabulary(marker: str) -> None:
    text = f"5000 {marker} 6000"
    spans = [
        _span(text, "5000", cls="money", start=0),
        _span(text, "6000", cls="money", start=text.index("6000")),
    ]
    assert detect_self_corrections(text, spans) == {0, 1}
