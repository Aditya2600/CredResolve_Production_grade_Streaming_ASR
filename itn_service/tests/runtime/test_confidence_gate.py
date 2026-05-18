"""Tests for ``runtime.confidence_gate.gate``."""

from __future__ import annotations

import pytest

from itn_service.runtime.confidence_gate import gate, load_thresholds
from itn_service.runtime.contract import Span


@pytest.fixture(scope="module")
def thresholds():
    return load_thresholds()


def _span(cls: str, conf: float = 0.99, *, ambiguous: bool = False, raw: str = "x", canon: str = "X") -> Span:
    return Span(
        cls=cls, raw=raw, canonical=canon, rule_id=f"r.{cls}",
        conf=conf, ambiguous=ambiguous,
    )


# ---------------------------------------------------------------------------
# Acceptance.
# ---------------------------------------------------------------------------

def test_high_confidence_with_cue_accepted(thresholds) -> None:
    span = _span("currency", conf=0.99, raw="एक हज़ार रुपये", canon="₹1,000")
    out = gate(span, asr_conf=0.95, has_lex_cue=True, is_partial=False, thresholds=thresholds)
    assert out is span
    assert out.canonical == "₹1,000"
    assert out.fallback_reason is None


def test_cardinal_no_cue_required(thresholds) -> None:
    span = _span("cardinal", conf=0.95, raw="एक सौ", canon="100")
    out = gate(span, asr_conf=0.85, has_lex_cue=False, is_partial=False, thresholds=thresholds)
    assert out is span


# ---------------------------------------------------------------------------
# Rejection paths — fallback_reason populated.
# ---------------------------------------------------------------------------

def test_low_classifier_conf_rejected(thresholds) -> None:
    span = _span("currency", conf=0.50, raw="X", canon="₹1")
    out = gate(span, asr_conf=0.95, has_lex_cue=True, is_partial=False, thresholds=thresholds)
    assert out.canonical == "X"
    assert out.fallback_reason is not None
    assert "classifier_conf" in out.fallback_reason


def test_low_asr_conf_rejected(thresholds) -> None:
    span = _span("currency", conf=0.99, raw="X", canon="₹1")
    out = gate(span, asr_conf=0.30, has_lex_cue=True, is_partial=False, thresholds=thresholds)
    assert out.canonical == "X"
    assert out.fallback_reason is not None
    assert "asr_conf" in out.fallback_reason


def test_missing_lex_cue_rejected(thresholds) -> None:
    span = _span("currency", conf=0.99, raw="1000", canon="₹1,000")
    out = gate(span, asr_conf=0.95, has_lex_cue=False, is_partial=False, thresholds=thresholds)
    assert out.canonical == "1000"
    assert "missing_lex_cue" in (out.fallback_reason or "")


def test_money_uses_the_same_gate_as_currency(thresholds) -> None:
    span = _span("money", conf=0.99, raw="1000", canon="₹1,000")
    out = gate(span, asr_conf=0.95, has_lex_cue=False, is_partial=False, thresholds=thresholds)
    assert out.canonical == "1000"
    assert "missing_lex_cue" in (out.fallback_reason or "")


def test_partial_defer_for_phone(thresholds) -> None:
    span = _span("phone", conf=0.99, raw="9876543210", canon="+91 98765 43210")
    out = gate(span, asr_conf=0.95, has_lex_cue=True, is_partial=True, thresholds=thresholds)
    assert out.canonical == "9876543210"
    assert "defer_on_partial" in (out.fallback_reason or "")


def test_ambiguous_always_falls_back(thresholds) -> None:
    span = _span("cardinal", conf=0.99, ambiguous=True, raw="X", canon="Y")
    out = gate(span, asr_conf=0.99, has_lex_cue=True, is_partial=False, thresholds=thresholds)
    assert out.canonical == "X"
    assert "ambiguous" in (out.fallback_reason or "")


def test_unknown_class_passes_through_when_unambiguous(thresholds) -> None:
    span = _span("nonexistent_class", conf=0.99)
    out = gate(span, asr_conf=0.99, has_lex_cue=True, is_partial=False, thresholds=thresholds)
    assert out is span


def test_unknown_class_falls_back_when_ambiguous(thresholds) -> None:
    span = _span("nonexistent_class", conf=0.99, ambiguous=True, raw="X", canon="Y")
    out = gate(span, asr_conf=0.99, has_lex_cue=True, is_partial=False, thresholds=thresholds)
    assert out.canonical == "X"


def test_multiple_failure_reasons_joined(thresholds) -> None:
    span = _span("currency", conf=0.10, raw="X", canon="Y")
    out = gate(span, asr_conf=0.10, has_lex_cue=False, is_partial=False, thresholds=thresholds)
    assert out.fallback_reason is not None
    parts = out.fallback_reason.split(";")
    assert len(parts) >= 3, parts
    assert any("classifier_conf" in p for p in parts)
    assert any("asr_conf" in p for p in parts)
    assert any("missing_lex_cue" in p for p in parts)


def test_threshold_table_loads_all_classes(thresholds) -> None:
    # Sanity: the YAML has the classes the implementation blueprint
    # mandates ("Confidence gating and fallback" table).
    expected = {"cardinal", "decimal", "percent", "currency", "money", "time",
                "date", "phone", "id", "health_dose"}
    assert expected <= set(thresholds.classes.keys())
    assert thresholds.partial_stable_min >= 1
    assert "cardinal" in thresholds.partial_safe_classes
