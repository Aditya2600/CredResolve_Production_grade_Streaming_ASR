"""Tests for Stage-B normalizer safety wiring."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from itn_service.runtime.contract import Span
from itn_service.runtime.locale_policy import TenantPolicy
from itn_service.runtime.normalizer import default_classifier, normalize_segment
from itn_service.runtime.regex_prefilter import prefilter
from itn_service.runtime.stream_state import StreamState
from itn_service.runtime.wfst_classifier import make_wfst_classifier


_DMY_POLICY = TenantPolicy(
    tenant_id="default",
    region="IN",
    date_order="DMY",
    currency="INR",
)


def _span(
    text: str,
    raw: str,
    *,
    cls: str,
    canonical: str,
    rule_id: str,
    start_at: int = 0,
) -> Span:
    start = text.index(raw, start_at)
    return Span(
        cls=cls,
        raw=raw,
        canonical=canonical,
        rule_id=rule_id,
        conf=0.99,
        start=start,
        end=start + len(raw),
    )


def _normalise(
    text: str,
    *,
    classifier: Callable[[str, str], list[Span]] = default_classifier,
):
    return normalize_segment(
        raw_text=text,
        tokens=[],
        is_final=True,
        state=StreamState(),
        lang_hint="hi",
        locale_policy="default",
        classifier=classifier,
    )


def test_default_classifier_remains_regex_only_passthrough() -> None:
    text = "call 9876543210 on 12/05/2026 at 17:30 and pay ₹1,250"

    direct = default_classifier(text, "hi")
    result = _normalise(text)

    assert direct
    assert all(span.rule_id.startswith("prefilter.") for span in direct)
    assert all(span.canonical == span.raw for span in direct)
    assert result.canonical_text == text
    assert result.display_text == text


def test_bare_prefilter_risky_shapes_do_not_gain_lexical_cues_from_location() -> None:
    text = "9876543210 12/05/2026 17:30"

    result = _normalise(text, classifier=lambda working, lang: prefilter(working))
    by_class = {span.cls: span for span in result.spans}

    assert result.canonical_text == text
    assert "missing_lex_cue" in (by_class["phone"].fallback_reason or "")
    assert "missing_lex_cue" in (by_class["date"].fallback_reason or "")
    assert "missing_lex_cue" in (by_class["time"].fallback_reason or "")


def test_supported_wfst_and_formatter_spans_still_pass() -> None:
    text = "9876543210 12-05-2026 05:30 PM ₹1250 12 %"

    def _classifier(working: str, lang: str) -> list[Span]:
        del lang
        return [
            _span(
                working,
                "9876543210",
                cls="phone",
                canonical="+91 98765 43210",
                rule_id="fmt.phone",
            ),
            _span(
                working,
                "12-05-2026",
                cls="date",
                canonical="12/05/2026",
                rule_id="wfst.date",
            ),
            _span(
                working,
                "05:30 PM",
                cls="time",
                canonical="5:30 PM",
                rule_id="wfst.time",
            ),
            _span(
                working,
                "₹1250",
                cls="money",
                canonical="₹1,250",
                rule_id="wfst.money",
            ),
            _span(
                working,
                "12 %",
                cls="percent",
                canonical="12%",
                rule_id="wfst.percent",
            ),
        ]

    result = _normalise(text, classifier=_classifier)

    assert result.canonical_text == "+91 98765 43210 12/05/2026 5:30 PM ₹1,250 12%"
    assert all(span.fallback_reason is None for span in result.spans)


def test_self_correction_spans_are_rejected_safely() -> None:
    text = "9876543210 no sorry 9876543211"

    def _classifier(working: str, lang: str) -> list[Span]:
        del lang
        first = _span(
            working,
            "9876543210",
            cls="phone",
            canonical="+91 98765 43210",
            rule_id="fmt.phone",
        )
        second = _span(
            working,
            "9876543211",
            cls="phone",
            canonical="+91 98765 43211",
            rule_id="fmt.phone",
            start_at=first.end or 0,
        )
        return [first, second]

    result = _normalise(text, classifier=_classifier)

    assert result.canonical_text == text
    assert len(result.spans) == 2
    assert all(span.ambiguous for span in result.spans)
    assert all(span.canonical == span.raw for span in result.spans)
    assert all("self_correction" in (span.fallback_reason or "") for span in result.spans)


def test_spoken_hindi_examples_rewrite_through_wfst_classifier() -> None:
    classifier = make_wfst_classifier(_DMY_POLICY)
    cases = [
        ("मुझे एक सौ पच्चीस रुपये भेजने हैं", "मुझे ₹125 भेजने हैं"),
        ("आज बारह मई दो हजार छब्बीस है", "आज 12/05/2026 है"),
        ("शाम पाँच बजे कॉल करो", "5:00 PM कॉल करो"),
        ("बारह दशमलव पाँच प्रतिशत", "12.5%"),
        (
            "मेरा मोबाइल नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है",
            "मेरा मोबाइल नंबर +91 98765 43210 है",
        ),
    ]

    for raw, expected in cases:
        result = _normalise(raw, classifier=classifier)
        assert result.raw_text == raw
        assert result.canonical_text == expected
        assert all(span.fallback_reason is None for span in result.spans)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("टू थाउजेंड", "2000"),
        ("मेरा अमाउंट टू थाउजेंड है", "मेरा अमाउंट 2000 है"),
        ("पेमेंट फाइव हंड्रेड रुपये", "पेमेंट 500 रुपये"),
    ],
)
def test_code_switched_devanagari_english_numbers_rewrite_through_normalizer(
    raw: str,
    expected: str,
) -> None:
    classifier = make_wfst_classifier(_DMY_POLICY)

    result = _normalise(raw, classifier=classifier)

    assert result.canonical_text == expected


def test_number_itn_regressions_remain_stable() -> None:
    classifier = make_wfst_classifier(_DMY_POLICY)
    cases = [
        ("दो हजार", "2000"),
        ("दो हज़ार", "2000"),
        ("two thousand", "two thousand"),
        ("यह सामान्य वाक्य है", "यह सामान्य वाक्य है"),
    ]

    for raw, expected in cases:
        result = _normalise(raw, classifier=classifier)
        assert result.canonical_text == expected


def test_spoken_digits_without_phone_cue_do_not_become_phone() -> None:
    raw = "नौ आठ सात छह पाँच चार तीन दो एक शून्य"

    result = _normalise(raw, classifier=make_wfst_classifier(_DMY_POLICY))

    assert result.raw_text == raw
    assert result.canonical_text == raw
    assert all(span.cls != "phone" for span in result.spans)


def test_spoken_no_parse_span_falls_back_with_reason_and_preserves_raw() -> None:
    raw = "मुझे एक सौ rupee भेजने हैं"

    result = _normalise(raw, classifier=make_wfst_classifier(_DMY_POLICY))

    assert result.raw_text == raw
    assert result.canonical_text == raw
    assert len(result.spans) == 1
    assert result.spans[0].cls == "money"
    assert result.spans[0].canonical == "एक सौ rupee"
    assert "wfst_no_parse" in (result.spans[0].fallback_reason or "")


def test_spoken_self_correction_still_blocks_risky_rewrite() -> None:
    raw = "एक सौ पच्चीस रुपये नहीं दो सौ रुपये"

    result = _normalise(raw, classifier=make_wfst_classifier(_DMY_POLICY))

    assert result.raw_text == raw
    assert result.canonical_text == raw
    assert len(result.spans) == 2
    assert all(span.cls == "money" for span in result.spans)
    assert all(span.canonical == span.raw for span in result.spans)
    assert all("self_correction" in (span.fallback_reason or "") for span in result.spans)
