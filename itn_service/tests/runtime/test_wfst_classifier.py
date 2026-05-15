"""Tests for the additive WFST-backed classifier surface."""

from __future__ import annotations

from types import SimpleNamespace

from itn_service.runtime.dateparser_fallback import DateParseResult
from itn_service.runtime.locale_policy import TenantPolicy
from itn_service.runtime.wfst_classifier import make_wfst_classifier


_DMY_POLICY = TenantPolicy(
    tenant_id="default",
    region="IN",
    date_order="DMY",
    currency="INR",
)


class _FakePipeline:
    def __init__(
        self,
        *,
        span_outputs: dict[tuple[str, str], str | None] | None = None,
        date_canonical: str | None = None,
        date_reason: str | None = None,
    ) -> None:
        self.span_outputs = span_outputs or {}
        self.date_canonical = date_canonical
        self.date_reason = date_reason
        self.span_calls: list[tuple[str, str]] = []
        self.date_calls: list[tuple[str, str]] = []

    def normalize_span(self, raw: str, cls: str) -> str | None:
        self.span_calls.append((raw, cls))
        return self.span_outputs.get((raw, cls))

    def normalize_date(self, raw: str, *, date_order: str) -> SimpleNamespace:
        self.date_calls.append((raw, date_order))
        return SimpleNamespace(
            canonical=self.date_canonical,
            fallback_reason=self.date_reason,
        )


def test_phone_uses_formatter_even_when_wfst_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.get_pipeline",
        lambda lang: None,
    )

    spans = make_wfst_classifier(_DMY_POLICY)("call 9876543210", "hi")

    assert len(spans) == 1
    assert spans[0].cls == "phone"
    assert spans[0].canonical == "+91 98765 43210"
    assert spans[0].rule_id == "fmt.phone"
    assert spans[0].ambiguous is False


def test_amount_routes_to_wfst_money(monkeypatch) -> None:
    pipeline = _FakePipeline(span_outputs={("₹1,250", "money"): "1250"})
    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.get_pipeline",
        lambda lang: pipeline,
    )

    spans = make_wfst_classifier(_DMY_POLICY)("pay ₹1,250 now", "hi")

    assert len(spans) == 1
    assert spans[0].cls == "money"
    assert spans[0].canonical == "1250"
    assert spans[0].rule_id == "wfst.money"
    assert pipeline.span_calls == [("₹1,250", "money")]


def test_date_honours_tenant_date_order(monkeypatch) -> None:
    pipeline = _FakePipeline(date_canonical="12/05/2026")
    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.get_pipeline",
        lambda lang: pipeline,
    )

    spans = make_wfst_classifier(_DMY_POLICY)("on 12/05/2026", "hi")

    assert len(spans) == 1
    assert spans[0].cls == "date"
    assert spans[0].canonical == "12/05/2026"
    assert spans[0].rule_id == "wfst.date"
    assert pipeline.date_calls == [("12/05/2026", "DMY")]


def test_date_falls_back_to_dateparser_only_with_cue(monkeypatch) -> None:
    pipeline = _FakePipeline(date_canonical=None, date_reason=None)
    fallback_calls: list[dict[str, object]] = []

    def _fake_dateparser(raw: str, **kwargs: object) -> DateParseResult:
        fallback_calls.append({"raw": raw, **kwargs})
        return DateParseResult(canonical="12/05/2026", fallback_reason=None)

    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.get_pipeline",
        lambda lang: pipeline,
    )
    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.try_dateparser_fallback",
        _fake_dateparser,
    )

    spans = make_wfst_classifier(_DMY_POLICY)("date 12/05/2026", "hi")

    assert len(spans) == 1
    assert spans[0].canonical == "12/05/2026"
    assert spans[0].rule_id == "dateparser.date"
    assert fallback_calls == [
        {
            "raw": "12/05/2026",
            "locale_date_order": "DMY",
            "classifier_conf": 1.0,
            "asr_conf": 1.0,
            "context_text": "date 12/05/2026",
        }
    ]


def test_date_without_cue_stays_raw_and_skips_dateparser(monkeypatch) -> None:
    pipeline = _FakePipeline(date_canonical=None, date_reason=None)
    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.get_pipeline",
        lambda lang: pipeline,
    )

    spans = make_wfst_classifier(_DMY_POLICY)("12/05/2026", "hi")

    assert len(spans) == 1
    assert spans[0].canonical == "12/05/2026"
    assert spans[0].rule_id == "wfst.date"
    assert spans[0].ambiguous is True
    assert spans[0].fallback_reason == "wfst_no_parse"


def test_excluded_prefilter_classes_are_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.get_pipeline",
        lambda lang: None,
    )

    spans = make_wfst_classifier(_DMY_POLICY)(
        "www.example.com x@y.com IFSC HDFC0001234 PAN ABCDE1234F Aadhaar 234567890123",
        "hi",
    )

    assert spans == []
