"""Request-path coverage for the gRPC ITN servicer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from itn_service.service import grpc_server
from itn_service.service.grpc_server import ItnServicer, ServicePolicy, load_service_policy


class _ProtoMessage(SimpleNamespace):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)


class _StubModule:
    Span = _ProtoMessage
    NormalizeResponse = _ProtoMessage


class _Context:
    def peer(self) -> str:
        return "test-peer"


def _request(
    text: str,
    *,
    lang_hint: str = "hi",
    locale_policy: str = "default",
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        tokens=[],
        is_final=True,
        lang_hint=lang_hint,
        locale_policy=locale_policy,
    )


def _responses(
    servicer: ItnServicer,
    *requests: SimpleNamespace,
) -> list[_ProtoMessage]:
    return list(servicer.StreamNormalize(iter(requests), _Context()))


def _servicer(*, enabled: bool) -> ItnServicer:
    return ItnServicer(
        service_policy=ServicePolicy(wfst_classifier_enabled=enabled),
        stub_module=_StubModule,
    )


def test_policy_config_enables_wfst_by_default_and_can_disable(tmp_path: Path) -> None:
    assert load_service_policy().wfst_classifier_enabled is True

    policy = tmp_path / "policy.yaml"
    policy.write_text("wfst_classifier_enabled: false\n", encoding="utf-8")
    assert load_service_policy(policy).wfst_classifier_enabled is False


def test_flag_false_keeps_regex_only_passthrough_behavior() -> None:
    text = "call 9876543210"

    response = _responses(_servicer(enabled=False), _request(text))[0]

    assert response.raw_text == text
    assert response.canonical_text == text
    assert response.display_text == text
    assert response.spans
    assert all(span.rule_id.startswith("prefilter.") for span in response.spans)


def test_flag_true_rewrites_supported_hindi_example_through_service_path() -> None:
    text = "आज 12-05-2026 है"

    response = _responses(_servicer(enabled=True), _request(text))[0]

    assert response.raw_text == text
    assert response.canonical_text == "आज 12/05/2026 है"
    assert response.display_text == "आज 12/05/2026 है"
    assert response.lang == "hi"
    assert [(span.cls, span.rule_id) for span in response.spans] == [
        ("date", "wfst.date")
    ]


def test_itn_failure_returns_raw_passthrough(monkeypatch) -> None:
    text = "आज 12-05-2026 है"

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(grpc_server, "normalize_segment", _raise)

    response = _responses(_servicer(enabled=True), _request(text))[0]

    assert response.raw_text == text
    assert response.canonical_text == text
    assert response.display_text == text
    assert response.spans == []
    assert response.deferred is True


def test_language_hint_selects_hindi_pipeline_for_common_surface() -> None:
    text = "05:30 PM"

    response = _responses(_servicer(enabled=True), _request(text, lang_hint="hi"))[0]

    assert response.lang == "hi"
    assert response.canonical_text == "5:30 PM"
    assert response.spans[0].rule_id == "wfst.time"


def test_date_policy_is_resolved_per_request_even_within_one_stream() -> None:
    text = "on 12-05-2026"

    dmy, mdy = _responses(
        _servicer(enabled=True),
        _request(text, locale_policy="default"),
        _request(text, locale_policy="acme_us"),
    )

    assert dmy.canonical_text == "on 12/05/2026"
    assert dmy.spans[0].fallback_reason == ""

    assert mdy.canonical_text == text
    assert mdy.spans[0].canonical == "12-05-2026"
    assert "ambiguous_numeric_date" in mdy.spans[0].fallback_reason


def test_one_bad_span_falls_back_locally_without_poisoning_request(monkeypatch) -> None:
    class _BrokenMoneyPipeline:
        def normalize_span(self, raw: str, cls: str) -> str | None:
            if cls == "money":
                raise RuntimeError("bad money span")
            return None

        def normalize_date(self, raw: str, *, date_order: str) -> SimpleNamespace:
            del raw, date_order
            return SimpleNamespace(canonical=None, fallback_reason=None)

    monkeypatch.setattr(
        "itn_service.runtime.wfst_classifier.get_pipeline",
        lambda lang: _BrokenMoneyPipeline(),
    )

    text = "call 9876543210 and pay ₹1,250"
    response = _responses(_servicer(enabled=True), _request(text))[0]
    spans = {span.cls: span for span in response.spans}

    assert response.canonical_text == "call +91 98765 43210 and pay ₹1,250"
    assert spans["phone"].canonical == "+91 98765 43210"
    assert spans["money"].canonical == "₹1,250"
    assert spans["money"].fallback_reason == "wfst_error;ambiguous"
