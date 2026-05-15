"""WFST-backed classifier built on top of the regex prefilter.

This is the additive Stage 1 surface: it keeps the existing classifier protocol
stable while upgrading the subset of prefilter spans that the current rollout
actually needs. By design, this classifier only emits:

* ``phone`` via the deterministic Indian mobile formatter,
* ``amount`` prefilter spans rewritten as threshold-facing ``money``,
* ``percent``, ``date``, and ``time`` via the WFST pipeline.

Other prefilter-only classes are deliberately ignored here. The legacy
``default_classifier`` remains the broad regex-only surface until a later commit
chooses to wire this classifier into the live path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .contract import Span
from .dateparser_fallback import has_date_cue, try_dateparser_fallback
from .formatters.phone_in import parse_indian_mobile
from .locale_policy import TenantPolicy
from .regex_prefilter import prefilter
from .wfst_factory import get_pipeline


class _DateNormalizationResult(Protocol):
    canonical: str | None
    fallback_reason: str | None


class _Pipeline(Protocol):
    def normalize_span(self, raw: str, cls: str) -> str | None: ...

    def normalize_date(
        self,
        raw: str,
        *,
        date_order: str,
    ) -> _DateNormalizationResult: ...


Classifier = Callable[[str, str], list[Span]]


_PREFILTER_TO_WFST_CLASS: dict[str, str] = {
    "amount": "money",
    "percent": "percent",
    "time": "time",
}


def make_wfst_classifier(tenant_policy: TenantPolicy) -> Classifier:
    """Build a per-tenant classifier closure.

    The closure captures locale policy so the public classifier shape remains
    ``(working_text, lang) -> list[Span]``. This module intentionally does not
    run self-correction; that is a cross-span orchestration concern and lands in
    ``normalizer.py`` in the next wiring commit.
    """

    def classify(working_text: str, lang: str) -> list[Span]:
        pipeline = get_pipeline(lang)
        rewritten: list[Span] = []

        for span in prefilter(working_text):
            if span.cls == "phone":
                rewritten.append(_rewrite_phone(span))
                continue

            if span.cls == "date":
                rewritten.append(
                    _rewrite_date(
                        span,
                        pipeline=pipeline,
                        tenant_policy=tenant_policy,
                        context_text=working_text,
                    )
                )
                continue

            wfst_cls = _PREFILTER_TO_WFST_CLASS.get(span.cls)
            if wfst_cls is not None:
                rewritten.append(
                    _rewrite_wfst(span, pipeline=pipeline, wfst_cls=wfst_cls)
                )

        return rewritten

    return classify


def _rewrite_phone(span: Span) -> Span:
    canonical = parse_indian_mobile(span.raw)
    if canonical is None:
        return _fallback(span, cls="phone", rule_id="fmt.phone", reason="fmt_no_parse")
    return span.model_copy(
        update={
            "cls": "phone",
            "canonical": canonical,
            "rule_id": "fmt.phone",
            "fallback_reason": None,
        }
    )


def _rewrite_wfst(span: Span, *, pipeline: _Pipeline | None, wfst_cls: str) -> Span:
    rule_id = f"wfst.{wfst_cls}"
    if pipeline is None:
        return _fallback(span, cls=wfst_cls, rule_id=rule_id, reason="wfst_unavailable")

    canonical = pipeline.normalize_span(span.raw, wfst_cls)
    if canonical is None:
        return _fallback(span, cls=wfst_cls, rule_id=rule_id, reason="wfst_no_parse")

    return span.model_copy(
        update={
            "cls": wfst_cls,
            "canonical": canonical,
            "rule_id": rule_id,
            "fallback_reason": None,
        }
    )


def _rewrite_date(
    span: Span,
    *,
    pipeline: _Pipeline | None,
    tenant_policy: TenantPolicy,
    context_text: str,
) -> Span:
    if pipeline is None:
        return _fallback(span, cls="date", rule_id="wfst.date", reason="wfst_unavailable")

    normalized = pipeline.normalize_date(span.raw, date_order=tenant_policy.date_order)
    if normalized.canonical is not None:
        return span.model_copy(
            update={
                "cls": "date",
                "canonical": normalized.canonical,
                "rule_id": "wfst.date",
                "fallback_reason": None,
            }
        )

    # ``ambiguous_numeric_date`` is a policy rejection, not a cue to ask another
    # parser to guess. Preserve it as the final reason.
    if normalized.fallback_reason is not None:
        return _fallback(
            span,
            cls="date",
            rule_id="wfst.date",
            reason=normalized.fallback_reason,
        )

    if not has_date_cue(context_text):
        return _fallback(span, cls="date", rule_id="wfst.date", reason="wfst_no_parse")

    fallback = try_dateparser_fallback(
        span.raw,
        locale_date_order=tenant_policy.date_order,
        classifier_conf=span.conf,
        asr_conf=1.0,
        context_text=context_text,
    )
    if fallback.canonical is not None:
        return span.model_copy(
            update={
                "cls": "date",
                "canonical": fallback.canonical,
                "rule_id": "dateparser.date",
                "fallback_reason": None,
            }
        )

    return _fallback(
        span,
        cls="date",
        rule_id="dateparser.date",
        reason=fallback.fallback_reason or "wfst_no_parse",
    )


def _fallback(span: Span, *, cls: str, rule_id: str, reason: str) -> Span:
    return span.model_copy(
        update={
            "cls": cls,
            "canonical": span.raw,
            "rule_id": rule_id,
            "ambiguous": True,
            "fallback_reason": reason,
        }
    )


__all__ = ["make_wfst_classifier"]
