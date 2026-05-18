"""WFST-backed classifier built on top of regex + spoken prefilters.

This is the additive Stage 1 surface: it keeps the existing classifier protocol
stable while upgrading the subset of prefilter spans that the current rollout
actually needs. By design, this classifier only emits:

* ``phone`` via the deterministic Indian mobile formatter,
* ``amount`` prefilter spans rewritten as threshold-facing ``money``,
* ``percent``, ``date``, and ``time`` via the WFST pipeline.

Other prefilter-only classes are deliberately ignored here. The lower-level
``default_classifier`` remains the broad regex-only surface; the gRPC service
opts into this classifier behind its rollout flag.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .contract import Span
from .dateparser_fallback import has_date_cue, try_dateparser_fallback
from .formatters.phone_in import parse_indian_mobile
from .locale_policy import TenantPolicy
from .regex_prefilter import prefilter as regex_prefilter
from .spoken_prefilter import prefilter as spoken_prefilter
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
    "cardinal": "cardinal",
    "decimal": "decimal",
}

_MERGE_PRIORITY: dict[str, int] = {
    # Preserve the regex prefilter's existing structural priority first.
    "url": 1,
    "email": 2,
    "ifsc": 3,
    "pan": 4,
    "aadhaar": 5,
    "phone": 6,
    "date": 7,
    "time": 8,
    "amount": 9,
    "percent": 10,
    # Spoken generic number spans are useful only when nothing more
    # semantically specific already covers them.
    "decimal": 11,
    "cardinal": 12,
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

        for span in _merged_prefilter_spans(working_text, lang=lang):
            try:
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
            except Exception:  # noqa: BLE001 — isolate one failed span
                fallback = _unexpected_error_fallback(span)
                if fallback is not None:
                    rewritten.append(fallback)

        return rewritten

    return classify


def _merged_prefilter_spans(text: str, *, lang: str) -> list[Span]:
    """Merge written + spoken candidates with one overlap policy.

    The regex prefilter already emits a non-overlapping set internally, but the
    new spoken detector can legitimately find nested alternatives such as:

        बारह दशमलव पाँच प्रतिशत
        ├─ percent  (specific, keep)
        └─ decimal  (generic, drop)

    Resolve all candidates together so offsets remain exact and generic number
    spans never shadow cue-bearing classes.
    """
    candidates = [*regex_prefilter(text), *spoken_prefilter(text, lang)]
    candidates.sort(
        key=lambda span: (
            _MERGE_PRIORITY.get(span.cls, 99),
            -(0 if span.start is None or span.end is None else span.end - span.start),
            span.start if span.start is not None else 10**9,
        )
    )
    selected: list[Span] = []
    for candidate in candidates:
        if candidate.start is None or candidate.end is None:
            selected.append(candidate)
            continue
        if any(
            selected_span.start is not None
            and selected_span.end is not None
            and selected_span.start < candidate.end
            and candidate.start < selected_span.end
            for selected_span in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda span: span.start if span.start is not None else 10**9)
    return selected


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


def _unexpected_error_fallback(span: Span) -> Span | None:
    """Collapse one formatter / WFST exception to a raw local fallback.

    The service-level handler still protects the whole request when the
    pipeline itself raises outside a span rewrite. Inside the classifier,
    however, one malformed span must not poison unrelated rewrites in
    the same segment.
    """
    if span.cls == "phone":
        return _fallback(span, cls="phone", rule_id="fmt.phone", reason="fmt_error")
    if span.cls == "date":
        return _fallback(span, cls="date", rule_id="wfst.date", reason="wfst_error")

    wfst_cls = _PREFILTER_TO_WFST_CLASS.get(span.cls)
    if wfst_cls is None:
        return None
    return _fallback(
        span,
        cls=wfst_cls,
        rule_id=f"wfst.{wfst_cls}",
        reason="wfst_error",
    )


__all__ = ["make_wfst_classifier"]
