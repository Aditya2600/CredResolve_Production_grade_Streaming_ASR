"""Per-segment normalisation entry point.

Implements ``normalize_segment`` matching the streaming pseudo-code in
``docs/implementation_bluprint_INR.md``. The flow is:

    raw_text  ──► (unchanged, returned verbatim)
       │
       ├─► working_copy ──► route_language ──► classify_tokens ──► gate
       │                                                              │
       │                                          (filter on partials) │
       │                                                              ▼
       └────────────────────────────────────── apply_spans ──► canonical_text
                                                                      │
                                                                      ▼
                                                               render_display
                                                                      │
                                                                      ▼
                                                               display_text

The four invariants from ``CONTRIBUTING.md`` are enforced here:

1. ``raw_text`` is propagated unchanged from input to output. The
   working-copy folding (``unicode_clean.working_copy``) never touches
   the verbatim string the decoder produced.
2. No LLM call — only the deterministic prefilter + WFST classifier and
   the confidence gate run.
3. No FAR compilation — the classifier loads pre-built FARs lazily on
   first use and caches them on the call-site object.
4. Canonical storage uses Latin digits and ICU-canonical separators;
   the display surface is produced by ``runtime.display_renderer``.

The classifier callable is injectable so tests can pin a deterministic
mock without touching disk. The default classifier wires the regex
prefilter as the only span source, which is correct (canonical == raw
with conf=1.0) even before WFST FARs are built — i.e. this module is
runnable as soon as ``runtime/regex_prefilter.py`` is in place.
"""

from __future__ import annotations

import re
from typing import Callable, Protocol

from .confidence_gate import ThresholdTable, gate, load_thresholds
from .contract import SegmentResult, Span, Token
from .dateparser_fallback import has_date_cue
from .regex_prefilter import prefilter
from .script_router import route_language
from .self_correction import detect_self_corrections
from .stream_state import StreamState
from .unicode_clean import working_copy

# ---------------------------------------------------------------------------
# Classifier hook.
# ---------------------------------------------------------------------------


class Classifier(Protocol):
    """Pluggable span classifier.

    Implementations take the *working-copy* text and the routed
    language code and return a list of :class:`~contract.Span` objects.
    Each span carries codepoint offsets (``start`` / ``end``) into the
    working copy so :func:`apply_spans` can splice it back in.

    The default classifier is :func:`default_classifier`, which runs
    the regex prefilter only. The gRPC service may inject the
    tenant-aware classifier from ``runtime.wfst_classifier`` when its
    rollout flag is enabled.
    """

    def __call__(self, working_text: str, lang: str) -> list[Span]: ...


def default_classifier(working_text: str, lang: str) -> list[Span]:
    """Regex-prefilter-only classifier.

    Safe even before the WFST FARs are built: every span carries
    ``canonical == raw`` with full confidence, so whether the gate
    accepts or rejects a located prefilter span, the emitted surfaces
    remain verbatim passthrough.
    """
    del lang  # unused at the prefilter layer; FAR routing happens in WFST classifier
    return prefilter(working_text)


# ---------------------------------------------------------------------------
# Span aggregation helpers.
# ---------------------------------------------------------------------------


def _aggregate_asr_conf(tokens: list[Token]) -> float:
    """Conservative aggregation: minimum over token confidences.

    The pseudo-code uses a free-form ``asr_confidence`` for each span;
    in the absence of codepoint-to-token alignment we use the minimum
    across the segment, which is conservative (it will defer more, not
    less, than a per-span average). A no-token segment returns 1.0 —
    text-only callers explicitly trust the text.
    """
    if not tokens:
        return 1.0
    return min(t.conf for t in tokens)


_CURRENCY_CUE_RE = re.compile(
    r"(?:₹|Rs\.?|INR|US\$|\$|USD|£|GBP|€|EUR|rupees?|रुपये|रुपया|रू)",
    re.IGNORECASE,
)
_PERCENT_CUE_RE = re.compile(r"(?:%|percent|प्रतिशत|टक्के|टक्का)", re.IGNORECASE)
_DECIMAL_CUE_RE = re.compile(r"(?:\b(?:point|dot|decimal)\b|दशमलव)", re.IGNORECASE)
_TIME_CUE_RE = re.compile(
    r"(?:\b(?:AM|PM|A\.M\.|P\.M\.)\b|बजे|बजकर|मिनट|सुबह|दोपहर|शाम|रात)",
    re.IGNORECASE,
)
_PHONE_CUE_RE = re.compile(
    r"(?:\b(?:phone|mobile|number|otp)\b|फ़ोन|फोन|मोबाइल|नंबर|नम्बर|ओटीपी)",
    re.IGNORECASE,
)


def _span_has_lex_cue(span: Span, *, context_text: str) -> bool:
    """Whether a span carries sufficient cue evidence to auto-rewrite.

    The prefilter is a locator, not a semantic witness: a bare numeric
    phone/date/time shape is intentionally *not* enough on its own.
    Stronger downstream rules may satisfy the cue requirement when the
    rule itself proves the missing piece (for example ``fmt.phone``'s
    strict structural parse or ``wfst.date``'s locale-aware date path).
    """
    if span.cls in {"amount", "currency", "money"}:
        # Amount regexes and the money WFST both require an explicit
        # currency surface. Keep the cue tied to that surface rather
        # than to the fact that a regex located the span.
        return bool(_CURRENCY_CUE_RE.search(span.raw))

    if span.cls == "percent":
        # A literal percent symbol or lexical percent word is self-cuing.
        return bool(_PERCENT_CUE_RE.search(span.raw))

    if span.cls == "decimal":
        # Generic spoken decimals are only safe when an explicit decimal marker
        # is in the span. Percent-bearing decimals arrive as ``percent`` spans.
        return bool(_DECIMAL_CUE_RE.search(span.raw))

    if span.cls == "time":
        # ``17:30`` is only a shape; ``5:30 PM`` or a Hindi time cue is
        # semantic evidence. Do not let ``prefilter.time`` alone pass.
        return bool(_TIME_CUE_RE.search(span.raw))

    if span.cls == "date":
        # Plain prefilter date matches are merely located. A date WFST
        # result means the policy-aware date path accepted it; the
        # dateparser branch is likewise already cue/policy-gated.
        return (
            has_date_cue(span.raw)
            or has_date_cue(context_text)
            or span.rule_id.startswith(("wfst.", "dateparser."))
        )

    if span.cls == "phone":
        # The strict formatter is accepted structural evidence. Raw
        # prefilter phone shapes still need actual phone/OTP context.
        return (
            span.rule_id.startswith("fmt.")
            or bool(_PHONE_CUE_RE.search(context_text))
        )

    # Other classes should earn their own semantics explicitly instead
    # of inheriting trust from a generic ``prefilter.*`` location.
    return False


def _mark_self_corrections(text: str, spans: list[Span]) -> list[Span]:
    """Mark spans participating in a correction pair as unsafe."""
    unsafe = detect_self_corrections(text, spans)
    if not unsafe:
        return spans
    return [
        span.model_copy(
            update={
                "ambiguous": True,
                "fallback_reason": "self_correction",
            }
        )
        if idx in unsafe
        else span
        for idx, span in enumerate(spans)
    ]


def apply_spans(text: str, spans: list[Span]) -> str:
    """Splice each span's ``canonical`` back into ``text``.

    Spans without ``start`` / ``end`` offsets are skipped — they cannot
    be located deterministically in the working copy. The list is
    sorted by start offset before splicing; overlapping accepted spans
    are not expected (the prefilter resolves overlap before emit and
    the gate never widens a span).
    """
    located = [s for s in spans if s.start is not None and s.end is not None]
    if not located:
        return text

    located.sort(key=lambda s: s.start or 0)
    out: list[str] = []
    cursor = 0
    for span in located:
        # mypy: the filter above guarantees these are not None.
        assert span.start is not None and span.end is not None
        if span.start < cursor:
            # Overlap with a previously emitted span — should not
            # happen (regex_prefilter resolves overlaps), but keeping
            # the rest of the text is safer than raising in the hot
            # path.
            continue
        out.append(text[cursor : span.start])
        out.append(span.canonical)
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Display.
# ---------------------------------------------------------------------------


def _render_display(canonical_text: str, lang: str) -> str:
    """Locale-rendered display surface.

    The full ICU/CLDR digit-shaping path lives in
    ``runtime/display_renderer.py``; until that is wired in we return
    the canonical text unchanged. This preserves invariant 4
    (canonical storage uses Latin digits) — the display layer is
    additive, never a fallback that mutates canonical.
    """
    del lang
    return canonical_text


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

# Classes that may render on a partial when stable + unambiguous, per
# ``configs/thresholds.yaml § streaming.partial_safe_classes`` and the
# blueprint's partial-display rules.
_PARTIAL_SAFE_CLASSES: frozenset[str] = frozenset({"cardinal", "money", "percent"})


def normalize_segment(
    raw_text: str,
    tokens: list[Token],
    is_final: bool,
    state: StreamState,
    lang_hint: str | None,
    locale_policy: str,
    *,
    thresholds: ThresholdTable | None = None,
    classifier: Classifier = default_classifier,
    display_fn: Callable[[str, str], str] = _render_display,
) -> SegmentResult:
    """Normalise one segment (partial or final) and return a SegmentResult.

    This is the request-path entry point invoked by the streaming gRPC
    handler. It is pure (no I/O beyond what the injected classifier
    does on its first call to load a FAR) and safe to call from a
    thread pool. ``state`` is mutated exactly once per call by
    :meth:`StreamState.update_stability`.

    Args:
        raw_text: verbatim ASR output. Never mutated; copied to
            ``SegmentResult.raw_text`` as-is.
        tokens: optional per-word tokens with timing + confidence.
            Used to derive the aggregated ASR confidence for the gate.
        is_final: whether this is a segment-final hypothesis. Final
            segments run the full pipeline; partials only render
            ``_PARTIAL_SAFE_CLASSES`` and only when stable.
        state: per-call ``StreamState`` instance. Mutated.
        lang_hint: BCP-47-ish ASR language hint, or None.
        locale_policy: tenant id for downstream date/currency policy.
            Currently propagated through the result via ``lang`` /
            ``script``; the date branch consumes it in the WFST stage.
        thresholds: parsed ``thresholds.yaml`` table. Loaded lazily on
            first call when None — production callers should preload
            it on a long-lived service object (see ``grpc_server``).
        classifier: span source. Defaults to the regex prefilter.
        display_fn: ``(canonical, lang) -> display`` renderer. Defaults
            to a passthrough until ``display_renderer`` is wired in.

    Returns:
        A populated :class:`~contract.SegmentResult`. ``deferred`` is
        True when the partial was unstable (the result mirrors the raw
        text in every surface) — callers should not persist a deferred
        partial, only display it.
    """
    # --- 1. stability bookkeeping --------------------------------------------
    stable = state.update_stability(raw_text)

    # Partial + unstable: defer entirely. Surfaces mirror raw_text so
    # the UI shows what the decoder produced verbatim until the segment
    # firms up.
    if not is_final and stable < state.stability_threshold:
        route = route_language(raw_text, asr_hint=lang_hint)
        return SegmentResult(
            raw_text=raw_text,
            canonical_text=raw_text,
            display_text=raw_text,
            spans=[],
            deferred=True,
            lang=route.lang,
            script=route.script,
        )

    # --- 2. routing + classification ----------------------------------------
    working = working_copy(raw_text)
    route = route_language(working, asr_hint=lang_hint)
    spans = classifier(working, route.lang)
    spans = _mark_self_corrections(working, spans)

    # On partials, restrict to the small set of low-risk classes
    # (cardinal/money/percent) per the blueprint's partial-display
    # rules. Ambiguous partial spans are filtered out before the gate;
    # final spans go through the gate so fallback provenance survives.
    if not is_final:
        spans = [
            s for s in spans
            if s.cls in _PARTIAL_SAFE_CLASSES and not s.ambiguous
        ]

    # --- 3. confidence gate --------------------------------------------------
    if thresholds is None:
        thresholds = load_thresholds()

    asr_conf = _aggregate_asr_conf(tokens)
    safe_spans: list[Span] = []
    for span in spans:
        gated = gate(
            span,
            asr_conf=asr_conf,
            has_lex_cue=_span_has_lex_cue(span, context_text=working),
            is_partial=not is_final,
            thresholds=thresholds,
        )
        # ``gate`` returns a copy with ``canonical = raw`` and a
        # populated ``fallback_reason`` when a check fails; we keep
        # that copy so provenance is preserved in the span log even
        # when no rewrite happened.
        safe_spans.append(gated)

    # --- 4. assemble canonical + display ------------------------------------
    canonical = apply_spans(working, safe_spans)
    display = display_fn(canonical, route.lang)

    return SegmentResult(
        raw_text=raw_text,
        canonical_text=canonical,
        display_text=display,
        spans=safe_spans,
        deferred=False,
        lang=route.lang,
        script=route.script,
    )


__all__ = [
    "Classifier",
    "apply_spans",
    "default_classifier",
    "normalize_segment",
]
