"""Streaming-pipeline invariants.

Three properties verified against 20 fixture streams in
``fixtures.STREAMS``:

1. **No flicker.** Once :func:`normalize_segment` emits a span at
   offset ``[a, b)`` for a non-deferred partial, the same
   ``(cls, canonical, a, b)`` tuple appears in every subsequent
   non-deferred response within the same segment. The fixture streams
   are monotonically prefix-extending by construction, so an offset
   that contained a span at partial *N* still contains the same
   substring at partial *N+1*; any disappearance is a regression.

2. **Batch == streaming on finals.** Replaying a stream's partials
   through one ``StreamState`` and then submitting the final must
   produce the same :class:`SegmentResult` as running the final text
   in batch (a fresh ``StreamState``, no partials). The current
   implementation is path-independent at the final boundary by
   design — this test pins that contract so future changes that
   accidentally let partial history leak into final output are
   caught.

3. **p95 added latency < 10 ms per final segment.** Measured as the
   wall-clock delta between :func:`normalize_segment` on the final
   text and a raw-passthrough baseline. p95 is over the 20 finals.

The latency test is marked ``slow`` so it is skipped by default; run
with ``pytest -m slow tests/streaming/test_streaming.py``.
"""

from __future__ import annotations

import statistics
import time
from typing import Iterable

import pytest

from itn_service.runtime.contract import SegmentResult, Span
from itn_service.runtime.normalizer import normalize_segment
from itn_service.runtime.stream_state import StreamState

from .fixtures import STREAMS, Hyp, Stream


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _span_key(s: Span) -> tuple[str, str, int, int]:
    """Identity tuple used by the no-flicker check.

    Includes ``cls``, ``canonical``, and the start/end offsets. We
    deliberately do *not* include ``rule_id`` so a rule rename inside
    the prefilter does not look like a flicker; the user-facing
    identity is (class, output, location).
    """
    assert s.start is not None and s.end is not None
    return (s.cls, s.canonical, s.start, s.end)


def _run_stream(stream: Stream) -> list[SegmentResult]:
    """Run ``stream`` through one ``StreamState`` and collect responses."""
    state = StreamState()
    out: list[SegmentResult] = []
    for hyp in stream.hyps:
        out.append(_run_one(hyp, state))
    return out


def _run_one(hyp: Hyp, state: StreamState) -> SegmentResult:
    return normalize_segment(
        raw_text=hyp.text,
        tokens=list(hyp.tokens),
        is_final=hyp.is_final,
        state=state,
        lang_hint=hyp.lang_hint,
        locale_policy="",
    )


def _non_deferred(results: Iterable[SegmentResult]) -> list[SegmentResult]:
    return [r for r in results if not r.deferred]


# ---------------------------------------------------------------------------
# 1. No flicker.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", STREAMS, ids=lambda s: s.name)
def test_no_flicker_on_stable_spans(stream: Stream) -> None:
    """Stable spans never disappear in a later partial of the same segment.

    Walks the per-stream responses in order. Whenever a span appears
    in a non-deferred response, it is recorded; every subsequent
    non-deferred response must re-emit the same span (matched by
    ``_span_key``). Since the fixture stream is prefix-extending, the
    underlying substring at the span's offsets is unchanged, so the
    span must persist.
    """
    results = _run_stream(stream)
    seen: set[tuple[str, str, int, int]] = set()

    for idx, res in enumerate(results):
        if res.deferred:
            # A deferred response carries no spans; it cannot flicker
            # anything off. Skip it.
            continue
        current = {_span_key(s) for s in res.spans if s.start is not None}
        missing = seen - current
        # Sanity: the raw text grows monotonically, so any previously
        # emitted offset is still inside the current raw text. (If
        # this assertion fails, the fixture is wrong, not the code.)
        for cls, canon, a, b in missing:
            assert b <= len(res.raw_text), (
                f"stream {stream.name!r} fixture not prefix-extending at "
                f"hyp #{idx}: span ({cls!r}, [{a},{b})) is beyond current raw "
                f"length {len(res.raw_text)}"
            )
        assert not missing, (
            f"stream {stream.name!r}: spans disappeared between partials "
            f"at hyp #{idx}: {sorted(missing)}\n"
            f"  prev seen={sorted(seen)}\n"
            f"  current  ={sorted(current)}"
        )
        seen |= current


# ---------------------------------------------------------------------------
# 2. Batch == streaming on finals.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", STREAMS, ids=lambda s: s.name)
def test_final_equals_batch(stream: Stream) -> None:
    """The final ``SegmentResult`` is identical whether the partials were replayed or not.

    The streaming pipeline must be deterministic at the final
    boundary: nothing from earlier partials should leak into the
    final's canonical_text, display_text, or span list. Compares
    canonical fields directly; ``raw_text`` is obviously identical
    (we never mutate it).
    """
    streamed_results = _run_stream(stream)
    streaming_final = streamed_results[-1]
    assert streaming_final.deferred is False, (
        f"stream {stream.name!r}: final was unexpectedly deferred"
    )

    # Replay only the final in a fresh state.
    final_hyp = stream.hyps[-1]
    batch_final = _run_one(final_hyp, StreamState())

    assert batch_final.raw_text == streaming_final.raw_text
    assert batch_final.canonical_text == streaming_final.canonical_text, (
        f"stream {stream.name!r}: canonical_text drift\n"
        f"  streaming={streaming_final.canonical_text!r}\n"
        f"  batch    ={batch_final.canonical_text!r}"
    )
    assert batch_final.display_text == streaming_final.display_text
    assert batch_final.lang == streaming_final.lang
    assert batch_final.script == streaming_final.script
    # Spans: compare as ordered list of identity tuples so a list-vs-tuple
    # or pydantic-rev difference doesn't fail us spuriously.
    assert [_span_key(s) for s in batch_final.spans] == [
        _span_key(s) for s in streaming_final.spans
    ]


# ---------------------------------------------------------------------------
# 3. p95 added latency < 10 ms per final segment.
# ---------------------------------------------------------------------------


# Warmup: import / load yaml / compile regex caches happen on the first
# call. We exclude that first call from the timed sample to avoid
# punishing CI for one-time costs that production amortises.
_WARMUP_RUNS = 3

# p95 bound from the spec. The blueprint's latency table caps each
# final's normalisation at the request budget for live telephony; the
# streaming pseudo-code uses 10 ms as the canonical added-latency
# target over raw passthrough.
_P95_LIMIT_MS = 10.0


@pytest.mark.slow
def test_p95_latency_under_10ms() -> None:
    """p95 added latency over raw passthrough is below 10 ms across the 20 finals."""
    finals = [stream.hyps[-1] for stream in STREAMS]

    # Warmup once.
    for _ in range(_WARMUP_RUNS):
        for hyp in finals:
            _run_one(hyp, StreamState())

    # Measure: end-to-end normalize_segment minus a baseline that does
    # only the SegmentResult construction (raw-text passthrough). This
    # isolates the *added* latency the ITN layer introduces.
    deltas_ms: list[float] = []
    for hyp in finals:
        state = StreamState()
        t0 = time.perf_counter()
        _run_one(hyp, state)
        t1 = time.perf_counter()

        # Baseline: building the equivalent passthrough SegmentResult.
        t2 = time.perf_counter()
        SegmentResult(
            raw_text=hyp.text,
            canonical_text=hyp.text,
            display_text=hyp.text,
            spans=[],
            deferred=True,
            lang=(hyp.lang_hint or "und"),
            script="Common",
        )
        t3 = time.perf_counter()

        delta_ms = ((t1 - t0) - (t3 - t2)) * 1000.0
        # Clamp negative deltas to 0 — measurement noise on very small
        # operations can push the baseline above the test.
        deltas_ms.append(max(delta_ms, 0.0))

    # Sort and take the p95. With 20 samples, p95 is the 19th (0-indexed 18).
    deltas_ms.sort()
    p95 = deltas_ms[int(0.95 * len(deltas_ms))] if len(deltas_ms) > 1 else deltas_ms[-1]
    p50 = statistics.median(deltas_ms)
    assert p95 < _P95_LIMIT_MS, (
        f"p95 added latency {p95:.2f} ms exceeds budget {_P95_LIMIT_MS} ms "
        f"(p50={p50:.2f} ms, max={deltas_ms[-1]:.2f} ms, samples={len(deltas_ms)})"
    )
