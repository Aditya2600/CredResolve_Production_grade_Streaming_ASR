"""p95 latency microbenchmark for the cleanup + routing + prefilter
hot path.

Per the plan's "Latency avoidance techniques", classification latency
must stay well below the streaming partial cadence. This test enforces
the design contract: ``working_copy + detect_script + prefilter`` on a
representative 200-character mixed-script segment p95 below **0.5 ms**.

The benchmark runs 5,000 iterations after a warm-up, computes the
sorted p95, and asserts. It is opt-in (marked ``slow``) so casual
``pytest`` runs from a developer laptop don't pay the cost; CI runs it
explicitly via ``pytest -m slow``.
"""

from __future__ import annotations

import time

import pytest

from itn_service.runtime.regex_prefilter import prefilter
from itn_service.runtime.script_router import detect_script
from itn_service.runtime.unicode_clean import working_copy


# Representative ~200-char Hinglish segment with phone, amount, date,
# time, email, IFSC and PAN — exercises every prefilter pattern.
_SEGMENT = (
    "Sir aapka payment ₹1,25,000 ka HDFC0001234 par 12/05/2026 ko "
    "5:30 PM tak settle ho jayega; PAN ABCDE1234F use kiya. Call "
    "9876543210 ya mail user@example.com. नमस्ते."
)


def _ensure_segment_length() -> None:
    # Sanity: must be roughly 200 chars to match the latency contract.
    assert 180 <= len(_SEGMENT) <= 230, len(_SEGMENT)


@pytest.mark.slow
def test_p95_under_half_ms() -> None:
    _ensure_segment_length()

    iterations = 5000

    # Warm caches: PyICU script lookup, regex JIT, lru_cache.
    for _ in range(200):
        wc = working_copy(_SEGMENT)
        detect_script(wc)
        prefilter(wc)

    samples_ns: list[int] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        wc = working_copy(_SEGMENT)
        detect_script(wc)
        prefilter(wc)
        samples_ns.append(time.perf_counter_ns() - t0)

    samples_ns.sort()
    p50 = samples_ns[iterations // 2] / 1e6
    p95 = samples_ns[int(iterations * 0.95)] / 1e6
    p99 = samples_ns[int(iterations * 0.99)] / 1e6

    print(f"\nlatency ms — p50={p50:.4f}  p95={p95:.4f}  p99={p99:.4f}")
    assert p95 < 0.5, f"p95={p95:.4f}ms exceeds 0.5ms budget"
