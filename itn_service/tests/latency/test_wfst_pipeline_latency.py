"""p95 latency benchmark for ``WFSTPipeline.normalize_span``.

Per the implementation blueprint:

    > Latency: p95 < 5 ms per span on a single CPU core, FAR loaded once.

The benchmark warms cache, then measures 5 000 iterations across a
representative input mix that exercises every recursive scale layer
(units, hundreds, thousands, lakhs, crores, अरब) plus the half /
quarter compounds. Marked ``slow`` so casual ``pytest`` runs from a
developer laptop don't pay the cost; CI runs ``pytest -m slow``.
"""

from __future__ import annotations

import time

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline


# Inputs span the full range of grammar paths so the benchmark
# exercises the worst-case pipeline cost, not just the easy ones.
_CARDINAL_INPUTS = [
    "एक",
    "पच्चीस",
    "एक सौ पच्चीस",
    "एक हज़ार पाँच सौ",
    "एक लाख पच्चीस हज़ार",
    "निन्यानवे करोड़ निन्यानवे लाख निन्यानवे हज़ार नौ सौ निन्यानवे",
    "एक अरब",
    "सवा हज़ार",
    "पौने चार हज़ार",
    "साढ़े पाँच लाख",
]

_DECIMAL_INPUTS = [
    "बारह दशमलव पाँच",
    "शून्य दशमलव एक दो पाँच",
    "एक सौ दशमलव पाँच",
    "डेढ़",
    "ढाई",
    "साढ़े पाँच",
    "पौने चार",
    "सवा बारह",
    "12.5",
    "0.125",
]


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


def _bench(pipeline: WFSTPipeline, inputs: list[str], cls: str) -> dict[str, float]:
    # Warm — pull the FAR + composition machinery into cache.
    for _ in range(200):
        for s in inputs:
            pipeline.normalize_span(s, cls)

    iterations = 5000
    samples_ns: list[int] = []
    for i in range(iterations):
        s = inputs[i % len(inputs)]
        t0 = time.perf_counter_ns()
        pipeline.normalize_span(s, cls)
        samples_ns.append(time.perf_counter_ns() - t0)
    samples_ns.sort()
    return {
        "p50": samples_ns[iterations // 2] / 1e6,
        "p95": samples_ns[int(iterations * 0.95)] / 1e6,
        "p99": samples_ns[int(iterations * 0.99)] / 1e6,
    }


@pytest.mark.slow
def test_cardinal_p95_under_5ms(pipeline: WFSTPipeline) -> None:
    stats = _bench(pipeline, _CARDINAL_INPUTS, "cardinal")
    print(
        f"\ncardinal latency ms — "
        f"p50={stats['p50']:.4f}  p95={stats['p95']:.4f}  p99={stats['p99']:.4f}"
    )
    assert stats["p95"] < 5.0, (
        f"cardinal p95={stats['p95']:.4f}ms exceeds 5ms budget "
        f"(blueprint § 'Latency')"
    )


@pytest.mark.slow
def test_decimal_p95_under_5ms(pipeline: WFSTPipeline) -> None:
    stats = _bench(pipeline, _DECIMAL_INPUTS, "decimal")
    print(
        f"\ndecimal latency ms — "
        f"p50={stats['p50']:.4f}  p95={stats['p95']:.4f}  p99={stats['p99']:.4f}"
    )
    assert stats["p95"] < 5.0, (
        f"decimal p95={stats['p95']:.4f}ms exceeds 5ms budget "
        f"(blueprint § 'Latency')"
    )
