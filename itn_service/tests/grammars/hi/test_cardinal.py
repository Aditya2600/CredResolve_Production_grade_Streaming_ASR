"""Gold-set tests for the Hindi cardinal grammar.

Reads ``tests/gold/hi/cardinal.jsonl`` (one ``{"raw": ..., "expected":
...}`` per line) and asserts that the WFST pipeline produces the
expected canonical Latin integer for every entry. Two acceptance
contracts come out of this test module:

* Per-case correctness — every gold case must round-trip.
* Sentence-level accuracy — the plan caps acceptable failure at 2 %
  (i.e. ≥ 98 % accuracy on the gold set).

The gold set is required to have ≥ 200 entries and to include every
half/quarter compound family explicitly (सवा / डेढ़ / ढाई / साढ़े /
पौने), as called out in the implementation blueprint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline


_GOLD_PATH = (
    Path(__file__).resolve().parents[2] / "gold" / "hi" / "cardinal.jsonl"
)


def _load_gold() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with _GOLD_PATH.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "raw" in obj and "expected" in obj, (
                f"malformed gold entry: {line!r}"
            )
            items.append(obj)
    return items


_GOLD: list[dict[str, str]] = _load_gold()


# --- module-scoped fixture: load FAR exactly once ---------------------------


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


# --- gold corpus shape ------------------------------------------------------


def test_gold_set_has_minimum_size() -> None:
    """Plan: at least 200 cases."""
    assert len(_GOLD) >= 200, (
        f"gold set has {len(_GOLD)} cases, want ≥ 200 — see "
        f"docs/implementation_bluprint_INR.md § 'Implementation blueprint'"
    )


def test_gold_set_includes_each_compound_family() -> None:
    """Plan: each of सवा / डेढ़ / ढाई / साढ़े / पौने must appear."""
    raws = [c["raw"] for c in _GOLD]
    for prefix in ("सवा", "डेढ़", "ढाई", "साढ़े", "पौने"):
        assert any(prefix in r for r in raws), (
            f"gold set missing {prefix} compounds; "
            f"the plan requires explicit coverage"
        )


# --- per-case correctness ---------------------------------------------------

# Parametrise by *index* rather than the raw string so the test ID stays
# readable when the entry is a long compound.
@pytest.mark.parametrize(
    "case",
    _GOLD,
    ids=[f"{i:03d}:{c['expected']}" for i, c in enumerate(_GOLD)],
)
def test_cardinal_normalises(pipeline: WFSTPipeline, case: dict[str, str]) -> None:
    raw = case["raw"]
    expected = case["expected"]
    out = pipeline.normalize_span(raw, "cardinal")
    assert out == expected, (
        f"cardinal normalisation mismatch:\n"
        f"  raw      = {raw!r}\n"
        f"  expected = {expected!r}\n"
        f"  got      = {out!r}"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("टू थाउजेंड", "2000"),
        ("टू थाउजंड", "2000"),
        ("वन थाउजेंड", "1000"),
        ("फाइव हंड्रेड", "500"),
        ("टू थाउजेंड फाइव हंड्रेड", "2500"),
        ("टू हजार", "2000"),
        ("दो थाउजेंड", "2000"),
    ],
)
def test_cardinal_normalises_devanagari_english_code_switch_aliases(
    pipeline: WFSTPipeline,
    raw: str,
    expected: str,
) -> None:
    assert pipeline.normalize_span(raw, "cardinal") == expected


# --- sentence accuracy ------------------------------------------------------


def test_sentence_accuracy_meets_98_percent(pipeline: WFSTPipeline) -> None:
    """Sentence-level accuracy on the gold corpus must be ≥ 98 %."""
    correct = 0
    total = len(_GOLD)
    misses: list[tuple[str, str, str | None]] = []
    for case in _GOLD:
        out = pipeline.normalize_span(case["raw"], "cardinal")
        if out == case["expected"]:
            correct += 1
        else:
            misses.append((case["raw"], case["expected"], out))

    accuracy = correct / total
    assert accuracy >= 0.98, (
        f"sentence accuracy {accuracy:.3%} below 98% bar.\n"
        f"misses (first 10): {misses[:10]}"
    )
