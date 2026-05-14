"""Gold-set tests for the Marathi cardinal grammar.

Mirror of ``tests/grammars/hi/test_cardinal.py``. Reads
``tests/gold/mr/cardinal.jsonl`` and asserts that the WFST pipeline
produces the expected canonical Latin integer for every entry. The
template-port acceptance bar is identical to Hindi:

* Per-case correctness — every gold case must round-trip.
* Sentence-level accuracy — must be ≥ 98 % on the gold corpus.
* Corpus shape — at least 200 entries, every half/quarter compound
  family present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline


_GOLD_PATH = (
    Path(__file__).resolve().parents[2] / "gold" / "mr" / "cardinal.jsonl"
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


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("mr")


def test_gold_set_has_minimum_size() -> None:
    assert len(_GOLD) >= 200, (
        f"gold set has {len(_GOLD)} cases, want ≥ 200 — see the Marathi "
        f"stage spec in the implementation blueprint"
    )


def test_gold_set_includes_each_compound_family() -> None:
    """Each Marathi half/quarter compound family must appear."""
    raws = [c["raw"] for c in _GOLD]
    for prefix in ("सव्वा", "दीड", "अडीच", "साडे", "पावणे"):
        assert any(prefix in r for r in raws), (
            f"gold set missing {prefix} compounds"
        )


def test_gold_set_includes_marathi_hundred_compounds() -> None:
    """The Marathi N-शे hundred-compound shape is the key structural
    difference from Hindi and must be exercised explicitly."""
    raws = [c["raw"] for c in _GOLD]
    for w in ("एकशे", "दोनशे", "पाचशे", "नऊशे"):
        assert any(w in r for r in raws), (
            f"gold set missing {w} — N-शे hundred shape required"
        )


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


def test_sentence_accuracy_meets_98_percent(pipeline: WFSTPipeline) -> None:
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
