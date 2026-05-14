"""Hindi gold-set regression gate.

The Marathi addition (stage 5+) is a per-language extension, not a
shared-code change. The contract for this stage is explicit: the
Hindi gold set must still pass at ≥ 98 % sentence accuracy after
Marathi is added. If this test fails, something in the shared
runtime (``runtime/script_router.py``, ``runtime/wfst_pipeline.py``,
``compile.py``) drifted in a way that affected Hindi normalisation.

The test deliberately duplicates the sentence-accuracy assertion
from ``tests/grammars/hi/test_cardinal.py`` rather than re-importing
it, so that a future refactor of the Hindi gold-set test cannot
silently weaken this regression guarantee. The Hindi-side test is
the *correctness* check; this one is the *non-regression* check.

Per the implementation blueprint's stages-2-4 acceptance bar and the
Marathi addition's stated deliverable, the 98 % threshold is fixed:
do not soften it without an explicit policy decision recorded in
``docs/`` and a corresponding update in the Marathi PR description.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_HI_GOLD_DIR = Path(__file__).resolve().parents[1] / "gold" / "hi"
_REGRESSION_BAR = 0.98


def _load_gold(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "raw" in obj and "expected" in obj, (
                f"malformed gold entry in {path}: {line!r}"
            )
            items.append(obj)
    return items


# Load every Hindi gold file present at collection time. Each class is
# parametrised separately so a regression in (say) ``date.jsonl`` does
# not mask one in ``cardinal.jsonl`` and vice-versa.
_GOLD_FILES: list[Path] = sorted(_HI_GOLD_DIR.glob("*.jsonl"))


@pytest.fixture(scope="module")
def pipeline():  # type: ignore[no-untyped-def]
    """Hindi WFSTPipeline. Imported lazily so the module's collection
    does not depend on the compiled FAR being present in environments
    that only want to run unit tests."""
    from itn_service.runtime.wfst_pipeline import WFSTPipeline

    return WFSTPipeline("hi")


def _class_from_filename(path: Path) -> str:
    """Map a gold filename to the WFST class identifier.

    Mirrors the per-class gold-file layout: ``cardinal.jsonl``,
    ``money.jsonl``, ``percent.jsonl``, ``date.jsonl``, ``time.jsonl``,
    ``decimal.jsonl``. The ``phone`` and ``id`` files map to classes
    that may not yet be present on the pipeline; the per-file test
    handles that by skipping gracefully.
    """
    return path.stem  # "cardinal.jsonl" -> "cardinal"


@pytest.mark.parametrize(
    "gold_path",
    _GOLD_FILES,
    ids=[p.name for p in _GOLD_FILES],
)
def test_hindi_gold_still_passes_98_percent(
    pipeline, gold_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    """After Marathi addition: Hindi gold accuracy must still be ≥ 98 %.

    Runs class-by-class so a fail message identifies the regressed
    class. ``phone`` / ``id`` gold files skip when the underlying
    Hindi grammar is still a stub (consistent with the staged
    rollout of those classes).
    """
    cls = _class_from_filename(gold_path)
    cases = _load_gold(gold_path)
    if not cases:
        pytest.skip(f"gold file {gold_path.name} is empty")

    if cls not in pipeline.supported_classes:
        pytest.skip(
            f"class {cls!r} not (yet) supported by the Hindi pipeline; "
            f"see grammars/hi/{cls}.py (likely still a stub)"
        )

    correct = 0
    misses: list[tuple[str, str, str | None]] = []
    for case in cases:
        out = pipeline.normalize_span(case["raw"], cls)
        if out == case["expected"]:
            correct += 1
        else:
            misses.append((case["raw"], case["expected"], out))

    accuracy = correct / len(cases)
    assert accuracy >= _REGRESSION_BAR, (
        f"Hindi {cls!r} gold accuracy {accuracy:.3%} fell below the "
        f"{_REGRESSION_BAR:.0%} regression bar after the Marathi addition. "
        f"This is the gate the Marathi PR must clear. First 10 misses: "
        f"{misses[:10]}"
    )
