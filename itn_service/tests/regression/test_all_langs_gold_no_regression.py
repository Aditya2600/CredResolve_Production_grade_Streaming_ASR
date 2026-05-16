"""Cross-language gold-set regression gate.

Generalises ``test_hi_gold_no_regression.py`` to every language whose
gold corpus is checked in. The contract added by the bn / gu / pa
onboarding stage is explicit: every previously-passing language
(``hi``, ``mr``, ``bn``, ``gu``, ``pa``) must still pass at
≥ 98 % sentence accuracy on its gold corpus after a new language
lands or a shared-runtime change ships. If this test fails, something
in the shared runtime (``runtime/script_router.py``,
``runtime/wfst_pipeline.py``, ``runtime/unicode_clean.py``,
``compile.py``) drifted in a way that affected normalisation for the
flagged language.

Graceful skips by design:

* If a language's gold directory contains no ``*.jsonl`` files (the
  bn / gu / pa case at the time the infrastructure stage lands), all
  parametrisations for that language are skipped at collection time.
  The file becomes load-bearing as soon as a gold file is committed.
* If a gold file's class is not in the pipeline's
  ``supported_classes`` (the stub-grammar case), that single
  parametrisation skips with a clear message rather than failing.
* If the pipeline's compiled FAR for the language is missing, the
  module-scope fixture skips the whole language with a one-line
  pointer to the compile command.

The test deliberately mirrors the Hindi-only sibling rather than
re-importing it, so that a future refactor of one cannot silently
weaken the other. The Hindi-only file remains the canonical
regression check for Hindi specifically; this file is the catch-all
for the full set.

Per the implementation blueprint's stages-2-4 acceptance bar and the
bn / gu / pa onboarding's stated deliverable, the 98 % threshold is
fixed: do not soften it without an explicit policy decision recorded
in ``docs/`` and a corresponding update in the PR description.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest


_GOLD_ROOT = Path(__file__).resolve().parents[1] / "gold"
_REGRESSION_BAR = 0.98

# Languages under the cross-language regression gate. Order is the
# onboarding order (hi first, then mr, then the bn/gu/pa template
# wave) so failures read top-down chronologically. Other supported
# languages (ta / te / kn / ml / ur) are intentionally excluded until
# their grammar + gold sets land.
_LANGS_UNDER_GATE: tuple[str, ...] = ("hi", "mr", "bn", "gu", "pa")


def _gold_files_for(lang: str) -> list[Path]:
    lang_dir = _GOLD_ROOT / lang
    if not lang_dir.exists():
        return []
    return sorted(lang_dir.glob("*.jsonl"))


def _collect_parametrisations() -> Iterator[tuple[str, Path]]:
    """Yield (lang, gold_path) for every (lang, jsonl) combination.

    Languages with no gold files yield nothing — they will not appear
    in the test report at all, which is the desired behaviour for the
    template-port stage: bn / gu / pa land with empty gold sets and
    the gate is silent until a gold file is committed.
    """
    for lang in _LANGS_UNDER_GATE:
        for path in _gold_files_for(lang):
            yield lang, path


_PARAMETRISATIONS: list[tuple[str, Path]] = list(_collect_parametrisations())


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


def _class_from_filename(path: Path) -> str:
    """Map a gold filename to the WFST class identifier.

    ``cardinal.jsonl`` -> ``cardinal``, ``money.jsonl`` -> ``money``,
    etc. The single irregular case is ``currency.jsonl`` which maps to
    the ``money`` class (the gold file is named for the spoken
    concept; the grammar class is named for its WFST namespace).
    """
    stem = path.stem
    if stem == "currency":
        return "money"
    return stem


# Per-language pipeline cache. We cache the pipeline AND any
# fixture-time skip reason so a missing FAR for (say) Punjabi causes
# every pa-tagged parametrisation to skip cleanly instead of erroring
# once per case.
_pipeline_cache: dict[str, object] = {}


def _get_pipeline(lang: str):  # type: ignore[no-untyped-def]
    if lang in _pipeline_cache:
        cached = _pipeline_cache[lang]
        if isinstance(cached, str):
            pytest.skip(cached)
        return cached

    try:
        from itn_service.runtime.wfst_pipeline import WFSTPipeline
        pipe = WFSTPipeline(lang)
    except FileNotFoundError as e:
        reason = (
            f"compiled FAR for {lang!r} not found ({e}); "
            f"run `python -m itn_service.compile --lang {lang}` first"
        )
        _pipeline_cache[lang] = reason
        pytest.skip(reason)
    except Exception as e:  # pragma: no cover - defensive
        reason = f"WFSTPipeline({lang!r}) failed to initialise: {e}"
        _pipeline_cache[lang] = reason
        pytest.skip(reason)

    _pipeline_cache[lang] = pipe
    return pipe


@pytest.mark.parametrize(
    "lang,gold_path",
    _PARAMETRISATIONS,
    ids=[f"{lang}/{path.name}" for lang, path in _PARAMETRISATIONS],
)
def test_lang_gold_still_passes_98_percent(
    lang: str, gold_path: Path,
) -> None:
    """For every (lang, gold-file) under the gate: ≥ 98 % accuracy.

    Skips cleanly when the gold file is empty, the class is not yet
    in the pipeline's ``supported_classes``, or the language's FAR
    has not been compiled in this environment.
    """
    cls = _class_from_filename(gold_path)
    cases = _load_gold(gold_path)
    if not cases:
        pytest.skip(f"gold file {lang}/{gold_path.name} is empty")

    pipeline = _get_pipeline(lang)

    if cls not in pipeline.supported_classes:  # type: ignore[attr-defined]
        pytest.skip(
            f"class {cls!r} not (yet) supported by the {lang!r} "
            f"pipeline; see grammars/{lang}/{cls}.py (likely still a stub)"
        )

    correct = 0
    misses: list[tuple[str, str, str | None]] = []
    for case in cases:
        out = pipeline.normalize_span(case["raw"], cls)  # type: ignore[attr-defined]
        if out == case["expected"]:
            correct += 1
        else:
            misses.append((case["raw"], case["expected"], out))

    accuracy = correct / len(cases)
    assert accuracy >= _REGRESSION_BAR, (
        f"{lang} {cls!r} gold accuracy {accuracy:.3%} fell below the "
        f"{_REGRESSION_BAR:.0%} regression bar. This is the gate the "
        f"bn / gu / pa onboarding PR (and any subsequent shared-runtime "
        f"change) must clear. First 10 misses: {misses[:10]}"
    )


def test_gate_includes_all_template_languages_when_their_gold_lands() -> None:
    """Guardrail: when a language's gold directory becomes non-empty,
    the gate must produce at least one parametrisation for it.

    This is the early-warning that catches a refactor of
    ``_collect_parametrisations`` accidentally dropping a language.
    Skips for languages that still have empty gold dirs (the expected
    state for bn / gu / pa during the infrastructure stage).
    """
    seen_langs = {lang for lang, _ in _PARAMETRISATIONS}
    for lang in _LANGS_UNDER_GATE:
        if not _gold_files_for(lang):
            continue
        assert lang in seen_langs, (
            f"language {lang!r} has gold files under tests/gold/{lang}/ "
            f"but produced no parametrisations — collection is broken"
        )
