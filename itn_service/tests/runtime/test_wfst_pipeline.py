"""Unit tests for ``runtime.wfst_pipeline.WFSTPipeline``.

Exercises construction, FAR loading, the ``normalize_span`` /
``classify_span`` API surface, and the contracts the pipeline upholds
on top of the underlying grammars.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from itn_service.runtime.wfst_pipeline import WFSTPipeline


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


def test_construction_loads_far_once(pipeline: WFSTPipeline) -> None:
    # Smoke: FAR exists and supported_classes is non-empty.
    assert pipeline.lang == "hi"
    assert "cardinal" in pipeline.supported_classes
    assert "decimal" in pipeline.supported_classes


def test_unknown_language_raises() -> None:
    with pytest.raises(FileNotFoundError):
        WFSTPipeline("xx")


def test_explicit_far_root_works(tmp_path: Path) -> None:
    # Pointing at a directory without the FAR should also raise — the
    # error message must mention the missing path so the operator can
    # build it.
    with pytest.raises(FileNotFoundError) as exc:
        WFSTPipeline("hi", far_root=tmp_path)
    assert "compile" in str(exc.value)


def test_unknown_class_raises(pipeline: WFSTPipeline) -> None:
    with pytest.raises(ValueError):
        pipeline.normalize_span("एक", "not_a_class")


def test_empty_input_returns_none(pipeline: WFSTPipeline) -> None:
    assert pipeline.normalize_span("", "cardinal") is None
    assert pipeline.normalize_span("", "decimal") is None


def test_unparseable_returns_none(pipeline: WFSTPipeline) -> None:
    # Random Hindi greetings / decimal-ish nonsense never parse.
    assert pipeline.normalize_span("नमस्ते", "cardinal") is None
    assert pipeline.normalize_span("नमस्ते दोस्तों", "decimal") is None


def test_cardinal_passes_latin(pipeline: WFSTPipeline) -> None:
    assert pipeline.normalize_span("12500", "cardinal") == "12500"


def test_decimal_passes_latin(pipeline: WFSTPipeline) -> None:
    assert pipeline.normalize_span("12.5", "decimal") == "12.5"


def test_classify_emits_nemo_tag(pipeline: WFSTPipeline) -> None:
    assert (
        pipeline.classify_span("एक सौ पच्चीस", "cardinal")
        == 'cardinal { value: "125" }'
    )
    assert (
        pipeline.classify_span("बारह दशमलव पाँच", "decimal")
        == 'decimal { value: "12.5" }'
    )


def test_classify_returns_none_when_no_match(pipeline: WFSTPipeline) -> None:
    assert pipeline.classify_span("नमस्ते", "cardinal") is None


def test_classify_unknown_class_returns_none(pipeline: WFSTPipeline) -> None:
    # Non-fatal here because the bare ``normalize_span`` already
    # validates; this method is the looser "best-effort" surface.
    assert pipeline.classify_span("एक", "money") is None


# --- the streaming invariant: no graph build in the request path -----------


def test_normalize_span_does_not_construct_fst(pipeline: WFSTPipeline) -> None:
    """Repeated calls must not re-import grammar modules.

    Concrete check: the FST objects pinned on the pipeline are the same
    Python objects across calls. A regression that re-loads the FAR
    per request would replace them on each call.
    """
    fst_before = pipeline._bare["cardinal"]            # noqa: SLF001
    pipeline.normalize_span("एक सौ पच्चीस", "cardinal")
    pipeline.normalize_span("नमस्ते", "cardinal")       # also exercises the no-match path
    fst_after = pipeline._bare["cardinal"]             # noqa: SLF001
    assert fst_before is fst_after
