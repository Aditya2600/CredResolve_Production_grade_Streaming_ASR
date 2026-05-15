"""Tests for the lazy WFST pipeline factory."""

from __future__ import annotations

import sys
from types import ModuleType

from itn_service.runtime.wfst_factory import get_pipeline


def test_get_pipeline_caches_successful_construction(monkeypatch) -> None:
    constructed: list[str] = []

    class _FakePipeline:
        def __init__(self, lang: str) -> None:
            constructed.append(lang)

    fake_module = ModuleType("itn_service.runtime.wfst_pipeline")
    fake_module.WFSTPipeline = _FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "itn_service.runtime.wfst_pipeline", fake_module)
    get_pipeline.cache_clear()

    first = get_pipeline("hi")
    second = get_pipeline("hi")

    assert first is second
    assert constructed == ["hi"]
    get_pipeline.cache_clear()


def test_get_pipeline_returns_none_when_far_is_missing(monkeypatch) -> None:
    class _MissingFarPipeline:
        def __init__(self, lang: str) -> None:
            raise FileNotFoundError(lang)

    fake_module = ModuleType("itn_service.runtime.wfst_pipeline")
    fake_module.WFSTPipeline = _MissingFarPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "itn_service.runtime.wfst_pipeline", fake_module)
    get_pipeline.cache_clear()

    assert get_pipeline("hi") is None
    get_pipeline.cache_clear()
