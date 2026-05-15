"""Lazy construction for per-language WFST pipelines.

The live request path should be able to import the ITN runtime even when the
optional native WFST stack or compiled FAR artefacts are not present yet. This
module keeps that edge cold: callers ask for a language pipeline on demand, and
missing dependencies / artefacts collapse to ``None`` rather than breaking
regex-only unit tests.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wfst_pipeline import WFSTPipeline


@cache
def get_pipeline(lang: str) -> "WFSTPipeline | None":
    """Return the cached WFST pipeline for ``lang`` when available.

    ``pynini`` is imported by :mod:`runtime.wfst_pipeline`, so that import stays
    inside this function. The factory intentionally treats a missing native
    dependency, absent FAR, or incomplete FAR as "pipeline unavailable" for the
    language; the caller can then emit raw text while keeping the rest of the
    process healthy.
    """
    try:
        from .wfst_pipeline import WFSTPipeline

        return WFSTPipeline(lang)
    except (ImportError, FileNotFoundError, KeyError):
        return None


__all__ = ["get_pipeline"]
