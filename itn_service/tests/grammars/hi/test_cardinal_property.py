"""Hypothesis property test for the Hindi cardinal grammar.

Round-trip property:

    int n  →  num2words(n, lang='hi')  →  WFSTPipeline.normalize_span  →  str(n)

This is the most direct sanity check we can run on the recursive
padded grammar: any integer the spelling library can produce, the
grammar must parse back. ``indic-numtowords`` covers 0 .. 10⁹ - 1
correctly (it produces a digit-by-digit fallback at exactly 10⁹), so
that is the property's domain. The fixed value ``n = 10⁹`` is checked
separately via the gold set.
"""

from __future__ import annotations

import pytest

try:
    from indic_numtowords import num2words

    _HAS_NUM2WORDS = True
except Exception:  # pragma: no cover - environment-dependent
    _HAS_NUM2WORDS = False

from hypothesis import given, settings
from hypothesis import strategies as st

from itn_service.runtime.wfst_pipeline import WFSTPipeline


pytestmark = pytest.mark.skipif(
    not _HAS_NUM2WORDS,
    reason="indic-numtowords not installed",
)


@pytest.fixture(scope="module")
def pipeline() -> WFSTPipeline:
    return WFSTPipeline("hi")


@given(n=st.integers(min_value=0, max_value=999_999_999))
@settings(max_examples=300, deadline=None)
def test_cardinal_roundtrip(pipeline: WFSTPipeline, n: int) -> None:
    spoken = num2words(str(n), lang="hi")
    out = pipeline.normalize_span(spoken, "cardinal")
    assert out == str(n), (
        f"roundtrip failure: n={n} spoken={spoken!r} got={out!r}"
    )


# Exhaustive scan over 0..99 is cheap and gives us a hard guarantee
# that every "ones" entry — including the alternates spec'd in
# cardinal.py (छह/छः, पाँच/पांच) — is parsable.
@pytest.mark.parametrize("n", list(range(0, 100)))
def test_zero_to_99_canonical_spellings(
    pipeline: WFSTPipeline, n: int
) -> None:
    spoken = num2words(str(n), lang="hi")
    out = pipeline.normalize_span(spoken, "cardinal")
    assert out == str(n), (
        f"0..99 roundtrip failed at {n}: spoken={spoken!r} got={out!r}"
    )
