"""PR-merge block on the Punjabi gold corpus.

The Punjabi gold sets in ``tests/gold/pa/`` were drafted by an
automated template-port from Hindi / Marathi and must be reviewed by
a native Punjabi speaker before they serve as the acceptance truth
source for the Punjabi grammar. While the marker file
``REVIEW_REQUIRED.md`` exists in ``tests/gold/pa/``, this test fails
— any CI that gates merge on the test suite will refuse to merge
until a reviewer signs off and deletes the marker.

Rationale: the Punjabi grammar topology is a direct port of
Hindi/Marathi's (by design — the stage exists to validate that the
per-language template is reusable across scripts). Two
Punjabi-specific concerns must be reviewed in addition to the
ordinary lexical-correctness check:

1. The Gurmukhi bindi-fold in ``runtime/unicode_clean.py`` collapses
   five letter pairs (ਫ਼/ਜ਼/ਗ਼/ਖ਼/ਸ਼ -> ਫ/ਜ/ਗ/ਖ/ਸ). Reviewers must
   confirm no in-vocabulary lemma depends on a bindi distinction the
   fold collapses, and flag any false-merge in the marker file.
2. The half/quarter compound family (ਸਵਾ / ਡੇਢ / ਢਾਈ / ਸਾਢੇ / ਪੌਣੇ)
   diverges from the Hindi stems even though many other Punjabi
   numerals share Hindi's spelling — the gold set must exercise the
   divergence.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REVIEW_MARKER = (
    Path(__file__).resolve().parents[2]
    / "gold"
    / "pa"
    / "REVIEW_REQUIRED.md"
)


def test_no_punjabi_review_required_marker() -> None:
    if _REVIEW_MARKER.exists():
        pytest.fail(
            "Punjabi gold-set review is pending. "
            f"Remove {_REVIEW_MARKER.relative_to(_REVIEW_MARKER.parents[3])} "
            "after a native Punjabi speaker has reviewed every file in "
            "tests/gold/pa/ and signed off in the marker. See the file "
            "itself for the reviewer checklist (note the bindi-fold and "
            "the half/quarter compound family)."
        )
