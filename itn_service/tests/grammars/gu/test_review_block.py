"""PR-merge block on the Gujarati gold corpus.

The Gujarati gold sets in ``tests/gold/gu/`` were drafted by an
automated template-port from Hindi / Marathi and must be reviewed by
a native Gujarati speaker before they serve as the acceptance truth
source for the Gujarati grammar. While the marker file
``REVIEW_REQUIRED.md`` exists in ``tests/gold/gu/``, this test fails
— any CI that gates merge on the test suite will refuse to merge
until a reviewer signs off and deletes the marker.

Rationale: the Gujarati grammar topology is a direct port of
Hindi/Marathi's (by design — the stage exists to validate that the
per-language template is reusable across scripts). The lexical
changes are the only place a Gujarati-specific error can land, and
every gold entry inherits whatever spelling decision the drafter
made. Catching a lexical mistake pre-merge — especially around lakh
/ crore vocabulary and Indian grouping invariants — is much cheaper
than catching it after the gold corpus has become the truth source
for downstream tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REVIEW_MARKER = (
    Path(__file__).resolve().parents[2]
    / "gold"
    / "gu"
    / "REVIEW_REQUIRED.md"
)


def test_no_gujarati_review_required_marker() -> None:
    if _REVIEW_MARKER.exists():
        pytest.fail(
            "Gujarati gold-set review is pending. "
            f"Remove {_REVIEW_MARKER.relative_to(_REVIEW_MARKER.parents[3])} "
            "after a native Gujarati speaker has reviewed every file in "
            "tests/gold/gu/ and signed off in the marker. See the file "
            "itself for the reviewer checklist (note the lakh/crore "
            "vocabulary and Indian-grouping invariant)."
        )
