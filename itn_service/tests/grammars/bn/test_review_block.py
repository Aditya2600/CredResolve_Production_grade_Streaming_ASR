"""PR-merge block on the Bengali gold corpus.

The Bengali gold sets in ``tests/gold/bn/`` were drafted by an
automated template-port from Hindi / Marathi and must be reviewed by
a native Bengali speaker before they serve as the acceptance truth
source for the Bengali grammar. While the marker file
``REVIEW_REQUIRED.md`` exists in ``tests/gold/bn/``, this test fails
— any CI that gates merge on the test suite will refuse to merge
until a reviewer signs off and deletes the marker.

Rationale: the Bengali grammar topology is a direct port of
Hindi/Marathi's (by design — the stage exists to validate that the
per-language template is reusable across scripts). The lexical
changes are the only place a Bengali-specific error can land, and
every gold entry inherits whatever spelling / cue-policy decision
the drafter made. Catching a lexical mistake pre-merge — including
the bn-IN vs bn-BD currency-cue ambiguity called out in the marker
file — is much cheaper than catching it after the gold corpus has
become the truth source for downstream tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REVIEW_MARKER = (
    Path(__file__).resolve().parents[2]
    / "gold"
    / "bn"
    / "REVIEW_REQUIRED.md"
)


def test_no_bengali_review_required_marker() -> None:
    if _REVIEW_MARKER.exists():
        pytest.fail(
            "Bengali gold-set review is pending. "
            f"Remove {_REVIEW_MARKER.relative_to(_REVIEW_MARKER.parents[3])} "
            "after a native Bengali speaker has reviewed every file in "
            "tests/gold/bn/ and signed off in the marker. See the file "
            "itself for the reviewer checklist (note the bn-IN vs bn-BD "
            "currency-cue policy split)."
        )
