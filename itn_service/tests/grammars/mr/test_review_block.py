"""PR-merge block on the Marathi gold corpus.

The Marathi gold sets in ``tests/gold/mr/`` were drafted by an
automated template-port from Hindi and must be reviewed by a native
Marathi speaker before they serve as the acceptance truth source for
the Marathi grammar. While the marker file ``REVIEW_REQUIRED.md``
exists in ``tests/gold/mr/``, this test fails — any CI that gates
merge on the test suite will refuse to merge until a reviewer signs
off and deletes the marker.

Rationale: the Marathi grammar topology is a direct port of Hindi's
(by design — the stage exists to validate that the per-language
template from stages 2-4 is reusable). The lexical changes are the
only place a Marathi-specific error can land, and every gold entry
inherits whatever spelling decision the drafter made. Catching a
lexical mistake pre-merge is much cheaper than catching it after the
gold corpus has become the truth source for downstream tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REVIEW_MARKER = (
    Path(__file__).resolve().parents[2]
    / "gold"
    / "mr"
    / "REVIEW_REQUIRED.md"
)


def test_no_marathi_review_required_marker() -> None:
    if _REVIEW_MARKER.exists():
        pytest.fail(
            "Marathi gold-set review is pending. "
            f"Remove {_REVIEW_MARKER.relative_to(_REVIEW_MARKER.parents[3])} "
            "after a native Marathi speaker has reviewed every file in "
            "tests/gold/mr/ and signed off in the marker. See the file "
            "itself for the reviewer checklist."
        )
