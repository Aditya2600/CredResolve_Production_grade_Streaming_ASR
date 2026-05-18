"""Self-correction detection for normalisation spans.

Speakers correct themselves in patterns like::

    "9876543210, no sorry, 9876543211"
    "मेरा खाता ABC123 — गलत, मतलब ABC1234"
    "the amount is 500 actually 5000 rupees"

When a span A is followed within ``window`` tokens by a self-correction
marker plus another span B of the same class, we mark BOTH spans as
unsafe. The downstream normaliser then emits raw text for both so a
human reviewer (agent UI / QA) can disambiguate. The system never
silently picks one over the other.

This is deliberately conservative: any uncertainty defers to raw, per
the stage-2 deliverable contract.

Marker vocabulary (case-insensitive):

    English single tokens:   no, sorry, actually, wait, oops
    English phrases:         "i mean", "no sorry", "no no"
    Hindi single tokens:     मतलब, गलत, नहीं, माफ़, माफ, क्षमा, अरे
    Hindi phrases:           "मेरा मतलब", "गलत है"
"""

from __future__ import annotations

import unicodedata
from typing import Sequence

from .contract import Span


_MARKERS_SINGLE: frozenset[str] = frozenset({
    # English
    "no",
    "sorry",
    "actually",
    "wait",
    "oops",
    "scratch",
    # Hindi
    "मतलब",
    "गलत",
    "ग़लत",
    "नहीं",
    "नही",
    "माफ़",
    "माफ",
    "क्षमा",
    "अरे",
})

_MARKERS_MULTI: tuple[str, ...] = (
    "i mean",
    "no sorry",
    "no no",
    "scratch that",
    "मेरा मतलब",
    "गलत है",
    "गलती हो",
)

def _tokenise(text: str) -> list[str]:
    """Whitespace-split + light leading/trailing punctuation trim."""
    out: list[str] = []
    for raw in text.split():
        t = _strip_edge_punctuation(raw)
        if t:
            out.append(t)
    return out


def _strip_edge_punctuation(token: str) -> str:
    """Trim punctuation without discarding Indic combining marks."""
    start = 0
    end = len(token)
    while start < end and unicodedata.category(token[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(token[end - 1]).startswith("P"):
        end -= 1
    return token[start:end]


def _gap_has_marker(text: str, max_tokens: int) -> bool:
    """``True`` iff the gap text is a non-empty token sequence of length
    ``<= max_tokens`` and contains at least one self-correction marker.
    """
    toks = _tokenise(text)
    if not toks or len(toks) > max_tokens:
        return False
    lower = [t.lower() for t in toks]
    if any(t in _MARKERS_SINGLE for t in lower):
        return True
    joined = " ".join(lower)
    return any(p in joined for p in _MARKERS_MULTI)


def detect_self_corrections(
    text: str,
    spans: Sequence[Span],
    *,
    window: int = 6,
) -> set[int]:
    """Return indices of ``spans`` that should be emitted as raw because
    they participate in a self-correction pattern.

    A span at index ``i`` is unsafe iff there exists ``j > i`` with:

        * ``spans[i].cls == spans[j].cls``,
        * both spans carry codepoint offsets (``end`` / ``start``),
        * the text between them is non-empty and at most ``window``
          tokens long, and
        * that gap contains at least one self-correction marker.

    Both ``i`` and ``j`` are added to the returned set. The function
    only pairs each earlier span with the *first* later same-class span
    it can correct toward; chained corrections (A -> B -> C) all end up
    flagged because each adjacent pair contributes its two indices.
    """
    if len(spans) < 2:
        return set()
    unsafe: set[int] = set()
    for i in range(len(spans)):
        ai = spans[i]
        if ai.end is None:
            continue
        for j in range(i + 1, len(spans)):
            aj = spans[j]
            if aj.cls != ai.cls or aj.start is None:
                continue
            if aj.start <= ai.end:
                continue  # overlap / out-of-order — skip silently
            if _gap_has_marker(text[ai.end : aj.start], window):
                unsafe.add(i)
                unsafe.add(j)
                break
    return unsafe


__all__ = ["detect_self_corrections"]
