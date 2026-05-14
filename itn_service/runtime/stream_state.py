"""Streaming stability tracker.

Implements the ``StreamState`` from the streaming pseudo-code in
``docs/implementation_bluprint_INR.md`` (section: "End-to-end pipeline").
The contract is intentionally tiny: count how many consecutive partial
hypotheses have arrived with identical text, so that the
:func:`runtime.normalizer.normalize_segment` confidence gate can decide
whether a partial is stable enough to render.

A new ``StreamState`` is created **per call** (i.e. per ASR session).
The object is *not* thread-safe; the bidi gRPC handler that owns the
stream is the only writer (see ``service/grpc_server.py``).

The default stability threshold of **2 consecutive identical partials**
matches ``configs/thresholds.yaml § streaming.partial_stable_min`` and
the blueprint's confidence-gating section ("only rewrite spans that
have been stable for at least two consecutive partial hypotheses").
"""

from __future__ import annotations

DEFAULT_STABILITY_THRESHOLD: int = 2


class StreamState:
    """Per-call partial-hypothesis stability counter.

    The state machine has two observables:

    * ``last_partial`` — the most recently observed partial text.
    * ``stable_count`` — how many consecutive observations of that
      exact text have been seen, **counting the latest one**. A fresh
      object reports zero until the first update arrives.

    ``update_stability(text)`` is the only mutator. It returns the
    new ``stable_count`` so callers can branch on the value without
    a second attribute read.
    """

    __slots__ = ("last_partial", "stable_count", "_threshold")

    def __init__(self, *, stability_threshold: int = DEFAULT_STABILITY_THRESHOLD) -> None:
        if stability_threshold < 1:
            raise ValueError(
                f"stability_threshold must be >= 1, got {stability_threshold}"
            )
        self.last_partial: str = ""
        self.stable_count: int = 0
        self._threshold: int = stability_threshold

    @property
    def stability_threshold(self) -> int:
        """Number of consecutive identical partials required to call a span stable."""
        return self._threshold

    def update_stability(self, partial_text: str) -> int:
        """Record one partial observation and return the updated count.

        Matches the blueprint pseudo-code exactly: identical text
        increments the count, any change resets to 1 and updates the
        anchor. The first call on a fresh state therefore returns 1,
        not 0.
        """
        if partial_text == self.last_partial and self.stable_count > 0:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_partial = partial_text
        return self.stable_count

    def is_stable(self) -> bool:
        """True once :attr:`stable_count` has reached the threshold."""
        return self.stable_count >= self._threshold

    def reset(self) -> None:
        """Clear the anchor and count; called on segment-final boundaries.

        After a final, the next partial starts a new segment, so the
        stability counter must not carry over.
        """
        self.last_partial = ""
        self.stable_count = 0


__all__ = ["DEFAULT_STABILITY_THRESHOLD", "StreamState"]
