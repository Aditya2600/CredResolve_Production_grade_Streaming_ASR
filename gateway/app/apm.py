"""Audio framing helpers for the gateway pipeline.

The gateway frames PCM into fixed-size 10 ms windows before forwarding to
the VAD. ``NoOpAudioProcessor`` performs the framing without altering
samples. See ``docs/audio/apm-decision.md`` for why no real audio
processing module runs server-side.
"""
from __future__ import annotations

from typing import Protocol

FRAME_MS = 10
SAMPLE_RATE = 16000
FRAME_BYTES = int(SAMPLE_RATE * (FRAME_MS / 1000.0) * 2)


class AudioProcessor(Protocol):
    frame_bytes: int

    def process_frame(self, frame: bytes) -> bytes: ...

    def reset(self) -> None: ...


class NoOpAudioProcessor:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"NoOpAudioProcessor.sample_rate must be {SAMPLE_RATE}")
        self.sample_rate = sample_rate
        self.frame_bytes = FRAME_BYTES

    def process_frame(self, frame: bytes) -> bytes:
        if len(frame) != self.frame_bytes:
            raise ValueError(
                f"expected {self.frame_bytes} bytes for audio frame, got {len(frame)}"
            )
        return frame

    def reset(self) -> None:
        return None
