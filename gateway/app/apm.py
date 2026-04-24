from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class APMConfig:
    enabled: bool = False
    sample_rate: int = 16000
    frame_ms: int = 10
    noise_suppression: bool = True
    agc: bool = True
    high_pass_filter: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError("APMConfig.sample_rate must be 16000")
        if self.frame_ms != 10:
            raise ValueError("APMConfig.frame_ms must be 10")

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0) * 2)


class AudioProcessor(Protocol):
    frame_bytes: int

    def process_frame(self, frame: bytes) -> bytes: ...

    def reset(self) -> None: ...


class WebRTCAPMBackend(Protocol):
    def process_frame(
        self,
        frame: bytes,
        *,
        sample_rate: int,
        noise_suppression: bool,
        agc: bool,
        high_pass_filter: bool,
    ) -> bytes: ...

    def reset(self) -> None: ...


class NoOpAudioProcessor:
    def __init__(self, config: APMConfig):
        self.config = config
        self.frame_bytes = config.frame_bytes

    def process_frame(self, frame: bytes) -> bytes:
        if len(frame) != self.frame_bytes:
            raise ValueError(
                f"expected {self.frame_bytes} bytes for APM frame, got {len(frame)}"
            )
        return frame

    def reset(self) -> None:
        return None


class WebRTCAudioProcessor:
    def __init__(self, config: APMConfig, backend: WebRTCAPMBackend):
        self.config = config
        self.backend = backend
        self.frame_bytes = config.frame_bytes

    def process_frame(self, frame: bytes) -> bytes:
        if len(frame) != self.frame_bytes:
            raise ValueError(
                f"expected {self.frame_bytes} bytes for APM frame, got {len(frame)}"
            )
        processed = self.backend.process_frame(
            frame,
            sample_rate=self.config.sample_rate,
            noise_suppression=self.config.noise_suppression,
            agc=self.config.agc,
            high_pass_filter=self.config.high_pass_filter,
        )
        if len(processed) != len(frame):
            raise ValueError(
                "APM backend must return the same number of bytes it receives"
            )
        return processed

    def reset(self) -> None:
        self.backend.reset()
