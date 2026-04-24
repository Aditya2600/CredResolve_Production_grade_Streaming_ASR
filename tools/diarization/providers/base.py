from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

if False:  # pragma: no cover
    from tools.diarization.config import DiarizationConfig


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_label: str
    start_sec: float
    end_sec: float
    confidence: float | None = None
    overlap_flag: bool = False
    provider_speaker_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True)
class PreparedAudio:
    audio_path: Path
    waveform: np.ndarray
    sample_rate: int
    source_audio_path: Path
    source_kind: str


@dataclass(frozen=True)
class DiarizationResult:
    provider: str
    turns: tuple[SpeakerTurn, ...]
    normalized_audio_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DiarizationProvider(Protocol):
    def diarize(
        self,
        prepared_audio: PreparedAudio,
        config: "DiarizationConfig",
        *,
        working_dir: Path,
    ) -> DiarizationResult:
        ...
