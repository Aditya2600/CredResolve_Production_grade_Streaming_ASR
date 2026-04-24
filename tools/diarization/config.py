from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MULTISCALE_WINDOW_SEC = (1.5, 1.25, 1.0, 0.75, 0.5)
DEFAULT_MULTISCALE_HOP_SEC = (0.75, 0.625, 0.5, 0.375, 0.25)


@dataclass(frozen=True)
class DiarizationConfig:
    min_segment_sec: float = 0.5
    max_segment_sec: float = 20.0
    multiscale_window_sec: tuple[float, ...] = DEFAULT_MULTISCALE_WINDOW_SEC
    multiscale_hop_sec: tuple[float, ...] = DEFAULT_MULTISCALE_HOP_SEC
    speaker_count_mode: str = "estimate"
    fixed_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 2
    clustering_threshold: float = 0.25
    overlap: bool = False
    sample_rate: int = 16000
    vad_window_sec: float = 0.15
    vad_shift_sec: float = 0.01
    vad_onset: float = 0.1
    vad_offset: float = 0.1
    vad_pad_onset: float = 0.1
    vad_pad_offset: float = 0.0
    vad_min_duration_on: float = 0.0
    vad_min_duration_off: float = 0.2
    vad_smoothing: str = "median"
    vad_overlap: float = 0.5
    nemo_vad_model: str = "vad_multilingual_marblenet"
    nemo_speaker_model: str = "titanet_large"
    nemo_msdd_model: str = "diar_msdd_telephonic"
