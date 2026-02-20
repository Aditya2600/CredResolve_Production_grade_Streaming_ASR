from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 20
    energy_threshold: float = 0.015
    min_speech_ms: int = 200
    min_silence_ms: int = 300
    max_utt_ms: int = 3000


@dataclass(frozen=True)
class AudioSegment:
    start_sample: int
    end_sample: int


class AudioPreprocessError(ValueError):
    pass


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Load PCM16 WAV and return float32 mono in [-1,1]."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        if width != 2:
            raise AudioPreprocessError("Only PCM16 WAV is supported")
        pcm = wf.readframes(wf.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), int(sr)


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise AudioPreprocessError("Audio must be 1D mono or 2D multi-channel")
    if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
        # channels-first heuristic
        return arr.mean(axis=0)
    return arr.mean(axis=1)


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if src_rate <= 0 or dst_rate <= 0:
        raise AudioPreprocessError("Sample rate must be positive")
    arr = np.asarray(audio, dtype=np.float32)
    if src_rate == dst_rate:
        return arr
    if arr.size == 0:
        return arr
    duration = arr.size / float(src_rate)
    dst_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=arr.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_size, endpoint=False)
    out = np.interp(dst_x, src_x, arr)
    return out.astype(np.float32)


def ensure_16k_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = downmix_to_mono(audio)
    return resample_linear(mono, src_rate=sample_rate, dst_rate=TARGET_SAMPLE_RATE)


def pcm16_bytes_to_float32(pcm16le: bytes) -> np.ndarray:
    return np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16le(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def frame_audio(audio: np.ndarray, frame_size: int) -> Iterable[np.ndarray]:
    if frame_size <= 0:
        raise AudioPreprocessError("frame_size must be > 0")
    for i in range(0, len(audio), frame_size):
        yield audio[i : i + frame_size]


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))


def energy_vad(audio: np.ndarray, sample_rate: int, config: VADConfig | None = None) -> list[AudioSegment]:
    cfg = config or VADConfig()
    if sample_rate <= 0:
        raise AudioPreprocessError("sample_rate must be > 0")
    if len(audio) == 0:
        return []

    frame_size = max(1, int(sample_rate * cfg.frame_ms / 1000))
    min_speech_frames = max(1, cfg.min_speech_ms // cfg.frame_ms)
    min_silence_frames = max(1, cfg.min_silence_ms // cfg.frame_ms)
    max_utt_frames = max(1, cfg.max_utt_ms // cfg.frame_ms)

    segments: list[AudioSegment] = []
    in_speech = False
    speech_start = 0
    speech_frames = 0
    silence_frames = 0

    frames = list(frame_audio(np.asarray(audio, dtype=np.float32), frame_size))
    for frame_idx, frame in enumerate(frames):
        is_speech = _rms(frame) >= cfg.energy_threshold

        if is_speech:
            if not in_speech:
                in_speech = True
                speech_start = frame_idx * frame_size
                speech_frames = 0
            speech_frames += 1
            silence_frames = 0
        elif in_speech:
            silence_frames += 1

        if in_speech and speech_frames >= max_utt_frames:
            end = min(len(audio), (frame_idx + 1) * frame_size)
            segments.append(AudioSegment(start_sample=speech_start, end_sample=end))
            in_speech = False
            speech_frames = 0
            silence_frames = 0
            continue

        if in_speech and silence_frames >= min_silence_frames:
            if speech_frames >= min_speech_frames:
                end = max(speech_start + 1, frame_idx * frame_size)
                segments.append(AudioSegment(start_sample=speech_start, end_sample=min(len(audio), end)))
            in_speech = False
            speech_frames = 0
            silence_frames = 0

    if in_speech and speech_frames >= min_speech_frames:
        segments.append(AudioSegment(start_sample=speech_start, end_sample=len(audio)))

    return segments


def split_by_vad(audio: np.ndarray, sample_rate: int, config: VADConfig | None = None) -> list[np.ndarray]:
    segs = energy_vad(audio, sample_rate, config=config)
    if not segs:
        return []
    return [audio[s.start_sample : s.end_sample].astype(np.float32) for s in segs if s.end_sample > s.start_sample]
