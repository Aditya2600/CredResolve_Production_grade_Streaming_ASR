from __future__ import annotations

import numpy as np

from asr.preprocess import (
    VADConfig,
    downmix_to_mono,
    ensure_16k_mono,
    energy_vad,
    float32_to_pcm16le,
    pcm16_bytes_to_float32,
    resample_linear,
)


def _tone(sr: int, seconds: float, freq: float = 440.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / float(sr)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_resample_8k_to_16k_doubles_length() -> None:
    x = _tone(8000, 1.0)
    y = resample_linear(x, 8000, 16000)
    assert abs(len(y) - 16000) <= 1


def test_downmix_stereo_to_mono_channels_first() -> None:
    x = np.stack([_tone(16000, 0.25), _tone(16000, 0.25, freq=220.0)], axis=0)
    y = downmix_to_mono(x)
    assert y.ndim == 1
    assert y.shape[0] == x.shape[1]


def test_downmix_stereo_to_mono_channels_last() -> None:
    x = np.stack([_tone(16000, 0.25), _tone(16000, 0.25, freq=220.0)], axis=1)
    y = downmix_to_mono(x)
    assert y.ndim == 1
    assert y.shape[0] == x.shape[0]


def test_ensure_16k_mono_from_8k_stereo() -> None:
    x = np.stack([_tone(8000, 0.5), _tone(8000, 0.5, freq=220.0)], axis=1)
    y = ensure_16k_mono(x, 8000)
    assert y.ndim == 1
    assert abs(len(y) - 8000) <= 1


def test_energy_vad_detects_speech_segment() -> None:
    sr = 16000
    silence = np.zeros(int(sr * 0.4), dtype=np.float32)
    speech = _tone(sr, 0.8, amp=0.25)
    audio = np.concatenate([silence, speech, silence])
    segs = energy_vad(audio, sr, config=VADConfig(energy_threshold=0.01, min_speech_ms=150, min_silence_ms=200))
    assert len(segs) >= 1
    first = segs[0]
    assert first.end_sample > first.start_sample


def test_pcm16_roundtrip_preserves_shape() -> None:
    x = _tone(16000, 0.4)
    b = float32_to_pcm16le(x)
    y = pcm16_bytes_to_float32(b)
    assert y.shape == x.shape
    assert np.max(np.abs(x - y)) < 0.02
