from __future__ import annotations

import numpy as np
import pytest

from worker.app import audio_processing
from worker.app.audio_processing import (
    AudioPreprocessor,
    _RNNoiseWrapperDenoiser,
    get_audio_preprocessor,
)


SAMPLE_RATE = 16000


def _silence(num_samples: int) -> bytes:
    return np.zeros(num_samples, dtype=np.int16).tobytes()


def _tone(
    num_samples: int,
    amplitude: int = 5000,
    freq: float = 440.0,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


@pytest.fixture
def make_preprocessor(monkeypatch):
    """Build an AudioPreprocessor without loading real models."""
    monkeypatch.setattr(AudioPreprocessor, "_load_models", lambda self: None)

    def _build(speech_timestamps=None, vad_select_mode=None, vad_concat_padding_ms=None):
        ap = AudioPreprocessor(
            vad_select_mode=vad_select_mode,
            vad_concat_padding_ms=vad_concat_padding_ms,
        )
        if speech_timestamps is not None:
            # Provide a stub VAD model + utils.
            ap.vad_model = object()
            stub_get_speech_timestamps = lambda *_args, **_kwargs: speech_timestamps
            ap.utils = (stub_get_speech_timestamps,)
        # No denoiser in tests.
        ap.rnnoise = None
        return ap

    return _build


def test_empty_input_returns_empty_bytes(make_preprocessor):
    ap = make_preprocessor(speech_timestamps=[])
    assert ap.process(b"", SAMPLE_RATE, vad_enabled=True, denoise_enabled=False) == b""


def test_pure_silence_returns_empty_bytes(make_preprocessor):
    ap = make_preprocessor(speech_timestamps=[])
    pcm = _silence(SAMPLE_RATE)  # 1s of silence
    out = ap.process(pcm, SAMPLE_RATE, vad_enabled=True, denoise_enabled=False)
    assert out == b""


def test_single_speech_segment_returns_that_segment(make_preprocessor):
    pcm = _tone(SAMPLE_RATE)  # 1s tone
    seg_start = 4000
    seg_end = 12000
    ap = make_preprocessor(
        speech_timestamps=[{"start": seg_start, "end": seg_end}],
        vad_select_mode="concat",
    )
    out = ap.process(pcm, SAMPLE_RATE, vad_enabled=True, denoise_enabled=False)
    expected = np.frombuffer(pcm, dtype=np.int16)[seg_start:seg_end].tobytes()
    assert out == expected


def test_concat_mode_combines_all_segments_with_padding(make_preprocessor):
    total_samples = 32000  # 2s
    pcm = _tone(total_samples)
    segments = [
        {"start": 0, "end": 4000},
        {"start": 8000, "end": 12000},
        {"start": 20000, "end": 30000},
    ]
    padding_ms = 100
    ap = make_preprocessor(
        speech_timestamps=segments,
        vad_select_mode="concat",
        vad_concat_padding_ms=padding_ms,
    )
    out = ap.process(pcm, SAMPLE_RATE, vad_enabled=True, denoise_enabled=False)

    seg_lens = [s["end"] - s["start"] for s in segments]
    padding_samples = int(padding_ms * SAMPLE_RATE / 1000)
    expected_samples = sum(seg_lens) + padding_samples * (len(segments) - 1)

    out_samples = len(out) // 2  # int16 = 2 bytes
    assert out_samples == expected_samples


def test_concat_mode_zero_padding(make_preprocessor):
    pcm = _tone(20000)
    segments = [
        {"start": 0, "end": 4000},
        {"start": 8000, "end": 12000},
    ]
    ap = make_preprocessor(
        speech_timestamps=segments,
        vad_select_mode="concat",
        vad_concat_padding_ms=0,
    )
    out = ap.process(pcm, SAMPLE_RATE, vad_enabled=True, denoise_enabled=False)
    expected = sum(s["end"] - s["start"] for s in segments)
    assert len(out) // 2 == expected


def test_loudest_mode_returns_only_loudest_segment(make_preprocessor):
    sr = SAMPLE_RATE
    quiet = (1000 * np.sin(2 * np.pi * 440 * np.arange(8000, dtype=np.float32) / sr)).astype(np.int16)
    loud = (15000 * np.sin(2 * np.pi * 440 * np.arange(6000, dtype=np.float32) / sr)).astype(np.int16)
    medium = (5000 * np.sin(2 * np.pi * 440 * np.arange(10000, dtype=np.float32) / sr)).astype(np.int16)
    pad = np.zeros(2000, dtype=np.int16)

    audio = np.concatenate([quiet, pad, loud, pad, medium])
    pcm = audio.tobytes()

    quiet_start = 0
    quiet_end = quiet_start + len(quiet)
    loud_start = quiet_end + len(pad)
    loud_end = loud_start + len(loud)
    medium_start = loud_end + len(pad)
    medium_end = medium_start + len(medium)

    segments = [
        {"start": quiet_start, "end": quiet_end},
        {"start": loud_start, "end": loud_end},
        {"start": medium_start, "end": medium_end},
    ]
    ap = make_preprocessor(
        speech_timestamps=segments,
        vad_select_mode="loudest",
    )
    out = ap.process(pcm, sr, vad_enabled=True, denoise_enabled=False)
    assert len(out) // 2 == len(loud)
    # And the bytes match the loud segment exactly.
    assert out == audio[loud_start:loud_end].tobytes()


def test_get_audio_preprocessor_returns_same_instance(monkeypatch):
    monkeypatch.setattr(AudioPreprocessor, "_load_models", lambda self: None)
    # Reset any prior cached instance so the test is hermetic.
    get_audio_preprocessor.cache_clear()
    a = get_audio_preprocessor()
    b = get_audio_preprocessor()
    assert a is b
    get_audio_preprocessor.cache_clear()


def test_denoise_adapter_is_used(make_preprocessor):
    class StubDenoiser:
        def __init__(self):
            self.inputs = []

        def process(self, pcm_48k: bytes) -> bytes:
            self.inputs.append(pcm_48k)
            return (np.frombuffer(pcm_48k, dtype=np.int16) // 2).astype(np.int16).tobytes()

    ap = make_preprocessor()
    ap.rnnoise = StubDenoiser()
    pcm = _tone(480, sample_rate=48000)

    out, stats = ap.process_with_stats(
        pcm,
        48000,
        vad_enabled=False,
        denoise_enabled=True,
    )

    assert ap.rnnoise.inputs == [pcm]
    assert np.array_equal(
        np.frombuffer(out, dtype=np.int16),
        np.frombuffer(pcm, dtype=np.int16) // 2,
    )
    assert stats["denoise_seconds"] is not None
    assert stats["denoise_input_samples"] == 480
    assert stats["denoise_output_samples"] == 480


def test_legacy_rnnoise_wrapper_adapter_supports_filter_frame():
    class StubLegacyDenoiser:
        def __init__(self):
            self.frame_lengths = []

        def filter_frame(self, frame: bytes):
            self.frame_lengths.append(len(frame))
            samples = np.frombuffer(frame, dtype=np.int16)
            return 0.5, (samples + 1).astype(np.int16).tobytes()

    stub = StubLegacyDenoiser()
    denoiser = _RNNoiseWrapperDenoiser(stub)
    pcm = np.arange(500, dtype=np.int16).tobytes()

    out = denoiser.process(pcm)

    assert stub.frame_lengths == [960, 960]
    assert len(out) == len(pcm)
    assert np.array_equal(
        np.frombuffer(out, dtype=np.int16),
        np.arange(500, dtype=np.int16) + 1,
    )


def test_load_denoiser_dispatches_to_deepfilternet_when_configured(monkeypatch):
    """DENOISER=deepfilternet must route through _DeepFilterNetDenoiser, not RNNoise."""
    # Don't load Silero etc. — only exercise the denoiser dispatch.
    monkeypatch.setattr(
        AudioPreprocessor,
        "_load_models",
        lambda self: setattr(self, "rnnoise", self._load_denoiser()),
    )
    monkeypatch.setattr(audio_processing.config, "DENOISER", "deepfilternet")

    constructed = []

    class StubDFN:
        def __init__(self):
            constructed.append(self)

        def process(self, pcm_48k: bytes) -> bytes:  # pragma: no cover - not called here
            return pcm_48k

    monkeypatch.setattr(audio_processing, "_DeepFilterNetDenoiser", StubDFN)
    ap = AudioPreprocessor()
    assert isinstance(ap.rnnoise, StubDFN)
    assert len(constructed) == 1


def test_load_denoiser_falls_back_to_rnnoise_when_dfn_missing(monkeypatch):
    """If deepfilternet is selected but the package isn't installed, fall back to RNNoise."""
    monkeypatch.setattr(
        AudioPreprocessor,
        "_load_models",
        lambda self: setattr(self, "rnnoise", self._load_denoiser()),
    )
    monkeypatch.setattr(audio_processing.config, "DENOISER", "deepfilternet")

    def _raise_import_error():
        raise ImportError("deepfilternet not installed")

    class FakeDFN:
        def __init__(self):
            _raise_import_error()

    sentinel = object()
    monkeypatch.setattr(audio_processing, "_DeepFilterNetDenoiser", FakeDFN)
    monkeypatch.setattr(AudioPreprocessor, "_load_rnnoise", lambda self: sentinel)

    ap = AudioPreprocessor()
    assert ap.rnnoise is sentinel


def test_invalid_vad_select_mode_falls_back_to_config_default(make_preprocessor, monkeypatch):
    monkeypatch.setattr(audio_processing.config, "VAD_SELECT_MODE", "concat")
    ap = make_preprocessor(
        speech_timestamps=[{"start": 0, "end": 4000}, {"start": 8000, "end": 12000}],
        vad_select_mode="not_a_real_mode",
        vad_concat_padding_ms=0,
    )
    pcm = _tone(20000)
    out = ap.process(pcm, SAMPLE_RATE, vad_enabled=True, denoise_enabled=False)
    # Concat behavior expected: total = 4000 + 4000 samples.
    assert len(out) // 2 == 8000
