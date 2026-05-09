"""Numeric equivalence: audioop.ratecv (current) vs soxr.resample (P2 candidate).

This test gates the planned swap of the resample primitive used inside
``AudioPreprocessor`` (worker/app/audio_processing.py:287, :296). It is a direct
primitive-level comparison; it does NOT exercise AudioPreprocessor. Higher-level
equivalence is the job of the audio characterization tests once P2 lands.

Production tuples
-----------------
Both ratecv calls in audio_processing.py use width=2 (int16) and channels=1.
``sample_rate`` is the value of the ``X-Sample-Rate`` HTTP header; the runtime
is hard-wired to 16 kHz everywhere downstream (Silero VAD passes
``sampling_rate=16000``; the Triton ASR ensemble expects 16 kHz). The only
tuples that production hits are therefore::

    (in_rate=16000, out_rate=48000, width=2, channels=1)   # upsample to RNNoise rate
    (in_rate=48000, out_rate=16000, width=2, channels=1)   # downsample back

Observation run (2026-05-08, soxr 1.1.0 quality="HQ", numpy 2.2.6)
------------------------------------------------------------------
Inputs: log-chirp 100->7500 Hz / 5 s, white noise 2 s, plus the 4 vaani WAVs in
tests/fixtures/audio_bench/. Worst-case RMSE across all six inputs per tuple:

    (16000 -> 48000):   worst RMSE = -20.20 dBFS,   max |delta| ~ 15329 LSB
    (48000 -> 16000):   worst RMSE = -40.03 dBFS,   max |delta| ~ 1497  LSB

The 16k->48k direction diverges hard on broadband content because ``ratecv`` is
linear interpolation (no anti-imaging filter) while soxr is bandlimited
polyphase, so spectral images near Nyquist account for most of the energy
difference. The 48k->16k direction is much closer because both backends
low-pass before decimation.

Thresholds (pinned with ~12 dB margin from observed worst, per direction)
-------------------------------------------------------------------------
::

    (16000 -> 48000):   RMSE <= -8 dBFS,   max |delta| <  25000 LSB
    (48000 -> 16000):   RMSE <= -28 dBFS,  max |delta| <  3000  LSB

Tightening either RMSE bound by 20 dB (e.g. -8 -> -28 or -28 -> -48) is
expected to fail on the broadband cases - that is the deliberate "binding
threshold" check; see test_threshold_is_binding_when_tightened_20db below.

If/when the production primitive is swapped to soxr, this file should be
deleted (it would compare soxr to itself and become vacuous). The
AudioPreprocessor-level characterization tests carry equivalence forward.
"""

from __future__ import annotations

import audioop
import math
import wave
from pathlib import Path

import numpy as np
import pytest


soxr = pytest.importorskip("soxr", minversion="0.3.7")


FIXTURES_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "audio_bench"
)

# (in_rate, out_rate) -> (rmse_dbfs_threshold, max_abs_diff_lsb_threshold).
# Values pinned 2026-05-08 from a one-time observation pass; see module
# docstring for the underlying numbers and the 12 dB margin reasoning.
TUPLE_THRESHOLDS: dict[tuple[int, int], tuple[float, int]] = {
    (16000, 48000): (-8.0, 25000),
    (48000, 16000): (-28.0, 3000),
}

# Length tolerance from the spec: each backend's output length must be within
# +/-2 samples of the ideal in_samples * out_rate / in_rate. ratecv is
# observed at -2 on the 16k->48k direction; soxr is exact.
LEN_TOLERANCE_SAMPLES = 2

WIDTH_BYTES = 2
CHANNELS = 1

CHIRP_SEED = 1234
NOISE_SEED = 5678


# ---------------------------------------------------------------------------
# Signal generation and IO
# ---------------------------------------------------------------------------

def _make_log_chirp_int16(
    seed: int,
    sr: int = 16000,
    dur_s: float = 5.0,
    f0: float = 100.0,
    f1: float = 7500.0,
) -> np.ndarray:
    """Deterministic logarithmic chirp 100 Hz -> 7500 Hz, mono int16 @ ``sr``."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    k = (f1 / f0) ** (1.0 / dur_s)
    phase = 2.0 * math.pi * f0 * (k**t - 1.0) / math.log(k)
    sig = np.sin(phase) * 0.6
    sig += rng.standard_normal(n) * 1e-6  # tiny dither to break ties at edges
    return (sig * 32767.0).astype(np.int16)


def _make_white_noise_int16(
    seed: int, sr: int = 16000, dur_s: float = 2.0
) -> np.ndarray:
    """Deterministic white noise, mono int16 @ ``sr``."""
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    sig = rng.standard_normal(n) * 0.3
    sig = np.clip(sig, -1.0, 1.0)
    return (sig * 32767.0).astype(np.int16)


def _load_wav_int16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == CHANNELS, f"{path}: expected mono"
        assert w.getsampwidth() == WIDTH_BYTES, f"{path}: expected 16-bit"
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).copy(), sr


# ---------------------------------------------------------------------------
# Resample primitives under test
# ---------------------------------------------------------------------------

def _ratecv_int16(pcm_i16: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Mirror the production audioop.ratecv call (width=2, channels=1)."""
    out_bytes, _state = audioop.ratecv(
        pcm_i16.tobytes(), WIDTH_BYTES, CHANNELS, in_rate, out_rate, None
    )
    return np.frombuffer(out_bytes, dtype=np.int16).copy()


def _soxr_int16(pcm_i16: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Candidate primitive: soxr at HQ quality, int16 in / int16 out."""
    f = pcm_i16.astype(np.float64) / 32768.0
    y = soxr.resample(f, in_rate, out_rate, quality="HQ")
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse_dbfs(a: np.ndarray, b: np.ndarray) -> float:
    """RMS error of (a - b), expressed in dBFS relative to int16 full-scale."""
    n = min(a.size, b.size)
    if n == 0:
        return float("-inf")
    diff = a[:n].astype(np.int64) - b[:n].astype(np.int64)
    rms = math.sqrt(float(np.mean(diff.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms / 32768.0)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> int:
    n = min(a.size, b.size)
    if n == 0:
        return 0
    return int(np.max(np.abs(a[:n].astype(np.int64) - b[:n].astype(np.int64))))


def _expected_out_len(in_len: int, in_rate: int, out_rate: int) -> int:
    return int(round(in_len * out_rate / in_rate))


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Each case is (case_id, pcm_int16, native_sr). For each case we exercise both
# production tuples: native -> 48k (using the original PCM) and 48k -> native
# (using a soxr-upsampled 48k version, so the downsample step is judged on the
# downsample primitive itself rather than on stacked artefacts from ratecv's
# upsample stage).

def _build_case_inputs() -> list[tuple[str, np.ndarray, int]]:
    cases: list[tuple[str, np.ndarray, int]] = [
        ("chirp_100_to_7500_5s_16k", _make_log_chirp_int16(CHIRP_SEED), 16000),
        ("white_noise_2s_16k", _make_white_noise_int16(NOISE_SEED), 16000),
    ]
    for wav_path in sorted(FIXTURES_DIR.glob("*.wav")):
        pcm, sr = _load_wav_int16_mono(wav_path)
        cases.append((wav_path.name, pcm, sr))
    return cases


def _flatten_for_pytest() -> list[pytest.param]:
    out: list[pytest.param] = []
    for case_id, pcm, native_sr in _build_case_inputs():
        for in_rate, out_rate in TUPLE_THRESHOLDS:
            if in_rate == native_sr:
                pcm_in = pcm
            else:
                # Native -> in_rate via soxr to obtain a clean source for the
                # other direction. soxr is "ground truth" for resampling here;
                # we are not measuring this conversion, only feeding both
                # backends the same bytes for the tuple under test.
                pcm_in = _soxr_int16(pcm, native_sr, in_rate)
            param_id = f"{case_id}__{in_rate}->{out_rate}"
            out.append(pytest.param(case_id, pcm_in, in_rate, out_rate, id=param_id))
    return out


@pytest.mark.equivalence
@pytest.mark.parametrize("case_id,pcm_in,in_rate,out_rate", _flatten_for_pytest())
def test_ratecv_vs_soxr_within_pinned_thresholds(
    case_id: str, pcm_in: np.ndarray, in_rate: int, out_rate: int
) -> None:
    """audioop.ratecv vs soxr.resample stays within pinned thresholds.

    Per-tuple thresholds are pinned in TUPLE_THRESHOLDS at the top of this
    file; see the module docstring for the observation run from 2026-05-08
    and the 12 dB margin used to derive them.
    """
    rmse_thresh, max_abs_thresh = TUPLE_THRESHOLDS[(in_rate, out_rate)]

    out_ratecv = _ratecv_int16(pcm_in, in_rate, out_rate)
    out_soxr = _soxr_int16(pcm_in, in_rate, out_rate)

    expected_n = _expected_out_len(pcm_in.size, in_rate, out_rate)

    rcv_drift = out_ratecv.size - expected_n
    soxr_drift = out_soxr.size - expected_n
    assert abs(rcv_drift) <= LEN_TOLERANCE_SAMPLES, (
        f"[{case_id} {in_rate}->{out_rate}] ratecv length drift "
        f"{rcv_drift:+d} samples exceeds +/-{LEN_TOLERANCE_SAMPLES} of expected "
        f"{expected_n}"
    )
    assert abs(soxr_drift) <= LEN_TOLERANCE_SAMPLES, (
        f"[{case_id} {in_rate}->{out_rate}] soxr length drift "
        f"{soxr_drift:+d} samples exceeds +/-{LEN_TOLERANCE_SAMPLES} of expected "
        f"{expected_n}"
    )

    rmse = _rmse_dbfs(out_ratecv, out_soxr)
    max_abs = _max_abs_diff(out_ratecv, out_soxr)

    assert rmse <= rmse_thresh, (
        f"[{case_id} {in_rate}->{out_rate}] RMSE {rmse:.2f} dBFS exceeds "
        f"threshold {rmse_thresh:.2f} dBFS"
    )
    assert max_abs < max_abs_thresh, (
        f"[{case_id} {in_rate}->{out_rate}] max |delta| {max_abs} LSB "
        f"exceeds sanity ceiling {max_abs_thresh} LSB"
    )


@pytest.mark.equivalence
@pytest.mark.parametrize("in_rate,out_rate", list(TUPLE_THRESHOLDS.keys()))
def test_threshold_is_binding_when_tightened_20db(
    in_rate: int, out_rate: int
) -> None:
    """Tightening the RMSE bound by 20 dB must fail on at least one input.

    Proves the threshold is the binding constraint, not a vacuous check. If
    this stops failing, either the resample backends got dramatically closer
    (in which case the main thresholds should be tightened) or the test
    inputs no longer stress the comparison.
    """
    rmse_thresh, _ = TUPLE_THRESHOLDS[(in_rate, out_rate)]
    tightened = rmse_thresh - 20.0  # smaller (more negative) dBFS == tighter

    worst_rmse = float("-inf")
    for case_id, pcm, native_sr in _build_case_inputs():
        if in_rate == native_sr:
            pcm_in = pcm
        else:
            pcm_in = _soxr_int16(pcm, native_sr, in_rate)
        rmse = _rmse_dbfs(
            _ratecv_int16(pcm_in, in_rate, out_rate),
            _soxr_int16(pcm_in, in_rate, out_rate),
        )
        if rmse > worst_rmse:
            worst_rmse = rmse

    assert worst_rmse > tightened, (
        f"[{in_rate}->{out_rate}] tightening RMSE threshold by 20 dB "
        f"({rmse_thresh:.1f} -> {tightened:.1f} dBFS) did NOT fail on any "
        f"input (worst observed = {worst_rmse:.2f} dBFS). The main threshold "
        f"may be vacuously loose; consider tightening TUPLE_THRESHOLDS."
    )
