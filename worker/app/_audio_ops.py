"""Byte-level PCM helpers replacing the stdlib ``audioop`` module.

Used by:
  - ``worker.app.audio_processing`` (resample only, in the denoise path).
  - ``tools.benchmarks.audio_bench`` (sample-width conversion, stereo->mono,
    resample — the WAV ingestion path).

Lives under ``worker/app/`` (not a separate top-level package) because the
bench tool already imports ``worker.app.*`` and the worker Docker image
already ships this directory. Both call sites import from here so the two
audio paths cannot drift.
"""
from __future__ import annotations

import numpy as np
import soxr

_INT16_MIN = -32768
_INT16_MAX = 32767


def int_bytes_to_int16(buf: bytes, sampwidth: int) -> bytes:
    """Convert signed-integer PCM bytes of width N to signed int16 bytes.

    Mirrors ``audioop.lin2lin(buf, sampwidth, 2)`` for ``sampwidth in {1,2,3,4}``:
    audioop treats every width as signed and rescales by an arithmetic shift of
    ``(newwidth - width) * 8`` bits.

    The bench currently exercises only ``sampwidth == 2``; the other branches
    exist for parity with the audioop API but are not covered by the fixtures.
    """
    if sampwidth == 2:
        return buf
    if sampwidth == 1:
        i8 = np.frombuffer(buf, dtype=np.int8).astype(np.int16)
        return (i8 << 8).astype(np.int16).tobytes()
    if sampwidth == 4:
        i32 = np.frombuffer(buf, dtype=np.int32)
        return (i32 >> 16).astype(np.int16).tobytes()
    if sampwidth == 3:
        n = len(buf) // 3
        b3 = np.frombuffer(buf, dtype=np.uint8).reshape(n, 3)
        i32 = (
            b3[:, 0].astype(np.int32)
            | (b3[:, 1].astype(np.int32) << 8)
            | (b3[:, 2].view(np.int8).astype(np.int32) << 16)
        )
        return (i32 >> 8).astype(np.int16).tobytes()
    raise ValueError(f"unsupported sampwidth: {sampwidth}")


def stereo_to_mono_int16(buf: bytes) -> bytes:
    """Mix interleaved stereo int16 PCM down to mono with 50/50 weights.

    Mirrors ``audioop.tomono(buf, 2, 0.5, 0.5)``: per-pair mean rounded toward
    zero (audioop performs ``(L * 0.5 + R * 0.5)`` with truncation).
    """
    arr = np.frombuffer(buf, dtype=np.int16).reshape(-1, 2).astype(np.int32)
    mono = (arr.sum(axis=1) // 2).astype(np.int16)
    return mono.tobytes()


def resample_int16(
    buf: bytes,
    in_rate: int,
    out_rate: int,
    *,
    quality: str = "HQ",
) -> bytes:
    """Resample mono int16 PCM bytes from ``in_rate`` to ``out_rate``.

    Replaces ``audioop.ratecv(buf, 2, 1, in_rate, out_rate, None)``. The
    float64 round-trip mirrors ``tests/test_resample_equivalence.py`` so
    production output stays inside the equivalence thresholds pinned there.
    """
    if in_rate == out_rate or not buf:
        return buf
    arr = np.frombuffer(buf, dtype=np.int16)
    if arr.size == 0:
        return b""
    f = arr.astype(np.float64) / 32768.0
    y = soxr.resample(f, in_rate, out_rate, quality=quality)
    return (
        np.clip(np.round(y * 32768.0), _INT16_MIN, _INT16_MAX)
        .astype(np.int16)
        .tobytes()
    )
