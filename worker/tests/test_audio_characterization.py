"""Characterization tests pinning AudioPreprocessor.process() byte output.

These run the REAL Silero VAD and REAL RNNoise denoiser (no monkeypatch) over
the four ``tests/fixtures/audio_bench/*.wav`` clips for every supported
(vad_enabled, denoise_enabled, vad_select_mode) combination. The output bytes
are compared to ``tests/golden/audio_processing/<stem>.npy`` files using a
two-tier strategy:

Tier 1 — three-way SHA agreement: ``sha256(live output)`` ==
    ``sha256(<stem>.npy bytes)`` == hash committed in ``<stem>.npy.sha256``.
    Catches any regression in CPU determinism, library versions, or the
    algorithm itself, byte for byte. The three-way form also catches a
    single-file edit (e.g., bit-rot in the ``.npy`` or a stale ``.sha256``)
    that a two-way check would silently pass.

Tier 2 — On Tier 1 mismatch only, classify the failure: golden
    self-inconsistency (.npy and .sha256 disagree, file edited in isolation),
    drift within tolerance (``np.allclose(out, golden, atol=1, rtol=0)``
    passes — likely a non-deterministic numpy/torch update worth
    investigating before regenerating), or real divergence. The failure
    message names which case applies so reviewers can decide whether to
    investigate or regenerate.

The denoise=True path also exercises the ``audioop.ratecv`` 16k → 48k → 16k
resampler at audio_processing.py lines 287 and 296. Those goldens are what a
future P2 audioop-replacement PR has to diff against.

Regenerate the goldens via ``tests/regenerate_golden_audio.py`` ONLY when
intentionally changing process() output (e.g., P2 audioop swap, P4 denoiser
swap, or an upstream model bump we accept).

Structural tests in test_audio_processing.py exercise the same control flow
with mocked models and are independent of these goldens.
"""
from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "audio_bench"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "audio_processing"

FIXTURES = (
    "01_iisc_vaaniproject_m_biha",
    "02_iisc_vaaniproject_m_biha",
    "03_iisc_vaaniproject_k_utta",
    "04_iisc_vaaniproject_k_jhar",
)
MODES = ("loudest", "concat")
VAD_FLAGS = (True,)            # vad=False bypasses both modes (see README)
DENOISE_FLAGS = (False, True)
VAD_CONCAT_PADDING_MS = 100    # mirrors config.VAD_CONCAT_PADDING_MS default


def _golden_stem(fixture: str, vad_enabled: bool, denoise_enabled: bool, mode: str) -> str:
    return (
        f"{fixture}__vad{int(vad_enabled)}__denoise{int(denoise_enabled)}__{mode}"
    )


def _read_wav_int16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, f"{path}: expected mono"
        assert w.getsampwidth() == 2, f"{path}: expected int16 PCM"
        return w.readframes(w.getnframes()), w.getframerate()


def _all_combos():
    return [
        (fixture, vad, denoise, mode)
        for fixture in FIXTURES
        for vad in VAD_FLAGS
        for denoise in DENOISE_FLAGS
        for mode in MODES
    ]


@pytest.mark.characterization
@pytest.mark.parametrize(
    ("fixture", "vad_enabled", "denoise_enabled", "mode"),
    _all_combos(),
    ids=lambda v: str(v),
)
def test_process_output_pinned(
    fixture,
    vad_enabled,
    denoise_enabled,
    mode,
    real_preprocessor,
    rnnoise_available,
):
    if denoise_enabled and not rnnoise_available:
        pytest.skip(
            "Denoise path requires pyrnnoise or rnnoise_wrapper. Install one "
            "of them to run this combo."
        )

    wav_path = FIXTURE_DIR / f"{fixture}.wav"
    pcm, sample_rate = _read_wav_int16(wav_path)

    ap = real_preprocessor
    ap.vad_select_mode = mode
    ap.vad_concat_padding_ms = VAD_CONCAT_PADDING_MS

    out_bytes = ap.process(
        pcm, sample_rate, vad_enabled=vad_enabled, denoise_enabled=denoise_enabled
    )

    stem = _golden_stem(fixture, vad_enabled, denoise_enabled, mode)
    npy_path = GOLDEN_DIR / f"{stem}.npy"
    sha_path = GOLDEN_DIR / f"{stem}.npy.sha256"

    if not npy_path.is_file() or not sha_path.is_file():
        pytest.fail(
            f"Golden missing for '{stem}':\n"
            f"  expected: {npy_path}\n"
            f"            {sha_path}\n"
            f"Generate it with: python tests/regenerate_golden_audio.py"
        )

    committed_sha = sha_path.read_text().strip().split()[0]
    golden_arr = np.load(npy_path)
    golden_sha = hashlib.sha256(golden_arr.tobytes()).hexdigest()
    actual_sha = hashlib.sha256(out_bytes).hexdigest()

    # Tier 1: live output bytes, the .npy bytes, and the committed .sha256
    # all agree. Any single-file edit (npy or sha256) fails this.
    if actual_sha == golden_sha == committed_sha:
        return

    if golden_sha != committed_sha:
        pytest.fail(
            f"[{stem}] golden self-inconsistency — sha256(.npy) = {golden_sha} "
            f"but committed .sha256 = {committed_sha}. One file was edited "
            "without regenerating the other. Run "
            "tests/regenerate_golden_audio.py."
        )

    out_arr = np.frombuffer(out_bytes, dtype=np.int16)

    if out_arr.shape != golden_arr.shape:
        pytest.fail(
            f"[{stem}] real divergence — live shape {out_arr.shape} != "
            f"golden shape {golden_arr.shape}.\n"
            f"  golden sha: {golden_sha}\n"
            f"  live   sha: {actual_sha}\n"
            "If this change is intentional, regenerate goldens via "
            "tests/regenerate_golden_audio.py."
        )

    if np.allclose(out_arr, golden_arr, atol=1, rtol=0):
        pytest.fail(
            f"[{stem}] drift within tolerance (atol=1, rtol=0) but byte "
            "hash diverged.\n"
            f"  golden sha: {golden_sha}\n"
            f"  live   sha: {actual_sha}\n"
            "This usually means a non-deterministic numpy/torch update. "
            "Investigate before regenerating — characterization is supposed "
            "to be byte-stable."
        )

    delta = out_arr.astype(np.int32) - golden_arr.astype(np.int32)
    pytest.fail(
        f"[{stem}] real divergence — max |delta| = {int(np.max(np.abs(delta)))} "
        f"(atol=1, rtol=0).\n"
        f"  golden sha: {golden_sha}\n"
        f"  live   sha: {actual_sha}\n"
        "If this is an intentional algorithm change (e.g., audioop swap, "
        "denoiser swap), regenerate goldens via "
        "tests/regenerate_golden_audio.py."
    )
