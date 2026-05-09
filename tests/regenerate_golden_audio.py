#!/usr/bin/env python3
"""Regenerate tests/golden/audio_processing/*.npy and *.npy.sha256.

WHEN TO RUN
-----------
Only as part of a PR that intentionally changes
``worker.app.audio_processing.AudioPreprocessor.process()`` output. Examples:

  - P2: replacing ``audioop.ratecv`` with a different resampler.
  - P4: swapping the RNNoise denoiser implementation.
  - Upstream Silero VAD or RNNoise version bumps that we accept as new ground truth.

DO NOT run this script to silence a flaky failure. Investigate divergence first
— the characterization tests are designed to be byte-stable, and "rewrite the
golden" is the wrong response unless you are deliberately changing behavior.

WHAT IT DOES
------------
1. Loads the real ``AudioPreprocessor`` (Silero VAD + RNNoise binding).
2. Iterates the same (fixture, vad_enabled, denoise_enabled, mode) combos as
   ``worker/tests/test_audio_characterization.py``.
3. Writes ``<stem>.npy`` and a sibling ``<stem>.npy.sha256`` (sha256sum-format
   line) into ``tests/golden/audio_processing/``.
4. Prints a summary: per-file SHA prefix and byte size, plus total runtime.

The combo list is loaded directly from the test module so the regen script
cannot drift out of sync with what the test expects.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_TEST_MODULE_PATH = REPO_ROOT / "worker" / "tests" / "test_audio_characterization.py"
_spec = importlib.util.spec_from_file_location(
    "_audio_characterization_spec", _TEST_MODULE_PATH
)
assert _spec is not None and _spec.loader is not None, (
    f"could not load combo spec from {_TEST_MODULE_PATH}"
)
_spec_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec_module)

FIXTURES = _spec_module.FIXTURES
MODES = _spec_module.MODES
VAD_FLAGS = _spec_module.VAD_FLAGS
DENOISE_FLAGS = _spec_module.DENOISE_FLAGS
VAD_CONCAT_PADDING_MS = _spec_module.VAD_CONCAT_PADDING_MS
FIXTURE_DIR: Path = _spec_module.FIXTURE_DIR
GOLDEN_DIR: Path = _spec_module.GOLDEN_DIR
_golden_stem = _spec_module._golden_stem
_read_wav_int16 = _spec_module._read_wav_int16


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    from worker.app.audio_processing import (
        AudioPreprocessor,
        get_audio_preprocessor,
    )

    get_audio_preprocessor.cache_clear()
    ap = AudioPreprocessor()

    if ap.vad_model is None:
        dependency_hint = ""
        if importlib.util.find_spec("torchaudio") is None:
            dependency_hint = (
                "\n\nMissing dependency detected: torchaudio is not installed "
                "in this environment. Install the worker audio dependencies "
                "first, for example:\n"
                "    python -m pip install -r worker/requirements.txt\n"
                "or, for this test venv:\n"
                "    .venv-tests/bin/python -m pip install torchaudio==2.4.1"
            )
        print(
            "error: Silero VAD failed to load. Pre-cache it with internet "
            "access:\n"
            "    python -c 'import torch; "
            "torch.hub.load(\"snakers4/silero-vad\", \"silero_vad\")'"
            f"{dependency_hint}",
            file=sys.stderr,
        )
        return 2

    rnnoise_loaded = ap.rnnoise is not None
    if not rnnoise_loaded:
        print(
            "warning: no RNNoise binding (pyrnnoise / rnnoise_wrapper) is "
            "installed. denoise=True goldens will be written but they will "
            "reflect the pre-denoise output (since process() no-ops when "
            "rnnoise is None). Re-run after installing pyrnnoise to get "
            "real denoise goldens.",
            file=sys.stderr,
        )

    written: list[tuple[Path, str, int]] = []
    skipped: list[str] = []
    t0 = time.perf_counter()

    for fixture in FIXTURES:
        wav_path = FIXTURE_DIR / f"{fixture}.wav"
        pcm, sample_rate = _read_wav_int16(wav_path)
        for vad_enabled in VAD_FLAGS:
            for denoise_enabled in DENOISE_FLAGS:
                for mode in MODES:
                    stem = _golden_stem(fixture, vad_enabled, denoise_enabled, mode)
                    if denoise_enabled and not rnnoise_loaded:
                        skipped.append(stem)
                        continue

                    ap.vad_select_mode = mode
                    ap.vad_concat_padding_ms = VAD_CONCAT_PADDING_MS

                    out_bytes = ap.process(
                        pcm,
                        sample_rate,
                        vad_enabled=vad_enabled,
                        denoise_enabled=denoise_enabled,
                    )

                    arr = np.frombuffer(out_bytes, dtype=np.int16)
                    npy_path = GOLDEN_DIR / f"{stem}.npy"
                    sha_path = GOLDEN_DIR / f"{stem}.npy.sha256"

                    np.save(npy_path, arr)
                    digest = hashlib.sha256(out_bytes).hexdigest()
                    sha_path.write_text(f"{digest}  {stem}.npy\n")

                    written.append((npy_path, digest, npy_path.stat().st_size))

    elapsed = time.perf_counter() - t0
    print(f"\nWrote {len(written)} golden file(s) to {GOLDEN_DIR} in {elapsed:.2f}s")
    for path, digest, size in written:
        print(f"  {digest[:12]}  {size:>10,} B  {path.name}")

    if skipped:
        print(
            f"\nSkipped {len(skipped)} denoise=True combo(s) because no RNNoise "
            "binding is installed:",
            file=sys.stderr,
        )
        for stem in skipped:
            print(f"  - {stem}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
