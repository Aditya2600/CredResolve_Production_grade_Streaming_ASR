"""Shared fixtures for worker/tests/.

The characterization tests in test_audio_characterization.py need the real
Silero VAD and a real RNNoise binding loaded once per session. These fixtures:

- Verify the Silero VAD model cache exists locally and SKIP cleanly with a
  clear setup hint if it is missing. Tests must NOT touch the network at
  collect or run time, so we never call ``torch.hub.load(force_reload=True)``.
- Probe whether ``pyrnnoise`` or ``rnnoise_wrapper`` is importable so denoise
  combinations can be skipped in environments that lack the C binding.
- Build one ``AudioPreprocessor`` per session and yield a per-test view that
  saves/restores ``vad_select_mode`` / ``vad_concat_padding_ms`` so tests can
  re-configure without leaking state.

Structural tests in test_audio_processing.py monkeypatch ``_load_models`` and
do not touch this fixture stack.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


_TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
_SILERO_HUB_DIR = _TORCH_HOME / "hub" / "snakers4_silero-vad_master"
_SILERO_REQUIRED_FILES = (
    _SILERO_HUB_DIR / "hubconf.py",
    _SILERO_HUB_DIR / "src" / "silero_vad" / "data" / "silero_vad.jit",
)
_SILERO_SETUP_HINT = (
    "Pre-cache the model once with internet access:\n"
    "    python -c 'import torch; torch.hub.load(\"snakers4/silero-vad\", \"silero_vad\")'\n"
    f"This populates {_SILERO_HUB_DIR}. After that, characterization tests run offline."
)


def _silero_vad_cached() -> bool:
    return all(p.is_file() for p in _SILERO_REQUIRED_FILES)


def _detect_rnnoise() -> str | None:
    try:
        import pyrnnoise  # noqa: F401

        return "pyrnnoise"
    except Exception:
        pass
    try:
        import rnnoise_wrapper  # noqa: F401

        return "rnnoise_wrapper"
    except Exception:
        return None


@pytest.fixture(scope="session")
def rnnoise_available() -> bool:
    """True if at least one RNNoise Python binding is importable in this env."""
    return _detect_rnnoise() is not None


@pytest.fixture(scope="session")
def _real_session_preprocessor():
    """Load Silero VAD + RNNoise once and return a real AudioPreprocessor.

    Skips the dependent tests with a clear hint if the local Silero cache is
    missing — we deliberately do not network at test time.
    """
    if not _silero_vad_cached():
        pytest.skip(
            "Silero VAD model cache not found at "
            f"{_SILERO_HUB_DIR}.\n{_SILERO_SETUP_HINT}"
        )
    from worker.app.audio_processing import (
        AudioPreprocessor,
        get_audio_preprocessor,
    )

    get_audio_preprocessor.cache_clear()
    return AudioPreprocessor()


@pytest.fixture
def real_preprocessor(_real_session_preprocessor):
    """Per-test view of the session AudioPreprocessor.

    Saves and restores the two configuration attributes the tests mutate so
    state does not leak between cases. The VAD model and RNNoise denoiser
    objects themselves are real and shared across the session.
    """
    ap = _real_session_preprocessor
    saved = (ap.vad_select_mode, ap.vad_concat_padding_ms)
    try:
        yield ap
    finally:
        ap.vad_select_mode, ap.vad_concat_padding_ms = saved
