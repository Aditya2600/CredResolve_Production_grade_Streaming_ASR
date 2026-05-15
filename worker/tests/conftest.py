"""Shared fixtures for worker/tests/.

The characterization tests in test_audio_characterization.py need the real
Silero VAD and a real RNNoise binding loaded once per session. These fixtures:

- Verify that Silero VAD is available, either via the `silero-vad` PyPI package
  (preferred, bundled weights) or via the legacy torch.hub cache directory.
  Tests SKIP cleanly with a clear hint when neither is present.
  We never call ``torch.hub.load(force_reload=True)`` so tests stay offline.
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
_SILERO_HUB_REQUIRED_FILES = (
    _SILERO_HUB_DIR / "hubconf.py",
    _SILERO_HUB_DIR / "src" / "silero_vad" / "data" / "silero_vad.jit",
)
_SILERO_SETUP_HINT = (
    "Install the silero-vad PyPI package (preferred, no git needed):\n"
    "    pip install 'silero-vad>=5.1.2'\n"
    "Or pre-cache the hub model once with internet access:\n"
    "    python -c 'import torch; torch.hub.load(\"snakers4/silero-vad\", \"silero_vad\")'\n"
    f"The hub path would be {_SILERO_HUB_DIR}."
)


def _silero_vad_available() -> bool:
    """Return True if Silero VAD can be loaded without network access.

    Checks the PyPI package first (always available once pip-installed),
    then the legacy torch.hub cache directory.
    """
    try:
        import silero_vad  # noqa: F401
        return True
    except Exception:
        pass
    return all(p.is_file() for p in _SILERO_HUB_REQUIRED_FILES)


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

    Skips the dependent tests with a clear hint if Silero VAD is unavailable
    (neither the PyPI package nor the torch.hub cache exists).
    We deliberately do not touch the network at test time.
    """
    if not _silero_vad_available():
        pytest.skip(
            "Silero VAD not available (neither silero-vad PyPI package nor "
            f"hub cache at {_SILERO_HUB_DIR}).\n{_SILERO_SETUP_HINT}"
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
