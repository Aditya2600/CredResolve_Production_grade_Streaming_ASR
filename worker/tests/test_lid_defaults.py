from __future__ import annotations

import importlib

import worker.app.config as config
from worker.app.model import ONNXIndicASRWorker


def test_worker_defaults_to_vakgyata_primary_with_speechbrain_fallback():
    worker = ONNXIndicASRWorker(
        model_name="test-model",
        default_decoder="rnnt",
        hf_token="",
        inference_timeout_ms=200,
        default_language="hi",
    )

    assert worker.lid_model_source == "onecxi/vakgyata-small"
    assert worker.lid_primary_provider == "vakgyata"
    assert worker.lid_primary_source == "onecxi/vakgyata-small"
    assert worker.lid_primary_model_dir == "models/lid_primary"
    assert worker.lid_fallback_provider == "speechbrain"
    assert worker.lid_fallback_source == "speechbrain/lang-id-voxlingua107-ecapa"
    assert worker.lid_fallback_model_dir == "models/lid_fallback"


def test_config_uses_default_chain_when_legacy_vars_are_unset(monkeypatch):
    for name in (
        "ASR_LID_MODEL_SOURCE",
        "ASR_LID_MODEL_DIR",
        "ASR_LID_PRIMARY_PROVIDER",
        "ASR_LID_PRIMARY_SOURCE",
        "ASR_LID_PRIMARY_MODEL_DIR",
        "ASR_LID_FALLBACK_PROVIDER",
        "ASR_LID_FALLBACK_SOURCE",
        "ASR_LID_FALLBACK_MODEL_DIR",
        "ASR_LID_CONFIDENCE_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)

    importlib.reload(config)
    try:
        assert config.ASR_LID_PRIMARY_PROVIDER == "vakgyata"
        assert config.ASR_LID_PRIMARY_SOURCE == "onecxi/vakgyata-small"
        assert config.ASR_LID_FALLBACK_PROVIDER == "speechbrain"
        assert config.ASR_LID_FALLBACK_SOURCE == "speechbrain/lang-id-voxlingua107-ecapa"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_config_preserves_explicit_legacy_single_provider(monkeypatch):
    for name in (
        "ASR_LID_PRIMARY_PROVIDER",
        "ASR_LID_PRIMARY_SOURCE",
        "ASR_LID_PRIMARY_MODEL_DIR",
        "ASR_LID_FALLBACK_PROVIDER",
        "ASR_LID_FALLBACK_SOURCE",
        "ASR_LID_FALLBACK_MODEL_DIR",
        "ASR_LID_CONFIDENCE_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASR_LID_MODEL_SOURCE", "speechbrain/lang-id-voxlingua107-ecapa")
    monkeypatch.setenv("ASR_LID_MODEL_DIR", "models/lid_model")

    importlib.reload(config)
    try:
        assert config.ASR_LID_PRIMARY_PROVIDER == "speechbrain"
        assert config.ASR_LID_PRIMARY_SOURCE == "speechbrain/lang-id-voxlingua107-ecapa"
        assert config.ASR_LID_PRIMARY_MODEL_DIR == "models/lid_model"
        assert config.ASR_LID_FALLBACK_PROVIDER == ""
        assert config.ASR_LID_FALLBACK_SOURCE == ""
        assert config.ASR_LID_FALLBACK_MODEL_DIR == ""
    finally:
        monkeypatch.undo()
        importlib.reload(config)
