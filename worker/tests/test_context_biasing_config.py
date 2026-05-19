from __future__ import annotations

import importlib
import logging

import worker.app.config as config
from worker.app.context_biasing import ContextBiasingConfig, NeMoContextBiasingRuntime


_CONCURRENCY_ENV_VARS = (
    "ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES",
    "ASR_CONTEXT_BIASING_EXECUTOR_WORKERS",
    "ASR_CONTEXT_BIASING_QUEUE_TIMEOUT_MS",
    "ASR_CONTEXT_BIASING_MODEL_POOL_SIZE",
    "ASR_CONTEXT_BIASING_MODEL_POOL_LOAD_MODE",
)


def test_context_biasing_config_defaults_preserve_single_concurrency(monkeypatch):
    for name in _CONCURRENCY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASR_CONTEXT_BIASING_TIMEOUT_MS", "4321")

    importlib.reload(config)
    try:
        assert config.ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES == 1
        assert config.ASR_CONTEXT_BIASING_EXECUTOR_WORKERS == 1
        assert config.ASR_CONTEXT_BIASING_QUEUE_TIMEOUT_MS == 4321
        assert config.ASR_CONTEXT_BIASING_MODEL_POOL_SIZE == 1
        assert config.ASR_CONTEXT_BIASING_EFFECTIVE_MAX_CONCURRENCY == 1
        assert config.ASR_CONTEXT_BIASING_MODEL_POOL_LOAD_MODE == "eager"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_context_biasing_config_reads_executor_and_max_concurrency(monkeypatch):
    monkeypatch.setenv("ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES", "3")
    monkeypatch.setenv("ASR_CONTEXT_BIASING_EXECUTOR_WORKERS", "5")
    monkeypatch.setenv("ASR_CONTEXT_BIASING_QUEUE_TIMEOUT_MS", "250")
    monkeypatch.setenv("ASR_CONTEXT_BIASING_MODEL_POOL_SIZE", "4")
    monkeypatch.setenv("ASR_CONTEXT_BIASING_MODEL_POOL_LOAD_MODE", "lazy")

    importlib.reload(config)
    try:
        assert config.ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES == 3
        assert config.ASR_CONTEXT_BIASING_EXECUTOR_WORKERS == 5
        assert config.ASR_CONTEXT_BIASING_QUEUE_TIMEOUT_MS == 250
        assert config.ASR_CONTEXT_BIASING_MODEL_POOL_SIZE == 4
        assert config.ASR_CONTEXT_BIASING_EFFECTIVE_MAX_CONCURRENCY == 3
        assert config.ASR_CONTEXT_BIASING_MODEL_POOL_LOAD_MODE == "lazy"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_context_biasing_config_inherits_executor_workers_from_max_concurrency(monkeypatch):
    for name in _CONCURRENCY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES", "3")
    monkeypatch.setenv("ASR_CONTEXT_BIASING_MODEL_POOL_SIZE", "3")

    importlib.reload(config)
    try:
        assert config.ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES == 3
        assert config.ASR_CONTEXT_BIASING_EXECUTOR_WORKERS == 3
        assert config.ASR_CONTEXT_BIASING_EFFECTIVE_MAX_CONCURRENCY == 3
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_context_biasing_config_clamps_effective_concurrency(monkeypatch, caplog):
    for name in _CONCURRENCY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES", "4")
    monkeypatch.setenv("ASR_CONTEXT_BIASING_MODEL_POOL_SIZE", "2")

    with caplog.at_level(logging.WARNING, logger="worker.config"):
        importlib.reload(config)
    try:
        assert config.ASR_CONTEXT_BIASING_MAX_CONCURRENT_INFERENCES == 4
        assert config.ASR_CONTEXT_BIASING_MODEL_POOL_SIZE == 2
        assert config.ASR_CONTEXT_BIASING_EFFECTIVE_MAX_CONCURRENCY == 2
        assert "clamping" in caplog.text
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_context_biasing_runtime_clamps_unsafe_concurrency(caplog):
    with caplog.at_level(logging.WARNING, logger="worker.context_biasing"):
        runtime = NeMoContextBiasingRuntime(
            ContextBiasingConfig(
                mode="active",
                method="ctc_ws",
                nemo_source="demo.nemo",
                nemo_model_class="EncDecCTCModelBPE",
                phrases_dir="",
                timeout_ms=100,
                device="cpu",
                shadow_sample_rate=1.0,
                beam_threshold=8.0,
                context_score=3.0,
                ctc_ali_token_weight=0.6,
                max_dynamic_phrases=32,
                max_concurrent_inferences=4,
                executor_workers=4,
                queue_timeout_ms=100,
                model_pool_size=2,
            )
        )

    assert runtime.configured_max_concurrency == 4
    assert runtime.effective_max_concurrency == 2
    assert runtime._slots._bound_value == 2
    assert "clamping" in caplog.text
