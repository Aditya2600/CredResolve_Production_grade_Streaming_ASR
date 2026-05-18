from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import worker.app.context_biasing as context_biasing_module
from worker.app.context_biasing import (
    ContextBiasingConfig,
    ContextBiasingError,
    ContextBiasingModelPool,
    ContextBiasingTimeoutError,
    NeMoContextBiasingRuntime,
)


PCM16 = b"\x01\x00" * 160


@dataclass
class _FakeModel:
    name: str
    applied_phrase_files: list[str] = field(default_factory=list)
    current_phrase_file: str = ""


class _ObservedMetric:
    def __init__(self):
        self.values: list[float] = []

    def observe(self, value):
        self.values.append(value)


class _GaugeMetric:
    def __init__(self):
        self.values: list[float] = []

    def set(self, value):
        self.values.append(value)


def _runtime_with_models(
    models: list[_FakeModel],
    *,
    max_concurrency: int | None = None,
    timeout_ms: int = 500,
    queue_timeout_ms: int = 500,
    transcribe_impl=None,
) -> NeMoContextBiasingRuntime:
    runtime = NeMoContextBiasingRuntime(
        ContextBiasingConfig(
            mode="active",
            method="ctc_ws",
            nemo_source="demo.nemo",
            nemo_model_class="EncDecCTCModelBPE",
            phrases_dir="",
            timeout_ms=timeout_ms,
            device="cpu",
            shadow_sample_rate=1.0,
            beam_threshold=8.0,
            context_score=3.0,
            ctc_ali_token_weight=0.6,
            max_dynamic_phrases=32,
            max_concurrent_inferences=max_concurrency or len(models),
            executor_workers=max_concurrency or len(models),
            queue_timeout_ms=queue_timeout_ms,
            model_pool_size=len(models),
        )
    )
    runtime.ready = True
    runtime.model = models[0]
    runtime.model_pool = ContextBiasingModelPool.from_models(models)

    def fake_apply(*, phrase_file: str, model=None):
        model.current_phrase_file = phrase_file
        model.applied_phrase_files.append(phrase_file)
        return "ctc"

    def fake_transcribe(_wav_path, *, language: str, model=None):
        if transcribe_impl is not None:
            return transcribe_impl(model, language)
        return model.current_phrase_file

    runtime._apply_decoding_strategy = fake_apply
    runtime._transcribe_file = fake_transcribe
    runtime._refresh_pool_metrics()
    return runtime


def _transcribe(runtime: NeMoContextBiasingRuntime, phrase_file: str):
    return runtime.transcribe_with_timeout(
        pcm16le=PCM16,
        sample_rate=16000,
        language="hi",
        phrase_file=phrase_file,
        session_id=f"session-{phrase_file}",
        utterance_id=f"utt-{phrase_file}",
        mode="active",
    )


def test_pool_size_one_serializes_concurrent_work():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_transcribe(model, _language):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return model.current_phrase_file

    runtime = _runtime_with_models([_FakeModel("m0")], transcribe_impl=fake_transcribe)

    async def scenario():
        return await asyncio.gather(_transcribe(runtime, "a.txt"), _transcribe(runtime, "b.txt"))

    results = asyncio.run(scenario())

    assert [result.text for result in results] == ["a.txt", "b.txt"]
    assert max_active == 1


def test_pool_size_two_allows_distinct_concurrent_decodes_without_phrase_cross_contamination():
    models = [_FakeModel("m0"), _FakeModel("m1")]
    barrier = threading.Barrier(2)
    active_model_ids: set[int] = set()
    lock = threading.Lock()

    def fake_transcribe(model, _language):
        with lock:
            assert id(model) not in active_model_ids
            active_model_ids.add(id(model))
        barrier.wait(timeout=1.0)
        observed_phrase = model.current_phrase_file
        time.sleep(0.01)
        with lock:
            active_model_ids.remove(id(model))
        return observed_phrase

    runtime = _runtime_with_models(models, transcribe_impl=fake_transcribe)

    async def scenario():
        return await asyncio.gather(_transcribe(runtime, "a.txt"), _transcribe(runtime, "b.txt"))

    results = asyncio.run(scenario())

    assert sorted(result.text for result in results) == ["a.txt", "b.txt"]
    assert sorted(model.applied_phrase_files for model in models) == [["a.txt"], ["b.txt"]]
    assert runtime.model_pool is not None
    assert runtime.model_pool.available_count == 2


def test_queue_timeout_has_distinct_reason():
    started = threading.Event()
    finish = threading.Event()

    def fake_transcribe(model, _language):
        started.set()
        finish.wait(timeout=1.0)
        return model.current_phrase_file

    runtime = _runtime_with_models(
        [_FakeModel("m0")],
        timeout_ms=500,
        queue_timeout_ms=20,
        transcribe_impl=fake_transcribe,
    )

    async def scenario():
        first = asyncio.create_task(_transcribe(runtime, "a.txt"))
        while not started.is_set():
            await asyncio.sleep(0.001)
        with pytest.raises(ContextBiasingTimeoutError) as excinfo:
            await _transcribe(runtime, "b.txt")
        finish.set()
        await first
        return excinfo.value

    error = asyncio.run(scenario())

    assert error.reason == "queue_timeout"


def test_inference_timeout_drains_until_underlying_transcribe_finishes():
    started = threading.Event()
    finish = threading.Event()

    def fake_transcribe(model, _language):
        started.set()
        finish.wait(timeout=1.0)
        return model.current_phrase_file

    runtime = _runtime_with_models(
        [_FakeModel("m0")],
        timeout_ms=20,
        queue_timeout_ms=100,
        transcribe_impl=fake_transcribe,
    )

    async def scenario():
        with pytest.raises(ContextBiasingTimeoutError) as excinfo:
            await _transcribe(runtime, "a.txt")
        assert started.is_set()
        assert runtime.model_pool is not None
        assert runtime.model_pool.available_count == 0
        assert runtime.model_pool.slots[0].state == "draining_after_timeout"
        finish.set()
        for _ in range(100):
            if runtime.model_pool.available_count == 1:
                break
            await asyncio.sleep(0.005)
        return excinfo.value

    error = asyncio.run(scenario())

    assert error.reason == "inference_timeout"
    assert runtime.model_pool is not None
    assert runtime.model_pool.available_count == 1
    assert runtime.model_pool.slots[0].state == "available"


def test_inference_timeout_defers_request_phrase_file_cleanup_until_worker_finishes(tmp_path: Path):
    started = threading.Event()
    finish = threading.Event()
    phrase_file = tmp_path / "dynamic.txt"
    phrase_file.write_text("ravi kumar\n", encoding="utf-8")

    def fake_transcribe(model, _language):
        started.set()
        finish.wait(timeout=1.0)
        return model.current_phrase_file

    runtime = _runtime_with_models(
        [_FakeModel("m0")],
        timeout_ms=20,
        queue_timeout_ms=100,
        transcribe_impl=fake_transcribe,
    )

    async def scenario():
        with pytest.raises(ContextBiasingTimeoutError):
            await runtime.transcribe_with_timeout(
                pcm16le=PCM16,
                sample_rate=16000,
                language="hi",
                phrase_file=str(phrase_file),
                session_id="session-a",
                utterance_id="utt-a",
                mode="active",
                cleanup_phrase_file=True,
            )
        assert phrase_file.exists()
        finish.set()
        for _ in range(100):
            if not phrase_file.exists():
                break
            await asyncio.sleep(0.005)

    asyncio.run(scenario())

    assert not phrase_file.exists()


def test_failed_model_is_not_reused_unsafely():
    calls = 0

    def fake_transcribe(_model, _language):
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    runtime = _runtime_with_models(
        [_FakeModel("m0")],
        timeout_ms=100,
        queue_timeout_ms=20,
        transcribe_impl=fake_transcribe,
    )

    async def scenario():
        with pytest.raises(ContextBiasingError):
            await _transcribe(runtime, "a.txt")
        for _ in range(100):
            if runtime.model_pool and runtime.model_pool.failed_count == 1:
                break
            await asyncio.sleep(0.001)
        with pytest.raises(ContextBiasingTimeoutError) as excinfo:
            await _transcribe(runtime, "b.txt")
        return excinfo.value

    error = asyncio.run(scenario())

    assert error.reason == "queue_timeout"
    assert calls == 1
    assert runtime.model_pool is not None
    assert runtime.model_pool.failed_count == 1


def test_logs_and_metrics_distinguish_queue_wait_from_inference_latency(monkeypatch, caplog):
    queue_metric = _ObservedMetric()
    inference_metric = _ObservedMetric()
    total_metric = _ObservedMetric()
    monkeypatch.setattr(context_biasing_module, "CONTEXT_BIASING_QUEUE_WAIT_MS", queue_metric)
    monkeypatch.setattr(context_biasing_module, "CONTEXT_BIASING_LATENCY", inference_metric)
    monkeypatch.setattr(context_biasing_module, "CONTEXT_BIASING_TOTAL_LATENCY", total_metric)
    monkeypatch.setattr(context_biasing_module, "CONTEXT_BIASING_INFLIGHT", _GaugeMetric())
    monkeypatch.setattr(context_biasing_module, "CONTEXT_BIASING_POOL_AVAILABLE", _GaugeMetric())

    runtime = _runtime_with_models([_FakeModel("m0")], transcribe_impl=lambda model, _lang: model.current_phrase_file)

    async def scenario():
        with caplog.at_level(logging.INFO, logger="worker.context_biasing"):
            result = await _transcribe(runtime, "a.txt")
            await asyncio.sleep(0)
        return result

    result = asyncio.run(scenario())
    attempt_payloads = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "worker.context_biasing" and record.message.startswith("{")
    ]

    assert result.text == "a.txt"
    assert queue_metric.values
    assert inference_metric.values
    assert total_metric.values
    assert attempt_payloads[-1]["bias_timeout_reason"] is None
    assert "bias_queue_wait_ms" in attempt_payloads[-1]
