from __future__ import annotations

import sys
from types import SimpleNamespace

from tools import indicvoices_dataset


def test_load_hf_speech_stream_uses_streaming_dataset_and_decodes_false(monkeypatch):
    calls = {}

    class FakeDataset:
        def decode(self, value):
            calls["decode"] = value
            return [{"ok": True}]

    def fake_load_dataset(dataset_id, dataset_config, split, **kwargs):
        calls["load_dataset"] = {
            "dataset_id": dataset_id,
            "dataset_config": dataset_config,
            "split": split,
            "kwargs": kwargs,
        }
        return FakeDataset()

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    rows = indicvoices_dataset.load_hf_speech_stream(
        dataset_id="demo/dataset",
        dataset_config="audio/Hindi",
        split="train",
        token="token-123",
        cache_dir="/tmp/hf-cache",
    )

    assert rows == [{"ok": True}]
    assert calls["decode"] is False
    assert calls["load_dataset"] == {
        "dataset_id": "demo/dataset",
        "dataset_config": "audio/Hindi",
        "split": "train",
        "kwargs": {
            "token": "token-123",
            "streaming": True,
            "cache_dir": "/tmp/hf-cache",
        },
    }


def test_load_hf_speech_stream_uses_explicit_parquet_fallback(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        del args, kwargs
        raise ValueError("Invalid pattern: '**' can only be an entire path component")

    def fake_build_explicit_parquet_paths(**kwargs):
        calls["build_paths"] = kwargs
        return ["audio/Hindi/train-00000-of-00001.parquet"]

    def fake_iter_explicit_parquet_rows(**kwargs):
        calls["iter_rows"] = kwargs
        return iter([{"transcript": "hello"}])

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setattr(indicvoices_dataset, "_build_explicit_parquet_paths", fake_build_explicit_parquet_paths)
    monkeypatch.setattr(indicvoices_dataset, "_iter_explicit_parquet_rows", fake_iter_explicit_parquet_rows)

    rows = list(
        indicvoices_dataset.load_hf_speech_stream(
            dataset_id="demo/dataset",
            dataset_config="audio/Hindi",
            split="train",
            token="token-123",
            cache_dir=None,
        )
    )

    assert rows == [{"transcript": "hello"}]
    assert calls["build_paths"] == {
        "dataset_id": "demo/dataset",
        "dataset_config": "audio/Hindi",
        "split": "train",
        "token": "token-123",
    }
    assert calls["iter_rows"] == {
        "dataset_id": "demo/dataset",
        "parquet_paths": ["audio/Hindi/train-00000-of-00001.parquet"],
        "token": "token-123",
    }


def test_load_vaani_stream_defaults_to_hindi_train(monkeypatch):
    calls = {}

    def fake_load_hf_speech_stream(**kwargs):
        calls.update(kwargs)
        return [{"ok": True}]

    monkeypatch.setattr(indicvoices_dataset, "load_hf_speech_stream", fake_load_hf_speech_stream)

    rows = indicvoices_dataset.load_vaani_stream(token="token-123", cache_dir=None)

    assert rows == [{"ok": True}]
    assert calls["dataset_id"] == indicvoices_dataset.VAANI_DATASET_ID
    assert calls["dataset_config"] == "audio/Hindi"
    assert calls["split"] == "train"
