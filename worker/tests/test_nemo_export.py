from __future__ import annotations

from pathlib import Path

import pytest

from worker.app.nemo_export import (
    _load_nemo_model,
    _normalize_nemo_restore_config,
    is_local_nemo_source,
    NemoExportError,
    normalize_model_class_name,
    select_supported_kwargs,
)


def _fake_target(required: int, optional: int = 0):
    return required + optional


def test_normalize_model_class_name_is_case_insensitive():
    assert normalize_model_class_name("encdecrnntbpemodel") == "EncDecRNNTBPEModel"


def test_normalize_model_class_name_rejects_unknown_values():
    with pytest.raises(ValueError):
        normalize_model_class_name("DefinitelyNotAModel")


def test_is_local_nemo_source_detects_nemo_suffix_even_if_missing(tmp_path: Path):
    candidate = tmp_path / "checkpoint.nemo"

    assert is_local_nemo_source(str(candidate)) is True


def test_is_local_nemo_source_detects_existing_files(tmp_path: Path):
    candidate = tmp_path / "checkpoint.ckpt"
    candidate.write_bytes(b"test")

    assert is_local_nemo_source(str(candidate)) is True


def test_is_local_nemo_source_keeps_remote_model_names_remote():
    assert is_local_nemo_source("nvidia/parakeet-ctc-1.1b") is False


def test_select_supported_kwargs_filters_unknown_options():
    def fake_export(output: str, check_trace: bool = False, verbose: bool = False):
        return output, check_trace, verbose

    selected = select_supported_kwargs(
        fake_export,
        check_trace=False,
        verbose=True,
        use_dynamo=False,
        onnx_opset_version=17,
    )

    assert selected == {"check_trace": False, "verbose": True}


def test_normalize_nemo_restore_config_rewrites_multilingual_tokenizer_type():
    normalized = _normalize_nemo_restore_config(
        {
            "tokenizer": {
                "type": "multilingual",
                "langs": {
                    "hi": {
                        "dir": "/tmp/hi",
                        "type": "bpe",
                    }
                },
            },
            "decoder": {
                "_target_": "nemo.collections.asr.modules.RNNTDecoder",
                "vocab_size": 5632,
                "multisoftmax": True,
            },
            "joint": {
                "_target_": "worker.tests.test_nemo_export._fake_target",
                "required": 7,
                "optional": 2,
                "unsupported": 99,
            },
        }
    )

    assert normalized["tokenizer"]["type"] == "agg"
    assert normalized["tokenizer"]["langs"]["hi"]["dir"] == "/tmp/hi"
    assert "multisoftmax" not in normalized["decoder"]
    assert normalized["joint"] == {
        "_target_": "worker.tests.test_nemo_export._fake_target",
        "required": 7,
        "optional": 2,
    }


def test_load_nemo_model_falls_back_to_hf_nemo_archive(monkeypatch, tmp_path: Path):
    archive_path = tmp_path / "model.nemo"
    archive_path.write_bytes(b"nemo")
    override_path = tmp_path / "override.yaml"
    override_path.write_text("tokenizer:\n  type: agg\n", encoding="utf-8")
    calls: list[tuple[str, object, dict[str, object]]] = []

    class FakeModelClass:
        @staticmethod
        def from_pretrained(source: str, map_location=None, strict=False):
            calls.append(
                (
                    "from_pretrained",
                    source,
                    {"map_location": map_location, "strict": strict},
                )
            )
            raise FileNotFoundError("missing model_config.yaml")

        @staticmethod
        def restore_from(source: str, map_location=None, override_config_path=None):
            kwargs = {
                "map_location": map_location,
                "override_config_path": override_config_path,
            }
            calls.append(("restore_from", source, kwargs))
            return {"loaded_from": source, "kwargs": kwargs}

    monkeypatch.setattr(
        "worker.app.nemo_export._resolve_nemo_model_class",
        lambda _: ("ASRModel", FakeModelClass),
    )
    monkeypatch.setattr("worker.app.nemo_export._download_hf_nemo_archive", lambda _: archive_path)
    monkeypatch.setattr("worker.app.nemo_export._write_nemo_restore_override_config", lambda _: override_path)

    model_class, load_method, model = _load_nemo_model(
        source="ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
        model_class_name="ASRModel",
        device="cuda",
    )

    assert model_class == "ASRModel"
    assert load_method == "restore_from_hf_hub"
    assert model == {
        "loaded_from": str(archive_path),
        "kwargs": {
            "map_location": "cuda",
            "override_config_path": str(override_path),
        },
    }
    assert calls == [
        (
            "from_pretrained",
            "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
            {"map_location": "cuda", "strict": False},
        ),
        (
            "restore_from",
            str(archive_path),
            {
                "map_location": "cuda",
                "override_config_path": str(override_path),
            },
        ),
    ]
    assert override_path.exists() is False


def test_load_nemo_model_applies_compat_override_for_local_archive(monkeypatch, tmp_path: Path):
    archive_path = tmp_path / "model.nemo"
    archive_path.write_bytes(b"nemo")
    override_path = tmp_path / "override.yaml"
    override_path.write_text("tokenizer:\n  type: agg\n", encoding="utf-8")
    calls: list[tuple[str, object, dict[str, object]]] = []

    class FakeModelClass:
        @staticmethod
        def restore_from(source: str, map_location=None, override_config_path=None):
            kwargs = {
                "map_location": map_location,
                "override_config_path": override_config_path,
            }
            calls.append(("restore_from", source, kwargs))
            return {"loaded_from": source, "kwargs": kwargs}

    monkeypatch.setattr(
        "worker.app.nemo_export._resolve_nemo_model_class",
        lambda _: ("ASRModel", FakeModelClass),
    )
    monkeypatch.setattr("worker.app.nemo_export._write_nemo_restore_override_config", lambda _: override_path)

    model_class, load_method, model = _load_nemo_model(
        source=str(archive_path),
        model_class_name="ASRModel",
        device="cuda",
    )

    assert model_class == "ASRModel"
    assert load_method == "restore_from"
    assert model == {
        "loaded_from": str(archive_path),
        "kwargs": {
            "map_location": "cuda",
            "override_config_path": str(override_path),
        },
    }
    assert calls == [
        (
            "restore_from",
            str(archive_path),
            {
                "map_location": "cuda",
                "override_config_path": str(override_path),
            },
        )
    ]
    assert override_path.exists() is False


def test_download_hf_nemo_archive_requires_single_archive(monkeypatch):
    monkeypatch.setattr(
        "worker.app.nemo_export._import_huggingface_hub",
        lambda: (
            lambda **_: "/tmp/ignored.nemo",
            lambda **_: ["first.nemo", "second.nemo"],
        ),
    )

    with pytest.raises(NemoExportError, match="multiple `.nemo` archives"):
        from worker.app.nemo_export import _download_hf_nemo_archive

        _download_hf_nemo_archive("ai4bharat/example")
