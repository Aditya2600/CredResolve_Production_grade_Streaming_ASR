from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.asr_text_normalizer import normalize_asr_text
from tools.check_tokenizer_coverage import check_rows, tokenize_text
from tools.normalize_manifest_text import normalize_manifest_file
from tools.prepare_vaani_8khz_manifest import prepare_splits
from tools.run_nemo_adapter_peft import (
    assert_tokenizer_unchanged,
    configure_adapter_optimizer_cfg,
    disable_metric_prediction_logging,
    inspect_matching_module_names,
    log_adapter_gradient_norms_to_tensorboard,
    log_adapter_tensorboard_metadata,
    parse_args as parse_adapter_args,
    parse_module_name_patterns,
    resolve_validation_manifest,
    sample_manifest_rows,
    summarize_trainable_parameters,
    tensorboard_tag_component,
    tokenizer_fingerprint,
    trainable_gradient_norms,
    validate_manifest_text,
    validate_trainable_parameters,
)
from tools.split_vaani_manifest import split_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_normalize_asr_text_cleans_vaani_annotations_without_transliteration():
    text = "\u200b<noise> हिंदी [inhaling] {cement} -- ... \u2026! </noise> <pause>"

    assert normalize_asr_text(text, "hi") == "हिंदी"


def test_normalize_asr_text_drops_latin_glosses_spans_and_keeps_indic_braces():
    text = (
        "પાસે એક સફેદ ટેબલ {table} છે જેના પર પર્પલ {purple} "
        "કલર {colour} ની એક પાણીની બોટલ {bottle} છે. {પર્પલ} "
        "State Bank Of India class ડેન્જરdangerous 4G"
    )

    assert normalize_asr_text(text, "gu") == (
        "પાસે એક સફેદ ટેબલ છે જેના પર પર્પલ કલર ની એક પાણીની બોટલ છે પર્પલ ડેન્જર 4"
    )


def test_normalize_asr_text_normalizes_partial_words_nukta_digits_and_symbols():
    text = "स्-बालू क़िला १२,૫૦० | ೪G mobile"

    assert normalize_asr_text(text, "hi") == "बालू क़िला 12 500 4"


def test_normalize_asr_text_rejects_unintelligible_markers_and_preserves_indic():
    assert normalize_asr_text("<unintelligible>") == ""
    assert normalize_asr_text("ಕನ್ನಡ ಪಠ್ಯ।") == "ಕನ್ನಡ ಪಠ್ಯ"


def test_normalize_manifest_file_writes_raw_text_and_rejects(tmp_path: Path):
    src = tmp_path / "manifest.jsonl"
    out = tmp_path / "normalized.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps({"id": "ok", "text": "<noise> नमस्ते {test}. </noise>", "lang": "hi"}, ensure_ascii=False),
                json.dumps({"id": "short", "text": "अ", "lang": "hi"}, ensure_ascii=False),
                json.dumps({"id": "long", "text": "x" * 30, "lang": "hi"}, ensure_ascii=False),
                json.dumps({"id": "words", "text": "one two three", "lang": "hi"}, ensure_ascii=False),
                "{not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = normalize_manifest_file(src, out, rejects, min_chars=2, max_chars=20, max_words=2)

    kept = _read_jsonl(out)
    rejected = _read_jsonl(rejects)
    assert summary["kept"] == 1
    assert summary["rejected"] == 4
    assert kept[0]["raw_text"] == "<noise> नमस्ते {test}. </noise>"
    assert kept[0]["text"] == "नमस्ते"
    assert {row["reject_reason"] for row in rejected} == {
        "too_short",
        "too_long",
        "too_many_words",
        "invalid_json",
    }


def test_split_rows_is_stratified_deterministic_and_non_overlapping():
    rows = [
        {"id": f"hi-{i}", "audio_filepath": f"/a/hi-{i}.wav", "lang": "hi", "duration": 1.0}
        for i in range(10)
    ] + [
        {"id": f"ta-{i}", "audio_filepath": f"/a/ta-{i}.wav", "lang": "ta", "duration": 1.0}
        for i in range(10)
    ]

    first = split_rows(rows, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2, seed=7)
    second = split_rows(rows, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2, seed=7)

    assert first == second
    assert {name: len(split) for name, split in first.items()} == {"train": 12, "dev": 4, "test": 4}
    split_by_id = {
        row["id"]: split_name
        for split_name, split in first.items()
        for row in split
    }
    assert len(split_by_id) == len(rows)


def test_split_rows_rejects_duplicate_audio_or_id():
    rows = [
        {"id": "same", "audio_filepath": "/a/1.wav", "lang": "hi"},
        {"id": "same", "audio_filepath": "/a/2.wav", "lang": "hi"},
    ]

    with pytest.raises(SystemExit):
        split_rows(rows, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=1)


def test_prepare_vaani_8khz_manifest_rewrites_audio_and_preserves_metadata(tmp_path: Path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    source_audio = tmp_path / "source.wav"
    sf.write(source_audio, np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
    input_dir = tmp_path / "input"
    _write_jsonl(
        input_dir / "train.jsonl",
        [
            {
                "id": "row/one",
                "audio_filepath": str(source_audio),
                "text": "hello |",
                "lang": "hi",
                "duration": 0.1,
            }
        ],
    )

    out_dir = tmp_path / "out"
    summary = prepare_splits(
        input_dir,
        out_dir,
        splits=("train",),
        target_sample_rate=8000,
        model_sample_rate=16000,
        progress_every=0,
    )

    rows = _read_jsonl(out_dir / "train.jsonl")
    assert summary["target_sample_rate"] == 8000
    assert rows[0]["source_audio_filepath"] == str(source_audio.resolve())
    assert rows[0]["raw_text"] == "hello |"
    assert rows[0]["text"] == "hello"
    assert rows[0]["source_sample_rate"] == 16000
    assert rows[0]["telephony_sample_rate"] == 8000
    assert rows[0]["model_sample_rate"] == 16000
    assert rows[0]["telephony_simulation"] == "downsampled_to_8khz"
    assert sf.info(rows[0]["audio_filepath"]).samplerate == 8000


class _FakeTokenizer:
    def text_to_ids(self, text: str, lang: str | None = None):
        assert lang == "hi"
        return [ord(char) for char in text]


class _FakeTokenizerModel:
    tokenizer = _FakeTokenizer()


def test_tokenizer_coverage_uses_language_aware_text_to_ids():
    rows = [{"id": "one", "text": "नमस्ते", "lang": "hi"}]

    assert tokenize_text(_FakeTokenizerModel(), "नमस्ते", "hi")[0] is True
    summary = check_rows(_FakeTokenizerModel(), rows, max_examples=3)
    assert summary["failed_rows"] == 0
    assert summary["methods"] == {"_FakeTokenizer.text_to_ids": 1}


class _VocabOnlyModel:
    labels = ["a", "b"]


def test_tokenizer_coverage_reports_bad_examples_with_vocab_fallback():
    summary = check_rows(_VocabOnlyModel(), [{"text": "abc", "lang": "hi"}], max_examples=1)

    assert summary["failed_rows"] == 1
    assert summary["bad_examples"][0]["error"] == "missing characters: c"


def test_adapter_cli_defaults_are_adapter_safe(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nemo_adapter_peft.py",
            "--model",
            "/tmp/model.nemo",
            "--train-manifest",
            "/tmp/train.jsonl",
            "--val-manifest",
            "/tmp/dev.jsonl",
            "--exp-dir",
            "/tmp/exp",
            "--name",
            "run",
        ],
    )

    args = parse_adapter_args()

    assert args.batch_size == 1
    assert args.val_batch_size == 1
    assert args.accumulate_grad_batches == 4
    assert args.lr == 1e-3
    assert args.max_steps == 1000
    assert args.warmup_steps == 100
    assert args.val_check_interval == 200
    assert args.validation_mode == "diagnostic"
    assert args.diagnostic_val_size == 200
    assert args.diagnostic_val_seed == 42
    assert args.max_duration == 8.0
    assert args.val_max_duration == 10.0
    assert args.precision == "16-mixed"
    assert args.resume_from_checkpoint is None
    assert args.print_grad_norms is False
    assert args.grad_norm_log_every_n_steps == 1


def test_adapter_cli_accepts_resume_checkpoint(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nemo_adapter_peft.py",
            "--model",
            "/tmp/model.nemo",
            "--train-manifest",
            "/tmp/train.jsonl",
            "--val-manifest",
            "/tmp/dev.jsonl",
            "--exp-dir",
            "/tmp/exp",
            "--name",
            "run",
            "--resume-from-checkpoint",
            "/tmp/run/checkpoints/last.ckpt",
        ],
    )

    args = parse_adapter_args()

    assert args.resume_from_checkpoint == Path("/tmp/run/checkpoints/last.ckpt")


def test_adapter_cli_accepts_final_validation_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nemo_adapter_peft.py",
            "--model",
            "/tmp/model.nemo",
            "--train-manifest",
            "/tmp/train.jsonl",
            "--val-manifest",
            "/tmp/dev.jsonl",
            "--exp-dir",
            "/tmp/exp",
            "--name",
            "run",
            "--validation-mode",
            "final",
        ],
    )

    args = parse_adapter_args()

    assert args.validation_mode == "final"


def test_diagnostic_manifest_sampling_is_deterministic_and_ordered():
    rows = [{"id": str(index), "text": "नमस्ते", "lang": "hi"} for index in range(20)]

    sample_a, indices_a = sample_manifest_rows(rows, sample_size=5, seed=123)
    sample_b, indices_b = sample_manifest_rows(rows, sample_size=5, seed=123)

    assert indices_a == indices_b
    assert indices_a == sorted(indices_a)
    assert [row["id"] for row in sample_a] == [row["id"] for row in sample_b]
    assert len(sample_a) == 5


def test_resolve_validation_manifest_writes_diagnostic_sample(tmp_path: Path):
    manifest = tmp_path / "dev.jsonl"
    _write_jsonl(
        manifest,
        [{"id": str(index), "text": "नमस्ते", "lang": "hi"} for index in range(20)],
    )

    effective_manifest, summary = resolve_validation_manifest(
        manifest,
        run_dir=tmp_path / "run",
        mode="diagnostic",
        diagnostic_val_size=5,
        diagnostic_val_seed=123,
    )

    sampled_rows = _read_jsonl(effective_manifest)
    assert effective_manifest != manifest.resolve()
    assert len(sampled_rows) == 5
    assert summary["sampled"] is True
    assert summary["source_rows"] == 20
    assert summary["rows"] == 5
    assert summary["manifest"] == str(effective_manifest)


def test_resolve_validation_manifest_keeps_full_dev_for_final_mode(tmp_path: Path):
    manifest = tmp_path / "dev.jsonl"
    _write_jsonl(
        manifest,
        [{"id": str(index), "text": "नमस्ते", "lang": "hi"} for index in range(20)],
    )

    effective_manifest, summary = resolve_validation_manifest(
        manifest,
        run_dir=tmp_path / "run",
        mode="final",
        diagnostic_val_size=5,
        diagnostic_val_seed=123,
    )

    assert effective_manifest == manifest.resolve()
    assert summary["sampled"] is False
    assert summary["rows"] == 20


def test_adapter_cli_allows_model_only_module_inspection(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nemo_adapter_peft.py",
            "--model",
            "/tmp/model.nemo",
            "--inspect-module-names",
        ],
    )

    args = parse_adapter_args()

    assert args.inspect_module_names is True
    assert args.train_manifest is None
    assert parse_module_name_patterns(args.inspect_module_name_patterns) == ["q", "k", "v", "proj", "linear", "ffn"]


def test_adapter_optimizer_replaces_inherited_noam_scheduler():
    optim_cfg = {
        "name": "adamw",
        "lr": 2.357,
        "betas": [0.9, 0.98],
        "weight_decay": 0.01,
        "sched": {
            "name": "NoamAnnealing",
            "d_model": 1024,
            "warmup_steps": 21703,
            "warmup_ratio": None,
            "min_lr": 1e-6,
        },
    }

    summary = configure_adapter_optimizer_cfg(
        optim_cfg,
        lr=1e-3,
        weight_decay=0.0,
        warmup_steps=100,
        max_steps=1000,
    )

    assert optim_cfg["name"] == "adamw"
    assert optim_cfg["lr"] == 1e-3
    assert optim_cfg["weight_decay"] == 0.0
    assert optim_cfg["betas"] == [0.9, 0.98]
    assert optim_cfg["sched"] == {
        "name": "WarmupAnnealing",
        "warmup_steps": 100,
        "warmup_ratio": None,
        "min_lr": 0.0,
        "last_epoch": -1,
        "max_steps": 1000,
    }
    assert "d_model" not in optim_cfg["sched"]
    assert summary["expected_peak_lr"] == 1e-3


class _ModuleTree:
    def named_modules(self):
        return [
            ("", object()),
            ("encoder.layers.0.self_attn.linear_q", object()),
            ("encoder.layers.0.self_attn.out_proj", object()),
            ("decoder.blank", object()),
        ]


def test_inspect_matching_module_names_filters_by_requested_fragments():
    matches = inspect_matching_module_names(_ModuleTree(), ["linear", "proj"])

    assert [item["name"] for item in matches] == [
        "encoder.layers.0.self_attn.linear_q",
        "encoder.layers.0.self_attn.out_proj",
    ]


def test_disable_metric_prediction_logging_turns_off_wer_metrics():
    class _Metric:
        log_prediction = True

    class _Model:
        wer = _Metric()
        ctc_wer = _Metric()

    model = _Model()

    disable_metric_prediction_logging(model)

    assert model.wer.log_prediction is False
    assert model.ctc_wer.log_prediction is False


def test_validate_manifest_text_fails_on_empty_normalized_text(tmp_path: Path):
    manifest = tmp_path / "bad.jsonl"
    _write_jsonl(manifest, [{"id": "bad", "text": "<unintelligible>", "lang": "hi"}])

    with pytest.raises(SystemExit):
        validate_manifest_text(manifest, split="train")


def test_validate_manifest_text_fails_on_unnormalized_training_text(tmp_path: Path):
    manifest = tmp_path / "bad.jsonl"
    _write_jsonl(manifest, [{"id": "bad", "text": "<noise> नमस्ते {test}", "lang": "hi"}])

    with pytest.raises(SystemExit, match="not normalized"):
        validate_manifest_text(manifest, split="train")


def test_tokenizer_fingerprint_detects_vocab_changes():
    before = tokenizer_fingerprint({"labels": ["a"], "decoder": {"vocab_size": 1}})
    after = tokenizer_fingerprint({"labels": ["a", "b"], "decoder": {"vocab_size": 2}})

    with pytest.raises(SystemExit):
        assert_tokenizer_unchanged(before, after)


class _Param:
    def __init__(self, count: int, requires_grad: bool, grad=None) -> None:
        self.count = count
        self.requires_grad = requires_grad
        self.grad = grad

    def numel(self) -> int:
        return self.count


class _ParamModel:
    def __init__(self, params: list[tuple[str, _Param]]) -> None:
        self.params = params

    def named_parameters(self):
        return list(self.params)


def test_trainable_parameter_guard_allows_only_small_adapter_set():
    model = _ParamModel(
        [
            ("encoder.layers.0.adapter.weight", _Param(10, True)),
            ("encoder.layers.0.weight", _Param(990, False)),
        ]
    )

    summary = summarize_trainable_parameters(model, adapter_name="vaani_adapter")
    validate_trainable_parameters(summary)
    assert summary["trainable_params"] == 10


def test_trainable_parameter_guard_rejects_non_adapter_trainables():
    model = _ParamModel(
        [
            ("decoder.weight", _Param(1, True)),
            ("encoder.layers.0.adapter.weight", _Param(10, True)),
            ("encoder.layers.0.weight", _Param(1000, False)),
        ]
    )

    summary = summarize_trainable_parameters(model, adapter_name="vaani_adapter")
    with pytest.raises(SystemExit):
        validate_trainable_parameters(summary)


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Grad:
    def __init__(self, norm: float) -> None:
        self._norm = norm

    def detach(self):
        return self

    def float(self):
        return self

    def norm(self):
        return _Scalar(self._norm)


def test_trainable_gradient_norms_reports_only_trainable_params_with_grad():
    model = _ParamModel(
        [
            ("encoder.layers.0.adapter.weight", _Param(10, True, _Grad(1.5))),
            ("encoder.layers.0.adapter.bias", _Param(2, True, None)),
            ("encoder.layers.0.weight", _Param(990, False, _Grad(9.0))),
        ]
    )

    assert trainable_gradient_norms(model) == [
        {"name": "encoder.layers.0.adapter.weight", "grad_norm": 1.5}
    ]


def test_tensorboard_tag_component_keeps_scalar_paths_readable():
    assert tensorboard_tag_component("encoder.layers.0/adapter weight") == "encoder.layers.0_adapter_weight"
    assert tensorboard_tag_component(":/") == "unnamed"


class _FakeExperiment:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.layouts: list[dict] = []
        self.flushed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.texts.append((tag, text, step))

    def add_custom_scalars(self, layout: dict) -> None:
        self.layouts.append(layout)

    def flush(self) -> None:
        self.flushed = True


class _FakeLogger:
    def __init__(self) -> None:
        self.experiment = _FakeExperiment()


def test_log_adapter_gradient_norms_to_tensorboard_writes_aggregate_and_parameter_scalars():
    logger = _FakeLogger()

    log_adapter_gradient_norms_to_tensorboard(
        logger,
        [
            {"name": "encoder.layers.0.adapter.weight", "grad_norm": 1.0},
            {"name": "encoder.layers.1.adapter.weight", "grad_norm": 3.0},
        ],
        step=5,
    )

    assert ("adapter_grad_norm/mean", 2.0, 5) in logger.experiment.scalars
    assert ("adapter_grad_norm/max", 3.0, 5) in logger.experiment.scalars
    assert (
        "adapter_grad_norm/parameters/encoder.layers.1.adapter.weight",
        3.0,
        5,
    ) in logger.experiment.scalars


def test_log_adapter_tensorboard_metadata_writes_adapter_scope_scalars():
    logger = _FakeLogger()
    run_config = {
        "adapter": {"adapter_dim": 32},
        "parameters": {"trainable_params": 100, "trainable_fraction": 0.001},
        "batch_size": 1,
        "accumulate_grad_batches": 4,
    }

    log_adapter_tensorboard_metadata(logger, run_config)

    assert logger.experiment.layouts
    assert ("adapter/trainable_params", 100.0, 0) in logger.experiment.scalars
    assert ("adapter/trainable_fraction_percent", 0.1, 0) in logger.experiment.scalars
    assert ("adapter/effective_batch_size", 4.0, 0) in logger.experiment.scalars
    assert logger.experiment.texts[0][0] == "adapter_peft/run_config"
    assert logger.experiment.flushed is True
