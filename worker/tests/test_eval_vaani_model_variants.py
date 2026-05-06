from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.validate_vaani_model_variants import build_eval_command


class FakeAPMProcessor:
    frame_bytes = 4

    def process_frame(self, frame: bytes) -> bytes:
        return frame[::-1]


def test_process_pcm_with_apm_keeps_partial_tail_unmodified():
    pytest.importorskip("omegaconf")
    from tools.eval_nemo_manifest_wer import process_pcm_with_apm

    processed = process_pcm_with_apm(b"abcdefghijkl", FakeAPMProcessor())

    assert processed == b"dcbahgfelkji"


def test_process_pcm_with_apm_preserves_non_frame_tail():
    pytest.importorskip("omegaconf")
    from tools.eval_nemo_manifest_wer import process_pcm_with_apm

    processed = process_pcm_with_apm(b"abcdefghij", FakeAPMProcessor())

    assert processed == b"dcbahgfeij"


def test_vaani_reference_normalization_drops_latin_glosses_and_keeps_indic_braces():
    pytest.importorskip("omegaconf")
    from tools.eval_nemo_manifest_wer import clean_reference

    raw = "स्-बालू {cement} [noise] <music> म्-यहाँँ {और}"

    assert clean_reference(raw, "vaani") == "बालू यहाँँ और"


def test_serving_reference_normalization_uses_shared_vaani_policy():
    from tools.eval_serving_model_manifest_wer import clean_reference

    raw = "પાસે એક સફેદ ટેબલ {table} છે જેના પર પર્પલ {purple} કલર {colour} ની એક પાણીની બોટલ {bottle} છે."

    assert clean_reference(raw, "vaani") == "પાસે એક સફેદ ટેબલ છે જેના પર પર્પલ કલર ની એક પાણીની બોટલ છે"


def test_raw_reference_normalization_only_normalizes_space():
    pytest.importorskip("omegaconf")
    from tools.eval_nemo_manifest_wer import clean_reference

    raw = "  {cement}\n[noise]  "

    assert clean_reference(raw, "raw") == "{cement} [noise]"


def test_build_eval_command_adds_variant_toggles(tmp_path: Path):
    args = argparse.Namespace(
        model=tmp_path / "model.nemo",
        batch_size=2,
        num_workers=0,
        device="cpu",
        decoder="rnnt",
        language="hi",
        reference_normalization="vaani",
        limit=5,
        apm_backend="demo.module:Backend",
        context_biasing_mode="active",
        context_biasing_phrases_dir=tmp_path / "phrases",
        top_errors=3,
    )

    command = build_eval_command(
        args,
        manifest=tmp_path / "manifest.jsonl",
        variant={"name": "all", "denoise": True, "apm": True, "context_biasing": True},
        variant_dir=tmp_path / "variant",
    )

    assert "--denoise" in command
    assert "--apm" in command
    assert command[command.index("--apm-backend") + 1] == "demo.module:Backend"
    assert command[command.index("--context-biasing-mode") + 1] == "active"
    assert command[command.index("--limit") + 1] == "5"


def test_serving_eval_context_biasing_device_follows_cpu_run():
    from tools.eval_serving_model_manifest_wer import resolve_context_biasing_device

    args = argparse.Namespace(device="cpu", context_biasing_device="")

    assert resolve_context_biasing_device(args) == "cpu"


def test_serving_eval_context_biasing_device_preserves_explicit_override():
    from tools.eval_serving_model_manifest_wer import resolve_context_biasing_device

    args = argparse.Namespace(device="cpu", context_biasing_device="cuda")

    assert resolve_context_biasing_device(args) == "cuda"
