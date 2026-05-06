from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.export_vaani_hindi_to_nemo import (
    VaaniExportFilters,
    export_samples_to_nemo,
    process_sample,
)


def _audio(seconds: float, *, path: str = "clip.wav") -> dict:
    return {
        "array": np.ones(int(16000 * seconds), dtype=np.float32) * 0.01,
        "sampling_rate": 16000,
        "path": path,
    }


def _sample(
    *,
    transcript: str = "नमस्ते दुनिया",
    seconds: float = 1.0,
    language: str = "Hindi",
    path: str = "clip.wav",
) -> dict:
    return {
        "audio": _audio(seconds, path=path),
        "transcript": transcript,
        "language": language,
        "state": "Delhi",
    }


def _process(sample: dict, seen: set[tuple[str, str]] | None = None) -> dict:
    return process_sample(
        sample,
        dataset_index=7,
        text_field="transcript",
        audio_field="audio",
        filters=VaaniExportFilters(),
        seen_dedupe_keys=seen if seen is not None else set(),
        dataset_config="audio/Hindi",
        split="train",
        language_code="hi",
    )


def _process_language(
    sample: dict,
    *,
    dataset_config: str,
    language_code: str,
    seen: set[tuple[str, str]] | None = None,
) -> dict:
    return process_sample(
        sample,
        dataset_index=7,
        text_field="transcript",
        audio_field="audio",
        filters=VaaniExportFilters(),
        seen_dedupe_keys=seen if seen is not None else set(),
        dataset_config=dataset_config,
        split="train",
        language_code=language_code,
    )


def test_process_sample_accepts_clean_hindi_audio():
    decision = _process(_sample())

    assert decision["status"] == "accepted"
    assert decision["row"]["lang"] == "hi"
    assert decision["row"]["dataset_config"] == "audio/Hindi"
    assert decision["row"]["duration"] == 1.0
    assert decision["row"]["text"] == "नमस्ते दुनिया"


def test_process_sample_rejects_empty_transcript():
    decision = _process(_sample(transcript="   "))

    assert decision["status"] == "rejected"
    assert decision["reject"]["reason"] == "empty_text"


def test_process_sample_rejects_non_hindi_language():
    decision = _process(_sample(language="English"))

    assert decision["status"] == "rejected"
    assert decision["reject"]["reason"] == "non_hindi_language"


def test_process_sample_accepts_configured_non_hindi_language():
    decision = _process_language(
        _sample(transcript="ગુજરાતી વાક્ય", language="Gujarati"),
        dataset_config="Gujarati",
        language_code="gu",
    )

    assert decision["status"] == "accepted"
    assert decision["row"]["lang"] == "gu"
    assert decision["row"]["dataset_config"] == "Gujarati"


def test_process_sample_rejects_wrong_configured_language():
    decision = _process_language(
        _sample(transcript="ગુજરાતી વાક્ય", language="Hindi"),
        dataset_config="Gujarati",
        language_code="gu",
    )

    assert decision["status"] == "rejected"
    assert decision["reject"]["reason"] == "non_target_language"


def test_process_sample_rejects_duration_outliers():
    decision = _process(_sample(seconds=9.0))

    assert decision["status"] == "rejected"
    assert decision["reject"]["reason"] == "too_long"


def test_process_sample_rejects_duplicate_audio_text():
    seen: set[tuple[str, str]] = set()
    first = _process(_sample(path="same.wav"), seen)
    second = _process(_sample(path="same.wav"), seen)

    assert first["status"] == "accepted"
    assert second["status"] == "rejected"
    assert second["reject"]["reason"] == "duplicate_audio_text"


def test_export_samples_stops_after_target_duration(tmp_path: Path):
    samples = [
        (0, _sample(transcript="पहला वाक्य", seconds=1.0, path="a.wav")),
        (1, _sample(transcript="दूसरा वाक्य", seconds=1.0, path="b.wav")),
        (2, _sample(transcript="तीसरा वाक्य", seconds=1.0, path="c.wav")),
    ]

    summary = export_samples_to_nemo(
        samples,
        out_dir=tmp_path,
        dataset_config="audio/Hindi",
        split="train",
        target_hours=1.5 / 3600.0,
        seed=42,
        shuffle_buffer_size=1,
        filters=VaaniExportFilters(),
        text_field="transcript",
        audio_field="audio",
        language_code="hi",
    )

    manifest_rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["target_reached"] is True
    assert summary["rows_scanned"] == 2
    assert summary["rows_kept"] == 2
    assert summary["kept_seconds"] == 2.0
    assert len(manifest_rows) == 2
    assert all(Path(row["audio_filepath"]).exists() for row in manifest_rows)
