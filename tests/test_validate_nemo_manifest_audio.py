from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_nemo_manifest_audio import validate_manifest_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_validate_manifest_file_writes_filtered_rejects_and_summary(tmp_path: Path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    good = tmp_path / "good.wav"
    short = tmp_path / "short.wav"
    stereo = tmp_path / "stereo.wav"
    wrong_sr = tmp_path / "wrong_sr.wav"
    sf.write(good, np.zeros(16000, dtype=np.float32), 16000)
    sf.write(short, np.zeros(3200, dtype=np.float32), 16000)
    sf.write(stereo, np.zeros((16000, 2), dtype=np.float32), 16000)
    sf.write(wrong_sr, np.zeros(8000, dtype=np.float32), 8000)

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"id": "good", "audio_filepath": str(good), "duration": 1.0, "text": "hello"},
            {"id": "short", "audio_filepath": str(short), "duration": 0.2, "text": "hello"},
            {"id": "empty", "audio_filepath": str(good), "duration": 1.0, "text": " "},
            {"id": "stereo", "audio_filepath": str(stereo), "duration": 1.0, "text": "hello"},
            {"id": "wrong_sr", "audio_filepath": str(wrong_sr), "duration": 1.0, "text": "hello"},
            {"id": "fast", "audio_filepath": str(good), "duration": 1.0, "text": "x" * 40},
            {"id": "mismatch", "audio_filepath": str(good), "duration": 3.0, "text": "hello"},
            {"id": "missing", "audio_filepath": str(tmp_path / "missing.wav"), "duration": 1.0, "text": "hello"},
        ],
    )

    output = tmp_path / "filtered.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    summary_json = tmp_path / "summary.json"
    summary = validate_manifest_file(
        manifest,
        output,
        rejects,
        summary_json,
        min_duration_sec=1.0,
        max_duration_sec=2.0,
        max_chars_per_sec=30.0,
        max_words_per_sec=4.5,
        expected_sample_rate=16000,
        require_mono=True,
        max_duration_delta_sec=0.25,
        progress_every=0,
    )

    kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rejected = [json.loads(line) for line in rejects.read_text(encoding="utf-8").splitlines()]
    counts = summary["reject_counts"]

    assert [row["id"] for row in kept] == ["good"]
    assert kept[0]["validation_source_lineno"] == 1
    assert len(rejected) == 7
    assert counts["too_short_audio"] == 1
    assert counts["empty_transcript"] == 1
    assert counts["non_mono_audio"] == 1
    assert counts["wrong_sample_rate"] == 1
    assert counts["high_chars_per_sec"] == 1
    assert counts["duration_mismatch"] == 1
    assert counts["missing_audio_file"] == 1
    assert json.loads(summary_json.read_text(encoding="utf-8"))["kept"] == 1
