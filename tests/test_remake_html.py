from __future__ import annotations

import json
from pathlib import Path

from remake_html import load_groups


def test_load_groups_mirrors_parent_audio_into_output_dir(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    target_html_path = tmp_path / "dashboard" / "segmented_audio_review.html"
    segment_audio_path = target_html_path.parent / "audio" / "seg0001.wav"
    source_audio_path = tmp_path / "source" / "call.wav"

    segment_audio_path.parent.mkdir(parents=True, exist_ok=True)
    source_audio_path.parent.mkdir(parents=True, exist_ok=True)
    segment_audio_path.write_bytes(b"segment")
    source_audio_path.write_bytes(b"parent")

    manifest_row = {
        "audio_filepath": str(segment_audio_path),
        "duration": 1.0,
        "text": "hello",
        "lang": "en",
        "segment_id": "seg-1",
        "row_id": "row-1",
        "call_id": "call-1",
        "segment_index": 1,
        "segment_start_sec": 0.0,
        "segment_end_sec": 1.0,
        "source_audio_filepath": str(source_audio_path),
        "transcript_roles": ["agent"],
    }
    manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

    groups = load_groups(manifest_path, target_html_path)

    assert len(groups) == 1
    group = groups[0]
    assert group["parent_audio_href"].startswith("_parent_audio/")
    mirrored_parent_path = target_html_path.parent / group["parent_audio_href"]
    assert mirrored_parent_path.exists()
    assert mirrored_parent_path.read_bytes() == b"parent"

