#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.asr_text_normalizer import normalize_asr_text
from tools.compute_wer import edit_distance, normalize


AUDIO_SOURCE_SUFFIX_RE = re.compile(r"\s+\([^)]*\)\s+as\s+PCM$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join a websocket transcript log with its source manifest for frontend review."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Source manifest JSONL with file + transcript fields.")
    parser.add_argument("--transcripts", type=Path, required=True, help="Websocket transcript JSONL from tools/ws_burst_barrier.py.")
    parser.add_argument("--out-jsonl", type=Path, required=True, help="Output JSONL compatible with the frontend review panel.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive CLI error path
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def extract_audio_path(audio_source: str) -> Path:
    cleaned = AUDIO_SOURCE_SUFFIX_RE.sub("", audio_source.strip())
    return Path(cleaned).expanduser()


def build_review_rows(
    manifest_rows: list[dict[str, Any]],
    transcript_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    manifest_by_file = {str(row["file"]): row for row in manifest_rows if row.get("file")}
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in transcript_rows:
        audio_path = extract_audio_path(str(row.get("audio_source", "")))
        key = (
            int(row.get("client_id", -1)),
            str(row.get("request_id", "")),
            audio_path.name,
        )
        grouped[key].append(row)

    review_rows: list[dict[str, Any]] = []
    for output_index, ((client_id, request_id, filename), chunks) in enumerate(
        sorted(grouped.items(), key=lambda item: item[0][0]),
        start=1,
    ):
        manifest_row = manifest_by_file.get(filename)
        if manifest_row is None:
            raise SystemExit(f"missing manifest row for transcript audio file: {filename}")

        ordered_chunks = sorted(chunks, key=lambda row: int(row.get("index", 0)))
        raw_reference = str(manifest_row.get("transcript", "") or "").strip()
        reference = normalize_asr_text(raw_reference, str(manifest_row.get("language", "") or ""))
        hypothesis = " ".join(str(row.get("transcript", "") or "").strip() for row in ordered_chunks).strip()
        audio_path = extract_audio_path(str(ordered_chunks[0].get("audio_source", "")))
        if not audio_path.is_absolute():
            audio_path = (repo_root / audio_path).resolve()

        ref_words = normalize(reference)
        hyp_words = normalize(hypothesis)
        substitutions, deletions, insertions = edit_distance(ref_words, hyp_words)

        review_rows.append(
            {
                "index": output_index,
                "client_id": client_id,
                "request_id": request_id,
                "audio_filepath": str(audio_path),
                "reference": reference,
                "raw_reference": raw_reference,
                "hypothesis": hypothesis,
                "reference_words": len(ref_words),
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "sample_wer": ((substitutions + deletions + insertions) / len(ref_words)) if ref_words else 0.0,
                "language_id": str(ordered_chunks[0].get("language_code", "") or ""),
                "language_source": str(ordered_chunks[0].get("language_source", "") or ""),
                "source_id": filename,
                "dataset_index": manifest_row.get("slot"),
                "chunk_count": len(ordered_chunks),
                "phases": [str(row.get("phase", "") or "") for row in ordered_chunks],
                "audio_duration": round(sum(float(row.get("audio_duration", 0) or 0) for row in ordered_chunks), 4),
            }
        )

    return review_rows


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    transcript_rows = read_jsonl(args.transcripts)
    review_rows = build_review_rows(manifest_rows, transcript_rows, repo_root=REPO_ROOT)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    print(f"wrote {len(review_rows)} review rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
