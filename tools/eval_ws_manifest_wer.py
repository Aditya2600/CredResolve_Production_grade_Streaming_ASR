#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import soundfile as sf
import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.asr_text_normalizer import normalize_asr_text
from tools.compute_wer import edit_distance, normalize


TEXT_KEY_CANDIDATES = ("text", "reference", "normalized_text", "transcript", "sentence")
AUDIO_KEY_CANDIDATES = ("audio_filepath", "audio_path", "audio", "path", "file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a manifest through the gateway WebSocket path and score both "
            "raw ASR output and client-visible ITN display text."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ws", default="ws://localhost/ws/stt")
    parser.add_argument("--api-key", default="dev")
    parser.add_argument("--language-code", default="hi")
    parser.add_argument("--model", default="credresolve:v1")
    parser.add_argument("--mode", default="transcribe")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--locale-policy", default="")
    parser.add_argument("--text-field")
    parser.add_argument("--audio-field")
    parser.add_argument("--reference-normalization", choices=("raw", "vaani"), default="vaani")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--idle-timeout-s", type=float, default=2.0)
    parser.add_argument("--open-timeout-s", type=float, default=30.0)
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument("--out-summary-json", type=Path)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: expected JSON object rows")
        rows.append(row)
    return rows


def detect_key(rows: list[dict[str, Any]], candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        return explicit
    if not rows:
        raise SystemExit("Manifest is empty")
    for key in candidates:
        if key in rows[0]:
            return key
    raise SystemExit(f"Could not auto-detect a key from: {', '.join(candidates)}")


def resolve_path(value: Any, manifest_path: Path) -> Path:
    candidate = Path(str(value or "").strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    return candidate


def update_ws_query(base_ws_url: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_ws_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: value for key, value in params.items() if value != ""})
    return urlunparse(parsed._replace(query=urlencode(query)))


def to_pcm16_mono(path: Path, target_sample_rate: int) -> bytes:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1, dtype=np.float32)
    if sample_rate != target_sample_rate:
        if arr.size == 0:
            arr = np.zeros(0, dtype=np.float32)
        else:
            src_positions = np.arange(arr.shape[0], dtype=np.float32)
            dst_length = max(1, int(round(arr.shape[0] * (target_sample_rate / sample_rate))))
            dst_positions = np.linspace(0.0, arr.shape[0] - 1, num=dst_length, dtype=np.float32)
            arr = np.interp(dst_positions, src_positions, arr).astype(np.float32)
    return np.rint(np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def clean_reference(text: str, mode: str) -> str:
    if mode == "raw":
        return text
    return normalize_asr_text(text)


def score(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)
    substitutions, deletions, insertions = edit_distance(ref_words, hyp_words)
    wer = ((substitutions + deletions + insertions) / len(ref_words)) if ref_words else 0.0
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "ref_words": len(ref_words),
        "wer": wer,
    }


async def stream_one(
    *,
    ws_url: str,
    api_key: str,
    pcm16le: bytes,
    frame_bytes: int,
    idle_timeout_s: float,
    open_timeout_s: float,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    data_messages: list[dict[str, Any]] = []
    async with websockets.connect(
        ws_url,
        max_size=20_000_000,
        open_timeout=open_timeout_s,
        additional_headers={"Api-Subscription-Key": api_key},
    ) as ws:
        for offset in range(0, len(pcm16le), frame_bytes):
            chunk = pcm16le[offset : offset + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk += b"\x00" * (frame_bytes - len(chunk))
            await ws.send(chunk)

        await ws.send(json.dumps({"type": "flush"}))

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=idle_timeout_s)
            except asyncio.TimeoutError:
                break
            payload = json.loads(raw)
            if payload.get("type") == "error":
                code = payload.get("code", "UNKNOWN")
                message = payload.get("message", "")
                raise RuntimeError(f"{code}: {message}".strip())
            if payload.get("type") == "data":
                data_messages.append(payload["data"])

    return data_messages, time.perf_counter() - started


async def main_async(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    text_key = detect_key(rows, TEXT_KEY_CANDIDATES, args.text_field)
    audio_key = detect_key(rows, AUDIO_KEY_CANDIDATES, args.audio_field)
    ws_url = update_ws_query(
        args.ws,
        {
            "language-code": args.language_code,
            "model": args.model,
            "mode": args.mode,
            "sample_rate": str(args.sample_rate),
            "input_audio_codec": "pcm_s16le",
            "binary_audio": "1",
            "flush_signal": "true",
            "locale-policy": args.locale_policy,
        },
    )
    frame_bytes = int(args.sample_rate * (args.frame_ms / 1000.0) * 2)

    detailed_rows: list[dict[str, Any]] = []
    totals = {
        "raw": {"s": 0, "d": 0, "i": 0, "words": 0},
        "display": {"s": 0, "d": 0, "i": 0, "words": 0},
    }

    for index, row in enumerate(rows, start=1):
        audio_path = resolve_path(row[audio_key], manifest_path)
        reference = clean_reference(str(row.get(text_key, "") or ""), args.reference_normalization)
        pcm16le = to_pcm16_mono(audio_path, args.sample_rate)
        try:
            messages, wall_s = await stream_one(
                ws_url=ws_url,
                api_key=args.api_key,
                pcm16le=pcm16le,
                frame_bytes=frame_bytes,
                idle_timeout_s=args.idle_timeout_s,
                open_timeout_s=args.open_timeout_s,
            )
            raw_text = " ".join(str(msg.get("raw_text") or msg.get("transcript") or "") for msg in messages).strip()
            display_text = " ".join(str(msg.get("display_text") or "") for msg in messages).strip()
            raw_score = score(reference, raw_text)
            display_score = score(reference, display_text)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - evaluation should continue row-by-row
            messages = []
            wall_s = 0.0
            raw_text = ""
            display_text = ""
            raw_score = score(reference, raw_text)
            display_score = score(reference, display_text)
            status = f"error:{exc}"

        for key, row_score in (("raw", raw_score), ("display", display_score)):
            totals[key]["s"] += row_score["substitutions"]
            totals[key]["d"] += row_score["deletions"]
            totals[key]["i"] += row_score["insertions"]
            totals[key]["words"] += row_score["ref_words"]

        result = {
            "index": index,
            "audio": str(audio_path),
            "status": status,
            "reference": reference,
            "raw_text": raw_text,
            "display_text": display_text,
            "data_messages": len(messages),
            "wall_s": wall_s,
            "raw_score": raw_score,
            "display_score": display_score,
            "normalization_spans": [
                span
                for message in messages
                for span in (message.get("normalization_spans") or [])
            ],
        }
        detailed_rows.append(result)
        print(
            f"[{index}/{len(rows)}] {audio_path.name} status={status} "
            f"raw_wer={raw_score['wer']:.4f} display_wer={display_score['wer']:.4f}"
        )

    summary = {
        "samples": len(detailed_rows),
        "ok": sum(1 for row in detailed_rows if row["status"] == "ok"),
        "errors": sum(1 for row in detailed_rows if row["status"] != "ok"),
        "raw_wer": (
            (totals["raw"]["s"] + totals["raw"]["d"] + totals["raw"]["i"]) / totals["raw"]["words"]
            if totals["raw"]["words"]
            else 0.0
        ),
        "display_wer": (
            (totals["display"]["s"] + totals["display"]["d"] + totals["display"]["i"]) / totals["display"]["words"]
            if totals["display"]["words"]
            else 0.0
        ),
        "raw_totals": totals["raw"],
        "display_totals": totals["display"],
    }

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.out_jsonl.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in detailed_rows) + "\n",
            encoding="utf-8",
        )
    if args.out_summary_json:
        args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["errors"] == 0 else 1


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
