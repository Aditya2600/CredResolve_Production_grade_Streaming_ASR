#!/usr/bin/env python3
import argparse
import asyncio
import base64
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import soundfile as sf
import websockets
from datasets import load_dataset

from compute_wer import edit_distance, normalize


LANGUAGE_MAP = {
    "assamese": "as",
    "bengali": "bn",
    "bodo": "brx",
    "dogri": "doi",
    "english": "en",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "kashmiri": "ks",
    "konkani": "gom",
    "maithili": "mai",
    "malayalam": "ml",
    "manipuri": "mni",
    "marathi": "mr",
    "nepali": "ne",
    "odia": "or",
    "punjabi": "pa",
    "sanskrit": "sa",
    "santali": "sat",
    "sindhi": "sd",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}

TEXT_CANDIDATES = (
    "normalized_text",
    "transcript",
    "transcription",
    "sentence",
    "text",
    "label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate gateway ASR WER on ai4bharat/IndicVoices via the Sarvam-like websocket path."
    )
    parser.add_argument("--ws", default="ws://localhost/ws/stt", help="Gateway websocket endpoint")
    parser.add_argument("--api-key", default="dev", help="Api-Subscription-Key for the websocket handshake")
    parser.add_argument("--dataset-config", default="hindi", help="IndicVoices config, e.g. hindi")
    parser.add_argument("--split", default="valid", help="Dataset split, e.g. valid")
    parser.add_argument("--language", help="ASR language code sent as language-code; defaults from dataset config")
    parser.add_argument("--model", default="credresolve:v1", help="Public model query param")
    parser.add_argument("--mode", default="transcribe", help="Public mode query param")
    parser.add_argument("--limit", type=int, default=10, help="Number of dataset samples to score")
    parser.add_argument("--hf-token", help="Hugging Face token for gated IndicVoices access")
    parser.add_argument("--cache-dir", help="Optional datasets cache dir")
    parser.add_argument("--text-field", help="Override transcript column name")
    parser.add_argument("--audio-field", help="Override audio column name")
    parser.add_argument("--out-tsv", type=Path, help="Optional output TSV with reference<TAB>hypothesis")
    return parser.parse_args()


def resolve_hf_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "IndicVoices is gated. Provide --hf-token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
        )
    return token


def resolve_language_code(dataset_config: str, cli_language: str | None) -> str:
    if cli_language:
        return cli_language
    code = LANGUAGE_MAP.get(dataset_config.strip().lower())
    if not code:
        raise SystemExit(
            f"Don't know how to map dataset config '{dataset_config}' to the gateway language code. "
            "Pass --language explicitly."
        )
    return code


def detect_text_field(sample: dict[str, Any]) -> str:
    for candidate in TEXT_CANDIDATES:
        value = sample.get(candidate)
        if isinstance(value, str) and value.strip():
            return candidate

    for key, value in sample.items():
        if isinstance(value, str) and value.strip():
            lowered = key.lower()
            if any(token in lowered for token in ("text", "trans", "sentence")):
                return key

    for key, value in sample.items():
        if isinstance(value, str) and value.strip():
            return key

    raise SystemExit(f"Unable to detect transcript field from sample keys: {sorted(sample.keys())}")


def detect_audio_field(sample: dict[str, Any]) -> str:
    for key, value in sample.items():
        if isinstance(value, dict):
            if "array" in value or "bytes" in value or "path" in value:
                return key
        if isinstance(value, (str, Path)) and str(value).lower().endswith((".wav", ".flac", ".mp3")):
            return key

    raise SystemExit(f"Unable to detect audio field from sample keys: {sorted(sample.keys())}")


def audio_to_float32(audio_value: Any) -> tuple[np.ndarray, int]:
    if isinstance(audio_value, dict):
        if "array" in audio_value and audio_value["array"] is not None:
            arr = np.asarray(audio_value["array"], dtype=np.float32)
            sr = int(audio_value.get("sampling_rate") or 16000)
            return arr, sr
        if audio_value.get("bytes") is not None:
            arr, sr = sf.read(io.BytesIO(audio_value["bytes"]), dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)
        if audio_value.get("path"):
            arr, sr = sf.read(audio_value["path"], dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)

    if isinstance(audio_value, (str, Path)):
        arr, sr = sf.read(str(audio_value), dtype="float32")
        return np.asarray(arr, dtype=np.float32), int(sr)

    raise SystemExit(f"Unsupported audio payload type: {type(audio_value)!r}")


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1, dtype=np.float32)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(audio.shape[0], dtype=np.float32)
    dst_length = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, num=dst_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def write_temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="indicvoices_", suffix=".wav", delete=False)
    handle.close()
    path = Path(handle.name)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def update_ws_query(base_ws_url: str, params: dict[str, str]) -> str:
    parsed = urlparse(base_ws_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


async def transcribe_wav(
    *,
    ws_uri: str,
    wav_path: Path,
    api_key: str,
    call_id: str,
    language: str,
    model: str,
    mode: str,
) -> str:
    final_texts: list[str] = []
    ws_url = update_ws_query(
        ws_uri,
        {
            "language-code": language,
            "model": model,
            "mode": mode,
            "sample_rate": "16000",
            "high_vad_sensitivity": "false",
            "vad_signals": "false",
            "flush_signal": "true",
            "input_audio_codec": "wav",
        },
    )
    auth_headers = {"Api-Subscription-Key": api_key}

    async with websockets.connect(ws_url, max_size=20_000_000, additional_headers=auth_headers) as ws:
        wav_bytes = wav_path.read_bytes()
        await ws.send(
            json.dumps(
                {
                    "audio": {
                        "data": base64.b64encode(wav_bytes).decode("ascii"),
                        "sample_rate": "16000",
                        "encoding": "wav",
                    }
                }
            )
        )
        await ws.send(json.dumps({"type": "flush"}))

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            event = json.loads(raw)
            if event.get("type") == "data":
                transcript = str(event.get("data", {}).get("transcript", "")).strip()
                if transcript:
                    final_texts.append(transcript)
            elif event.get("type") == "error":
                raise RuntimeError(f"gateway error for {call_id}: {event}")

    return " ".join(part for part in final_texts if part).strip()


async def run() -> None:
    args = parse_args()
    token = resolve_hf_token(args.hf_token)
    language = resolve_language_code(args.dataset_config, args.language)

    dataset = load_dataset(
        "ai4bharat/IndicVoices",
        args.dataset_config,
        split=args.split,
        token=token,
        streaming=True,
        cache_dir=args.cache_dir,
    )
    dataset = dataset.decode(False)

    iterator = iter(dataset)
    try:
        first_sample = next(iterator)
    except StopIteration as exc:
        raise SystemExit("Dataset split is empty.") from exc

    text_field = args.text_field or detect_text_field(first_sample)
    audio_field = args.audio_field or detect_audio_field(first_sample)

    samples = [(0, first_sample)]
    for index, sample in enumerate(iterator, start=1):
        if index >= args.limit:
            break
        samples.append((index, sample))

    total_s = total_d = total_i = total_words = 0
    processed = 0
    failures = 0
    tsv_lines: list[str] = []

    print(
        f"dataset=ai4bharat/IndicVoices config={args.dataset_config} split={args.split} "
        f"limit={args.limit} language={language} text_field={text_field} audio_field={audio_field}"
    )

    for index, sample in samples:
        reference = str(sample[text_field]).strip()
        if not reference:
            print(f"skip index={index} reason=empty_reference")
            continue

        temp_wav: Path | None = None
        try:
            audio, sr = audio_to_float32(sample[audio_field])
            audio = to_mono(audio)
            audio = resample_linear(audio, sr, 16000)
            temp_wav = write_temp_wav(audio, 16000)

            hypothesis = await transcribe_wav(
                ws_uri=args.ws,
                wav_path=temp_wav,
                api_key=args.api_key,
                call_id=f"indicvoices-{args.dataset_config}-{index}",
                language=language,
                model=args.model,
                mode=args.mode,
            )

            ref_words = normalize(reference)
            hyp_words = normalize(hypothesis)
            s, d, ins = edit_distance(ref_words, hyp_words)
            total_s += s
            total_d += d
            total_i += ins
            total_words += len(ref_words)
            processed += 1
            tsv_lines.append(f"{reference}	{hypothesis}")
            print(
                f"sample={index} ref_words={len(ref_words)} hyp_words={len(hyp_words)} "
                f"s={s} d={d} i={ins} hyp={json.dumps(hypothesis, ensure_ascii=False)}"
            )
        except Exception as exc:
            failures += 1
            print(f"sample={index} status=error detail={exc}")
        finally:
            if temp_wav is not None:
                temp_wav.unlink(missing_ok=True)

    wer = ((total_s + total_d + total_i) / total_words) if total_words else 0.0

    if args.out_tsv:
        args.out_tsv.write_text("\n".join(tsv_lines) + ("\n" if tsv_lines else ""), encoding="utf-8")


    print(f"processed={processed}")
    print(f"failures={failures}")
    print(f"reference_words={total_words}")
    print(f"substitutions={total_s}")
    print(f"deletions={total_d}")
    print(f"insertions={total_i}")
    print(f"wer={wer:.4f}")
    print(f"wer_percent={wer * 100:.2f}")


if __name__ == "__main__":
    asyncio.run(run())
