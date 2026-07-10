#!/usr/bin/env python3
"""Pseudo-label a batch_transcript CSV with Sarvam sync STT and flag mismatches vs indic conformer."""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import mimetypes
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_URL = "https://api.sarvam.ai/speech-to-text"
MEDHA_URL_DEFAULT = "http://164.52.192.196:8002/v1/chat/completions"
MEDHA_MODEL_DEFAULT = "Medha"
MEDHA_KEY_DEFAULT = "medha-prod-2026-secure"


class RateLimiter:
    """Caps calls to N per second across threads."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next_slot = time.monotonic()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        sleep_for = slot - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def download(url: str, path: Path, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "sarvam-pseudo-label/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"download failed for {url}: {last_error}")


def transcribe(path: Path, *, api_key: str, model: str, language_code: str, timeout_sec: float) -> str:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    fields = {"model": model, "language_code": language_code}
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n".encode()
        + path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "api-subscription-key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            import json

            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"sarvam transcribe failed for {path.name}: HTTP {exc.code}: {exc.read().decode()}") from exc
    return str(data.get("transcript") or "").strip()


SYSTEM_PROMPT = (
    "You are a Hindi ASR text normalizer. You rewrite one transcript into the exact spoken-word "
    "form a speech recognizer would emit. You output only the normalized transcript — no quotes, "
    "labels, explanations, or extra lines."
)

USER_PROMPT_TEMPLATE = (
    "Rewrite the ASR transcript below into the exact words a Hindi speech recognizer emits — "
    "spoken-word Devanagari, nothing else.\n"
    "Rules:\n"
    "- Output pure Devanagari. Transliterate romanized/English words to Devanagari as they sound "
    "(do not translate them).\n"
    "- Spell out EVERY number, amount, currency, percent, phone number, and date as spoken Hindi "
    "words — never leave digits or symbols. Read multi-digit groups the way a person says them.\n"
    "    ₹4,45,000 -> चार लाख पैंतालीस हज़ार रुपये\n"
    "    ₹1,90,000 -> एक लाख नब्बे हज़ार रुपये\n"
    "    90% -> नब्बे प्रतिशत    |    2 -> दो    |    15/03 -> पंद्रह तीन\n"
    "- CRITICAL: change ONLY script and number spelling. Keep every word exactly as given — never "
    "add, drop, translate, reorder, correct grammar, or 'improve' anything. Same word count.\n"
    "- Single spaces between words. No punctuation, quotes, or labels.\n"
    "Output only the rewritten transcript on one line.\n\n"
    "TRANSCRIPT: {sarvam_text}"
)


def medha_normalize(
    sarvam_text: str,
    *,
    medha_url: str,
    medha_key: str,
    model: str,
    timeout_sec: float,
) -> str:
    """Rewrite a Sarvam ASR transcript into spoken-word Devanagari via Medha. Returns plain text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(sarvam_text=sarvam_text)},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False,
    }
    request = urllib.request.Request(
        medha_url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {medha_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = str(data["choices"][0]["message"]["content"]).strip()
    content = re.sub(r"^```[a-zA-Z]*\n?|```$", "", content).strip()
    return content


_COMPARE_STRIP = re.compile(r"[।॥,.\?!\"'`~:;\-–—_/\\|()\[\]{}<>*]+")
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")


def canon_devanagari(text: str) -> str:
    """Fold trivial Devanagari variance so only real word differences survive the comparison.

    Removes nukta (ज़->ज, क़->क), zero-width joiners, punctuation/danda, and collapses whitespace.
    These vary freely between ASR systems and human typists and should never count as a mismatch.
    """
    text = unicodedata.normalize("NFD", text)
    text = text.replace("़", "")  # combining nukta
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _COMPARE_STRIP.sub(" ", text)
    return normalize(text)


def texts_match(a: str, b: str, *, threshold: float) -> bool:
    """True when two spoken-word Devanagari transcripts are the same up to trivial variance.

    Canonicalization already folds punctuation, spacing, nukta and zero-width joiners, so what
    remains is word-level difference. We compare at WORD granularity (not characters): a single
    swapped, added, or dropped word measurably lowers the ratio, whereas a character ratio would
    hide a missing short word like "ही" behind a 0.95 score. Ratio >= `threshold` counts as a
    match; anything below flags for manual review.
    """
    ta, tb = canon_devanagari(a).split(), canon_devanagari(b).split()
    if ta == tb:
        return True
    if not ta or not tb:
        return False
    return difflib.SequenceMatcher(None, ta, tb).ratio() >= threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Add sarvam_transcript + mismatch columns to a batch_transcript CSV.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audio-column", default="s3_audio_link")
    parser.add_argument("--reference-column", default="transcript", help="indic conformer transcript column to diff against")
    parser.add_argument("--download-dir", type=Path, default=Path("artifacts/sarvam_pseudo_label_audio"))
    parser.add_argument("--model", default="saarika:v2.5")
    parser.add_argument("--language-code", default="hi-IN")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="SARVAM_API")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=16, help="Parallel workers (rate-capped separately).")
    parser.add_argument("--requests-per-sec", type=float, default=80.0, help="Shared cap across sarvam+medha calls.")
    parser.add_argument("--medha-url", default=os.environ.get("MEDHA_URL", MEDHA_URL_DEFAULT))
    parser.add_argument("--medha-key", default=os.environ.get("MEDHA_API_KEY", MEDHA_KEY_DEFAULT))
    parser.add_argument("--medha-model", default=MEDHA_MODEL_DEFAULT)
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.95,
        help="Word-similarity (0-1) at/above which normalized sarvam counts as matching reference. "
        "Lower = fewer mismatches flagged; raise toward 1.0 to flag more aggressively (1.0 = every "
        "word must match after normalization).",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Input CSV already has sarvam_transcript filled; only normalize+compare via Medha.",
    )
    args = parser.parse_args()

    api_key = ""
    if not args.skip_transcribe:
        load_dotenv(args.env_file)
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is not set; pass --env-file or export it.")

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    for column in ("sarvam_transcript", "sarvam_transcript_devanagari", "mismatch_vs_indic_conformer"):
        if column not in fieldnames:
            fieldnames.append(column)

    limiter = RateLimiter(args.requests_per_sec)

    def process_row(row: dict[str, str]) -> None:
        reference = row.get(args.reference_column) or ""

        if args.skip_transcribe:
            sarvam_text = (row.get("sarvam_transcript") or "").strip()
            if not sarvam_text or sarvam_text.startswith("ERROR:"):
                row["sarvam_transcript_devanagari"] = ""
                row["mismatch_vs_indic_conformer"] = ""
                return
            try:
                limiter.wait()
                normalized = medha_normalize(
                    sarvam_text,
                    medha_url=args.medha_url,
                    medha_key=args.medha_key,
                    model=args.medha_model,
                    timeout_sec=args.timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001
                row["sarvam_transcript_devanagari"] = f"ERROR: {exc}"
                row["mismatch_vs_indic_conformer"] = ""
                return
            row["sarvam_transcript_devanagari"] = normalized
            is_match = texts_match(normalized, reference, threshold=args.match_threshold)
            row["mismatch_vs_indic_conformer"] = "" if is_match else "YES"
            print(f"{row.get('filename')}: mismatch={row['mismatch_vs_indic_conformer'] or 'no'}", flush=True)
            return

        url = (row.get(args.audio_column) or "").strip()
        if not url:
            row["sarvam_transcript"] = ""
            row["sarvam_transcript_devanagari"] = ""
            row["mismatch_vs_indic_conformer"] = ""
            return
        audio_path = args.download_dir / (row.get("filename") or f"{uuid.uuid4().hex}.wav")
        try:
            download(url, audio_path)
            limiter.wait()
            sarvam_text = transcribe(
                audio_path,
                api_key=api_key,
                model=args.model,
                language_code=args.language_code,
                timeout_sec=args.timeout_sec,
            )
            limiter.wait()
            normalized = medha_normalize(
                sarvam_text,
                medha_url=args.medha_url,
                medha_key=args.medha_key,
                model=args.medha_model,
                timeout_sec=args.timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            row["sarvam_transcript"] = f"ERROR: {exc}"
            row["sarvam_transcript_devanagari"] = ""
            row["mismatch_vs_indic_conformer"] = ""
            return
        row["sarvam_transcript"] = sarvam_text
        row["sarvam_transcript_devanagari"] = normalized
        is_match = texts_match(normalized, reference, threshold=args.match_threshold)
        row["mismatch_vs_indic_conformer"] = "" if is_match else "YES"
        print(f"{row.get('filename')}: mismatch={row['mismatch_vs_indic_conformer'] or 'no'}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        list(executor.map(process_row, rows))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
