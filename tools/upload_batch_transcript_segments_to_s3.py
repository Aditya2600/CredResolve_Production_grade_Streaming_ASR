#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import hmac
import os
import pathlib
import re
import shutil
import time
import urllib.parse
import urllib.request


SEGMENT_RE = re.compile(r"_seg_\d+\.wav$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload segment WAVs to S3 and keep only segment CSV rows.")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("model_training/batch_transcript (3).csv"))
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("vad_chunks"))
    parser.add_argument("--env", type=pathlib.Path, default=pathlib.Path(".env"))
    parser.add_argument("--state", type=pathlib.Path, default=pathlib.Path("artifacts/s3_upload_batch_transcript_3_done.txt"))
    parser.add_argument("--link-column", default="s3_audio_link")
    return parser.parse_args()


def load_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if text and not text.startswith("#") and "=" in text:
            key, value = text.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def signing_key(secret: str, date_stamp: str, region: str) -> bytes:
    date_key = sign(("AWS4" + secret).encode(), date_stamp)
    region_key = sign(date_key, region)
    service_key = sign(region_key, "s3")
    return sign(service_key, "aws4_request")


def upload_once(
    path: pathlib.Path,
    *,
    bucket: str,
    key: str,
    region: str,
    access_key: str,
    secret_key: str,
) -> None:
    body = path.read_bytes()
    payload_hash = hashlib.sha256(body).hexdigest()
    now = dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    host = f"{bucket}.s3.{region}.amazonaws.com"
    uri = "/" + urllib.parse.quote(key, safe="/")
    headers = {
        "content-type": "audio/wav",
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
    canonical_request = "\n".join(["PUT", uri, "", canonical_headers, signed_headers, payload_hash])
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(signing_key(secret_key, date_stamp, region), string_to_sign.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    request = urllib.request.Request(f"https://{host}{uri}", data=body, headers=headers, method="PUT")
    with urllib.request.urlopen(request, timeout=90) as response:
        response.read()


def upload_with_retries(path: pathlib.Path, **kwargs: str) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            upload_once(path, **kwargs)
            return
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(attempt * 2)
    raise RuntimeError(f"upload failed for {path}: {last_error}")


def main() -> int:
    args = parse_args()
    env = load_env(args.env)
    s3_url = env["S3_URL"].rstrip("/") + "/"
    region = env.get("AWS_DEFAULT_REGION", "ap-south-1")
    parsed = urllib.parse.urlparse(s3_url)
    bucket = parsed.netloc.split(".")[0]
    prefix = parsed.path.strip("/")
    if prefix:
        prefix += "/"
    public_base = f"https://{bucket}.s3.{region}.amazonaws.com/{prefix}"

    args.state.parent.mkdir(parents=True, exist_ok=True)
    done = set(args.state.read_text(encoding="utf-8").splitlines()) if args.state.exists() else set()
    file_map = {path.name: path for path in args.audio_root.rglob("*.wav")}

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig", newline="")))
    segment_rows = [row for row in rows if SEGMENT_RE.search((row.get("filename") or "").strip())]
    missing = [row["filename"] for row in segment_rows if row["filename"] not in file_map]
    if missing:
        raise SystemExit("Missing local audio files:\n" + "\n".join(missing[:20]))

    for index, row in enumerate(segment_rows, start=1):
        name = row["filename"].strip()
        row[args.link_column] = public_base + urllib.parse.quote(name)
        if name not in done:
            upload_with_retries(
                file_map[name],
                bucket=bucket,
                key=prefix + name,
                region=region,
                access_key=env["AWS_ACCESS_KEY_ID"],
                secret_key=env["AWS_SECRET_ACCESS_KEY"],
            )
            with args.state.open("a", encoding="utf-8") as handle:
                handle.write(name + "\n")
            done.add(name)
        if index == 1 or index % 100 == 0 or index == len(segment_rows):
            print(f"done {len(done)}/{len(segment_rows)}", flush=True)

    fieldnames = list(rows[0].keys())
    if args.link_column not in fieldnames:
        fieldnames.append(args.link_column)

    backup = args.csv.with_suffix(args.csv.suffix + ".bak")
    shutil.copy2(args.csv, backup)
    tmp = args.csv.with_suffix(args.csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(segment_rows)
    os.replace(tmp, args.csv)

    print(f"kept_segments={len(segment_rows)}")
    print(f"dropped_complete_recordings={len(rows) - len(segment_rows)}")
    print(f"backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
