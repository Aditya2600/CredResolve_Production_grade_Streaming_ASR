#!/usr/bin/env python3
"""Upload the full-path WAVs listed in the Dataset-generation parent CSV to S3 and add an s3_audio_link column.

Parent CSV layout: row0 = title, row1 = header (filename,audio_link,language_id,transcript,ground_truth_transcript,Remarks),
row2+ = data, where `filename` holds the absolute local path to the WAV.
Reuses SigV4 upload helpers from upload_batch_transcript_segments_to_s3.py.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import hmac
import os
import pathlib
import shutil
import urllib.parse

from upload_batch_transcript_segments_to_s3 import load_env, signing_key, upload_with_retries


def presign_get(
    *, bucket: str, key: str, region: str, access_key: str, secret_key: str, expires_sec: int
) -> str:
    """Build a SigV4 presigned GET URL so the object is fetchable on a private bucket."""
    host = f"{bucket}.s3.{region}.amazonaws.com"
    now = dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    uri = "/" + urllib.parse.quote(key, safe="/")
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires_sec),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(query[k], safe='')}" for k in sorted(query)
    )
    canonical_request = "\n".join(
        ["GET", uri, canonical_query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"]
    )
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()]
    )
    signature = hmac.new(
        signing_key(secret_key, date_stamp, region), string_to_sign.encode(), hashlib.sha256
    ).hexdigest()
    return f"https://{host}{uri}?{canonical_query}&X-Amz-Signature={signature}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "data_correction/Dataset generation - batch_transcript-parent 2 - Sheet 1 - batch_transcript-pare.csv"
        ),
    )
    parser.add_argument("--env", type=pathlib.Path, default=pathlib.Path(".env"))
    parser.add_argument(
        "--state",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/s3_upload_parent_csv_done.txt"),
    )
    parser.add_argument("--link-column", default="s3_audio_link")
    parser.add_argument("--header-row", type=int, default=1, help="0-based index of the real header row.")
    parser.add_argument(
        "--presign-days",
        type=int,
        default=7,
        help="Write a presigned GET URL valid this many days (0 = bare public URL for a public bucket). "
        "Max 7 (S3 SigV4 limit).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = load_env(args.env)
    region = env.get("AWS_DEFAULT_REGION", "ap-south-1")
    parsed = urllib.parse.urlparse(env["S3_URL"].rstrip("/") + "/")
    bucket = parsed.netloc.split(".")[0]
    prefix = parsed.path.strip("/")
    prefix = prefix + "/" if prefix else ""
    public_base = f"https://{bucket}.s3.{region}.amazonaws.com/{prefix}"

    args.state.parent.mkdir(parents=True, exist_ok=True)
    done = set(args.state.read_text(encoding="utf-8").splitlines()) if args.state.exists() else set()

    all_rows = list(csv.reader(args.csv.open(encoding="utf-8-sig", newline="")))
    header = all_rows[args.header_row]
    data = all_rows[args.header_row + 1 :]
    idx = {name: i for i, name in enumerate(header)}
    fn_col = idx["filename"]
    if args.link_column in idx:
        link_col = idx[args.link_column]
    else:
        link_col = len(header)
        header.append(args.link_column)

    # Pre-flight: every referenced path must exist locally.
    missing = [r[fn_col] for r in data if r and r[fn_col].strip() and not os.path.exists(r[fn_col].strip())]
    if missing:
        raise SystemExit("Missing local audio files:\n" + "\n".join(missing[:20]))

    total = sum(1 for r in data if r and r[fn_col].strip())
    processed = 0
    for row in data:
        if not row or not row[fn_col].strip():
            continue
        local_path = pathlib.Path(row[fn_col].strip())
        name = local_path.name  # S3 key uses basename, not the local absolute path
        while len(row) <= link_col:
            row.append("")
        if args.presign_days > 0:
            row[link_col] = presign_get(
                bucket=bucket,
                key=prefix + name,
                region=region,
                access_key=env["AWS_ACCESS_KEY_ID"],
                secret_key=env["AWS_SECRET_ACCESS_KEY"],
                expires_sec=min(args.presign_days, 7) * 86400,
            )
        else:
            row[link_col] = public_base + urllib.parse.quote(name)
        if name not in done:
            upload_with_retries(
                local_path,
                bucket=bucket,
                key=prefix + name,
                region=region,
                access_key=env["AWS_ACCESS_KEY_ID"],
                secret_key=env["AWS_SECRET_ACCESS_KEY"],
            )
            with args.state.open("a", encoding="utf-8") as handle:
                handle.write(name + "\n")
            done.add(name)
        processed += 1
        if processed == 1 or processed % 100 == 0 or processed == total:
            print(f"done {processed}/{total}", flush=True)

    backup = args.csv.with_suffix(args.csv.suffix + ".bak")
    shutil.copy2(args.csv, backup)
    tmp = args.csv.with_suffix(args.csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(all_rows)
    os.replace(tmp, args.csv)

    print(f"uploaded_or_present={total}")
    print(f"link_column={args.link_column}")
    print(f"backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
