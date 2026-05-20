#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
API_BASE = "https://api.sarvam.ai/speech-to-text/job/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a recording-url CSV with Sarvam Batch STT.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url-column", default="cr_recording_url")
    parser.add_argument("--id-column", default="job_id")
    parser.add_argument("--start-row", type=int, default=1, help="1-based CSV data row, excluding header.")
    parser.add_argument("--limit", type=int, help="Maximum CSV rows to include after --start-row.")
    parser.add_argument("--batch-size", type=int, default=20, help="Sarvam supports up to 20 files per job.")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel downloads/uploads within a batch.")
    parser.add_argument("--batch-concurrency", type=int, default=1, help="Parallel Sarvam jobs to run at once.")
    parser.add_argument("--max-batches", type=int, help="Stop after this many Sarvam jobs.")
    parser.add_argument("--poll-sec", type=float, default=10.0)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--language-code", default="unknown")
    parser.add_argument("--model", default="saaras:v3")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="SARVAM_API_KEY")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def window_rows(rows: list[dict[str, str]], *, start_row: int, limit: int | None) -> list[dict[str, str]]:
    start = max(0, start_row - 1)
    return rows[start:] if limit is None else rows[start : start + max(0, limit)]


def sanitize_stem(name: str) -> str:
    stem = Path(name or "recording").stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "recording"


def audio_name_for_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    suffix = Path(name).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        suffix = ".mp3"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{sanitize_stem(name)}__{digest}{suffix}"


def request_json(
    url: str,
    *,
    api_key: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_sec: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {body}") from exc
    parsed = json.loads(body or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON")
    return parsed


def download_url(url: str, path: Path, *, timeout_sec: float, retries: int) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CredResolve-Sarvam-batch/1.0"})
            with urllib.request.urlopen(request, timeout=timeout_sec) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_path.replace(path)
            return
        except Exception as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError(f"download_failed:{last_error}")


def put_signed_file(upload_url: str, path: Path, *, timeout_sec: float) -> None:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Content-Type": content_type, "x-ms-blob-type": "BlockBlob"}
    request = urllib.request.Request(upload_url, data=path.read_bytes(), headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"upload failed for {path.name}: HTTP {exc.code}: {body}") from exc


def download_signed_json(download_url: str, path: Path, *, timeout_sec: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(download_url)
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        path.write_bytes(response.read())


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def progress_iter(items: Any, *, enabled: bool, **kwargs: Any) -> Any:
    if enabled and tqdm is not None:
        return tqdm(items, **kwargs)
    return items


def run_parallel(
    items: list[Any],
    fn: Any,
    *,
    concurrency: int,
    progress: bool,
    desc: str,
    unit: str,
) -> list[Any]:
    if not items:
        return []
    workers = max(1, int(concurrency))
    if workers == 1:
        return [fn(item) for item in progress_iter(items, enabled=progress, desc=desc, unit=unit)]

    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        iterator = as_completed(futures)
        iterator = progress_iter(iterator, enabled=progress, total=len(futures), desc=desc, unit=unit)
        for future in iterator:
            results.append(future.result())
    return results


def transcript_from_result(path: Path) -> str:
    if not path.is_file():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("transcript") or data.get("text") or "").strip()


def diarized_from_result(path: Path) -> str:
    if not path.is_file():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = ((data.get("diarized_transcript") or {}).get("entries") or [])
    lines = []
    for entry in entries:
        text = str(entry.get("transcript") or "").strip()
        if text:
            speaker = entry.get("speaker_id")
            lines.append(f"speaker_{speaker}: {text}")
    return "\n".join(lines)


def process_batch(
    *,
    batch_index: int,
    files: list[dict[str, Any]],
    api_key: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_path = output_dir / "sarvam_jobs" / f"batch_{batch_index:04d}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        params: dict[str, Any] = {
            "model": args.model,
            "language_code": args.language_code,
            "with_timestamps": not args.no_timestamps,
            "with_diarization": not args.no_diarization,
        }
        if args.num_speakers is not None:
            params["num_speakers"] = args.num_speakers
        created = request_json(
            API_BASE,
            api_key=api_key,
            method="POST",
            payload={"job_parameters": params},
            timeout_sec=args.timeout_sec,
        )
        state = {"batch_index": batch_index, "job_id": created["job_id"], "files": files, "created": created}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    job_id = state["job_id"]
    result_dir = output_dir / "sarvam_results" / job_id
    expected_done = all((result_dir / item["result_name"]).exists() for item in state["files"])
    if args.resume and expected_done:
        return state

    if "upload" not in state:
        upload = request_json(
            f"{API_BASE}/upload-files",
            api_key=api_key,
            method="POST",
            payload={"job_id": job_id, "files": [item["sarvam_name"] for item in state["files"]]},
            timeout_sec=args.timeout_sec,
        )
        def upload_one(item: dict[str, Any]) -> None:
            upload_info = (upload.get("upload_urls") or {}).get(item["sarvam_name"]) or {}
            put_signed_file(upload_info["file_url"], Path(item["audio_path"]), timeout_sec=args.timeout_sec)

        run_parallel(
            state["files"],
            upload_one,
            concurrency=args.concurrency,
            progress=args.progress,
            desc=f"upload batch {batch_index}",
            unit="file",
        )
        state["upload"] = upload
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if "started" not in state:
        state["started"] = request_json(
            f"{API_BASE}/{job_id}/start",
            api_key=api_key,
            method="POST",
            payload={},
            timeout_sec=args.timeout_sec,
        )
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    while True:
        status = request_json(
            f"{API_BASE}/{job_id}/status",
            api_key=api_key,
            timeout_sec=args.timeout_sec,
        )
        state["status"] = status
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if status.get("job_state") in {"Completed", "Failed"}:
            break
        time.sleep(max(0.005, float(args.poll_sec)))

    status = state["status"]
    output_by_input: dict[str, str] = {}
    errors_by_input: dict[str, str] = {}
    for detail in status.get("job_details") or []:
        input_name = str(((detail.get("inputs") or [{}])[0]).get("file_name") or "")
        outputs = detail.get("outputs") or []
        if outputs:
            output_by_input[input_name] = str(outputs[0].get("file_name") or "")
        if detail.get("error_message") or detail.get("exception_name"):
            errors_by_input[input_name] = f"{detail.get('exception_name') or ''}:{detail.get('error_message') or ''}"

    output_names = [name for name in output_by_input.values() if name]
    if output_names:
        download = request_json(
            f"{API_BASE}/download-files",
            api_key=api_key,
            method="POST",
            payload={"job_id": job_id, "files": output_names},
            timeout_sec=args.timeout_sec,
        )
        for item in state["files"]:
            output_name = output_by_input.get(item["sarvam_name"], "")
            item["sarvam_output_name"] = output_name
            item["sarvam_error"] = errors_by_input.get(item["sarvam_name"], "")
            if not output_name:
                continue
            url_info = (download.get("download_urls") or {}).get(output_name) or {}
            result_path = result_dir / item["result_name"]
            download_signed_json(url_info["file_url"], result_path, timeout_sec=args.timeout_sec)
            item["result_path"] = str(result_path)
        state["download"] = download
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return state


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set; pass --env-file or export it.")

    rows = window_rows(csv_rows(args.input_csv), start_row=args.start_row, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = args.output_dir / "downloaded_audio"
    manifest_path = args.output_dir / "sarvam_manifest.csv"
    summary_path = args.output_dir / "sarvam_transcripts.csv"

    by_url: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows, start=args.start_row):
        url = str(row.get(args.url_column) or "").strip()
        if not url:
            continue
        item = by_url.setdefault(
            url,
            {
                "url": url,
                "source_name": audio_name_for_url(url),
                "row_indices": [],
                "job_ids": [],
            },
        )
        item["row_indices"].append(row_index)
        item["job_ids"].append(str(row.get(args.id_column) or ""))

    work_items = list(by_url.values())
    for index, item in enumerate(work_items):
        audio_path = download_dir / item["source_name"]
        item["audio_path"] = str(audio_path)
        item["sarvam_name"] = f"{index:06d}_{item['source_name']}"
        item["result_name"] = f"{Path(item['sarvam_name']).stem}.json"

    def download_one(item: dict[str, Any]) -> None:
        audio_path = Path(item["audio_path"])
        download_url(item["url"], audio_path, timeout_sec=args.timeout_sec, retries=args.download_retries)

    run_parallel(
        work_items,
        download_one,
        concurrency=args.concurrency,
        progress=args.progress,
        desc="download recordings",
        unit="file",
    )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sarvam_name", "source_name", "url", "audio_path", "row_indices"])
        writer.writeheader()
        for item in work_items:
            writer.writerow(
                {
                    "sarvam_name": item["sarvam_name"],
                    "source_name": item["source_name"],
                    "url": item["url"],
                    "audio_path": item["audio_path"],
                    "row_indices": json.dumps(item["row_indices"]),
                }
            )

    batches = chunks(work_items, min(20, max(1, int(args.batch_size))))
    if args.max_batches is not None:
        batches = batches[: max(0, args.max_batches)]

    def run_batch(batch_pair: tuple[int, list[dict[str, Any]]]) -> dict[str, Any]:
        batch_index, batch_files = batch_pair
        print(f"batch {batch_index}/{len(batches)}: {len(batch_files)} files", flush=True)
        return process_batch(
            batch_index=batch_index,
            files=batch_files,
            api_key=api_key,
            output_dir=args.output_dir,
            args=args,
        )

    processed_by_url: dict[str, dict[str, Any]] = {}
    batch_iter = list(enumerate(batches, start=1))
    states = run_parallel(
        batch_iter,
        run_batch,
        concurrency=args.batch_concurrency,
        progress=args.progress,
        desc="sarvam batches",
        unit="batch",
    )
    for state in states:
        for item in state["files"]:
            processed_by_url[item["url"]] = item

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "input_row",
            args.id_column,
            args.url_column,
            "status",
            "transcript",
            "diarized_transcript",
            "sarvam_json_path",
            "sarvam_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(rows, start=args.start_row):
            url = str(row.get(args.url_column) or "").strip()
            item = processed_by_url.get(url)
            result_path = Path(item["result_path"]) if item and item.get("result_path") else Path("/dev/null")
            status = "ok" if result_path.exists() else ("not_processed" if url else "missing_url")
            writer.writerow(
                {
                    "input_row": row_index,
                    args.id_column: row.get(args.id_column, ""),
                    args.url_column: url,
                    "status": status,
                    "transcript": transcript_from_result(result_path),
                    "diarized_transcript": diarized_from_result(result_path),
                    "sarvam_json_path": str(result_path) if result_path.exists() else "",
                    "sarvam_error": (item or {}).get("sarvam_error", ""),
                }
            )

    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
