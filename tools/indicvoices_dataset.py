#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


DATASET_ID = "ai4bharat/IndicVoices"


def _resolve_cache_dir(cache_dir: Path | str | None) -> str | None:
    if cache_dir is None:
        return None
    return str(Path(cache_dir).expanduser().resolve())


def _build_explicit_parquet_paths(
    *,
    dataset_config: str,
    split: str,
    token: str,
) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(DATASET_ID, repo_type="dataset", token=token)
    prefix = f"{dataset_config}/{split}-"
    parquet_files = sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))
    if not parquet_files:
        raise SystemExit(
            f"Could not find parquet files for dataset config={dataset_config!r} split={split!r} in {DATASET_ID}."
        )
    return parquet_files


def _iter_explicit_parquet_rows(*, parquet_paths: list[str], token: str):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem(token=token)
    for path in parquet_paths:
        with fs.open(f"datasets/{DATASET_ID}/{path}", "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            for batch in parquet_file.iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    yield row


def load_indicvoices_stream(
    *,
    dataset_config: str,
    split: str,
    token: str,
    cache_dir: Path | str | None,
):
    from datasets import load_dataset

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    common_kwargs = {
        "token": token,
        "streaming": True,
    }
    if resolved_cache_dir is not None:
        common_kwargs["cache_dir"] = resolved_cache_dir

    try:
        dataset = load_dataset(
            DATASET_ID,
            dataset_config,
            split=split,
            **common_kwargs,
        )
        return dataset.decode(False)
    except ValueError as exc:
        # `fsspec` 2026 tightened `**` glob parsing, which can break the
        # automatic dataset pattern inference on some Hub dataset repos.
        if "Invalid pattern: '**' can only be an entire path component" not in str(exc):
            raise

    parquet_paths = _build_explicit_parquet_paths(
        dataset_config=dataset_config,
        split=split,
        token=token,
    )
    return _iter_explicit_parquet_rows(
        parquet_paths=parquet_paths,
        token=token,
    )
