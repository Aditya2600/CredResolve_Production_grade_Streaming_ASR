#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable


INDICVOICES_DATASET_ID = "ai4bharat/IndicVoices"
VAANI_DATASET_ID = "ARTPARK-IISc/Vaani-transcription-part"
VAANI_HINDI_CONFIG = "audio/Hindi"

# Backward-compatible name used by existing tools.
DATASET_ID = INDICVOICES_DATASET_ID


def _resolve_cache_dir(cache_dir: Path | str | None) -> str | None:
    if cache_dir is None:
        return None
    return str(Path(cache_dir).expanduser().resolve())


def _build_explicit_parquet_paths(
    *,
    dataset_id: str,
    dataset_config: str,
    split: str,
    token: str,
) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(dataset_id, repo_type="dataset", token=token)
    
    prefixes = [
        f"{dataset_config}/{split}-",
        f"audio/{dataset_config}/{split}-",
        f"data/{dataset_config}/{split}-",
        f"data/{dataset_config}-{split}-",
        f"{dataset_config}/data/{split}-",
    ]
    
    parquet_files = []
    for prefix in prefixes:
        parquet_files = sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))
        if parquet_files:
            break
            
    if not parquet_files:
        raise SystemExit(
            f"Could not find parquet files for dataset config={dataset_config!r} split={split!r} in {dataset_id}."
        )
    return parquet_files


def _iter_explicit_parquet_rows(
    *,
    dataset_id: str,
    parquet_paths: list[str],
    token: str,
) -> Iterable[dict]:
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem(token=token)
    for path in parquet_paths:
        with fs.open(f"datasets/{dataset_id}/{path}", "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            for batch in parquet_file.iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    yield row


def load_hf_speech_stream(
    *,
    dataset_id: str,
    dataset_config: str,
    split: str,
    token: str,
    cache_dir: Path | str | None,
    force_explicit_parquet: bool = False,
):
    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    common_kwargs = {
        "token": token,
        "streaming": True,
    }
    if resolved_cache_dir is not None:
        common_kwargs["cache_dir"] = resolved_cache_dir

    if not force_explicit_parquet:
        from datasets import load_dataset

        try:
            dataset = load_dataset(
                dataset_id,
                dataset_config,
                split=split,
                **common_kwargs,
            )
            return dataset.decode(False) if hasattr(dataset, "decode") else dataset
        except ValueError as exc:
            # `fsspec` 2026 tightened `**` glob parsing, which can break the
            # automatic dataset pattern inference on some Hub dataset repos.
            if "Invalid pattern: '**' can only be an entire path component" not in str(exc):
                raise

    parquet_paths = _build_explicit_parquet_paths(
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        split=split,
        token=token,
    )
    return _iter_explicit_parquet_rows(
        dataset_id=dataset_id,
        parquet_paths=parquet_paths,
        token=token,
    )


def load_indicvoices_stream(
    *,
    dataset_config: str,
    split: str,
    token: str,
    cache_dir: Path | str | None,
):
    return load_hf_speech_stream(
        dataset_id=INDICVOICES_DATASET_ID,
        dataset_config=dataset_config,
        split=split,
        token=token,
        cache_dir=cache_dir,
    )


def load_vaani_stream(
    *,
    dataset_config: str = VAANI_HINDI_CONFIG,
    split: str = "train",
    token: str,
    cache_dir: Path | str | None,
):
    return load_hf_speech_stream(
        dataset_id=VAANI_DATASET_ID,
        dataset_config=dataset_config,
        split=split,
        token=token,
        cache_dir=cache_dir,
        force_explicit_parquet=True,
    )
