from __future__ import annotations

import math
import random
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Sampler


@dataclass(frozen=True)
class DurationBucketingConfig:
    enabled: bool
    bucket_edges: tuple[float, ...]
    curriculum_first_epoch: bool
    long_tail_threshold: float
    long_tail_batch_size: int
    max_total_batch_duration: float | None
    bucketing_strategy: str
    log_batch_duration_stats: bool
    seed: int


def parse_bucket_edges(raw_value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw_value, str):
        pieces = raw_value.replace(",", " ").split()
        values = [float(piece) for piece in pieces]
    else:
        values = [float(piece) for piece in raw_value]
    cleaned = sorted({value for value in values if value > 0.0})
    return tuple(cleaned)


def build_duration_bucketing_config(args: Any) -> DurationBucketingConfig:
    return DurationBucketingConfig(
        enabled=bool(args.use_duration_bucketing),
        bucket_edges=parse_bucket_edges(args.bucket_edges),
        curriculum_first_epoch=bool(args.curriculum_first_epoch),
        long_tail_threshold=float(args.long_tail_threshold),
        long_tail_batch_size=int(args.long_tail_batch_size),
        max_total_batch_duration=(
            float(args.max_total_batch_duration) if args.max_total_batch_duration is not None else None
        ),
        bucketing_strategy=str(args.bucketing_strategy),
        log_batch_duration_stats=bool(args.log_batch_duration_stats),
        seed=int(args.duration_bucketing_seed),
    )


class DurationAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        bucket_edges: Sequence[float],
        curriculum_first_epoch: bool,
        long_tail_threshold: float,
        long_tail_batch_size: int,
        max_total_batch_duration: float | None,
        bucketing_strategy: str,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if long_tail_batch_size <= 0:
            raise ValueError(f"long_tail_batch_size must be positive, got {long_tail_batch_size}")
        if max_total_batch_duration is not None and max_total_batch_duration <= 0.0:
            raise ValueError(
                f"max_total_batch_duration must be positive when provided, got {max_total_batch_duration}"
            )
        if bucketing_strategy not in {"fixed_order", "synced_randomized", "fully_randomized"}:
            raise ValueError(
                "bucketing_strategy must be one of "
                "['fixed_order', 'synced_randomized', 'fully_randomized']"
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.bucket_edges = tuple(sorted(float(edge) for edge in bucket_edges if float(edge) > 0.0))
        self.curriculum_first_epoch = bool(curriculum_first_epoch)
        self.long_tail_threshold = float(long_tail_threshold)
        self.long_tail_batch_size = int(long_tail_batch_size)
        self.max_total_batch_duration = (
            float(max_total_batch_duration) if max_total_batch_duration is not None else None
        )
        self.bucketing_strategy = str(bucketing_strategy)
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_global_batches: list[list[int]] | None = None
        self._cached_local_batches: list[list[int]] | None = None
        self._durations = self._extract_durations(dataset)
        self._bucket_to_indices = self._build_bucket_index()

    def _extract_durations(self, dataset: Any) -> list[float]:
        collection = getattr(getattr(dataset, "manifest_processor", None), "collection", None)
        if collection is None:
            raise ValueError(
                "Duration-aware bucketing requires a map-style NeMo dataset with "
                "dataset.manifest_processor.collection."
            )
        durations: list[float] = []
        for sample in collection:
            duration = getattr(sample, "duration", None)
            if duration is None:
                raise ValueError("Manifest sample is missing duration, cannot build duration buckets.")
            durations.append(float(duration))
        return durations

    def _base_bucket_index(self, duration: float) -> int:
        return bisect_left(self.bucket_edges, duration)

    def _logical_bucket_key(self, duration: float) -> tuple[int, int]:
        base_bucket = self._base_bucket_index(duration)
        is_long_tail = 1 if duration > self.long_tail_threshold else 0
        return is_long_tail, base_bucket

    def _build_bucket_index(self) -> dict[tuple[int, int], list[int]]:
        bucket_to_indices: dict[tuple[int, int], list[int]] = {}
        for index, duration in enumerate(self._durations):
            key = self._logical_bucket_key(duration)
            bucket_to_indices.setdefault(key, []).append(index)
        return bucket_to_indices

    def _bucket_sort_key(self, key: tuple[int, int]) -> tuple[int, int]:
        is_long_tail, base_bucket = key
        return is_long_tail, base_bucket

    def _ordered_bucket_keys(self, rng: random.Random) -> list[tuple[int, int]]:
        keys = sorted(self._bucket_to_indices, key=self._bucket_sort_key)
        if self.curriculum_first_epoch and self.epoch == 0:
            return keys
        if self.bucketing_strategy == "fixed_order":
            return keys
        shuffled = list(keys)
        rng.shuffle(shuffled)
        return shuffled

    def _target_batch_size(self, key: tuple[int, int]) -> int:
        is_long_tail, _ = key
        return self.long_tail_batch_size if is_long_tail else self.batch_size

    def _finalize_batch(self, batches: list[list[int]], current: list[int], *, min_items: int) -> None:
        if not current:
            return
        if self.drop_last and len(current) < min_items:
            return
        batches.append(list(current))

    def _build_global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        bucket_to_indices = {key: list(indices) for key, indices in self._bucket_to_indices.items()}
        for indices in bucket_to_indices.values():
            rng.shuffle(indices)

        ordered_keys = self._ordered_bucket_keys(rng)
        batches: list[list[int]] = []
        for key in ordered_keys:
            current: list[int] = []
            current_total_duration = 0.0
            max_items = self._target_batch_size(key)
            for index in bucket_to_indices[key]:
                duration = self._durations[index]
                if (
                    current
                    and self.max_total_batch_duration is not None
                    and current_total_duration + duration > self.max_total_batch_duration
                ):
                    self._finalize_batch(batches, current, min_items=max_items)
                    current = []
                    current_total_duration = 0.0

                current.append(index)
                current_total_duration += duration

                if len(current) >= max_items:
                    self._finalize_batch(batches, current, min_items=max_items)
                    current = []
                    current_total_duration = 0.0

            self._finalize_batch(batches, current, min_items=max_items)

        if not (self.curriculum_first_epoch and self.epoch == 0) and self.bucketing_strategy == "fully_randomized":
            rng.shuffle(batches)
        return batches

    def _get_epoch_batches(self) -> tuple[list[list[int]], list[list[int]]]:
        if self._cached_epoch == self.epoch and self._cached_global_batches is not None and self._cached_local_batches is not None:
            return self._cached_global_batches, self._cached_local_batches

        global_batches = self._build_global_batches()
        if self.num_replicas > 1 and self.drop_last:
            keep_batches = len(global_batches) - (len(global_batches) % self.num_replicas)
            global_batches = global_batches[:keep_batches]
        local_batches = global_batches[self.rank :: self.num_replicas]
        self._cached_epoch = self.epoch
        self._cached_global_batches = global_batches
        self._cached_local_batches = local_batches
        return global_batches, local_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_global_batches = None
        self._cached_local_batches = None

    def __iter__(self) -> Iterator[list[int]]:
        _, local_batches = self._get_epoch_batches()
        return iter(local_batches)

    def __len__(self) -> int:
        _, local_batches = self._get_epoch_batches()
        return len(local_batches)

    def _bucket_range_label(self, base_bucket: int) -> str:
        if not self.bucket_edges:
            return "all"
        if base_bucket == 0:
            return f"<= {self.bucket_edges[0]:g}s"
        if base_bucket >= len(self.bucket_edges):
            return f"> {self.bucket_edges[-1]:g}s"
        return f"({self.bucket_edges[base_bucket - 1]:g}s, {self.bucket_edges[base_bucket]:g}s]"

    def summary(self) -> dict[str, Any]:
        global_batches, local_batches = self._get_epoch_batches()
        bucket_rows: list[dict[str, Any]] = []
        for key in sorted(self._bucket_to_indices, key=self._bucket_sort_key):
            is_long_tail, base_bucket = key
            indices = self._bucket_to_indices[key]
            durations = [self._durations[index] for index in indices]
            bucket_rows.append(
                {
                    "bucket_label": self._bucket_range_label(base_bucket),
                    "is_long_tail": bool(is_long_tail),
                    "sample_count": len(indices),
                    "batch_size": self._target_batch_size(key),
                    "min_duration_sec": round(min(durations), 4),
                    "max_duration_sec": round(max(durations), 4),
                    "mean_duration_sec": round(sum(durations) / len(durations), 4),
                }
            )
        return {
            "epoch": int(self.epoch),
            "num_replicas": int(self.num_replicas),
            "rank": int(self.rank),
            "batch_size": int(self.batch_size),
            "bucket_edges": [float(edge) for edge in self.bucket_edges],
            "curriculum_first_epoch": bool(self.curriculum_first_epoch),
            "long_tail_threshold": float(self.long_tail_threshold),
            "long_tail_batch_size": int(self.long_tail_batch_size),
            "max_total_batch_duration": self.max_total_batch_duration,
            "bucketing_strategy": self.bucketing_strategy,
            "sample_count": len(self._durations),
            "global_batch_count": len(global_batches),
            "local_batch_count": len(local_batches),
            "long_tail_sample_count": sum(
                len(indices) for key, indices in self._bucket_to_indices.items() if key[0] == 1
            ),
            "bucket_rows": bucket_rows,
        }


def build_duration_bucketed_dataloader(
    dataset: Any,
    *,
    collate_fn: Any,
    batch_size: int,
    bucket_edges: Sequence[float],
    curriculum_first_epoch: bool,
    long_tail_threshold: float,
    long_tail_batch_size: int,
    max_total_batch_duration: float | None,
    bucketing_strategy: str,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    num_replicas: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DurationAwareBatchSampler]:
    sampler = DurationAwareBatchSampler(
        dataset,
        batch_size=batch_size,
        bucket_edges=bucket_edges,
        curriculum_first_epoch=curriculum_first_epoch,
        long_tail_threshold=long_tail_threshold,
        long_tail_batch_size=long_tail_batch_size,
        max_total_batch_duration=max_total_batch_duration,
        bucketing_strategy=bucketing_strategy,
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
        drop_last=drop_last,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader, sampler
