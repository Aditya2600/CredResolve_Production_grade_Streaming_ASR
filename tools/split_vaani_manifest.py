#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratify a normalized Vaani NeMo manifest into train/dev/test JSONL files.")
    parser.add_argument("--input", type=Path, required=True, help="Normalized multilingual Vaani manifest.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for train/dev/test manifests.")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{lineno}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: manifest is empty")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def language_key(row: dict[str, Any]) -> str:
    return str(row.get("language") or row.get("language_id") or row.get("lang") or "unknown")


def validate_unique_rows(rows: list[dict[str, Any]]) -> None:
    for key in ("id", "audio_filepath"):
        seen: set[str] = set()
        duplicates: list[str] = []
        for row in rows:
            value = row.get(key)
            if value is None or str(value) == "":
                continue
            text = str(value)
            if text in seen:
                duplicates.append(text)
            seen.add(text)
        if duplicates:
            sample = ", ".join(sorted(set(duplicates))[:5])
            raise SystemExit(f"duplicate {key} values would make overlap checks unsafe: {sample}")


def allocate_counts(n_rows: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    raw_counts = [n_rows * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remaining = n_rows - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (raw_counts[index] - counts[index], ratios[index]), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1

    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if n_rows >= len(positive):
        for index in positive:
            if counts[index] == 0:
                donor = max((i for i in positive if counts[i] > 1), key=lambda i: counts[i], default=None)
                if donor is not None:
                    counts[donor] -= 1
                    counts[index] += 1
    return counts[0], counts[1], counts[2]


def split_rows(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise SystemExit(f"split ratios must sum to 1.0, got {ratio_sum:g}")
    if min(train_ratio, dev_ratio, test_ratio) < 0:
        raise SystemExit("split ratios must be non-negative")

    validate_unique_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[language_key(row)].append(row)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for language in sorted(grouped):
        group = list(grouped[language])
        rng.shuffle(group)
        n_train, n_dev, n_test = allocate_counts(len(group), (train_ratio, dev_ratio, test_ratio))
        splits["train"].extend(group[:n_train])
        splits["dev"].extend(group[n_train : n_train + n_dev])
        splits["test"].extend(group[n_train + n_dev : n_train + n_dev + n_test])

    for name in SPLITS:
        rng.shuffle(splits[name])
    return splits


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_counts: Counter[str] = Counter()
    duration_counts: Counter[str] = Counter()
    for row in rows:
        lang = language_key(row)
        row_counts[lang] += 1
        try:
            duration_counts[lang] += float(row.get("duration") or 0.0)
        except (TypeError, ValueError):
            pass
    return {
        "rows": len(rows),
        "rows_by_language": dict(sorted(row_counts.items())),
        "duration_hours_by_language": {
            lang: round(seconds / 3600.0, 6)
            for lang, seconds in sorted(duration_counts.items())
        },
        "duration_hours_total": round(sum(duration_counts.values()) / 3600.0, 6),
    }


def assert_no_overlap(splits: dict[str, list[dict[str, Any]]]) -> None:
    for key in ("id", "audio_filepath"):
        owners: dict[str, str] = {}
        for split, rows in splits.items():
            for row in rows:
                value = row.get(key)
                if value is None or str(value) == "":
                    continue
                text = str(value)
                if text in owners:
                    raise SystemExit(f"{key} overlap between {owners[text]} and {split}: {text}")
                owners[text] = split


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input.expanduser().resolve())
    splits = split_rows(
        rows,
        train_ratio=float(args.train_ratio),
        dev_ratio=float(args.dev_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    assert_no_overlap(splits)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        write_jsonl(out_dir / f"{name}.jsonl", splits[name])

    summary = {
        "input": str(args.input.expanduser().resolve()),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "dev": float(args.dev_ratio),
            "test": float(args.test_ratio),
        },
        "source": summarize(rows),
        "splits": {name: summarize(splits[name]) for name in SPLITS},
        "outputs": {name: str(out_dir / f"{name}.jsonl") for name in SPLITS},
    }
    summary["outputs"]["summary_json"] = str(out_dir / "split_summary.json")
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
