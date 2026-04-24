#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine one or more CSV files and sample rows to match a target language distribution."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the sampled CSV.",
    )
    parser.add_argument(
        "--language-column",
        default="language",
        help="Language column name. Default: language",
    )
    parser.add_argument(
        "--target",
        required=True,
        help=(
            "Comma-separated language=percent list, for example "
            "'HINDI=27,TELUGU=20,MARATHI=14'."
        ),
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        help=(
            "Optional total sampled rows. If omitted, uses the largest possible total "
            "without oversampling."
        ),
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Allow sampling with replacement when a target count exceeds available rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling. Default: 42",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        help="Optional path to write the train split CSV.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        help="Optional path to write the validation split CSV.",
    )
    parser.add_argument(
        "--val-percent",
        type=float,
        default=10.0,
        help="Validation split percentage when train/val outputs are requested. Default: 10.0",
    )
    return parser.parse_args()


def parse_target_distribution(raw_value: str) -> OrderedDict[str, float]:
    target: OrderedDict[str, float] = OrderedDict()
    for item in raw_value.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"Invalid target item '{chunk}'. Expected LANGUAGE=percent.")
        language, percent = chunk.split("=", 1)
        language_key = language.strip().upper()
        if not language_key:
            raise SystemExit(f"Invalid target item '{chunk}'. Language name is empty.")
        try:
            percent_value = float(percent.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid percentage in target item '{chunk}'.") from exc
        if percent_value <= 0:
            raise SystemExit(f"Invalid percentage in target item '{chunk}'. Must be > 0.")
        if language_key in target:
            raise SystemExit(f"Duplicate language in target distribution: {language_key}")
        target[language_key] = percent_value

    if not target:
        raise SystemExit("Target distribution is empty.")
    return target


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise SystemExit(f"Input CSV not found: {resolved}")
        frame = pd.read_csv(resolved, dtype=str, keep_default_na=True, encoding="utf-8-sig")
        if frame.empty:
            raise SystemExit(f"Input CSV is empty: {resolved}")
        frame = frame.copy()
        frame["distribution_source_file"] = resolved.name
        frame["distribution_source_path"] = str(resolved)
        frame["distribution_source_row_number"] = range(2, len(frame) + 2)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_language(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def resolve_total_rows(
    *,
    available_counts: dict[str, int],
    target_distribution: OrderedDict[str, float],
    requested_total_rows: int | None,
    oversample: bool,
) -> int:
    if requested_total_rows is not None:
        if requested_total_rows <= 0:
            raise SystemExit("--total-rows must be greater than 0 when provided.")
        if not oversample:
            for language, share in target_distribution.items():
                required = requested_total_rows * (share / sum(target_distribution.values()))
                if required - 1e-9 > available_counts.get(language, 0):
                    raise SystemExit(
                        "Requested --total-rows requires oversampling for "
                        f"{language}. Re-run with --oversample or a smaller total."
                    )
        return requested_total_rows

    if oversample:
        return sum(available_counts.get(language, 0) for language in target_distribution)

    total_target_share = sum(target_distribution.values())
    max_total = min(
        available_counts.get(language, 0) / (share / total_target_share)
        for language, share in target_distribution.items()
    )
    return int(max_total)


def allocate_counts(
    *,
    total_rows: int,
    target_distribution: OrderedDict[str, float],
) -> OrderedDict[str, int]:
    total_share = sum(target_distribution.values())
    exact_counts = OrderedDict(
        (language, total_rows * (share / total_share))
        for language, share in target_distribution.items()
    )
    allocated = OrderedDict((language, int(count)) for language, count in exact_counts.items())
    remainder = total_rows - sum(allocated.values())

    ranked_remainders = sorted(
        (
            (exact_counts[language] - allocated[language], language)
            for language in target_distribution
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, language in ranked_remainders[:remainder]:
        allocated[language] += 1
    return allocated


def sample_rows(
    frame: pd.DataFrame,
    *,
    language_column: str,
    target_counts: OrderedDict[str, int],
    oversample: bool,
    seed: int,
) -> pd.DataFrame:
    sampled_frames: list[pd.DataFrame] = []
    for offset, (language, count) in enumerate(target_counts.items()):
        language_frame = frame.loc[frame[language_column] == language].copy()
        if language_frame.empty:
            raise SystemExit(f"No rows found for target language {language}")
        replace = oversample and count > len(language_frame)
        sampled = language_frame.sample(n=count, replace=replace, random_state=seed + offset)
        sampled_frames.append(sampled)

    combined = pd.concat(sampled_frames, ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_summary(
    *,
    input_paths: list[Path],
    output_path: Path,
    total_rows: int,
    target_counts: OrderedDict[str, int],
    available_counts: dict[str, int],
    sampled_frame: pd.DataFrame,
    language_column: str,
) -> dict[str, object]:
    sampled_counts = (
        sampled_frame[language_column]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    percentages = {
        language: round((count / total_rows) * 100, 4) if total_rows else 0.0
        for language, count in sampled_counts.items()
    }
    return {
        "input_files": [str(path.expanduser().resolve()) for path in input_paths],
        "output_file": str(output_path.expanduser().resolve()),
        "total_rows": total_rows,
        "available_counts": available_counts,
        "target_counts": dict(target_counts),
        "sampled_counts": sampled_counts,
        "sampled_percentages": percentages,
    }


def allocate_fractional_split_counts(
    counts_by_language: OrderedDict[str, int],
    fraction: float,
) -> OrderedDict[str, int]:
    exact_counts = OrderedDict(
        (language, count * fraction)
        for language, count in counts_by_language.items()
    )
    allocated = OrderedDict((language, int(count)) for language, count in exact_counts.items())
    target_total = int(round(sum(counts_by_language.values()) * fraction))
    remainder = target_total - sum(allocated.values())

    ranked_remainders = sorted(
        (
            (exact_counts[language] - allocated[language], language)
            for language in counts_by_language
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, language in ranked_remainders[:remainder]:
        allocated[language] += 1
    return allocated


def split_sampled_rows(
    frame: pd.DataFrame,
    *,
    language_column: str,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, OrderedDict[str, int], OrderedDict[str, int]]:
    if not 0.0 < val_fraction < 1.0:
        raise SystemExit("--val-percent must be greater than 0 and less than 100.")

    counts_by_language = OrderedDict(
        (
            language,
            int(count),
        )
        for language, count in frame[language_column].value_counts().sort_index().items()
    )
    val_counts = allocate_fractional_split_counts(counts_by_language, val_fraction)
    train_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []

    for offset, (language, val_count) in enumerate(val_counts.items()):
        language_frame = frame.loc[frame[language_column] == language].copy()
        if val_count <= 0:
            train_frames.append(language_frame)
            continue

        val_frame = language_frame.sample(n=val_count, random_state=seed + offset)
        train_frame = language_frame.drop(index=val_frame.index)
        train_frames.append(train_frame)
        val_frames.append(val_frame)

    train_df = pd.concat(train_frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if val_frames:
        val_df = pd.concat(val_frames, ignore_index=True).sample(frac=1.0, random_state=seed + 999).reset_index(drop=True)
    else:
        val_df = frame.iloc[0:0].copy()

    train_counts = OrderedDict(
        (language, int((train_df[language_column] == language).sum()))
        for language in counts_by_language
    )
    val_counts_actual = OrderedDict(
        (language, int((val_df[language_column] == language).sum()))
        for language in counts_by_language
    )
    return train_df, val_df, train_counts, val_counts_actual


def main() -> int:
    args = parse_args()
    target_distribution = parse_target_distribution(args.target)
    frame = load_inputs(args.input)

    if args.language_column not in frame.columns:
        raise SystemExit(
            f"Language column '{args.language_column}' not found. "
            f"Available columns: {sorted(frame.columns)}"
        )

    frame = frame.copy()
    frame[args.language_column] = frame[args.language_column].map(normalize_language)
    frame = frame.loc[frame[args.language_column].isin(target_distribution.keys())].copy()
    if frame.empty:
        raise SystemExit("No rows remain after filtering to the target languages.")

    available_counts = {
        language: int((frame[args.language_column] == language).sum())
        for language in target_distribution
    }
    missing_languages = [language for language, count in available_counts.items() if count == 0]
    if missing_languages:
        raise SystemExit(
            "Some target languages have no rows in the input data: "
            + ", ".join(missing_languages)
        )

    total_rows = resolve_total_rows(
        available_counts=available_counts,
        target_distribution=target_distribution,
        requested_total_rows=args.total_rows,
        oversample=args.oversample,
    )
    target_counts = allocate_counts(total_rows=total_rows, target_distribution=target_distribution)
    sampled_frame = sample_rows(
        frame,
        language_column=args.language_column,
        target_counts=target_counts,
        oversample=args.oversample,
        seed=args.seed,
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_frame.to_csv(output_path, index=False)

    summary = build_summary(
        input_paths=args.input,
        output_path=output_path,
        total_rows=total_rows,
        target_counts=target_counts,
        available_counts=available_counts,
        sampled_frame=sampled_frame,
        language_column=args.language_column,
    )

    split_requested = args.train_output is not None or args.val_output is not None
    if split_requested:
        if args.train_output is None or args.val_output is None:
            raise SystemExit("Pass both --train-output and --val-output to write a train/val split.")
        train_df, val_df, train_counts, val_counts = split_sampled_rows(
            sampled_frame,
            language_column=args.language_column,
            val_fraction=args.val_percent / 100.0,
            seed=args.seed,
        )
        train_output_path = args.train_output.expanduser().resolve()
        val_output_path = args.val_output.expanduser().resolve()
        train_output_path.parent.mkdir(parents=True, exist_ok=True)
        val_output_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)
        summary["train_split"] = {
            "output_file": str(train_output_path),
            "total_rows": int(len(train_df)),
            "counts": dict(train_counts),
        }
        summary["val_split"] = {
            "output_file": str(val_output_path),
            "total_rows": int(len(val_df)),
            "counts": dict(val_counts),
        }

    if args.summary_json is not None:
        summary_path = args.summary_json.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Output CSV: {output_path}")
    print(f"Total rows: {summary['total_rows']}")
    print("Target counts:")
    for language, count in target_counts.items():
        print(f"  {language}: {count} (available={available_counts[language]})")
    print("Sampled percentages:")
    for language in target_counts:
        percent = summary["sampled_percentages"].get(language, 0.0)
        print(f"  {language}: {percent:.4f}%")
    if split_requested:
        print(f"Train CSV: {summary['train_split']['output_file']}")
        print(f"Train rows: {summary['train_split']['total_rows']}")
        for language, count in summary["train_split"]["counts"].items():
            print(f"  train {language}: {count}")
        print(f"Val CSV: {summary['val_split']['output_file']}")
        print(f"Val rows: {summary['val_split']['total_rows']}")
        for language, count in summary["val_split"]["counts"].items():
            print(f"  val {language}: {count}")
    if args.summary_json is not None:
        print(f"Summary JSON: {args.summary_json.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
