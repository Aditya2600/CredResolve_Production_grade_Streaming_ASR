#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar events from a run directory or event file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="TensorBoard run directory or a single events.out.tfevents* file.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Output CSV path. Default: <run-dir>/tensorboard_scalars.csv",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        help="Output summary JSON path. Default: <run-dir>/tensorboard_scalar_summary.json",
    )
    parser.add_argument(
        "--tags",
        help="Optional comma-separated scalar tag allowlist, e.g. val_wer,val_wer_ctc,train_loss.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Also print a compact tag summary to stdout.",
    )
    return parser.parse_args()


def import_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "TensorBoard is required. Install it in this environment, for example: "
            "`pip install tensorboard`."
        ) from exc
    return EventAccumulator


def default_output_dir(input_path: Path) -> Path:
    resolved = input_path.expanduser().resolve()
    return resolved.parent if resolved.is_file() else resolved


def parse_tag_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    tags = {item.strip() for item in raw.split(",") if item.strip()}
    return tags or None


def scalar_summary(events: list[Any]) -> dict[str, Any]:
    values = [float(event.value) for event in events]
    steps = [int(event.step) for event in events]
    wall_times = [float(event.wall_time) for event in events]

    if not events:
        return {
            "count": 0,
            "first_step": None,
            "last_step": None,
            "first_value": None,
            "last_value": None,
            "min_value": None,
            "min_step": None,
            "max_value": None,
            "max_step": None,
        }

    min_index = min(range(len(values)), key=values.__getitem__)
    max_index = max(range(len(values)), key=values.__getitem__)
    return {
        "count": len(events),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first_wall_time": wall_times[0],
        "last_wall_time": wall_times[-1],
        "first_value": values[0],
        "last_value": values[-1],
        "min_value": values[min_index],
        "min_step": steps[min_index],
        "max_value": values[max_index],
        "max_step": steps[max_index],
    }


def export_scalars(
    *,
    input_path: Path,
    out_csv: Path,
    out_summary_json: Path,
    tag_filter: set[str] | None,
) -> dict[str, Any]:
    EventAccumulator = import_event_accumulator()
    accumulator = EventAccumulator(str(input_path.expanduser().resolve()), size_guidance={"scalars": 0})
    accumulator.Reload()

    tags = sorted(accumulator.Tags().get("scalars", []))
    if tag_filter is not None:
        tags = [tag for tag in tags if tag in tag_filter]

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        summary[tag] = scalar_summary(events)
        for event in events:
            rows.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "value": float(event.value),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tag", "step", "wall_time", "value"))
        writer.writeheader()
        writer.writerows(rows)

    out_summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def print_pretty(summary: dict[str, Any]) -> None:
    for tag, item in summary.items():
        last_value = item.get("last_value")
        min_value = item.get("min_value")
        print(
            f"{tag}: count={item.get('count')} "
            f"last_step={item.get('last_step')} last={last_value:.8g} "
            f"min_step={item.get('min_step')} min={min_value:.8g}"
        )


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")

    output_dir = default_output_dir(input_path)
    out_csv = (args.out_csv or output_dir / "tensorboard_scalars.csv").expanduser().resolve()
    out_summary_json = (
        args.out_summary_json or output_dir / "tensorboard_scalar_summary.json"
    ).expanduser().resolve()

    summary = export_scalars(
        input_path=input_path,
        out_csv=out_csv,
        out_summary_json=out_summary_json,
        tag_filter=parse_tag_filter(args.tags),
    )
    print(f"wrote {out_csv}")
    print(f"wrote {out_summary_json}")
    if args.pretty:
        print_pretty(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
