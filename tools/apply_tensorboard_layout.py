#!/usr/bin/env python3
"""Write a readable TensorBoard custom-scalars layout for NeMo ASR runs.

This is intended for post-hoc use on an existing training log directory.
It reads the available scalar tags from the run's event files and emits a
TensorBoard custom-scalars summary into the same log directory so the Scalars
tab opens with a more useful grouping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


DEFAULT_LAYOUT = {
    "01 Model Quality": {
        "Validation WER": ["Multiline", ["val_wer", "val_wer_ctc"]],
        "Training Batch WER": [
            "Multiline",
            ["training_batch_wer", "training_batch_wer_ctc"],
        ],
    },
    "02 Losses": {
        "Loss Comparison": [
            "Multiline",
            ["train_loss", "train_rnnt_loss", "train_ctc_loss"],
        ],
    },
    "03 Optimization": {
        "Learning Rate": ["Multiline", ["learning_rate"]],
        "Progress": ["Multiline", ["global_step", "epoch"]],
    },
    "04 Speed": {
        "Train Timing": [
            "Multiline",
            ["train_step_timing in s", "train_backward_timing in s"],
        ],
        "Validation Timing": ["Multiline", ["validation_step_timing in s"]],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a custom TensorBoard Scalars layout to an existing NeMo "
            "training run directory."
        )
    )
    parser.add_argument(
        "logdir",
        help="Run directory containing one or more events.out.tfevents files.",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Print detected scalar tags before writing the layout.",
    )
    return parser.parse_args()


def load_scalar_tags(logdir: Path) -> list[str]:
    accumulator = event_accumulator.EventAccumulator(str(logdir))
    accumulator.Reload()
    return sorted(accumulator.Tags().get("scalars", []))


def filter_layout(tags: set[str]) -> dict[str, dict[str, list[object]]]:
    filtered: dict[str, dict[str, list[object]]] = {}
    for category, charts in DEFAULT_LAYOUT.items():
        category_layout: dict[str, list[object]] = {}
        for title, chart in charts.items():
            chart_type, chart_tags = chart
            present_tags = [tag for tag in chart_tags if tag in tags]
            if present_tags:
                category_layout[title] = [chart_type, present_tags]
        if category_layout:
            filtered[category] = category_layout
    return filtered


def main() -> int:
    args = parse_args()
    logdir = Path(args.logdir).expanduser().resolve()
    if not logdir.exists():
        print(f"logdir does not exist: {logdir}", file=sys.stderr)
        return 1

    tags = load_scalar_tags(logdir)
    if args.list_tags:
        print("Detected scalar tags:")
        for tag in tags:
            print(f"  - {tag}")

    layout = filter_layout(set(tags))
    if not layout:
        print(
            "No known scalar tags were found. Use --list-tags to inspect the run.",
            file=sys.stderr,
        )
        return 2

    writer = SummaryWriter(log_dir=str(logdir))
    writer.add_custom_scalars(layout)
    writer.flush()
    writer.close()

    print(f"Wrote TensorBoard custom layout into: {logdir}")
    print("Open TensorBoard Scalars and use the Custom Scalars groups:")
    for category, charts in layout.items():
        print(f"  {category}")
        for title, chart in charts.items():
            print(f"    - {title}: {', '.join(chart[1])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
