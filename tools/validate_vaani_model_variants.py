#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_TOOL = REPO_ROOT / "tools" / "eval_nemo_manifest_wer.py"
SERVING_EVAL_TOOL = REPO_ROOT / "tools" / "eval_serving_model_manifest_wer.py"
EXPORT_TOOL = REPO_ROOT / "tools" / "export_vaani_hindi_to_nemo.py"
DEFAULT_BASE_MODEL = "ai4bharat/indic-conformer-600m-multilingual"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "baseline", "denoise": False, "apm": False, "context_biasing": False},
    {"name": "denoise", "denoise": True, "apm": False, "context_biasing": False},
    {"name": "apm", "denoise": False, "apm": True, "context_biasing": False},
    {"name": "context_biasing", "denoise": False, "apm": False, "context_biasing": True},
    {"name": "denoise_apm", "denoise": True, "apm": True, "context_biasing": False},
    {"name": "denoise_context_biasing", "denoise": True, "apm": False, "context_biasing": True},
    {"name": "apm_context_biasing", "denoise": False, "apm": True, "context_biasing": True},
    {"name": "denoise_apm_context_biasing", "denoise": True, "apm": True, "context_biasing": True},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a direct NeMo model on a Vaani manifest across baseline, denoise, APM, "
            "context-biasing, and combination variants."
        )
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=Path, help="Path to a .nemo or .ckpt model artifact.")
    model_group.add_argument(
        "--model-name",
        default=None,
        help=(
            "Serving HF model name for direct worker-model evaluation. "
            f"Use `{DEFAULT_BASE_MODEL}` for the base model currently used by serving."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Existing Vaani NeMo JSONL manifest. If omitted, the script exports one first.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/vaani_model_validation"))
    parser.add_argument("--dataset-config", default="Hindi", help="Vaani dataset config for export. Default: Hindi")
    parser.add_argument("--split", default="train", help="Vaani split to export when --manifest is omitted. Default: train")
    parser.add_argument("--target-hours", type=float, default=0.25, help="Export size when --manifest is omitted. Default: 0.25")
    parser.add_argument("--hf-token", help="Hugging Face token for gated Vaani access when exporting.")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache dir when exporting.")
    parser.add_argument("--max-scan-rows", type=int, help="Optional source row scan cap when exporting.")
    parser.add_argument("--limit", type=int, help="Optional manifest row limit for each validation variant.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--decoder", choices=("rnnt", "ctc"), default="rnnt")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--reference-normalization", choices=("raw", "vaani"), default="vaani")
    parser.add_argument(
        "--context-biasing-mode",
        choices=("shadow", "active"),
        default="active",
        help="Context-biasing mode for variants that include context biasing. Default: active",
    )
    parser.add_argument("--context-biasing-phrases-dir", type=Path, default=Path("context_biasing/phrases"))
    parser.add_argument("--apm-backend", help="Optional module.path:ClassName backend for --apm variants.")
    parser.add_argument("--top-errors", type=int, default=20)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variant["name"] for variant in VARIANTS),
        help="Optional subset of variants to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def read_dotenv_value(*keys: str) -> str | None:
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return None
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in keys:
            return value.strip().strip("\"'")
    return None


def resolve_hf_token(cli_token: str | None) -> str | None:
    return (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or read_dotenv_value("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    )


def build_export_command(args: argparse.Namespace, export_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(EXPORT_TOOL),
        "--dataset-config",
        args.dataset_config,
        "--split",
        args.split,
        "--out-dir",
        str(export_dir),
        "--target-hours",
        str(args.target_hours),
        "--language",
        args.language,
    ]
    token = resolve_hf_token(args.hf_token)
    if token:
        command.extend(["--hf-token", token])
    if args.cache_dir:
        command.extend(["--cache-dir", str(args.cache_dir)])
    if args.max_scan_rows is not None:
        command.extend(["--max-scan-rows", str(args.max_scan_rows)])
    return command


def build_eval_command(
    args: argparse.Namespace,
    *,
    manifest: Path,
    variant: dict[str, Any],
    variant_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(SERVING_EVAL_TOOL if getattr(args, "model_name", None) else EVAL_TOOL),
    ]
    if getattr(args, "model_name", None):
        command.extend(["--model-name", str(args.model_name)])
    else:
        command.extend(["--model", str(args.model)])

    command.extend([
        "--manifest",
        str(manifest),
        "--decoder",
        args.decoder,
        "--language",
        args.language,
        "--reference-normalization",
        args.reference_normalization,
        "--out-tsv",
        str(variant_dir / "hypotheses.tsv"),
        "--out-jsonl",
        str(variant_dir / "errors.jsonl"),
        "--out-summary-json",
        str(variant_dir / "summary.json"),
        "--processed-audio-dir",
        str(variant_dir / "processed_audio"),
        "--top-errors",
        str(args.top_errors),
    ])
    if not getattr(args, "model_name", None):
        command.extend([
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ])
    else:
        command.extend(["--device", args.device])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if variant["denoise"]:
        command.append("--denoise")
    if variant["apm"]:
        command.append("--apm")
        if args.apm_backend:
            command.extend(["--apm-backend", args.apm_backend])
    if variant["context_biasing"]:
        command.extend(
            [
                "--context-biasing-mode",
                args.context_biasing_mode,
                "--context-biasing-phrases-dir",
                str(args.context_biasing_phrases_dir),
            ]
        )
    return command


def command_for_display(command: list[str]) -> str:
    return " ".join(json.dumps(part) if " " in part else part for part in command)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if args.model is None and args.model_name is None:
        args.model_name = DEFAULT_BASE_MODEL

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.expanduser().resolve() if args.manifest else None
    commands: list[list[str]] = []
    if manifest is None:
        export_dir = out_dir / "vaani_export"
        manifest = export_dir / "manifest.jsonl"
        if not manifest.exists():
            commands.append(build_export_command(args, export_dir))

    selected = set(args.variants or [variant["name"] for variant in VARIANTS])
    variant_commands: list[tuple[str, Path, list[str]]] = []
    for variant in VARIANTS:
        if variant["name"] not in selected:
            continue
        variant_dir = out_dir / "variants" / str(variant["name"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        command = build_eval_command(args, manifest=manifest, variant=variant, variant_dir=variant_dir)
        variant_commands.append((str(variant["name"]), variant_dir, command))
        commands.append(command)

    if args.dry_run:
        for command in commands:
            print(command_for_display(command))
        return 0

    for command in commands:
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)

    aggregate: dict[str, Any] = {
        "model": (
            str(args.model.expanduser().resolve())
            if args.model is not None
            else str(args.model_name or DEFAULT_BASE_MODEL)
        ),
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "context_biasing_mode": args.context_biasing_mode,
        "variants": {},
    }
    for name, variant_dir, _ in variant_commands:
        summary_path = variant_dir / "summary.json"
        if summary_path.exists():
            summary = load_summary(summary_path)
            aggregate["variants"][name] = {
                "utterances": summary.get("utterances"),
                "reference_words": summary.get("reference_words"),
                "wer": summary.get("wer"),
                "wer_percent": summary.get("wer_percent"),
                "summary_json": str(summary_path),
                "errors_jsonl": str(variant_dir / "errors.jsonl"),
                "hypotheses_tsv": str(variant_dir / "hypotheses.tsv"),
            }

    aggregate_path = out_dir / "variant_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
