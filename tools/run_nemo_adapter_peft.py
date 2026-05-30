#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import importlib.util
import json
import os
import random
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    if importlib.util.find_spec("cuda.cuda") is not None:
        os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
except ModuleNotFoundError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.asr_text_normalizer import normalize_asr_text


ADAPTER_COMPATIBLE_ENCODERS = {
    "nemo.collections.asr.modules.ConformerEncoder": (
        "nemo.collections.asr.modules.conformer_encoder.ConformerEncoderAdapter"
    ),
    "nemo.collections.asr.modules.conformer_encoder.ConformerEncoder": (
        "nemo.collections.asr.modules.conformer_encoder.ConformerEncoderAdapter"
    ),
}

RNNT_LOSS_NAME_CHOICES = ("default", "warprnnt_numba", "pytorch", "warprnnt")
TRAINING_OBJECTIVE_CHOICES = ("rnnt", "ctc")


ADAPTER_TENSORBOARD_LAYOUT = {
    "01 Adapter PEFT": {
        "Adapter Scope": [
            "Multiline",
            [
                "adapter/trainable_params",
                "adapter/trainable_fraction_percent",
                "adapter/adapter_dim",
            ],
        ],
        "Adapter Grad Norms": [
            "Multiline",
            [
                "adapter_grad_norm/mean",
                "adapter_grad_norm/max",
            ],
        ],
    },
    "02 Model Quality": {
        "Validation WER": ["Multiline", ["val_wer", "val_wer_ctc"]],
        "Training Batch WER": [
            "Multiline",
            ["training_batch_wer", "training_batch_wer_ctc"],
        ],
    },
    "03 Losses": {
        "Loss Comparison": [
            "Multiline",
            ["train_loss", "train_rnnt_loss", "train_ctc_loss"],
        ],
    },
    "04 Optimization": {
        "Learning Rate": ["Multiline", ["learning_rate"]],
        "Batching": ["Multiline", ["adapter/effective_batch_size"]],
        "Progress": ["Multiline", ["global_step", "epoch"]],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only NeMo encoder adapters on normalized Vaani data. Full fine-tuning is not exposed."
    )
    parser.add_argument("--model", type=Path, required=True, help="Base .nemo model.")
    parser.add_argument("--train-manifest", type=Path, help="Normalized Vaani train manifest. Required for training.")
    parser.add_argument("--val-manifest", type=Path, help="Normalized Vaani validation manifest. Required for training.")
    parser.add_argument("--exp-dir", type=Path, help="Experiment root directory. Required for training.")
    parser.add_argument("--name", help="Experiment name. Required for training.")
    parser.add_argument("--adapter-name", default="vaani_adapter")
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-check-interval", type=int, default=200)
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Number of best validation checkpoints to keep. Default: 1.",
    )
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--val-max-duration", type=float, default=10.0)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument(
        "--validation-mode",
        choices=("diagnostic", "final"),
        default="diagnostic",
        help=(
            "Use a fixed dev subset for sweep diagnostics, or the full dev manifest "
            "for the final selected config. Default: diagnostic."
        ),
    )
    parser.add_argument(
        "--diagnostic-val-size",
        type=int,
        default=200,
        help="Number of validation utterances to sample in --validation-mode diagnostic. Default: 200.",
    )
    parser.add_argument(
        "--diagnostic-val-seed",
        type=int,
        default=42,
        help="Seed for the diagnostic validation manifest sample. Default: 42.",
    )
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument(
        "--rnnt-loss-name",
        choices=RNNT_LOSS_NAME_CHOICES,
        default="default",
        help=(
            "RNN-T loss implementation to request in the restored NeMo config. "
            "Use 'pytorch' to bypass the Numba CUDA RNN-T loss kernel. Default: default."
        ),
    )
    parser.add_argument(
        "--training-objective",
        choices=TRAINING_OBJECTIVE_CHOICES,
        default="rnnt",
        help=(
            "Training objective for adapter PEFT. 'ctc' bypasses decoder/joint/RNN-T loss "
            "and trains through the hybrid model's CTC head. Default: rnnt."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--return-language-id", action="store_true")
    parser.add_argument("--save-final-nemo", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume training from a Lightning .ckpt checkpoint, typically checkpoints/last.ckpt.",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        help="Initialize model and adapter weights from a Lightning .ckpt checkpoint without resuming optimizer/scheduler states or starting step.",
    )
    parser.add_argument(
        "--print-grad-norms",
        action="store_true",
        help="Print trainable parameter gradient norms after backward for debugging adapter training.",
    )
    parser.add_argument(
        "--grad-norm-log-every-n-steps",
        type=int,
        default=1,
        help="When --print-grad-norms is enabled, print once every N trainer steps. Default: 1.",
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 before importing torch to make CUDA stack traces synchronous.",
    )
    parser.add_argument(
        "--debug-bad-batch",
        action="store_true",
        help=(
            "Make training order deterministic for bad-sample isolation: disable train shuffle, "
            "force dataloader workers to 0, use train batch size 1, and print likely manifest "
            "line/audio path before each train batch."
        ),
    )
    parser.add_argument(
        "--debug-rnnt-targets",
        action="store_true",
        help=(
            "For multilingual multisoftmax RNNT training, print global/local target ID ranges "
            "after language-local remapping and before RNNT loss."
        ),
    )
    parser.add_argument(
        "--inspect-module-names",
        action="store_true",
        help="Restore the adapter-compatible model, print likely PEFT/LoRA target module names, and exit.",
    )
    parser.add_argument(
        "--inspect-module-name-patterns",
        default="q,k,v,proj,linear,ffn",
        help="Comma-separated lowercase name fragments used by --inspect-module-names.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    return parser.parse_args()


def read_jsonl_manifest(path: Path) -> list[dict[str, Any]]:
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
            raise SystemExit(f"{path}:{lineno}: expected JSON object")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: manifest is empty")
    return rows


def write_jsonl_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def row_language(row: dict[str, Any]) -> str | None:
    for key in ("language", "language_id", "lang", "locale", "language_code"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def validate_manifest_text(path: Path, *, split: str) -> dict[str, Any]:
    rows = read_jsonl_manifest(path)
    empty_rows: list[dict[str, Any]] = []
    unnormalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = str(row.get("text") or "")
        normalized = normalize_asr_text(text, row_language(row))
        if not normalized:
            if len(empty_rows) >= 5:
                continue
            empty_rows.append({"index": index, "id": row.get("id"), "text": text})
        elif normalized != text:
            if len(unnormalized_rows) >= 5:
                continue
            unnormalized_rows.append(
                {
                    "index": index,
                    "id": row.get("id"),
                    "text": text,
                    "normalized_text": normalized,
                }
            )
    if empty_rows:
        raise SystemExit(
            f"{split} manifest has empty text after ASR normalization; first offenders: "
            + json.dumps(empty_rows, ensure_ascii=False)
        )
    if unnormalized_rows:
        raise SystemExit(
            f"{split} manifest text is not normalized with tools.asr_text_normalizer.normalize_asr_text; "
            "run tools/normalize_manifest_text.py before training. First offenders: "
            + json.dumps(unnormalized_rows, ensure_ascii=False)
        )
    return {"path": str(path), "rows": len(rows)}


def sample_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    if sample_size < 0:
        raise SystemExit("--diagnostic-val-size must be non-negative")
    if sample_size == 0 or sample_size >= len(rows):
        return list(rows), list(range(len(rows)))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices], indices


def row_duration_for_filter(row: dict[str, Any]) -> float | None:
    try:
        duration = float(row.get("duration"))
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return duration


def build_debug_manifest_order(
    manifest: Path,
    *,
    min_duration: float,
    max_duration: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{manifest}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{manifest}:{lineno}: expected JSON object")

        duration = row_duration_for_filter(row)
        if duration is not None and duration < float(min_duration):
            continue
        if duration is not None and duration > float(max_duration):
            continue

        rows.append(
            {
                "debug_order_index": len(rows),
                "manifest_lineno": lineno,
                "source_lineno": row.get("source_lineno"),
                "validation_source_lineno": row.get("validation_source_lineno"),
                "id": row.get("id"),
                "audio_filepath": row.get("audio_filepath"),
                "duration": row.get("duration"),
                "text": row.get("text"),
            }
        )
    if not rows:
        raise SystemExit(f"{manifest}: no rows left for debug order after duration filtering")
    return rows


def debug_batch_rows(
    rows: list[dict[str, Any]],
    *,
    batch_idx: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    start = (int(batch_idx) * max(1, int(batch_size))) % len(rows)
    end = min(start + max(1, int(batch_size)), len(rows))
    return rows[start:end]


def compact_debug_row(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or "")
    if len(text) > 120:
        text = text[:117] + "..."
    return {
        "debug_order_index": row.get("debug_order_index"),
        "manifest_lineno": row.get("manifest_lineno"),
        "source_lineno": row.get("source_lineno"),
        "validation_source_lineno": row.get("validation_source_lineno"),
        "id": row.get("id"),
        "audio_filepath": row.get("audio_filepath"),
        "duration": row.get("duration"),
        "text": text,
    }


def resolve_validation_manifest(
    source_manifest: Path,
    *,
    run_dir: Path,
    mode: str,
    diagnostic_val_size: int,
    diagnostic_val_seed: int,
) -> tuple[Path, dict[str, Any]]:
    source_manifest = source_manifest.expanduser().resolve()
    rows = read_jsonl_manifest(source_manifest)
    summary: dict[str, Any] = {
        "mode": mode,
        "source_manifest": str(source_manifest),
        "source_rows": len(rows),
        "diagnostic_val_size": int(diagnostic_val_size),
        "diagnostic_val_seed": int(diagnostic_val_seed),
    }
    if mode == "final":
        summary.update(
            {
                "manifest": str(source_manifest),
                "rows": len(rows),
                "sampled": False,
                "reason": "final mode uses the full validation manifest",
            }
        )
        return source_manifest, summary

    sample_rows, sample_indices = sample_manifest_rows(
        rows,
        sample_size=int(diagnostic_val_size),
        seed=int(diagnostic_val_seed),
    )
    if len(sample_rows) == len(rows):
        summary.update(
            {
                "manifest": str(source_manifest),
                "rows": len(rows),
                "sampled": False,
                "reason": "diagnostic sample size covers the full validation manifest",
            }
        )
        return source_manifest, summary

    sample_manifest = run_dir / f"dev_diagnostic_{len(sample_rows)}utt_seed{int(diagnostic_val_seed)}.jsonl"
    write_jsonl_manifest(sample_manifest, sample_rows)
    encoded_indices = json.dumps(sample_indices, separators=(",", ":")).encode("utf-8")
    summary.update(
        {
            "manifest": str(sample_manifest),
            "rows": len(sample_rows),
            "sampled": True,
            "sample_indices_sha256": hashlib.sha256(encoded_indices).hexdigest(),
            "sample_indices_preview": sample_indices[:20],
        }
    )
    return sample_manifest, summary


def ensure_inputs(args: argparse.Namespace, *, require_manifests: bool = True) -> dict[str, Any]:
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"model file does not exist: {model_path}")
    if not require_manifests:
        return {"model": str(model_path)}
    missing = [
        name
        for name, value in (
            ("--train-manifest", args.train_manifest),
            ("--val-manifest", args.val_manifest),
            ("--exp-dir", args.exp_dir),
            ("--name", args.name),
        )
        if value is None
    ]
    if missing:
        raise SystemExit("training requires " + ", ".join(missing))
    train_manifest = args.train_manifest.expanduser().resolve()
    val_manifest = args.val_manifest.expanduser().resolve()
    if not train_manifest.exists():
        raise SystemExit(f"train manifest does not exist: {train_manifest}")
    if not val_manifest.exists():
        raise SystemExit(f"val manifest does not exist: {val_manifest}")
    resume_from_checkpoint = None
    if args.resume_from_checkpoint is not None:
        resume_from_checkpoint = args.resume_from_checkpoint.expanduser().resolve()
        if not resume_from_checkpoint.exists():
            raise SystemExit(f"resume checkpoint does not exist: {resume_from_checkpoint}")
    init_from_checkpoint = None
    if args.init_from_checkpoint is not None:
        init_from_checkpoint = args.init_from_checkpoint.expanduser().resolve()
        if not init_from_checkpoint.exists():
            raise SystemExit(f"init checkpoint does not exist: {init_from_checkpoint}")
    return {
        "model": str(model_path),
        "train_manifest": validate_manifest_text(train_manifest, split="train"),
        "val_manifest": validate_manifest_text(val_manifest, split="val"),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "init_from_checkpoint": str(init_from_checkpoint) if init_from_checkpoint else None,
    }


def _cfg_get(cfg: Any, *path: str) -> Any:
    current = cfg
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, "get"):
            try:
                current = current.get(key)
            except Exception:
                return None
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return None
    return current


def _to_plain(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=False)
    except Exception:
        pass
    return value


def tokenizer_fingerprint(model_or_cfg: Any) -> dict[str, Any]:
    cfg = getattr(model_or_cfg, "_cfg", getattr(model_or_cfg, "cfg", model_or_cfg))
    payload = {
        "tokenizer": _to_plain(_cfg_get(cfg, "tokenizer")),
        "labels": _to_plain(_cfg_get(cfg, "labels")),
        "decoder_vocab_size": _to_plain(_cfg_get(cfg, "decoder", "vocab_size")),
        "decoder_num_classes": _to_plain(_cfg_get(cfg, "decoder", "num_classes")),
        "decoder_vocabulary": _to_plain(_cfg_get(cfg, "decoder", "vocabulary")),
        "joint_num_classes": _to_plain(_cfg_get(cfg, "joint", "num_classes")),
        "joint_vocabulary": _to_plain(_cfg_get(cfg, "joint", "vocabulary")),
        "ctc_decoder_num_classes": _to_plain(_cfg_get(cfg, "aux_ctc", "decoder", "num_classes")),
        "ctc_decoder_vocabulary": _to_plain(_cfg_get(cfg, "aux_ctc", "decoder", "vocabulary")),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "label_count": len(payload["labels"]) if isinstance(payload["labels"], list) else None,
        "decoder_vocab_size": payload["decoder_vocab_size"],
        "decoder_num_classes": payload["decoder_num_classes"],
        "joint_num_classes": payload["joint_num_classes"],
    }


def assert_tokenizer_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before["sha256"] != after["sha256"]:
        raise SystemExit(
            "Tokenizer/vocabulary fingerprint changed during adapter setup: "
            + json.dumps({"before": before, "after": after}, ensure_ascii=True, sort_keys=True)
        )


def load_nemo_model_config(model_path: Path):
    from omegaconf import OmegaConf

    with tarfile.open(model_path, "r:*") as archive:
        member = next(
            (item for item in archive.getmembers() if item.name.strip("./") == "model_config.yaml"),
            None,
        )
        if member is None:
            raise SystemExit(f"{model_path}: model_config.yaml not found")
        handle = archive.extractfile(member)
        if handle is None:
            raise SystemExit(f"{model_path}: could not read model_config.yaml")
        return OmegaConf.load(handle)


def parse_module_name_patterns(raw: str) -> list[str]:
    patterns = [item.strip().casefold() for item in raw.split(",")]
    return [item for item in patterns if item]


def inspect_matching_module_names(model: Any, patterns: list[str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for name, module in model.named_modules():
        lowered = name.casefold()
        if any(pattern in lowered for pattern in patterns):
            module_type = type(module)
            matches.append(
                {
                    "name": name,
                    "type": f"{module_type.__module__}.{module_type.__qualname__}",
                }
            )
    return matches


def normalize_aggregate_tokenizer_config(cfg: Any) -> bool:
    from omegaconf import DictConfig, open_dict

    tokenizer = _cfg_get(cfg, "tokenizer")
    tokenizer_type = _cfg_get(tokenizer, "type")
    tokenizer_langs = _cfg_get(tokenizer, "langs")
    if tokenizer_type != "multilingual" or not isinstance(tokenizer_langs, (dict, DictConfig)):
        return False
    with open_dict(tokenizer):
        tokenizer.type = "agg"
    return True


def make_encoder_adapter_compatible(cfg: Any) -> bool:
    from omegaconf import open_dict

    target = _cfg_get(cfg, "encoder", "_target_")
    if not isinstance(target, str) or target.endswith("Adapter"):
        return False
    replacement = ADAPTER_COMPATIBLE_ENCODERS.get(target)
    if replacement is None and target.endswith(".ConformerEncoder"):
        replacement = f"{target}Adapter"
    if replacement is None:
        return False
    with open_dict(cfg.encoder):
        cfg.encoder._target_ = replacement
    return True


def configure_rnnt_loss_cfg(cfg: Any, loss_name: str) -> dict[str, Any]:
    from omegaconf import open_dict

    requested = str(loss_name or "default")
    if requested not in RNNT_LOSS_NAME_CHOICES:
        raise SystemExit(f"unsupported --rnnt-loss-name: {requested}")

    loss_cfg = _cfg_get(cfg, "loss")
    original_loss_name = _cfg_get(loss_cfg, "loss_name")
    summary = {
        "requested": requested,
        "original_loss_name": original_loss_name,
        "effective_loss_name": original_loss_name,
        "changed": False,
        "removed_loss_kwargs": [],
    }
    if requested == "default":
        return summary

    with open_dict(cfg):
        if _cfg_get(cfg, "loss") is None:
            cfg.loss = {}
    loss_cfg = _cfg_get(cfg, "loss")
    removed: list[str] = []
    with open_dict(loss_cfg):
        loss_cfg.loss_name = requested
        if requested == "pytorch":
            for key in list(loss_cfg.keys()):
                if str(key).endswith("_kwargs"):
                    removed.append(str(key))
                    del loss_cfg[key]

    summary.update(
        {
            "effective_loss_name": requested,
            "changed": requested != original_loss_name,
            "removed_loss_kwargs": removed,
        }
    )
    return summary


def _language_id_to_key(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _mask_to_bool_list(mask: Any) -> list[bool]:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().tolist()
    return [bool(item) for item in list(mask)]


def build_multisoftmax_label_maps(language_masks: Any) -> dict[str, Any] | None:
    if language_masks is None:
        return None

    if isinstance(language_masks, dict) or hasattr(language_masks, "items"):
        items = list(language_masks.items())
    else:
        items = list(enumerate(language_masks))
    if not items:
        raise SystemExit("multisoftmax language_masks is empty")

    maps: dict[Any, dict[int, int]] = {}
    mask_lengths: set[int] = set()
    local_class_counts: set[int] = set()
    language_summaries: list[dict[str, Any]] = []
    for raw_key, raw_mask in items:
        key = _language_id_to_key(raw_key)
        bool_mask = _mask_to_bool_list(raw_mask)
        true_indices = [index for index, enabled in enumerate(bool_mask) if enabled]
        if not true_indices:
            raise SystemExit(f"multisoftmax language mask {key!r} has no enabled classes")
        label_map = {global_id: local_id for local_id, global_id in enumerate(true_indices)}
        maps[key] = label_map
        mask_lengths.add(len(bool_mask))
        local_class_counts.add(len(true_indices))
        language_summaries.append(
            {
                "language_id": key,
                "global_classes_with_blank": len(bool_mask),
                "local_classes_with_blank": len(true_indices),
                "blank_global_id": true_indices[-1],
                "blank_local_id": len(true_indices) - 1,
            }
        )

    if len(local_class_counts) != 1:
        raise SystemExit(
            "multisoftmax language masks have different local class counts; "
            "the current CTC-only adapter path requires equal per-language vocab sizes: "
            + json.dumps(language_summaries[:20], ensure_ascii=True, sort_keys=True)
        )

    local_classes_with_blank = next(iter(local_class_counts))
    return {
        "enabled": True,
        "maps": maps,
        "language_count": len(maps),
        "global_classes_with_blank": next(iter(mask_lengths)) if len(mask_lengths) == 1 else sorted(mask_lengths),
        "local_classes_with_blank": local_classes_with_blank,
        "local_blank_id": local_classes_with_blank - 1,
        "languages_preview": language_summaries[:20],
    }


def remap_multisoftmax_targets(
    targets: Any,
    target_lengths: Any,
    language_ids: Any,
    label_maps: dict[str, Any] | None,
) -> Any:
    if not label_maps:
        return targets
    if language_ids is None:
        raise RuntimeError("multisoftmax CTC training requires language_ids in the dataloader batch")

    maps = label_maps["maps"]
    remapped = targets.clone()
    batch_size = int(targets.shape[0])
    for batch_index in range(batch_size):
        lang_key = _language_id_to_key(language_ids[batch_index])
        token_map = maps.get(lang_key)
        if token_map is None and isinstance(lang_key, str):
            try:
                token_map = maps.get(int(lang_key))
            except ValueError:
                token_map = None
        if token_map is None:
            raise RuntimeError(f"no multisoftmax label map for language id {lang_key!r}")

        length_value = _language_id_to_key(target_lengths[batch_index])
        target_length = int(length_value)
        for token_index in range(target_length):
            global_token = int(_language_id_to_key(targets[batch_index, token_index]))
            local_token = token_map.get(global_token)
            if local_token is None:
                raise RuntimeError(
                    "target token is outside the language-local multisoftmax vocabulary: "
                    + json.dumps(
                        {
                            "batch_index": batch_index,
                            "language_id": lang_key,
                            "target_position": token_index,
                            "global_token_id": global_token,
                            "target_length": target_length,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
            remapped[batch_index, token_index] = int(local_token)
    return remapped


def _unpack_asr_batch(model: Any, batch: Any) -> tuple[Any, Any, Any, Any, Any]:
    cfg = getattr(model, "cfg", getattr(model, "_cfg", None))
    is_multisoftmax = _cfg_get(cfg, "decoder", "multisoftmax") is not None or _cfg_get(cfg, "joint", "multisoftmax") is not None
    if not is_multisoftmax:
        signal, signal_len, transcript, transcript_len = batch
        return signal, signal_len, transcript, transcript_len, None
    if len(batch) < 6:
        raise RuntimeError("multisoftmax CTC training expected batch with sample_ids and language_ids")
    signal, signal_len, transcript, transcript_len, _sample_ids, language_ids = batch
    return signal, signal_len, transcript, transcript_len, language_ids


def _ctc_wer_update(metric: Any, *, log_probs: Any, targets: Any, target_lengths: Any, encoded_len: Any, language_ids: Any) -> tuple[Any, Any, Any]:
    if language_ids is not None:
        metric.update(
            predictions=log_probs,
            targets=targets,
            targets_lengths=target_lengths,
            predictions_lengths=encoded_len,
            lang_ids=language_ids,
        )
    else:
        metric.update(
            predictions=log_probs,
            targets=targets,
            targets_lengths=target_lengths,
            predictions_lengths=encoded_len,
        )
    wer, wer_num, wer_denom = metric.compute()
    metric.reset()
    return wer, wer_num, wer_denom


def enable_ctc_only_training(model: Any) -> dict[str, Any]:
    from types import MethodType

    import torch
    from nemo.collections.asr.losses.ctc import CTCLoss

    language_masks = getattr(getattr(model, "ctc_decoder", None), "language_masks", None)
    label_maps = build_multisoftmax_label_maps(language_masks)
    if label_maps:
        reduction = _cfg_get(getattr(model, "cfg", None), "aux_ctc", "ctc_reduction") or "mean_batch"
        model.ctc_loss = CTCLoss(
            num_classes=int(label_maps["local_blank_id"]),
            zero_infinity=True,
            reduction=reduction,
        )
        ctc_decoding = getattr(model, "ctc_decoding", None)
        if ctc_decoding is not None and hasattr(ctc_decoding, "blank_id"):
            ctc_decoding.blank_id = int(label_maps["local_blank_id"])
            inner_decoding = getattr(ctc_decoding, "decoding", None)
            if inner_decoding is not None and hasattr(inner_decoding, "blank_id"):
                inner_decoding.blank_id = int(label_maps["local_blank_id"])
    model.ctc_loss_weight = 1.0
    model.cur_decoder = "ctc"

    def ctc_only_training_step(self: Any, batch: Any, batch_nb: int):
        signal, signal_len, transcript, transcript_len, language_ids = _unpack_asr_batch(self, batch)
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        loss_targets = remap_multisoftmax_targets(transcript, transcript_len, language_ids, label_maps)
        log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
        loss_value = self.ctc_loss(
            log_probs=log_probs,
            targets=loss_targets,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )

        trainer = getattr(self, "trainer", getattr(self, "_trainer", None))
        global_step = int(getattr(trainer, "global_step", 0)) if trainer is not None else int(batch_nb)
        log_every = int(getattr(trainer, "log_every_n_steps", 1)) if trainer is not None else 1
        optimizer = getattr(self, "_optimizer", None)
        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
        tensorboard_logs = {
            "learning_rate": lr,
            "global_step": torch.tensor(global_step, dtype=torch.float32, device=loss_value.device),
            "train_ctc_loss": loss_value.detach(),
            "train_loss": loss_value,
        }

        if (global_step + 1) % max(1, log_every) == 0:
            ctc_wer, _ctc_scores, _ctc_words = _ctc_wer_update(
                self.ctc_wer,
                log_probs=log_probs,
                targets=loss_targets,
                target_lengths=transcript_len,
                encoded_len=encoded_len,
                language_ids=language_ids,
            )
            tensorboard_logs["training_batch_wer_ctc"] = ctc_wer
            tensorboard_logs["training_batch_wer"] = ctc_wer

        self.log_dict(tensorboard_logs)
        return {"loss": loss_value}

    def ctc_only_validation_pass(self: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        del batch_idx, dataloader_idx
        signal, signal_len, transcript, transcript_len, language_ids = _unpack_asr_batch(self, batch)
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        loss_targets = remap_multisoftmax_targets(transcript, transcript_len, language_ids, label_maps)
        log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
        ctc_loss = self.ctc_loss(
            log_probs=log_probs,
            targets=loss_targets,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = _ctc_wer_update(
            self.ctc_wer,
            log_probs=log_probs,
            targets=loss_targets,
            target_lengths=transcript_len,
            encoded_len=encoded_len,
            language_ids=language_ids,
        )
        logs = {
            "val_loss": ctc_loss,
            "val_ctc_loss": ctc_loss,
            "val_wer": ctc_wer,
            "val_wer_num": ctc_wer_num,
            "val_wer_denom": ctc_wer_denom,
            "val_wer_ctc": ctc_wer,
            "val_wer_num_ctc": ctc_wer_num,
            "val_wer_denom_ctc": ctc_wer_denom,
        }
        self.log("global_step", torch.tensor(self.trainer.global_step, dtype=torch.float32, device=ctc_loss.device))
        return logs

    def ctc_only_validation_step(self: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(logs)
        else:
            self.validation_step_outputs.append(logs)
        return logs

    def ctc_only_test_step(self: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    model.training_step = MethodType(ctc_only_training_step, model)
    model.validation_pass = MethodType(ctc_only_validation_pass, model)
    model.validation_step = MethodType(ctc_only_validation_step, model)
    model.test_step = MethodType(ctc_only_test_step, model)

    return {
        "name": "ctc",
        "rnnt_bypassed": True,
        "ctc_loss_weight": 1.0,
        "multisoftmax_label_remap": {key: value for key, value in (label_maps or {}).items() if key != "maps"},
    }


def _target_min_max(targets: Any, target_lengths: Any) -> tuple[int | None, int | None]:
    mins: list[int] = []
    maxes: list[int] = []
    batch_size = int(targets.shape[0])
    for batch_index in range(batch_size):
        length = int(_language_id_to_key(target_lengths[batch_index]))
        if length <= 0:
            continue
        if length > int(targets.shape[1]):
            raise RuntimeError(
                "RNNT target length exceeds padded target dimension: "
                + json.dumps(
                    {
                        "batch_index": batch_index,
                        "target_length": length,
                        "target_dim": int(targets.shape[1]),
                    },
                    sort_keys=True,
                )
            )
        segment = targets[batch_index, :length].detach().cpu()
        mins.append(int(segment.min().item()))
        maxes.append(int(segment.max().item()))
    if not mins:
        return None, None
    return min(mins), max(maxes)


def _language_ids_to_list(language_ids: Any) -> list[Any] | None:
    if language_ids is None:
        return None
    values: list[Any] = []
    for item in language_ids:
        values.append(_language_id_to_key(item))
    return values


def _target_lengths_to_list(target_lengths: Any) -> list[int]:
    return [int(_language_id_to_key(item)) for item in target_lengths]


def validate_rnnt_local_targets(
    *,
    targets_before: Any,
    targets_after: Any,
    target_lengths: Any,
    language_ids: Any,
    rnnt_num_classes: int,
    debug: bool,
    phase: str,
    batch_idx: int | None = None,
) -> dict[str, Any]:
    before_min, before_max = _target_min_max(targets_before, target_lengths)
    after_min, after_max = _target_min_max(targets_after, target_lengths)
    payload = {
        "event": "debug_rnnt_targets",
        "phase": phase,
        "batch_idx": batch_idx,
        "language_ids": _language_ids_to_list(language_ids),
        "targets_before_min": before_min,
        "targets_before_max": before_max,
        "targets_after_min": after_min,
        "targets_after_max": after_max,
        "rnnt_num_classes": int(rnnt_num_classes),
        "rnnt_blank_id": int(rnnt_num_classes) - 1,
        "target_lengths": _target_lengths_to_list(target_lengths),
    }
    if debug:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

    if after_min is None or after_max is None:
        raise RuntimeError("RNNT target id out of range after language-local remap: empty target batch")
    if after_min < 0 or after_max >= int(rnnt_num_classes) - 1:
        raise RuntimeError(
            "RNNT target id out of range after language-local remap: "
            + json.dumps(payload, ensure_ascii=True, sort_keys=True)
        )
    return payload


def remap_rnnt_targets_for_loss(
    *,
    transcript: Any,
    transcript_len: Any,
    language_ids: Any,
    label_maps: dict[str, Any] | None,
    rnnt_num_classes: int,
    debug: bool,
    phase: str,
    batch_idx: int | None = None,
) -> Any:
    try:
        remapped = remap_multisoftmax_targets(transcript, transcript_len, language_ids, label_maps)
    except Exception as exc:
        raise RuntimeError(f"RNNT target id out of range after language-local remap: {exc}") from exc
    validate_rnnt_local_targets(
        targets_before=transcript,
        targets_after=remapped,
        target_lengths=transcript_len,
        language_ids=language_ids,
        rnnt_num_classes=int(rnnt_num_classes),
        debug=debug,
        phase=phase,
        batch_idx=batch_idx,
    )
    return remapped


def configure_local_ctc_loss_for_multisoftmax(model: Any, label_maps: dict[str, Any] | None) -> None:
    if not label_maps:
        return
    from nemo.collections.asr.losses.ctc import CTCLoss

    reduction = _cfg_get(getattr(model, "cfg", None), "aux_ctc", "ctc_reduction") or "mean_batch"
    model.ctc_loss = CTCLoss(
        num_classes=int(label_maps["local_blank_id"]),
        zero_infinity=True,
        reduction=reduction,
    )
    ctc_decoding = getattr(model, "ctc_decoding", None)
    if ctc_decoding is not None and hasattr(ctc_decoding, "blank_id"):
        ctc_decoding.blank_id = int(label_maps["local_blank_id"])
        inner_decoding = getattr(ctc_decoding, "decoding", None)
        if inner_decoding is not None and hasattr(inner_decoding, "blank_id"):
            inner_decoding.blank_id = int(label_maps["local_blank_id"])


def enable_rnnt_local_target_training(model: Any, *, debug_targets: bool = False) -> dict[str, Any]:
    from types import MethodType

    import torch
    from nemo.core.classes.mixins import AccessMixin

    language_masks = getattr(getattr(model, "joint", None), "language_masks", None)
    label_maps = build_multisoftmax_label_maps(language_masks)
    if not label_maps:
        return {
            "name": "rnnt",
            "rnnt_bypassed": False,
            "multisoftmax_label_remap": None,
            "debug_rnnt_targets": bool(debug_targets),
        }

    rnnt_num_classes = int(label_maps["local_classes_with_blank"])
    configure_local_ctc_loss_for_multisoftmax(model, label_maps)

    def rnnt_local_training_step(self: Any, batch: Any, batch_nb: int):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len, language_ids = _unpack_asr_batch(self, batch)
        if hasattr(batch, "has_processed_signal") and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        decoder, target_length, _states = self.decoder(targets=transcript, target_length=transcript_len)
        trainer = getattr(self, "_trainer", getattr(self, "trainer", None))
        if trainer is not None:
            log_every_n_steps = int(getattr(trainer, "log_every_n_steps", 1))
            sample_id = int(getattr(trainer, "global_step", batch_nb))
        else:
            log_every_n_steps = 1
            sample_id = int(batch_nb)
        compute_wer = (sample_id + 1) % max(1, log_every_n_steps) == 0

        rnnt_targets = remap_rnnt_targets_for_loss(
            transcript=transcript,
            transcript_len=transcript_len,
            language_ids=language_ids,
            label_maps=label_maps,
            rnnt_num_classes=rnnt_num_classes,
            debug=bool(debug_targets),
            phase="train",
            batch_idx=int(batch_nb),
        )

        if not self.joint.fuse_loss_wer:
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, language_ids=language_ids)
            validate_rnnt_local_targets(
                targets_before=transcript,
                targets_after=rnnt_targets,
                target_lengths=transcript_len,
                language_ids=language_ids,
                rnnt_num_classes=int(joint.shape[-1]),
                debug=bool(debug_targets),
                phase="train_joint",
                batch_idx=int(batch_nb),
            )
            loss_value = self.loss(
                log_probs=joint,
                targets=rnnt_targets,
                input_lengths=encoded_len,
                target_lengths=target_length,
            )
            loss_value = self.add_auxiliary_losses(loss_value)
            tensorboard_logs = {
                "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=rnnt_targets,
                    targets_lengths=transcript_len,
                    lang_ids=language_ids,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({"training_batch_wer": scores.float() / words})
        else:
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=rnnt_targets,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
                language_ids=language_ids,
            )
            loss_value = self.add_auxiliary_losses(loss_value)
            tensorboard_logs = {
                "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
            if compute_wer:
                tensorboard_logs.update({"training_batch_wer": wer})

        ctc_targets = rnnt_targets
        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs,
                targets=ctc_targets,
                input_lengths=encoded_len,
                target_lengths=transcript_len,
            )
            tensorboard_logs["train_rnnt_loss"] = loss_value
            tensorboard_logs["train_ctc_loss"] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=ctc_targets,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                    lang_ids=language_ids,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({"training_batch_wer_ctc": ctc_wer})

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            ctc_targets,
            transcript_len,
            compute_wer=compute_wer,
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs["train_loss"] = loss_value
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        self.log_dict(tensorboard_logs)
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]
        return {"loss": loss_value}

    def rnnt_local_validation_pass(self: Any, batch: Any, batch_idx: int, dataloader_idx: int):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len, language_ids = _unpack_asr_batch(self, batch)
        if hasattr(batch, "has_processed_signal") and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        rnnt_targets = remap_rnnt_targets_for_loss(
            transcript=transcript,
            transcript_len=transcript_len,
            language_ids=language_ids,
            label_maps=label_maps,
            rnnt_num_classes=rnnt_num_classes,
            debug=bool(debug_targets),
            phase="val",
            batch_idx=int(batch_idx),
        )

        tensorboard_logs: dict[str, Any] = {}
        loss_value = None
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, _states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, language_ids=language_ids)
                validate_rnnt_local_targets(
                    targets_before=transcript,
                    targets_after=rnnt_targets,
                    target_lengths=transcript_len,
                    language_ids=language_ids,
                    rnnt_num_classes=int(joint.shape[-1]),
                    debug=bool(debug_targets),
                    phase="val_joint",
                    batch_idx=int(batch_idx),
                )
                loss_value = self.loss(
                    log_probs=joint,
                    targets=rnnt_targets,
                    input_lengths=encoded_len,
                    target_lengths=target_length,
                )
                tensorboard_logs["val_loss"] = loss_value
            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=rnnt_targets,
                targets_lengths=transcript_len,
                lang_ids=language_ids,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()
            tensorboard_logs["val_wer_num"] = wer_num
            tensorboard_logs["val_wer_denom"] = wer_denom
            tensorboard_logs["val_wer"] = wer
        else:
            if self.compute_eval_loss:
                decoded, target_len, _states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=rnnt_targets,
                transcript_lengths=target_len,
                compute_wer=True,
                language_ids=language_ids,
            )
            if loss_value is not None:
                tensorboard_logs["val_loss"] = loss_value
            tensorboard_logs["val_wer_num"] = wer_num
            tensorboard_logs["val_wer_denom"] = wer_denom
            tensorboard_logs["val_wer"] = wer

        ctc_targets = rnnt_targets
        log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs,
                targets=ctc_targets,
                input_lengths=encoded_len,
                target_lengths=transcript_len,
            )
            tensorboard_logs["val_ctc_loss"] = ctc_loss
            tensorboard_logs["val_rnnt_loss"] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs["val_loss"] = loss_value
        self.ctc_wer.update(
            predictions=log_probs,
            targets=ctc_targets,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
            lang_ids=language_ids,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs["val_wer_num_ctc"] = ctc_wer_num
        tensorboard_logs["val_wer_denom_ctc"] = ctc_wer_denom
        tensorboard_logs["val_wer_ctc"] = ctc_wer
        self.log("global_step", torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            ctc_targets,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            tensorboard_logs["val_loss"] = loss_value
        tensorboard_logs.update(additional_logs)
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        return tensorboard_logs

    model.training_step = MethodType(rnnt_local_training_step, model)
    model.validation_pass = MethodType(rnnt_local_validation_pass, model)

    return {
        "name": "rnnt",
        "rnnt_bypassed": False,
        "multisoftmax_label_remap": {key: value for key, value in label_maps.items() if key != "maps"},
        "debug_rnnt_targets": bool(debug_targets),
        "remap_location": "tools/run_nemo_adapter_peft.py:enable_rnnt_local_target_training",
        "language_ids_passed_to_joint": True,
    }


def set_joint_memory_safe(cfg: Any) -> dict[str, Any]:
    from omegaconf import open_dict

    summary = {"preserve_memory": None, "fused_batch_size": None}
    joint = _cfg_get(cfg, "joint")
    if joint is None:
        return summary
    with open_dict(joint):
        joint.preserve_memory = True
        summary["preserve_memory"] = True
        if "fused_batch_size" in joint:
            joint.fused_batch_size = 1
            summary["fused_batch_size"] = 1
    return summary


def disable_metric_prediction_logging(model: Any) -> None:
    for name in ("wer", "ctc_wer"):
        metric = getattr(model, name, None)
        if metric is not None:
            metric.log_prediction = False


def update_dataset_cfg(
    cfg: Any,
    *,
    manifest: Path,
    batch_size: int,
    shuffle: bool,
    max_duration: float,
    min_duration: float,
    num_workers: int,
    sample_rate: int,
    return_language_id: bool,
) -> None:
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.manifest_filepath = str(manifest)
        cfg.sample_rate = int(sample_rate)
        cfg.batch_size = int(batch_size)
        cfg.shuffle = bool(shuffle)
        cfg.num_workers = int(num_workers)
        cfg.pin_memory = False
        cfg.max_duration = float(max_duration)
        cfg.min_duration = float(min_duration)
        if "is_concat" in cfg:
            cfg.is_concat = False
        if "concat_sampling_technique" in cfg:
            cfg.concat_sampling_technique = "temperature"
        if "concat_sampling_temperature" in cfg:
            cfg.concat_sampling_temperature = 1.0
        if "concat_sampling_scale" in cfg:
            cfg.concat_sampling_scale = 1
        if "concat_sampling_probabilities" in cfg:
            cfg.concat_sampling_probabilities = None
        if "tarred_audio_filepaths" in cfg:
            cfg.tarred_audio_filepaths = None
        if "is_tarred" in cfg:
            cfg.is_tarred = False
        if "bucketing_batch_size" in cfg:
            cfg.bucketing_batch_size = None
        if return_language_id:
            cfg.return_language_id = True


def infer_encoder_dim(model: Any, cfg: Any) -> int:
    for value in (
        _cfg_get(cfg, "encoder", "d_model"),
        _cfg_get(cfg, "model_defaults", "enc_hidden"),
        getattr(getattr(model, "encoder", None), "d_model", None),
        getattr(getattr(model, "encoder", None), "_feat_out", None),
    ):
        if isinstance(value, int) and value > 0:
            return value
    raise SystemExit("Could not infer encoder adapter input dimension from model config")


def build_linear_adapter_config(in_features: int, adapter_dim: int) -> Any:
    try:
        from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig

        return LinearAdapterConfig(in_features=int(in_features), dim=int(adapter_dim))
    except Exception:
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                "in_features": int(in_features),
                "dim": int(adapter_dim),
                "activation": "swish",
                "norm_position": "pre",
                "dropout": 0.0,
            }
        )


def is_adapter_added(model: Any) -> bool:
    for target in (model, getattr(model, "encoder", None)):
        method = getattr(target, "is_adapter_available", None)
        if method is None:
            continue
        try:
            if bool(method()):
                return True
        except Exception:
            continue
    return False


def add_encoder_adapter(model: Any, *, adapter_name: str, adapter_dim: int) -> dict[str, Any]:
    cfg = getattr(model, "_cfg", getattr(model, "cfg", None))
    adapter_cfg = build_linear_adapter_config(infer_encoder_dim(model, cfg), adapter_dim)
    module_names = list(getattr(model, "adapter_module_names", []) or [])
    candidate_names = [f"encoder:{adapter_name}", adapter_name] if "encoder" in module_names else [adapter_name]
    errors: list[str] = []
    added_name: str | None = None

    add_adapter = getattr(model, "add_adapter", None)
    if add_adapter is not None:
        for candidate in candidate_names:
            try:
                add_adapter(candidate, cfg=adapter_cfg)
                added_name = candidate
                break
            except Exception as exc:
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    if added_name is None and hasattr(getattr(model, "encoder", None), "add_adapter"):
        try:
            model.encoder.add_adapter(adapter_name, cfg=adapter_cfg)
            added_name = f"encoder:{adapter_name}"
        except Exception as exc:
            errors.append(f"encoder.{adapter_name}: {type(exc).__name__}: {exc}")

    if added_name is None or not is_adapter_added(model):
        raise SystemExit("adapter was not added; errors: " + "; ".join(errors))
    return {"requested_name": adapter_name, "added_name": added_name, "adapter_dim": int(adapter_dim)}


def freeze_all_parameters(model: Any) -> None:
    method = getattr(model, "freeze", None)
    if method is not None:
        method()
    for parameter in model.parameters():
        parameter.requires_grad = False


def enable_only_adapter(model: Any, *, adapter_name: str, added_name: str) -> None:
    freeze_all_parameters(model)
    for target in (model, getattr(model, "encoder", None)):
        set_enabled = getattr(target, "set_enabled_adapters", None)
        if set_enabled is None:
            continue
        try:
            set_enabled(enabled=False)
        except TypeError:
            pass
        for name in (added_name, adapter_name):
            try:
                set_enabled(name=name, enabled=True)
            except Exception:
                continue
    for target in (model, getattr(model, "encoder", None)):
        unfreeze = getattr(target, "unfreeze_enabled_adapters", None)
        if unfreeze is None:
            continue
        try:
            unfreeze()
        except Exception:
            continue
    for name, parameter in model.named_parameters():
        if is_adapter_parameter_name(name, adapter_name):
            parameter.requires_grad = True


def is_adapter_parameter_name(name: str, adapter_name: str) -> bool:
    lowered = name.casefold()
    return "adapter" in lowered or adapter_name.casefold() in lowered


def summarize_trainable_parameters(model: Any, *, adapter_name: str) -> dict[str, Any]:
    total_params = 0
    trainable_params = 0
    trainable_names: list[str] = []
    non_adapter_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_params += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable_params += count
            trainable_names.append(name)
            if not is_adapter_parameter_name(name, adapter_name):
                non_adapter_trainable.append(name)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_fraction": (trainable_params / total_params) if total_params else 0.0,
        "trainable_names": trainable_names,
        "non_adapter_trainable": non_adapter_trainable,
    }


def validate_trainable_parameters(summary: dict[str, Any]) -> None:
    if int(summary["trainable_params"]) <= 0:
        raise SystemExit("no adapter parameters are trainable")
    if summary["non_adapter_trainable"]:
        raise SystemExit(
            "non-adapter parameters are trainable: " + ", ".join(summary["non_adapter_trainable"][:20])
        )
    if float(summary["trainable_fraction"]) >= 0.05 or int(summary["trainable_params"]) >= 50_000_000:
        raise SystemExit(
            "trainable parameter count is suspiciously close to full fine-tuning: "
            + json.dumps(
                {
                    "trainable_params": summary["trainable_params"],
                    "total_params": summary["total_params"],
                    "trainable_fraction": summary["trainable_fraction"],
                },
                sort_keys=True,
            )
        )


def trainable_gradient_norms(model: Any) -> list[dict[str, Any]]:
    norms: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        try:
            value = float(grad.detach().float().norm().item())
        except AttributeError:
            value = float(grad.norm().item())
        norms.append({"name": name, "grad_norm": value})
    return norms


def tensorboard_tag_component(value: str) -> str:
    tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return tag.strip("._-") or "unnamed"


def log_adapter_gradient_norms_to_tensorboard(logger: Any, norms: list[dict[str, Any]], step: int) -> None:
    if not norms:
        return
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return
    values = [float(item["grad_norm"]) for item in norms]
    experiment.add_scalar("adapter_grad_norm/mean", sum(values) / len(values), step)
    experiment.add_scalar("adapter_grad_norm/max", max(values), step)
    for item in norms:
        parameter_tag = tensorboard_tag_component(str(item["name"]))
        experiment.add_scalar(
            f"adapter_grad_norm/parameters/{parameter_tag}",
            float(item["grad_norm"]),
            step,
        )


def log_adapter_tensorboard_metadata(logger: Any, run_config: dict[str, Any]) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    add_custom_scalars = getattr(experiment, "add_custom_scalars", None)
    if add_custom_scalars is not None:
        add_custom_scalars(ADAPTER_TENSORBOARD_LAYOUT)

    parameters = run_config["parameters"]
    experiment.add_scalar("adapter/trainable_params", float(parameters["trainable_params"]), 0)
    experiment.add_scalar(
        "adapter/trainable_fraction_percent",
        float(parameters["trainable_fraction"]) * 100.0,
        0,
    )
    experiment.add_scalar("adapter/adapter_dim", float(run_config["adapter"]["adapter_dim"]), 0)
    experiment.add_scalar(
        "adapter/effective_batch_size",
        float(run_config["batch_size"] * run_config["accumulate_grad_batches"]),
        0,
    )

    add_text = getattr(experiment, "add_text", None)
    if add_text is not None:
        add_text(
            "adapter_peft/run_config",
            "```json\n" + json.dumps(run_config, ensure_ascii=True, indent=2, sort_keys=True) + "\n```",
            0,
        )

    flush = getattr(experiment, "flush", None)
    if flush is not None:
        flush()


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def configure_adapter_optimizer_cfg(
    optim_cfg: Any,
    *,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    scheduler_cfg: dict[str, Any] = {
        "name": "WarmupAnnealing",
        "warmup_steps": int(warmup_steps),
        "warmup_ratio": None,
        "min_lr": 0.0,
        "last_epoch": -1,
    }
    if int(max_steps) > 0:
        scheduler_cfg["max_steps"] = int(max_steps)

    _cfg_set(optim_cfg, "name", "adamw")
    _cfg_set(optim_cfg, "lr", float(lr))
    _cfg_set(optim_cfg, "weight_decay", float(weight_decay))
    _cfg_set(optim_cfg, "sched", scheduler_cfg)
    return {
        "name": "adamw",
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "sched": scheduler_cfg,
        "expected_peak_lr": float(lr),
    }


def save_adapters_if_supported(model: Any, path: Path, *, adapter_name: str, added_name: str) -> dict[str, Any]:
    save_adapters = getattr(model, "save_adapters", None)
    if save_adapters is None:
        return {"saved": False, "reason": "model does not expose save_adapters"}
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for name in (adapter_name, added_name, None):
        try:
            if name is None:
                save_adapters(str(path))
            else:
                save_adapters(str(path), name=name)
            return {"saved": True, "path": str(path), "name": name}
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return {"saved": False, "path": str(path), "errors": errors}


def main() -> int:
    args = parse_args()
    if args.resume_from_checkpoint is not None and args.init_from_checkpoint is not None:
        raise ValueError("Cannot pass both --resume-from-checkpoint and --init-from-checkpoint.")
    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    input_summary = ensure_inputs(args, require_manifests=not args.inspect_module_names)
    model_path = Path(input_summary["model"])

    try:
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf, open_dict
    except ModuleNotFoundError as exc:
        raise SystemExit("NeMo training dependencies are missing. Activate the nemo_asr/export environment.") from exc

    if args.inspect_module_names:
        base_cfg = load_nemo_model_config(model_path)
        tokenizer_type_changed = normalize_aggregate_tokenizer_config(base_cfg)
        encoder_target_changed = make_encoder_adapter_compatible(base_cfg)
        rnnt_loss_summary = configure_rnnt_loss_cfg(base_cfg, args.rnnt_loss_name)
        set_joint_memory_safe(base_cfg)
        with tempfile.TemporaryDirectory(prefix="nemo_adapter_peft_inspect_") as tmpdir:
            override_path = Path(tmpdir) / "model_config_adapter.yaml"
            OmegaConf.save(config=base_cfg, f=str(override_path))
            model = ASRModel.restore_from(
                restore_path=str(model_path),
                map_location="cpu",
                override_config_path=str(override_path),
            )
        patterns = parse_module_name_patterns(args.inspect_module_name_patterns)
        matches = inspect_matching_module_names(model, patterns)
        for item in matches:
            print(f"{item['name']} {item['type']}")
        print(
            json.dumps(
                {
                    "matched_modules": len(matches),
                    "patterns": patterns,
                    "encoder_target_changed": encoder_target_changed,
                    "rnnt_loss": rnnt_loss_summary,
                    "tokenizer_type_changed": tokenizer_type_changed,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0

    assert args.train_manifest is not None
    assert args.val_manifest is not None
    assert args.exp_dir is not None
    assert args.name is not None
    train_manifest = args.train_manifest.expanduser().resolve()
    val_manifest = args.val_manifest.expanduser().resolve()
    effective_batch_size = 1 if args.debug_bad_batch else int(args.batch_size)
    effective_num_workers = 0 if args.debug_bad_batch else int(args.num_workers)
    train_shuffle = not bool(args.debug_bad_batch)
    debug_order_rows: list[dict[str, Any]] = []
    debug_order_path: Path | None = None
    if args.debug_bad_batch:
        debug_order_rows = build_debug_manifest_order(
            train_manifest,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )

    try:
        import pytorch_lightning as pl
        import torch
        from nemo.collections.asr.metrics.wer import WER
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger
    except ModuleNotFoundError as exc:
        raise SystemExit("NeMo/Lightning training dependencies are missing. Activate the nemo_asr/export environment.") from exc

    exp_dir = args.exp_dir.expanduser().resolve()
    run_version = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger(save_dir=str(exp_dir), name=args.name, version=run_version)
    run_dir = Path(str(logger.log_dir)).resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_bad_batch:
        debug_order_path = run_dir / "debug_bad_batch_order.jsonl"
        write_jsonl_manifest(debug_order_path, debug_order_rows)
    validation_manifest, validation_summary = resolve_validation_manifest(
        val_manifest,
        run_dir=run_dir,
        mode=args.validation_mode,
        diagnostic_val_size=args.diagnostic_val_size,
        diagnostic_val_seed=args.diagnostic_val_seed,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{args.name}" + "--{val_wer:.4f}-step={step}",
        monitor="val_wer",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
    )
    callbacks: list[Any] = [checkpoint_callback]
    if args.print_grad_norms:
        every_n_steps = max(1, int(args.grad_norm_log_every_n_steps))

        class TrainableGradientNormCallback(pl.Callback):
            def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
                step = int(getattr(trainer, "global_step", 0))
                if step % every_n_steps != 0:
                    return
                norms = trainable_gradient_norms(pl_module)
                log_adapter_gradient_norms_to_tensorboard(trainer.logger, norms, step)
                for item in norms:
                    print(
                        f"grad_norm step={step} {item['name']} {item['grad_norm']}",
                        flush=True,
                    )

        callbacks.append(TrainableGradientNormCallback())

    if args.debug_bad_batch:

        class DebugBadBatchCallback(pl.Callback):
            def __init__(self, rows: list[dict[str, Any]], batch_size: int) -> None:
                self.rows = rows
                self.batch_size = max(1, int(batch_size))
                self.last_payload: dict[str, Any] | None = None

            def on_train_batch_start(
                self,
                trainer: Any,
                pl_module: Any,
                batch: Any,
                batch_idx: int,
            ) -> None:
                del pl_module, batch
                rows = [
                    compact_debug_row(row)
                    for row in debug_batch_rows(
                        self.rows,
                        batch_idx=int(batch_idx),
                        batch_size=self.batch_size,
                    )
                ]
                self.last_payload = {
                    "event": "debug_bad_batch",
                    "epoch": int(getattr(trainer, "current_epoch", 0)),
                    "global_step": int(getattr(trainer, "global_step", 0)),
                    "batch_idx": int(batch_idx),
                    "rows": rows,
                }
                print(json.dumps(self.last_payload, ensure_ascii=False, sort_keys=True), flush=True)

            def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
                del trainer, pl_module
                if self.last_payload is None:
                    return
                payload = dict(self.last_payload)
                payload["event"] = "debug_bad_batch_last_seen_before_exception"
                payload["exception"] = f"{type(exception).__name__}: {exception}"
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

        callbacks.append(DebugBadBatchCallback(debug_order_rows, effective_batch_size))

    resume_checkpoint = (
        args.resume_from_checkpoint.expanduser().resolve()
        if args.resume_from_checkpoint is not None
        else None
    )
    init_checkpoint = (
        args.init_from_checkpoint.expanduser().resolve()
        if args.init_from_checkpoint is not None
        else None
    )
    trainer_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "strategy": "auto",
        "logger": logger,
        "callbacks": callbacks,
        "precision": args.precision,
        "max_steps": args.max_steps,
        "max_epochs": args.max_epochs,
        "val_check_interval": args.val_check_interval,
        "limit_val_batches": 1.0,
        "log_every_n_steps": args.log_every_n_steps,
        "num_sanity_val_steps": 0,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "enable_progress_bar": True,
        "default_root_dir": str(run_dir),
    }
    fit_kwargs: dict[str, Any] = {}
    if resume_checkpoint is not None:
        trainer_init_params = inspect.signature(pl.Trainer.__init__).parameters
        if "resume_from_checkpoint" in trainer_init_params:
            trainer_kwargs["resume_from_checkpoint"] = str(resume_checkpoint)
        else:
            fit_kwargs["ckpt_path"] = str(resume_checkpoint)

    trainer = pl.Trainer(**trainer_kwargs)

    base_cfg = load_nemo_model_config(model_path)
    tokenizer_type_changed = normalize_aggregate_tokenizer_config(base_cfg)
    encoder_target_changed = make_encoder_adapter_compatible(base_cfg)
    rnnt_loss_summary = configure_rnnt_loss_cfg(base_cfg, args.rnnt_loss_name)
    set_joint_memory_safe(base_cfg)
    with tempfile.TemporaryDirectory(prefix="nemo_adapter_peft_") as tmpdir:
        override_path = Path(tmpdir) / "model_config_adapter.yaml"
        OmegaConf.save(config=base_cfg, f=str(override_path))
        model = ASRModel.restore_from(
            restore_path=str(model_path),
            map_location="cpu",
            override_config_path=str(override_path),
        )

    if hasattr(model, "set_trainer"):
        model.set_trainer(trainer)

    cfg = model._cfg
    with open_dict(cfg):
        cfg.sample_rate = int(args.sample_rate)
        cfg.log_prediction = False
        if "skip_nan_grad" in cfg:
            cfg.skip_nan_grad = False
    disable_metric_prediction_logging(model)
    joint_memory_summary = set_joint_memory_safe(cfg)
    if hasattr(model, "setup_optimization_flags"):
        model.setup_optimization_flags()

    tokenizer_before = tokenizer_fingerprint(model)
    cfg_is_multisoftmax = _cfg_get(cfg, "decoder", "multisoftmax") is not None
    effective_return_language_id = bool(args.return_language_id or (args.training_objective in {"ctc", "rnnt"} and cfg_is_multisoftmax))
    update_dataset_cfg(
        cfg.train_ds,
        manifest=train_manifest,
        batch_size=effective_batch_size,
        shuffle=train_shuffle,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        num_workers=effective_num_workers,
        sample_rate=args.sample_rate,
        return_language_id=effective_return_language_id,
    )
    update_dataset_cfg(
        cfg.validation_ds,
        manifest=validation_manifest,
        batch_size=args.val_batch_size,
        shuffle=False,
        max_duration=args.val_max_duration,
        min_duration=args.min_duration,
        num_workers=effective_num_workers,
        sample_rate=args.sample_rate,
        return_language_id=effective_return_language_id,
    )

    optimizer_summary: dict[str, Any] | None = None
    if hasattr(cfg, "optim") and cfg.optim is not None:
        with open_dict(cfg.optim):
            optimizer_summary = configure_adapter_optimizer_cfg(
                cfg.optim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
            )

    if hasattr(model, "decoding"):
        model.cer = WER(decoding=model.decoding, batch_dim_index=0, use_cer=True, log_prediction=False)
    if hasattr(model, "ctc_decoding"):
        model.ctc_cer = WER(decoding=model.ctc_decoding, use_cer=True, log_prediction=False)

    model.setup_training_data(cfg.train_ds)
    model.setup_validation_data(cfg.validation_ds)
    disable_metric_prediction_logging(model)

    adapter_summary = add_encoder_adapter(
        model,
        adapter_name=args.adapter_name,
        adapter_dim=args.adapter_dim,
    )
    enable_only_adapter(model, adapter_name=args.adapter_name, added_name=adapter_summary["added_name"])
    parameter_summary = summarize_trainable_parameters(model, adapter_name=args.adapter_name)
    validate_trainable_parameters(parameter_summary)
    tokenizer_after = tokenizer_fingerprint(model)
    assert_tokenizer_unchanged(tokenizer_before, tokenizer_after)

    if init_checkpoint is not None:
        print(f"Loading checkpoint weights from {init_checkpoint} in strict=True mode...", flush=True)
        checkpoint = torch.load(str(init_checkpoint), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint weights loaded successfully.", flush=True)

    if args.training_objective == "ctc":
        training_objective_summary = enable_ctc_only_training(model)
    else:
        training_objective_summary = enable_rnnt_local_target_training(
            model,
            debug_targets=bool(args.debug_rnnt_targets),
        )

    run_config = {
        "model": str(model_path),
        "train_manifest": str(train_manifest),
        "val_manifest": str(validation_manifest),
        "full_val_manifest": str(val_manifest),
        "validation": validation_summary,
        "input_summary": input_summary,
        "adapter": adapter_summary,
        "encoder_target_changed": encoder_target_changed,
        "parameters": {
            key: value
            for key, value in parameter_summary.items()
            if key != "trainable_names"
        },
        "trainable_names": parameter_summary["trainable_names"][:200],
        "tokenizer_fingerprint": tokenizer_before,
        "joint_memory": joint_memory_summary,
        "rnnt_loss": rnnt_loss_summary,
        "training_objective": training_objective_summary,
        "batch_size": effective_batch_size,
        "requested_batch_size": args.batch_size,
        "train_shuffle": train_shuffle,
        "val_batch_size": args.val_batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "optimizer": optimizer_summary,
        "max_steps": args.max_steps,
        "max_epochs": args.max_epochs,
        "val_check_interval": args.val_check_interval,
        "save_top_k": args.save_top_k,
        "validation_mode": args.validation_mode,
        "diagnostic_val_size": args.diagnostic_val_size,
        "diagnostic_val_seed": args.diagnostic_val_seed,
        "max_duration": args.max_duration,
        "val_max_duration": args.val_max_duration,
        "min_duration": args.min_duration,
        "precision": args.precision,
        "num_workers": effective_num_workers,
        "requested_num_workers": args.num_workers,
        "return_language_id": effective_return_language_id,
        "requested_return_language_id": args.return_language_id,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "init_from_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "print_grad_norms": args.print_grad_norms,
        "grad_norm_log_every_n_steps": args.grad_norm_log_every_n_steps,
        "cuda_launch_blocking": args.cuda_launch_blocking,
        "debug_bad_batch": {
            "enabled": bool(args.debug_bad_batch),
            "order_manifest": str(debug_order_path) if debug_order_path is not None else None,
            "order_rows": len(debug_order_rows),
            "note": "Rows are likely batch order after local duration filtering; NeMo internals may still skip rows.",
        },
        "debug_rnnt_targets": bool(args.debug_rnnt_targets),
        "cuda_env": {
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "NUMBA_CUDA_USE_NVIDIA_BINDING": os.environ.get("NUMBA_CUDA_USE_NVIDIA_BINDING"),
        },
        "tensorboard": {
            "log_dir": str(run_dir),
            "custom_scalars": ADAPTER_TENSORBOARD_LAYOUT,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "adapter_run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "resolved_model_cfg.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    print(json.dumps({"adapter_training": run_config}, ensure_ascii=True, indent=2, sort_keys=True))
    log_adapter_tensorboard_metadata(logger, run_config)

    restored = (resume_checkpoint is not None)
    init_mode = "restored" if restored else "freshly initialized"
    opt_lr = args.lr
    sched_max_steps = args.max_steps
    trainer_max_steps = args.max_steps

    if restored:
        try:
            ckpt = torch.load(resume_checkpoint, map_location="cpu")
            opt_states = ckpt.get("optimizer_states", [])
            if opt_states and isinstance(opt_states, list):
                param_groups = opt_states[0].get("param_groups", [])
                if param_groups and isinstance(param_groups, list):
                    opt_lr = param_groups[0].get("lr", args.lr)
            sched_states = ckpt.get("lr_schedulers", [])
            if sched_states and isinstance(sched_states, list):
                sched_max_steps = sched_states[0].get("max_steps", args.max_steps)
        except Exception as e:
            print(f"Warning: Could not read optimizer/scheduler states from resume checkpoint: {e}", flush=True)

    print("=========================================", flush=True)
    print("STARTUP LOGGING:", flush=True)
    print(f"- optimizer lr: {opt_lr}", flush=True)
    print(f"- scheduler max_steps: {sched_max_steps}", flush=True)
    print(f"- trainer max_steps: {trainer_max_steps}", flush=True)
    print(f"- whether optimizer/scheduler were restored or freshly initialized: {init_mode}", flush=True)
    print("=========================================", flush=True)

    trainer.fit(model, **fit_kwargs)

    adapter_path = ckpt_dir / f"{args.name}_{args.adapter_name}.pt"
    adapter_save_summary = save_adapters_if_supported(
        model,
        adapter_path,
        adapter_name=args.adapter_name,
        added_name=adapter_summary["added_name"],
    )
    final_summary = {
        "run_dir": str(run_dir),
        "best_model_path": checkpoint_callback.best_model_path,
        "best_model_score": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "last_model_path": checkpoint_callback.last_model_path,
        "adapter_save": adapter_save_summary,
        "trainable_params": parameter_summary["trainable_params"],
        "total_params": parameter_summary["total_params"],
        "validation": validation_summary,
    }
    if args.save_final_nemo:
        final_nemo = ckpt_dir / f"{args.name}_final.nemo"
        model.save_to(str(final_nemo))
        final_summary["final_nemo"] = str(final_nemo)

    (run_dir / "adapter_fit_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_summary, ensure_ascii=True, indent=2, sort_keys=True))
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
