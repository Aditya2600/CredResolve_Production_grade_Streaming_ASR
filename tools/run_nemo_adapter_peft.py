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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--return-language-id", action="store_true")
    parser.add_argument("--save-final-nemo", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume training from a Lightning .ckpt checkpoint, typically checkpoints/last.ckpt.",
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
    return {
        "model": str(model_path),
        "train_manifest": validate_manifest_text(train_manifest, split="train"),
        "val_manifest": validate_manifest_text(val_manifest, split="val"),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
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
    input_summary = ensure_inputs(args, require_manifests=not args.inspect_module_names)
    model_path = Path(input_summary["model"])

    try:
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf, open_dict
    except ModuleNotFoundError as exc:
        raise SystemExit("NeMo training dependencies are missing. Activate the nemo_asr/export environment.") from exc

    if args.inspect_module_names:
        base_cfg = load_nemo_model_config(model_path)
        encoder_target_changed = make_encoder_adapter_compatible(base_cfg)
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
        save_top_k=1,
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

    resume_checkpoint = (
        args.resume_from_checkpoint.expanduser().resolve()
        if args.resume_from_checkpoint is not None
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
    encoder_target_changed = make_encoder_adapter_compatible(base_cfg)
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
    update_dataset_cfg(
        cfg.train_ds,
        manifest=train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        return_language_id=args.return_language_id,
    )
    update_dataset_cfg(
        cfg.validation_ds,
        manifest=validation_manifest,
        batch_size=args.val_batch_size,
        shuffle=False,
        max_duration=args.val_max_duration,
        min_duration=args.min_duration,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        return_language_id=args.return_language_id,
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
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "optimizer": optimizer_summary,
        "max_steps": args.max_steps,
        "max_epochs": args.max_epochs,
        "val_check_interval": args.val_check_interval,
        "validation_mode": args.validation_mode,
        "diagnostic_val_size": args.diagnostic_val_size,
        "diagnostic_val_seed": args.diagnostic_val_seed,
        "max_duration": args.max_duration,
        "val_max_duration": args.val_max_duration,
        "min_duration": args.min_duration,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "return_language_id": args.return_language_id,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "print_grad_norms": args.print_grad_norms,
        "grad_norm_log_every_n_steps": args.grad_norm_log_every_n_steps,
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
