from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
import inspect
import json
import logging
from pathlib import Path
import tarfile
import tempfile
from typing import Any

log = logging.getLogger("worker.nemo_export")

HYDRA_META_KEYS = frozenset({"_target_", "_recursive_", "_convert_", "_partial_"})

SUPPORTED_NEMO_ASR_MODEL_CLASSES = (
    "ASRModel",
    "EncDecCTCModel",
    "EncDecCTCModelBPE",
    "EncDecHybridRNNTCTCModel",
    "EncDecHybridRNNTCTCBPEModel",
    "EncDecRNNTModel",
    "EncDecRNNTBPEModel",
)


class NemoExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExportSummary:
    source: str
    output_path: str
    model_class: str
    load_method: str
    device: str
    onnx_opset: int
    validate: bool
    use_dynamo: bool
    metadata_path: str
    exported_at_utc: str


def normalize_model_class_name(name: str) -> str:
    raw = (name or "ASRModel").strip()
    if not raw:
        return "ASRModel"

    for supported in SUPPORTED_NEMO_ASR_MODEL_CLASSES:
        if supported.lower() == raw.lower():
            return supported

    supported_list = ", ".join(SUPPORTED_NEMO_ASR_MODEL_CLASSES)
    raise ValueError(f"Unsupported NeMo ASR model class `{raw}`. Supported: {supported_list}")


def is_local_nemo_source(source: str) -> bool:
    path = Path((source or "").strip()).expanduser()
    if not path.name:
        return False
    if path.suffix.lower() == ".nemo":
        return True
    return path.is_file()


def select_supported_kwargs(func: Any, **candidate_kwargs: Any) -> dict[str, Any]:
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {key: value for key, value in candidate_kwargs.items() if value is not None}

    selected: dict[str, Any] = {}
    for key, value in candidate_kwargs.items():
        if value is None:
            continue
        if key in parameters:
            selected[key] = value
    return selected


def _import_torch():
    try:
        import torch
    except Exception as exc:
        raise NemoExportError(
            "PyTorch is required for NeMo export. Create a Python 3.11 environment and "
            "install the export dependencies from `worker/requirements-export.txt`."
        ) from exc
    return torch


def _import_onnx():
    try:
        import onnx
    except Exception as exc:
        raise NemoExportError(
            "ONNX validation requires the `onnx` package. Install the export environment from "
            "`worker/requirements-export.txt` in Python 3.11, or rerun with `--no-validate`."
        ) from exc
    return onnx


def _import_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as exc:
        raise NemoExportError(
            "Hugging Face Hub support is required to download remote NeMo archives. "
            "Create a Python 3.11 environment and install the worker context-biasing environment from "
            "`worker/requirements-context-biasing.txt`."
        ) from exc
    return hf_hub_download, list_repo_files


def _resolve_nemo_model_class(name: str):
    try:
        import nemo.collections.asr.models as nemo_asr_models
    except Exception as exc:
        raise NemoExportError(
            "NeMo ASR export dependencies are missing. Create a Python 3.11 environment and "
            "install the export environment from `worker/requirements-export.txt`."
        ) from exc

    normalized = normalize_model_class_name(name)
    model_class = getattr(nemo_asr_models, normalized, None)
    if model_class is None:
        raise NemoExportError(
            f"Installed NeMo package does not expose `{normalized}`. "
            f"Available choices in this repo: {', '.join(SUPPORTED_NEMO_ASR_MODEL_CLASSES)}"
        )
    return normalized, model_class


def _resolve_hydra_target(target: str) -> Any | None:
    raw = (target or "").strip()
    if not raw or "." not in raw:
        return None
    module_name, attr_name = raw.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, attr_name, None)


def _filter_hydra_target_kwargs(value: Any) -> Any:
    if isinstance(value, list):
        return [_filter_hydra_target_kwargs(item) for item in value]
    if not isinstance(value, dict):
        return value

    filtered = {key: _filter_hydra_target_kwargs(item) for key, item in value.items()}
    target_name = filtered.get("_target_")
    if not isinstance(target_name, str):
        return filtered

    target = _resolve_hydra_target(target_name)
    if target is None:
        return filtered

    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return filtered

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return filtered

    allowed = HYDRA_META_KEYS | set(parameters)
    return {key: item for key, item in filtered.items() if key in allowed}


def _normalize_nemo_restore_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _filter_hydra_target_kwargs(deepcopy(config))
    tokenizer_cfg = normalized.get("tokenizer")
    if not isinstance(tokenizer_cfg, dict):
        decoder_cfg = normalized.get("decoder")
        if isinstance(decoder_cfg, dict):
            decoder_cfg.pop("multisoftmax", None)
        return normalized

    tokenizer_type = str(tokenizer_cfg.get("type") or "").strip().lower()
    if tokenizer_type == "multilingual" and isinstance(tokenizer_cfg.get("langs"), dict):
        tokenizer_cfg["type"] = "agg"

    decoder_cfg = normalized.get("decoder")
    if isinstance(decoder_cfg, dict):
        decoder_cfg.pop("multisoftmax", None)
    return normalized


def _write_nemo_restore_override_config(archive_path: str | Path) -> Path | None:
    try:
        import yaml
    except Exception as exc:
        raise NemoExportError(
            "PyYAML is required to normalize NeMo restore configs for context-biasing checkpoints."
        ) from exc

    archive = Path(archive_path).expanduser().resolve()
    try:
        with tarfile.open(archive) as handle:
            try:
                config_member = handle.extractfile("./model_config.yaml")
            except KeyError:
                config_member = handle.extractfile("model_config.yaml")
            if config_member is None:
                return None
            config = yaml.safe_load(config_member.read().decode("utf-8"))
    except (tarfile.TarError, OSError):
        return None

    if not isinstance(config, dict):
        return None

    normalized = _normalize_nemo_restore_config(config)
    if normalized == config:
        return None

    temp_handle = tempfile.NamedTemporaryFile(prefix="nemo_restore_override_", suffix=".yaml", delete=False)
    temp_handle.close()
    override_path = Path(temp_handle.name)
    override_path.write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8")
    return override_path


def _restore_nemo_archive(model_class: Any, source: str | Path, *, device: str):
    override_config_path = _write_nemo_restore_override_config(source)
    restore_kwargs = select_supported_kwargs(
        model_class.restore_from,
        map_location=device,
        override_config_path=str(override_config_path) if override_config_path is not None else None,
    )
    try:
        return model_class.restore_from(str(source), **restore_kwargs)
    finally:
        if override_config_path is not None:
            override_config_path.unlink(missing_ok=True)


def _download_hf_nemo_archive(source: str) -> Path:
    repo_id = (source or "").strip()
    if not repo_id:
        raise NemoExportError("A Hugging Face repo id is required to resolve a remote `.nemo` archive.")

    hf_hub_download, list_repo_files = _import_huggingface_hub()
    repo_files = list_repo_files(repo_id=repo_id)
    nemo_files = sorted(path for path in repo_files if path.lower().endswith(".nemo"))
    if not nemo_files:
        raise NemoExportError(f"Hugging Face repo `{repo_id}` does not contain a `.nemo` archive.")
    if len(nemo_files) > 1:
        choices = ", ".join(nemo_files[:5])
        if len(nemo_files) > 5:
            choices = f"{choices}, ..."
        raise NemoExportError(
            f"Hugging Face repo `{repo_id}` contains multiple `.nemo` archives. "
            f"Disambiguate the source before loading: {choices}"
        )

    filename = nemo_files[0]
    log.info("Downloading NeMo archive from Hugging Face repo=%s file=%s", repo_id, filename)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _load_nemo_model(*, source: str, model_class_name: str, device: str):
    normalized_name, model_class = _resolve_nemo_model_class(model_class_name)
    load_method = "restore_from" if is_local_nemo_source(source) else "from_pretrained"

    if load_method == "restore_from":
        model = _restore_nemo_archive(model_class, source, device=device)
    else:
        load_kwargs = select_supported_kwargs(
            model_class.from_pretrained,
            map_location=device,
            strict=False,
        )
        try:
            model = model_class.from_pretrained(source, **load_kwargs)
        except Exception as exc:
            log.warning(
                "NeMo from_pretrained failed for source=%s model_class=%s error=%s. "
                "Falling back to a downloaded `.nemo` archive from Hugging Face Hub.",
                source,
                normalized_name,
                exc,
            )
            archive_path = _download_hf_nemo_archive(source)
            model = _restore_nemo_archive(model_class, archive_path, device=device)
            load_method = "restore_from_hf_hub"

    return normalized_name, load_method, model


def _prepare_model_for_export(model: Any, device: str) -> None:
    torch = _import_torch()
    target_device = (device or "cpu").strip().lower() or "cpu"
    if target_device.startswith("cuda") and not torch.cuda.is_available():
        raise NemoExportError("CUDA export requested but torch.cuda.is_available() is False")

    if hasattr(model, "to"):
        model.to(target_device)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "freeze"):
        model.freeze()


def _validate_onnx_model(output_path: Path) -> None:
    onnx = _import_onnx()
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)


def export_nemo_asr_model_to_onnx(
    *,
    source: str,
    output_path: str | Path,
    model_class_name: str = "ASRModel",
    device: str = "cpu",
    onnx_opset: int = 17,
    validate: bool = True,
    verbose: bool = False,
    use_dynamo: bool = False,
) -> ExportSummary:
    if not source or not source.strip():
        raise NemoExportError("A NeMo model source is required.")

    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() != ".onnx":
        raise NemoExportError(f"Expected an `.onnx` output path, got `{target}`")
    target.parent.mkdir(parents=True, exist_ok=True)

    model_class, load_method, model = _load_nemo_model(
        source=source.strip(),
        model_class_name=model_class_name,
        device=device,
    )
    _prepare_model_for_export(model, device=device)

    export_kwargs = select_supported_kwargs(
        model.export,
        check_trace=False,
        onnx_opset_version=max(int(onnx_opset), 11),
        verbose=verbose,
        do_constant_folding=True,
        use_dynamo=use_dynamo,
    )

    # Prefer the legacy non-dynamo path when available; it is generally more stable for ASR ONNX export.
    if "use_dynamo" not in export_kwargs and use_dynamo:
        log.warning("Installed NeMo export API does not expose `use_dynamo`; continuing without it")

    model.export(str(target), **export_kwargs)

    if validate:
        _validate_onnx_model(target)

    summary = ExportSummary(
        source=source.strip(),
        output_path=str(target),
        model_class=model_class,
        load_method=load_method,
        device=(device or "cpu").strip().lower() or "cpu",
        onnx_opset=max(int(onnx_opset), 11),
        validate=validate,
        use_dynamo=bool(use_dynamo),
        metadata_path=str(target.with_suffix(f"{target.suffix}.metadata.json")),
        exported_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    metadata_path = Path(summary.metadata_path)
    metadata_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a NeMo ASR checkpoint to ONNX.")
    parser.add_argument("--source", required=True, help="Local `.nemo` path or NeMo pretrained model name.")
    parser.add_argument("--output", required=True, help="Destination `.onnx` file.")
    parser.add_argument(
        "--model-class",
        default="ASRModel",
        help="NeMo ASR model class to use when loading. Default: ASRModel",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device, typically `cpu` or `cuda`. Default: cpu",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ONNX checker validation after export.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose export logging when supported by the installed NeMo version.",
    )
    parser.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Opt into the dynamo-based exporter when the installed NeMo version supports it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    try:
        summary = export_nemo_asr_model_to_onnx(
            source=args.source,
            output_path=args.output,
            model_class_name=args.model_class,
            device=args.device,
            onnx_opset=args.opset,
            validate=not args.no_validate,
            verbose=args.verbose,
            use_dynamo=args.use_dynamo,
        )
    except Exception as exc:
        log.error("NeMo ASR export failed: %s", exc)
        return 1

    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
