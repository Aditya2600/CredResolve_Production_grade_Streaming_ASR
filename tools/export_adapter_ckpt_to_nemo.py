#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_nemo_manifest_wer import load_nemo_model_config, resolve_checkpoint_restore_source, suppress_embedded_dataset_setup_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PEFT adapter .ckpt checkpoint together with a base model into a self-contained .nemo file."
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the PEFT adapter .ckpt checkpoint.")
    parser.add_argument("--out-nemo", type=Path, required=True, help="Destination path for the exported self-contained .nemo model.")
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Optional base .nemo model path. If omitted, discovered from checkpoint folder metadata or falls back to default.",
    )
    parser.add_argument(
        "--adapter-name",
        default="vaani_adapter",
        help="Adapter name. Default: vaani_adapter",
    )
    parser.add_argument(
        "--adapter-dim",
        type=int,
        default=32,
        help="Adapter dimension. Default: 32",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = args.ckpt.expanduser().resolve()
    out_nemo_path = args.out_nemo.expanduser().resolve()

    if not ckpt_path.exists():
        print(f"Error: checkpoint file does not exist: {ckpt_path}", file=sys.stderr)
        return 1

    # 1. Resolve base model path and adapter config from metadata or arguments
    restore_source, _ = resolve_checkpoint_restore_source(ckpt_path)
    if args.base_model is not None:
        restore_source = args.base_model.expanduser().resolve()
    elif restore_source is None:
        # Fallback to the standard path
        default_base = Path("/home/ubuntu/models/indicconformer/IndicConformer.nemo")
        if default_base.exists():
            restore_source = default_base
        else:
            print(
                f"Error: Could not determine base model path for checkpoint {ckpt_path}. "
                f"Please specify --base-model explicitly.",
                file=sys.stderr
            )
            return 1

    # Read adapter config if available in run folder
    checkpoint_dir = ckpt_path.parent
    run_dir = checkpoint_dir.parent
    run_config_path = run_dir / "adapter_run_config.json"
    meta_adapter_name = args.adapter_name
    meta_adapter_dim = args.adapter_dim
    if run_config_path.exists():
        try:
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
            adapter_meta = payload.get("adapter", {})
            name = adapter_meta.get("requested_name") or adapter_meta.get("added_name")
            if name:
                if name.startswith("encoder:"):
                    name = name[len("encoder:"):]
                meta_adapter_name = name
            dim = adapter_meta.get("adapter_dim")
            if dim:
                meta_adapter_dim = int(dim)
        except Exception as e:
            print(f"Warning: could not parse adapter config metadata: {e}", file=sys.stderr)

    print(f"Loading base model: {restore_source}", file=sys.stderr)
    print(f"Adapter settings: name={meta_adapter_name}, dim={meta_adapter_dim}", file=sys.stderr)

    # 2. Load dependencies
    try:
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        print("Error: NeMo dependencies are missing.", file=sys.stderr)
        return 1

    # Import helper functions from run_nemo_adapter_peft
    from tools.run_nemo_adapter_peft import (
        normalize_aggregate_tokenizer_config,
        make_encoder_adapter_compatible,
        add_encoder_adapter,
        enable_only_adapter,
    )

    # 3. Load the base model config and make encoder adapter compatible
    try:
        base_cfg = load_nemo_model_config(restore_source)
    except Exception as e:
        print(f"Error: Failed to read model config from base model {restore_source}: {e}", file=sys.stderr)
        return 1

    tokenizer_type_changed = False
    try:
        tokenizer_type_changed = normalize_aggregate_tokenizer_config(base_cfg)
    except Exception:
        pass

    encoder_target_changed = make_encoder_adapter_compatible(base_cfg)
    print(f"Base model config patched: tokenizer_changed={tokenizer_type_changed}, encoder_changed={encoder_target_changed}", file=sys.stderr)

    # Suppress dataset setups that might fail or emit warnings
    suppress_embedded_dataset_setup_warnings(base_cfg)

    # Restore base model with override config
    with tempfile.TemporaryDirectory(prefix="nemo_adapter_export_") as tmpdir:
        override_path = Path(tmpdir) / "model_config_adapter.yaml"
        OmegaConf.save(config=base_cfg, f=str(override_path))
        model = ASRModel.restore_from(
            restore_path=str(restore_source),
            map_location="cpu",
            override_config_path=str(override_path),
        )

    # 4. Add and enable the adapter modules in the model instance
    adapter_info = add_encoder_adapter(model, adapter_name=meta_adapter_name, adapter_dim=meta_adapter_dim)
    print(f"Added encoder adapter: {adapter_info}", file=sys.stderr)

    # Enable adapter
    enable_only_adapter(model, adapter_name=meta_adapter_name, added_name=adapter_info["added_name"])

    # 5. Load checkpoint weights into the model
    print(f"Loading checkpoint weights from {ckpt_path} in strict=True mode...", file=sys.stderr)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        print(f"Error: Checkpoint does not contain a state_dict: {ckpt_path}", file=sys.stderr)
        return 1

    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint weights loaded successfully.", file=sys.stderr)

    # 6. Save the exported self-contained .nemo model
    out_nemo_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving self-contained model to {out_nemo_path}...", file=sys.stderr)
    model.save_to(str(out_nemo_path))
    print("Export completed successfully!", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
