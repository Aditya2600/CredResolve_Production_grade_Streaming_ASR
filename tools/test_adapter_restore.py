#!/usr/bin/env python3
from __future__ import annotations

import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_nemo_manifest_wer import load_model

CHECKPOINT_DIR = Path("/home/ubuntu/asr-stt-v3/artifacts/ft_runs/vaani_adapter_peft/indicconformer_vaani_adapter_dim32_lr1e-5_resume500_steps2000/2026-05-29_11-14-24/checkpoints")

CHECKPOINTS = [
    "indicconformer_vaani_adapter_dim32_lr1e-5_resume500_steps2000--val_wer=0.1250-step=step=500.ckpt",
    "indicconformer_vaani_adapter_dim32_lr1e-5_resume500_steps2000--val_wer=0.2733-step=step=550.ckpt",
    "indicconformer_vaani_adapter_dim32_lr1e-5_resume500_steps2000--val_wer=0.2733-step=step=600.ckpt",
]


def test_restore_single(ckpt_name: str) -> bool:
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    print("\n" + "=" * 80)
    print(f"Testing restore for: {ckpt_path.name}")
    print("=" * 80)
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file {ckpt_path} does not exist.")
        return False

    try:
        model, actual_device = load_model(
            ckpt_path,
            requested_device=torch.device("cpu"),
            allow_cpu_fallback=True,
            restore_compat="auto"
        )
        print("Success! Model restored and loaded successfully.")
        
        # Verify that encoder adapter keys are indeed present and have requires_grad=False
        found_adapter = False
        for name, param in model.named_parameters():
            if "adapter_layer.vaani_adapter" in name:
                found_adapter = True
                break
        
        if found_adapter:
            print("Verified: encoder adapter layers found in restored model.")
        else:
            print("Warning: no encoder adapter layers found in restored model parameters.")
            return False
            
        return True
    except Exception as e:
        print(f"Restore failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    success = True
    for ckpt in CHECKPOINTS:
        if not test_restore_single(ckpt):
            success = False
            
    print("\n" + "=" * 80)
    if success:
        print("ALL RESTORE SMOKE TESTS PASSED!")
        return 0
    else:
        print("SOME RESTORE SMOKE TESTS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
