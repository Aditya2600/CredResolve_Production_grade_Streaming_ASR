from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _write_fake_nemo(path: Path, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    config_path = tmp_path / "model_config.yaml"
    weights_path = tmp_path / "model_weights.ckpt"
    config_path.write_text(
        """
tokenizer:
  type: multilingual
  langs:
    hi: hi_tokenizer.model
    ta: ta_tokenizer.model
train_ds:
  manifest_filepath: stale_train.jsonl
validation_ds:
  manifest_filepath: stale_valid.jsonl
test_ds:
  manifest_filepath: stale_test.jsonl
decoder:
  vocab_size: 512
joint:
  num_classes: 512
  jointnet:
    joint_hidden: 4
""".lstrip(),
        encoding="utf-8",
    )
    torch.save(
        {
            "joint.joint_net.2.hi.weight": torch.zeros(257, 4),
            "joint.joint_net.2.hi.bias": torch.zeros(257),
            "joint.joint_net.2.ta.weight": torch.zeros(257, 4),
            "joint.joint_net.2.ta.bias": torch.zeros(257),
        },
        weights_path,
    )
    with tarfile.open(path, "w:gz") as archive:
        archive.add(config_path, arcname="./model_config.yaml")
        archive.add(weights_path, arcname="./model_weights.ckpt")


def test_restore_compat_preserves_multisoftmax_joint_for_per_language_heads(tmp_path: Path, capsys):
    from tools.eval_nemo_manifest_wer import build_nemo_compat_override_config

    nemo_path = tmp_path / "model.nemo"
    _write_fake_nemo(nemo_path, tmp_path)

    override_path = build_nemo_compat_override_config(nemo_path, tmp_path)

    assert override_path is not None
    cfg = OmegaConf.load(override_path)
    assert cfg.tokenizer.type == "agg"
    assert cfg.train_ds is None
    assert cfg.validation_ds is None
    assert cfg.test_ds is None
    assert cfg.decoder.multisoftmax is True
    assert cfg.joint.multilingual is True
    assert list(cfg.joint.language_keys) == ["hi", "ta"]
    assert "restore_compat: patched RNNT joint to multilingual multisoftmax" in capsys.readouterr().err


def _real_nemo_path(env_name: str, default: str) -> Path:
    return Path(os.environ.get(env_name, default)).expanduser()


def _restore_smoke(path: Path) -> None:
    if os.environ.get("RUN_NEMO_RESTORE_SMOKE") != "1":
        pytest.skip("set RUN_NEMO_RESTORE_SMOKE=1 to run real NeMo restore smoke tests")
    if not path.exists():
        pytest.skip(f"missing local smoke-test model: {path}")
    pytest.importorskip("nemo")
    torch = pytest.importorskip("torch")

    from tools.eval_nemo_manifest_wer import load_model

    model, actual_device = load_model(
        path,
        requested_device=torch.device("cpu"),
        allow_cpu_fallback=False,
        restore_compat="auto",
    )

    assert model is not None
    assert actual_device.type == "cpu"


@pytest.mark.restore_smoke
def test_base_indicconformer_restore_smoke():
    _restore_smoke(
        _real_nemo_path(
            "BASE_INDICCONFORMER_NEMO",
            "/home/ubuntu/models/indicconformer/IndicConformer.nemo",
        )
    )


@pytest.mark.restore_smoke
def test_peft_final_nemo_restore_smoke():
    _restore_smoke(
        _real_nemo_path(
            "PEFT_FINAL_NEMO",
            "artifacts/ft_runs/vaani_adapter_peft/indicconformer_vaani_adapter_dim32_rnnt_debug20/2026-05-28_17-05-32/checkpoints/indicconformer_vaani_adapter_dim32_rnnt_debug20_final.nemo",
        )
    )
