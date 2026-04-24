from __future__ import annotations

import sys

from tools.run_stable_nemo_finetune import freeze_lower_encoder_layers, parse_args


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _Block:
    def __init__(self) -> None:
        self.parameter = _Parameter()

    def parameters(self):
        return [self.parameter]


class _Encoder:
    def __init__(self, count: int) -> None:
        self.layers = [_Block() for _ in range(count)]


class _Model:
    def __init__(self, count: int) -> None:
        self.encoder = _Encoder(count)


def test_parse_args_uses_vaani_pilot_training_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stable_nemo_finetune.py",
            "--model",
            "/tmp/model.nemo",
            "--train-manifest",
            "/tmp/train.jsonl",
            "--val-manifest",
            "/tmp/dev.jsonl",
            "--exp-dir",
            "/tmp/exp",
        ],
    )

    args = parse_args()

    assert args.batch_size == 2
    assert args.accumulate_grad_batches == 4
    assert args.lr == 5e-6
    assert args.warmup_steps == 500
    assert args.max_steps == 8000
    assert args.max_epochs == -1
    assert args.val_check_interval == 500
    assert args.gradient_clip_val == 1.0
    assert args.use_duration_bucketing is True
    assert args.max_duration is None
    assert args.val_max_duration is None
    assert args.freeze_encoder_fraction == 0.5


def test_freeze_lower_encoder_layers_freezes_lower_half():
    model = _Model(count=10)

    summary = freeze_lower_encoder_layers(model, 0.5)

    assert summary["status"] == "enabled"
    assert summary["layer_attr"] == "layers"
    assert summary["total_layers"] == 10
    assert summary["frozen_layers"] == 5
    assert [block.parameter.requires_grad for block in model.encoder.layers] == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_freeze_lower_encoder_layers_is_safe_when_unsupported():
    class ModelWithoutEncoder:
        pass

    summary = freeze_lower_encoder_layers(ModelWithoutEncoder(), 0.5)

    assert summary["status"] == "unsupported"
    assert summary["total_layers"] == 0
