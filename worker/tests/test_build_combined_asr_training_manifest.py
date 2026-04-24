from __future__ import annotations

import json
from pathlib import Path

from tools.build_combined_asr_training_manifest import (
    build_combined_manifests,
    compute_domain_repeat_factor,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row(name: str, duration: float = 1.0) -> dict:
    return {
        "audio_filepath": f"/tmp/{name}.wav",
        "duration": duration,
        "text": f"text {name}",
        "lang": "hi",
        "source_id": name,
    }


def test_compute_domain_repeat_factor_respects_target_and_cap():
    factor, target_met = compute_domain_repeat_factor(
        domain_rows=2,
        vaani_rows=10,
        strategy="domain_weighted",
        min_fraction=0.25,
        repeat_cap=3,
    )

    assert factor == 2
    assert target_met is True

    capped_factor, capped_target_met = compute_domain_repeat_factor(
        domain_rows=1,
        vaani_rows=20,
        strategy="domain_weighted",
        min_fraction=0.25,
        repeat_cap=3,
    )

    assert capped_factor == 3
    assert capped_target_met is False


def test_build_combined_manifests_repeats_domain_and_keeps_dev_domain_only(tmp_path: Path):
    domain_train = tmp_path / "domain_train.jsonl"
    domain_dev = tmp_path / "domain_dev.jsonl"
    vaani_train = tmp_path / "vaani_train.jsonl"
    output_dir = tmp_path / "combined"

    _write_jsonl(domain_train, [_row("domain-a"), _row("domain-b")])
    _write_jsonl(domain_dev, [_row("domain-dev")])
    _write_jsonl(vaani_train, [_row(f"vaani-{index}") for index in range(10)])

    summary = build_combined_manifests(
        domain_train_manifest=domain_train,
        domain_dev_manifest=domain_dev,
        vaani_train_manifest=vaani_train,
        output_dir=output_dir,
        strategy="domain_weighted",
        domain_min_row_fraction=0.25,
        domain_repeat_cap=3,
        seed=42,
    )

    train_rows = _read_jsonl(output_dir / "combined_train.jsonl")
    dev_rows = _read_jsonl(output_dir / "combined_dev.jsonl")

    assert summary["domain_repeat_factor"] == 2
    assert summary["domain_fraction_target_met"] is True
    assert summary["actual_domain_row_fraction"] == round(4 / 14, 6)
    assert len(train_rows) == 14
    assert sum(row["mix_source"] == "domain" for row in train_rows) == 4
    assert sum(row["mix_source"] == "vaani" for row in train_rows) == 10
    assert [row["mix_source"] for row in dev_rows] == ["domain_dev"]
    assert summary["validation_policy"] == "domain_only"


def test_build_combined_manifests_is_deterministic(tmp_path: Path):
    domain_train = tmp_path / "domain_train.jsonl"
    domain_dev = tmp_path / "domain_dev.jsonl"
    vaani_train = tmp_path / "vaani_train.jsonl"

    _write_jsonl(domain_train, [_row("domain-a"), _row("domain-b")])
    _write_jsonl(domain_dev, [_row("domain-dev")])
    _write_jsonl(vaani_train, [_row(f"vaani-{index}") for index in range(5)])

    build_combined_manifests(
        domain_train_manifest=domain_train,
        domain_dev_manifest=domain_dev,
        vaani_train_manifest=vaani_train,
        output_dir=tmp_path / "combined-a",
        seed=123,
    )
    build_combined_manifests(
        domain_train_manifest=domain_train,
        domain_dev_manifest=domain_dev,
        vaani_train_manifest=vaani_train,
        output_dir=tmp_path / "combined-b",
        seed=123,
    )

    assert _read_jsonl(tmp_path / "combined-a" / "combined_train.jsonl") == _read_jsonl(
        tmp_path / "combined-b" / "combined_train.jsonl"
    )
