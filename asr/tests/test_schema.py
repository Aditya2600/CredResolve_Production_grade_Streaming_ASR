from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr.data.schema_validator import SchemaValidationError, load_schema, validate_record


SCHEMA_PATH = Path("asr/data/schema/annotation_schema.json")
TRAIN_MANIFEST = Path("asr/data/manifests/train.jsonl")


def _first_train_record() -> dict:
    line = TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


def test_schema_accepts_valid_record() -> None:
    schema = load_schema(SCHEMA_PATH)
    rec = _first_train_record()
    validate_record(rec, schema)


def test_schema_rejects_missing_required_field() -> None:
    schema = load_schema(SCHEMA_PATH)
    rec = _first_train_record()
    rec.pop("id")
    with pytest.raises(SchemaValidationError):
        validate_record(rec, schema)


def test_schema_rejects_mismatched_token_lengths() -> None:
    schema = load_schema(SCHEMA_PATH)
    rec = _first_train_record()
    rec["token_lang"] = rec["token_lang"][:-1]
    with pytest.raises(SchemaValidationError):
        validate_record(rec, schema)
