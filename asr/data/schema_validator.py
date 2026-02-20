from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    pass


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "array": list,
    "object": dict,
    "boolean": bool,
}


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SchemaValidationError("Schema must be a JSON object")
    return data


def _validate_type(name: str, value: Any, spec: dict[str, Any]) -> None:
    expected_type = spec.get("type")
    if expected_type not in _TYPE_MAP:
        return
    py_t = _TYPE_MAP[expected_type]
    if not isinstance(value, py_t):
        raise SchemaValidationError(f"{name} expected type {expected_type}, got {type(value).__name__}")


def _validate_string(name: str, value: str, spec: dict[str, Any]) -> None:
    min_len = spec.get("minLength")
    if isinstance(min_len, int) and len(value) < min_len:
        raise SchemaValidationError(f"{name} length < {min_len}")


def _validate_number(name: str, value: float | int, spec: dict[str, Any]) -> None:
    minimum = spec.get("minimum")
    if isinstance(minimum, (int, float)) and float(value) < float(minimum):
        raise SchemaValidationError(f"{name} must be >= {minimum}")


def _validate_enum(name: str, value: Any, spec: dict[str, Any]) -> None:
    enum = spec.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise SchemaValidationError(f"{name} must be one of {enum}")


def _validate_array(name: str, value: list[Any], spec: dict[str, Any]) -> None:
    min_items = spec.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        raise SchemaValidationError(f"{name} size < {min_items}")
    item_spec = spec.get("items")
    if not isinstance(item_spec, dict):
        return
    for idx, item in enumerate(value):
        _validate_type(f"{name}[{idx}]", item, item_spec)
        if isinstance(item, str):
            _validate_string(f"{name}[{idx}]", item, item_spec)
        if isinstance(item, (int, float)):
            _validate_number(f"{name}[{idx}]", item, item_spec)
        _validate_enum(f"{name}[{idx}]", item, item_spec)


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise SchemaValidationError("Record must be JSON object")
    required = schema.get("required", [])
    for key in required:
        if key not in record:
            raise SchemaValidationError(f"Missing required field: {key}")

    props = schema.get("properties", {})
    for name, spec in props.items():
        if name not in record:
            continue
        value = record[name]
        if isinstance(spec, dict):
            _validate_type(name, value, spec)
            _validate_enum(name, value, spec)
            if isinstance(value, str):
                _validate_string(name, value, spec)
            elif isinstance(value, (int, float)):
                _validate_number(name, value, spec)
            elif isinstance(value, list):
                _validate_array(name, value, spec)

    # Common alignment checks for token-level labels.
    tokens = record.get("tokens")
    token_lang = record.get("token_lang")
    token_script = record.get("token_script")
    if isinstance(tokens, list) and isinstance(token_lang, list) and len(tokens) != len(token_lang):
        raise SchemaValidationError("tokens and token_lang must have equal length")
    if isinstance(tokens, list) and isinstance(token_script, list) and len(tokens) != len(token_script):
        raise SchemaValidationError("tokens and token_script must have equal length")


def validate_manifest(manifest_path: str | Path, schema_path: str | Path) -> list[dict[str, Any]]:
    schema = load_schema(schema_path)
    records: list[dict[str, Any]] = []

    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"Invalid JSON at line {line_no}: {exc}") from exc
            validate_record(rec, schema)
            records.append(rec)

    return records
