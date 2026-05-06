#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


LANGUAGE_KEY_CANDIDATES = ("language", "language_id", "lang", "locale", "language_code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check tokenizer coverage for a NeMo ASR manifest without changing it.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the .nemo model.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest to check.")
    parser.add_argument("--max-examples", type=int, default=20, help="Maximum bad examples to print. Default: 20")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def row_language(row: dict[str, Any]) -> str | None:
    for key in LANGUAGE_KEY_CANDIDATES:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def add_candidate(candidates: list[Any], seen: set[int], candidate: Any) -> None:
    if candidate is None:
        return
    identity = id(candidate)
    if identity in seen:
        return
    seen.add(identity)
    candidates.append(candidate)


def tokenizer_candidates(model: Any, language: str | None = None) -> list[Any]:
    candidates: list[Any] = []
    seen: set[int] = set()
    owners = [
        model,
        getattr(model, "decoding", None),
        getattr(model, "ctc_decoding", None),
        getattr(model, "decoder", None),
        getattr(model, "joint", None),
    ]
    for owner in owners:
        add_candidate(candidates, seen, getattr(owner, "tokenizer", None))
    for candidate in list(candidates):
        tokenizers = getattr(candidate, "tokenizers", None)
        if isinstance(tokenizers, dict):
            if language and language in tokenizers:
                add_candidate(candidates, seen, tokenizers[language])
            for value in tokenizers.values():
                add_candidate(candidates, seen, value)
    return candidates


def _try_call(method: Callable[..., Any], text: str, language: str | None) -> tuple[bool, Any | None, str | None]:
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [((text,), {})]
    if language:
        attempts = [
            ((text,), {"lang": language}),
            ((text,), {"lang_id": language}),
            ((text,), {"language": language}),
            ((text,), {"language_id": language}),
            ((text, language), {}),
        ] + attempts

    last_error: str | None = None
    for args, kwargs in attempts:
        try:
            result = method(*args, **kwargs)
        except TypeError as exc:
            last_error = str(exc)
            continue
        except Exception as exc:  # pragma: no cover - depends on tokenizer internals
            return False, None, f"{type(exc).__name__}: {exc}"
        return True, result, None
    return False, None, last_error


def tokenize_text(model: Any, text: str, language: str | None = None) -> tuple[bool, str | None, str | None]:
    last_method: str | None = None
    last_error: str | None = None
    for tokenizer in tokenizer_candidates(model, language):
        for method_name in ("text_to_ids", "text_to_tokens"):
            method = getattr(tokenizer, method_name, None)
            if method is None:
                continue
            ok, result, error = _try_call(method, text, language)
            if ok and result is not None:
                return True, f"{type(tokenizer).__name__}.{method_name}", None
            if error:
                last_method = f"{type(tokenizer).__name__}.{method_name}"
                last_error = error

    vocab = extract_decoder_vocabulary(model)
    if vocab:
        vocab_chars = set("".join(vocab))
        missing = sorted({char for char in text if not char.isspace() and char not in vocab_chars})
        if missing:
            return False, "decoder_vocabulary_chars", "missing characters: " + "".join(missing[:20])
        return True, "decoder_vocabulary_chars", None
    if last_error is not None:
        return False, last_method, last_error
    return False, None, "no tokenizer or decoder vocabulary exposed"


def extract_decoder_vocabulary(model: Any) -> list[str]:
    for owner in (model, getattr(model, "decoder", None), getattr(model, "joint", None)):
        for attr in ("vocabulary", "labels"):
            value = getattr(owner, attr, None)
            plain = _to_plain(value)
            if isinstance(plain, (list, tuple)):
                return [str(item) for item in plain]
    cfg = getattr(model, "_cfg", getattr(model, "cfg", None))
    for key in ("labels", "vocabulary"):
        value = _to_plain(_cfg_get(cfg, key))
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
    return []


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


def check_rows(model: Any, rows: list[dict[str, Any]], *, max_examples: int) -> dict[str, Any]:
    common_chars: Counter[str] = Counter()
    failed = 0
    bad_examples: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()

    for index, row in enumerate(rows):
        text = str(row.get("text") or "")
        common_chars.update(char for char in text if not char.isspace())
        language = row_language(row)
        ok, method, error = tokenize_text(model, text, language)
        if method:
            methods[method] += 1
        if not ok:
            failed += 1
            if len(bad_examples) < max_examples:
                bad_examples.append(
                    {
                        "index": index,
                        "id": row.get("id"),
                        "language": language,
                        "text": text,
                        "method": method,
                        "error": error,
                    }
                )

    return {
        "total_rows": len(rows),
        "failed_rows": failed,
        "methods": dict(methods),
        "common_characters": [
            {"character": char, "count": count}
            for char, count in common_chars.most_common(100)
        ],
        "bad_examples": bad_examples,
    }


def load_nemo_model(path: Path) -> Any:
    try:
        from nemo.collections.asr.models import ASRModel
    except ModuleNotFoundError as exc:  # pragma: no cover - host env may not include NeMo
        raise SystemExit("NeMo is not installed in this Python environment. Activate the nemo_asr/export environment.") from exc
    return ASRModel.restore_from(restore_path=str(path), map_location="cpu")


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"model file does not exist: {model_path}")
    if not manifest_path.exists():
        raise SystemExit(f"manifest file does not exist: {manifest_path}")

    rows = read_jsonl(manifest_path)
    model = load_nemo_model(model_path)
    summary = check_rows(model, rows, max_examples=int(args.max_examples))
    summary["model"] = str(model_path)
    summary["manifest"] = str(manifest_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
