"""Build-time FAR archiver for itn_service grammars.

Compiles the per-language Pynini grammars into a single FAR archive per
language under ``<out-dir>/<lang>.far``. Runs at build / deploy time so
the request path never has to invoke ``pynini.compile`` — see
``CONTRIBUTING.md`` invariant 3.

Usage::

    python -m itn_service.compile                        # all languages
    python -m itn_service.compile --lang hi              # one language
    python -m itn_service.compile --out-dir build/grammars

Each FAR contains one named entry per grammar surface; the runtime
loader (``runtime/wfst_pipeline.py``) reads them by name.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Iterable

import pynini

# Per-language registry: language code -> (module path -> list of names
# to export from that module). Adding a language means adding an entry
# here; the rest of the pipeline picks it up automatically.
_LANGUAGE_REGISTRY: dict[str, list[tuple[str, list[str]]]] = {
    "hi": [
        ("itn_service.grammars.hi.cardinal", ["CARDINAL", "CARDINAL_CLASSIFIER"]),
        ("itn_service.grammars.hi.decimal", ["DECIMAL", "DECIMAL_CLASSIFIER"]),
        ("itn_service.grammars.hi.money", ["MONEY", "MONEY_CLASSIFIER"]),
        ("itn_service.grammars.hi.percent", ["PERCENT", "PERCENT_CLASSIFIER"]),
        # Date exposes three FAR entries: the always-safe month-word
        # branch, the DMY-only numeric branch, and the union the runtime
        # uses for DMY tenants. The classifier wraps the union.
        (
            "itn_service.grammars.hi.date",
            ["DATE", "DATE_MONTHWORD", "DATE_NUMERIC", "DATE_CLASSIFIER"],
        ),
        ("itn_service.grammars.hi.time", ["TIME", "TIME_CLASSIFIER"]),
    ],
    "mr": [
        ("itn_service.grammars.mr.cardinal", ["CARDINAL", "CARDINAL_CLASSIFIER"]),
        ("itn_service.grammars.mr.decimal", ["DECIMAL", "DECIMAL_CLASSIFIER"]),
        ("itn_service.grammars.mr.money", ["MONEY", "MONEY_CLASSIFIER"]),
        ("itn_service.grammars.mr.percent", ["PERCENT", "PERCENT_CLASSIFIER"]),
        (
            "itn_service.grammars.mr.date",
            ["DATE", "DATE_MONTHWORD", "DATE_NUMERIC", "DATE_CLASSIFIER"],
        ),
        ("itn_service.grammars.mr.time", ["TIME", "TIME_CLASSIFIER"]),
    ],
}


def _resolve_fsts(modules: Iterable[tuple[str, list[str]]]) -> list[tuple[str, pynini.Fst]]:
    """Import every module and pull the named attributes out as FSTs."""
    out: list[tuple[str, pynini.Fst]] = []
    seen: set[str] = set()
    for module_name, attr_names in modules:
        module = importlib.import_module(module_name)
        for attr in attr_names:
            if attr in seen:
                raise ValueError(
                    f"duplicate FAR entry name {attr!r} from {module_name}; "
                    f"each FAR entry must be unique"
                )
            fst = getattr(module, attr)
            if not isinstance(fst, pynini.Fst):
                raise TypeError(
                    f"{module_name}.{attr} is {type(fst).__name__}, expected pynini.Fst"
                )
            out.append((attr, fst))
            seen.add(attr)
    return out


def compile_language(lang: str, out_dir: Path) -> Path:
    """Compile one language's grammars into ``<out_dir>/<lang>.far``.

    Returns the path of the written FAR.
    """
    if lang not in _LANGUAGE_REGISTRY:
        raise ValueError(
            f"unknown language {lang!r}; registered: "
            f"{sorted(_LANGUAGE_REGISTRY)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    far_path = out_dir / f"{lang}.far"
    fsts = _resolve_fsts(_LANGUAGE_REGISTRY[lang])
    with pynini.Far(str(far_path), mode="w") as far:
        for name, fst in sorted(fsts, key=lambda item: item[0]):
            far[name] = fst
    return far_path


def compile_all(out_dir: Path) -> list[Path]:
    """Compile every registered language."""
    return [compile_language(lang, out_dir) for lang in _LANGUAGE_REGISTRY]


def _default_out_dir() -> Path:
    """Pick a default output directory next to this file.

    Mirrors the layout in the implementation blueprint
    (``compiled_grammars/<lang>.far``).
    """
    return Path(__file__).resolve().parent / "compiled_grammars"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang",
        action="append",
        choices=sorted(_LANGUAGE_REGISTRY),
        help="language(s) to compile; default = all",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="directory to write FAR archives into",
    )
    args = parser.parse_args(argv)

    langs = args.lang if args.lang else list(_LANGUAGE_REGISTRY)
    written: list[Path] = []
    for lang in langs:
        far_path = compile_language(lang, args.out_dir)
        written.append(far_path)
        print(f"wrote {far_path}", file=sys.stderr)
    print(f"compiled {len(written)} language(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
