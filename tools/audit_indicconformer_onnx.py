"""Audit the IndicConformer ONNX bundle.

Prints, for each .onnx asset in the snapshot directory:
  - file size
  - producer / opset / IR version
  - input names, dtypes, shapes (with dynamic dims)
  - output names, dtypes, shapes
Also reports whether trtexec is on PATH and which onnxruntime EPs are available.

Usage:
  python tools/audit_indicconformer_onnx.py [--snapshot DIR]

If --snapshot is omitted, the script searches under
worker/hub/models--ai4bharat--indic-conformer-600m-multilingual/snapshots/*.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _format_dim(d) -> str:
    if d.HasField("dim_value"):
        return str(d.dim_value)
    if d.HasField("dim_param") and d.dim_param:
        return d.dim_param
    return "?"


def _format_tensor(t) -> dict:
    import onnx
    from onnx import TensorProto

    elem_type = TensorProto.DataType.Name(t.type.tensor_type.elem_type)
    shape = [_format_dim(d) for d in t.type.tensor_type.shape.dim]
    return {"name": t.name, "dtype": elem_type, "shape": shape}


def _resolve_snapshot(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / "worker/hub/models--ai4bharat--indic-conformer-600m-multilingual/snapshots"
    if not base.exists():
        raise SystemExit(f"snapshot base not found: {base}")
    candidates = sorted(p for p in base.iterdir() if p.is_dir())
    if not candidates:
        raise SystemExit(f"no snapshot dirs under {base}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    try:
        import onnx
    except ImportError:
        print("ERROR: onnx is not installed (pip install onnx)", file=sys.stderr)
        return 2

    snapshot = _resolve_snapshot(args.snapshot)
    assets = snapshot / "assets"
    if not assets.exists():
        raise SystemExit(f"assets/ not found in {snapshot}")

    onnx_files = sorted(assets.glob("*.onnx"))
    report: dict = {
        "snapshot": str(snapshot),
        "asset_count": len(onnx_files),
        "models": [],
        "tooling": {
            "trtexec_on_path": shutil.which("trtexec"),
        },
    }

    try:
        import onnxruntime as ort

        report["tooling"]["onnxruntime_version"] = ort.__version__
        report["tooling"]["onnxruntime_providers"] = list(ort.get_available_providers())
    except ImportError:
        report["tooling"]["onnxruntime_version"] = None

    for path in onnx_files:
        try:
            model = onnx.load(str(path), load_external_data=False)
        except Exception as exc:
            report["models"].append({"file": path.name, "error": str(exc)})
            continue

        opsets = [{"domain": op.domain or "ai.onnx", "version": op.version} for op in model.opset_import]
        entry = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "ir_version": model.ir_version,
            "producer": f"{model.producer_name} {model.producer_version}".strip(),
            "opsets": opsets,
            "inputs": [_format_tensor(t) for t in model.graph.input],
            "outputs": [_format_tensor(t) for t in model.graph.output],
        }
        report["models"].append(entry)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"snapshot:        {report['snapshot']}")
    print(f"onnx assets:     {report['asset_count']}")
    print(f"trtexec on PATH: {report['tooling']['trtexec_on_path']}")
    print(f"onnxruntime:     {report['tooling'].get('onnxruntime_version')}")
    print(f"ort providers:   {report['tooling'].get('onnxruntime_providers')}")
    print()

    for m in report["models"]:
        print(f"== {m['file']} ==")
        if "error" in m:
            print(f"  load error: {m['error']}")
            continue
        size_mb = m["size_bytes"] / (1024 * 1024)
        print(f"  size: {size_mb:.2f} MB  ir_version={m['ir_version']}  producer={m['producer']!r}")
        opset_str = ", ".join(f"{o['domain']}={o['version']}" for o in m["opsets"])
        print(f"  opsets: {opset_str}")
        for t in m["inputs"]:
            print(f"  in : {t['name']:30s} {t['dtype']:10s} {t['shape']}")
        for t in m["outputs"]:
            print(f"  out: {t['name']:30s} {t['dtype']:10s} {t['shape']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
