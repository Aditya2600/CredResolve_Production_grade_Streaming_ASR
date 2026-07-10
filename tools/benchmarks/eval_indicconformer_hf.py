#!/usr/bin/env python3
"""Transcribe a NeMo JSONL manifest with AI4Bharat IndicConformer-600M (multilingual),
the production model, emitting scorer-compatible hypotheses.

`ai4bharat/indic-conformer-600m-multilingual` is a GATED, ONNX-packaged repo (its config
auto-maps to `model_onnx.IndicASRModel`). We load it exactly the way the production worker
does (worker/app/model.py): snapshot_download -> import model_onnx.py -> IndicASRConfig
-> IndicASRModel, then call `model(wav, lang, decoding=...)`. Runs offline/full-context per
clip — the fair production-accuracy baseline vs Nemotron's streaming operating points.

Requirements (in the venv): `pip install onnxruntime-gpu` (the model uses ONNX Runtime),
plus huggingface_hub / soundfile / numpy / torch. The HF token (HUGGINGFACE_HUB_TOKEN/HF_TOKEN,
auto-loaded from .env) must have access to the gated repo — the same token production uses.

Output JSONL rows: {"audio_filepath","reference","hypothesis"} -> feed as --indic-jsonl to
tools/benchmarks/score_nemotron_manifest.py (matches by audio basename, hyp key "hypothesis").

Example:
  python tools/benchmarks/eval_indicconformer_hf.py \
    --manifest artifacts/benchmarks/nemotron_vs_indic/vaani.valid.jsonl \
    --language hi --decoder rnnt --limit 3 \
    --out-jsonl /tmp/indic_smoke.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(*paths) -> None:
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if sep and key and val and key not in os.environ:
                os.environ[key] = val


_load_dotenv(REPO_ROOT / ".env", Path.cwd() / ".env")


def resample_linear(audio, src_sr: int, dst_sr: int):
    import numpy as np
    if src_sr == dst_sr or audio.size == 0:
        return audio
    n = max(1, int(round(audio.shape[0] * (dst_sr / src_sr))))
    pos = np.linspace(0.0, audio.shape[0] - 1, num=n, dtype="float32")
    return np.interp(pos, np.arange(audio.shape[0], dtype="float32"), audio).astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-name", default="ai4bharat/indic-conformer-600m-multilingual")
    ap.add_argument("--language", default="hi", help="IndicConformer language id (production default: hi).")
    ap.add_argument("--decoder", default="rnnt", choices=["rnnt", "ctc"], help="Production default: rnnt.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0, help="Cap clips (0 = all); use a few first to verify the interface.")
    ap.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
        help="HF token with access to the gated repo. Default: HUGGINGFACE_HUB_TOKEN/HF_TOKEN (auto-loaded from .env).",
    )
    ap.add_argument("--out-jsonl", type=Path, required=True)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    if not args.hf_token:
        print("WARNING: no HF token found — this repo is gated and will 401.", file=sys.stderr, flush=True)

    print(f"Downloading snapshot {args.model_name} ...", flush=True)
    snap = snapshot_download(repo_id=args.model_name, token=args.hf_token)
    model_onnx_path = Path(snap) / "model_onnx.py"
    if not model_onnx_path.exists():
        raise SystemExit(f"model_onnx.py not found in snapshot {snap}")

    spec = importlib.util.spec_from_file_location("ai4bharat_model_onnx", str(model_onnx_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["ai4bharat_model_onnx"] = module
    spec.loader.exec_module(module)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Building IndicASRModel on {device} ...", flush=True)
    # model_onnx.py selects ORT providers from torch.cuda.is_available(), ignoring config.device,
    # so mask it for explicit CPU runs (mirrors worker/app/model._instantiate_indic_asr_model).
    _cuda_avail = torch.cuda.is_available
    if device == "cpu":
        torch.cuda.is_available = lambda: False
    try:
        config = module.IndicASRConfig(ts_folder=snap, device=device, FRAME_DURATION_MS=0.08)
        model = module.IndicASRModel(config)
    finally:
        torch.cuda.is_available = _cuda_avail

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} clips; decoder={args.decoder} language={args.language}", flush=True)

    out = []
    for i, r in enumerate(rows):
        path = r.get("audio_filepath") or r.get("audio_path")
        wav, sr = sf.read(path, dtype="float32")
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1).astype("float32")
        if sr != args.target_sr:
            wav = resample_linear(wav, sr, args.target_sr)
        wav_t = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)  # [1, T], CPU (model handles device)
        with torch.inference_mode():
            res = model(wav_t, args.language, decoding=args.decoder)
        if isinstance(res, tuple):
            res = res[0]
        if isinstance(res, list):
            res = res[0] if res else ""
        hyp = str(res or "").strip()
        out.append({"audio_filepath": path, "reference": r.get("text", ""), "hypothesis": hyp})
        if i < 2:
            print(f"  [{i}] ref={r.get('text','')[:50]!r}  hyp={hyp[:50]!r}", flush=True)
        elif (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} rows -> {args.out_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
