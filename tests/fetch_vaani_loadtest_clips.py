"""Download N Vaani Hindi clips and cache them as 8 kHz mono int16 .raw files.

One-time prep for the L40s load test. Re-run only to refresh the snapshot.

Requires HUGGINGFACE_HUB_TOKEN env var with access to ARTPARK-IISc/Vaani-*.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_hf_token_from_dotenv(env_path: Path) -> str:
    if not env_path.is_file():
        raise SystemExit(f"{env_path} not found")
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != "HUGGINGFACE_HUB_TOKEN":
            continue
        token = value.strip().strip('"').strip("'")
        if not token:
            raise SystemExit(f"HUGGINGFACE_HUB_TOKEN in {env_path} is empty")
        return token
    raise SystemExit(f"HUGGINGFACE_HUB_TOKEN not set in {env_path}")

from tools.indicvoices_dataset import load_vaani_stream
from tools.eval_indicvoices_wer import (
    audio_to_float32,
    detect_audio_field,
    resample_linear,
    to_mono,
)

TARGET_SR = 8000
MIN_DURATION_S = 1.5
MAX_DURATION_S = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tests/fixtures/loadtest_vaani_8k")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--scan-limit", type=int, default=2000,
                    help="max samples to inspect from the stream")
    ap.add_argument("--seed", type=int, default=20260509)
    ap.add_argument("--cache-dir", default=None,
                    help="HF datasets cache dir (defaults to ~/.cache/huggingface)")
    ap.add_argument("--env-file", default=str(REPO_ROOT / ".env"),
                    help="path to .env file containing HUGGINGFACE_HUB_TOKEN")
    args = ap.parse_args()

    token = _load_hf_token_from_dotenv(Path(args.env_file))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] streaming Vaani Hindi (scan up to {args.scan_limit}, want {args.n})")
    stream = load_vaani_stream(token=token, cache_dir=args.cache_dir)

    audio_field = None
    candidates = []  # (idx, float32_pcm_at_target_sr, duration_s)
    for idx, sample in enumerate(stream):
        if idx >= args.scan_limit:
            break
        if audio_field is None:
            audio_field = detect_audio_field(sample)
        try:
            pcm, sr = audio_to_float32(sample[audio_field])
            pcm = to_mono(pcm)
            duration = len(pcm) / sr
            if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
                continue
            if sr != TARGET_SR:
                pcm = resample_linear(pcm, sr, TARGET_SR)
            candidates.append((idx, pcm.astype(np.float32), len(pcm) / TARGET_SR))
        except Exception as e:
            print(f"[skip {idx}] {e!r}")
            continue

        if len(candidates) >= args.n * 4:  # gather a pool, then sample
            break

    if len(candidates) < args.n:
        print(f"[warn] only {len(candidates)} candidates met filter; using all")
    rng = random.Random(args.seed)
    chosen = rng.sample(candidates, min(args.n, len(candidates)))
    if not chosen:
        raise SystemExit(
            "no clips met the duration filter; try increasing --scan-limit or "
            "relaxing MIN_DURATION_S/MAX_DURATION_S"
        )

    total_audio_s = 0.0
    for i, (orig_idx, pcm, duration) in enumerate(chosen):
        int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
        path = out_dir / f"vaani_{i:03d}_idx{orig_idx}_dur{int(duration*1000)}ms.raw"
        int16.tofile(path)
        total_audio_s += duration

    print(f"[done] wrote {len(chosen)} clips to {out_dir}")
    print(f"[done] total audio: {total_audio_s:.2f}s "
          f"(mean {total_audio_s/len(chosen):.2f}s/clip)")


if __name__ == "__main__":
    main()
