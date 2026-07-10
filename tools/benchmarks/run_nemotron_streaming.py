#!/usr/bin/env python3
"""Run NVIDIA Nemotron 3.5 ASR cache-aware streaming inference at one or more
latency operating points, timing each run for throughput RTF.

This is a thin, reproducible wrapper around NeMo's official
``examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py``.
We keep Nemotron fully DECOUPLED from the repo's pinned NeMo 2.4.1 stack: this
script only shells out, so it must run inside the isolated Nemotron conda env
(NeMo 26.06 from git main, Python >= 3.11). See tools/benchmarks/README.md.

For each --att-context-size it writes an output manifest (Nemotron predictions,
hyp key usually ``pred_text``) and records wall-clock + RTF into a timing sidecar
JSON that tools/benchmarks/score_nemotron_manifest.py can read.

Example (on the L4, env `nemotron-asr` active):
  python tools/benchmarks/run_nemotron_streaming.py \
    --manifest artifacts/benchmarks/nemotron_vs_indic/contextual.valid.jsonl \
    --script ~/NeMo-src/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    --target-lang hi-IN --att-context-sizes "[56,3]" "[56,13]" --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Published chunk/emission latency for each Nemotron att_context_size [left, right].
EMISSION_LATENCY_MS: dict[str, int] = {
    "[56,0]": 80,
    "[56,1]": 160,
    "[56,3]": 320,
    "[56,6]": 560,
    "[56,13]": 1120,
}


def canon_att(att: str) -> str:
    """Normalize an att_context_size token so '[56, 3]' and '[56,3]' map the same."""
    return att.replace(" ", "")


def att_tag(att: str) -> str:
    return canon_att(att).strip("[]").replace(",", "_")


def total_audio_seconds(manifest: Path) -> float:
    total = 0.0
    missing = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        dur = row.get("duration")
        if dur is None:
            missing += 1
            continue
        total += float(dur)
    if missing:
        print(f"WARNING: {missing} manifest rows lack 'duration'; RTF will undercount.", file=sys.stderr)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="NeMo JSONL manifest (audio_filepath, duration, text).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: manifest's parent).")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b", help="HF repo id or local .nemo path.")
    parser.add_argument(
        "--script",
        default=os.path.expanduser("~/NeMo-src/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"),
        help="Path to NeMo's cache-aware streaming infer script (clone NeMo for examples/).",
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for the NeMo run.")
    parser.add_argument("--target-lang", default="hi-IN", help="Language tag; 'auto' for LID. Hindi = hi-IN.")
    parser.add_argument("--att-context-sizes", nargs="+", default=["[56,3]", "[56,13]"], help="One or more [left,right] points.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model-arg", default="auto", choices=["auto", "pretrained_name", "model_path"],
        help="Hydra key for the model. 'auto' (default) -> pretrained_name for an HF repo id, "
             "model_path for a local .nemo. (NeMo's infer script restore_from's model_path, so an HF id needs pretrained_name.)",
    )
    parser.add_argument("--strip-lang-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra", nargs="*", default=[], help="Extra key=value args forwarded verbatim to the NeMo script.")
    parser.add_argument("--label", default=None, help="Output filename prefix (default: manifest stem).")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    out_dir = (args.out_dir or manifest.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or manifest.stem
    script = Path(os.path.expanduser(args.script))
    if not args.dry_run and not script.exists():
        raise SystemExit(
            f"NeMo infer script not found: {script}\n"
            "Clone it: git clone --depth 1 https://github.com/NVIDIA/NeMo.git ~/NeMo-src"
        )

    audio_s = total_audio_seconds(manifest)
    print(f"manifest={manifest}\n total_audio={audio_s/60.0:.1f} min  label={label}\n", flush=True)

    if args.model_arg == "auto":
        is_local = args.model.endswith(".nemo") or os.path.exists(args.model)
        model_arg = "model_path" if is_local else "pretrained_name"
    else:
        model_arg = args.model_arg
    print(f"model arg: {model_arg}={args.model}\n", flush=True)

    records = []
    for att in args.att_context_sizes:
        att = canon_att(att)
        tag = att_tag(att)
        # NeMo's infer script treats output_path as a DIRECTORY and writes
        # streaming_out_<model>_<manifest>.json inside it. Give it a dedicated dir,
        # then flatten to an att-tagged file the scorer can label/distinguish.
        nemo_out_dir = out_dir / f"nemotron_{label}_att{tag}.d"
        final_path = out_dir / f"nemotron_{label}_att{tag}.json"
        cmd = [
            args.python, str(script),
            f"{model_arg}={args.model}",
            f"dataset_manifest={manifest}",
            f"output_path={nemo_out_dir}",
            f"target_lang={args.target_lang}",
            f"att_context_size={att}",
            f"strip_lang_tags={'true' if args.strip_lang_tags else 'false'}",
            f"batch_size={args.batch_size}",
            *args.extra,
        ]
        print("RUN:", shlex.join(cmd), flush=True)
        rec = {
            "att_context_size": att,
            "emission_latency_ms": EMISSION_LATENCY_MS.get(att),
            "output_path": str(final_path),
            "target_lang": args.target_lang,
            "total_audio_s": round(audio_s, 3),
        }
        if args.dry_run:
            records.append({**rec, "dry_run": True})
            continue

        start = time.perf_counter()
        proc = subprocess.run(cmd, check=False)
        wall = time.perf_counter() - start
        rec["returncode"] = proc.returncode
        rec["wall_clock_s"] = round(wall, 3)
        rec["rtf"] = round(wall / audio_s, 5) if audio_s > 0 else None
        if proc.returncode != 0:
            print(f"  !! NeMo run exited {proc.returncode} for att={att} — see output above.", file=sys.stderr, flush=True)
        else:
            produced = []
            if nemo_out_dir.is_dir():
                produced = sorted(nemo_out_dir.glob("*.json")) + sorted(nemo_out_dir.glob("*.jsonl"))
            if nemo_out_dir.is_file():            # some builds may write a single file
                shutil.copyfile(nemo_out_dir, final_path)
            elif produced:
                shutil.copyfile(produced[0], final_path)
            else:
                rec["output_path"] = str(nemo_out_dir)
                print(f"  !! no produced file found under {nemo_out_dir}", file=sys.stderr, flush=True)
            print(f"  ok att={att}  wall={wall:.1f}s  rtf={rec['rtf']}  -> {rec['output_path']}", flush=True)
        records.append(rec)

    timing_path = out_dir / f"nemotron_{label}_timing.json"
    timing_path.write_text(json.dumps({"manifest": str(manifest), "model": args.model, "runs": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote timing sidecar: {timing_path}", flush=True)
    if not args.dry_run:
        print("Verify the hyp key on the first output row (expected 'pred_text'):")
        print(f"  head -1 {records[0]['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
