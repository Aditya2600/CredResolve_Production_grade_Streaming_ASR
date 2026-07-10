#!/usr/bin/env python3
"""Score Nemotron 3.5 ASR and the production IndicConformer on the SAME manifest,
through the SAME normalizer + WER, for an apples-to-apples comparison.

Gold references come from the manifest (single source of truth). Hypotheses come
from each model's output and are matched by audio basename (IndicConformer's
out-jsonl is re-sorted by WER, so we never match by row order). Both sides are
scored via the repo's shared path -- this mirrors
tools/benchmarks/contextual_asr_hindi_benchmark.compute_sample_wer exactly:
    normalize_asr_text(text)  ->  compute_wer.normalize  ->  compute_wer.edit_distance
normalize_asr_text strips punctuation/casing/Latin/<tags>/[brackets], neutralizing
Nemotron's rich text vs IndicConformer's bare Devanagari.

Pure-python (no torch/nemo) -> runs in either env.

Example:
  python tools/benchmarks/score_nemotron_manifest.py \
    --manifest artifacts/benchmarks/nemotron_vs_indic/contextual.valid.jsonl \
    --nemotron-json artifacts/benchmarks/nemotron_vs_indic/nemotron_contextual.valid_att56_3.json \
                    artifacts/benchmarks/nemotron_vs_indic/nemotron_contextual.valid_att56_13.json \
    --indic-jsonl artifacts/benchmarks/nemotron_vs_indic/indic_contextual.jsonl \
    --timing-json artifacts/benchmarks/nemotron_vs_indic/nemotron_contextual.valid_timing.json \
    --out-md artifacts/benchmarks/nemotron_vs_indic/report.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.asr_text_normalizer import normalize_asr_text
    from tools.compute_wer import edit_distance, normalize as wer_normalize
except ImportError:  # pragma: no cover - allow running from tools/
    from asr_text_normalizer import normalize_asr_text
    from compute_wer import edit_distance, normalize as wer_normalize

EMISSION_LATENCY_MS = {"[56,0]": 80, "[56,1]": 160, "[56,3]": 320, "[56,6]": 560, "[56,13]": 1120}


def score_pair(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(s, d, i, ref_words) using the repo's shared normalize+WER path."""
    ref_words = wer_normalize(normalize_asr_text(ref))
    hyp_words = wer_normalize(normalize_asr_text(hyp))
    s, d, i = edit_distance(ref_words, hyp_words)
    return s, d, i, len(ref_words)


def load_jsonl_or_json(path: Path) -> list[dict]:
    """Robustly load JSONL (one obj per line) or a single JSON array."""
    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            rows = []
            break
        if isinstance(obj, dict):
            rows.append(obj)
    if not rows:
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
    return rows


def base(path_value: str) -> str:
    return Path(str(path_value)).name


AUDIO_KEYS = ("audio_filepath", "audio_path", "audio", "path")
TEXT_KEYS = ("text", "reference", "raw_reference", "transcript", "sentence")


def row_audio(row: dict) -> str | None:
    for k in AUDIO_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return base(v)
        if isinstance(v, dict) and v.get("path"):
            return base(v["path"])
    return None


def build_hyps(rows: list[dict], hyp_key: str):
    """Return (by_audio_basename, by_row_order).

    NeMo's streaming output has no audio path (keys are just pred_text/text/wer), so we
    also keep row order and fall back to index alignment with the manifest.
    """
    by_audio: dict[str, str] = {}
    by_order: list[str] = []
    for row in rows:
        hyp = str(row.get(hyp_key, "") or "")
        by_order.append(hyp)
        audio = row_audio(row)
        if audio is not None:
            by_audio[audio] = hyp
    return by_audio, by_order


def att_from_name(name: str) -> str | None:
    m = re.search(r"att(\d+)_(\d+)", name)
    return f"[{m.group(1)},{m.group(2)}]" if m else None


def nemotron_label(path: Path) -> tuple[str, int | None]:
    att = att_from_name(path.name)
    if att and att in EMISSION_LATENCY_MS:
        lat = EMISSION_LATENCY_MS[att]
        return f"Nemotron {lat}ms {att}", lat
    if att:
        return f"Nemotron {att}", None
    return f"Nemotron ({path.stem})", None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="Reference manifest (audio_filepath, text).")
    p.add_argument("--nemotron-json", type=Path, nargs="*", default=[], help="Nemotron output manifest(s).")
    p.add_argument("--nemotron-hyp-key", default="pred_text", help="Hyp field in Nemotron output (default pred_text).")
    p.add_argument("--indic-jsonl", type=Path, default=None, help="IndicConformer eval_nemo_manifest_wer.py --out-jsonl.")
    p.add_argument("--indic-hyp-key", default="hypothesis")
    p.add_argument("--indic-label", default="IndicConformer (prod)")
    p.add_argument("--timing-json", type=Path, nargs="*", default=[], help="run_nemotron_streaming timing sidecar(s) for RTF.")
    p.add_argument("--out-md", type=Path, default=None, help="Markdown report path.")
    p.add_argument("--out-json", type=Path, default=None, help="Summary JSON path.")
    p.add_argument("--out-jsonl", type=Path, default=None, help="Per-sample JSONL path.")
    p.add_argument("--examples", type=int, default=8, help="Number of example rows in the report.")
    args = p.parse_args()

    # Reference: manifest is the single source of truth.
    ref_map: dict[str, str] = {}
    order: list[str] = []
    for row in load_jsonl_or_json(args.manifest):
        audio = row_audio(row)
        if audio is None:
            continue
        ref = next((str(row[k]) for k in TEXT_KEYS if row.get(k)), "")
        ref_map[audio] = ref
        order.append(audio)
    if not ref_map:
        raise SystemExit(f"no usable reference rows in {args.manifest}")

    # RTF / latency lookup keyed by output basename.
    rtf_by_output: dict[str, dict] = {}
    for tpath in args.timing_json:
        for run in json.loads(tpath.read_text(encoding="utf-8")).get("runs", []):
            op = run.get("output_path")
            if op:
                rtf_by_output[base(op)] = run

    # Assemble models: label -> {by_audio, by_order, rtf, latency_ms, source}
    models: dict[str, dict] = {}
    for npath in args.nemotron_json:
        label, lat = nemotron_label(npath)
        run = rtf_by_output.get(base(str(npath)), {})
        ba, bo = build_hyps(load_jsonl_or_json(npath), args.nemotron_hyp_key)
        models[label] = {
            "by_audio": ba, "by_order": bo,
            "rtf": run.get("rtf"),
            "latency_ms": run.get("emission_latency_ms", lat),
            "source": str(npath),
        }
    if args.indic_jsonl:
        ba, bo = build_hyps(load_jsonl_or_json(args.indic_jsonl), args.indic_hyp_key)
        models[args.indic_label] = {
            "by_audio": ba, "by_order": bo,
            "rtf": None, "latency_ms": None, "source": str(args.indic_jsonl),
        }
    if not models:
        raise SystemExit("provide at least one of --nemotron-json / --indic-jsonl")

    # Warn if a model relies on index alignment but its row count != manifest count.
    for label, m in models.items():
        if not m["by_audio"] and len(m["by_order"]) != len(order):
            print(f"WARNING: {label} has no audio keys and {len(m['by_order'])} rows vs "
                  f"{len(order)} manifest rows — row-order alignment may be off.", file=sys.stderr)

    # Score.
    agg = {label: {"s": 0, "d": 0, "i": 0, "words": 0, "matched": 0, "missing": 0} for label in models}
    per_sample = []
    for idx, audio in enumerate(order):
        ref = ref_map[audio]
        entry = {"audio": audio, "reference": ref, "models": {}}
        for label, m in models.items():
            hyp = m["by_audio"].get(audio)
            if hyp is None and idx < len(m["by_order"]):
                hyp = m["by_order"][idx]   # NeMo output has no audio path -> align by manifest order
            if hyp is None:
                agg[label]["missing"] += 1
                continue
            s, d, i, words = score_pair(ref, hyp)
            a = agg[label]
            a["s"] += s; a["d"] += d; a["i"] += i; a["words"] += words; a["matched"] += 1
            entry["models"][label] = {
                "hypothesis": hyp,
                "wer": round(((s + d + i) / words) if words else (0.0 if not hyp.strip() else 1.0), 4),
                "substitutions": s, "deletions": d, "insertions": i, "ref_words": words,
            }
        per_sample.append(entry)

    summary = {"manifest": str(args.manifest), "clips": len(order), "models": {}}
    for label, a in agg.items():
        w = a["words"]
        summary["models"][label] = {
            "wer": (a["s"] + a["d"] + a["i"]) / w if w else 0.0,
            "wer_percent": round(((a["s"] + a["d"] + a["i"]) / w * 100) if w else 0.0, 2),
            "substitutions": a["s"], "deletions": a["d"], "insertions": a["i"],
            "reference_words": w, "matched": a["matched"], "missing": a["missing"],
            "rtf": models[label]["rtf"], "latency_ms": models[label]["latency_ms"],
            "source": models[label]["source"],
        }

    # Console + markdown (sorted by WER ascending).
    ranked = sorted(summary["models"].items(), key=lambda kv: kv[1]["wer"])
    print("\n" + "=" * 92)
    print(f"Nemotron vs IndicConformer  |  {len(order)} clips  |  {args.manifest.name}")
    print("=" * 92)
    hdr = f"{'Model':<28} | {'Lat(ms)':>7} | {'Clips':>5} | {'RefW':>6} | {'S':>5} | {'D':>5} | {'I':>5} | {'WER%':>7} | {'RTF':>6}"
    print(hdr); print("-" * len(hdr))
    for label, s in ranked:
        lat = s["latency_ms"] if s["latency_ms"] is not None else "-"
        rtf = f"{s['rtf']:.4f}" if s["rtf"] is not None else "-"
        print(f"{label:<28} | {str(lat):>7} | {s['matched']:>5} | {s['reference_words']:>6} | {s['substitutions']:>5} | {s['deletions']:>5} | {s['insertions']:>5} | {s['wer_percent']:>6.2f}% | {rtf:>6}")
        if s["missing"]:
            print(f"  (note: {s['missing']} clips missing a hypothesis for {label})")
    print("=" * 92 + "\n")

    md = ["# Nemotron 3.5 ASR vs IndicConformer — Hindi benchmark\n",
          f"Manifest: `{args.manifest}` · **{len(order)}** clips · scoring = `normalize_asr_text` + corpus WER (apples-to-apples).\n",
          "| Model | Emission latency | Clips | Ref words | Sub | Del | Ins | WER % | Throughput RTF |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, s in ranked:
        lat = f"{s['latency_ms']} ms" if s["latency_ms"] is not None else "—"
        rtf = f"{s['rtf']:.4f}" if s["rtf"] is not None else "—"
        md.append(f"| **{label}** | {lat} | {s['matched']} | {s['reference_words']} | {s['substitutions']} | {s['deletions']} | {s['insertions']} | **{s['wer_percent']:.2f}%** | {rtf} |")
    md.append("\n## Spot-check examples\n")
    for entry in per_sample[: max(0, args.examples)]:
        md.append(f"- **ref**: `{entry['reference']}`")
        for label, mp in entry["models"].items():
            md.append(f"  - {label} (WER {mp['wer'] * 100:.0f}%): `{mp['hypothesis']}`")
        md.append("")
    md_text = "\n".join(md) + "\n"

    out_md = args.out_md or (args.manifest.parent / "nemotron_vs_indic_report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")
    print(f"Wrote markdown report: {out_md}")

    if args.out_json:
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote summary JSON: {args.out_json}")
    if args.out_jsonl:
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for entry in per_sample:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Wrote per-sample JSONL: {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
