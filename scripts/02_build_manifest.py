#!/usr/bin/env python3
"""
02_build_manifest.py — Build NeMo JSONL manifests from OpenSLR 118 (Gram Vaani).

Pipeline (in order):
  1. Convert MP3 -> 16kHz mono WAV (ffmpeg, ThreadPoolExecutor, idempotent).
  2. Load Kaldi `text` transcripts + metadata TSV, join on uttid.
  3. Filter by duration and metadata `Other` flags.
  4. Write NeMo manifests to /data/manifests/{train,dev,eval}.json
  5. Print per-split summary.
  6. Tokenizer <unk> check against the pretrained SPE tokenizer (train only).

Non-negotiables honoured:
  - Reuses the pretrained SPE tokenizer (loads it, never rebuilds).
  - Keeps code-switched (Latin) transcripts verbatim; does not strip anything.
  - Every path is telephony-realistic; noisy-but-valid utts (inaudible/audio_jump)
    are KEPT and tagged via the `quality` field.
  - --dry-run prints what it would do and writes nothing.

Log to /data/exp/logs/02_manifest.log via `tee` from the shell, e.g.:
    python3 scripts/02_build_manifest.py 2>&1 | tee -a /data/exp/logs/02_manifest.log
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Paths -------------------------------------------------------------------
RAW_ROOT = "/data/raw/gramvaani"
META_DIR = os.path.join(RAW_ROOT, "Metadata")
WAV_ROOT = "/data/wav16k"
MANIFEST_DIR = "/data/manifests"

SPLITS = {
    "train": {"dir": "GV_Train_100h", "meta": "utt2labels_GV_Train_100h.txt"},
    "dev":   {"dir": "GV_Dev_5h",     "meta": "utt2labels_GV_Dev_5h.txt"},
    "eval":  {"dir": "GV_Eval_3h",    "meta": "utt2labels_GV_Eval_3h.txt"},
}

# --- Filter policy -----------------------------------------------------------
MIN_DUR = 0.5    # seconds
MAX_DUR = 20.0   # seconds
DROP_FLAGS = {"clipping", "truncated", "empty", "No_speaking"}   # unusable audio
KEEP_NOISY = {"inaudible", "audio_jump"}                          # noisy but valid
HIST_EDGES = [0, 2, 4, 6, 8, 11, 15, 20]

LATIN_RE = re.compile(r"[A-Za-z]")

# --- Text normalization ------------------------------------------------------
# The 256-piece Hindi SPE tokenizer has NO punctuation pieces, so any of these
# chars become <unk> TARGETS and poison the CTC/RNNT heads. Strip them. Latin
# letters (code-switch) are deliberately preserved.
INCOMPLETE_RE = re.compile(r"#incomp", re.IGNORECASE)   # detect incomplete marker
HASHTAG_RE = re.compile(r"#[A-Za-z]+")                  # remove #incomplete etc,
#                                                        # keeps glued Hindi (#..दिया)
PUNCT_CHARS = "'\"[](),.:;!?/\\-#"
ZERO_WIDTH = [0x200B, 0x200C, 0x200D, 0xFEFF]           # ZWSP, ZWNJ, ZWJ, BOM
PUNCT_TABLE = {ord(c): " " for c in PUNCT_CHARS}
PUNCT_TABLE.update({cp: " " for cp in ZERO_WIDTH})
WS_RE = re.compile(r"\s+")


def normalize_text(text):
    """Return (clean_text, was_incomplete). Strips annotation/punct artifacts."""
    was_incomplete = bool(INCOMPLETE_RE.search(text))
    t = HASHTAG_RE.sub(" ", text)      # drop #incomplete-family, keep glued Hindi
    t = t.translate(PUNCT_TABLE)       # drop remaining punctuation -> space
    t = WS_RE.sub(" ", t).strip()      # collapse whitespace
    return t, was_incomplete

# Gated repo(s) to try, in order. The account may only be granted access to one
# of these two naming variants — try both before giving up.
PRETRAINED_MODELS = (
    "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
    "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large",
)


# --- Small helpers -----------------------------------------------------------
def log(msg=""):
    print(msg, flush=True)


def wav_path(split, uttid):
    return os.path.join(WAV_ROOT, split, f"{uttid}.wav")


def mp3_path(split, uttid):
    return os.path.join(RAW_ROOT, SPLITS[split]["dir"], "Audio", f"{uttid}.mp3")


def speaker_of(uttid):
    return uttid.rsplit("-", 1)[0]


def load_hf_token():
    """Resolve a HuggingFace token for the gated IndicConformer repo.

    Checks already-exported env vars first, then parses the repo-root .env
    (stdlib only, no python-dotenv). Accepts several common var names — note
    the repo uses the non-standard HUGGINGFACE_HUB_TOKEN which huggingface_hub
    does NOT auto-detect. Returns the token string or None. Does not print it.
    """
    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
             "HUGGINGFACEHUB_API_TOKEN")
    for n in names:
        if os.environ.get(n):
            return os.environ[n].strip()

    # repo root = parent of scripts/ (this file lives in scripts/)
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return None
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k in names:
                    v = v.strip().strip('"').strip("'")
                    if v:
                        return v
    except Exception:                                # noqa: BLE001
        return None
    return None


def ffmpeg_cmd(src_mp3, dst_wav):
    """16kHz mono PCM WAV, quiet."""
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src_mp3,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", dst_wav,
    ]


def read_text(path):
    """Parse Kaldi `text`: '<uttid> <transcript...>'. Returns {uttid: text}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 1:
                out[parts[0]] = ""          # uttid with empty transcript
            else:
                out[parts[0]] = parts[1].strip()
    return out


def read_metadata(path):
    """Parse tab-delimited metadata TSV with header. Returns {uttid: {col: val}}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        uid_i = idx.get("Uttids", 0)
        other_i = idx.get("Other")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            uid = cols[uid_i]
            other = cols[other_i] if (other_i is not None and other_i < len(cols)) else "NA"
            out[uid] = {"Other": other.strip() or "NA"}
    return out


# --- Step 1: convert ---------------------------------------------------------
def convert_split(split, uttids, workers, dry_run):
    """Convert MP3 -> 16kHz mono WAV. Idempotent. Returns list of failures."""
    dst_dir = os.path.join(WAV_ROOT, split)
    to_convert = []
    already = 0
    missing_src = []
    for uid in uttids:
        src = mp3_path(split, uid)
        dst = wav_path(split, uid)
        if not os.path.exists(src):
            missing_src.append(uid)
            continue
        if os.path.exists(dst):
            already += 1
            continue
        to_convert.append((uid, src, dst))

    log(f"[{split}] convert: {len(uttids)} candidates | "
        f"{already} already present | {len(to_convert)} to convert | "
        f"{len(missing_src)} missing source mp3")

    if missing_src:
        log(f"[{split}]   WARNING: {len(missing_src)} uttids have no source mp3, "
            f"e.g. {missing_src[:3]}")

    if dry_run:
        return []

    os.makedirs(dst_dir, exist_ok=True)

    failures = []
    done = 0

    def _work(item):
        uid, src, dst = item
        tmp = dst + ".tmp.wav"
        try:
            r = subprocess.run(ffmpeg_cmd(src, tmp),
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
            if r.returncode != 0 or not os.path.exists(tmp):
                return uid, r.stderr.decode("utf-8", "replace")[-200:]
            os.replace(tmp, dst)   # atomic; keeps re-runs idempotent
            return uid, None
        except Exception as e:                       # noqa: BLE001
            return uid, str(e)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, it) for it in to_convert]
        for fut in as_completed(futs):
            uid, err = fut.result()
            done += 1
            if err:
                failures.append((uid, err))
            if done % 2000 == 0 or done == len(to_convert):
                log(f"[{split}]   converted {done}/{len(to_convert)} "
                    f"({len(failures)} failures)")

    if failures:
        log(f"[{split}]   {len(failures)} conversion FAILURES, "
            f"e.g. {failures[0][0]}: {failures[0][1]}")
    return failures


# --- Step 3 helper: durations ------------------------------------------------
def probe_durations(split, uttids, workers):
    """Header-only WAV duration probe via soundfile. Returns {uttid: seconds}."""
    import soundfile as sf

    durs = {}

    def _probe(uid):
        p = wav_path(split, uid)
        try:
            info = sf.info(p)
            return uid, info.frames / float(info.samplerate)
        except Exception:                            # noqa: BLE001
            return uid, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for uid, d in ex.map(_probe, uttids):
            durs[uid] = d
    return durs


# --- Step 3/4: filter + build entries ---------------------------------------
def build_entries(split, texts, meta, durs, keep_incomplete=False):
    """Apply filters + text normalization, return (entries, drop_counts, norm)."""
    entries = []
    drops = Counter()
    norm = Counter()   # normalization bookkeeping
    for uid, text in texts.items():
        dur = durs.get(uid)
        if dur is None:
            drops["no_audio"] += 1
            continue
        if dur < MIN_DUR:
            drops["too_short"] += 1
            continue
        if dur > MAX_DUR:
            drops["too_long"] += 1
            continue
        quality = meta.get(uid, {}).get("Other", "NA")
        # `Other` may hold comma-combined flags, e.g. "audio_jump,inaudible" or
        # "clipping,inaudible". Drop if ANY component is an unusable-audio flag.
        flag_set = {f.strip() for f in quality.split(",") if f.strip()}
        bad = flag_set & DROP_FLAGS
        if bad:
            drops[f"flag_{'+'.join(sorted(bad))}"] += 1
            continue

        # Text normalization: strip punctuation/annotation artifacts that the
        # 256-piece Hindi SPE cannot represent (they would become <unk> targets).
        clean, was_incomplete = normalize_text(text)
        if was_incomplete:
            norm["incomplete_marked"] += 1
            if not keep_incomplete:
                # Partial transcript over full audio = misalignment; drop it.
                drops["incomplete_transcript"] += 1
                continue
        if clean != text:
            norm["normalized"] += 1
        if not clean:
            drops["empty_after_norm"] += 1
            continue

        entries.append({
            "audio_filepath": wav_path(split, uid),
            "duration": round(dur, 3),
            "text": clean,
            "speaker": speaker_of(uid),
            "lang": "hi",
            "quality": quality,
        })
    return entries, drops, norm


# --- Step 5: summary ---------------------------------------------------------
def print_summary(split, entries, drops, norm=None):
    log(f"\n===== SUMMARY [{split}] =====")
    total = len(entries)
    hours = sum(e["duration"] for e in entries) / 3600.0
    log(f"kept entries      : {total}")
    log(f"total hours       : {hours:.2f}")

    log("dropped:")
    if drops:
        for reason, n in sorted(drops.items()):
            log(f"  {reason:22s}: {n}")
    else:
        log("  (none)")

    if norm:
        log("text normalization:")
        for reason, n in sorted(norm.items()):
            log(f"  {reason:22s}: {n}")

    qdist = Counter(e["quality"] for e in entries)
    log("quality distribution (kept):")
    for q, n in sorted(qdist.items(), key=lambda kv: -kv[1]):
        log(f"  {q:22s}: {n}")

    speakers = {e["speaker"] for e in entries}
    log(f"unique speakers   : {len(speakers)}")

    # Duration histogram
    buckets = [0] * (len(HIST_EDGES) - 1)
    for e in entries:
        d = e["duration"]
        for i in range(len(HIST_EDGES) - 1):
            if HIST_EDGES[i] <= d < HIST_EDGES[i + 1] or (
                i == len(HIST_EDGES) - 2 and d == HIST_EDGES[-1]):
                buckets[i] += 1
                break
    log("duration histogram:")
    for i, c in enumerate(buckets):
        log(f"  [{HIST_EDGES[i]:>2},{HIST_EDGES[i+1]:>2})s : {c}")

    latin = sum(1 for e in entries if LATIN_RE.search(e["text"]))
    log(f"utts with Latin chars (code-switch): {latin}")

    log("sample manifest lines:")
    for e in entries[:3]:
        log("  " + json.dumps(e, ensure_ascii=False))


# --- Step 6: tokenizer <unk> check ------------------------------------------
def extract_spe_from_nemo(token):
    """Download the gated .nemo and extract EVERY SentencePiece .model in it.

    Bypasses NeMo model instantiation (broken for this checkpoint). IndicConformer
    bundles multiple per-language sub-tokenizers (hash-named *_tokenizer.model),
    so we extract them all and let the caller pick the one that encodes Hindi.
    Returns a list of (member_name, local_path), or [] on failure.
    """
    import tarfile
    import tempfile
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as e:                           # noqa: BLE001
        log(f"[skip] huggingface_hub not importable: {e}")
        return []

    for repo in PRETRAINED_MODELS:
        try:
            files = list_repo_files(repo, token=token)
        except Exception as e:                       # noqa: BLE001
            log(f"[warn] list_repo_files failed for {repo}: {str(e)[:120]}")
            continue
        nemo_files = [f for f in files if f.endswith(".nemo")]
        if not nemo_files:
            log(f"[warn] no .nemo file in {repo}")
            continue
        try:
            log(f"downloading {repo}/{nemo_files[0]} ...")
            local = hf_hub_download(repo_id=repo, filename=nemo_files[0],
                                    token=token)
        except Exception as e:                       # noqa: BLE001
            log(f"[warn] download failed for {repo}: {str(e)[:120]}")
            continue

        try:
            extracted = []
            tmpdir = tempfile.mkdtemp(prefix="gv_spe_")
            with tarfile.open(local, "r:*") as tar:
                members = [m for m in tar.getnames() if m.endswith(".model")]
                if not members:
                    log(f"[warn] no *.model inside {nemo_files[0]}; "
                        f"members e.g. {tar.getnames()[:8]}")
                    continue
                for m in members:
                    data = tar.extractfile(m).read()
                    out = os.path.join(tmpdir, os.path.basename(m))
                    with open(out, "wb") as fo:
                        fo.write(data)
                    extracted.append((m, out))
            log(f"extracted {len(extracted)} SPE model(s) from {repo}: "
                f"{[os.path.basename(m) for m, _ in extracted]}")
            return extracted
        except Exception as e:                       # noqa: BLE001
            log(f"[warn] tar extract failed for {repo}: {str(e)[:120]}")
            continue
    return []


def tokenizer_unk_check(entries):
    log("\n===== TOKENIZER <unk> CHECK (train) =====")
    # Authenticate for the gated repo BEFORE importing/downloading.
    token = load_hf_token()
    if token:
        # Export under the names huggingface_hub actually checks.
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            log("authenticated to HuggingFace Hub via .env token")
        except Exception as e:                       # noqa: BLE001
            log(f"[warn] huggingface_hub.login failed ({e}); "
                f"relying on HF_TOKEN env var")
    else:
        log("[warn] no HF token found (checked env + .env); "
            "gated model load will likely 401")

    # NeMo 25.02 cannot instantiate this IndicConformer checkpoint
    # (_setup_monolingual_tokenizer does cfg.tokenizer.pop('dir') and the
    # checkpoint has no 'dir' key -> KeyError). We don't need the model: a
    # .nemo is a tar archive holding the SentencePiece .model. Extract it and
    # load with `sentencepiece` directly.
    candidates = extract_spe_from_nemo(token)
    if not candidates:
        log("[skip] could not obtain SentencePiece model from any repo")
        return

    try:
        import sentencepiece as spm_lib
    except Exception as e:                           # noqa: BLE001
        log(f"[skip] sentencepiece not importable: {e}")
        return

    # Sample of real transcripts to score each sub-tokenizer.
    sample = [e["text"] for e in entries if e["text"]][:2000]

    def unk_rate(sp, unk_id, texts):
        tot = unk = 0
        for t in texts:
            ids = sp.EncodeAsIds(t)
            tot += len(ids)
            unk += sum(1 for i in ids if i == unk_id)
        return (unk / tot) if tot else 1.0

    # Pick the sub-tokenizer with the lowest <unk> rate on Hindi text.
    best = None  # (rate, sp, unk_id, name, vocab)
    log("scoring sub-tokenizer candidates on 2000 train transcripts:")
    for name, path in candidates:
        sp = spm_lib.SentencePieceProcessor()
        try:
            sp.Load(path)
        except Exception as e:                       # noqa: BLE001
            log(f"  {os.path.basename(name):40s} load failed: {str(e)[:60]}")
            continue
        uid = sp.unk_id()
        r = unk_rate(sp, uid, sample)
        log(f"  {os.path.basename(name):40s} vocab={sp.GetPieceSize():>6} "
            f"unk={r*100:6.2f}%")
        if best is None or r < best[0]:
            best = (r, sp, uid, name, sp.GetPieceSize())

    if best is None:
        log("[skip] no loadable SentencePiece model")
        return

    _, sp, unk_id, name, vocab = best
    log(f"selected sub-tokenizer: {os.path.basename(name)} "
        f"(vocab={vocab}, <unk> id={unk_id})")

    total_tokens = 0
    unk_tokens = 0
    unk_words = Counter()

    for e in entries:
        text = e["text"]
        if not text:
            continue
        ids = sp.EncodeAsIds(text)
        total_tokens += len(ids)
        unk_tokens += sum(1 for i in ids if i == unk_id)
        # Attribute <unk> back to source words.
        for w in text.split():
            if unk_id in sp.EncodeAsIds(w):
                unk_words[w] += 1

    if total_tokens == 0:
        log("[skip] no tokens produced")
        return

    pct = 100.0 * unk_tokens / total_tokens
    log(f"total tokens      : {total_tokens}")
    log(f"<unk> tokens      : {unk_tokens} ({pct:.3f}%)")
    log("top 20 source words producing <unk>:")
    for w, n in unk_words.most_common(20):
        log(f"  {n:6d}  {w}")

    if pct > 5.0:
        log(f"\n[WARNING] <unk> rate {pct:.2f}% > 5% — vocab may need extending.")
    else:
        log(f"\n[ok] <unk> rate {pct:.2f}% <= 5%.")


# --- Dry-run -----------------------------------------------------------------
def do_dry_run(splits):
    log("=== DRY RUN — no audio converted, no files written ===\n")
    log("filter thresholds:")
    log(f"  duration: keep {MIN_DUR}s <= d <= {MAX_DUR}s")
    log(f"  drop metadata Other in : {sorted(DROP_FLAGS)}")
    log(f"  keep+tag noisy Other in: {sorted(KEEP_NOISY)}")
    log(f"  keep code-switched (Latin) letters (not stripped)")
    log(f"  strip punctuation chars: {PUNCT_CHARS!r} + zero-width")
    log(f"  drop #incomplete-marked utts (override: --keep-incomplete)\n")

    # 5 sample ffmpeg commands from train.
    split = "train" if "train" in splits else splits[0]
    text_file = os.path.join(RAW_ROOT, SPLITS[split]["dir"], "text")
    texts = read_text(text_file)
    sample_uids = list(texts.keys())[:5]
    log(f"5 sample ffmpeg commands ({split}):")
    for uid in sample_uids:
        cmd = ffmpeg_cmd(mp3_path(split, uid), wav_path(split, uid))
        log("  " + " ".join(cmd))

    # 3 sample manifest lines built from the text file (duration unknown here).
    meta_file = os.path.join(META_DIR, SPLITS[split]["meta"])
    meta = read_metadata(meta_file)
    log(f"\n3 sample manifest lines (normalized text, duration TBD after convert):")
    for uid in list(texts.keys())[:3]:
        clean, _ = normalize_text(texts[uid])
        e = {
            "audio_filepath": wav_path(split, uid),
            "duration": None,
            "text": clean,
            "speaker": speaker_of(uid),
            "lang": "hi",
            "quality": meta.get(uid, {}).get("Other", "NA"),
        }
        log("  " + json.dumps(e, ensure_ascii=False))

    # Show normalization on a few artifact-bearing samples.
    log("\nnormalization examples (raw -> clean):")
    shown = 0
    for uid, raw in texts.items():
        clean, inc = normalize_text(raw)
        if clean != raw:
            tag = " [DROP: incomplete]" if inc else ""
            log(f"  RAW : {raw[:90]}")
            log(f"  NORM: {clean[:90]}{tag}")
            shown += 1
            if shown >= 3:
                break


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned actions; convert nothing, write nothing")
    ap.add_argument("--workers", type=int, default=8,
                    help="ThreadPoolExecutor workers for ffmpeg + probing")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "eval"],
                    choices=["train", "dev", "eval"],
                    help="which splits to process")
    ap.add_argument("--keep-incomplete", action="store_true",
                    help="keep #incomplete-marked utts (strip marker) instead of "
                         "dropping them; default drops (audio/text misalignment)")
    ap.add_argument("--skip-tokenizer-check", action="store_true",
                    help="skip Step 6 (loading the pretrained model can be heavy)")
    ap.add_argument("--only-tokenizer-check", action="store_true",
                    help="skip Steps 1-5; run Step 6 against the existing "
                         "/data/manifests/train.json")
    args = ap.parse_args()

    for s in args.splits:
        if s not in SPLITS:
            log(f"[FAIL] unknown split: {s}")
            sys.exit(1)

    if args.dry_run:
        do_dry_run(args.splits)
        return

    if args.only_tokenizer_check:
        train_manifest = os.path.join(MANIFEST_DIR, "train.json")
        if not os.path.exists(train_manifest):
            log(f"[FAIL] {train_manifest} not found — run the full build first")
            sys.exit(1)
        entries = []
        with open(train_manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        log(f"loaded {len(entries)} entries from {train_manifest}")
        tokenizer_unk_check(entries)
        return

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join("/data/exp/logs", "x")), exist_ok=True)

    train_entries = None
    for split in args.splits:
        log(f"\n########## SPLIT: {split} ##########")
        text_file = os.path.join(RAW_ROOT, SPLITS[split]["dir"], "text")
        meta_file = os.path.join(META_DIR, SPLITS[split]["meta"])

        texts = read_text(text_file)
        meta = read_metadata(meta_file)
        log(f"[{split}] transcripts: {len(texts)} | metadata rows: {len(meta)}")

        # Step 1: convert
        uttids = list(texts.keys())
        convert_split(split, uttids, args.workers, dry_run=False)

        # Step 3 prep: durations from converted WAVs
        durs = probe_durations(split, uttids, args.workers)

        # Step 3/4: filter + normalize + build
        entries, drops, norm = build_entries(
            split, texts, meta, durs, keep_incomplete=args.keep_incomplete)

        # Step 4: write manifest
        out_path = os.path.join(MANIFEST_DIR, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        log(f"[{split}] wrote {len(entries)} entries -> {out_path}")

        # Step 5: summary
        print_summary(split, entries, drops, norm)

        if split == "train":
            train_entries = entries

    # Step 6: tokenizer <unk> check on train
    if train_entries is not None and not args.skip_tokenizer_check:
        tokenizer_unk_check(train_entries)


if __name__ == "__main__":
    main()
