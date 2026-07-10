"""Probe Gram Vaani (OpenSLR 118) before building manifests."""
import collections, csv, re
from pathlib import Path

BASE = Path("/data/raw/gramvaani")
SPLITS = {
    "train": BASE / "GV_Train_100h",
    "dev":   BASE / "GV_Dev_5h",
    "eval":  BASE / "GV_Eval_3h",
}
META = {
    "train": BASE / "Metadata/utt2labels_GV_Train_100h.txt",
    "dev":   BASE / "Metadata/utt2labels_GV_Dev_5h.txt",
    "eval":  BASE / "Metadata/utt2labels_GV_Eval_3h.txt",
}

# need pydub or ffprobe for mp3 duration/samplerate
# try mutagen first — header-only, fast
try:
    from mutagen.mp3 import MP3
    def probe(p):
        m = MP3(str(p))
        return m.info.sample_rate, m.info.length
    print("[ok] using mutagen for mp3 probing\n")
except ImportError:
    import subprocess, json
    def probe(p):
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(p)],
            capture_output=True, text=True)
        s = json.loads(r.stdout)["streams"][0]
        return int(s["sample_rate"]), float(s["duration"])
    print("[ok] using ffprobe for mp3 probing (slower)\n")

for name, split_dir in SPLITS.items():
    mp3s = sorted(split_dir.glob("Audio/*.mp3"))
    text_file = split_dir / "text"

    # --- transcripts ---
    utt2txt = {}
    if text_file.exists():
        for ln in text_file.read_text("utf-8").strip().split("\n"):
            parts = ln.split(maxsplit=1)
            if len(parts) == 2:
                utt2txt[parts[0]] = parts[1]

    inaudible = sum("<inaudible>" in v for v in utt2txt.values())
    empty = sum(len(v.strip()) == 0 for v in utt2txt.values())
    latin = sum(bool(re.search(r'[A-Za-z]', v)) for v in utt2txt.values())
    digits = sum(bool(re.search(r'[0-9०-९]', v)) for v in utt2txt.values())

    # --- speakers ---
    spk = collections.Counter(u.rsplit("-", 1)[0] for u in utt2txt)

    # --- audio properties (sample first 500 for speed) ---
    rates = collections.Counter()
    durs = []
    for i, p in enumerate(mp3s):
        try:
            sr, dur = probe(p)
            if i < 500:
                rates[sr] += 1
            durs.append(dur)
        except Exception as e:
            if i < 5:
                print(f"  [warn] {p.name}: {e}")
    total_h = sum(durs) / 3600
    durs.sort()
    pct = lambda p: durs[int(len(durs) * p)] if durs else 0

    print(f"=== {name} ===")
    print(f"  mp3 files    : {len(mp3s)}")
    print(f"  transcripts  : {len(utt2txt)}")
    print(f"  orphan mp3   : {len(mp3s) - len(utt2txt)}  (mp3 without transcript)")
    print(f"  total hours  : {total_h:.2f}")
    print(f"  sample rates : {dict(rates)}  (first 500)")
    print(f"  dur p05/p25/p50/p75/p95/max : "
          f"{pct(.05):.1f} / {pct(.25):.1f} / {pct(.50):.1f} / "
          f"{pct(.75):.1f} / {pct(.95):.1f} / {durs[-1]:.1f}")
    print(f"  mean dur     : {sum(durs)/len(durs):.1f}s")
    print(f"  <0.5s : {sum(d<0.5 for d in durs)}   >20s : {sum(d>20 for d in durs)}   >60s : {sum(d>60 for d in durs)}")
    print(f"  speakers     : {len(spk)}")
    if spk:
        sv = sorted(spk.values())
        print(f"  utts/speaker : min {sv[0]}, med {sv[len(sv)//2]}, max {sv[-1]}")
    print(f"  <inaudible>  : {inaudible}  ({100*inaudible/max(len(utt2txt),1):.1f}%)")
    print(f"  empty text   : {empty}")
    print(f"  has latin    : {latin}   has digits : {digits}")

    # duration buckets
    edges = [0, 2, 4, 6, 8, 11, 15, 20, 60, 999]
    print(f"  duration buckets:")
    for lo, hi in zip(edges, edges[1:]):
        n = sum(lo <= d < hi for d in durs)
        print(f"    [{lo:>3},{hi:>3}) : {n:>6}  ({100*n/max(len(durs),1):5.1f}%)")
    print()

# --- metadata analysis ---
print("=== METADATA ===")
for name, mpath in META.items():
    if not mpath.exists():
        print(f"  {name}: NOT FOUND"); continue
    rows = list(csv.DictReader(mpath.open("r", encoding="utf-8"), delimiter="\t"))
    print(f"\n  --- {name} ({len(rows)} rows) ---")
    for col in ["Accent", "Gender", "Background", "Sentiment", "District", "State", "Other"]:
        vals = collections.Counter(r.get(col, "NA") for r in rows)
        na = vals.get("NA", 0)
        print(f"  {col:12s}: {len(vals)} unique, NA={na} ({100*na/max(len(rows),1):.0f}%)"
              f"  top3: {vals.most_common(3)}")

print("\n--- DECISIONS ---")
print("1. If >10% <inaudible> in train: strip tag vs drop utterance")
print("2. If >30s utterances exist: segment or drop")
print("3. If metadata Background is populated: use as quality filter")
print("4. Speaker key = uttid.rsplit('-',1)[0] for disjoint splits")
