# Gram Vaani Manifest Build — Session Runbook

Chronological record of building NeMo manifests from OpenSLR 118 (Gram Vaani) for
the IndicConformer Hindi telephony fine-tune. Every command, every failure, every
fix. Companion to [gramvaani_data_distribution.md](gramvaani_data_distribution.md).

- **Host:** `e2e-60-249` (bare metal, L40S 45 GB, 123 GB free on /dev/vda2)
- **Container:** `nvcr.io/nvidia/nemo:25.02`, hostname `742eaa1c22c3`
- **Repo (host):** `~/CredResolve_Production_grade_Streaming_ASR`
- **Repo (container):** `/workspace/repo`  ·  **Data:** `/data` (shared mount)
- **Date:** 2026-07-10

> ⚠️ **Host vs container.** `/data` is mounted in both, but Python deps
> (`soundfile`, `mutagen`, `sentencepiece`, `huggingface_hub`, NeMo) live **only
> in the container**. `/workspace/repo` exists **only in the container**. Two
> commands in this session failed purely because they ran on the host by mistake
> — always confirm the shell prompt (`root@742eaa1c22c3` = container).

---

## 0. Container launch

```bash
docker run --gpus all -it \
  --name nemo-ft \
  --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data:/data \
  -v ~/CredResolve_Production_grade_Streaming_ASR:/workspace/repo \
  -w /workspace/repo \
  nvcr.io/nvidia/nemo:25.02 bash
```

Re-enter / restart later:

```bash
docker exec -it nemo-ft bash      # if still running
docker start -ai nemo-ft          # if exited
docker ps -a                      # check status
```

Benign startup warning (ignore): `groups: cannot find name for group ID 993`
— a host GID with no matching entry in the container's `/etc/group`.

---

## 1. Install mutagen

```bash
pip install mutagen --break-system-packages
```

`mutagen` = pure-Python audio metadata (MP3 probing). `--break-system-packages`
is harmless in this container (not strictly needed). Ephemeral — reinstall after
any fresh `docker run` unless baked into an image.

---

## 2. The build script

Written to `scripts/02_build_manifest.py` (git-tracked, maps to
`/workspace/repo/scripts/`). Pipeline:

1. MP3 → 16 kHz mono WAV (ffmpeg, `ThreadPoolExecutor` max_workers=8, idempotent
   via `.tmp.wav` + atomic `os.replace`).
2. Parse Kaldi `text` + metadata TSV, join on uttid.
3. Filter (duration + metadata flags) **and normalize text**.
4. Write NeMo JSONL to `/data/manifests/{train,dev,eval}.json`.
5. Per-split summary.
6. Tokenizer `<unk>` check against the pretrained SPE.

### Flags

| Flag | Effect |
|------|--------|
| `--dry-run` | print plan (5 ffmpeg cmds, thresholds, sample lines, normalization examples); write nothing |
| `--workers N` | ffmpeg + probe threads (default 8) |
| `--splits ...` | subset of `train dev eval` |
| `--keep-incomplete` | keep `#incomplete` utts (strip marker) instead of dropping |
| `--skip-tokenizer-check` | skip Step 6 |
| `--only-tokenizer-check` | run **only** Step 6 against existing `train.json` |

---

## 3. Commands run (in order)

```bash
# Dry-run sanity check
python3 scripts/02_build_manifest.py --dry-run

# Full build, tee to log (project convention)
mkdir -p /data/exp/logs
python3 scripts/02_build_manifest.py 2>&1 | tee -a /data/exp/logs/02_manifest.log

# Re-run just the tokenizer check (after fixes)
python3 scripts/02_build_manifest.py --only-tokenizer-check 2>&1 \
  | tee -a /data/exp/logs/02_manifest.log
```

Always `cd /workspace/repo` first (or use the absolute script path).

---

## 4. Problems hit & fixes (the interesting part)

### 4.1 Wrong working directory → `No such file or directory`
```
python3: can't open file '/data/raw/gramvaani/scripts/02_build_manifest.py'
```
**Cause:** shell was `cd`'d into `/data/raw/gramvaani`; relative `scripts/...`
didn't resolve. **Fix:** `cd /workspace/repo` or use the absolute path.

### 4.2 Gated HuggingFace repo → 401
```
401 Client Error ... Cannot access gated repo ... Access to model
ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large is restricted.
```
**Cause:** the SPE tokenizer lives in a gated `.nemo`. Token was in `.env` as
`HUGGINGFACE_HUB_TOKEN` — a **non-standard var name** `huggingface_hub` does NOT
auto-detect (it checks `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`), and `.env` isn't
auto-loaded. **Fix:** added `load_hf_token()` — stdlib `.env` parser that reads
the token and re-exports it as `HF_TOKEN` + calls `login()` before download.

### 4.3 NeMo can't instantiate the checkpoint → `KeyError: 'dir'`
```
_setup_monolingual_tokenizer -> self.tokenizer_cfg.pop('dir')  KeyError: 'dir'
... Can't instantiate abstract class ASRModel ...
```
**Cause:** NeMo 25.02's monolingual-tokenizer loader assumes a `dir` key this
IndicConformer checkpoint's config lacks. **Fix:** stopped loading the model
entirely. A `.nemo` is a tar archive — added `extract_spe_from_nemo()` to
download it (`hf_hub_download`), extract the SentencePiece `.model`, and load it
directly with `sentencepiece`. No 600 M model instantiation needed.

### 4.4 Wrong sub-tokenizer picked → exactly 50% `<unk>`
```
selected: 03571f..._tokenizer.model  vocab=256  <unk>=50.000%
top <unk> words: के, है, की, में ...  (the commonest Hindi words!)
```
**Cause:** the `.nemo` bundles **multiple** hash-named per-language SPE
sub-tokenizers (IndicConformer is multilingual). The first-match grab pulled a
**non-Hindi** one, so all Devanagari fell back to `<unk>`. The *exactly* 50% and
"commonest words are unk" were the tells. **Fix:** extract **all** `.model`
files, score each on 2 000 real transcripts, auto-select the lowest-`<unk>` one.

### 4.5 Ran Step 6 on the host by mistake
```
[warn] huggingface_hub.login failed (No module named 'huggingface_hub')
bash: cd: /workspace/repo: No such file or directory
```
**Cause:** prompt was `root@e2e-60-249` (host), not the container. Host Python
has none of the ML deps. **Fix:** re-enter the container (`docker exec -it
nemo-ft bash`) and re-run.

### 4.6 Transcript artifacts poison the targets → text normalization added
The corrected tokenizer check (0.11% `<unk>`) exposed **annotation garbage** in
the transcripts, not a vocab gap. The 256-piece SPE has **no punctuation
pieces**, so these all become `<unk>` *training targets*:

| Artifact | Train utts | Example |
|----------|-----------:|---------|
| `#incomplete` (+misspellings, glued forms) | 416 (1.13%) | `#incomplete मिशन सोसाइटी ...` |
| `['word', 'word']` (stringified Python lists) | 143 (0.39%) | `['मोबाइल', 'मिडिया'] रिपोर्टर` |
| apostrophes `'` | 608 (1.65%) | `महत्वपूर्ण' हिसाव 'मधुबनी` |
| other punct `, . : - \` + zero-width | — | — |

**Fix:** added `normalize_text()` (Step 3):
- strip `#…` Latin hashtags (`#incomplete` etc), **preserving glued Hindi**
  (`#incompleteदिया` → `दिया`);
- strip punctuation `' " [ ] ( ) , . : ; ! ? / \ -` + zero-width
  (U+200B/C/D, U+FEFF) → space; collapse whitespace;
- **keep Latin code-switch** (per non-negotiable);
- **drop** `#incomplete`-marked utts by default (partial transcript over full
  audio = misalignment, worse for CTC/RNNT). Override with `--keep-incomplete`.

---

## 5. Findings

### 5.1 Data distribution
Full tables in [gramvaani_data_distribution.md](gramvaani_data_distribution.md).
Headline: train 37,152 utts / 20,266 speakers / ~100 h; all 8 kHz MP3; ~10 s
mean; 60% clean / 22% inaudible / 9% audio_jump; 3.3% code-switch.

### 5.2 ⚠️ Official splits are NOT speaker-disjoint
Overlap of `<district>-<speaker>` keys: **train∩dev = 1,338**, **train∩eval =
758**, dev∩eval = 79. Violates the CLAUDE.md non-negotiable; official-split WER
is optimistically biased. **Must re-partition speaker-disjointly.** Eval also has
**zero** code-switch utts → code-switch WER unobservable there.

### 5.3 Tokenizer
Correct Hindi sub-tokenizer: **vocab 256, `<unk>` id 0, 0.11% `<unk>`** →
**no vocab extension needed.** Residual `<unk>` is isolated Latin capitals
(`A`, `M`, `N`) from code-switch — expected and left as-is.

### 5.4 Build result (pre-normalization run)
`train.json`: **36,797** entries (37,152 − 355 duration drops). Post-normalization
rebuild expected to drop a further ~416 `#incomplete` utts → ~36,381.

---

## 6. Current state & what to run next

```bash
cd /workspace/repo

# Regenerate manifests WITH text normalization (conversion is cached; fast).
python3 scripts/02_build_manifest.py 2>&1 | tee -a /data/exp/logs/02_manifest.log
```

| Item | Status |
|------|--------|
| MP3→16k WAV conversion (37,152) | ✅ done, idempotent |
| Manifests (`/data/manifests/*.json`) | ⚠️ rebuild for normalized text |
| Tokenizer `<unk>` check | ✅ 0.11%, no action |
| Text normalization | ✅ added + dry-run validated |
| **Speaker-disjoint re-split** | ❌ **TODO — blocker for trustworthy metrics** |

### Outstanding
1. **`scripts/03_speaker_disjoint_split.py`** — pool utts, group by speaker key,
   reassign whole speakers into disjoint train/dev/eval (~100/5/3 h), assert zero
   overlap. **Highest priority** (non-negotiable).
2. Decide whether to keep the AMR-NB held-out `dev_codec` path (per CLAUDE.md)
   once splits are fixed.

---

## 7. Artifact / path reference

| Path | What |
|------|------|
| `/data/raw/gramvaani/GV_{Train_100h,Dev_5h,Eval_3h}/` | raw MP3 + `text` + `utt2labels` |
| `/data/raw/gramvaani/Metadata/utt2labels_GV_*.txt` | metadata TSVs (`Other` col) |
| `/data/wav16k/{train,dev,eval}/` | converted 16 kHz mono WAVs |
| `/data/manifests/{train,dev,eval}.json` | NeMo JSONL manifests |
| `/data/exp/logs/02_manifest.log` | build log (tee target) |
| `scripts/02_build_manifest.py` | the builder |
| `~/.cache/huggingface/` (`HF_HOME`) | cached gated `.nemo` |

Manifest schema:
```json
{"audio_filepath": "/data/wav16k/train/01-00003-02.wav", "duration": 10.234,
 "text": "...", "speaker": "01-00003", "lang": "hi", "quality": "inaudible"}
```
