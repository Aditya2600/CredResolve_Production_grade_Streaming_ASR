# IndicConformer 600M Telephony Fine-tune — Full Project Runbook

Complete chronological record of every command, decision, pivot, failure, and fix.
From bare metal to NeMo manifests. Nothing omitted.

- **Host:** `e2e-60-249` (bare metal, NVIDIA L40S 45 GB VRAM, 233 GB disk)
- **Container:** `nvcr.io/nvidia/nemo:25.02`, hostname `742eaa1c22c3`
- **GPU:** NVIDIA L40S, 46068 MiB, Ada Lovelace (sm_89), Driver 570.133.20, CUDA 12.8
- **Repo (host):** `~/CredResolve_Production_grade_Streaming_ASR`
- **Repo (container):** `/workspace/repo` · **Data:** `/data` (shared mount)
- **Date started:** 2026-07-10
- **Model:** `ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large` (600M params)

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Hardware Reconnaissance](#2-hardware-reconnaissance)
3. [VRAM & Disk Budget](#3-vram--disk-budget)
4. [Dataset Pivot: MUCS → Gram Vaani](#4-dataset-pivot-mucs--gram-vaani)
5. [Stage 0 — Environment Setup](#5-stage-0--environment-setup)
6. [Stage 1 — Data Acquisition & Probing](#6-stage-1--data-acquisition--probing)
7. [Stage 2 — Manifest Building](#7-stage-2--manifest-building)
8. [Current State & Next Steps](#8-current-state--next-steps)
9. [Artifact / Path Reference](#9-artifact--path-reference)
10. [Architecture Decisions Log](#10-architecture-decisions-log)

---

## 1. Project Goal

Fine-tune the IndicConformer 600M (hybrid CTC+RNNT) for Hindi telephony/call-center
ASR. The pretrained checkpoint is a multilingual Indic model; we're adapting it to
spontaneous telephone Hindi with real channel artifacts.

**Training pipeline (3-stage):**
1. Supervised fine-tune on OpenSLR 118 Gram Vaani (100h labelled telephony Hindi) ← **current focus**
2. Pseudo-label 1000h unlabelled Gram Vaani with 3-model ensemble (Sarvam Saaras v3, Whisper large-v3, Gemini/GPT-4o), filter by 3-way agreement
3. Gold fine-tune with 25h human-labeled call-center data

**Non-negotiables (in CLAUDE.md):**
- Reuse the pretrained SPE tokenizer. Never rebuild it.
- Hybrid CTC+RNNT. `aux_ctc.ctc_loss_weight = 0.3`. Do not drop the CTC head.
- bf16-mixed only. No fp16.
- Every script takes `--dry-run`.
- Log to `/data/exp/logs/<script>.log`.

---

## 2. Hardware Reconnaissance

### Commands run (on host, before Docker)

```bash
df -h
```
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda2       233G   18G  216G   8% /
```

```bash
nvidia-smi
```
```
NVIDIA L40S | 46068MiB | Driver 570.133.20 | CUDA 12.8
0MiB used, 0% GPU-Util
```

### Assessment
- **VRAM:** 45 GB usable. Can do bs=16 at 20s max_duration, or batch_duration=360 with Lhotse bucketing.
- **Disk:** 216 GB free initially. NeMo container eats ~80 GB from `/var/lib/docker` on the same partition → effective ~130 GB free after pull.
- **Shared memory:** `/dev/shm` = 101 GB on host, capped to 64 GB inside container via `--shm-size=64g`.

---

## 3. VRAM & Disk Budget

### VRAM breakdown (600M params, bf16 AMP + AdamW)

| Component | Bytes/param | Total |
|-----------|-------------|-------|
| fp32 master weights | 4 | 2.4 GB |
| bf16 working copy | 2 | 1.2 GB |
| Gradients (fp32) | 4 | 2.4 GB |
| Adam m + v | 8 | 4.8 GB |
| **Static subtotal** | | **~10.8 GB** |

Activation memory per sample at 20s: ~1.2–1.4 GB (encoder ~1.0 GB + RNNT joint ~0.2–0.4 GB).

The RNNT joint tensor is the bottleneck: `B × T × U × V` where T=500 (encoder steps after 4× subsampling), U≈50 (label steps), V≈1024 (SPE vocab). Mitigated by `joint.fused_batch_size: 4`.

### Actual Gram Vaani data characteristics change the budget
- Mean duration 9.9s (not 20s) → activations much cheaper
- All 8 kHz → after 16 kHz upsample, mel frames are half what studio 16 kHz would produce in information content
- L40S 45 GB is comfortable for `batch_duration: 360`

### Disk budget (100h Hindi-only, post-MUCS deletion)

| Item | Size |
|------|------|
| Gram Vaani tarballs | 2.2 GB |
| Extracted MP3s | ~2.2 GB |
| Converted 16k WAVs | ~8 GB |
| MUSAN + RIR (if needed) | ~16 GB |
| Augmented copies (3×, speed perturb only) | ~24 GB |
| Checkpoints (save_top_k=3 + last) | ~29 GB |
| Scratch | ~10 GB |
| **Total** | **~92 GB** |

Available after container: ~123 GB. Fits.

---

## 4. Dataset Pivot: MUCS → Gram Vaani

### Original plan: OpenSLR 103 (MUCS)
MUCS Hindi = 95h read speech from storybooks. Clean, studio-quality, slow articulation. Zero telephony artifacts, zero spontaneous speech.

### What happened

Downloaded MUCS Hindi (4.2 GB train + 247 MB test), extracted to `/data/raw/hi/`:
```bash
cd /data/raw
wget -c "https://www.openslr.org/resources/103/Hindi_train.tar.gz"
wget -c "https://www.openslr.org/resources/103/Hindi_test.tar.gz"
tar xzf Hindi_train.tar.gz -C hi/
tar xzf Hindi_test.tar.gz  -C hi/
```
103,768 WAVs, Kaldi-style transcripts. Speaker ID = uttid prefix (e.g. `0001_030` → speaker `0001`).

Started downloading Marathi (partial, 2.45 GB of 4.1 GB), cancelled with Ctrl+C.

### Why we pivoted
MUCS is read storybook speech — the **opposite** of call-center audio. Codec simulation can add channel artifacts but NOT spontaneous disfluency, overlapping turns, dialect variation, background noise, or code-switching.

### Discovery: OpenSLR 118 (Gram Vaani)
Spontaneous telephone speech from Gram Vaani's Mobile Vaani platform. 100h labelled + 1000h unlabelled. Real telephony: 8 kHz MP3, channel distortions, clipping, speech truncation, audio jumps. Regional Hindi dialects from across India (heavy Bihar/Jharkhand). Crowd-sourced transcriptions with quality metadata.

SandLogic used this dataset to fine-tune a 769M ASR model and achieved ~21% relative WER reduction on LAHAJA benchmark, ~55% WER reduction for a healthcare client.

### Cleanup commands
```bash
# Rename MUCS for potential OOD eval use
mv /data/raw/hi /data/raw/mucs_hi

# Delete partial Marathi download
rm -f /data/raw/Marathi_train.tar.gz

# Later: decided to delete MUCS entirely to free 8 GB
rm -rf /data/raw/mucs_hi /data/raw/Hindi_train.tar.gz /data/raw/Hindi_test.tar.gz
rm -rf /data/raw/mr /data/raw/or
```

### Impact on pipeline
| Aspect | MUCS plan | Gram Vaani plan |
|--------|-----------|-----------------|
| Augmentation | Heavy codec sim (G.711, GSM, AMR-NB, packet loss) | Light: speed perturb + MUSAN noise only. Data already has real telephony artifacts. |
| Sample rate | Unknown (probe needed) | All 8 kHz. Upsample to 16 kHz in dataloader. |
| Text normalization | Clean storybook text | Noisy crowd transcriptions. Filter by quality metadata, heavier normalization. |
| Dev set design | Held-out codec (AMR-NB) | Held-out by speaker (official splits have speaker overlap — must re-split) |
| Pseudo-label corpus | Didn't exist | 1000h unlabelled ships with dataset |

---

## 5. Stage 0 — Environment Setup

### 5.1 Docker + NVIDIA Container Toolkit verification

```bash
docker --version
```
```
Docker version 29.6.1, build 8900f1d
```

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```
→ Printed L40S. Container GPU access confirmed. NVIDIA Container Toolkit already installed — skipped manual install.

### 5.2 Directory scaffold

```bash
sudo mkdir -p /data/{raw/{hi,mr,or},musan,rirs,manifests,shards,exp/logs}
sudo mkdir -p /data/scripts /data/configs
sudo chown -R $USER:$USER /data
```
```bash
df -h /data
```
```
/dev/vda2       233G   19G  214G   9% /
```

### 5.3 Pull NeMo container

```bash
docker pull nvcr.io/nvidia/nemo:25.02
```
~25 GB. No NGC login required (public pull succeeded).

### 5.4 Launch container

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

Key flags:
- `--name nemo-ft` → persistent container, reconnect with `docker start -ai nemo-ft`
- `--shm-size=64g` → prevents dataloader shared-memory crash (Docker defaults to 64 MB)
- `-v /data:/data` → data volume shared between host and container
- `-v ~/CredResolve...:/workspace/repo` → git repo mounted read-write

Benign warning: `groups: cannot find name for group ID 993` (host GID without container entry).

### 5.5 In-container verification

```bash
df -h /dev/shm
python -c "import torch,nemo; print(torch.__version__, torch.version.cuda, nemo.__version__)"
```
Confirmed `/dev/shm` = 64G, torch + NeMo importable.

### 5.6 Verification script

```bash
mkdir -p /workspace/repo/scripts

cat > /workspace/repo/scripts/00_verify_env.py <<'EOF'
# ... [full script as documented in conversation]
# Tests: CUDA, bf16, NeMo import, RNNT loss fwd+bwd, numba CUDA, /dev/shm, ffmpeg codecs
EOF

python /workspace/repo/scripts/00_verify_env.py
```

Output:
```
[ok] GPU        : NVIDIA L40S
[ok] VRAM       : 44.4 GiB
[ok] Capability : sm_89
[ok] torch      : <version>
[ok] torch CUDA : 12.8
[ok] bf16       : supported
[ok] NeMo       : <version>
[ok] RNNT loss  : 41.xxxx, grads finite
[ok] numba cuda : available
[ok] /dev/shm   : 64G
[ok] ffmpeg     : pcm_mulaw
[ok] ffmpeg     : libgsm
[ok] ffmpeg     : libopencore_amrnb

=== Stage 0 PASSED ===
```

**The RNNT loss fwd+bwd is the critical check.** `import warprnnt_numba` succeeds even on broken installs — the Numba kernel only gets JIT-compiled against the actual driver on first invocation.

### 5.7 Snapshot

```bash
# From the HOST
docker commit nemo-ft nemo-ft:stage0
```

---

## 6. Stage 1 — Data Acquisition & Probing

### 6.1 Gram Vaani download

Initial attempt with guessed URLs failed (404):
```bash
wget -c https://www.openslr.org/resources/118/train_labeled.tar.gz   # 404
wget -c https://www.openslr.org/resources/118/dev_labeled.tar.gz     # 404
```

Correct URLs found from openslr.org/118/:
```bash
cd /data/raw
mkdir -p gramvaani
cd gramvaani

wget -c https://www.openslr.org/resources/118/GV_Train_100h.tar.gz    # 2.0G
wget -c https://www.openslr.org/resources/118/GV_Dev_5h.tar.gz        # 98M
wget -c https://www.openslr.org/resources/118/GV_Eval_3h.tar.gz       # 62M
wget -c https://www.openslr.org/resources/118/Metadata.tar.gz         # 460K
```

Total: ~2.2 GB (vs 13.5 GB for MUCS all three languages).

**Did NOT download:** `GV_Train_1000h_unlabeled.tar.gz` (~30-40 GB). That's the pseudo-labeling corpus for Stage 4. Disk is tight; download later.

### 6.2 Integrity check + extraction

```bash
for f in *.tar.gz; do tar tzf "$f" > /dev/null && echo "ok $f" || echo "CORRUPT $f"; done
```
All passed.

```bash
tar xzf GV_Train_100h.tar.gz
tar xzf GV_Dev_5h.tar.gz
tar xzf GV_Eval_3h.tar.gz
tar xzf Metadata.tar.gz
```

### 6.3 Layout discovered

```
/data/raw/gramvaani/
├── GV_Train_100h/
│   ├── Audio/          (37,152 .mp3 files, flat directory)
│   └── text            (Kaldi-style: <uttid> <transcript>)
├── GV_Dev_5h/
│   ├── Audio/          (1,885 .mp3 files)
│   └── text
├── GV_Eval_3h/
│   ├── Audio/          (1,032 .mp3 files)
│   └── text
└── Metadata/
    ├── utt2labels_GV_Train_100h.txt   (TSV)
    ├── utt2labels_GV_Dev_5h.txt
    ├── utt2labels_GV_Eval_3h.txt
    └── utt2labels_GV_Train_1000h.txt  (for unlabelled set, unused)
```

Utterance ID format: `<district>-<speaker>-<segment>` (e.g. `02-22287-01`).
Speaker key: `uttid.rsplit("-", 1)[0]` → `02-22287`.

Transcript format (Kaldi-style):
```
01-00003-02 इस मामले में कोर्ट द्वारा निर्देश दिया गया है ...
01-00003-03 आदिवासी प्रदेश कार्यकाल में सदस्य रहे ...
```

Metadata TSV columns: `Uttids | Accent | Age | Gender | Background | Sentiment | District | State | Other`

### 6.4 Corpus probe

```bash
pip install mutagen --break-system-packages

cat > /workspace/repo/scripts/01_probe_gramvaani.py <<'PYEOF'
# ... [full probe script as in conversation]
PYEOF

mkdir -p /data/exp/logs
python /workspace/repo/scripts/01_probe_gramvaani.py 2>&1 | tee /data/exp/logs/01_probe_gv.log
```

### 6.5 Probe results — full

#### Audio properties

| Property | Train | Dev | Eval |
|----------|-------|-----|------|
| MP3 files | 37,152 | 1,885 | 1,032 |
| Transcripts | 37,152 | 1,885 | 1,032 |
| Orphan MP3 | 0 | 0 | 0 |
| Total hours | 102.18 | 5.08 | 2.83 |
| Sample rate | 8000 (100%) | 8000 (100%) | 8000 (93%), 44100 (5.6%), 48000 (1.2%), 16000 (0.2%) |
| Mean duration | 9.9s | 9.7s | 9.9s |
| p05/p50/p95 | 4.4/10.4/12.1 | 4.2/10.3/12.1 | 4.4/10.5/12.2 |
| Max duration | 141.0s | 29.6s | 33.7s |
| <0.5s | 0 | 0 | 0 |
| >20s | 363 | 12 | 12 |
| >60s | 13 | 0 | 0 |
| Speakers | 20,266 | 1,826 | 1,011 |
| Utts/speaker (median) | 2 | 1 | 1 |
| <inaudible> in text | 0 (0%) | 14 (0.7%) | 0 (0%) |
| Empty text | 0 | 0 | 0 |
| Has Latin chars | 1,228 (3.3%) | 141 | 0 |

#### Duration distribution (train)

| Bucket | Count | % |
|--------|------:|---:|
| [0, 2) | 191 | 0.5% |
| [2, 4) | 1,296 | 3.5% |
| [4, 6) | 2,156 | 5.8% |
| [6, 8) | 3,889 | 10.5% |
| [8, 11) | 16,188 | 43.6% |
| [11, 15) | 12,607 | 33.9% |
| [15, 20) | 462 | 1.2% |
| [20, 60) | 350 | 0.9% |
| [60, 999) | 13 | 0.0% |

77.5% of mass in [8, 15). Very tight distribution.

#### Metadata — `Other` column (the quality signal)

| Flag | Train count | % |
|------|------------:|---:|
| NA (clean) | 22,454 | 60% |
| inaudible | 8,300 | 22% |
| audio_jump | 3,227 | 9% |
| remaining | ~3,171 | 9% |

`Background` column is 90% NA — useless. `Other` is the only quality filter.

---

## 7. Stage 2 — Manifest Building

### 7.1 Build script

Script: `/workspace/repo/scripts/02_build_manifest.py`

Pipeline:
1. MP3 → 16 kHz mono WAV (ffmpeg, `ThreadPoolExecutor` max_workers=8, idempotent via `.tmp.wav` + atomic `os.replace`)
2. Parse Kaldi `text` + metadata TSV, join on uttid
3. Filter (duration 0.5–20s + metadata flags) + normalize text
4. Write NeMo JSONL to `/data/manifests/{train,dev,eval}.json`
5. Per-split summary
6. Tokenizer `<unk>` check against pretrained SPE

Flags: `--dry-run`, `--workers N`, `--splits ...`, `--keep-incomplete`, `--skip-tokenizer-check`, `--only-tokenizer-check`

### 7.2 Commands run

```bash
cd /workspace/repo

# Dry-run sanity check
python3 scripts/02_build_manifest.py --dry-run

# Full build
mkdir -p /data/exp/logs
python3 scripts/02_build_manifest.py 2>&1 | tee -a /data/exp/logs/02_manifest.log

# Re-run tokenizer check after fixes
python3 scripts/02_build_manifest.py --only-tokenizer-check 2>&1 \
  | tee -a /data/exp/logs/02_manifest.log
```

### 7.3 Problems hit & fixes

#### 7.3.1 Wrong working directory → `No such file or directory`
```
python3: can't open file '/data/raw/gramvaani/scripts/02_build_manifest.py'
```
**Cause:** Shell was `cd`'d into `/data/raw/gramvaani`; relative `scripts/...` didn't resolve.
**Fix:** `cd /workspace/repo` or use absolute path. Also: host vs container confusion — always confirm prompt is `root@742eaa1c22c3` (container), not `root@e2e-60-249` (host).

#### 7.3.2 Gated HuggingFace repo → 401
```
401 Client Error ... Cannot access gated repo ...
ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large is restricted.
```
**Cause:** SPE tokenizer is inside a gated `.nemo`. Token was in `.env` as `HUGGINGFACE_HUB_TOKEN` — non-standard var name. `huggingface_hub` checks `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, and `.env` isn't auto-loaded.
**Fix:** Added `load_hf_token()` — stdlib `.env` parser that reads token and re-exports as `HF_TOKEN` + calls `login()`.

#### 7.3.3 NeMo can't instantiate checkpoint → `KeyError: 'dir'`
```
_setup_monolingual_tokenizer -> self.tokenizer_cfg.pop('dir')  KeyError: 'dir'
... Can't instantiate abstract class ASRModel ...
```
**Cause:** NeMo 25.02's monolingual-tokenizer loader assumes a `dir` key the IndicConformer checkpoint config lacks.
**Fix:** Stopped loading the model entirely. A `.nemo` is a tar archive — added `extract_spe_from_nemo()` to download it (`hf_hub_download`), extract the SentencePiece `.model`, load directly with `sentencepiece`. No 600M model instantiation needed for tokenizer check.

#### 7.3.4 Wrong sub-tokenizer picked → exactly 50% `<unk>`
```
selected: 03571f..._tokenizer.model  vocab=256  <unk>=50.000%
top <unk> words: के, है, की, में ... (the commonest Hindi words!)
```
**Cause:** `.nemo` bundles **multiple** hash-named per-language SPE sub-tokenizers (IndicConformer is multilingual). First-match grabbed a non-Hindi one, so all Devanagari → `<unk>`.
**Telltale signs:** exactly 50% unk rate, and the most common Hindi stop words producing unk.
**Fix:** Extract **all** `.model` files, score each on 2,000 real transcripts, auto-select the lowest-`<unk>` one.

#### 7.3.5 Host vs container mix-up
```
[warn] huggingface_hub.login failed (No module named 'huggingface_hub')
bash: cd: /workspace/repo: No such file or directory
```
**Cause:** Ran on host (`root@e2e-60-249`), not container. Host Python lacks ML deps.
**Fix:** `docker exec -it nemo-ft bash` → re-run inside container.

#### 7.3.6 Transcript artifacts → text normalization added
The corrected tokenizer (0.11% `<unk>`) exposed annotation garbage, not vocab gaps. The 256-piece SPE has no punctuation pieces:

| Artifact | Train utts | Example |
|----------|------------|---------|
| `#incomplete` (+ misspellings, glued forms) | 416 (1.13%) | `#incomplete मिशन सोसाइटी ...` |
| `['word', 'word']` (stringified Python lists) | 143 (0.39%) | `['मोबाइल', 'मिडिया'] रिपोर्टर` |
| Apostrophes `'` | 608 (1.65%) | `महत्वपूर्ण' हिसाव 'मधुबनी` |
| Other punct `, . : - \` + zero-width | various | |

**Fix:** Added `normalize_text()` in Step 3:
- Strip `#...` Latin hashtags, preserving glued Hindi (`#incompleteदिया` → `दिया`)
- Strip punctuation `' " [ ] ( ) , . : ; ! ? / \ -` + zero-width (U+200B/C/D, U+FEFF) → space; collapse whitespace
- Keep Latin code-switch (non-negotiable)
- Drop `#incomplete`-marked utts by default (partial transcript over full audio = misalignment, fatal for CTC/RNNT). Override with `--keep-incomplete`.

### 7.4 Tokenizer result

Correct Hindi sub-tokenizer identified by auto-scoring.
- **Vocab size:** 256
- **`<unk>` id:** 0
- **`<unk>` rate:** 0.11%
- **No vocab extension needed.**
- Residual `<unk>`: isolated Latin capitals (`A`, `M`, `N`) from code-switch — expected, left as-is.

### 7.5 Build result

Pre-normalization: `train.json` = 36,797 entries (37,152 − 355 duration drops).
Post-normalization: expected ~36,381 (further ~416 `#incomplete` drops).

Manifest schema:
```json
{"audio_filepath": "/data/wav16k/train/01-00003-02.wav", "duration": 10.234,
 "text": "...", "speaker": "01-00003", "lang": "hi", "quality": "inaudible"}
```

### 7.6 Critical finding: official splits are NOT speaker-disjoint

Speaker key overlap:
- **train ∩ dev = 1,338 speakers**
- **train ∩ eval = 758 speakers**
- dev ∩ eval = 79 speakers

Official-split WER is optimistically biased. **Must re-partition speaker-disjointly.**
Also: eval has **zero** code-switch utterances → code-switch WER unobservable there.

---

## 8. Current State & Next Steps

### Status table

| Item | Status |
|------|--------|
| Docker + GPU + NeMo verified | ✅ Stage 0 passed |
| Container snapshot | ✅ `nemo-ft:stage0` |
| Gram Vaani downloaded + extracted | ✅ 2.2 GB |
| Corpus probed (audio + metadata) | ✅ All 8kHz, 102h, 20k speakers |
| MP3 → 16k WAV conversion (40,069 files) | ✅ Idempotent, cached at `/data/wav16k/` |
| Manifests (`/data/manifests/*.json`) | ⚠️ Rebuild needed for normalized text |
| Text normalization logic | ✅ Added + dry-run validated |
| Tokenizer `<unk>` check | ✅ 0.11%, no action needed |
| **Speaker-disjoint re-split** | ❌ **TODO — blocker for trustworthy metrics** |
| Augmentation (speed perturb + MUSAN) | ❌ Not started |
| Training config | ❌ Not started |
| Training | ❌ Not started |

### Immediate next steps (in order)

1. **Rebuild manifests** with text normalization enabled:
   ```bash
   cd /workspace/repo
   python3 scripts/02_build_manifest.py 2>&1 | tee -a /data/exp/logs/02_manifest.log
   ```

2. **Speaker-disjoint re-split** (`scripts/03_speaker_disjoint_split.py`):
   Pool all utterances, group by speaker key, assign whole speakers into disjoint
   train/dev/eval (~92h / 5h / 3h), assert zero overlap. Stratify by quality tier
   and dialect if possible.

3. **Augmentation** (`scripts/04_augment.py`):
   Speed perturb (0.9/1.0/1.1) + optional MUSAN noise at high SNR (15–25 dB).
   No codec simulation — data already has real telephony artifacts.

4. **Training config** — sized for L40S 45 GB:
   - Phase A (epochs 0–1): freeze encoder, lr=3e-4, batch_duration=600
   - Phase B (epochs 2–9): unfreeze all, LLRD, lr=1e-4, batch_duration=360
   - Phase C (epochs 10–11): lr=1e-5, disable SpecAugment

5. **Train + evaluate + WiSE-FT**

---

## 9. Artifact / Path Reference

| Path | What | Persist? |
|------|------|----------|
| `/data/raw/gramvaani/GV_{Train_100h,Dev_5h,Eval_3h}/` | Raw MP3 + `text` | Keep |
| `/data/raw/gramvaani/Metadata/utt2labels_GV_*.txt` | Metadata TSVs | Keep |
| `/data/wav16k/{train,dev,eval}/` | Converted 16 kHz mono WAVs | Keep (expensive to regenerate) |
| `/data/manifests/{train,dev,eval}.json` | NeMo JSONL manifests | Rebuild after each pipeline change |
| `/data/exp/` | Experiment outputs | Keep |
| `/data/exp/logs/` | All script logs | Keep |
| `/workspace/repo/scripts/` | All pipeline scripts (git-tracked) | Git |
| `/workspace/repo/CLAUDE.md` | Project invariants for Claude Code | Git |
| `~/.cache/huggingface/` | Cached gated `.nemo` checkpoint | Ephemeral (container-scoped) |

### Docker commands reference

```bash
# Enter running container
docker exec -it nemo-ft bash

# Restart stopped container
docker start -ai nemo-ft

# Check container status
docker ps -a | grep nemo-ft

# Snapshot container state
docker commit nemo-ft nemo-ft:<tag>

# Disk usage check
df -h /data
docker system df        # from host
```

---

## 10. Architecture Decisions Log

### ADR-001: Gram Vaani over MUCS
**Context:** Need Hindi telephony ASR training data.
**Decision:** Use OpenSLR 118 (Gram Vaani) instead of OpenSLR 103 (MUCS).
**Rationale:** MUCS is read storybook speech. Gram Vaani is spontaneous telephone speech with real channel artifacts, dialect variation, and code-switching. Eliminates need for codec simulation. Ships with 1000h unlabelled for pseudo-labeling.

### ADR-002: Keep noisy utterances (inaudible/audio_jump)
**Context:** 31% of train has quality flags.
**Decision:** Train on all quality tiers. Tag in manifest for potential filtering later.
**Rationale:** Production system will see noisy audio. Dropping 31% loses domain-realistic training signal. Can pull noisy data into Phase C only if it hurts clean dev WER.

### ADR-003: Drop >20s utterances
**Context:** 363 train utts (0.9%) exceed 20s, 13 exceed 60s (max 141s).
**Decision:** Drop all >20s.
**Rationale:** 141s is clearly unsegmented garbage. 20–60s range is 0.9% of data. Segmentation complexity not justified. Loses ~1h.

### ADR-004: Drop #incomplete utterances
**Context:** 416 utts (1.13%) have `#incomplete` marker — partial transcript over full audio.
**Decision:** Drop by default (`--keep-incomplete` flag to override).
**Rationale:** Partial transcript over full audio = CTC/RNNT alignment poison. The model learns to emit silence tokens for spoken content, actively harmful.

### ADR-005: Must re-split speaker-disjoint
**Context:** Official splits have 1,338 shared speakers between train and dev.
**Decision:** Re-partition all utterances into speaker-disjoint splits.
**Rationale:** Speaker overlap means dev WER is optimistically biased — model memorizes speaker characteristics, not language. Violates experimental integrity.

### ADR-006: Extract SPE tokenizer without model instantiation
**Context:** NeMo 25.02 can't instantiate the IndicConformer checkpoint (`KeyError: 'dir'`).
**Decision:** Treat `.nemo` as tar archive, extract SPE `.model` files directly, auto-select Hindi tokenizer by scoring against real transcripts.
**Rationale:** Faster, avoids loading 600M params just for tokenizer check, sidesteps NeMo version incompatibility.

### ADR-007: Minimal augmentation
**Context:** Data already has real telephony artifacts.
**Decision:** Speed perturb (0.9/1.0/1.1) + optional MUSAN noise at high SNR (15–25 dB). No codec simulation, no RIR.
**Rationale:** Codec sim on already-coded audio adds synthetic artifacts on top of real ones — double degradation with no domain benefit. Speed perturb is always free data. MUSAN adds variability in background noise.