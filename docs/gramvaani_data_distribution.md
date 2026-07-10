# Gram Vaani (OpenSLR 118) — Data Distribution

Reference for the Hindi telephony fine-tune corpus. All numbers computed directly
from `/data/raw/gramvaani/` (GV_Train_100h / GV_Dev_5h / GV_Eval_3h) on 2026-07-10.

- **Source:** OpenSLR 118 — Gram Vaani ASR Challenge, spontaneous Hindi telephone speech.
- **Utterance ID:** `<district>-<speaker>-<segment>`, e.g. `02-22287-01`.
- **Speaker key:** `uttid.rsplit("-", 1)[0]` → `<district>-<speaker>` (e.g. `02-22287`).

---

## 1. Split totals

| Split | Utts  | Speaker keys | Median utts/spk | MP3 files | Nominal hours |
|-------|------:|-------------:|----------------:|----------:|--------------:|
| train | 37,152 | 20,266 | 2 | 37,152 | ~100 h |
| dev   |  1,885 |  1,826 | 1 |  1,885 | ~5 h |
| eval  |  1,032 |  1,011 | 1 |  1,032 | ~3 h |

Extreme speaker diversity: train averages <2 utts/speaker. There is almost no
per-speaker data to overfit to — good for generalization, bad for any
speaker-adaptation assumptions.

---

## 2. Audio format

| Split | 8 kHz | 44 kHz | 48 kHz |
|-------|------:|-------:|-------:|
| train | 100% | – | – |
| dev   | 100% | – | – |
| eval  | 93% | 5.6% | 1.2% |

- All source audio is **MP3**. Train/dev are uniformly 8 kHz (native telephony).
- Eval has a small high-samplerate tail — resampling to 16 kHz normalizes it.
- Pipeline (`02_build_manifest.py`) transcodes **MP3 → 16 kHz mono PCM WAV** via
  ffmpeg. Duration is measured from the converted WAV header (soundfile), not the MP3.

> Note: upsampling 8 kHz → 16 kHz adds no acoustic information above 4 kHz. The
> model input is 16 kHz because the pretrained IndicConformer front-end expects it;
> the true signal bandwidth remains telephony-band.

---

## 3. Duration distribution (train, pre-filter)

| Stat | Value |
|------|------:|
| mean | 9.9 s |
| p50  | 10.4 s |
| p95  | 12.1 s |
| max  | 141 s |
| utts > 20 s | 363 (1.0%) |
| utts > 60 s | 13 |

Tight unimodal distribution around ~10 s. The long tail (>20 s) is dropped
(see §5) — 363 train utts, plus a handful in dev/eval.

---

## 4. Quality flags — the `Other` metadata column

`Other` holds zero or more **comma-combined** annotation tags per utterance
(e.g. `audio_jump,inaudible`). `NA` means clean.

### 4a. Component-level frequency (atomic tags, after splitting on comma)

| Tag | train | dev | eval | Disposition |
|-----|------:|----:|-----:|-------------|
| NA (clean)     | 22,454 | 1,293 | 716 | keep |
| inaudible      | 10,020 |   412 | 222 | **keep + tag** |
| audio_jump     |  4,950 |   217 | 115 | **keep + tag** |
| speaker_change |  1,144 |    — |   — | keep + tag |
| not_clear      |    303 |    12 |   6 | keep + tag |
| audio_break    |    272 |    12 |   9 | keep + tag |
| laugh          |     92 |     2 |   — | keep + tag |
| cough          |     31 |    — |   — | keep + tag |
| stutter        |      2 |    — |   — | keep + tag |
| people_talking |      2 |    — |   — | keep + tag |
| sneezing       |      1 |    — |   — | keep + tag |

(Counts are per-utterance occurrences of each atomic tag; an utterance with
`audio_jump,inaudible` contributes to both rows.)

### 4b. Top combined values (train, whole `Other` string)

| `Other` value | count | % |
|---------------|------:|--:|
| NA | 22,454 | 60.4% |
| inaudible | 8,300 | 22.3% |
| audio_jump | 3,227 | 8.7% |
| audio_jump,inaudible | 1,385 | 3.7% |
| speaker_change | 654 | 1.8% |
| not_clear | 267 | 0.7% |
| audio_jump,speaker_change | 187 | 0.5% |
| inaudible,speaker_change | 173 | 0.5% |
| audio_break | 173 | 0.5% |
| _remaining combos_ | ~332 | ~0.9% |

39 distinct combined values in train.

---

## 5. Filter policy (`02_build_manifest.py`)

| Filter | Rule | Expected effect on this release |
|--------|------|---------------------------------|
| Duration | keep `0.5 s ≤ d ≤ 20.0 s` | drops ~363 long / 0 short in train |
| Hard-drop flags | drop if any atomic tag ∈ `{clipping, truncated, empty, No_speaking}` | **0 rows** — none present in 100 h release |
| Noisy-but-valid | **keep + tag** `inaudible`, `audio_jump`, and all others | ~40% of train retained as tagged-noisy |
| Code-switch (Latin) | keep verbatim, never strip | see §6 |

> **On this release the only real drops are duration-based.** The four hard-drop
> flags do not occur in any split of GV_*_100h (verified). The guard is retained
> defensively for the 1000 h release, where they may appear.

The raw `Other` string is preserved verbatim in the manifest `quality` field
(e.g. `"audio_jump,inaudible"`) so downstream curriculum/filtering keeps full
resolution.

---

## 6. Code-switching (Hindi–English)

| Split | Utts with Latin chars | % |
|-------|----------------------:|--:|
| train | 1,228 | 3.3% |
| dev   |   141 | 7.5% |
| eval  |     0 | 0.0% |

Code-switched transcripts are **kept and never stripped**. The pretrained SPE
tokenizer's `<unk>` behaviour on these is checked in Step 6 of the manifest build
(WARNING if `<unk>` > 5%). Note eval has **zero** Latin — code-switch WER is not
observable on the official eval set.

---

## 7. ⚠️ Speaker-disjointness — official splits VIOLATE it

CLAUDE.md non-negotiable: *"Splits are speaker-disjoint, not utterance-random."*
**The shipped Gram Vaani splits are not.** Overlap of `<district>-<speaker>` keys:

| Pair | Overlapping speaker keys |
|------|-------------------------:|
| train ∩ dev  | **1,338** |
| train ∩ eval | **758** |
| dev ∩ eval   | 79 |

Nearly every dev/eval speaker also appears in train. Validation/eval WER on the
official splits is therefore **optimistically biased** (train-seen speakers leak
into dev/eval).

**Implication / TODO:** to honour the non-negotiable we must **re-partition
speaker-disjointly** rather than use the shipped `dev`/`eval` as-is — group by
speaker key, then assign whole speakers to a single split. Until then, treat
official-split metrics as an upper bound, not a generalization estimate.

(Caveat: speaker numbering may not be globally unique across districts; the key
above assumes `<district>-<speaker>` identity. Even so, 1,338 exact key collisions
between train and dev is far too many to be coincidental ID reuse.)

---

## 8. Manifest schema (output of `02_build_manifest.py`)

One JSON object per line, written to `/data/manifests/{train,dev,eval}.json`:

```json
{"audio_filepath": "/data/wav16k/train/01-00003-02.wav", "duration": 10.234,
 "text": "...", "speaker": "01-00003", "lang": "hi", "quality": "inaudible"}
```

| Field | Meaning |
|-------|---------|
| `audio_filepath` | 16 kHz mono WAV under `/data/wav16k/{split}/` |
| `duration` | seconds, from WAV header (3 dp) |
| `text` | transcript verbatim (code-switch preserved) |
| `speaker` | `uttid.rsplit("-", 1)[0]` |
| `lang` | `"hi"` |
| `quality` | raw `Other` string (`NA` = clean; comma-combined tags otherwise) |
