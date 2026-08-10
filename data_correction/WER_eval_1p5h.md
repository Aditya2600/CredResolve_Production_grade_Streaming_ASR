# Indic Conformer WER — 1.5h Test Set

Date: 2026-07-11

## Result

| metric | value |
|---|---|
| **WER** | **5.13%** |
| utterances | 588 (F1=439, F2=149) |
| duration | 1.500 h |
| reference words | 17,593 |
| substitutions / deletions / insertions | 489 / 183 / 231 |

Text-only comparison of columns already present in the sheets. Indic Conformer
output was pre-generated (not re-run during eval).

- **hyp** (hypothesis) = `transcript` column = Indic Conformer output
- **ref** (reference) = human-corrected ground truth
  - F1: `ground_truth_transcript` (col 4)
  - F2: `Ground Truth` (col 4)

## Sources

| id | sheet | audio dir |
|---|---|---|
| F1 | `Dataset generation - batch_transcript-parent 2 - Sheet 1 - batch_transcript-pare-2.csv` | `vad_chunks/` |
| F2 | `Transcription Phase 2 - Transcription Phase 2 - batch_transcript (3) (2).csv` | `vad_chunks/transcription level2/` |

## Selection rules

- **F1**: keep all rows EXCEPT ranges 1120–1148, 1202–1318, 1353–1398. Drop-remark
  rows kept (bad-audio rows included by choice). → 1203 eligible.
- **F2**: keep rows where `Remarks` ∈ {Done, Drop, Perfect}. → 435 eligible.
- Pool = 1638 → shuffled (`random.seed(42)`) → filled until cumulative wav
  duration reaches 1.5h (5400s) → 588 utts.
- Durations read from local wav headers (stdlib `wave`).

## Normalization

Repo `tools/compute_wer.py`: Unicode NFKC → strip punctuation (Unicode category
`P*` → space) → collapse whitespace → word split. Corpus WER =
(S+D+I) / reference_words.

## Reproduce

```bash
# 1. build test set (deterministic, seed=42)
python3 data_correction/build_testset_1p5h.py
#   -> data_correction/testset_1p5h.tsv          (ref<TAB>hyp per line)
#   -> data_correction/testset_1p5h_manifest.csv (filename,source,duration,ref,hyp)

# 2. compute WER
python3 tools/compute_wer.py --tsv data_correction/testset_1p5h.tsv
```

Same seed → same 588 utterances → same 5.13%.
