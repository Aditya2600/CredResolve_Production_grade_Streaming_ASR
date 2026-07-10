# Nemotron 3.5 ASR 0.6B vs IndicConformer-600M — Hindi benchmark

Local, reproducible benchmark of **`nvidia/nemotron-3.5-asr-streaming-0.6b`** (Cache-Aware
FastConformer-RNNT, 600M, OpenMDW-1.1, Hindi `hi-IN` is *transcription-ready*) against the
production model **`ai4bharat/indic-conformer-600m-multilingual`** on public Hindi audio.
Fills the "noisy/local Hindi WER — unknown (test locally)" gate in
[hindi_asr_model_research_2026.md](hindi_asr_model_research_2026.md).

Runs on the remote **NVIDIA L4** (23 GB, CUDA 12.8). Author locally, paste to the L4 to execute.

## Why two environments
Nemotron needs **NeMo 26.06 (git main) + Python ≥3.11**, which conflicts with the worker's pinned
`nemo_toolkit[asr]==2.4.1` / `torch==2.4.1` / `onnxruntime-gpu==1.20.1`. So Nemotron runs in an
**isolated `python3.12` venv** (the L4 has no conda) via NVIDIA's official cache-aware streaming infer
script; we never force it into [eval_nemo_manifest_wer.py](../tools/eval_nemo_manifest_wer.py). Both models' hypotheses are
then scored through the **same** `normalize_asr_text` + corpus-WER path
([score_nemotron_manifest.py](../tools/benchmarks/score_nemotron_manifest.py)), so the comparison is
apples-to-apples at the text layer (punctuation/casing/`<lang>` tags are normalized away).

## Test sets
- **Primary — Vaani Hindi (`ARTPARK-IISc/Vaani-transcription-part`, config `audio/Hindi`)**: the repo's
  in-domain Hindi anchor (also its PEFT eval set), head-to-head vs IndicConformer. **Gated** (needs an HF
  token) and large — cap with `--limit` (e.g. 500). Alt set: `sarvamai/contextual_asr_benchmark` (`hi-IN`).
  - ⚠️ **Leakage caveat:** Vaani is also the repo's PEFT *train/replay* source (only a `train` split ships). If
    your IndicConformer was adapted on Vaani, scoring it on freshly-pulled `train` rows is an optimistic
    baseline. For a fair head-to-head, point steps 3–5 at a **held-out** Vaani manifest instead (e.g. your
    `artifacts/vaani_50h_*/dev.jsonl` seed-42 split from `tools/split_vaani_manifest.py`) and skip the build —
    the manifest just needs `audio_filepath`/`duration`/`text` rows pointing at 16 kHz mono WAVs.
- **Anchor — FLEURS Hindi (`google/fleurs`, `hi_in`, test)**: clean read speech; reproduce Nemotron's
  published ~6.81% WER @1.12s to prove the pipeline is wired correctly *before* trusting other numbers.

## Operating points
`att_context_size` sets the streaming latency/accuracy point on one checkpoint:
`[56,3]` = **320 ms** (production-realistic) and `[56,13]` = **1.12 s** (best accuracy).
Published FLEURS-Hindi WER (with LangID): 80 ms 8.13% · 320 ms 7.41% · 560 ms 7.05% · 1.12 s **6.81%**.

---

## Procedure (paste-ready, on the L4)

### 1. Isolated Nemotron env (one-time)
The L4 has **no conda**, so we use a `python3.12` venv. One script does it all
(installs torch cu124 + NeMo git-main + audio deps, clones NeMo for `examples/`,
and runs the load sanity check):
```bash
bash tools/benchmarks/setup_nemotron_env.sh     # SLOW: NeMo git-main + torch, several GB
source ~/nemotron-asr/bin/activate
# Pin reproducibility once it works: re-run with  NEMO_REF=<sha> bash tools/benchmarks/setup_nemotron_env.sh
```
The infer script lands at `~/NeMo-src/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`.

### 2. Build + validate manifests (either env; needs numpy/soundfile/datasets)
```bash
python tools/benchmarks/build_hindi_eval_manifest.py --source vaani --limit 500 --hf-token $HF_TOKEN
python tools/benchmarks/build_hindi_eval_manifest.py --source fleurs        # clean anchor (ungated)
# gate each (the builder prints the exact validate command for its manifest):
python tools/validate_nemo_manifest_audio.py \
  --input  artifacts/benchmarks/nemotron_vs_indic/vaani.jsonl \
  --output artifacts/benchmarks/nemotron_vs_indic/vaani.valid.jsonl \
  --rejects artifacts/benchmarks/nemotron_vs_indic/vaani.rejects.jsonl \
  --summary-json artifacts/benchmarks/nemotron_vs_indic/vaani.validate.json \
  --expected-sample-rate 16000 --require-mono --min-duration-sec 0.3 --max-duration-sec 40
```
(For the contextual or FLEURS sets, swap `vaani` → `contextual`/`fleurs` in `--source` and in every path below.)

### 3. Run Nemotron at both latency points (isolated env)
```bash
source ~/nemotron-asr/bin/activate
python tools/benchmarks/run_nemotron_streaming.py \
  --manifest artifacts/benchmarks/nemotron_vs_indic/vaani.valid.jsonl \
  --script ~/NeMo-src/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  --target-lang hi-IN --att-context-sizes "[56,3]" "[56,13]" --batch-size 16
# -> nemotron_vaani.valid_att56_3.json, nemotron_vaani.valid_att56_13.json, *_timing.json
# Verify the hyp key on the first output row (expected pred_text): head -1 <output>.json
```
If `model_path=` rejects the HF id on this build, save once and pass the local file:
`python -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained('nvidia/nemotron-3.5-asr-streaming-0.6b').save_to('nemotron.nemo')"`
then add `--model ./nemotron.nemo`. (Optionally also run `--target-lang auto` once to quantify the LID penalty.)

### 4. IndicConformer baseline on the same manifest (production env)
Use your existing production env (the one with `nemo_toolkit==2.4.1`), NOT the Nemotron venv.
On this L4 the `.nemo` was **not** at the path the code assumes
(`/home/ubuntu/models/indicconformer/IndicConformer.nemo`) — point `--model` at wherever your
IndicConformer `.nemo` actually lives:
```bash
python tools/eval_nemo_manifest_wer.py \
  --model <path/to/IndicConformer.nemo> \
  --manifest artifacts/benchmarks/nemotron_vs_indic/vaani.valid.jsonl \
  --decoder rnnt --language hi --reference-normalization vaani \
  --out-jsonl artifacts/benchmarks/nemotron_vs_indic/indic_vaani.jsonl \
  --out-summary-json artifacts/benchmarks/nemotron_vs_indic/indic_vaani.summary.json
```
**No `.nemo` anywhere?** Use the HF ONNX worker instead (downloads
`ai4bharat/indic-conformer-600m-multilingual`), which also emits an `--out-jsonl`:
```bash
python tools/eval_serving_model_manifest_wer.py \
  --manifest artifacts/benchmarks/nemotron_vs_indic/vaani.valid.jsonl \
  --model-name ai4bharat/indic-conformer-600m-multilingual \
  --decoder rnnt --language hi --device cuda --reference-normalization vaani \
  --out-jsonl artifacts/benchmarks/nemotron_vs_indic/indic_vaani.jsonl \
  --out-summary-json artifacts/benchmarks/nemotron_vs_indic/indic_vaani.summary.json
```

### 5. Score both through the shared normalizer (either env; pure-python)
```bash
python tools/benchmarks/score_nemotron_manifest.py \
  --manifest artifacts/benchmarks/nemotron_vs_indic/vaani.valid.jsonl \
  --nemotron-json artifacts/benchmarks/nemotron_vs_indic/nemotron_vaani.valid_att56_3.json \
                  artifacts/benchmarks/nemotron_vs_indic/nemotron_vaani.valid_att56_13.json \
  --indic-jsonl artifacts/benchmarks/nemotron_vs_indic/indic_vaani.jsonl \
  --timing-json artifacts/benchmarks/nemotron_vs_indic/nemotron_vaani.valid_timing.json \
  --out-md   artifacts/benchmarks/nemotron_vs_indic/vaani_report.md \
  --out-json artifacts/benchmarks/nemotron_vs_indic/vaani_summary.json \
  --out-jsonl artifacts/benchmarks/nemotron_vs_indic/vaani_per_sample.jsonl
```
Repeat 2–5 for FLEURS (`--source fleurs`, manifest `fleurs.valid.jsonl`).

---

## Results — Vaani Hindi (audio/Hindi)
> Fill from `vaani_summary.json` after the L4 run. RTF = inference wall-clock ÷ total audio.

| Model | Emission latency | Clips | WER % | Sub | Del | Ins | Throughput RTF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IndicConformer-600M (prod) | — (full-context) | TBD | **TBD** | TBD | TBD | TBD | TBD |
| Nemotron 3.5 ASR 0.6B | 320 ms `[56,3]` | TBD | **TBD** | TBD | TBD | TBD | TBD |
| Nemotron 3.5 ASR 0.6B | 1.12 s `[56,13]` | TBD | **TBD** | TBD | TBD | TBD | TBD |

## Results — FLEURS Hindi (anchor)
> Sanity: Nemotron @`[56,13]` should land near **~6.81%**. Large deviation ⇒ debug manifest/lang-tag/normalizer wiring first.

| Model | Emission latency | WER % | Published ref |
| --- | --- | --- | --- |
| Nemotron 3.5 ASR 0.6B | 320 ms `[56,3]` | TBD | 7.41% |
| Nemotron 3.5 ASR 0.6B | 1.12 s `[56,13]` | TBD | 6.81% |
| IndicConformer-600M (prod) | — | TBD | ~10–15% (clean, published) |

## Run log (copy into your experiment log after the run)
- Date / NeMo commit SHA pinned: …
- Clips evaluated (vaani / FLEURS): … / …
- FLEURS anchor reproduced? Nemotron@1.12s = …% vs 6.81% published → pass/fail
- Decision: route Hindi → Nemotron, or stay single-model (IndicConformer)? …

## Gotchas
- **Env isolation** is mandatory — never `pip install` Nemotron's stack into the production env.
- **Verify the CLI** on first run: `model_path` vs `pretrained_name`, hyp key `pred_text`, quote `att_context_size="[56,3]"`.
- **`hi-IN` not `auto`** for the headline numbers (auto LID can misroute and inflate WER).
- **Pin the NeMo commit SHA** once a run succeeds (git main is a moving target).
- **L4 memory**: start `batch_size=16`, drop to 8/4 on OOM.
- **Spot-check** 5–10 ref/hyp pairs (the scorer prints examples) to confirm normalization isn't hiding real errors.
