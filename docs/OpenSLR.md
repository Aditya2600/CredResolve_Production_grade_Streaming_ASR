## Read of your box

`L40S 46068 MiB` → ~45 GB usable, Ada Lovelace, bf16 + TF32 native. `216 GB` free on `/`. `/dev/shm 101G` — that's the important one: you can run 12–16 dataloader workers without the shared-memory crashes that plague containerized NeMo.

Revised disk math for **100 h** (vs. the 285 h number I gave earlier):

| Item | Size |
|---|---|
| Raw 100 h @ 16 kHz WAV | 11.5 GB |
| MUSAN | 11 GB |
| OpenSLR 28 (RIR) | 5 GB |
| 3× codec-augmented, 8 kHz FLAC | ~10 GB |
| Tarred/Lhotse shards | ~10 GB |
| Checkpoints (`save_top_k=3` + last) | 29 GB |
| Scratch | 20 GB |
| **Total** | **~97 GB** |

You have 216 GB. **Disk is a non-issue at 100 h.** Don't bother with the aggressive trimming I suggested — keep the loose augmented FLACs around, you'll want them when you regenerate shards after a text-normalization bug (you will have a text-normalization bug).

---

## Stage 0 — Environment

Driver 570 / CUDA 12.8. Do **not** pip-install NeMo into a bare venv — the RNNT loss is a Numba CUDA kernel and version-matching Numba↔CUDA↔PyTorch by hand is a half-day you won't get back.

```bash
docker run --gpus all -it --rm \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data:/data \
  nvcr.io/nvidia/nemo:25.02 bash
```

Sanity check before anything else:

```python
import torch, nemo.collections.asr as nemo_asr
from nemo.collections.asr.losses.rnnt import RNNTLoss
assert torch.cuda.is_bf16_supported()
RNNTLoss(num_classes=128)  # fails loudly if warprnnt_numba is broken
```

If that last line throws, stop. Everything downstream is wasted.

---

## Stage 1 — Data layout and splits

```
/data/
  raw/{hi,mr,or}/           # OpenSLR 103 MUCS
  musan/  rirs/
  manifests/
  shards/
  exp/
```

Build manifests, then **carve the splits before augmenting**:

- `train.json` — 92 h
- `dev_clean.json` — 4 h, held out, **no augmentation**
- `dev_codec.json` — same 4 h utterances, augmented once with a *fixed seed* and a *held-out codec* (e.g. augment train with G.711 + GSM, hold AMR-NB for dev)

That second dev set is the whole point. If you validate only on codec-augmented audio drawn from the same augmentation distribution as training, your dev WER measures how well you learned your own ffmpeg pipeline, not how well you transcribe phone calls. Holding out a codec you never trained on is the cheapest available proxy for real telephony generalization.

Speaker-disjoint splits, not utterance-random. MUCS ships speaker IDs — use them.

---

## Stage 2 — Augmentation → shards

Offline, once. Per utterance emit 3 variants + keep the clean original (so 4× ≈ 400 h effective):

```
16k → 8k → [G.711 μ-law | GSM-FR | AMR-NB 12.2k] (random)
     → burst packet loss p∈[0, 0.03]
     → MUSAN noise, SNR ∈ [5, 20] dB
     → RIR convolution, p=0.5
     → save 8 kHz FLAC
```

Write FLAC at 8 kHz and upsample in the dataloader (`model.sample_rate: 16000` with a resample transform). The codec already annihilated everything above 4 kHz — storing 16 kHz is storing zeros.

Parallelize with GNU parallel across your CPU cores; this is a few hours, not overnight, at 100 h.

Then shard. Use **Lhotse** rather than legacy tarred manifests if your NeMo is 2.x — dynamic duration bucketing is worth 25–40% throughput over fixed batching, because a 2 s utterance and a 19 s utterance no longer share a padded batch.

---

## Stage 3 — Tokenizer

**Reuse the pretrained SPE tokenizer.** Do not rebuild it. A 600M model's decoder embedding + joint output layer are tied to that vocabulary; swapping it means reinitializing the joint and throwing away most of what the pretrain bought you.

Exception: if your held-out probe shows >8% of reference tokens hitting `<unk>` — likely if MUCS Odia orthography diverges from what AI4Bharat trained on — then you have a real problem and should extend rather than replace the vocab.

Normalize text first: `indic-num2words` for numerals, unify Devanagari punctuation, strip Latin-script code-mixed tokens or transliterate them consistently. Pick one policy and apply it identically to train, dev, and your eventual gold set.

---

## Stage 4 — Config

Sized for 45 GB. Static cost (fp32 master + bf16 copy + grads + Adam moments) is ~10.8 GB; you have ~34 GB for activations.

```yaml
init_from_pretrained_model: ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large

model:
  sample_rate: 16000

  train_ds:
    use_lhotse: true
    batch_duration: 360          # seconds per batch, not utterances
    quadratic_duration: 20       # penalizes long utts, guards the O(T²) tail
    bucket_duration_bins: [2.0, 4.0, 6.0, 8.0, 11.0, 15.0, 20.0]
    num_buckets: 7
    max_duration: 20.0
    min_duration: 0.5
    num_workers: 12
    pin_memory: true

  validation_ds:
    batch_size: 8
    num_workers: 4

  encoder:
    use_pytorch_sdpa: true       # flash/mem-efficient attention path
    stochastic_depth_drop_prob: 0.1

  joint:
    fuse_loss_wer: true
    fused_batch_size: 4          # keeps B×T×U×V from materializing

  aux_ctc:
    ctc_loss_weight: 0.3

  spec_augment:
    freq_masks: 2
    freq_width: 27
    time_masks: 10
    time_width: 0.05

  optim:
    name: adamw
    lr: 1e-4
    betas: [0.9, 0.98]
    weight_decay: 1e-3
    sched:
      name: CosineAnnealing
      warmup_steps: 2000
      min_lr: 1e-6

trainer:
  devices: 1
  precision: bf16-mixed
  max_epochs: 12
  accumulate_grad_batches: 2
  gradient_clip_val: 1.0
  val_check_interval: 0.25
  accelerator: gpu

exp_manager:
  checkpoint_callback_params:
    save_top_k: 3
    monitor: val_wer
    mode: min
  resume_if_exists: true
```

Also set, in your launcher:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Verify `use_pytorch_sdpa` and `quadratic_duration` exist in your NeMo version — the flag surface moved around across 1.23 → 2.x, and silently-ignored YAML keys are NeMo's signature failure mode. Print the resolved config and read it.

**Why `batch_duration: 360` and not `batch_size: 16`:** with duration bucketing, a batch of 2 s clips has 90 utterances and a batch of 19 s clips has 18, and both occupy roughly the same memory. Fixed batch size means you must size for the worst case and waste ~40% of VRAM on every other batch. Expect peak ~34–38 GB. If you OOM on the longest bucket, drop `batch_duration` to 300 before touching anything else.

---

## Stage 5 — Schedule

100 h × 4 (augmented) ≈ 400 h ≈ 180,000 utterances at ~8 s mean.

| Phase | Epochs | What | Why |
|---|---|---|---|
| **A** | 0–1 | Freeze encoder. Train decoder + joint + CTC head only. `lr: 3e-4`. | The joint is being fed by a pretrained encoder into a decoder that's seeing a new label distribution. Two epochs of letting the head catch up prevents the first few thousand high-gradient steps from wrecking encoder features. Static memory drops to ~3 GB here; you can raise `batch_duration` to 600. |
| **B** | 2–9 | Unfreeze all. LLRD: layer 0 gets `lr × 0.4`, top layer gets `lr × 1.0`, geometric interpolation. Peak `1e-4`. | Lower layers encode acoustics that mostly transfer; upper layers encode the language-model-ish structure that must adapt. LLRD is the single highest-leverage knob you have on a 600M model with 100 h. |
| **C** | 10–11 | `lr` → `1e-5`, disable SpecAugment, drop augmentation to 1× (clean + one codec). | Sharpens on the actual target distribution after the model has already generalized. |

~4,000 optimizer steps/epoch at `accumulate_grad_batches: 2`. On an L40S, budget **60–90 min/epoch** → **12–18 h total**. Measure your actual `it/s` in the first 200 steps and extrapolate before committing the GPU-hours.

**Early stopping is not optional.** 600M params on 100 h of read speech is an overfitting setup. `dev_clean` WER will keep improving after `dev_codec` WER has bottomed out — that divergence is your stop signal, and it typically appears around epoch 7–9.

---

## Stage 6 — Evaluate, then interpolate

Decode with both heads. RNNT is your production path; CTC decoding is a free 5-second diagnostic:

```bash
python examples/asr/transcribe_speech.py \
  model_path=exp/checkpoints/model.nemo \
  dataset_manifest=/data/manifests/dev_codec.json \
  decoder_type=rnnt batch_size=16
```

If CTC WER and RNNT WER move together, the encoder learned something. If CTC WER is flat while RNNT WER drops, the decoder is memorizing the training text distribution and you're overfitting — pull back to an earlier checkpoint.

**WiSE-FT:** interpolate the fine-tuned weights with the pretrained checkpoint, `θ = α·θ_ft + (1-α)·θ_pre`, sweep `α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}` on `dev_codec`. This costs one hour and reliably recovers 1–3 WER points of robustness lost to fine-tuning, at near-zero cost to in-domain accuracy. `α ≈ 0.7` is the usual winner. Do this before you conclude anything about whether the run worked.

Bucket your WER report by: language, utterance duration (<4 s / 4–10 s / >10 s), and codec. The aggregate number will hide that short utterances regressed — they almost always do, because the RNNT prediction network has less context to work with and codec artifacts dominate a larger fraction of the signal.

---

## What will actually go wrong

**Numba/CUDA mismatch on the RNNT kernel.** Catch it in Stage 0, not at step 50 of epoch 0.

**Silent YAML keys.** NeMo does not error on unrecognized config keys in many code paths. Your `fused_batch_size` may be doing nothing. Print the resolved `OmegaConf` and grep it.

**Checkpoint disk creep.** Each `.ckpt` is ~7.2 GB (fp32 weights + both Adam moments). `save_top_k: 3` + `last.ckpt` = 29 GB. Fine on 216 GB, but if you also enable `save_last` per-epoch backups you'll fill the volume around epoch 9.

**The distribution gap.** MUCS is read speech. Codec simulation buys you channel realism — bandwidth, quantization, packet loss. It does not buy you disfluency, overlapping turns, hold music, keyboard clatter, or the specific ambience of your call center's headsets. Expect this fine-tune to move you meaningfully off the 15% WER baseline on clean-ish calls and much less on hard ones. It is the *initialization* for your pseudo-labeling stage, not the deliverable. Plan the 25 h of gold labels accordingly, and spend them on the audio the ensemble disagrees about rather than a random sample.