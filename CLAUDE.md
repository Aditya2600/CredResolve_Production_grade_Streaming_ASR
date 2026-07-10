# IndicConformer Telephony Fine-tune

## Hardware
- Single L40S, 45GB usable VRAM. 216GB disk on /. /dev/shm = 101GB.
- Run everything inside `nvcr.io/nvidia/nemo:25.02`. Never pip-install NeMo bare.

## Non-negotiables
- Reuse the pretrained SPE tokenizer. Never rebuild it.
- Hybrid CTC+RNNT. aux_ctc.ctc_loss_weight = 0.3. Do not drop the CTC head.
- bf16-mixed only. No fp16.
- dev_codec uses a HELD-OUT codec (AMR-NB) never seen in training.
- Splits are speaker-disjoint, not utterance-random.

## Paths
/data/{raw,musan,rirs,manifests,shards,exp}

## Style
- No new deps without asking. Prefer stdlib + torchaudio + ffmpeg.
- Every script takes --dry-run and prints what it would do.
- Log to /data/exp/logs/<script>.log. Never print to stdout only.
- Ask before deleting anything under /data/raw.