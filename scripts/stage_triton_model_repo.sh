#!/usr/bin/env bash
# Stage IndicConformer artefacts into the Triton model_repository.
#
# Phase 1 layout (CTC ensemble + TensorRT encoder):
#   indic_asr_preproc/1/model.pt             ← preprocessor.ts (TorchScript)
#   indic_asr_encoder/1/model.onnx           ← encoder.onnx + external weights
#   indic_asr_encoder/1/layers.* / Constant_* / onnx__*  ← external weight blobs
#   indic_asr_encoder/1/model.plan           ← built later by trtexec (FP16 TRT engine)
#   indic_asr_ctc_decoder/1/model.onnx       ← ctc_decoder.onnx
#
# This script does NOT build the TensorRT engine (model.plan). That step needs
# the production GPU and the Triton+TRT container — see the runbook printed at
# the end.
#
# Implementation notes:
#   The local HF cache snapshot directory has been clobbered (every snapshot
#   path is a zero-byte regular file, not a symlink into blobs/). The blobs/
#   store still has all 2.4 GB of actual content, so this script bypasses
#   the broken snapshot dir entirely: it queries the HF API for filename →
#   blob_id mappings, then copies the matching blob directly into the model
#   repo. Falls back to a fresh snapshot if blobs are missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REPO="${REPO_ROOT}/triton/model_repository"

ENCODER_DIR="${MODEL_REPO}/indic_asr_encoder/1"
PREPROC_DIR="${MODEL_REPO}/indic_asr_preproc/1"
CTC_DIR="${MODEL_REPO}/indic_asr_ctc_decoder/1"

export ASR_MODEL_NAME="${ASR_MODEL_NAME:-ai4bharat/indic-conformer-600m-multilingual}"
HF_HOME_DEFAULT="${REPO_ROOT}/worker/hub"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"

echo "[stage] HF_HOME=${HF_HOME}"
echo "[stage] ASR_MODEL_NAME=${ASR_MODEL_NAME}"
echo "[stage] model_repo=${MODEL_REPO}"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        PYTHON="${REPO_ROOT}/.venv/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi
echo "[stage] python=${PYTHON}"

if ! "${PYTHON}" -c "import huggingface_hub" 2>/dev/null; then
    echo "[stage][error] huggingface_hub not importable from ${PYTHON}." >&2
    echo "  Install it with:  ${PYTHON} -m pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "${ENCODER_DIR}" "${PREPROC_DIR}" "${CTC_DIR}"

# ---------------------------------------------------------------------------
# Stage from blobs directly using HF metadata for filename → blob hash mapping.
# ---------------------------------------------------------------------------
export ENCODER_DIR PREPROC_DIR CTC_DIR

"${PYTHON}" - <<'PY'
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo_id = os.environ["ASR_MODEL_NAME"]
hf_home = Path(os.environ["HF_HOME"])
encoder_dir = Path(os.environ["ENCODER_DIR"])
preproc_dir = Path(os.environ["PREPROC_DIR"])
ctc_dir = Path(os.environ["CTC_DIR"])

# huggingface_hub stores blobs at:
#   {HF_HOME}/models--<org>--<name>/blobs/<git_sha or sha256>
repo_dir_name = "models--" + repo_id.replace("/", "--")
blob_root = hf_home / repo_dir_name / "blobs"

api = HfApi()
info = api.repo_info(repo_id=repo_id, files_metadata=True)

# Build filename → blob_id (small files) or sha256 (LFS) map.
file_blob = {}
for sib in info.siblings:
    blob_id = sib.lfs.sha256 if sib.lfs else sib.blob_id
    expected_size = sib.lfs.size if sib.lfs else sib.size
    file_blob[sib.rfilename] = (blob_id, expected_size)

print(f"[stage] HF repo has {len(file_blob)} files; blob_root={blob_root}", flush=True)


def resolve(rfilename: str) -> Path:
    """Return a local path containing the file's content."""
    blob_id, expected = file_blob[rfilename]
    blob_path = blob_root / blob_id
    if blob_path.is_file() and blob_path.stat().st_size == expected:
        return blob_path
    # blob is missing or wrong size — pull it down via hf_hub_download
    print(f"[stage]   blob missing/incorrect for {rfilename}; downloading…", flush=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=rfilename,
            cache_dir=str(hf_home),
            force_download=True,
        )
    )


def copy_to(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[stage]   {dst.name}  ←  {src.name}  ({dst.stat().st_size:,} bytes)", flush=True)


# 1. Preprocessor (TorchScript).
print("[stage] preprocessor.ts → indic_asr_preproc/1/model.pt", flush=True)
copy_to(resolve("assets/preprocessor.ts"), preproc_dir / "model.pt")

# 2. CTC decoder.
print("[stage] ctc_decoder.onnx → indic_asr_ctc_decoder/1/model.onnx", flush=True)
copy_to(resolve("assets/ctc_decoder.onnx"), ctc_dir / "model.onnx")

# 3. Encoder graph + external weights (must sit alongside model.onnx).
print("[stage] encoder.onnx → indic_asr_encoder/1/model.onnx", flush=True)
copy_to(resolve("assets/encoder.onnx"), encoder_dir / "model.onnx")

# Encoder external-tensor blobs: layers.*, Constant_*, onnx__*, plus anything
# starting with "pre" (some checkpoints embed pre/post-net biases as separate
# blobs). Skip preprocessor.ts itself.
prefixes = ("assets/layers.", "assets/Constant_", "assets/onnx__", "assets/pre")
weight_files = sorted(
    rf for rf in file_blob
    if rf.startswith(prefixes) and rf != "assets/preprocessor.ts"
)
print(f"[stage] encoder external weights ({len(weight_files)} blobs) → indic_asr_encoder/1/", flush=True)
copied = 0
for rf in weight_files:
    src = resolve(rf)
    dst = encoder_dir / Path(rf).name
    shutil.copyfile(src, dst)
    copied += 1
print(f"[stage] copied {copied} external weight blobs", flush=True)

# Sanity-check final sizes.
for label, path, expected_min in [
    ("preproc/model.pt",       preproc_dir / "model.pt",       50_000),
    ("encoder/model.onnx",     encoder_dir / "model.onnx",     1_000_000),
    ("ctc_decoder/model.onnx", ctc_dir / "model.onnx",         5_000_000),
]:
    sz = path.stat().st_size if path.is_file() else 0
    flag = "OK" if sz >= expected_min else "TOO SMALL"
    print(f"[stage] {label:<26} {sz:>12,} bytes  [{flag}]")

# Sum of all encoder weight blobs
total_w = sum((encoder_dir / Path(rf).name).stat().st_size for rf in weight_files)
print(f"[stage] encoder external weights total: {total_w:,} bytes")
PY

cat <<'NEXT'

----------------------------------------------------------------------
NEXT STEP — build the FP16 TensorRT engine for the encoder.
----------------------------------------------------------------------
The encoder config.pbtxt already declares backend: "tensorrt" and
default_model_filename: "model.plan". Triton will refuse to load the
encoder until model.plan exists, so build it inside the Triton+TRT
container against the production GPU:

  docker compose -f docker-compose.yml -f docker-compose.triton.yml \
      run --rm --entrypoint bash triton -c '
        trtexec \
          --onnx=/models/indic_asr_encoder/1/model.onnx \
          --fp16 \
          --minShapes=audio_signal:1x80x100,length:1 \
          --optShapes=audio_signal:1x80x800,length:1 \
          --maxShapes=audio_signal:1x80x3000,length:1 \
          --memPoolSize=workspace:4096 \
          --saveEngine=/models/indic_asr_encoder/1/model.plan
      '

Then bring the stack up:

  docker compose -f docker-compose.yml -f docker-compose.triton.yml up -d triton
  curl -sf http://localhost:8100/v2/health/ready && echo  READY

  for m in indic_asr_preproc indic_asr_encoder indic_asr_ctc_decoder \
           indic_asr_ctc indic_asr; do
      printf '%-25s ' "$m"
      curl -s -o /dev/null -w '%{http_code}\n' \
           http://localhost:8100/v2/models/$m/ready
  done

(All five should report 200.)
----------------------------------------------------------------------
NEXT
