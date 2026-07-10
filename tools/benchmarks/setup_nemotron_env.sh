#!/usr/bin/env bash
# Set up an ISOLATED environment for Nemotron 3.5 ASR on the L4.
# The L4 has no conda, so we use a python3.12 venv (kept fully separate from the
# production worker's pinned NeMo 2.4.1 / torch 2.4.1 / onnxruntime-gpu 1.20.1).
#
# Run ONCE on the L4:   bash tools/benchmarks/setup_nemotron_env.sh
# Then activate with:   source ~/nemotron-asr/bin/activate
#
# This is the slow, network-heavy step (NeMo git-main + torch = several GB, many
# minutes). Override defaults via env vars, e.g. NEMO_REF=<sha> for reproducibility.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/nemotron-asr}"
NEMO_SRC="${NEMO_SRC:-$HOME/NeMo-src}"
NEMO_REF="${NEMO_REF:-main}"   # pin a commit SHA here once a run succeeds
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo ">> [1/6] system libs (libsndfile, ffmpeg)"
if command -v apt-get >/dev/null 2>&1; then
  ( command -v sudo >/dev/null 2>&1 && sudo apt-get update -y && sudo apt-get install -y libsndfile1 ffmpeg ) \
    || ( apt-get update -y && apt-get install -y libsndfile1 ffmpeg ) || echo "  (skip apt; install libsndfile1+ffmpeg manually if soundfile/ffmpeg fail)"
fi

echo ">> [2/6] python3.12 venv at $ENV_DIR"
python3.12 -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install -U pip wheel setuptools

echo ">> [3/6] torch (CUDA 12.x wheels; bundles its own CUDA runtime, needs only driver >= 525)"
pip install torch --index-url "$TORCH_INDEX"

echo ">> [4/6] NeMo $NEMO_REF with ASR extra  (SLOW)"
pip install Cython packaging
# PEP 508 direct-URL syntax (modern pip rejects the old '#egg=nemo_toolkit[asr]' fragment).
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_REF}"

echo ">> [5/6] manifest + scoring deps"
pip install soundfile soxr huggingface_hub datasets

echo ">> [6/6] NeMo source for examples/ (the infer script is not shipped in the wheel)"
if [ ! -d "$NEMO_SRC" ]; then
  git clone --depth 1 --branch "$NEMO_REF" https://github.com/NVIDIA/NeMo.git "$NEMO_SRC" 2>/dev/null \
    || git clone --depth 1 https://github.com/NVIDIA/NeMo.git "$NEMO_SRC"
fi
INFER="$NEMO_SRC/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"

echo ">> sanity: load Nemotron + print class/params/sample_rate"
python - <<'PY'
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")
print("LOADED", type(m).__name__, round(sum(p.numel() for p in m.parameters())/1e6, 1), "M  sample_rate=", m.cfg.get("sample_rate"))
PY

echo
echo ">> DONE."
echo "   activate:  source $ENV_DIR/bin/activate"
echo "   infer script: $INFER"
echo "   record the resolved NeMo commit and re-run with NEMO_REF=<sha> for reproducibility:"
( cd "$NEMO_SRC" && git rev-parse HEAD 2>/dev/null | sed 's/^/     NEMO_REF=/' ) || true
