#!/usr/bin/env bash
# Validate that the worker image built from worker/Dockerfile correctly:
#   1. Removed the Rust/Cargo toolchain after pip install (build-time only).
#   2. Has deepfilternet / silero-vad / pyrnnoise importable.
#   3. Loads _DeepFilterNetDenoiser when DENOISER=deepfilternet and uses the
#      silero-vad PyPI package (not torch.hub).
#   4. Falls back to the pyrnnoise RNNoise adapter when DENOISER=rnnoise.
#
# Usage:
#   bash scripts/verify_worker_image.sh [IMAGE_TAG]
#
# IMAGE_TAG defaults to credresolve-worker:verify. If the image does not exist
# locally the script builds it from the repo root using worker/Dockerfile.
#
# The script does not push images, run pytest, or touch any compose stack.
set -euo pipefail

IMAGE_TAG="${1:-credresolve-worker:verify}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
red()   { printf '\033[0;31m%s\033[0m\n' "$1" >&2; }

# Per check: print PASS line with a ✓ on success, FAIL with stderr dump on failure.
pass() { green "  ✓ $1"; }
fail() {
  red "  ✗ $1"
  if [[ -n "${2:-}" ]]; then
    red "----- captured output -----"
    printf '%s\n' "$2" >&2
    red "---------------------------"
  fi
  exit 1
}

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Image $IMAGE_TAG not found locally; building from worker/Dockerfile..."
  docker build -f "$REPO_ROOT/worker/Dockerfile" -t "$IMAGE_TAG" "$REPO_ROOT"
fi

echo "Verifying image: $IMAGE_TAG"

# ---------------------------------------------------------------------------
# Check 1: Rust/Cargo toolchain was removed in the final image.
# ---------------------------------------------------------------------------
echo "[1/4] Build-time Rust toolchain cleanup"
out=$(docker run --rm --entrypoint bash "$IMAGE_TAG" -c \
  'test ! -e /root/.cargo && test ! -e /root/.rustup && ! command -v cargo >/dev/null 2>&1 && ! command -v rustc >/dev/null 2>&1 && echo cargo-absent' 2>&1) \
  || fail "Rust toolchain artifacts present in final image (expected /root/.cargo and /root/.rustup removed, cargo/rustc not on PATH)" "$out"
[[ "$out" == *"cargo-absent"* ]] \
  || fail "Rust toolchain artifacts present in final image" "$out"
pass "/root/.cargo and /root/.rustup removed; cargo/rustc not on PATH"

# ---------------------------------------------------------------------------
# Check 2: deepfilternet / silero-vad / pyrnnoise are importable.
# ---------------------------------------------------------------------------
echo "[2/4] Audio dependency imports"
out=$(docker run --rm --entrypoint python "$IMAGE_TAG" -c '
from df.enhance import init_df, enhance  # deepfilternet -> deepfilterlib
from silero_vad import load_silero_vad, get_speech_timestamps
from pyrnnoise import RNNoise
print("imports-ok")
' 2>&1) || fail "One or more audio dependencies failed to import" "$out"
[[ "$out" == *"imports-ok"* ]] \
  || fail "Import probe did not print imports-ok" "$out"
pass "df.enhance, silero_vad, pyrnnoise all import cleanly"

# ---------------------------------------------------------------------------
# Check 3: DENOISER=deepfilternet loads _DeepFilterNetDenoiser, and Silero
# VAD comes from the silero-vad PyPI package (not torch.hub).
# ---------------------------------------------------------------------------
echo "[3/4] DENOISER=deepfilternet load path (DFN + silero-vad PyPI)"
out=$(docker run --rm --entrypoint python \
  -e DENOISER=deepfilternet \
  -e PYTHONPATH=/srv \
  -w /srv \
  "$IMAGE_TAG" -c '
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(name)s %(levelname)s %(message)s")
from app.audio_processing import AudioPreprocessor
ap = AudioPreprocessor()
got = type(ap.rnnoise).__name__
assert got == "_DeepFilterNetDenoiser", f"expected _DeepFilterNetDenoiser, got {got}"
assert ap.vad_model is not None, "vad_model is None"
print(f"denoiser={got}")
' 2>&1) || fail "DENOISER=deepfilternet did not load _DeepFilterNetDenoiser" "$out"

[[ "$out" == *"denoiser=_DeepFilterNetDenoiser"* ]] \
  || fail "Denoiser type assertion failed" "$out"
[[ "$out" == *"DeepFilterNet3 denoiser loaded successfully"* ]] \
  || fail "Missing 'DeepFilterNet3 denoiser loaded successfully' log line" "$out"
[[ "$out" == *"Silero VAD model loaded via silero-vad PyPI package"* ]] \
  || fail "Silero VAD did not load via PyPI package (suspect torch.hub fallback)" "$out"
[[ "$out" != *"Silero VAD model loaded via torch.hub"* ]] \
  || fail "Silero VAD fell through to torch.hub legacy path" "$out"
pass "DeepFilterNet3 loaded; Silero VAD via PyPI package"

# ---------------------------------------------------------------------------
# Check 4: DENOISER=rnnoise loads the pyrnnoise adapter (fallback path works).
# ---------------------------------------------------------------------------
echo "[4/4] DENOISER=rnnoise fallback path"
out=$(docker run --rm --entrypoint python \
  -e DENOISER=rnnoise \
  -e PYTHONPATH=/srv \
  -w /srv \
  "$IMAGE_TAG" -c '
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(name)s %(levelname)s %(message)s")
from app.audio_processing import AudioPreprocessor
ap = AudioPreprocessor()
got = type(ap.rnnoise).__name__
assert got in {"_PyRNNoiseDenoiser", "_RNNoiseWrapperDenoiser"}, f"unexpected RNNoise type: {got}"
print(f"denoiser={got}")
' 2>&1) || fail "DENOISER=rnnoise did not load a pyrnnoise/rnnoise adapter" "$out"

[[ "$out" == *"denoiser=_PyRNNoiseDenoiser"* || "$out" == *"denoiser=_RNNoiseWrapperDenoiser"* ]] \
  || fail "RNNoise adapter type assertion failed" "$out"
[[ "$out" == *"pyrnnoise denoiser loaded successfully"* || "$out" == *"rnnoise_wrapper denoiser loaded successfully"* ]] \
  || fail "Missing RNNoise success log line" "$out"
pass "RNNoise fallback adapter loaded"

# ---------------------------------------------------------------------------
# Image size — sanity check that the Rust install didn't leak in. The PyTorch
# CUDA base image already accounts for the bulk (~7-9 GB); we print the size
# so reviewers can eyeball it.
# ---------------------------------------------------------------------------
size_bytes=$(docker image inspect "$IMAGE_TAG" --format '{{.Size}}')
size_human=$(awk -v b="$size_bytes" 'BEGIN { printf "%.2f GB", b/1024/1024/1024 }')
echo
echo "Final image size: $size_human ($size_bytes bytes)"
green "All checks passed for $IMAGE_TAG"
