# DeepFilterNet3 — Build, Runtime, and Cargo Dependency Reference

> **Scope:** Why the worker Docker image needs Rust/Cargo during `pip install`,
> how the Dockerfile handles it, the full runtime behaviour of the
> `_DeepFilterNetDenoiser` adapter, the fallback chain, and how to verify that
> the built image is correct. Audio evaluation results that drove the
> DeepFilterNet3 vs RNNoise decision live in [denoiser-eval.md](denoiser-eval.md).

---

## 1. Dependency chain: why Cargo is required

```
worker/requirements.txt
  └─ deepfilternet>=0.5.6          # Python package on PyPI
       ├─ deep-filter (Python API)  # pure Python — no compilation
       └─ deepfilterlib              # Rust extension — REQUIRES Cargo at build time
            └─ compiled from source via PyO3 + maturin
```

`deepfilterlib` is a Rust extension (PyO3/maturin) that wraps the DFN3 inference
kernel. PyPI publishes manylinux2014 wheels for `deepfilterlib` for some
Linux/Python combinations, but **the worker base image
(`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`) does not match any published
wheel** for Python 3.11 on its glibc/ABI target. pip therefore falls back to
building the extension from source, which requires:

- `rustc` and `cargo` (provided by rustup, installed inside the build RUN)
- `build-essential pkg-config libssl-dev` (native compile toolchain, already
  installed via apt-get)
- `curl` (to download rustup itself)

Neither `rustc` nor `cargo` ship in the PyTorch CUDA runtime base image.
Without them, `pip install deepfilternet` fails with an error from maturin
during the deepfilterlib sdist build.

---

## 2. Dockerfile pattern

```dockerfile
# apt layer — native toolchain + curl (separate layer; rarely changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl build-essential pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt /srv/requirements.txt
RUN python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# Cargo is BUILD-TIME ONLY. deepfilterlib is the Rust extension inside the
# deepfilternet wheel that pip compiles from source on this base image; once
# the wheel is installed into site-packages we no longer need rustc/cargo,
# so the toolchain is removed in the same RUN layer to keep it out of the
# final image. Verified end-to-end by scripts/verify_worker_image.sh.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --no-modify-path && \
    PATH="/root/.cargo/bin:${PATH}" pip install -r /srv/requirements.txt && \
    rm -rf /root/.cargo /root/.rustup
```

### Why a single chained `RUN` for rustup + pip + cleanup

Each `RUN` in a Dockerfile produces one image layer. Layers are additive — a
file added in layer N and deleted in layer N+1 still exists in the layer N
blob and inflates the final image size. By chaining `rustup install`, `pip
install`, and `rm -rf .cargo .rustup` in one `RUN`, the Rust toolchain never
appears in any layer that ships. The final image contains only the compiled
`deepfilterlib` `.so` wheel inside site-packages — not the compiler or cargo
registry cache.

### rustup install flags explained

| Flag | Effect |
|---|---|
| `--profile minimal` | Installs only `rustc` and `cargo`. Skips `clippy`, `rustfmt`, `rust-docs`, and other components that are irrelevant for a pip build. Reduces download and toolchain size. |
| `--no-modify-path` | Does not append `source ~/.cargo/env` or similar to shell profiles (`.bashrc`, `.profile`). We set `PATH` manually on the next line, so profile modification would only confuse later RUN steps. |
| `-y` | Non-interactive; skips the "press Enter to continue" prompt. Required for Docker. |

### What `rm -rf /root/.cargo /root/.rustup` removes

- `/root/.cargo` — the Cargo home directory: the `bin/` directory (contains
  `cargo`, `rustc` symlinks via `cargo` proxy), the local package registry
  cache, and any downloaded crates used when compiling deepfilterlib.
- `/root/.rustup` — the rustup toolchain directory: the actual `rustc`/`cargo`
  binaries, the standard library, and target-specific libraries.

Deleting both is necessary; deleting only one leaves orphaned binaries. After
the `rm`, `which cargo` and `which rustc` both fail inside a running container.

---

## 3. Runtime behaviour — `_DeepFilterNetDenoiser`

The `_DeepFilterNetDenoiser` adapter in
[`worker/app/audio_processing.py`](../../worker/app/audio_processing.py) wraps
the DFN3 inference call behind the same byte-in / byte-out contract as the
RNNoise adapters:

```
input:  48 kHz mono int16 PCM bytes
output: 48 kHz mono int16 PCM bytes (denoised)
```

The caller (`AudioPreprocessor.process_with_stats`) handles resampling from
the session sample rate to 48 kHz before calling the adapter, and from 48 kHz
back to the session rate after. The adapter itself always sees and returns
48 kHz audio.

### Initialisation (`__init__`)

```python
from df.enhance import init_df

self._model, self._df_state, _ = init_df()
sr = getattr(self._df_state, "sr", lambda: 48000)
self._sr = sr() if callable(sr) else sr
if self._sr != 48000:
    raise RuntimeError(f"DeepFilterNet sample rate is {self._sr}, expected 48000")
```

`init_df()` loads the bundled DFN3 model weights. `self._df_state` carries the
model configuration; `self._sr` is extracted at init time so the check runs
once at startup rather than per-utterance. If upstream ever releases a DFN
model at a different sample rate the `RuntimeError` surfaces it immediately
rather than silently producing wrong-rate audio.

### Per-utterance inference (`process`)

```python
from df.enhance import enhance

audio = np.frombuffer(pcm_48k, dtype=np.int16)   # bytes → int16 array
x = (audio.astype(np.float32) / 32768.0).reshape(1, -1)  # → float32 (1, samples)
tensor = torch.from_numpy(x)

with torch.no_grad():
    cleaned = enhance(self._model, self._df_state, tensor)

cleaned = cleaned.detach().cpu().numpy().reshape(-1)
cleaned = np.clip(cleaned * 32768.0, -32768, 32767).astype(np.int16)
return cleaned.tobytes()
```

Key design decisions:

| Decision | Reason |
|---|---|
| `torch.no_grad()` | Disables gradient tracking; reduces memory and improves throughput for inference-only use. |
| `reshape(1, -1)` | `enhance()` expects shape `(channels, samples)`. DFN3 is mono: channels=1. |
| `np.clip(... -32768, 32767)` | Prevents int16 overflow if `enhance()` returns values outside `[-1, 1]` (can happen on high-amplitude transients). |
| whole-utterance call | `enhance()` is called on the complete utterance buffer, not in streaming chunks. This gives DFN3 full temporal context and maximises quality, at the cost of latency that scales with utterance length. Accepted for this batch-style pipeline. |

### Thread safety and the `_denoise_lock`

DFN3's `enhance()` modifies shared state in `self._df_state`. The adapter is
**not thread-safe**: calling `enhance()` concurrently from multiple threads on
the same `_DeepFilterNetDenoiser` instance corrupts the model's internal
buffers. `AudioPreprocessor` serialises all calls through `self._denoise_lock`
(a `threading.Lock`):

```python
with self._denoise_lock:
    pcm_48k     = resample_int16(pcm_bytes, sample_rate, target_sr)
    cleaned_48k = self.rnnoise.process(pcm_48k)
    pcm_bytes   = resample_int16(cleaned_48k, target_sr, sample_rate)
```

The lock wraps the full resample → denoise → resample triple to prevent a
second request from seeing partially-resampled audio interleaved with its own.
`AudioPreprocessor` is a singleton (via `@lru_cache(maxsize=1)` on
`get_audio_preprocessor()`), so there is one lock per worker process.

---

## 4. Configuration — `DENOISER` env var

Declared in `worker/app/config.py`:

```python
_VALID_DENOISERS = {"rnnoise", "deepfilternet", "none"}
_raw_denoiser = getenv_str("DENOISER", "deepfilternet").lower()
DENOISER = _raw_denoiser if _raw_denoiser in _VALID_DENOISERS else "deepfilternet"
```

| Value | Effect |
|---|---|
| `deepfilternet` (default) | Load `_DeepFilterNetDenoiser`. Falls back to RNNoise if the package is missing or `init_df()` throws. |
| `rnnoise` | Skip DFN3, go directly to `_load_rnnoise()` (pyrnnoise, then rnnoise_wrapper). |
| `none` | Denoising disabled; `self.rnnoise = None`; audio passes through unmodified. |
| anything else | Treated as `deepfilternet` (silent coercion, not an error). |

Docker Compose default (in `docker-compose.yml`):

```yaml
- DENOISER=${DENOISER:-deepfilternet}
```

---

## 5. Fallback chain — `_load_denoiser()` / `_load_rnnoise()`

```
DENOISER=deepfilternet
    │
    ├─ try _DeepFilterNetDenoiser()
    │      ├─ success → "DeepFilterNet3 denoiser loaded successfully" [INFO]
    │      ├─ ImportError → "DENOISER=deepfilternet but ... not installed; falling back to RNNoise" [WARN]
    │      └─ any other Exception → "Failed to load DeepFilterNet3 (%s); falling back to RNNoise." [WARN]
    │
    └─ _load_rnnoise()
           ├─ try _PyRNNoiseDenoiser()        ← pyrnnoise>=0.4.3
           │      ├─ success → "pyrnnoise denoiser loaded successfully" [INFO]
           │      └─ ImportError / Exception → try next
           │
           ├─ try _RNNoiseWrapperDenoiser()   ← rnnoise_wrapper (legacy, not in requirements.txt)
           │      ├─ success → "rnnoise_wrapper denoiser loaded successfully" [INFO]
           │      └─ ImportError → "RNNoise Python binding not found. Denoising will be skipped." [WARN]
           │
           └─ return None  → denoise stage silently skipped (audio unchanged)

DENOISER=rnnoise → skip DFN3, go directly to _load_rnnoise()
DENOISER=none   → return None immediately (no attempt to load any denoiser)
```

The production image has `pyrnnoise==0.4.3` in `worker/requirements.txt`, so
the fallback chain never reaches `rnnoise_wrapper` or `None` in normal
operation.

---

## 6. Verification

Run `scripts/verify_worker_image.sh` against a freshly built image to confirm
all five properties hold:

```bash
docker build -f worker/Dockerfile -t credresolve-worker:verify .
bash scripts/verify_worker_image.sh credresolve-worker:verify
```

Expected output: four `✓` lines plus a final image size print.

| Check | What it asserts |
|---|---|
| 1 — Rust toolchain removed | `/root/.cargo` and `/root/.rustup` absent; `cargo`/`rustc` not on `PATH` |
| 2 — Audio dependency imports | `df.enhance.{init_df,enhance}`, `silero_vad.load_silero_vad`, `pyrnnoise.RNNoise` all import without error |
| 3 — DFN load + Silero PyPI path | `DENOISER=deepfilternet` → `type(ap.rnnoise).__name__ == "_DeepFilterNetDenoiser"`; log contains `"DeepFilterNet3 denoiser loaded successfully"` and `"Silero VAD model loaded via silero-vad PyPI package"` (not the `torch.hub` fallback line) |
| 4 — RNNoise fallback | `DENOISER=rnnoise` → `type(ap.rnnoise).__name__` is `_PyRNNoiseDenoiser` or `_RNNoiseWrapperDenoiser`; log contains the appropriate `"... denoiser loaded successfully"` line |
| Image size | Printed for eyeball check; bulk of image is the PyTorch CUDA base (~7–9 GB), not Rust artifacts |

---

## 7. Files involved

| File | Role |
|---|---|
| `worker/requirements.txt` | Pins `deepfilternet>=0.5.6` and `pyrnnoise==0.4.3` |
| `worker/Dockerfile` | Installs rustup + pkg-config, runs pip with Cargo on PATH, removes toolchain |
| `worker/app/audio_processing.py` | `_DeepFilterNetDenoiser`, `_PyRNNoiseDenoiser`, `_RNNoiseWrapperDenoiser`, `_load_denoiser()`, `_load_rnnoise()`, `AudioPreprocessor._denoise_lock` |
| `worker/app/config.py` | `DENOISER` env var declaration and valid-value coercion |
| `scripts/verify_worker_image.sh` | End-to-end image validation (five checks) |
| `docs/audio/denoiser-eval.md` | Evaluation data: WER and latency benchmarks that drove the DFN3 selection |
| `docs/audio/noise-cancellation-deep-dive.md` | Full audio pipeline context (VAD + denoiser + resampler) |
