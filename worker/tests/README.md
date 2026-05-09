# worker/tests

Two complementary test layers cover `worker/app/audio_processing.py`:

## 1. Structural — `test_audio_processing.py`
Exercises control flow (VAD select-mode branches, denoise adapter wiring,
config fallbacks, empty-input handling) with **mocked models**: Silero VAD is
replaced by a stub that returns a fixed `speech_timestamps` list and the
denoiser is a stub. Fast, hermetic, and runnable with no model cache.

## 2. Characterization — `test_audio_characterization.py`
Pins the **numerical byte output** of `AudioPreprocessor.process()` against
committed goldens in `tests/golden/audio_processing/`, using the **real
Silero VAD** (loaded from local `torch.hub` cache) and the **real RNNoise**
denoiser (`pyrnnoise` or `rnnoise_wrapper`). The `denoise=True` combos also
exercise the `audioop.ratecv` 16k → 48k → 16k resampling path, so a future
PR that swaps `audioop` (Python 3.13 deprecation) has these byte-level
goldens to diff against.

### Two-tier comparison
- **Tier 1**: `sha256(output_bytes)` vs the hash committed in
  `<stem>.npy.sha256`. Catches any byte-level regression.
- **Tier 2** (only on Tier 1 mismatch): `np.allclose(out, golden, atol=1,
  rtol=0)`. Distinguishes "drift within one LSB" (likely a numpy/torch
  determinism quirk) from a real numerical divergence. The failure message
  names which tier diverged.

### Regenerate goldens
**Only on intentional algorithm changes** (e.g., P2 audioop swap, P4
denoiser swap, accepted upstream model bump). Do **not** regenerate to
silence flaky failures — investigate first.

```bash
python tests/regenerate_golden_audio.py
```

The script prints per-file SHA prefix and byte size on completion and is
itself the source of truth for which combos exist (the spec is loaded from
`test_audio_characterization.py`, so the two cannot drift apart).

### Marker / running
The marker `characterization` is registered in the root `pytest.ini`.

```bash
pytest                                  # runs everything (default)
pytest -m "not characterization"        # fast iteration; skip characterization
pytest -m characterization              # only characterization
```

### Required local cache
- **Silero VAD**: cached at
  `$TORCH_HOME/hub/snakers4_silero-vad_master/` (defaults to
  `~/.cache/torch/hub/...`). The test session SKIPS with a clear hint if the
  cache is missing — we never network at test time. To populate it once:
  ```bash
  python -c 'import torch; torch.hub.load("snakers4/silero-vad", "silero_vad")'
  ```
- **RNNoise**: at least one of `pyrnnoise` (preferred) or `rnnoise_wrapper`
  must be importable. If neither is, the `denoise_enabled=True` combos are
  skipped with a clear message; the `denoise_enabled=False` combos still
  run.

### Why `vad_enabled=False` is not in the cross product
`vad_enabled=False` bypasses both `vad_select_mode` branches, so the mode
parameter is meaningless there. With denoise off it is a trivial passthrough
already covered by the structural suite; with denoise on it duplicates the
ratecv path that the `vad_enabled=True` combos already exercise on real
post-VAD segments. Skipping the four `vad=False` slots keeps the suite
focused on the 16 combos that actually exercise distinct numerical paths.

### Determinism notes
On the reference machine all 16 combos are byte-stable across repeated runs
(Tier 1 passes; suite runtime ~3.9s). If a fixture turns out to be
non-deterministic on some hardware (e.g., cuDNN nondet on a different CPU),
record it here and the Tier 2 fallback will keep the suite green while we
investigate.

### Loudest vs. concat coalesce on the current fixtures
On the four `audio_bench/*.wav` clips, Silero VAD finds exactly one speech
segment per clip, so `vad_select_mode={loudest, concat}` produce identical
byte output (matching SHA per fixture/denoise pair). The two-mode
parametrization is still worth keeping — it pins both code paths byte-for-byte
and will diverge automatically the moment a fixture lands with multiple
segments, or `_select_segments` / concat padding changes. If you want the
mode-divergence path covered today, add a multi-segment fixture (e.g., two
short utterances with a clear silent gap) and regenerate.
