# APM flag: keep or drop?

> **Superseded by implementation.** The recommendation in this document has shipped:
> the soft-deprecation phase landed in S5b (PR #TBD) and the hard removal landed in S5c (PR #TBD).
> Live behavior — `apm_enabled` is no longer parsed and is silently ignored if sent;
> `WebRTCAPMBackend` and `WebRTCAudioProcessor` have been deleted from `gateway/app/apm.py`;
> `STREAMING_APM_ENABLED` is no longer a config knob; the frontend no longer sends the field.
> This document is preserved as the historical rationale.

**Status:** Recommendation — **drop the `apm_enabled` flag from the public schema.** *(Implemented.)*
**Author:** audio pipeline review, 2026-05-07.
**Scope:** Decides the fate of the `apm_enabled` knob currently exposed by the gateway. Does *not* change code or schema in this turn.

---

## 1. What `apm_enabled` is supposed to do

`apm_enabled` is a session-level boolean that toggles a WebRTC-style Audio Processing Module in front of the gateway VAD. The intended pipeline (in code today):

```
PCM in -> [APM frame loop, 10 ms @ 16 kHz] -> WebRTC VAD (20 ms) -> RNNT stream
```

The APM contract is defined in [gateway/app/apm.py:7-46](../../gateway/app/apm.py#L7-L46):

- `APMConfig` — frozen 16 kHz / 10 ms config with three feature toggles: `noise_suppression`, `agc`, `high_pass_filter` (defaults all `True`).
- `WebRTCAPMBackend` — `Protocol` describing the binding the gateway expects: `process_frame(frame, *, sample_rate, noise_suppression, agc, high_pass_filter) -> bytes` plus `reset()`.
- `NoOpAudioProcessor` and `WebRTCAudioProcessor` — same interface; `WebRTCAudioProcessor` would delegate to a real backend.

The pipeline always runs *some* `AudioProcessor`: see [gateway/app/pipeline.py:120-131](../../gateway/app/pipeline.py#L120-L131) and [gateway/app/pipeline.py:162-182](../../gateway/app/pipeline.py#L162-L182). When APM is "off" (or "on" with no backend) it is `NoOpAudioProcessor`, which only re-emits its input bytes unchanged ([gateway/app/apm.py:54-59](../../gateway/app/apm.py#L54-L59)).

There is no implementation behind the toggle. [gateway/app/main.py:243-261](../../gateway/app/main.py#L243-L261):

```python
def build_apm_backend() -> object | None:
    # TODO: Bind a real WebRTC APM Python backend here.
    return None

def build_audio_processor(session: SessionConfig):
    config = APMConfig(enabled=session.apm_enabled, sample_rate=session.sample_rate)
    if not config.enabled:
        return NoOpAudioProcessor(config)
    backend = build_apm_backend()
    if backend is None:
        log.warning("APM enabled but no backend is configured; falling back to no-op audio processor")
        return NoOpAudioProcessor(config)
    return WebRTCAudioProcessor(config, backend)
```

Setting `apm_enabled=true` therefore yields exactly the same audio bytes as `apm_enabled=false`, plus a warning log line.

This is consistent with the existing internal note in [docs/plan_for_audio_processing_backend.md:14-15](../plan_for_audio_processing_backend.md#L14-L15) ("apm_enabled is dead on the worker side — accepted, logged, never consumed") and the production audit at [docs/production_system_design.md:197-201](../production_system_design.md#L197-L201) ("APM and speaker verification are stubbed").

## 2. Current call sites

| Layer    | Role         | Where                                                                                                                                                          |
|----------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Frontend | sender       | [frontend/src/utils/constants.ts:132](../../frontend/src/utils/constants.ts#L132) sets `?apm_enabled=...` query param; default `false` at [App.tsx:91](../../frontend/src/App.tsx#L91); UI toggle at [ContextBiasingPanel.tsx:14,54](../../frontend/src/components/ContextBiasingPanel.tsx#L14) |
| Frontend | session-update payload | [frontend/src/utils/contextBiasing.ts:26](../../frontend/src/utils/contextBiasing.ts#L26), [frontend/src/types/ws.ts:25](../../frontend/src/types/ws.ts#L25) |
| Gateway  | parse        | Query param: [gateway/app/main.py:497](../../gateway/app/main.py#L497). `audio_processing.apm_enabled` in WS update: [gateway/app/main.py:625](../../gateway/app/main.py#L625) |
| Gateway  | env default  | `STREAMING_APM_ENABLED` at [gateway/app/config.py:41](../../gateway/app/config.py#L41); compose default `false` at [docker-compose.yml:156](../../docker-compose.yml#L156)              |
| Gateway  | logging only | [main.py:765-777](../../gateway/app/main.py#L765-L777), [main.py:947-953](../../gateway/app/main.py#L947-L953)                                                                          |
| Gateway  | flag handed to pipeline context (unused by pipeline) | [main.py:329](../../gateway/app/main.py#L329), [main.py:963](../../gateway/app/main.py#L963), `PipelineSessionContext.apm_enabled` field at [main.py:153](../../gateway/app/main.py#L153) |
| Gateway  | builds processor object | [main.py:248-261](../../gateway/app/main.py#L248-L261) — always returns `NoOpAudioProcessor` because `build_apm_backend()` returns `None` |
| Worker   | receives flag | **Does not.** [gateway/app/worker_client.py:46-66](../../gateway/app/worker_client.py#L46-L66) sends `X-VAD-Enabled` and `X-Denoise-Enabled` only; `transcribe(...)` has no `apm_enabled` parameter. Worker `/v1/transcribe` headers at [worker/app/main.py:676-677](../../worker/app/main.py#L676-L677) define `x_vad_enabled` and `x_denoise_enabled` only. No code in `worker/app/audio_processing.py` references APM. |

**Confirmed: the worker neither receives nor uses `apm_enabled`.** Today the flag terminates inside the gateway pipeline as a parameter to a `NoOp` processor.

## 3. Integration options for *real* APM functionality

### 3.1 `webrtc-audio-processing` (xiongyihui/python-webrtc-audio-processing)

Verified 2026-05-07:

| Field                    | Value                                                                                                  |
|--------------------------|--------------------------------------------------------------------------------------------------------|
| PyPI latest              | **0.1.3, uploaded 2018-07-17** (per `pypi.org/pypi/webrtc-audio-processing/json`)                      |
| Repo last push           | 2025-05-07 (a batch of community PR merges, incl. macOS arm64; default branch `master`)                |
| Open issues              | 9                                                                                                      |
| Recurring open requests  | AEC support (#21 2026-01-28, #18 2024-12-06, #15 2022-11-02), build failures on Windows/macOS (#10, #8, #7, #13) |
| Status field on PyPI     | "Pre-Alpha (Development Status 2)"                                                                     |
| Build requirements       | Native build against bundled libwebrtc-audio-processing C++ source; needs C++ toolchain, autotools-style deps; binary wheels not published — `pip install` triggers from-source build |

Refines the late-2025 "dormant" reading: the GitHub repo accepted PRs in May 2025, but **no new release has been cut to PyPI in nearly 8 years**, AEC remains unimplemented, and build robustness across platforms is the dominant theme of open issues. Pinning `webrtc-audio-processing==0.1.3` ships an 8-year-old binding of an old libwebrtc snapshot. Pinning a git SHA from `master` exposes us to PyPI-less, source-built dependency hygiene in CI/CD.

### 3.2 Build the binding from source against a current `libwebrtc-audio-processing` system package

Distros now package newer libwebrtc APM (Debian/Ubuntu `libwebrtc-audio-processing-1`, recent versions of the Pipewire-aligned fork). Approach: write our own thin pybind11/cffi wrapper over the system `.so` and conform to the `WebRTCAPMBackend` Protocol already defined at [gateway/app/apm.py:35-46](../../gateway/app/apm.py#L35-L46).

| Pros | Cons |
|------|------|
| Up-to-date upstream code, real AEC/AGC2/NS3 available | Real engineering investment: 1–2 weeks for a robust binding + CI build matrix |
| Frees us from PyPI staleness | Adds a native dep to the gateway image (apt package + headers) |
| Matches the protocol contract that already exists in code | Maintenance burden across distro/python upgrades (ABI breaks) |

### 3.3 "Pick apart the pieces"

Rather than one APM blob, compose the capabilities we actually need:

| Need              | Library                              | Status (verified 2026-05-07) |
|-------------------|--------------------------------------|------------------------------|
| VAD               | `webrtcvad` (`py-webrtcvad`)         | PyPI **2.0.10, 2017-01-07**; dormant but functional. **Already in use** in gateway pipeline ([pipeline.py:102](../../gateway/app/pipeline.py#L102)). |
| VAD (modern)      | Silero VAD                           | Already in use in worker ([worker/app/audio_processing.py:64-69](../../worker/app/audio_processing.py#L64-L69)). |
| NS + AGC          | `webrtc-noise-gain` (dscripka)       | PyPI **1.2.5, 2024-07-08**; actively maintained; ships wheels; exposes WebRTC NS + AGC2 only (no AEC). |
| Denoise           | RNNoise (already wired)              | Already in use in worker ([worker/app/audio_processing.py:83](../../worker/app/audio_processing.py#L83)). |
| AEC               | None of the above                    | Echo-cancellation is not packaged in any maintained Python binding today. Would require either approach 3.2 or an external C/C++ process. |
| High-pass filter  | `scipy.signal.butter` + `lfilter`    | Trivial. |

The realistic "real APM" path through this menu is `webrtc-noise-gain` (NS + AGC) plus the existing `webrtcvad` and a 2-pole high-pass — i.e. APM minus AEC.

### 3.4 Cloud APMs

E.g. NVIDIA Riva noise reduction. Out of scope: the worker is a containerized GPU service running our own NeMo/Triton stack ([docs/bigger_gpu_migration_guide.md](../bigger_gpu_migration_guide.md)) and not authorized to call third-party endpoints per the production design ([production_system_design.md](../production_system_design.md)). Adding an outbound dependency for what is essentially NS+AGC is the wrong shape.

## 4. Do we actually need AEC?

Echo cancellation matters when **the same device captures both the microphone input and the speaker output** (loopback path). Conditions for our deployment:

- The audio source is the **browser** (`navigator.mediaDevices.getUserMedia`), running on the end-user's device. The browser already enables AEC + NS at capture time — see [frontend/IMPLEMENTATION_GUIDE.md:288-291](../../frontend/IMPLEMENTATION_GUIDE.md#L288-L291) (`echoCancellation: true, noiseSuppression: true`).
- The gateway runs on **CPU nodes**, the worker on **GPU nodes** ([docs/production_system_design.md:263-267](../production_system_design.md#L263-L267)). Server-side, there is no microphone, no speaker, no loopback — server-side AEC is not meaningful.
- There is **no real-time agent loop** in this codebase: TTS exists as a *helper* for emitting payloads downstream ([gateway/app/tts_helper.py](../../gateway/app/tts_helper.py)), but no code path plays TTS back through the same capture device that's feeding the ASR session. (`docs/knowledge_transfer.md:147,511-518` explicitly flags TTS round-tripping as a *future* concern.)

**Conclusion: AEC is not needed today.** It would only become relevant if the system grew an on-device agent loop where the ASR stream and a TTS playback share a capture device — and even then, AEC belongs at the capture point (browser / device SDK), not in the server gateway.

This is the load-bearing observation: it removes the only feature in the WebRTC APM bundle that we would have to build a native binding to get. NS, AGC, and high-pass are all available without that pain.

## 5. Recommendation: drop `apm_enabled` from the public schema

We should pick option **(a) Drop**, for these reasons:

1. The flag has been a no-op since it was introduced. There is no production traffic that depends on it doing anything.
2. The one part of WebRTC APM we *cannot* easily get from a maintained Python binding — AEC — is also the one we don't need (Section 4).
3. The remaining APM capabilities (NS, AGC, high-pass) overlap with audio processing the worker already performs (Silero VAD, RNNoise, denoise pipeline at [worker/app/audio_processing.py:124](../../worker/app/audio_processing.py#L124)) and with the browser's own `getUserMedia` constraints. Adding a *second* NS pass at the gateway is at best redundant and at worst a WER regression on Indian-English/Hindi audio that has already been tuned around RNNoise.
4. The `vad_enabled` and `denoise_enabled` flags already exist on the same audio-processing object ([main.py:498-499](../../gateway/app/main.py#L498-L499)), are honored end-to-end, and cover the use cases. They are the right place to land any future NS/AGC work — without inventing a new gateway-side native dependency.
5. Keeping a public flag that does nothing is an API-shape liability: clients build retries and dashboards around it, then we either have to honor a fake contract forever or break it later.

Outcomes (b) "Implement via path X" and (c) "Defer" are explicitly not recommended:
- (b) requires either pinning an 8-year-old PyPI release or owning a native binding to libwebrtc-audio-processing-1, for capabilities we either already have (NS/denoise) or don't need (AEC).
- (c) has no obvious signal that would change the answer absent an on-device agent-loop deployment, which is not on the roadmap.

### 5.1 Patch list (deferred — not part of this turn)

Public-API surface to remove:

| File                                                              | Lines              | Change                                                                                  |
|-------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------------------|
| [gateway/app/main.py](../../gateway/app/main.py)                  | 127, 153, 329, 497, 625, 765-777, 947-963 | Remove `apm_enabled` from `SessionConfig`, `PipelineSessionContext`, query parsing, session-update parsing, and log lines. |
| [gateway/app/main.py](../../gateway/app/main.py)                  | 243-261            | Delete `build_apm_backend` and `build_audio_processor`. Pass `NoOpAudioProcessor(APMConfig(sample_rate=session.sample_rate))` directly, or inline the framing into the pipeline if we want to drop the abstraction entirely. |
| [gateway/app/apm.py](../../gateway/app/apm.py)                    | whole file         | Either delete, or strip down to the framing helper (`frame_bytes`) + `NoOpAudioProcessor`. The `WebRTCAPMBackend` Protocol and `WebRTCAudioProcessor` class can go. |
| [gateway/app/config.py](../../gateway/app/config.py)              | 41                 | Remove `STREAMING_APM_ENABLED`.                                                         |
| [gateway/tests/test_ws_stt_partial_out_guard.py](../../gateway/tests/test_ws_stt_partial_out_guard.py) | 163 | Drop the `apm_enabled` kwarg.                                                           |
| [docker-compose.yml](../../docker-compose.yml)                    | 156                | Remove `STREAMING_APM_ENABLED`.                                                         |
| [frontend/src/App.tsx](../../frontend/src/App.tsx)                | 91                 | Remove `apmEnabled` from default state.                                                 |
| [frontend/src/components/ContextBiasingPanel.tsx](../../frontend/src/components/ContextBiasingPanel.tsx) | 14, 54  | Remove `apmEnabled` field and the "WebRTC APM" toggle row.                              |
| [frontend/src/utils/constants.ts](../../frontend/src/utils/constants.ts) | 4, 132    | Remove `apmEnabled` from the audio-processing type and from the URL builder.            |
| [frontend/src/utils/contextBiasing.ts](../../frontend/src/utils/contextBiasing.ts) | 26 | Remove `apm_enabled` from the session-update payload type.                              |
| [frontend/src/types/ws.ts](../../frontend/src/types/ws.ts)        | 25                 | Remove `apm_enabled?: boolean`.                                                         |
| [tools/eval_nemo_manifest_wer.py](../../tools/eval_nemo_manifest_wer.py), [tools/eval_serving_model_manifest_wer.py](../../tools/eval_serving_model_manifest_wer.py) | various | These imports of `gateway.app.apm` are evaluation-only. Either keep `NoOpAudioProcessor` for those, or migrate them to use the worker's `audio_processing.AudioPreprocessor`. |
| [docs/project_architecture_deepdive.md](../project_architecture_deepdive.md), [docs/production_system_design.md](../production_system_design.md), [docs/plan_for_audio_processing_backend.md](../plan_for_audio_processing_backend.md) | mentions of APM | Update prose to reflect removal. |

Worker side: nothing to change — the worker never accepted the flag.

### 5.2 Schema migration plan

- The flag is exposed in two public surfaces: the WebSocket connect query string (`?apm_enabled=...`) and the in-band `session.update` payload (`audio_processing.apm_enabled`).
- Backward compatibility: gateway should *accept and ignore* `apm_enabled` for one deprecation window (e.g. one minor release). Concretely: keep the parsing branches at [main.py:497](../../gateway/app/main.py#L497) and [main.py:625](../../gateway/app/main.py#L625) but discard the value, and emit a one-shot deprecation log per session. After the window, remove parsing entirely so unknown fields produce `BAD_MESSAGE`.
- Frontend: remove the toggle in the same release that begins ignoring it; ensure the default-`false` historical behavior is preserved (it always was — the value was a no-op).

### 5.3 Deprecation note (for the changelog / API doc)

> **Deprecated and removed: `apm_enabled` session flag.** This flag was historically accepted but had no effect — the gateway WebRTC Audio Processing Module never had a backend wired in. AEC is not required for this server-side deployment (echo cancellation belongs at the browser capture point, which already enables it via `getUserMedia` constraints). Noise suppression and AGC capabilities are available through the existing `denoise_enabled` and worker-side audio preprocessing path. Clients should remove `apm_enabled` from connect URLs and `session.update` payloads. The gateway will accept and ignore the field for one release before rejecting unknown fields.
