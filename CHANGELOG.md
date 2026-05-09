# Changelog

## Unreleased

### Removed

- **`apm_enabled` session flag.** This flag was historically accepted but had no effect — the gateway WebRTC Audio Processing Module never had a backend wired in. AEC is not required for this server-side deployment (echo cancellation belongs at the browser capture point, which already enables it via `getUserMedia` constraints). Noise suppression and AGC capabilities are available through the existing `denoise_enabled` and worker-side audio preprocessing path. Clients should remove `apm_enabled` from connect URLs and `session.update` payloads. The gateway accepted and ignored the field for one release (S5b) and now silently ignores it like any other unknown field. The `WebRTCAPMBackend` / `WebRTCAudioProcessor` types and the `STREAMING_APM_ENABLED` env knob have been removed. See [docs/audio/apm-decision.md](docs/audio/apm-decision.md).
