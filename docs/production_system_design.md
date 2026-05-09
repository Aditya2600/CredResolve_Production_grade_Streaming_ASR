# CredResolve ASR Production System Design

## Goal

Turn this repo from a strong production-style reference stack into an operationally safe production service for real customer traffic, multi-tenant access, and GPU-backed scaling.

This design is written against the current repository, not as a generic ASR checklist.

## Current State Summary

What already exists:

- Split runtime between `gateway` (public WebSocket edge) and `worker` (private ASR service).
- Optional Triton-backed serving path for cleaner model serving.
- VAD-based utterance segmentation and worker concurrency caps.
- Prometheus, Grafana, node-exporter, and optional DCGM exporter.
- Browser demo frontend and smoke/load-test tools.

What the repo is today:

- "Streaming-ish" utterance transcription, not true incremental token streaming.
- Docker Compose oriented.
- Strong for demos, internal pilots, and controlled deployments.
- Not yet complete for hardened internet-facing production.

## Recommended Production Architecture

```text
Browser / Telephony / Partner Client
    -> CDN / WAF / TLS termination / DDoS protection
    -> L7 Load Balancer with WebSocket support
    -> Gateway service (CPU, stateless, horizontally scaled)
    -> Internal service mesh / private LB
    -> Worker service (GPU-aware adapter, horizontally scaled)
    -> Triton inference service (GPU)
    -> Model artifacts + cache

Supporting plane
    -> Redis (rate limiting, quotas, ephemeral coordination)
    -> Postgres (tenants, API credentials, audit/config metadata)
    -> Object storage (eval logs, sampled audio, phrase packs, reports)
    -> Metrics + logs + traces + alerting
    -> CI/CD + image registry + secrets manager + IaC
```

## Service Roles

### 1. Edge Layer

- TLS termination with valid public certificates.
- WAF, IP filtering, request-size limits, bot/DDoS protection.
- WebSocket-aware load balancing.
- Access logging with request IDs and tenant IDs.

### 2. Gateway Layer

Keep the gateway as the only public runtime service.

Responsibilities:

- Authentication and authorization
- Session validation
- WebSocket lifecycle
- Audio normalization and VAD segmentation
- Admission control and backpressure
- Tenant-aware routing, quotas, and rate limits
- Correlation IDs, metrics, structured logs

The gateway should stay stateless per deployment unit so it can scale horizontally.

### 3. Worker Layer

Keep the worker private and CPU-light.

Responsibilities:

- Request validation from gateway
- Language resolution and LID
- Context-biasing policy
- Short-lived request orchestration
- Calling Triton
- Returning typed success/error responses

### 4. Triton Layer

Use Triton as the primary production inference plane.

Responsibilities:

- GPU model execution
- Model warmup and batching policy
- Versioned model rollout
- GPU metrics
- Canary and rollback support

### 5. Data and Control Plane

- `Redis`: distributed rate limiting, per-tenant quotas, short-lived session metadata if needed.
- `Postgres`: tenant registry, API credentials, plan limits, audit records, allowlists.
- `Object storage`: sampled eval artifacts, logs, exported manifests, phrase files, benchmark outputs.
- `Secrets manager`: API signing keys, Hugging Face tokens, Grafana credentials, TLS material.
- `CI/CD`: tests, image build, scan, deploy promotion, rollback.

## What Is Needed For Real Production

### A. Security and Access Control

- Replace static `WS_API_KEYS` with real auth.
- Support tenant-scoped API credentials or JWT/OIDC.
- Add credential rotation, expiration, revocation, and audit logging.
- Add per-tenant quotas and rate limiting.
- Put the public endpoint behind TLS and WAF.
- Move secrets out of `.env` files into a secret manager.
- Restrict worker and Triton to private network access only.

### B. Reliability and Failure Semantics

- Return explicit error responses for inference failure instead of transcript-like fallbacks.
- Separate liveness from readiness.
- Make readiness fail with non-200 when model load or Triton readiness is degraded.
- Add graceful connection draining on deploys.
- Add circuit breaking and retry policy between gateway and worker.
- Add bounded message size, session duration, and audio duration enforcement.
- Define SLOs for availability and end-to-end latency.

### C. Scalability

- Run multiple gateway replicas behind a WebSocket-capable load balancer.
- Run multiple GPU worker/Triton replicas across GPU nodes.
- Autoscale gateway on active connections, CPU, and p95 latency.
- Autoscale worker/Triton on queue wait, GPU utilization, and request latency.
- Introduce distributed rate limiting and quota enforcement.
- Benchmark per-GPU safe concurrency and set admission control from measured limits.

### D. ML Serving Discipline

- Standardize on Triton mode for production.
- Pin model versions and model configs per environment.
- Add shadow/canary rollout for model changes.
- Add pre-deploy WER and latency gates on held-out datasets.
- Store benchmark outputs and deployment decisions in artifacts.
- Keep context-biasing in `shadow` mode first, then promote by language/use case.

### E. Observability

- Keep Prometheus/Grafana, but add centralized logs and distributed traces.
- Propagate a single request/session/correlation ID from edge -> gateway -> worker -> Triton.
- Alert on auth failures, 5xx, queue wait, fallback rate, GPU saturation, and disconnect spikes.
- Record tenant, language, model version, and deployment version in logs and metrics.

### F. Platform and Delivery

- Add CI for tests, lint, build, image scan, and smoke tests.
- Add CD for staging -> canary -> production promotion.
- Add infrastructure as code for networking, compute, secrets, storage, and monitoring.
- Use immutable image tags and environment-specific config.
- Add backup/restore for stateful systems such as Postgres and Grafana data if retained.

### G. Compliance and Data Governance

- Decide whether audio and transcripts are stored, sampled, or never retained.
- Redact or hash PII in logs and evaluation outputs.
- Add retention windows and deletion workflows.
- Restrict access to sampled audio/eval artifacts.
- Add audit logging for credential and config changes.

## Immediate Gaps Found In This Repo

### 1. Auth is still static API-key matching

- Gateway auth is based on `WS_API_KEYS` loaded from environment.
- This is enough for local/dev, not for tenant-aware production auth.

Relevant files:

- `gateway/app/config.py`
- `.env.example`

### 2. Unauthorized sockets are accepted before auth rejection

- The WebSocket is accepted before API-key validation.
- In production, authentication should happen at or before upgrade completion when possible, with stricter connection admission and edge protections.

Relevant file:

- `gateway/app/main.py`

### 3. The gateway is not using a real streaming ASR provider

- The current runtime buffers an utterance and sends it to the worker only at finalization.
- This is fine for utterance streaming, but it is not true partial streaming.

Relevant file:

- `gateway/app/main.py`

### 4. Speaker verification backend is stubbed

- Gateway-side audio processing is intentionally a pass-through; noise suppression / AGC live in the worker `AudioPreprocessor` (denoise + RNNoise) and at the browser capture point (`getUserMedia` constraints). The gateway does **not** run a WebRTC APM. See [docs/audio/apm-decision.md](audio/apm-decision.md) and the end-to-end [noise-cancellation deep dive](audio/noise-cancellation-deep-dive.md).
- External speaker embedding backend is still TODO / not implemented.
- Speaker verification should not be marketed as production-ready until the real backend exists and is load-tested.

Relevant file:

- `gateway/app/main.py`

### 5. Worker fallback currently looks like a transcript

- On timeout/not-ready/inference failure, the worker can return `"text": "worker-fallback"`.
- In production this is risky because downstream consumers may treat it as a valid transcript instead of an error/degraded response.

Relevant file:

- `worker/app/main.py`

### 6. Health/readiness behavior needs hardening

- Gateway health always returns `ok`.
- Worker health returns degraded text but not an explicit non-200 failure.
- External orchestrators should be able to distinguish live, ready, and degraded states cleanly.

Relevant files:

- `gateway/app/main.py`
- `worker/app/main.py`

### 7. Current deployment defaults are single-process and intentionally small

- Gateway runs with one uvicorn worker.
- Worker runs with one uvicorn worker.
- Default concurrency is conservative.
- The README load test shows 50 connected sessions, but not 50 parallel transcriptions under the tested settings.

Relevant files:

- `docker-compose.yml`
- `gateway/Dockerfile`
- `worker/Dockerfile`
- `README.md`

### 8. Public ingress is still plain HTTP in repo default wiring

- Nginx listens on port 80 only in the current config.
- Browser microphone support also already hints that HTTPS/WSS is required for real deployment.

Relevant files:

- `nginx/nginx.conf`
- `frontend/src/lib/audioClient.ts`
- `README.md`

### 9. CI/CD automation is missing from the repo

- There is no `.github/workflows` pipeline in the repository today.
- That means tests, image builds, scans, smoke tests, and deploy promotion are not yet automated here.

## Recommended Production Topology

### Option A: Best target

Use Kubernetes/EKS or an equivalent orchestrator:

- `gateway` deployment on CPU nodes
- `worker` deployment on GPU nodes
- Triton as sidecar or sibling service on GPU nodes
- HPA/KEDA based on custom metrics
- PDBs, rolling deploys, readiness gates, network policy

### Option B: Transitional target

Use ECS or managed VM groups:

- ALB/NLB for WSS ingress
- gateway autoscaling group/service
- GPU worker service pool
- Triton on GPU instances
- Redis/Postgres/Object storage as managed services

### Option C: What not to keep long term

- Single-host Docker Compose as the primary production platform
- Static API keys in env
- Public HTTP-only ingress
- Implicit fallback transcripts

## Production Rollout Plan

### Phase 1: Hard blockers

- Replace auth with tenant-aware credentials.
- Fix error semantics so failures are not returned as transcript text.
- Implement proper readiness/liveness endpoints.
- Put ingress behind TLS/WAF/LB.
- Add CI and staging smoke tests.

### Phase 2: Stable scaling

- Move to Triton-backed serving as default.
- Add multiple gateway and GPU worker replicas.
- Add Redis-backed rate limits and quotas.
- Add centralized logs, traces, and alerts.
- Define and measure latency/error SLOs.

### Phase 3: Safer ML operations

- Add benchmark gates for WER and latency before deploy.
- Add canary/shadow model rollout.
- Add per-language and per-tenant routing controls.
- Promote context biasing only after shadow validation.

## Minimum Production Checklist

- WSS with valid certs
- Real auth and secret rotation
- Tenant quotas and rate limits
- Private worker/Triton network
- Non-200 readiness on degraded backends
- Explicit failure responses
- Horizontal scaling for gateway and GPU worker
- Observability with alerts
- CI/CD with staging
- Benchmark gates for model changes
- Data retention/redaction policy

## Practical Recommendation For This Repo

If you want the fastest credible path to production:

1. Keep the current `gateway -> worker -> Triton` shape.
2. Replace env API keys with tenant-aware auth.
3. Treat gateway as stateless and scale it horizontally.
4. Make worker a thin private adapter and standardize on Triton for inference.
5. Add Redis, Postgres, object storage, and secret management around the serving path.
6. Fix fallback and health semantics before exposing the system to external customers.

That keeps most of the current code structure, while adding the missing production control plane around it.
