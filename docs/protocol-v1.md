# Control / Worker Protocol v1

AstrumWeaver v1 uses an HTTP/JSON protocol between the Control Plane, clients, and Workers.

The protocol is workload-agnostic. Control understands capabilities, resource requirements, durable job state, and generic results; it does not understand LLM-, image-, TTS-, or application-specific payload semantics.

## Versioning

The HTTP namespace is:

```text
/v1/...
```

JSON mutation requests include:

```json
{
  "protocol_version": "v1"
}
```

A body declaring another protocol version is rejected.

Responses include `protocol_version: "v1"` where applicable.

Breaking transport changes require a new protocol namespace rather than silently changing v1 semantics.

## Authorities

v1 distinguishes two bearer-token authorities.

### Client authority

Configured with:

```text
ASTRUMWEAVER_CLIENT_TOKEN
```

Client authority may:

- submit jobs
- inspect full jobs/results
- cancel jobs

It may not register Workers, claim work, heartbeat a Worker, or write fenced terminal results.

### Worker authority

Configured on Control and Worker with:

```text
ASTRUMWEAVER_WORKER_TOKEN
```

Worker authority may:

- register Workers
- heartbeat and renew an active lease
- change Worker lifecycle state
- claim work
- inspect restricted status for a Worker-visible job
- submit fenced completion/failure

It may not submit or cancel client jobs and may not use the full client job-read endpoint.

v1 uses one Worker authority token for the Worker trust domain. This means possession of that token grants Worker-plane authority across the deployment. Per-Worker cryptographic identities are outside the v1 scope and may be introduced by a later protocol version.

The client and Worker tokens must be distinct.

## TLS boundary

The built-in Control daemon serves HTTP through Uvicorn.

For traffic that leaves a trusted host/network boundary, deploy TLS using a reviewed reverse proxy, service mesh, VPN, or equivalent transport-security layer.

Bearer tokens must not be sent over an untrusted plaintext network.

AstrumWeaver does not configure Proxmox networking or TLS infrastructure itself.

## Control endpoints

### Health

```text
GET /v1/health
```

Unauthenticated process-liveness endpoint.

### Readiness

```text
GET /v1/ready
```

Readiness requires the configured PostgreSQL service to be reachable and the AstrumWeaver `workers` and `jobs` schema to exist.

### Client job operations

```text
POST /v1/jobs
GET  /v1/jobs/{job_id}
POST /v1/jobs/{job_id}/cancel
```

These require client authority.

### Worker registration/lifecycle

```text
POST /v1/workers/register
POST /v1/workers/{worker_id}/heartbeat
POST /v1/workers/{worker_id}/state
```

These require Worker authority.

### Worker claim/status

```text
POST /v1/workers/{worker_id}/jobs/claim
GET  /v1/workers/{worker_id}/jobs/{job_id}
```

An empty claim returns HTTP 204 with no body.

The Worker job-status endpoint is intentionally restricted and does not expose the full payload/result record used by the client endpoint.

### Fenced terminal operations

```text
POST /v1/workers/{worker_id}/jobs/{job_id}/complete
POST /v1/workers/{worker_id}/jobs/{job_id}/fail
```

Both require the current lease token.

A stale worker, expired attempt, wrong Worker, or previous fencing token receives a conflict response and cannot finalize the durable job.

## Claim contract

A successful claim includes:

- job ID
- capability
- opaque executor payload
- attempt number
- current lease token
- lease expiry

The lease token identifies one specific attempt.

If the job is recovered and claimed again, the new attempt receives a new token.

The old token never becomes valid again.

## Heartbeat contract

A Worker without active work may heartbeat only its Worker identity.

A Worker renewing active work supplies both:

- active job ID
- current lease token

Control does not trust a Worker-reported active-job count. Durable running-job ownership remains authoritative.

## Cancellation

Client cancellation is durable and clears scheduler ownership.

A Worker executing that attempt will eventually see its fenced heartbeat rejected. It then uses the restricted job-status endpoint.

If the durable status is `cancelled`, the Worker:

1. calls `JobExecutor.cancel(job_id)`
2. stops that local execution
3. does not submit a stale completion/failure

A terminal-write race is also fenced: if execution finishes at the same time as cancellation, a completion conflict is treated as a discarded stale attempt only when Control confirms the job is cancelled.

## Worker lifecycle

v1 Worker states are:

```text
ONLINE
DRAINING
OFFLINE
```

- ONLINE: may claim work.
- DRAINING: may finish/heartbeat current work but receives no new claims.
- OFFLINE: receives no work and releases advertised ownership when no active job remains.

The Worker daemon uses DRAINING during graceful shutdown before moving OFFLINE.

## Control configuration

Example non-secret TOML:

```toml
[control]
host = "127.0.0.1"
port = 9000
worker_ttl_seconds = 60
lease_seconds = 300
maintenance_interval_seconds = 5
access_log = false
```

Secrets/authority are environment variables:

```text
ASTRUMWEAVER_DATABASE_URL
ASTRUMWEAVER_CLIENT_TOKEN
ASTRUMWEAVER_WORKER_TOKEN
```

Production Control never falls back to an in-memory repository.

## Database migration

Apply packaged migrations explicitly:

```sh
ASTRUMWEAVER_DATABASE_URL='postgresql://...' astrumweaver-migrate
```

The NixOS module can optionally perform this before Control startup:

```nix
services.astrumweaver.control.migrateOnStart = true;
```

It defaults to `false` because database schema mutation is an explicit deployment authority.

## Worker configuration

Example CPU/non-GPU Worker:

```toml
[worker]
id = "worker-example"
class = "cpu"
control_url = "https://control.example.invalid"
capabilities = ["debug.echo"]
gpu_uuids = []
gpu_count = 0
total_vram_mb = 0
max_single_gpu_vram_mb = 0
max_concurrency = 1
poll_interval_seconds = 1
heartbeat_interval_seconds = 5
health_host = "127.0.0.1"
health_port = 9100

[executor]
factory = "astrumweaver.executors.structured_echo:create_executor"

[executor.settings]
```

Example GPU Worker resource section:

```toml
[worker]
id = "worker-gpu-example"
class = "modern-single"
control_url = "https://control.example.invalid"
capabilities = ["llm.chat"]
gpu_uuids = ["GPU-example"]
gpu_count = 1
total_vram_mb = 24576
max_single_gpu_vram_mb = 24576
```

The daemon repeats exact guest-visible GPU UUID preflight before registration unless `gpu_preflight = false` is explicitly configured. Normal deployment should leave it enabled.

Worker authority is provided only through:

```text
ASTRUMWEAVER_WORKER_TOKEN
```

## Executor plugin contract

The Worker loads its executor through:

```text
module:attribute
```

The attribute may be:

- an existing object satisfying `JobExecutor`, or
- a synchronous factory receiving an executor-settings mapping and returning `JobExecutor`

The Worker never imports application-specific executors in Control.

The built-in `structured_echo` executor exists for smoke/integration validation, not as a workload-specific scheduler rule.

## Worker local health

The Worker daemon exposes a local diagnostic HTTP server, defaulting to:

```text
127.0.0.1:9100
```

Endpoints:

```text
GET /health
GET /ready
```

`/ready` returns ready only after successful Worker registration and before shutdown begins.

## Concurrency in v1

The durable scheduler model supports `max_concurrency`, but the initial v1 Worker daemon executes one active job at a time and therefore requires:

```text
max_concurrency = 1
```

Parallel execution can be introduced later without changing the durable lease/fencing model.

## Non-goals

The transport/runtime does not:

- create a VM or LXC
- call the Proxmox API
- configure PCI passthrough or IOMMU
- infer identity from VMID/CTID
- hard-code Ollama
- route based on application-specific conditionals
- allow stale attempts to write terminal state
