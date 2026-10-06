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

### Optional v1 extensions

Control advertises the optional extensions it supports from `GET /v1/health`
and `GET /v1/ready`. Unknown extensions fail closed.

#### Serving extension negotiation

Deployment-bound serving uses the optional v1 extension:

```text
serving-bindings-v1
```

Control advertises supported extensions from `GET /v1/health` and
`GET /v1/ready`. A Worker that intends to register a serving deployment must
check `/v1/ready` first and must not send serving registration state to a
Control that does not advertise this extension.

Requests carrying serving identity or a runtime-instance epoch include:

```json
{
  "protocol_version": "v1",
  "extensions": ["serving-bindings-v1"]
}
```

Unknown extensions fail closed. Worker-side protocol parsing also rejects a
response that contains serving identity without the extension marker. Legacy v1
Workers and direct Jobs continue to omit the extension and retain their existing
empty-body claim behavior.

A serving Job snapshots an immutable `ServingJobBinding` plus an absolute
timezone-aware deadline. Admission has three explicit bounded failure classes:

- HTTP 408: the request deadline has already expired;
- HTTP 429: compatible fresh ONLINE replicas exist but all are at capacity;
- HTTP 503: no fresh ONLINE Worker currently satisfies the exact serving
  deployment/contract plus generic Job requirements.

Admission compatibility is not a reservation. At claim, the Worker supplies its
current runtime-instance epoch; Control atomically rechecks Worker liveness,
state, generic requirements, serving binding and capacity before persisting the
attempt identity. Heartbeats, lifecycle writes and terminal writes for a serving
Worker are fenced by the same epoch, so a restarted Worker invalidates its stale
predecessor even when the immutable deployment revision did not change.

Idempotency is resolved before transient liveness/capacity/deadline admission
checks. An equivalent retry of an already-admitted request returns the same
durable Job; reuse of the key for different resolved serving intent is a
conflict.

#### Fenced Job event extension

Live incremental output uses:

```text
job-events-v1
```

This extension is workload-agnostic. Control does not interpret chat tokens,
tool-call fragments or other event payload semantics.

A Worker appends one event with:

```text
POST /v1/workers/{worker_id}/jobs/{job_id}/events
```

and includes `job-events-v1`, the current lease token, and, for serving Jobs,
the same `serving-bindings-v1` runtime-instance epoch used by heartbeat and
terminal writes. Control accepts the append only while that exact attempt is
RUNNING and the lease/deadline/Worker/epoch fences remain current.

Every durable event records Job ID, attempt number, monotonically increasing
per-Job sequence, Worker ID, runtime-instance epoch when applicable, opaque kind
and payload, and creation time. PostgreSQL serializes appends by locking the Job
row before allocating the next sequence.

Event buffers are bounded by per-event bytes, event count, total retained bytes,
and read page size. An overflow is rejected rather than dropping or bypassing
Control. Streaming executors treat rejected publication as a failed attempt.

The gateway reads events internally from the same Control repository; there is
no public client event-append authority and no direct runtime-to-client proxy.

## Authorities

v1 keeps the Worker trust domain bearer-authenticated and makes Client API
bearer authentication an explicit deployment choice.

### Client API authentication

Control uses:

```toml
[control]
client_auth = "bearer" # or "none"
```

If `client_auth` is omitted, the effective mode is `bearer`. This is the
secure upgrade/default behavior.

#### `client_auth = "bearer"`

Configured with:

```text
ASTRUMWEAVER_CLIENT_TOKEN
```

Client authority may:

- submit jobs
- inspect full jobs/results
- cancel jobs

The Client token may not register Workers, claim work, heartbeat a Worker, or
write fenced terminal results.

#### `client_auth = "none"`

The Client job endpoints do not require an AstrumWeaver bearer token:

```text
POST /v1/jobs
GET  /v1/jobs/{job_id}
POST /v1/jobs/{job_id}/cancel
```

`ASTRUMWEAVER_CLIENT_TOKEN` is not required or consulted in this mode. The
operator is explicitly delegating Client API access control to the deployment
boundary, for example a trusted network, VPN, reverse proxy, or upstream
identity layer.

This does **not** weaken Worker authorization. Supplying the Worker token on an
otherwise unauthenticated Client endpoint does not create a new cross-domain
authority; those endpoints are simply open according to the selected
`client_auth = "none"` policy.

### Worker authority

Configured on Control and Worker with:

```text
ASTRUMWEAVER_WORKER_TOKEN
```

Worker authority is always mandatory and may:

- register Workers
- heartbeat and renew an active lease
- change Worker lifecycle state
- claim work
- inspect restricted status for a Worker-visible job
- submit fenced completion/failure

Worker endpoints always require Worker authority regardless of Client auth
mode. A Client token cannot call Worker endpoints. In bearer mode, the Worker
token cannot submit, fully inspect, or cancel Client jobs.

v1 uses one Worker authority token for the Worker trust domain. This means
possession of that token grants Worker-plane authority across the deployment.
Per-Worker cryptographic identities are outside the v1 scope and may be
introduced by a later protocol version.

When Client auth is `bearer`, the Client and Worker tokens must be distinct.

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

Readiness requires PostgreSQL to be reachable, the AstrumWeaver control tables to exist, and every SQL migration packaged with the running Control binary to be recorded in `schema_migrations`. A Control binary whose required migration set has not been applied returns 503 rather than advertising Ready.

### Client job operations

```text
POST /v1/jobs
GET  /v1/jobs/{job_id}
POST /v1/jobs/{job_id}/cancel
```

These require Client authority only when `control.client_auth = "bearer"`.
They are unauthenticated by AstrumWeaver when `client_auth = "none"`.

### Worker registration/lifecycle

```text
POST /v1/workers/register
POST /v1/workers/{worker_id}/heartbeat
POST /v1/workers/{worker_id}/state
```

These require Worker authority.

### Worker event publication

```text
POST /v1/workers/{worker_id}/jobs/{job_id}/events
```

This requires Worker authority plus `job-events-v1`. For a serving attempt,
the body also carries `serving-bindings-v1` and the current runtime-instance
epoch. HTTP 409 indicates stale/cancelled/expired attempt authority; HTTP 429
indicates that the configured bounded event buffer cannot accept another event.

### Worker claim/status

```text
POST /v1/workers/{worker_id}/jobs/claim
GET  /v1/workers/{worker_id}/jobs/{job_id}
```

Legacy Workers use the existing empty claim request. A serving Worker sends its
`runtime_instance_epoch` with `serving-bindings-v1`; a stale or missing epoch
cannot claim for a serving registration. An empty eligible queue returns HTTP
204 with no body.

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

A local `SIGUSR1` is a non-terminating drain request: the process stops claiming locally and mirrors DRAINING into Control while allowing the current job to finish. This is the primitive used by [Borrowable GPU Worker](borrowable-worker.md).

## Control configuration

Example non-secret TOML:

```toml
[control]
host = "127.0.0.1"
port = 9000
client_auth = "bearer"
worker_ttl_seconds = 60
lease_seconds = 300
maintenance_interval_seconds = 5
event_max_count = 4096
event_max_payload_bytes = 65536
event_max_total_bytes = 4194304
event_max_read = 512
access_log = false
```

Secrets/authority are environment variables:

```text
ASTRUMWEAVER_DATABASE_URL
ASTRUMWEAVER_WORKER_TOKEN
ASTRUMWEAVER_CLIENT_TOKEN   # required only for client_auth = "bearer"
```

`ASTRUMWEAVER_WORKER_TOKEN` remains mandatory in both Client auth modes.

Production Control never falls back to an in-memory repository.

### Optional Stage A chat gateway

#94 Stage A may mount the bounded non-streaming edge surface on the same
Control process:

```toml
[chat_gateway]
enabled = true
catalog = "/etc/astrumweaver/chat-catalog.json"
poll_interval_seconds = 0.05
```

When omitted or `enabled = false`, existing v1 Control routes are unchanged.
The catalog contains only reviewed logical serving profiles and their resolved
#93 deployment/contract identities. `GET /v1/models` exposes those profile IDs,
not runtime model discovery.

The Stage A routes reuse the configured Client authentication policy:

```text
GET  /v1/models
POST /v1/chat/completions
```

`POST /v1/chat/completions` is non-streaming only. It compiles the validated
request into an opaque durable `llm.chat` Job, waits within the profile
deadline, and cancels only that owned Job on gateway timeout or disconnect.
`stream=true` remains unsupported until the separate #94 Stage B fenced event
channel exists.

### Optional embedding gateway

#95 may mount the bounded embedding edge surface independently of the chat
gateway:

```toml
[embedding_gateway]
enabled = true
catalog = "/etc/astrumweaver/embedding-catalog.json"
poll_interval_seconds = 0.05
```

When enabled, the configured Client authentication policy protects:

```text
GET  /v1/embedding-spaces
POST /v1/embeddings
```

`GET /v1/embedding-spaces` is an AstrumWeaver namespaced discovery surface.
It exposes configured logical profile IDs, immutable `embedding_space_id`
values, resolved deployment/contract/profile revisions, dimensions,
pooling/normalization policy, supported input roles, and effective limits. It
does not present these fields as standard OpenAI model metadata.

The first `POST /v1/embeddings` subset accepts one string or a non-empty
array of strings, `encoding_format = "float"` (or omission), and the required
`x_astrumweaver_input_type` value `query` or `document`. Token-array input,
base64 encoding, caller-selected dimensions, mixed input policies and unknown
fields fail closed.

Each accepted request compiles to a durable `text.embed` Job carrying the
exact #93 serving binding. The Worker locally revalidates the claimed serving
contract semantic revision before the provider adapter executes. Batch count,
per-item/aggregate byte limits and request bytes are checked at the gateway;
per-item and aggregate token limits are rechecked with the active runtime
tokenizer. Provider results must preserve input ordering and contain finite
vectors of the configured dimension. A batch is all-or-error.

Changing model, quantization, tokenizer, pooling, normalization, dimensions,
query/document preprocessing, adapter identity or deployment revision changes
the immutable embedding-space identity. Indexing and reindexing remain client
responsibilities.

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

A serving Worker additionally points at an operator-reviewed serving deployment
declaration:

```toml
[serving]
manifest = "/etc/astrumweaver/serving.json"
```

The same path may be supplied explicitly with
`astrumweaver-worker --serving-manifest ...`, which overrides the TOML value.
The declaration contains the immutable `DeploymentIdentity` and the exact
`ServingContract` set to advertise. It is configuration provenance inside the
existing trusted operator/Worker boundary, not cryptographic attestation of the
referenced artifacts.

The daemon parses this declaration before registration, verifies that every
declared contract capability is also in `worker.capabilities`, and, when a
managed RuntimeProvider deployment is used, requires its provider ID to match
the runtime deployment. It creates the per-start `RuntimeInstance` epoch only
after the selected executor/runtime has been prepared and its capabilities have
been validated. Restarting the Worker therefore preserves the immutable
deployment revision but produces a fresh epoch.

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

The daemon repeats GPU ownership preflight before registration unless
`gpu_preflight = false` is explicitly configured. Normal deployment should
leave it enabled.

Deployment selects one explicit preflight mode:

- `exact-visible` (default): raw NVIDIA-visible UUIDs must exactly equal the
  Worker GPU contract.
- `isolated-access`: only a reviewed deployment may select this mode. The
  Worker verifies the reviewed UUID/device map, configured CUDA UUID order, and
  actual physical device access; selected nodes must open and any unselected
  visible physical node must be denied.

`CUDA_VISIBLE_DEVICES` alone does not select or prove isolated mode.

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

The resulting executor must also expose a non-empty `capabilities` collection. Before registration, the Worker verifies that every capability it intends to advertise is included in that executor declaration. A mismatch fails startup rather than allowing the Worker to claim unsupported work.

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
