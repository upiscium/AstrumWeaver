# Worker liveness and Control acknowledgement

This contract covers Worker heartbeats and job-attempt authority.
[Post-start RuntimeProvider supervision](runtime-supervision.md) adds a separate
health gate; Control liveness is not proof that an owned model process is healthy.

## Cadence

The Worker keeps one monotonic heartbeat deadline across the entire execution
loop. Neither claiming nor finishing a job resets it. The deadline is checked
before another claim, while waiting for an active execution, and before its
terminal write when due. Idle and draining waits are capped by the same deadline.
A continuous queue of short jobs and an empty queue therefore both send
heartbeats; `poll_interval_seconds` does not delay a due heartbeat.

There is one execution owner, not an independent heartbeat writer. Heartbeat,
claim, terminal write and signal-driven drain RPCs are serialized under one
async lock. Once a completion/failure is acknowledged, the local attempt is
cleared before a waiting drain can issue another heartbeat. Every active
heartbeat carries that attempt's job ID and lease token together.

The relevant existing Worker settings are:

```toml
[worker]
poll_interval_seconds = 1.0
heartbeat_interval_seconds = 5.0
request_timeout_seconds = 10.0
```

The request timeout now also bounds total asynchronous Control request time,
not just HTTP read/connect phases. Configure the heartbeat interval and request
budget with ample margin below **both** Control's `worker_ttl_seconds` and
`lease_seconds`, accounting for serialized request latency and scheduling
jitter. The defaults are intended for the default Control TTL/lease settings;
shortening a Control deadline requires reviewing Worker timings too.

This is not hard real-time scheduling. A blocked event loop, uncooperative
executor, host suspension, prolonged network outage, or database stall can
still exceed a TTL/lease. Executors must yield during asynchronous work and
honor cancellation. Control remains the final authority for TTL and lease
fencing; Worker heartbeats never disable those checks.

## Readiness and state

The Worker validates registration/heartbeat acknowledgements against its
protocol version, Worker specification and concurrency contract. `/ready`
requires a fresh matching acknowledgement with Control state ONLINE, no local
stop/drain request, a known registration and (for RuntimeProvider-backed Workers)
a fresh runtime-health gate. The freshness budget is the
heartbeat interval plus the request timeout, measured from request start,
not from delayed response arrival. This is a local fail-closed freshness
budget, not a replacement for the server's independently configured TTL.

`/health` additionally reports:

- `control_state`: the last acknowledged `online`, `draining`, `offline`, or null;
- `control_available`: whether that acknowledgement is still fresh.

On transport/protocol failure, readiness is immediately withdrawn. Missing or
rejected Worker authority also clears local `registered`. The loop pauses
claims and retries heartbeat on its normal bounded cadence. It never retries
registration or sets ONLINE automatically while running. A successful later
matching heartbeat can restore the acknowledgement, but only the state
actually returned by Control can permit claims.

An OFFLINE acknowledgement is **not** silently converted to ONLINE. The Worker
keeps sending state-less heartbeats and stays not-ready. This preserves both
operator intent and transactional GPU ownership. A DRAINING acknowledgement
also stops new claims while an already admitted attempt may finish. Local
SIGUSR1/drain intent is latched for this service invocation and cannot be
cleared by a late ONLINE response.

A claim already in flight when drain is requested may have been admitted by
Control; that one attempt is finished under its lease, but no subsequent claim
is initiated. Graceful shutdown retains its existing drain/grace policy.

## Operator-controlled recovery

First inspect the Worker journal and local health endpoint. State-change and
lost-acknowledgement messages contain status/state, not tokens or raw Control
response bodies. Do not publish raw journals or private health payloads.

For an OFFLINE or remotely DRAINING Worker whose cause has been resolved,
explicitly return the existing identity ONLINE through Control's Worker state
endpoint (see [protocol-v1.md](protocol-v1.md)). For example, using locally set
`ASTRUMWEAVER_CONTROL_URL`, `WORKER_ID` and protected Worker credentials:

```sh
curl -fsS -X POST \
  -H "Authorization: Bearer $ASTRUMWEAVER_WORKER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"protocol_version":"v1","state":"online"}' \
  "$ASTRUMWEAVER_CONTROL_URL/v1/workers/$WORKER_ID/state"
```

This endpoint requires the **Worker** token even when Client job endpoints use
`client_auth = "none"`. Do not use a Client token or direct SQL as a substitute.
An ownership conflict must be resolved deliberately; do not repeatedly force
registration or overwrite another Worker's topology. The Worker observes the
accepted ONLINE transition on its next heartbeat.

If local drain was requested, credentials/configuration changed, or registration
was removed, use the documented ordinary service stop/start procedure after
resolving the cause. Startup registration again goes through Control's ownership
checks. Do not use a restart to bypass an unresolved conflict or conceal a
liveness failure. See [borrowable-worker.md](borrowable-worker.md) for local
personal/share ownership transitions.

## Active-attempt uncertainty

A failed lease heartbeat or unconfirmed Control authority cancels the local
execution instead of continuing with an unverified lease. No synthetic terminal
failure is sent merely because a Control request failed. Durable cancellation,
lease expiry/recovery and fencing decide the recorded attempt's outcome; new
claims cannot proceed without renewed Control acknowledgement and server-side
capacity/ownership approval. A known remote cancellation is respected without
writing an obsolete completion.

The Worker invokes the executor's cancellation hook and awaits cancellation of
its task. Daemon shutdown/error cleanup also joins its Worker/health/signal
tasks. This does not claim that arbitrary external side effects or a
non-cooperating executor are reversible.

## Release verification

Software regressions cover sustained short-job backlog beyond two TTL windows,
idle/draining heartbeat, explicit OFFLINE recovery and GPU conflicts, active
lease renewal/cancellation, delayed RPCs, and drain/completion/shutdown races.
These do not satisfy #80: the operator-led documentation-only Real Smoke still
requires separate final-candidate installation, runtime E2E and human sign-off.
