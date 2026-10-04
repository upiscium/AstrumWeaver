# Post-start RuntimeProvider supervision

A running Worker parent is not proof that its owned model server is healthy.
Runtime-backed Workers require both fresh Control acknowledgement (see
[Worker liveness](worker-liveness.md)) and fresh RuntimeProvider health before
advertising `/ready` or beginning another claim. Smoke/custom executors without
a ManagedRuntime keep their existing Control-liveness checks; `runtime_state`
is `unmanaged` for them, not a claim of provider health.

## Monitoring policy

After initial RuntimeProvider startup, the Worker starts one supervisor before
registration. The first check must succeed before registration/claims. The
v0.1 daemon policy uses a one-second probe interval and a five-second total
asynchronous probe budget, including any wait for a preceding probe. These are
fixed daemon defaults, not environment-overridable installer state. Constructor
timing overrides support tests/embedding only.

The supervisor invokes the provider's existing health contract, covering its
owned child, health endpoint and configured model identity. All five first-class
providers recheck the child after awaited model inventory, rejecting a success
from a child that exited during the request. Runtime health and Control heartbeat
are separate operations; model-server health does not replace Worker-token
validation, lease renewal or GPU ownership checks.

Health probes are serialized and bounded. A successful probe is fresh for at
most interval + timeout (six seconds with daemon defaults), measured from probe
start, not delayed response arrival. Expiry closes admission even before the
monitor resumes. Invalid results, not-ready states, exceptions, timeout,
staleness and unexpected monitor termination all latch failure. A late success
cannot reopen the gate. Executor errors request a serialized immediate probe
before reporting an ordinary job error or considering another job. An invalid
job on a demonstrably healthy runtime still uses the ordinary fenced job-failure
path rather than automatically quarantining the Worker.

Detection is sampled, not instantaneous or hard real-time. A runtime can fail
between observations; a blocked event loop, uncooperative custom provider or
executor, suspended host or stalled kernel cannot be bounded by asynchronous
timeouts alone. The runtime contracts must cooperate with cancellation. A
responsive event loop detects faults within the probe interval and request
budget; these budgets are not a claim about GPU hardware fault recovery.

## Quarantine instead of a hidden restart loop

An observed fault closes readiness and claim admission for the current Worker
invocation. The parent remains running, locally observable, and not-ready. It
does not exit merely to trigger systemd `Restart=on-failure`, automatically
restart the runtime, re-register itself or force ONLINE. This avoids repeatedly
spending queued jobs' retry budgets on an unhealthy model server.

Control heartbeats continue at the normal bounded cadence. The existing guarded
drain transition advertises DRAINING where allowed, without promoting OFFLINE
or bypassing transactional GPU ownership. If Control cannot be reached, local
readiness remains false and Control's TTL/lease rules remain authoritative.
The Worker retains responsibility for its owned runtime until ordinary service
shutdown. A still-alive degraded model server is not adopted by another Worker
or forcibly released as a side effect of a failed health probe.

`GET /health` includes these local diagnostic fields:

- `runtime_state`: `unmanaged`, `starting`, `ready`, `failed`, or `stopped`;
- `runtime_available`: current runtime-health gate;
- `runtime_failure`: a fixed reason code or null (no raw provider detail).

The usual `ready` and `/ready` now include this gate. An HTTP 200 `/health` only
says the Worker parent is serving diagnostics; HTTP 503 `/ready` is expected
while quarantined. Registration can remain true while runtime availability is
false. Do not interpret `systemctl is-active` alone as successful operation.

## Active and in-flight work

WorkerRuntime remains the single owner of executor cancellation and task
joining. A failed health gate wakes the active execution path; cancellation may
also wait for the currently bounded Control RPC. It cancels unfinished local
work without inventing a new terminal failure from runtime-health uncertainty.
Control lease expiry, recovery and stale-token rejection determine that
attempt's durable outcome. The cancellation hook plus local task cancellation
cannot undo arbitrary external side effects or guarantee a third-party server
has stopped already-admitted computation.

A claim already sent before fault detection may still be admitted by Control.
Its response is checked before execution, and an unconfirmed runtime does not
execute it or claim another job. That one admission/attempt is not retroactively
undone. Similarly, a fenced terminal write already in flight may have committed;
the Worker does not erase that result or fabricate a rollback. No terminal write
is initiated after observing runtime-health loss, and queued work not admitted
by such an in-flight claim retains its attempt budget on this Worker. Other
healthy Workers may still claim eligible queued/recovered work normally.

## Operator-controlled recovery

1. Inspect the local Worker health endpoint and the owning service journal:

   ```sh
   curl -fsS http://127.0.0.1:9100/health
   journalctl -u astrumweaver-worker.service -b
   ```

   Adapt the documented local health address if it was configured differently.
   Keep raw journals and private health payloads local; providers may log paths,
   model identity or request details even though the supervisor only logs fixed
   failure reason codes.

2. Resolve the runtime failure (for example, missing model, exhausted resources
   or failed runtime binary). Do not switch Control ONLINE or rerun setup merely
   to hide a quarantined runtime. Restoring the endpoint by itself does not clear
   the latched gate. Retain any needed diagnostic evidence before restart.

3. Use the documented ordinary restart procedure once the cause is resolved:

   ```sh
   sudo systemctl restart astrumweaver-worker.service
   ```

   Shutdown first joins execution and monitoring. Owned runtime stop/release has
   a total budget from `[runtime].shutdown_timeout_seconds` (default 60 seconds)
   and does not wait on the failed health endpoint before attempting stop. Control
   is marked OFFLINE only after owned-runtime cleanup succeeds; failed cleanup
   does not deliberately release an idle GPU reservation. Only
   owned children are targeted; an external server is not killed/adopted. Ollama
   does not send model-unload requests to a foreign endpoint during release.
   Cancellation-resistant providers still require the service manager's final
   process cleanup; a shutdown error is not reported as successful release.

4. Verify Control acknowledgement, runtime/model health, `/ready`, and a harmless
   Control job again using the [getting-started checks](getting-started.md).
   Startup repeats runtime validation and registration through Control's GPU
   ownership protocol. An unresolved ownership conflict still rejects startup.
   Old active attempts can occupy capacity until fenced recovery completes;
   a restart does not reset their attempts, leases or history.

Worker token authentication remains mandatory. Client jobs may continue to use
explicit `client_auth = "none"`; no Client token is introduced by supervision.

Software fault-injection, fake model servers and real OS-child lifetime tests
are not GPU hardware acceptance. Final release still requires the separate
operator-led documentation-only Real Smoke gate (#80) on the final revisions.
