# Durable composite Embedding + System-One GPU E2E (isolated)

Tracking: #122. Observed 2026-10-10 JST. This is a **public-safe aggregate**
from a disposable validation cluster. Raw request bodies, credentials, DB
connection settings, host identifiers, model outputs, full logs and GPU UUIDs
remain private.

## Candidate and execution boundary

- Candidate feature implementation: `82b49459d2452b899d6d0517c54ae44b7b510e2c`
  (Draft PR #123, not yet approved or integrated).
- Exact Nix-built Worker and Control packages at that commit.
- Both model runtimes use pinned CUDA 12.9 / SM61 llama.cpp;
  individual model SHA-256s retained from the frozen native study.
- One real 11-GiB NVIDIA SM61 GPU, two distinct owned loopback model
  servers, one composite Worker with `max_concurrency=1`.
- Distinct **disposable** PostgreSQL 16, Control, Worker, service ports,
  embedding and decision catalogs; legacy deployed services unchanged.
- Private bearer-token separation for Worker and Client. Both serving
  profiles bind one exact composite bundle identity but separate embedding
  space and System-One semantics identities.
- No shared production vector index, persistent GPU ownership change,
  production Control migration or automatic decision authority.

## Durably observed results

| Check | Result |
| --- | --- |
| PostgreSQL 16 isolated setup and 6 migrations | PASS |
| Control readiness and serving extension | PASS |
| Worker full dual-child GPU readiness and registration | PASS |
| Versioned embedding and decision profiles available | 2/2 PASS |
| Alternating embedding requests through Control / DB / Worker / GPU | 3/3 succeeded |
| Alternating native System-One requests through the same fabric | 3/3 succeeded |
| PostgreSQL durable terminal Job rows | 6/6 succeeded, no failed/queued |
| Composite simultaneous VRAM increase | 3,413 MiB |
| Missing Client bearer | 401 / rejected |
| Wrong embedding-space identity | 409 / rejected |
| Wrong decision-profile revision | 409 / rejected |
| Stale runtime-instance epoch claim | 409 / rejected |
| Worker and Control owned shutdown | PASS |
| All private test-only ports and model children released | PASS |
| GPU restored to 2-MiB idle baseline | PASS |

Client-side wall times for 3 observations per operation (milliseconds):
Embedding 110.577 / 77.163 / 78.081, median **78.081 ms**.
System-One 108.726 / 104.434 / 103.548, median **104.434 ms**.
These are **fabric-level HTTP timings** including Control, durable state,
Worker and model work, but are not a peak-load or fleet-capacity estimate.

The exact stored durable job counts were read directly from independent
PostgreSQL:
`text.embed / succeeded = 3` and `decision.system_one / succeeded = 3`.

The first private attempt already completed one embedding and one decision
durable Job before a **test assertion mistake** misread the Decision score
field; the provider actually returns `score`, not `probability`. An
intermediate retry stopped at a deliberate no-overwrite log gate. Both
attempts and their failure evidence were retained. The successful, isolated
final run above used a fresh per-attempt PostgreSQL database. These earlier
attempts are **not** counted in its six successes.

## Real child-death / resource-fencing negative test

A second, newly isolated PostgreSQL instance bound the same exact
candidate runtime and verified both children were ready before
unexpectedly killing **only the owned System-One native inference child**.
The observed behavior was:

- Worker switched `runtime_state=failed`, `runtime_available=false`,
  with latched failure, and `GET /ready` returned **503**.
- The Worker sent a fenced draining heartbeat; Control persisted
  `WorkerState=draining`.
- A new legitimate **serving-bound Embedding** request was refused with
  **HTTP 503** despite the other model having been healthy beforehand.
- PostgreSQL job count stayed **0** across the rejected request.
- Worker and Control ended, both private ports released and GPU memory
  returned to the original baseline.

One preliminary negative probe submitted a *legacy/unbound generic Job* and
observed HTTP 201: accepting such a queued generic Job is intentional
Control behavior, **not serving-deployment admission**. The final negative
check correctly exercised the versioned **serving-bound** Gateway request.
The initial probe is preserved privately, not misreported as a runtime
vulnerability.

## Trace integrity and limits

Only hash commitments to private mode-0600 JSON evidence are included here:

- Successful durable 6/6 E2E: SHA-256
  `7c70da51ed650429ce05449feb7c0db8f788693feb5346ae16b02012379c8769`.
- Real owned System-One-child death and fail-closed serving admission:
  SHA-256
  `2912a16c3ce96c04d9c4c4db158a4f9098a44f24d4676d8eba5d739b8b560c02`.

**Disposition: exact-candidate distributed durable E2E and one-child-death
fencing PASS.** Independent human review, on-host reversible deployment
review, generalized cancellation, queue-pressure and repeated peak-window
testing remain separate gates; they are **not accepted** by this report.
Decision remains **OBSERVED_ONLY**, uncalibrated and recommendation-only.
No subscription changes, `main` merge, production Worker/GPU activation,
or v0.1 manual Real Smoke #80 acceptance are claimed.
