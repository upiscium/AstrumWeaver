# Serving contracts and durable admission

Tracking: #91 / #93. Design baseline: #92 at
`2d34d72db1643c01a1782b00b8058fbedd942fc9`.

The pure-domain slice is developed in #98. The durable admission slice is
stacked in #99 on `feat/93-serving-admission`. Both target the isolated
capability-serving feature series rather than `main`. The #79/#89/#90
release-repair track and operator-only #80 gate remain separate. Do not merge
feature code into the release candidate implicitly.

## Boundary

`astrumweaver.serving` contains immutable identity/contract/profile values and
side-effect-free profile validation. It introduces no new Worker/GPU owner,
service, API route, database migration, registration field or runtime behavior.
Existing package entrypoints and core contracts do not import this module.

A successful profile resolution is still only a **configuration snapshot**.
The second slice persists an exact serving binding into the existing Worker/Job
records, negotiates it through the explicit `serving-v1` extension, requires a
bounded deadline, checks compatible ONLINE Workers at admission, rechecks exact
deployment/contract identity during the existing atomic pull claim, and binds
each running attempt to the registered runtime-instance epoch. No second
scheduler, Slot registry or GPU ownership authority is introduced.

## Identity and validation

Deployment identity binds provider, runtime/adapter/model artifacts,
quantization, optional tokenizer/template artifacts and a reviewed execution
configuration digest. SHA-256 values identify content, not mutable names/paths.
An artifact digest may name a reviewed canonical multi-file manifest.

Raw runtime configuration is deliberately not accepted here. The caller must
construct execution configuration identity from an allowlisted, non-secret
representation. Hashing a secret does not make it public-safe. This layer cannot
verify that a supplied digest really names reviewed content or evidence.

Serving contracts bind an operation/schema to one deployment revision, declared
verified features, numeric upper bounds, optional semantic identity and an
explicit validation-evidence digest. Evidence references are provenance within
the existing trusted Worker/operator boundary, not cryptographic attestation.
Only later provider integration can establish that evidence is valid/current.

Profiles pin an exact contract and deployment. Resolution rejects mismatches,
unknown features/usage keys, excessive bounds and incompatible explicit
provider/model selections. There is no recommendation or fallback mode. Limits
are generic inclusive upper bounds; operation-specific tokenization, aggregate
context arithmetic and vector-space semantics remain adapter responsibilities.

Value constructors reject unknown versions, noncanonical identifiers/digests,
boolean or noninteger quantities, duplicate features and malformed collections.
Mappings are defensively copied and read-only; set-like features are frozen and
sorted only for canonical serialization. Returned dictionaries are detached.
Revisions are domain-separated by schema identifier and SHA-256 over UTF-8 JSON
with sorted keys, compact separators, ASCII escaping and nonfinite values
forbidden. Runtime instance epochs are separate canonical UUIDs supplied by the
caller; constructing a value neither starts a runtime nor verifies uniqueness.

## Validation scope

Run `python -m pytest tests/test_serving_contracts.py` in the repository's normal
Python environment. Tests cover canonical identities, defensive immutability,
version/type rejection, exact binding, explicit selections, limits and snapshot
stability. The complete repository CI remains required for package integration.
These are not PostgreSQL concurrency, GPU, gateway, OpenCode or model-quality
acceptance tests. They satisfy no release or operator hardware gate.


## Durable admission slice

A serving-capable Worker advertises one deployment revision, one per-process
runtime-instance epoch and the serving-contract revision for each advertised
capability. The record remains inside the existing authenticated Worker trust
domain; this is not remote attestation.

A serving-bound Job snapshots profile, deployment, contract, capability and
operation-schema identity. It also requires `deadline_at`. The binding is
immutable for that Job even if operator profile configuration later changes.

Admission is intentionally bounded:

- a deadline already reached is rejected;
- no fresh ONLINE Worker matching generic requirements plus the exact
  deployment/contract binding returns a no-compatible-deployment error;
- if compatible Workers exist but all are at declared concurrency, admission
  returns overload rather than creating an unbounded interactive queue;
- an idempotency key is equivalent only when payload, requirements, retry
  policy, serving binding **and deadline** are equivalent.

Admission is advisory with respect to later availability. Atomic claim remains
authoritative. Claim rechecks Worker liveness/state/capacity, generic resources,
the exact serving deployment/contract and the current runtime-instance epoch.
The lease is capped at the Job deadline. Terminal completion/failure and active
heartbeat renewal must present the epoch captured by that attempt in addition
to the existing lease token. A restarted runtime therefore cannot finalize an
older attempt merely because it has the same Worker ID and deployment revision.

Before output is exposed, ordinary lease recovery may retry on a new epoch of
the same deployment/contract. Once the deadline is reached, recovery is
terminal and records `deadline-exceeded`. Queued expired Jobs are failed by
the normal maintenance loop. Streaming-specific “no retry after visible output”
semantics remain #94.

Legacy v1 Jobs and Workers omit the extension and retain their existing
behavior. New serving fields without `serving-v1`, unknown protocol
extensions, malformed epochs and stale epochs fail closed.

The generic WorkerRuntime can carry and locally recheck a
`WorkerServingAdvertisement`. Production provider adapters still have to
construct that advertisement from verified prepared artifacts/evidence. #99
does not turn an arbitrary label, mutable model tag or private path into trusted
deployment evidence, and does not add a user-facing model-serving endpoint.

## Acceptance scope

In addition to the pure contract tests, #99 adds reference-backend,
transport/WorkerRuntime and actual PostgreSQL tests for binding persistence,
profile-snapshot stability, deadline-capped leases, deadline terminal behavior,
stale-epoch rejection, concurrent claims and concurrent idempotent submission.

Passing these tests establishes durable serving admission semantics only. It is
not proof of llama.cpp/vLLM/Ollama model correctness, OpenCode compatibility,
embedding-space quality, System-One calibration, GPU performance or the v0.1
Real Smoke gate. Those remain in #94-#97 and #80 respectively.
