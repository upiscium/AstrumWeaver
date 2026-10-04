# Serving contracts and durable admission

Tracking: #91 / #93. Design baseline: #92 at
`2d34d72db1643c01a1782b00b8058fbedd942fc9`.

The pure-domain contract layer was the first slice of #93. The feature track now
also integrates those contracts with durable Worker registration, Job admission,
PostgreSQL persistence, bounded deadlines, extension negotiation and claim-time
runtime-instance fencing. The work remains on the #91 feature integration stack,
not on `main`; release-repair and operator-only acceptance tracks remain
independent.

## Boundary

`astrumweaver.serving` remains the pure identity/contract/profile layer. It
does not own GPUs, start runtimes or interpret workload payloads. Control and
Worker integration consume those values through the existing Worker/Job
authorities rather than introducing a second Slot or resource owner.

Profile resolution is still a **configuration snapshot**, not runtime readiness
or a reservation. Admission persists the resolved `ServingJobBinding` and
deadline only after at least one fresh ONLINE compatible Worker exists. That
preflight does not reserve a replica: authoritative compatibility, capacity and
runtime-instance identity are rechecked when a Worker atomically claims the Job.

A serving Worker advertises one immutable deployment revision and one per-start
`RuntimeInstance` epoch. Every serving claim, heartbeat/lifecycle mutation and
terminal write is fenced by the current epoch. Re-registering the same idle
Worker identity with the same deployment but a new epoch invalidates the old
process without changing the admitted deployment contract.

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

Schema/unit coverage includes `tests/test_serving_contracts.py` and
`tests/test_serving_control.py`. Durable integration coverage in
`tests/test_postgres_serving.py` exercises actual PostgreSQL transactions,
including competing claims, concurrent idempotency, cancellation/deadline races,
stale runtime epochs and database identity constraints.

Repository CI is the integration authority for this feature branch. These tests
are deliberately **not** real-runtime/model, gateway-client, OpenCode, embedding
quality or decision-quality evidence. Those acceptance levels belong to the
corresponding adapter issues (#94–#96); passing #93 must not be reported as proof
that a model/runtime pair supports one of those client surfaces.
