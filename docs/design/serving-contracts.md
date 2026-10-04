# Serving contracts: isolated first slice

Tracking: #91 / #93. Design baseline: #92 at
`2d34d72db1643c01a1782b00b8058fbedd942fc9`.

This is the first, pure-domain slice of #93. It is developed on
`feat/93-serving-contracts`, targeting `feature/91-capability-serving`, not
`main`. The integration branch initially contains only the #92 design snapshot.
The #79/#89/#90 release-repair track and operator-only #80 gate remain separate.
Do not merge feature code into the release candidate implicitly.

## Boundary

`astrumweaver.serving` contains immutable identity/contract/profile values and
side-effect-free profile validation. It introduces no new Worker/GPU owner,
service, API route, database migration, registration field or runtime behavior.
Existing package entrypoints and core contracts do not import this module.

A successful resolution is a **configuration snapshot**, not admission, a
reservation, hardware compatibility evidence or runtime readiness. #93 remains
open until the later persistence, protocol negotiation, atomic claim/epoch
binding, deadline and idempotency work is implemented and accepted.

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
