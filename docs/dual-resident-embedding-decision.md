# Joint resident Embedding + System-One Runtime (experimental)

Issue: #122 under #91. This is a separate, opt-in, post-v0.1 feature.
The historical single-model runtimes and embedding/decision serving profiles
are unchanged. Keep the stable `astrumweaver` runtime identifiers while the
independent TSUMGI migration #120 remains pending.

## Provider boundary

`llama-cpp-dual` hosts **two separately owned llama.cpp servers** behind
**one** GPU-owning Worker, **one** approved composite deployment and **one**
runtime instance epoch. The Worker has exactly one GPU and must set
`max_concurrency=1`. Neither a second Worker/GPU reservation nor a Control
bypass is introduced. Job execution dispatches by exact capability:

- `text.embed` → Qwen3-Embedding-0.6B Q8_0, `--embeddings --pooling last`
- `decision.system_one` → LiquidAI d1-3B Q4_K_M, native decision head

Each child has a distinct, credential-free **loopback port**, pinned prepared
GGUF artifact SHA-256, an absolute Nix-store `astrumweaver-llama-server`
wrapper, full GPU offload (`fit=off`), offline mode and no web UI. This wrapper
prevents inherited Worker/Client bearer and cloud tokens reaching llama-server.
Operator-approved composite model identity is:

`dual-sha256:<digest>` where digest = SHA-256 over JSON with sorted keys,
compact separators and exactly:
`{"schema":"dual-llama-cpp-models-v1","embedding_sha256":"...","decision_sha256":"..."}`.

RuntimeDeploymentSpec provider_config holds `embedding_model`,
`decision_model`, `embedding_provider`, `decision_provider` and
`required_total_vram_mb`, plus pinned `embedding_space_id` and `decision_semantics_id`. These child semantic digests must
equal the reviewed ServingContract.semantic_revision fields; a mismatch
prevents the Worker from advertising either capability.

Model entries have `model_ref` (absolute prepared
GGUF), `sha256` (lowercase 64-hex digest) and `estimated_size_mb`. Provider
entries are standard LlamaCppProviderConfig JSON using pinned Nix wrapper paths;
embedding must enable `embeddings=true,pooling=last`, decision must enable
`decision=true`. Endpoint ports must differ. The parent demand's
`model.model_ref` must equal the composite `dual-sha256:<digest>`,
`model.model_format=gguf`, `residency_policy=vram_only`,
`gpu_topology=single_gpu` and `min_single_gpu_vram_mb` must reserve
the combined model+KV+scratch+CUDA footprint.

The single ServingDeploymentDeclaration still binds every advertised contract
to **one composite deployment revision**; its DeploymentIdentity
`model_artifact_sha256` must be `sha256:<bundle-digest>`. Distinct
`ServingContract.semantic_revision` values continue to bind each capability:
the immutable Embedding-space ID and the uncalibrated Decision-semantics ID.
Existing logical profiles cannot be silently retargeted. Two exact child
aliases are used on the distinct ports.

## Lifecycle and fail-closed rules

Both exact GGUF SHA-256s are checked before any model process starts and
rechecked after both children are ready. Opened file descriptors are hashed
without following symlinks; accepted launchers must be pinned top-level Nix
store outputs. The model storage remains a trusted operator resource: a
host-privileged adversary can still rewrite it and is outside this Worker
threat model. Both children must pass owned health and model alias checks
before Worker admission. A dual Worker must have a reviewed serving manifest
with exactly two contracts (one embedding operation and one decision operation)
and the matching pinned semantic IDs; extra contracts fail closed.
If one child fails during startup, stop both. If either child dies/degrades
after startup, the whole Worker runtime becomes unavailable and the existing
supervisor fences claims/readiness. Stop/release attempt **both** children,
even if the other fails. The existing generic Control claim/lease/deadline,
job-level serving binding and runtime instance epoch fencing remain mandatory.
No model hot-swap, background eviction, concurrent jobs or extra GPU owners.

This is currently a **candidate implementation**; do not reuse a production
embedding index, reassign GPUs or perform an operator-irreversible cutover.

## Stage 1: real GPU feasibility

An isolated compute capability 6.1 GPU (11 GiB) ran both exact pinned GGUFs
with source-pinned CUDA 12.9 llama.cpp simultaneously resident:
Qwen3-Embedding-0.6B Q8_0
`sha256:06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`
and LiquidAI d1-3B Q4_K_M
`sha256:16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402`.
Both servers used distinct loopback sockets and full GPU layers. Alternating
two embedding and two native decision calls all passed. Native median HTTP:
embedding 19.437 ms, decision 32.006 ms. GPU VRAM increments: 1,519 MiB
embedding + 1,894 MiB decision = **3,413 MiB** jointly. Cleanup released
both process groups, sockets and GPU allocation to baseline.
Private evidence SHA-256:
`5a7d116545b32b95856c1158184b6a5520123c53731b80500bc6cc7f38658661`.

**Native PASS does not establish distributed Worker/Control acceptance.**
The decision head remains OBSERVED_ONLY, uncalibrated, recommendation-only.

## Outstanding acceptance gates

1. Integration probe for this exact composite provider on a disposable GPU
   Worker; negative foreign listeners, partial start, child crash, cancellation,
   concurrent Worker misconfiguration, runtime health and cleanup.
2. Isolated PostgreSQL Control → Worker durable E2E with two freshly approved
   serving contracts/profiles and alternating single-job execution, including
   incorrect semantic/embedding-space/stale-epoch rejection.
3. Independent correctness/security review plus a reviewed, reversible
   operator deployment migration. Do not claim production or v0.1 Real Smoke
   gate #80 completion from these tests.
