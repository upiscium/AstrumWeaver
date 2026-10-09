# Hardened composite Worker — exact-head real GPU durable E2E

Tracking: #122, 2026-10-10 JST. This is a privacy-safe aggregate from an
independent disposable PostgreSQL and GPU test environment, not a production
cutover or a public raw trace.

## Bound executable and model identity

- **Exact candidate code revision:** `a7b223d5d812323732adfcac0e6f3ef3ea669b6c`
  on Draft PR #123. Nix `worker` and `control` packages were rebuilt at
  this exact commit, after applying fail-closed dual-serving hardening.
- The GPU runtime is source-pinned CUDA 12.9 / NVIDIA SM61 llama.cpp.
  The two resident model artifacts are the frozen
  Qwen3-Embedding-0.6B Q8_0 and LiquidAI d1-3B Q4_K_M GGUF digests from
  `native-v1.md`. The complete composite model-bundle identity is
  `sha256:5ac27347cef0f2e6c01e6101855e7d62c9c9e74259e73e5b3290193586800730`.
- Single 11-GiB GPU, one Worker, two owned local model processes,
  **max_concurrency=1**, two separately versioned immutable
  Embedding-space and System-One semantics contracts.
- Fresh, separately generated PostgreSQL 16 database, Client and Worker
  bearer tokens, configuration, Control/Worker/model ports and private
  files. No shared production database, index, Control, Worker or GPU
  allocation was modified.

## Measured outcome

| Acceptance check | Result |
| --- | --- |
| PostgreSQL setup and 6 migrations | PASS |
| Fresh Control/Worker serving readiness and two profiles | PASS |
| Original alternating Embedding jobs | **3/3 succeeded** |
| Original alternating System-One jobs | **3/3 succeeded** |
| Versioned durable terminal rows for the original six Jobs | **6/6 succeeded** |
| Concurrent 8 bounded Embedding HTTP submissions | **7 HTTP 200, 1 HTTP 429** |
| Maximum running Job count sampled from PostgreSQL | **1 (26 samples)** |
| Future-eligible, serving-bound cancellation | **HTTP 200, durable CANCELLED, attempts=0** |
| Worker token attempting Client API submission | **401, rejected** |
| Missing Client bearer | 401, rejected |
| Wrong Embedding-space identity | 409, rejected |
| Wrong Decision profile revision | 409, rejected |
| Stale runtime-instance epoch claim | 409, rejected |
| Final PostgreSQL persisted status count | Embedding 10 succeeded / 1 cancelled; Decision 3 succeeded |
| Both owned runtime children, Worker and Control stopped | PASS |
| All test ports and PostgreSQL stopped | PASS |
| GPU memory restored to the idle baseline | PASS |

Original E2E median client-side HTTP times (3 tasks per capability):
Embedding **82.074 ms**, System-One **112.593 ms**. The two models
occupied a combined incremental **3,413 MiB** on one GPU. This is a
bounded small synthetic workload, not a sustained peak or real end-user
quality evaluation.

The single HTTP 429 is a **correctly rejected overload**, not a successful
inference. The sampled maximum-one-running-Job observation is
complementary to deterministic single-job scheduling tests, not a
continuous timing proof.

## Integrity, restrictions and outstanding gates

Private mode-0600 E2E JSON evidence SHA-256:
`86fd2c1b4304a1f55a3e1a027bcc3cba99d8bdd1876e556b4d11ca53b191ef98`.
Raw GPU UUIDs, host identity, internal ports, authentication material,
prompts, unredacted tool payloads and model outputs are not published.

The earlier child-death/serving-fencing test is in
(`durable-e2e-v1.md`). A **separate exact-head revalidation** of the same
single-child failure condition, with a fresh disposable PostgreSQL database,
is reported in `hardened-child-failure-v1.md`. The exact hardening
worktree passed **944 tests / 48 skips** before publication.

**Disposition: exact-head functional GPU+durable Control/Worker E2E PASS.**
Independent human correctness/security review, production deployment
design/rollback acceptance, long-running cancellation and repeated
peak-window capacity/quality remain outside this result. System-One
remains OBSERVED_ONLY, uncalibrated and recommendation-only. No merge to
`main`, v0.1 release gate #80, billing change or production cutover
is implied.
