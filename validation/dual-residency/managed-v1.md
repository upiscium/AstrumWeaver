# Composite ManagedRuntime GPU acceptance — opt-in prototype

Tracking: #122. Observed 2026-10-10 JST.

This is a **redacted** exact-candidate validation. No internal hostnames,
addresses, Worker IDs, GPU UUIDs, credentials, source paths, server logs, user
prompts or generated decision text are published.

## Candidate and model identities

- Source HEAD tested: `183daf12e0c7b448479c0acb375fa12d62008619`.
- Executable: exact Nix-built feature-branch Worker package at that HEAD;
  package output pinned with the source revision above.
- Runtime: source-pinned CUDA 12.9 / SM61 llama.cpp launcher package from
  immutable serving input `c65f129ad433fdb1304e5096fe266176b08f1c16`.
- GPU: one NVIDIA SM61 accelerator, 11 GiB, initially 2 MiB in use.
- Embedding Qwen3-Embedding-0.6B Q8_0 artifact SHA-256:
  `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.
- Decision LiquidAI d1-3B Q4_K_M artifact SHA-256:
  `16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402`.
- Joint canonical model bundle SHA-256:
  `5ac27347cef0f2e6c01e6101855e7d62c9c9e74259e73e5b3290193586800730`.
- Explicit minimum reserved GPU memory: **5,120 MiB**; concurrency policy:
  one active Job at a time.

## Observed composite runtime results

Unlike earlier standalone-native smoke, this test constructs an actual
`RuntimeDeploymentSpec` → serialized/reloaded manifest →
`managed_runtime_from_deployment` → `RuntimeLifecycleManager` →
`DualLlamaCppExecutor`, starts **both** children, and checks joint health,
model aliases, capability-based dispatch, child semantics and unified shutdown.

| Gate | Result |
| --- | --- |
| Both pinned GGUF digests before startup | PASS |
| Composite config/compatibility and manifest round trip | PASS |
| Joint ManagedRuntime startup | PASS (9,009.604 ms) |
| Two resident child models on one GPU | PASS |
| Combined VRAM increase | **3,413 MiB** |
| Embedding executor requests | **2/2 PASS** (17.257 / 12.046 ms) |
| Decision executor requests | **2/2 PASS** (52.317 / 42.384 ms) |
| Embedding-space mismatch rejection | PASS (no normal answer) |
| Decision-semantics mismatch rejection | PASS (no normal answer) |
| Residency enumeration | 2 distinct resident models |
| Joint health after alternating operations | PASS |
| Joint bounded shutdown and child process exit | PASS |
| Both temporary loopback listeners released | PASS |
| GPU memory restored to initial 2 MiB | PASS |
| Any installed Worker service started/reconfigured | NO |

The four operations were **sequential**, not concurrent. These are native
Worker-side executor timings, **not Control/DB/network end-to-end** timings.
Embedding returned finite 1024-dimensional unit-normalized vectors. System-One
returned finite normalized scores and retains **OBSERVED_ONLY / uncalibrated /
recommendation-only** policy. No labelled model quality acceptance is claimed.

An initial local test iteration mistakenly treated legitimate TCP `TIME_WAIT`
state as a still-listening port and reported an inconclusive/FAIL cleanup.
A new evidence file was generated after correcting the **test probe's** port
release check to test the absence of a listening acceptor rather than requiring
a raw socket bind. The first observation was preserved privately. The repeat
succeeded, with no GPU processes/listeners remaining.

## Provenance and outstanding gates

Private evidence file mode 0600, SHA-256:
`2ed0bb8a7bcd88800f880e063404459de9ac8e8f03d25cb2fc41804663368ce0`.

Candidate CI #1076: `unit` SUCCESS and `nix` SUCCESS; full local regression
**940 passed, 48 skipped**, including 11 focused dual-runtime tests.

**Disposition: GPU-local composite runtime PASS; durable Control→PostgreSQL→
Worker E2E NOT_RUN.** Two approved profiles and contracts need a disposable
distributed validation plus negative stale-epoch, partial-child failure,
cancellation, and queue-pressure tests before production cutover.
Existing hardware and production Control/Worker registrations are unchanged.
