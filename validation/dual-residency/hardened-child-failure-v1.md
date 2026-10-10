# Hardened composite Worker — exact-head native child-death fencing

Tracking: #122, 2026-10-10 JST. Metadata-only controlled fault injection.

## Exact candidate and isolation boundary

- Tested code revision: `a7b223d5d812323732adfcac0e6f3ef3ea669b6c`
  (Draft PR #123, source-pinned Worker and Control Nix packages).
- Fresh, disposable PostgreSQL 16 and independent Control, one dual-resident
  GPU Worker, two distinct model subprocesses and versioned Embedding and
  Decision contracts on a single real NVIDIA SM61 GPU.
- Both prepared model artifacts match the pinned Qwen3-Embedding-0.6B Q8_0
  and LiquidAI d1-3B Q4_K_M digests in `native-v1.md`.
- One explicit Worker/GPU owner, `max_concurrency=1`; no access to any
  deployed production Worker, Control database, embedding index or client.

## Failure-injection result

Both children first reached a READY and jointly resident state. The test
then identified and killed **only the native System-One child process that
belonged to the isolated Worker's process group**. The Embedding model was
not deliberately stopped before checking supervision.

| Expected fail-closed behavior | Observed |
| --- | --- |
| Both children confirmed ready before fault | PASS |
| Fault applied only to owned System-One child | PASS |
| Worker `GET /ready` | HTTP **503**, `ready=false` |
| Worker runtime status | `failed`, availability **false** |
| Runtime failure latched | PASS |
| Durable Control Worker status | `draining` |
| New correctly **serving-bound** Embedding Gateway request | HTTP **503**, rejected |
| Rejected request changed PostgreSQL job count | NO |
| Worker, Control and owned children stopped | PASS |
| All temporary sockets released | PASS |
| GPU memory recovered to initial idle baseline | PASS |

The semantic distinction matters: an unbound legacy generic Job may
remain queueable under separate scheduling rules; this test requires
the *versioned serving-bound* Embedding Gateway to refuse admission after
the composite's Decision child dies. No partial service is silently
advertised as ready.

## Private evidence and limits

Private mode-0600 SHA-256:
`5beed7e0c777cf3684507a2f6b69fae4def73567d070c797051f10e2f87b78fc`.

No internal hostnames, IP addresses, ports, authentication material,
GPU UUIDs, raw model text, source paths or database credentials are
published. The disposable PostgreSQL instance was stopped after the
test; no lingering model process, listener or GPU allocation was
observed.

**Disposition:** exact-head real child-death/quarantine/serving-fencing
PASS for one isolated failure injection. Repeated failure/restart storms,
long-running in-flight job cancellation, sustained peak load, independent
human correctness/security review and production rollback remain distinct
acceptance gates. System-One retains experimental OBSERVED_ONLY,
uncalibrated, recommendation-only authority.
