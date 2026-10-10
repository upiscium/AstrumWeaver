# Independent-review security remediation — authenticated native GPU E2E

Tracking: #122 / PR #123, tested 2026-10-10 JST. Public-safe
acceptance evidence only. Private tokens, model prompts, internal ports,
infrastructure names, GPU UUIDs, raw child logs and connection strings
are not published.

## Code and attack model

- Exact code revision tested: `44f71f43c03806665987d2f4bd9452cc7eb8be27`.
- Source-pinned Nix Worker and Control builds succeeded.
- One isolated NVIDIA SM61 11-GiB GPU, two GGUF models
  (Qwen3-Embedding-0.6B Q8_0 and LiquidAI d1-3B Q4_K_M), dedicated
  root-owned model files, two private native inference processes,
  `max_concurrency=1`, and new disposable PostgreSQL16/Control/Worker.
- The previous independent Security-Reviewer identified that local
  network-namespace clients could call unauthenticated llama.cpp
  loopback ports without passing Worker/Control job admission.
- The composite now creates independent ephemeral API keys in a
  Worker-owned mode-0700 directory as mode-0600 key files. A
  model-specific `--api-key-file` is passed without an argv secret;
  the corresponding Worker-side API uses its private Bearer header.
  Generic single-model runtimes remain unchanged.

## Observed negative and positive outcomes

| Gate | Measured outcome |
| --- | --- |
| Direct unauthenticated native Embedding request | **HTTP 401** |
| Direct unauthenticated native System-One request | **HTTP 401** |
| Authorized Control → Worker → Embedding durable Jobs | **3/3 succeeded** |
| Authorized Control → Worker → Decision durable Jobs | **3/3 succeeded** |
| Eight parallel Embedding Gateway requests | **6 HTTP 200; 2 HTTP 429** |
| Max sampled concurrent Worker job count | **1** |
| Future-eligible serving-bound Job cancellation | **CANCELLED, attempts=0** |
| Worker token attempting Client submission | **HTTP 401** |
| Wrong embedding-space / decision profile / stale epoch | **HTTP 409** |
| Both resident models combined GPU memory increase | **3,413 MiB** |
| Test-owned Worker, Control, PostgreSQL and model listeners stopped | **PASS** |
| Private ephemeral API-key directory after release | **absent** |
| GPU recovered to initial 2-MiB idle baseline | **PASS** |

Original six fabric HTTP median latencies are synthetic and short-run only.
No sustained capacity, quality improvement, calibrated decision authority or
long-running in-flight cancellation conclusion follows from these data.

Private mode-0600 JSON evidence commitment:
`sha256:c7891ee962437ac4f188a16b63974884f4cb7865e0876e29f55805dc5858e7d1`.

## Limits

The private-key file cannot be read by a **different unprivileged local UID**
under normal Unix owner/mode and trusted local filesystem semantics.
An attacker sharing the Worker's effective UID, or a privileged root attacker,
lies outside that boundary; production must use a dedicated Worker
service identity, appropriate service isolation and trustworthy model storage.

Code reviewer finding about startup exception masking and the previous
conditional model-path TOCTOU finding were addressed by
`72f04ac86c81ca87c999d9ea3cc79b2adcf88175`;
independent Security-Reviewer then identified the unauthenticated loopback
issue and this code revision addresses it. A **new exact-head independent
review after this patch** is still required before any merge. Decision
remains OBSERVED_ONLY / uncalibrated / recommendation-only. Production
rollout, v0.1 operator Real Smoke and a `main` merge were not performed.
