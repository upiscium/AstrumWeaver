# Durable dual-runtime queue pressure and cancellation evidence (isolated)

Tracking: #122, 2026-10-10 JST. This report contains aggregate validation
results only. Raw model outputs, GPU identity, host addresses, credentials,
database settings, logs and workload text remain private.

## Fixed execution scope

- The separately tested composite runtime and associated Control/Worker
  packages were pinned at implementation commit
  `82b49459d2452b899d6d0517c54ae44b7b510e2c` within Draft PR #123.
- Real PostgreSQL 16, one local Control process, one Worker owning one SM61
  11-GiB GPU, two distinct GPU-resident llama.cpp child processes, and both
  immutable Embedding/Decision serving profiles.
- Models and their SHA-256 identities are the two pinned artifacts already
  recorded in `native-v1.md` and `durable-e2e-v1.md`. GPU runtime was
  source-pinned CUDA 12.9 llama.cpp. Worker `max_concurrency=1`.
- Fresh, isolated configuration, token pair and PostgreSQL cluster were
  created for this run. No existing Control database, client index or
  production Worker was changed.

## Results

| Gate | Measured outcome |
| --- | --- |
| Fresh PostgreSQL 16 and migration history | 6 migrations PASS |
| Control and dual Worker ready with two versioned profiles | PASS |
| Baseline alternating Embedding and Decision jobs | 3/3 and 3/3 succeeded |
| Durably stored original six Jobs | 6/6 succeeded |
| Eight simultaneous bounded Embedding HTTP requests | 6 HTTP 200; 2 HTTP 429 |
| Concurrent running Jobs sampled in PostgreSQL | maximum 1 (24 samples) |
| Client API with Worker-only bearer token | 401, rejected |
| Valid, future-eligible serving-bound Job cancellation | HTTP 200, CANCELLED |
| Cancelled Job attempts | 0 (never claimed) |
| Final PostgreSQL rows | 9 Embedding succeeded; 3 Decision succeeded; 1 Embedding cancelled |
| Both owned Control/Worker processes stopped | PASS |
| All model/Control/Worker test sockets released | PASS |
| GPU idle VRAM restored to baseline | PASS |

Two HTTP 429 responses are **correct bounded overload behavior**, not
successes; the test did not retry or reclassify them as 200. The PostgreSQL
concurrency bound was **sampled**, not a continuous scheduler proof;
existing deterministic claim/capacity regressions remain complementary.

Model quality and calibration were not re-evaluated by these synthetic
requests. System-One is still OBSERVED_ONLY / uncalibrated /
recommendation-only. This does not authorize decision-based permissions,
unattended changes, or final acceptance.

## Evidence integrity and remaining gates

Private mode-0600 JSON result SHA-256:
`7418d8decfc5d1a648b6033021aecb2c04fc2d7549f4fbc6fe9c717f507197af`.

The isolated PostgreSQL server was stopped after collection, both private
inference children and Worker/Control were stopped, and GPU memory returned
to its idle baseline. No production change was made.

**Disposition:** bounded queue pressure, authorization-domain separation,
future-eligible cancellation and durable State retention PASS for this
isolated candidate. This small burst is **not** repeated sustained traffic,
peak-window sizing, arbitrary simultaneous GPU execution, or production
operations acceptance. Independent review and reversible deployment
approval remain separate release gates.
