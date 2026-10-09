# Native dual-residency smoke evidence

Tracking: #122, 2026-10-10 JST. Private-safe aggregate; no hostname,
network address, GPU UUID, raw model output or source text is published.

- Target: one NVIDIA SM61 GPU, 11 GiB; source-pinned CUDA 12.9 llama.cpp.
- Embedding artifact: Qwen3-Embedding-0.6B Q8_0,
  `sha256:06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.
- Decision artifact: LiquidAI d1-3B Q4_K_M,
  `sha256:16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402`.
- Both servers jointly resident with two owned loopback sockets, GPU layers all,
  `fit=off`, context 4096, concurrency one.
- Alternating operations: text.embed 2/2 PASS (1024 finite L2-normalized values),
  decision.system_one 2/2 PASS (finite normalized choice probabilities,
  zero generated output tokens).
- Native median HTTP: Embedding **19.437 ms**, Decision **32.006 ms**.
- Incremental GPU memory: Embedding 1,519 MiB, additional Decision 1,894 MiB,
  jointly **3,413 MiB** from the initial 2 MiB baseline.
- All test-created process groups terminated, both sockets released and GPU
  memory returned to baseline; no Worker or production configuration change.
- Private mode-0600 source JSON SHA-256:
  `5a7d116545b32b95856c1158184b6a5520123c53731b80500bc6cc7f38658661`.

**Disposition: native co-residency PASS; composite Worker and Control E2E
NOT_RUN.** Decision quality OBSERVED_ONLY. No production installation claimed.
