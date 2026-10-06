# AstrumWeaver Coding Pilot Evidence

This evidence is intentionally metadata-only. It contains no repository name, source text, prompt, credential, worktree path, hostname, Worker identity, GPU identity, or raw tool input/output.

## Support scope

| Field | Result |
| --- | --- |
| Evidence version | coding-pilot-v1 |
| AstrumWeaver revision | 35fb7295bb8802c63df5b66da006563ce2fa255a |
| OpenCode version | 1.18.30 |
| Provider adapter | llama-cpp-chat-v1 |
| Runtime revision | 6753a033f058fbf778d282556ed9b16c78de7c71 |
| Model artifact | sha256:3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597 |
| Quantization | Q4_K_M |
| Tokenizer artifact | sha256:a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3 |
| Template artifact | sha256:51f190aff3530ef8bdb2aad9fc1cf4afb49a6d1506a458af41e6d247b587bc34 |
| Deployment revision | sha256:43ced97ef6aaa2b0408a9990b13344c936d33705941af5a67697ef23f9ab3fb3 |
| Logical profile revision | sha256:f2b91848fc3f604ec5c353c1adcfab8b6d55a28d9166f14a49317e3729d583d9 |
| Serving contract revision | sha256:d476ebd2c0af5ff9ff988ae64080bbcabce6c139df0450b54defbf5665f2cd5f |
| Context tokens | 12288 |
| Output tokens | 1024 |
| Request timeout seconds | 600 |
| Concurrency | 1 |
| Disposition | experimental |
| Private values omitted | true |

## Bounded tasks

| Task | Kind | Baseline | Acceptance | Write | Max paths | Max tool calls |
| --- | --- | --- | --- | --- | ---: | ---: |
| task-01 | discovery | sha256:8d4c7698bb470532db53923745b63565a590a147268274c5dd0782e236f6d58f | sha256:a43e9e604f727214208bb28c6f226a607dc3b4aa66e2d7b243a94698abe0057f | no | 1 | 12 |
| task-02 | test_addition | sha256:8d4c7698bb470532db53923745b63565a590a147268274c5dd0782e236f6d58f | sha256:36636778d13edf7de70d0836f4e110e43a4f35062e11e43cc8f776b349a0c597 | yes | 1 | 16 |
| task-03 | scoped_fix | sha256:8d4c7698bb470532db53923745b63565a590a147268274c5dd0782e236f6d58f | sha256:be24825892c0415738f3b933e9be020701de85637ab5d81af709dc1fcd237744 | yes | 1 | 16 |

## Lane aggregates

| Lane | Observed | NOT_RUN | Accepted | Rejected | Failed | Escalated | Accepted rate | Median wall ms | Median queue ms | Median TTFT ms | Median generation ms | Remote rework events | Rework unknown runs | Escalation events | Observed remote usage | Usage unknown runs | Gaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| direct_remote | 0 | 3 | 0 | 0 | 0 | 0 | NOT_RUN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | 0 | 0 | UNKNOWN | 0 | electrical_cost,generation_time,queue_wait,remote_rework,remote_usage,ttft |
| delegated_local | 3 | 0 | 2 | 1 | 0 | 0 | 0.667 | 432355.98 | 288.91 | 3015.54 | 424721.63 | 0 | 0 | 0 | UNKNOWN | 3 | electrical_cost,remote_usage |

Accepted task rate counts only observed runs. NOT_RUN is never treated as a failure or a success. UNKNOWN metrics remain unknown rather than being inferred from local token counts.
