# Coding pilot — observed direct-remote lane (v1)

Tracking: #97 / #91. Observed 2026-10-09 (JST).

This evidence is **metadata-only**. Raw OpenCode JSON events, prompts, tool inputs
and outputs, credentials, system identity and local directories remain private.
Historical delegated-local evidence is preserved unchanged in
`validation/coding-pilot/coding-v1.md`; this report is its complementary
`direct_remote` lane, not a rewrite of that prior measurement.

## Fixed comparison scope

- Client: OpenCode `1.18.30`.
- Genuine remote provider: `openai/gpt-6.1-sol`, `high` variant; direct
  model execution through OpenCode. No simulated remote endpoint or local
  inference is included in this lane.
- Model/account billing identifiers, OAuth data, session IDs and raw tool
  content are omitted. Remote model artifact, tokenizer and template digests
  were **not exposed** by the hosted provider.
- Synthetic fixture source checkout: `5f7f72a5fa54f77e828bcb33ad9a609f510c26dc`.
- Frozen six-file baseline: `sha256:8d4c7698bb470532db53923745b63565a590a147268274c5dd0782e236f6d58f`.
- Frozen independent checkers:
  - task-01 discovery: `sha256:a43e9e604f727214208bb28c6f226a607dc3b4aa66e2d7b243a94698abe0057f`;
  - task-02 test addition: `sha256:36636778d13edf7de70d0836f4e110e43a4f35062e11e43cc8f776b349a0c597`;
  - task-03 scoped fix: `sha256:be24825892c0415738f3b933e9be020701de85637ab5d81af709dc1fcd237744`.
- Fresh isolated baseline per independent attempt; concurrency 1; no shell or
  external-directory tools available to the remote coding agent. Discovery
  had no edit authority. Write tasks were limited to synthetic isolated
  workspaces, and source-path constraints were checked by independent
  file-diff and acceptance checks. Max tool calls: 12 / 16 / 16.
- Instructions were functionally equivalent to the three earlier task
  classes, but the **original private delegated-local prompt wording is not
  verifiably identical**. OpenCode bootstrapping/client overhead may also
  differ. Do not interpret these wall-time medians as a controlled model-speed
  or causal serving comparison.

## Observed direct-remote results

| Task | Final status | Attempts | Client tool calls | Modified source files | Checker | Total wall (ms) | OpenCode-reported tokens |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| task-01 — discovery | rejected | 1 | 3 / 12 | 0 | FAIL | 16008 | 8814 |
| task-02 — test addition | accepted | 1 | 4 / 16 | 1 (allowed) | PASS | 20322 | 14833 |
| task-03 — scoped fix | accepted after setup retry | 2 | 8 across 2 attempts / 16 per attempt | 1 (allowed, second attempt) | PASS (second attempt) | 43928 | 24996 |

The discovery task reported the wrong definition line (one line earlier
than the frozen checker requires). The client tools completed successfully,
but **correctness failed**; no file was changed.

Task-03's **first attempt failed at the pilot permission gate**: the tool
patch was denied by a too-narrow synthetic-workspace edit rule. The initial
21,869 ms and 10,040 reported tokens remain charged to the observation.
It is a **client/experimental-harness failure**, not evidence that the model
produced an incorrect code fix. A second, fresh baseline allowed edits within
the isolated synthetic workspace, while disabling shell/external-directory
tools and enforcing the exact changed-path/checker budget afterwards; this
attempt passed (22,059 ms, 14,956 reported tokens). Neither attempt changed
unrelated source paths. Checkers may create local Python cache files during
post-run validation; these are not model-generated source modifications.

## Aggregates and historical local comparison

| Metric | delegated_local (historical) | direct_remote (new observation) |
| --- | ---: | ---: |
| Observed task classes | 3 | 3 |
| Final accepted / rejected | 2 / 1 | 2 / 1 |
| Accepted rate | 0.667 | 0.667 |
| Median task wall, ms | 432355.98 | 20322.00 (retry-inclusive) |
| Remote/CLI retry caused by harness failure | not separately established | 1 failed setup attempt retained |
| Queue wait / TTFT / pure generation time | earlier local measured | UNKNOWN for equivalent remote semantics |
| Explicit planning/review/repair attribution | historical local record | UNKNOWN |
| Remote token telemetry | UNKNOWN | 48643 OpenCode-reported total tokens across 4 invocations |
| Subscription quota units / electrical or operating cost | UNKNOWN | UNKNOWN |

The remote token figure is the **sum of `step_finish.tokens.total`
reported by OpenCode**, including context/cache accounting and the failed
setup attempt. It is **not verified billing, credits, subscription quotas,
effective cost, or an OpenAI plan limit**. The local lane has no equivalent
independently observed remote-usage figure, so a usage-reduction percentage
cannot be computed.

The three remote final task observations took 80,258 ms in aggregate,
including the permission-gate retry. This is a small three-task fixture,
not repeated production workload or a peak-window consumption study.
Different model capabilities, prompt wording and remote/local execution
conditions prevent a causal throughput comparison.

## Private trace integrity and validation limits

Only SHA-256 commitments to private OpenCode event traces are published:

| Invocation | Private trace SHA-256 |
| --- | --- |
| task-01 | `sha256:1ae5e0ebf736ee6194e2fa7450fa5f3a6c383ec3275abe9e034e8ba87d1b9552` |
| task-02 | `sha256:3ecba948a12cd873d178d8f793a670afd8e382511a3208a3386e5a549da32324` |
| task-03 attempt 1 (permission failure) | `sha256:a6216064e1d9a8f8c079a04c1b789376795dc949dc5556b612d5c5f363b2941e` |
| task-03 attempt 2 (accepted) | `sha256:818e5fe5cd73982930c0d00f944b69a338239f0a005291f4eee2744f9740862f` |

The private underlying traces and exact test instructions are not committed.
Accept/reject is determined by the existing frozen fixture checkers plus the
allowed changed-source-path and completed-tool-call budget, not by client prose.

## Disposition and next gate

Coding remains **experimental**: both lanes now have genuine observations,
but each accepted only 2/3 tasks. Embedding and decision retain their
separate prior dispositions. This result **resolves the NOT_RUN direct-remote
observation gap only**; it does not validate model-quality parity at scale,
provider quota accounting, total cost of ownership, peak-period headroom,
or subscription downgrading. No production Worker/GPU, private project,
plan/billing or active OpenCode session was changed by this pilot.
