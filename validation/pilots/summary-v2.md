# AstrumWeaver bounded consumer pilot roll-up (v2)

Tracking: #91 / #97. Updated 2026-10-09 (JST).

This report extends the historical, immutable
[roll-up v1](summary-v1.md) with a **genuinely observed**
coding `direct_remote` lane. It does not rewrite the original
[delegated-local coding evidence](../coding-pilot/coding-v1.md).
Raw prompts, tool events, credentials, paths and other private trial data
remain outside the repository.

## Adapter disposition

| Adapter | Evidence | Observed outcome | Disposition |
| --- | --- | --- | --- |
| Coding | [historical local](../coding-pilot/coding-v1.md) + [new direct remote](../coding-pilot/direct-remote-v1.md) | delegated-local 2/3 accepted; direct-remote 2/3 accepted (one recorded harness retry) | **experimental** |
| Embedding | [embedding v1](../embedding/embedding-v1.md) | disposable retrieval 4/4 top-1 correct, incompatible space rejected | **accepted for pinned disposable scope** |
| Decision | [initial shadow](../decision/shadow-v1.md), [later d1-3B](../decision/d1-3b-shadow-v1.md) | initial 2/3 correct; d1-3B 2/5 correct on a **different** fixture, unknown abstention 0/1 | **experimental / OBSERVED_ONLY** |

Decision quality numbers use different labelled fixtures and are **not**
an apples-to-apples model-quality ranking. No Decision model has execution
or acceptance authority.

## Coding: same frozen fixture and checkers

The first local lane and the newly observed remote lane both use the
six-source-file baseline
`sha256:8d4c7698bb470532db53923745b63565a590a147268274c5dd0782e236f6d58f`
and the original three frozen independent acceptance checkers.

| Metric | delegated_local (historical) | direct_remote (new) |
| --- | ---: | ---: |
| Observed task classes | 3 | 3 |
| Final accepted / rejected | 2 / 1 | 2 / 1 |
| Accepted rate | 0.667 | 0.667 |
| Task wall-time median | 432355.98 ms | 20322.00 ms (including a failed setup attempt for task-03) |
| Tool/path budget | original pinned checkers | original pinned checkers; completed tools under task budgets |
| Remote planning/review/repair attribution | not measured in this paired comparison | UNKNOWN |
| Remote input/output/processing token telemetry | local-side remote usage UNKNOWN | 48643 **OpenCode-reported processed tokens** across 4 invocations |
| Subscription use/quota, electrical and operations cost | UNKNOWN | UNKNOWN |

The newly observed remote provider is `openai/gpt-6.1-sol` with
`high` effort through OpenCode `1.18.30`. It was used directly;
no fake endpoint or local-model proxy stood in for the remote provider.
A too-restrictive pilot edit rule caused task-03's first attempt to fail.
That attempt's time and token usage are **included**, not discarded; a
new isolated baseline and corrected test-only edit boundary yielded
an accepted second attempt. The task-01 checker rejected an off-by-one
definition-line answer, despite completed tool calls.

Both lanes used the same test fixtures and correctness authorities, but
the **private local-vs-remote instruction wording is not confirmed identical**,
models and resource conditions differ, and these are only three tasks.
The median times therefore are observations, **not** evidence of causal
throughput improvement, model-quality parity, or general remote demand.

The 48643 figure is `step_finish.tokens.total` from OpenCode JSON events,
**not** remotely audited billing, plan-quota units, credits, net cost, or
a transferable percentage of ChatGPT usage. TTFT, queue and generation
metrics that do not share the local durable-Job semantics remain UNKNOWN.

## Gates outside this technical scope

The earlier `direct_remote = NOT_RUN` gap is now addressed as a
**genuine bounded observation**, but coding remains experimental
because quality was only 2/3 accepted in each lane. Before any Pro→Plus
subscription change, separately observe representative real workloads and
peak windows, provider-specific plan accounting, cost/headroom and
rework/repair; do not equate these fixture tokens with a billing plan.

No production Worker/GPU, private code, embedding index, billing settings,
or existing OpenCode session was changed in this validation.
