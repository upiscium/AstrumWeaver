# Bounded coding consumer pilot

Tracking: #91 / #97. Coding adapter prerequisite: accepted #94.

This procedure measures whether one pinned AstrumWeaver chat deployment is useful
to a coding client. It is deliberately narrower than a coding benchmark and does
not infer remote-provider savings from local token counts.

## Scope

The first pilot uses:

- one Worker and one active deployment;
- concurrency exactly 1;
- pinned OpenCode, provider adapter, runtime, model, quantization, tokenizer,
  template, deployment, profile and serving-contract revisions;
- a fixed context/output envelope and explicit request timeout;
- three task classes: read-only discovery, scoped test addition, scoped fix.

Client repository access, tool execution, worktrees, review and final acceptance
remain outside AstrumWeaver.

## Private/public boundary

Raw prompts, source, OpenCode JSON events, tool arguments/results, worktree paths,
hostnames, Worker IDs, credentials and remote-provider account data are private
pilot material and must not be committed.

Public evidence contains only opaque task IDs, fixture/baseline digests,
acceptance-checker digests, bounded budgets, exact serving identities and numeric
observations.

The recorder rejects unknown input fields. This is intentional: arbitrary notes
or prompt/source fields must not accidentally become public evidence.

## Synthetic fixture

The repository contains a public synthetic fixture under
`validation/coding-pilot/fixture`. It is not production source and contains no
private project material.

Canonical baseline digest:

`sha256:35cc690e909857f306f3ccf130ae3af244df1c742470a91a3f7cfd9f2fb289f4`

Acceptance checker identities:

| Task | Kind | Checker SHA-256 | Write | Max changed paths | Max completed tool calls |
| --- | --- | --- | --- | ---: | ---: |
| task-01 | discovery | `sha256:9dafceeecffecb2887668cad5f6ac93294e3244bfc67e403ef6824d5ed4ea0aa` | no | 1 | 12 |
| task-02 | test_addition | `sha256:1aa016f109b429a1fe80df3c736f2a2c2a41d55c05d45843b8b0f2f9f9cce9aa` | yes | 1 | 16 |
| task-03 | scoped_fix | `sha256:97e9c5e999794323c12db830c7e0800321874ed51cd01795bb88e0ed6117b82c` | yes | 1 | 16 |

For task-01, any modification is rejected even though the schema keeps a positive
path budget. For write tasks, the pilot runner must additionally enforce the
task-specific allowed path, not merely the aggregate path count.

The exact task instruction supplied to the client is private run input. The
public task contract and checker determine correctness.

## Run preparation

1. Verify Control and Worker readiness and the exact `GET /v1/models` identity.
2. Prepare a client dependency cache before timed runs. Dependency installation,
   model download and runtime startup are setup costs, not per-task inference.
3. Use a fresh isolated worktree/copy for every task run.
4. Pin the same baseline digest and acceptance checker for both comparison lanes.
5. Keep OpenCode project config, auth and persisted sessions isolated.
6. Do not use a broad permission bypass. Write tasks remain constrained by their
   explicit path/tool budget.
7. Start timing only after the prepared client environment and resident runtime
   are ready.

A warm resident runtime is the initial operating condition because #91 is
evaluating a serving layer, not cold model installation. Cold-start cost may be
reported separately but must not be silently mixed into task latency.

## Comparison lanes

`direct_remote` means the task is executed directly by the remote coding
provider/client path being compared.

`delegated_local` means the equivalent task is delegated through the pinned
AstrumWeaver local serving profile.

Both lanes use the same task baseline and acceptance checker. A different
baseline/checker is a different experiment.

If a genuine remote run or its usage accounting is unavailable, record an
explicit `not_run` record and leave unavailable metrics UNKNOWN. Do not
substitute a simulated remote model or infer remote usage from local tokens.

## Correctness vs tool completion

Client tool completion and task correctness are separate fields.

A tool call completing only proves that the client executed a tool round trip.
An accepted run requires both:

- `client_tool_completion = PASS`;
- `correctness = PASS`.

Correctness is determined only by the pinned checker plus the path/tool budget.
Generated prose is not acceptance authority.

## Metrics

Per observed run, record when available:

- wall time: client task start to client task termination;
- queue wait: durable Job creation to authoritative claim/start;
- TTFT: authoritative Job start to first externally visible job event;
- generation time: authoritative Job start to terminal completion;
- task attempts;
- remote planning/review/repair event counts;
- escalation count;
- actually observed remote usage units.

A metric that was not observed is null plus the corresponding measurement gap.
The recorder preserves null as UNKNOWN.

For repeated tasks, the public report uses lane-level counts and medians. It does
not create a universal quality score.

## Status and disposition

Run statuses are:

- `accepted`: tool completion and checker correctness both PASS;
- `rejected`: output completed but checker correctness FAIL;
- `failed`: infrastructure/client/model failure prevented accepted output;
- `escalated`: the local path explicitly escalated to another path;
- `not_run`: no observation exists.

Adapter disposition is one of `accepted`, `experimental`, or `rejected`.
The recorder refuses `accepted` when either comparison lane still contains
NOT_RUN records.

Therefore a local-only first pilot normally remains `experimental` until a
genuine direct-remote baseline exists.

## Recorder

Prepare a private JSON input following `coding-pilot-v1` and run:

```sh
astrumweaver-coding-pilot \
  --input /private/pilot.json \
  --evidence validation/coding-pilot/coding-v1.md
```

The input file is private and must not be committed. The generated Markdown is
the only public pilot evidence.

## Interpretation boundary

A successful local pilot establishes bounded consumer usefulness only for the
pinned scope. It does not prove compatibility with another model/client version,
GPU topology or repository.

A future subscription/plan decision additionally requires representative remote
usage observations, peak-window headroom, preserved task quality and separate
electrical/operational cost accounting. #97 records technical evidence; it does
not authorize a billing or plan change.