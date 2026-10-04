# Deployment-bound capability serving

Status: proposed detailed contract; approved direction, not implemented serving support.

Tracking: [Epic #91](https://github.com/upiscium/AstrumWeaver/issues/91),
[design PR #92](https://github.com/upiscium/AstrumWeaver/pull/92).

Inspected source baseline: `118baa14e525eb933e12bc0d79751c39ae66fd7f`.

This is a post-v0.1 feature track. It does not amend the v0.1.0 release decision,
repair #89, or operator-only Real Smoke #80. No installation procedure, live
configuration, database, GPU exposure, tag or release is changed by this document.
The JSON below illustrates proposed semantics; it is not accepted configuration
for the inspected baseline.

## 1. Reuse the existing fabric

AstrumWeaver is already a capability-based compute fabric, not an LLM proxy
that must now be generalized. The implementation work should fill serving gaps,
not rebuild the ownership model.

| Existing boundary | Retained responsibility | Proposed addition |
| --- | --- | --- |
| WorkerSpec | Exclusive resource owner and scheduler identity | Deployment-bound serving advertisement |
| RuntimeDeploymentSpec | Reviewed provider/configuration/execution demand | Immutable execution identity and contract revision |
| ManagedRuntime | Process, health, residency and executor lifecycle | Per-start instance identity and verified serving readiness |
| JobExecutor | Opaque request, structured result, cancellation | Workload adapters; optional generic progress events |
| Control repository | Durable Jobs, atomic claims, capacity, leases | Generic deployment binding and bounded serving admission |
| Existing-node setup | Reviewed installation and runtime preparation | Explicit preparation of supported serving deployments |
| New logical serving profile | No existing application alias is assumed | Versioned client name for an approved execution contract |

See [architecture](../architecture.md), [Worker contract](../worker-contract.md),
[Runtime Providers](../runtime-providers.md), [executor contract](../executor-contract.md),
[Control Plane](../control-plane.md), and [protocol v1](../protocol-v1.md).

At the inspected baseline, the public fabric API is the generic job protocol.
The universal executor contract does not require streaming. Neither generic
`llm.chat` advertisement nor a provider's HTTP endpoint establishes a tested
OpenCode-compatible edge API, embedding adapter, or decision adapter.

## 2. Responsibilities and topology

A Worker remains the smallest scheduler-visible exclusive compute unit. It may
own one GPU, multiple GPUs, or a supported non-GPU resource shape. A deployment
may choose one GPU per LXC and one Worker per LXC without making that a universal
core restriction.

Do not introduce a second persistent Slot identity or another GPU owner table.
For this increment, “slot” is simply a descriptive name for the resource envelope
owned by a Worker. LXC and physical-host identities stay outside scheduling.

```text
Client application / agent runner / indexer
                    |
           Serving gateway adapters
       profile resolution + schema validation
                    |
       Generic admission / durable Control Jobs
                    |
          Atomic Worker pull claim + lease
                    |
          Worker-owned deployment instance
                    |
           RuntimeProvider -> JobExecutor
```

The gateway is a logical module boundary; it need not force another distributed
service initially. It must not import workload-specific matching rules into
Control or become a second scheduler with independent GPU reservations.

A fabric Job is one inference request, bounded embedding batch, or decision
request. An OpenCode task may cause many Jobs. Repository access, worktrees,
compilation/tests, tool execution, task decomposition, permissions, final review
and remote-agent escalation belong to the client/agent system. A generated tool
call is data, not authority to execute the tool.

Initial deployment policy: one active deployment and one executing Job per
Worker. Sharing several resident models, automatic eviction and GPU regrouping
are deferred. Unused capacity does not authorize an unreviewed deployment change.

## 3. Four distinct identities

**Worker identity** names the resource owner and retains the existing exclusive
GPU UUID and lifecycle rules.

**Deployment revision** names immutable execution intent. It binds the explicit
provider, runtime/adapter build, prepared model artifact/revision, quantization,
tokenizer/template where applicable, and non-secret configuration affecting
execution. Use artifact identities, not mutable model tags or private paths as
identity. Credentials and endpoints are private references, never digest inputs
that purport to make secret values safe to publish.

**Runtime instance epoch** identifies one started deployment instance. Restarting
the same revision produces a new epoch. Changing weights, relevant preprocessing
or execution configuration changes the revision, not merely the epoch.

**Serving contract revision** binds the deployment to the operation schema,
features, effective limits and evidence scope it can actually serve. It is not
a benchmark score. Advertise only after preparation, identity checks, executor
validation and runtime readiness have passed.

Desired deployment, observed residency and healthy serving are different states.
A file on disk is not proof of a loaded model; a live process is not proof of a
supported tool-call format. Provider compatibility must be checked on the actual
Worker resource facts. Missing evidence is unknown, not an implicit pass.

Registration remains within the existing authenticated Worker trust domain;
these identities do not claim stronger per-Worker attestation than that protocol
provides. Exact canonical serialization, extension negotiation and persistence
are implementation obligations in #93.

## 4. Capabilities, features and performance

Keep capability identifiers opaque to Control. The first workload adapters use:

| Primary capability | Meaning | Adapter-owned contract details |
| --- | --- | --- |
| `llm.chat` | Structured conversational inference | Roles, tools, output format, actual streaming mode, context envelope |
| `text.embed` | Text-to-vector inference | Input policy, vector space, dimensions, normalization, bounded batching |
| `decision.system_one` | Bounded choice scoring | Choice IDs/order, scoring method, abstention, calibration status |

`text.embed` is a proposed identifier; the other names reuse existing vocabulary.
A capability name alone is not a claim that an adapter has been implemented.

“Coding” is initially a logical profile/use case, not a universal quality
capability. Likewise, `tools` is a feature of a particular chat contract, not
proof that every model behind a runtime uses tools correctly. Features must be
bound to the same deployment/operation, not combined across unrelated workers.

Limits reflect the deployed runtime, not merely a model-card maximum. Include
applicable input/output token limits, request/batch bounds and concurrency.
Execution memory planning must account for cache and runtime overhead rather
than treating weight-file size as the whole demand.

Measured latency, throughput, error rates and evaluation results are a separate,
versioned observation record. Include workload/context/concurrency and evidence
scope. Unknown performance stays unknown. Static operator preferences may later
rank compatible candidates, but must not override hard constraints or fabricate
numeric quality guarantees.

## 5. Logical serving profiles

A `LogicalServingProfile` is a versioned, operator-managed client contract. It is
not the existing host-sizing profile documented in `deployment-profiles.md`.

The first increment resolves each profile to one approved deployment revision
and compatible replicas. It does not automatically choose a different model,
quantization, runtime or cloud provider. Client model identifiers should name
versioned profiles; a future moving alias must be explicit and must not rewrite
an admitted request.

Illustrative profile:

```json
{
  "schema_version": "serving-profile-v1",
  "profile_id": "local-code-v1",
  "profile_revision": "example-profile-r1",
  "operation": "llm.chat",
  "deployment_revision": "example-deployment-r1",
  "serving_contract_revision": "example-contract-r1",
  "required_features": ["tools"],
  "limits": {
    "max_input_tokens": 8192,
    "max_output_tokens": 2048,
    "max_total_tokens": 10240
  },
  "fallback": "none"
}
```

The illustrative limits are not hardware recommendations. An adapter must reject
an internally inconsistent profile or one exceeding verified deployment limits.
For chat, token accounting includes system messages, templates and tool schemas,
with output reservation. Byte limits also bound work before tokenization.

A request snapshots the profile and serving/deployment revisions at admission.
At claim, the attempt binds the chosen runtime instance epoch. Retry before
external output may use a healthy new instance of the same approved revision,
subject to retry policy; it may not silently switch model contracts.

GPU identity and placement need not be exposed to ordinary clients. Exact model
or provider requests constrain resolution rather than becoming advisory hints.
The resolved execution identity must be available in private result metadata or
a documented gateway contract mechanism.

## 6. Admission and dispatch

Use the existing pull-claim scheduler, not an unowned direct proxy:

1. Resolve the requested profile revision and validate the operation schema.
2. Enforce requested features, effective limits and bounded request size.
3. Compile deployment/contract bindings into generic scheduling requirements.
4. Admit a durable Job with a deadline and an idempotency scope.
5. Atomically claim only on a healthy, ONLINE, compatible Worker with capacity.
6. Recheck the bound contract/instance locally before execution.
7. Publish fenced events/results and release capacity through existing lifecycle rules.

The current eligible-job priority-descending/FIFO ordering remains the starting
policy. Equivalent replicas compete through atomic claims; deterministic policy
does not imply deterministic Worker selection under races. Round-robin,
least-loaded placement and cross-model optimization are not promised here.

A gateway compatibility lookup is advisory until authoritative claim validation.
Do not let stale catalog data, arbitrary operator labels, re-registration or
profile changes bypass the bound execution contract. Control compares generic
identities/constraints; workload adapters interpret chat/vector/choice semantics.

Define structured errors for invalid input, unsupported features, unknown
profile, no compatible deployment, overload, deadline expiry and cancellation.
Transient lack of capacity may wait only within a bounded policy. Interactive
requests must not silently remain queued indefinitely. Bulk embedding batches
must also have bounded aggregate work; priority does not preempt an active GPU
Job in this increment.

Idempotency compares resolved intent, including contract identity. Equivalent
concurrent submissions share one Job; reuse for a different revision/payload is
a conflict. Specify client/request scoping without claiming multi-tenant isolation
that the current shared Client/Worker token model does not provide.

## 7. Gateway/API boundary

The proposed gateway exposes a documented subset, not blanket API compatibility:

| Edge surface | First scope |
| --- | --- |
| `GET /v1/models` | Operator-configured versioned serving profile names |
| `POST /v1/chat/completions` | Structured chat and the features proved for that profile |
| `POST /v1/embeddings` | Validated text/list input and supported vector encoding |
| `POST /v1/decisions` | AstrumWeaver-native experimental decision operation |
| `/v1/responses` | Deferred; not implemented or implied by chat compatibility |

Existing `/v1/jobs` behavior remains valid. New contracts need explicit extension
negotiation; old Workers must not ignore mandatory deployment-binding fields.
Breaking transport changes require a new version as specified by protocol v1.
Unknown schema versions and unsupported API fields are rejected explicitly.

Gateway Client access follows the configured Client authentication policy,
including explicitly selected `none` behind an appropriate deployment boundary.
Worker endpoints remain authenticated. Client applications never receive the
Worker token merely to call inference. No automatic paid-cloud fallback.

Payloads may contain private notes/source. The current durable job path persists
payloads/results; adding redacted metrics does not make that path metadata-only.
Specify access, payload/artifact retention and cleanup before a consumer pilot.
Do not log raw request bodies, model credentials or private topology publicly.

### Chat, tools and streaming

Preserve ordered roles, tool definitions, supported tool-choice semantics,
call IDs, argument JSON strings, tool-result messages and finish reasons. A tool
call cannot be flattened into assistant text without changing its meaning.
Tool support requires a full client round-trip test on the exact adapter/model.

Stage A may be bounded non-streaming inference. It is usable with a coding client
only after the pinned client version has demonstrated that mode. OpenCode
configuration and adapter package names must be pinned; do not mix examples from
unverified client versions.

Before advertising live streaming, provide an optional generic event channel
bound to Job, attempt, runtime instance, sequence and current lease authority.
Fence event publication and delivery against cancellation/recovery; bound buffers
and specify backpressure, disconnect handling and terminal events. Data already
delivered cannot be recalled, so no automatic retry once any output becomes
externally visible, especially tool-call output. Do not concatenate attempts.
A terminal success still requires the authoritative durable completion.

Do not simulate live token streaming by splitting a completed answer, or connect
the client directly to a runtime in a way that bypasses capacity and leases.
The detailed event transport and failure-ordering tests belong to #94; the
current executor contract is not evidence that this channel exists.

### Embedding-space identity

Dimension count does not define a vector space. Define `embedding_space_id`
from the exact model/revision, quantization, tokenizer, pooling, normalization,
dimension and query/document preprocessing/instruction policy. Record runtime
and adapter identity and validate implementation changes rather than assuming
numerical equivalence across backends.

A versioned embedding profile must not silently retarget to another space.
Changing any space-defining input requires a new identity and client-owned
index migration/reindexing. This is conservative identity, not a claim of
bitwise-identical output across arbitrary hardware. Approved replica scopes
must have numerical/quality compatibility evidence.

Validate finite vectors, dimensions, indices and input ordering. Declare accepted
text/list/token-array inputs and encodings rather than accepting every hosted-API
option. Enforce per-input and aggregate batch limits. Initially a batch returns
all results or an error, not an unnoticed partial vector set.

Indexing, chunking, corpus traversal, storage, search and reindex transactions
remain outside AstrumWeaver. Test new serving against a disposable fixture, not
by replacing an existing production index. Reranking is a later separate adapter.

### Decision scores and authority

Input is bounded state, a question and unique ordered choice IDs/labels. Output
is structured selection/abstention and explicitly typed scores. Illustrative
output from an uncalibrated scorer:

```json
{
  "choice_id": "inspect",
  "abstained": false,
  "score_kind": "normalized_choice_probability",
  "scores": [
    {"choice_id": "inspect", "value": 0.7},
    {"choice_id": "test", "value": 0.2},
    {"choice_id": "escalate", "value": 0.1}
  ],
  "calibration": {"status": "unvalidated", "reference": null},
  "deployment_revision": "example-deployment-r1"
}
```

These example values are not measured results. Normalized choice probabilities
are not calibrated probabilities of being correct. Do not invent a full score
distribution from incomplete logprobs; validate the chosen adapter's scoring
method, multi-token labels and choice-order behavior. Unknown/abstain is distinct
from a failed request. Thresholds are explicit reviewed client policy.

Decision results may recommend routing or further inspection. They never grant
permissions, bypass tests/review, approve protected writes or authorize release.
Initial use is shadow mode alongside existing decisions, not automatic control.

Measure full latency including gateway, persistence, queue/poll and inference.
A fast small model can still be a slow decision service. If fabric overhead
dominates, record that result before proposing a separate low-latency path;
do not remove ownership guarantees to obtain a better benchmark.

## 8. Lifecycle and compatibility

Retain [Worker liveness](../worker-liveness.md) and
[runtime supervision](../runtime-supervision.md). Provider failure withdraws
readiness and admission; successful startup does not imply indefinite health.

Deployment change is initially explicit: stop new claims, drain or cancel under
policy, settle fenced attempts, stop/release the old runtime, apply the reviewed
SetupPlan, verify resources and the new runtime, then register its contract and
resume admission. Do not mutate a live instance underneath an active Job.

Changing from multiple single-GPU Workers to a multi-GPU Worker remains an
operator/infrastructure operation. Control's exclusive ownership must still
reject overlap, and old processes must actually stop before reassignment. This
feature does not edit hypervisor/LXC device mappings, and environment-variable
visibility is not a substitute for the existing GPU isolation preflight.

The documentation PR changes no schema or live installation. #93 must specify
migration/version negotiation and test compatibility before code is integrated.
Legacy debug.echo/direct jobs must continue to work without claiming new serving
features. Keep feature changes out of the active v0.1 release acceptance candidate
unless the operator explicitly reschedules and revalidates that candidate.

## 9. Staged delivery and acceptance

| Issue | Scope | Dependency / first acceptance |
| --- | --- | --- |
| #93 | Serving identities, descriptors, profiles, admission and claim binding | Reviewed design; pure-contract tests, then real PostgreSQL integration |
| #94 | Chat gateway, structured tools, bounded non-streaming then fenced streaming | #93; exact client/runtime tool round-trip and failure tests |
| #95 | Embedding adapter and space-safe gateway | #93; vector validation and incompatible-space negative control |
| #96 | Experimental decision adapter | #93; score validation, abstention and shadow evaluation |
| #97 | Consumer pilots and remote-rework accounting | Each pilot depends only on its corresponding adapter |

Prioritize #93, then a single-Worker coding vertical slice in #94 and its pilot.
Embedding/decision adapter work can proceed independently once shared contracts
are fixed. Do not make all adapters or a broad automated benchmarking system a
prerequisite for the first useful client test.

The coding pilot begins with read-only code discovery, a bounded test addition
and a scoped fix, using explicit tool/path budgets and isolated worktrees for
writes. Compare direct remote execution with local delegation on equivalent task
baselines and the same acceptance tests. Track remote planning/review/repair as
well as local execution; rejected local output is not saved work.

Record accepted task rate, wall-clock time, queue wait, TTFT when available,
inference time, escalation/rework and actual remote-usage observations where
available. Missing remote accounting remains unknown. Gateway-local token counts
cannot establish remote-provider quota savings. Net cost also requires separate
electrical/operational accounting. A subscription decision is client/operator
policy, not a scheduler feature or a promised outcome of this design.

Evidence must name exact code/client/runtime/model/contract identities and its
level: schema/unit, integration, real-runtime compatibility, or consumer quality.
One level does not establish the others. Keep private payloads and site inventory
out of public evidence. Existing #80 installation acceptance remains independent.

## 10. Deliberately deferred

No AI scheduler, benchmark-derived automatic quality ranking, multi-model
co-residency, automatic GPU repartitioning, cold model downloads on requests,
paid-cloud fallback, repository execution service, vector database, Responses
API parity or image/media serving is included in this increment. Future
reranking, performance-aware replica policy and explicit multi-GPU deployment
profiles can extend these contracts after measurements justify them.

## External interface references

These references describe client/API semantics, not evidence of AstrumWeaver
implementation or blanket provider compatibility. Pin exact versions during
adapter acceptance.

- [OpenCode provider configuration](https://opencode.ai/docs/providers/)
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI streaming responses](https://developers.openai.com/api/docs/guides/streaming-responses)
- [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
