# System-One decision serving contract

Tracking: #91 / #96. Shared serving substrate: #93.

## First implementation scope

The first decision backend is llama.cpp's native System-One endpoint,
`POST /v1/systemone`, running a GGUF model that advertises
`architecture.output_modalities = ["decisions"]` (or includes
`"decisions"`) through `GET /v1/models`.

A normal text-generating LLM that can be prompted to classify is **not** a
supported decision deployment. Worker startup and health fail closed unless the
loaded model advertises native decision output.

The public AstrumWeaver surface is namespaced rather than presented as an
OpenAI API:

```text
GET  /v1/decision-profiles
POST /v1/decisions
```

The adapter is shadow-mode only. It produces recommendations and measurements;
it does not grant execution, permission, review, merge, release, billing or
acceptance authority.

## Input contract

`POST /v1/decisions` accepts exactly:

- `profile`: one configured logical decision profile ID;
- `profile_revision`: the exact immutable profile revision expected by the
  caller;
- `state`: non-empty decision context;
- `question`: non-empty decision question;
- `choices`: an ordered list of at least two objects containing exactly
  `id` and `label`.

Choice IDs are stable client identifiers and must be unique. Labels are
descriptions and may tokenize to multiple tokens.

The gateway bounds request bytes, state/question/choice bytes, choice count,
state/question/label token counts and aggregate token count. The Worker repeats
the semantic identity and all applicable byte/count checks, and performs token
checks with the active llama.cpp tokenizer before calling the decision endpoint.

## Ordering

Client choice order is part of the request contract.

The provider request does not reuse caller IDs as map keys. AstrumWeaver maps
the ordered choices to deterministic positional provider keys:

```text
0000
0001
0002
...
```

and maps returned scores back to the original ordered IDs. This prevents
lexicographic ordering of arbitrary client identifiers from changing the
meaning of an otherwise identical request.

A changed choice order is still a changed model input and may affect model
behavior. The shadow acceptance procedure explicitly probes this rather than
claiming order invariance.

## Score semantics

The first adapter exposes exactly:

```text
score_kind = choice_set_probability
provider_score_semantics = llama-cpp-systemone-temperature-softmax-v1
```

These values mean:

1. llama.cpp evaluates the native decision head supplied by the loaded decision
   model;
2. model-provided temperature metadata is applied by the provider;
3. a softmax normalizes scores over the choices in the current question;
4. the returned finite choice scores sum to approximately one.

The scores are **conditional on the supplied choice set**. They are not
unconditional event probabilities and are not comparable across arbitrary
choice sets without a separate validation contract.

AstrumWeaver validates that the provider returns exactly the expected positional
keys, finite scores in `[0, 1]`, a normalized sum, a maximal selected choice,
and zero generated output tokens. Invalid provider output fails the Job rather
than being repaired or silently renormalized.

## Calibration

Probability normalization is not calibration.

The initial identity is:

```text
calibration_status = uncalibrated
calibration_reference_sha256 = null
```

The upstream/provider `confidence` field is not surfaced as correctness
confidence and is not accepted as calibration evidence.

A future `calibrated` status requires an explicit immutable calibration
reference digest and therefore creates a different
`DecisionSemanticsIdentity`. A profile may never silently promote an
uncalibrated deployment to calibrated.

## Abstention

Each reviewed decision profile pins `abstain_below` inside its immutable
decision-semantics identity.

If the largest choice-set probability is below that threshold, the request is a
successful decision observation with:

```json
{
  "choice_id": null,
  "abstained": true
}
```

Abstention is not a backend failure. The ordered score vector is still returned
for shadow evaluation.

Threshold selection is reviewed client/configuration policy. It is not a
permission threshold and never authorizes an action.

## Immutable semantics identity

`DecisionSemanticsIdentity` includes:

- exact deployment revision;
- provider adapter identity;
- score kind;
- provider score semantics;
- calibration status/reference;
- abstention threshold;
- shadow-mode status.

Its canonical SHA-256 revision is stored as the serving contract's
`semantic_revision`. The gateway snapshots that binding in the durable Job and
the Worker rechecks the semantic revision before provider execution.

Changing any of these fields creates a new semantics ID and therefore requires a
new reviewed serving contract/profile.

## Durable execution

Each accepted request compiles to one durable `decision.system_one` Job using
the shared #93 admission, priority/FIFO, capacity, claim, lease, runtime epoch,
health, cancellation and retry fencing.

Control does not interpret decision state, choices or scores. Workload semantics
remain at the gateway/provider adapter boundary.

Provider-local execution uses `POST /v1/systemone`. The managed runtime is
considered ready for decision serving only if `GET /v1/models` confirms the
configured alias and includes the native `decisions` output modality.

## Response contract

A successful response contains:

- `choice_id` or explicit abstention;
- ordered `{id, score}` entries matching the caller's choices;
- `score_kind`;
- explicit calibration status/reference;
- provider usage;
- namespaced exact profile/deployment/contract/semantics identities;
- provider score-semantics identifier;
- abstention threshold;
- `mode = shadow`;
- `authority = recommendation-only`;
- queue wait, execution and durable-job duration measurements.

The response never interprets a score as permission or task correctness.

## Shadow acceptance

The packaged `astrumweaver-decision-shadow-accept` runner consumes a private
labelled fixture. Fixture state/question/choice text and case IDs are not
published.

The public evidence records only:

- fixture digest and case counts;
- answered/abstained counts;
- correct/incorrect observations;
- deliberate-unknown abstention coverage;
- false-safe count;
- coverage and observed accuracy values;
- changed-order probe result;
- end-to-end and fabric latency aggregates;
- exact serving/semantics identity;
- explicit uncalibrated/shadow/recommendation-only boundaries.

Quality measurements are `OBSERVED_ONLY`; a successful protocol run does not
promote the model to an authority. A poor accuracy or non-zero false-safe count
is retained as evidence rather than rerun away.

See [Decision shadow acceptance](../decision-shadow-acceptance.md).

## Observed real shadow acceptance

The first real native OpenJev shadow run completed on 2026-10-07 against
AstrumWeaver revision
`fd412bee45a8c3f1c2487d4878137f7348d4c868`.

The live deployment/profile/contract/semantics identity was preflighted before
the fixture was run. All three labelled cases and the reversed-order probe
completed through the durable decision Job path.

Observed quality is deliberately retained as measurement rather than promoted
to an acceptance threshold:

- 3 cases, all answered;
- 2 correct / 1 incorrect;
- overall and answered accuracy: 0.666667;
- deliberate-unknown abstention: 0/1;
- false-safe count: 0;
- reversed-order probe did not change the selected decision;
- median end-to-end wall latency: 22521.894 ms;
- median provider/Worker execution: 22462.674 ms;
- median queue wait: 13.030 ms;
- median wall-minus-execution overhead: 69.910 ms.

The result is therefore `OBSERVED_ONLY`: protocol, identity, score validation
and shadow-mode execution passed, while the small fixture does not establish
model quality or calibration. In particular, the failure to abstain on the
deliberate unknown is retained as a real limitation.

Canonical redacted evidence is
[`validation/decision/shadow-v1.md`](../../validation/decision/shadow-v1.md).