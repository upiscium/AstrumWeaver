# Decision shadow acceptance

Tracking: #91 / #96.

This procedure validates the first native System-One decision adapter without
granting the decision model any automation authority.

## Preconditions

Use an exact reviewed decision deployment and logical profile.

The runtime must expose a native decision model through llama.cpp. The Worker
fails readiness if the configured model does not advertise the `decisions`
output modality.

The first accepted semantics are intentionally:

```text
score_kind = choice_set_probability
provider_score_semantics = llama-cpp-systemone-temperature-softmax-v1
calibration_status = uncalibrated
mode = shadow
authority = recommendation-only
```

## Private fixture

The runner consumes a JSON file with schema
`decision-shadow-fixture-v1`:

```json
{
  "schema_version": "decision-shadow-fixture-v1",
  "choices": [
    {"id": "choice-a", "label": "private description"},
    {"id": "choice-b", "label": "private description"}
  ],
  "cases": [
    {
      "id": "private-case-id",
      "state": "private state",
      "question": "private question",
      "expected_choice_id": "choice-a"
    },
    {
      "id": "private-unknown-id",
      "state": "private ambiguous state",
      "question": "private question",
      "expected_choice_id": null
    }
  ],
  "unsafe_expected_choice_ids": ["choice-b"],
  "false_safe_choice_ids": ["choice-a"],
  "ordering_probe_case_id": "private-case-id"
}
```

`expected_choice_id = null` is an intentionally unknown case. It is counted as
correct only when the configured profile abstains.

`unsafe_expected_choice_ids` and `false_safe_choice_ids` define the fixture's
false-safe metric. They are private fixture semantics; the public evidence
contains only the aggregate count.

The fixture file itself must not be committed. Its canonical JSON SHA-256 is
recorded in evidence.

## Run

Provide the live gateway URL and Client credential only through the environment:

```sh
export ASTRUMWEAVER_DECISION_BASE_URL='https://gateway.example.invalid/v1'
export ASTRUMWEAVER_CLIENT_TOKEN='REDACTED'

astrumweaver-decision-shadow-accept \
  --profile-id decision-local-v1 \
  --profile-revision sha256:PROFILE_REVISION \
  --deployment-revision sha256:DEPLOYMENT_REVISION \
  --serving-contract-revision sha256:CONTRACT_REVISION \
  --decision-semantics-id sha256:SEMANTICS_REVISION \
  --revision ASTRUMWEAVER_GIT_REVISION \
  --fixture /private/decision-shadow.json \
  --evidence validation/decision/shadow-v1.md
```

The runner first performs a live `GET /v1/decision-profiles` identity
preflight. It refuses to run if profile/deployment/contract/semantics revisions,
score semantics, calibration status, shadow mode or recommendation-only
authority differ from the expected contract.

It then executes every labelled case through the live
`POST /v1/decisions` durable path.

## Measurements

The public evidence records aggregate observations only:

- answered vs abstained;
- correct vs incorrect under the private fixture labels;
- deliberate-unknown abstention rate;
- false-safe count;
- coverage;
- overall and answered accuracy;
- changed-order probe result;
- median and p95 client wall latency;
- median queue wait;
- median Worker/provider execution;
- median wall-minus-execution fabric overhead.

The ordering probe reruns one fixture case with the choice list reversed. A
changed result is recorded, not silently treated as impossible.

## Interpretation

`Overall = PASS` means the pinned protocol/identity/validation path completed
correctly and emitted private-safe evidence.

It does **not** mean the model has passed a universal quality threshold.

Quality is reported as:

```text
Quality disposition = OBSERVED_ONLY
```

until a separate experimental design defines a justified quality/calibration
gate.

Likewise, normalized choice probabilities remain uncalibrated. The decision
adapter cannot authorize shell execution, repository writes, permissions,
protected-branch operations, human-review bypass, release, or billing changes.

## Public evidence boundary

Generated evidence omits:

- gateway URL and credentials;
- fixture state/question/choice text;
- fixture case IDs;
- local paths;
- host/Worker/GPU identifiers;
- model artifact path.

It may include immutable public-safe content digests/revisions and aggregate
numeric observations.