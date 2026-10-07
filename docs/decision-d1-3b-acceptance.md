# Liquid AI d1-3B System-One acceptance

Tracking: #91, #96, #114. This is an additional immutable deployment qualification, **not** a replacement for historical OpenJev evidence.

## Exact model target

- Publisher: Liquid AI
- Source model: `LiquidAI/d1-3B`, released 2026-10-07
- GGUF repo: `LiquidAI/d1-3B-GGUF`
- Pinned GGUF repo revision: `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`
- Initial text-only artifact: `d1-3B-Q4_K_M.gguf`
- Runtime interface: llama.cpp native `POST /v1/systemone`
- Multimodal image-input qualification: **NOT_RUN**, separate mmproj/adapter validation required

The GGUF artifact digest and runtime binary/revision must be measured locally before publishing a serving profile. A model swap always changes the deployment revision; never retarget an existing profile revision.

## Acceptance contract

The first candidate uses the existing `decision.system_one` adapter with:

```text
score_kind = choice_set_probability
provider_score_semantics = llama-cpp-systemone-temperature-softmax-v1
calibration_status = uncalibrated
mode = shadow
authority = recommendation-only
```

The provider must advertise native `decisions` output, return complete finite normalized scores, preserve choice-ID ordering, and report zero generated output tokens. This contract is specific to **the verified llama.cpp build and model**; no externally claimed model calibration implies empirically calibrated correctness for our private fixture.

The client retains all tool, review, permission and acceptance authority. Scores never authorize actions.

## Exact qualified deployment

Text-only, CPU-backed real acceptance was performed using:

- GGUF repository revision: `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`.
- GGUF artifact SHA-256: `16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402` (verified against upstream LFS identity).
- llama.cpp source revision: `bd4eeaa047006cb1fe71999fbd11134b5836e167`.
- compiled CPU llama-server SHA-256: `ee742744ca602d90349c527b2cf638e64007b73e6b31b68cb2a0e748ecdfdcbe`.
- AstrumWeaver acceptance source: `996678b1fb5820bc619e81df244f32dade21fc3c`.
- approved logical profile ID: `d1-3b-system-one-v1`.
- profile revision: `sha256:8a2739ea6d7b504f7e55f7e225959a7b04aed191aa1ea9466daa35bf30d58a4a`.
- deployment revision: `sha256:d7265af19eef56107bde73f92addb55526eacbd2e7aad6b7df80c4de6bf4062b`.
- serving contract revision: `sha256:3a238cf59e7fb1fc0ba451ef23b79b4082a07beee472a0b68dbb655ed281f779`.
- decision semantics ID: `sha256:246687d981a1cc4da310834de147f83153829f84d38c3265c5796140ae683d5e`.
- context size: 4096, single concurrent request, timeout: 120 seconds, pinned abstention threshold: 0.65.
- model tokenizer: embedded in the verified GGUF; no external tokenizer substitution.

The original older llama.cpp build `42b021b` failed at model initialization with `unsupported decision model type: lfm2-d1`. The exact qualified build above recognizes that type and must not be silently replaced with the older binary.

## Real direct-provider verification

With the pinned model and binary, the real `GET /v1/models` advertised `input_modalities=["text"]` and `output_modalities=["decisions"]`. The real `POST /v1/systemone` returned all three expected positional choice keys with finite normalized probabilities, a maximal selected key, and `usage.output_tokens=0`; a descriptive choice label tokenized to multiple tokens. The response passed the unmodified AstrumWeaver `validate_decision_provider_response` validator.

## Real shadow acceptance

The existing packaged `astrumweaver-decision-shadow-accept` ran with an isolated private labelled fixture through the full PostgreSQL-backed Control/Worker admission path: 5 cases + 1 reversed-choice ordering probe, **6/6 durable Jobs succeeded**. The exact preflight verified the new, separate profile/deployment/contract/semantics identities.

Observed aggregate (not a general model benchmark):

| Metric | Value |
| --- | ---: |
| Protocol/identity acceptance | **PASS** |
| Overall task accuracy | 2/5 (0.400000) |
| Answered | 3/5 |
| Abstained | 2/5 |
| Correct among answered | 2/3 |
| Deliberate unknown abstention | 0/1 |
| False-safe | 0 |
| Changed choice ordering changed decision | false |
| Median client wall time | 2043.302 ms |
| P95 client wall time | 2077.260 ms |
| Median Worker/provider execution | 1968.296 ms |
| Median queue wait | 40.230 ms |
| Median wall-minus-execution overhead | 75.663 ms |

- Canonical redacted evidence: [`validation/decision/d1-3b-shadow-v1.md`](../validation/decision/d1-3b-shadow-v1.md).
- Generated evidence SHA-256: `b148d3c2a05164bfb67df568e528daad6e1ec4ed63b6d0c861a664d7173d101e`.
- Fixture identity: `sha256:0ce83f866ff1172343318cb4915c312a99ef112ebf6ea00e2e6977e1885a515c` (raw fixture never committed).
- Quality disposition: **OBSERVED_ONLY / experimental**; the deliberate unknown was not abstained on. Preserve the result without tuning the threshold against the same fixture and relabeling it as calibrated.

This is a **new model qualification**; the older OpenJev evidence remains in `validation/decision/shadow-v1.md`. The runs use different labelled fixtures and model builds, so these latencies and accuracies must not be presented as a matched head-to-head benchmark.

## Reproduction outline

1. Download the pinned `d1-3B-Q4_K_M.gguf` from the revision above and verify the artifact SHA-256.
2. Build the pinned llama.cpp revision and explicitly verify `lfm2-d1` support in `server-decision.cpp`.
3. Validate the local model in a text-only `llama-server` instance with the native decision modality and `/v1/systemone` response.
4. Build an operator-reviewed **new** serving declaration and decision catalog with the exact identities, limits and abstention policy above. Do not edit the historical OpenJev binding.
5. Start a single decision-capable Worker and Control, then run the existing [decision shadow acceptance](decision-shadow-acceptance.md) CLI against the new profile and a private labelled fixture.
6. Compare redacted aggregates only. Before autonomous routing or quality acceptance, obtain larger held-out and calibrated task-specific evidence.

No image support was qualified: the GGUF model's optional vision projector was not loaded, so this deployment advertises text input only. Scores retain **uncalibrated** and **recommendation-only** status regardless of the model card's broader calibration claims. No production deployment/GPU change or permission/merge/release authority was granted.
