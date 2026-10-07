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

## Qualification and evidence

1. Verify downloaded GGUF against its pinned repository content and record exact SHA-256.
2. Directly probe llama.cpp `GET /v1/models`, `POST /v1/systemone`, multi-token labels, score mapping and tokenization.
3. Generate a new reviewed `DeploymentIdentity`, `ServingContract`, `DecisionSemanticsIdentity` and `LogicalServingProfile`; no OpenJev identity reuse.
4. Run the existing private-safe `astrumweaver-decision-shadow-accept` through Control and a single managed Worker, including the reversed-order probe.
5. Record accepted/rejected/abstained cases, false-safe cases, wall/queue/execution overhead and explicit model-quality limitations.
6. Generate redacted `validation/decision/d1-3b-shadow-v1.md` only from a real run.

**Current status: REAL ACCEPTANCE NOT_RUN.** Preserve the existing `validation/decision/shadow-v1.md` as OpenJev historical evidence.
