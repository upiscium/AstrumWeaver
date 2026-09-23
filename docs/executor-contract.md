# Job Execution Contract

## 1. Purpose

AstrumWeaver separates scheduling from workload execution.

The Control Plane reasons about:

- job identity
- capability requirements
- resource requirements
- worker lifecycle
- leases and fencing

It does not need to know how a particular workload is executed.

The Worker delegates a claimed job to a configured `JobExecutor`.

## 2. Generic request

`JobRequest` contains:

- `job_id`
- requested `capability`
- opaque executor payload
- optional metadata

The request intentionally does not encode LLM/chat/TTS/image-specific fields in the core contract.

Lease tokens and other fencing state belong to the Worker/Control lifecycle rather than the executor payload.

## 3. Generic result

`JobResult` may contain:

- structured `outputs`
- zero or more `ArtifactRef` values
- numeric `metrics`
- optional convenience `text`
- executor metadata

Examples:

```text
LLM/chat
  outputs = { ...provider-neutral structured values... }
  text    = "..."

Image
  outputs   = { width, height, seed }
  artifacts = [image/png reference]
  text      = None

TTS
  outputs   = { sample_rate, duration }
  artifacts = [audio/wav reference]
  text      = None

Decision/System-One
  outputs = { choice, probabilities, confidence }
  text    = None
```

The Control Plane persists generic terminal results without interpreting these application semantics.

## 4. Executor protocol

The v0.1 worker-local executor boundary is:

```python
class JobExecutor(Protocol):
    async def execute(self, job: JobRequest) -> JobResult: ...
    async def cancel(self, job_id: str) -> None: ...
    async def residency(self) -> ResidencyReport: ...
```

Returning normally means the executor completed successfully. Execution failure is represented by an exception at this boundary and is translated by the Worker into the durable job failure lifecycle.

Cancellation is a request to the executor; the Worker/Control layer remains responsible for durable cancellation and lease/fencing rules.

## 5. Residency

Residency reporting is intentionally generic.

A `ResidencyItem` has:

- a name
- an opaque kind
- optional accelerator-memory usage
- optional metadata

This can describe model weights, runtime caches, pipelines, or other resident executor state without requiring the core to assume that every worker hosts an LLM.

## 6. Text generation compatibility

`TextGenerationExecutor` demonstrates how an existing generation runtime can be wrapped behind the generic contract.

The dependency direction is:

```text
Control / scheduler
      |
      v
generic JobExecutor contract
      |
      +-- TextGenerationExecutor -> generation runtime
      +-- future TTS executor
      +-- future image executor
      +-- future decision executor
```

The text adapter is not imported by Control Plane code and does not change generic scheduling semantics.

## 7. Streaming

Token or progress streaming is not required by the universal executor contract.

An executor/worker protocol may add optional progress events later, for example:

- text token chunks
- image diffusion progress
- training progress
- media-generation progress

Terminal success/failure/cancellation must remain valid even when no streaming interface is implemented.
