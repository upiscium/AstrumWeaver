# Ollama Runtime Provider

AstrumWeaver supports Ollama as a first-class Worker-local RuntimeProvider.

Ollama is intended as the easy/general-purpose local serving option. It is not required by the Control Plane and is not a fallback target when the operator explicitly selects another runtime.

## Scope

The provider owns:

- Ollama package/setup intent
- local runtime process lifecycle
- exact Worker GPU visibility configuration
- model preparation intent
- runtime health/readiness
- model preload/unload
- generic JobExecutor integration
- cancellation of in-flight local HTTP requests
- residency reporting

The provider does not own:

- VM/LXC creation
- GPU passthrough
- NVIDIA driver installation
- host CUDA driver replacement
- private site networking

## Model format

The initial provider accepts:

```text
model_format = "ollama"
```

The model reference is an Ollama model name, for example:

```text
qwen3:8b
```

GGUF/safetensors import workflows are intentionally not claimed by this provider yet. Those require an Ollama model-creation flow and should be added explicitly rather than silently treating arbitrary files as Ollama-native models.

## Setup planning

`OllamaProvider.setup_intent()` requests:

- package reference `ollama`
- local loopback API configuration
- explicit model download
- exact Worker GPU UUID set
- one loaded model
- one parallel request
- runtime capabilities:
  - `llm.chat`
  - `text.generate`

The generic setup backend converts this intent into a reviewed SetupPlan.

Model download is therefore network/confirmation gated and does not happen during compatibility planning.

## Local endpoint ownership

The initial provider requires a loopback HTTP endpoint, normally:

```text
http://127.0.0.1:11434
```

The ManagedRuntime refuses to attach to an already-running external Ollama server that it does not own.

This avoids accidentally reusing a developer/user Ollama daemon with unrelated models, GPU visibility, or lifecycle policy.

#31 may replace the subprocess controller with a systemd/NixOS service controller while preserving the same ManagedRuntime contract.

## GPU ownership

For NVIDIA Workers, the runtime process receives:

```text
CUDA_VISIBLE_DEVICES=<exact Worker GPU UUID set>
```

GPU UUIDs are used rather than CUDA ordinals.

This preserves the existing AstrumWeaver accelerator identity contract.

The provider also sets:

```text
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=<configured value>
OLLAMA_NO_CLOUD=true
```

by default.

The one-request/one-loaded-model defaults align with the current AstrumWeaver Worker concurrency contract and make residency/ownership easier to reason about.

## Single GPU

A `single_gpu` execution demand runs Ollama with exactly the one GPU UUID owned by the Worker.

If the execution policy is `vram_only`, planning requires model-size evidence and runtime startup performs a second check after preload.

## VRAM-only verification

For `vram_only`:

1. the estimated model size must fit the Worker total VRAM at planning time
2. the model is preloaded
3. `/api/ps` is read
4. the provider requires reported `size_vram >= size`

If Ollama reports part of the model outside VRAM, startup fails and the model/process started by that attempt is cleaned up.

This is a model-weight residency check based on the information Ollama exposes; it is not a claim that every runtime allocation or KV-cache byte is resident in VRAM.

## CPU/GPU hybrid

`cpu_gpu_hybrid` is accepted.

Ollama controls its own CPU/GPU offload placement, so compatibility returns a non-blocking advisory reason.

For operators who need exact layer/offload control, llama.cpp is the intended RuntimeProvider.

AstrumWeaver does not silently switch the selected runtime.

## Multi GPU

For an explicit `multi_gpu` execution demand:

- the Ollama process is restricted to the Worker-owned GPU UUID set
- `OLLAMA_SCHED_SPREAD=true` is set

Ollama therefore receives an explicit request to spread scheduling across the selected GPU set.

The current `/api/ps` response reports total VRAM residency but not a per-device split. AstrumWeaver intentionally does not infer per-GPU placement from total `size_vram`.

## Model identity

Ollama treats an omitted tag as `:latest`.

The provider therefore treats:

```text
model
model:latest
```

as the same model identity for health/residency checks.

Explicit non-latest tags remain distinct.

## Executor contract

`OllamaExecutor` advertises:

```text
llm.chat
text.generate
```

### llm.chat

The generic job payload supplies Ollama chat fields such as:

```json
{
  "messages": [
    {"role": "user", "content": "hello"}
  ]
}
```

AstrumWeaver injects the Worker-bound model, disables streaming for the universal terminal JobExecutor contract, and preserves the configured keep-alive policy.

### text.generate

The payload supplies fields such as:

```json
{
  "prompt": "hello"
}
```

The model is again bound by the Worker/runtime rather than freely overridden per job.

A job that tries to select a different model fails locally instead of silently changing residency.

## Results

The raw Ollama response is retained in generic `JobResult.outputs`.

Where present, the adapter also extracts:

- convenience text
- total/load duration
- prompt evaluation counts/duration
- evaluation counts/duration

The Control Plane does not interpret Ollama-specific output fields.

## Cancellation

The executor tracks the in-flight HTTP request task per job ID.

`cancel(job_id)` cancels that local request.

Durable cancellation and stale-result fencing remain Worker/Control responsibilities, exactly as for other JobExecutors.

## Residency

`residency()` maps `GET /api/ps` into generic ResidencyItems.

When Ollama exposes them, metadata includes:

- total model size
- context length

`size_vram` becomes `accelerator_memory_bytes`.

## Load and unload

The provider preloads the configured model with an empty generation request.

On stop/release it unloads the model using:

```text
keep_alive = 0
```

The ManagedRuntime is idempotent about release and closes the local API client once released.

## Failure behavior

Startup fails closed when:

- an unmanaged Ollama server already occupies the managed endpoint
- the server does not become reachable
- the reviewed model is not available after setup
- residency is absent after preload
- `vram_only` is violated

If a process was started by the failed startup attempt, the provider unloads the model when possible and stops that process before returning the failure.

## Current limitations

The first implementation deliberately does not claim:

- arbitrary GGUF/safetensors import
- a per-GPU residency breakdown from Ollama's API
- exact layer-level CPU/GPU offload control
- concurrent multi-model serving
- more than one concurrent AstrumWeaver job
- remote/shared Ollama daemon ownership

These can be extended later without adding Ollama-specific logic to Control.
