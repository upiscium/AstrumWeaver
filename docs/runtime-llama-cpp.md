# llama.cpp Runtime Provider

AstrumWeaver supports llama.cpp as the primary generic GGUF RuntimeProvider.

Its main role is to cover the execution shapes that need more explicit model-placement control than Ollama provides:

- single-GPU GGUF inference
- full/mostly-full VRAM residency
- CPU/GPU hybrid execution
- RAM-rich / VRAM-constrained inference
- heterogeneous multi-GPU workers
- supported dense and MoE offload paths

## Scope

The provider owns:

- llama.cpp package/setup intent
- local llama-server process lifecycle
- exact Worker GPU visibility
- GGUF model reference handling
- launch-policy generation
- runtime health/readiness
- generic JobExecutor integration
- cancellation of in-flight HTTP requests
- generic residency reporting

The provider does not own:

- VM/LXC creation
- GPU passthrough
- NVIDIA driver installation
- model conversion into GGUF
- arbitrary model download without explicit setup policy
- Control-plane runtime-specific scheduling

## Model format

The initial provider requires:

```text
model_format = "gguf"
```

The model reference is an existing GGUF path, for example:

```text
/models/qwen.gguf
```

The initial setup intent uses `reference_only`.

Hugging Face / URL download support can be added later as an explicit reviewed model-preparation path. It is intentionally not inferred merely from the model name.

## Server

The provider runs the upstream llama.cpp HTTP server on a local loopback endpoint, normally:

```text
http://127.0.0.1:8080
```

The server exposes OpenAI-compatible endpoints used by the generic executor:

```text
/v1/chat/completions
/v1/completions
```

Readiness is checked with:

```text
/health
```

The configured model alias is verified through:

```text
/v1/models
```

The first implementation refuses to attach to an already-running external llama.cpp server on the managed endpoint.

#31 may replace the subprocess controller with deployment-specific NixOS/systemd service controllers without changing the provider/executor contracts.

## GPU identity

AstrumWeaver restricts the subprocess with:

```text
CUDA_VISIBLE_DEVICES=<exact Worker GPU UUID set>
```

Inside that restricted process, the selected GPUs become:

```text
CUDA0
CUDA1
...
```

The provider passes this visible set through llama.cpp `--device`.

This keeps physical accelerator ownership based on stable UUIDs while allowing llama.cpp to use its native backend device identifiers.

## Residency policies

### vram_only

The provider derives:

```text
--n-gpu-layers all
--fit off
```

Planning requires an estimated GGUF size.

For a single-GPU Worker, the estimate must fit the single device.

For a multi-GPU Worker, the estimate must fit aggregate Worker VRAM.

The provider does not silently reduce GPU layers if the model cannot fit. Startup therefore fails rather than converting a requested `vram_only` execution into CPU offload.

This policy means all llama.cpp model layers that are eligible for GPU offload are explicitly requested on GPU. It does not claim that every transient runtime allocation is VRAM-only.

### prefer_vram

The provider derives:

```text
--n-gpu-layers auto
--fit on
```

This asks llama.cpp to use available device memory while retaining the ability to fit the runtime safely.

### cpu_gpu_hybrid

The default also uses:

```text
--n-gpu-layers auto
--fit on
```

This is well suited to a model larger than the available VRAM when the host has sufficient RAM.

If the operator wants an explicit partition, provider configuration can override the GPU-layer count and CPU offload controls.

## Single GPU

For `single_gpu`:

```text
--device CUDA0
--split-mode none
--main-gpu 0
```

AstrumWeaver exposes only the Worker-owned GPU to the process, so a non-zero main-GPU index is invalid.

## Multi GPU

For `multi_gpu`, the default is:

```text
--device CUDA0,CUDA1,...
--split-mode layer
--fit on
```

No `tensor-split` is emitted by default.

This allows llama.cpp to use its own device-memory information and fit/split logic, which is the preferred starting point for heterogeneous GPU sets.

### Explicit tensor split

The operator may configure:

```text
tensor_split = [2, 1]
```

AstrumWeaver then emits:

```text
--tensor-split 2,1
```

The number of proportions must match the number of Worker GPUs.

### fit target

The operator may configure one global fit margin or one per selected GPU:

```text
fit_target_mb = [2048]
```

or:

```text
fit_target_mb = [2048, 1024]
```

### split modes

Supported provider values are:

```text
none
layer
row
tensor
```

`layer` is the default multi-GPU policy.

`tensor` is exposed because upstream supports it, but the provider reports a non-blocking experimental compatibility advisory.

## CPU offload controls

The provider exposes upstream llama.cpp placement controls without moving them into Control.

### Dense models

For dense models, the operator may use:

```text
n_cpu_ffn
```

This maps to:

```text
--n-cpu-ffn
```

### MoE models

For MoE models, the operator may use:

```text
cpu_moe
n_cpu_moe
```

mapping to:

```text
--cpu-moe
--n-cpu-moe
```

The provider rejects MoE-only controls on a declared dense model, and rejects the dense FFN shortcut on a declared MoE model.

A future provider revision may add finer-grained tensor overrides, but those should remain llama.cpp-local policy rather than scheduler logic.

## Fit behavior and heterogeneous GPUs

When no explicit `tensor_split` is supplied, llama.cpp's own fit/device-memory logic is used.

AstrumWeaver deliberately does not invent per-GPU VRAM proportions from the current aggregate WorkerSpec.

This avoids duplicating llama.cpp's placement heuristics and is especially useful for heterogeneous Worker GPU sets.

A future Worker resource-shape extension could expose per-device capacity for more explicit planning, without changing Control into a llama.cpp-aware scheduler.

## SetupPlan integration

The provider setup intent contains:

- package reference `llama-cpp`
- local server configuration
- model path
- exact GPU UUID set
- residency policy-derived launch options
- split/offload settings
- generic capabilities

The shared setup backend creates a reviewable SetupPlan.

Because the initial GGUF model path is externally supplied, the plan uses:

```text
VERIFY_MODEL_REFERENCE
```

rather than silently downloading a model.

## Executor

`LlamaCppExecutor` advertises:

```text
llm.chat
text.generate
```

### Chat

Jobs use the OpenAI-compatible chat endpoint.

The generic payload supplies fields such as:

```json
{
  "messages": [
    {"role": "user", "content": "hello"}
  ]
}
```

### Text generation

Jobs use the OpenAI-compatible completions endpoint.

Example payload:

```json
{
  "prompt": "hello"
}
```

Streaming is disabled at the terminal JobExecutor boundary.

The provider binds each Worker runtime to its configured model alias; a job cannot silently switch to another model.

## Results

The raw server response remains in `JobResult.outputs`.

Where available, OpenAI-compatible usage values are copied into generic metrics:

- prompt tokens
- completion tokens
- total tokens

The assistant/completion text is also exposed through the convenience `text` field.

## Cancellation

The executor tracks one in-flight HTTP task per job ID.

`cancel(job_id)` cancels the local request.

Durable cancellation, retry, leases, and stale-result fencing remain Worker/Control responsibilities.

## Residency

llama-server does not currently expose the same direct per-model VRAM byte value that Ollama exposes through `/api/ps`.

The provider therefore does **not** invent an observed accelerator-memory number.

Instead, generic ResidencyItem metadata records the reviewed launch policy:

- residency policy
- GPU topology/count
- GPU-layer policy
- split mode
- fit policy
- tensor split
- CPU MoE/FFN offload controls

For `vram_only`, the enforcement mechanism is the fail-closed launch configuration (`all` GPU layers + fit disabled).

## Runtime ownership

The initial ManagedRuntime owns one local server process.

If a healthy server already occupies the configured endpoint but is not the process owned by this ManagedRuntime, startup fails.

This prevents AstrumWeaver from silently taking over a developer llama-server instance.

## Failure cleanup

If a process was started by the current startup attempt and the server:

- never becomes healthy, or
- does not expose the configured model alias

the process is stopped before startup returns failure.

## Current limitations

The first implementation deliberately does not claim:

- automatic GGUF conversion
- automatic model download
- observed per-GPU memory accounting
- dynamic model switching inside one Worker runtime
- router-mode multi-model serving
- RPC/distributed multi-host llama.cpp
- exact tensor overrides
- speculative-decoding configuration
- multimedia projector lifecycle

These can be added as provider-local extensions later.
