# ExLlamaV3 Runtime Provider

AstrumWeaver uses ExLlamaV3 for quantized, VRAM-focused local inference on
consumer NVIDIA GPUs.

The provider intentionally uses
[TabbyAPI](https://github.com/theroyallab/tabbyAPI) as its serving surface.
ExLlamaV3 upstream names TabbyAPI as the official and recommended API backend,
so AstrumWeaver does not maintain a second custom HTTP server around the
ExLlamaV3 library.

## Scope

The v0.x provider supports:

- EXL3 model format
- `vram_only`
- `prefer_vram`
- single-GPU Workers
- multi-GPU autosplit
- multi-GPU tensor parallelism
- OpenAI-compatible chat and text completions
- cancellation of in-flight HTTP calls
- deterministic SetupPlan configuration
- explicit Worker GPU UUID ownership

The provider does not claim:

- GGUF
- arbitrary Hugging Face FP16/BF16 checkpoints
- RAM-heavy `cpu_gpu_hybrid` execution
- automatic GPU-driver/CUDA replacement
- VM/LXC or passthrough configuration

ExLlamaV3 itself has broader functionality, including CPU MoE offload and
unquantized model support. Those features are outside this AstrumWeaver v0.x
provider role so runtime selection stays predictable.

## Serving backend

The managed process is TabbyAPI with the ExLlamaV3 backend.

TabbyAPI provides the stable endpoints used by the provider:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

Authentication is disabled only because AstrumWeaver binds the managed endpoint
to loopback and sets an empty browser-origin allowlist. Exposing that endpoint
outside the Worker would require a separate authentication design.

## Package/setup boundary

Current upstream TabbyAPI is a source-style application rather than a stable
standalone console command. The provider therefore records:

- TabbyAPI/ExLlamaV3 package intent
- the TabbyAPI entrypoint
- the rendered config path
- the full deterministic TabbyAPI config

The concrete NixOS/systemd materialization belongs to the deployment driver in
#31. The provider itself does not clone repositories or mutate Python/CUDA
installations behind SetupPlan.

The default logical package intent is:

```text
tabbyAPI[cu12]
exllamav3
```

and the default launcher contract is:

```text
python /opt/tabbyAPI/main.py --config /var/lib/astrumweaver/runtime/exllamav3/config.yml
```

Those paths are deployment inputs, not hidden installation actions.

## Hardware compatibility

Current TabbyAPI explicitly checks that every selected GPU is NVIDIA compute
capability 8.0 (Ampere) or newer and rejects ROCm.

AstrumWeaver therefore understands the optional Worker label:

```text
gpu.compute_capability.min = 8.6
```

When present, values below 8.x fail compatibility. This means Pascal-class
devices such as GTX 1080 Ti are not valid ExLlamaV3 Workers, while Ampere-class
RTX 30 series devices satisfy the architecture floor.

When the label is absent the provider emits a non-blocking
`compute-capability-unverified` advisory rather than pretending the generic
Worker resource shape proves GPU architecture. Host/deployment preflight should
materialize this fact before production use.

## Model format

The first-class provider accepts:

```text
exl3
exllamav3
```

Other formats fail with `model-format-unsupported`.

Remote model references produce the shared reviewed model-download SetupPlan
action. Local filesystem references use reference verification.

For a remote reference such as:

```text
org/Qwen-EXL3
```

the provider derives `Qwen-EXL3` as the TabbyAPI model folder/name. For a
local path, the parent directory becomes `model_dir` and the basename becomes
`model_name`.

## GPU ownership

The Worker remains the exclusive scheduler-visible unit.

Every GPU UUID owned by the Worker is passed to the child process through:

```text
CUDA_VISIBLE_DEVICES=<uuid1>,<uuid2>,...
```

The UUID order is the canonical visible-device order used by TabbyAPI's
`gpu_split`.

Runtime construction revalidates the SetupIntent UUID set against the current
WorkerSpec. A modified setup payload cannot redirect the process to GPUs that
the Worker does not own.

## Single GPU

For `single_gpu`:

- Worker GPU count must be exactly one
- tensor parallelism is disabled
- explicit multi-GPU `gpu_split` is rejected

## Multi GPU

All Worker-owned GPUs participate. AstrumWeaver does not silently use a subset.

Two modes are supported.

### autosplit

```text
multi_gpu_mode = autosplit
```

TabbyAPI/ExLlamaV3 chooses placement automatically unless an explicit
`gpu_split` is supplied.

This is the default because it works naturally with heterogeneous consumer GPU
sets.

### tensor_parallel

```text
multi_gpu_mode = tensor_parallel
```

TabbyAPI receives:

```text
tensor_parallel = true
tensor_parallel_backend = native | nccl
```

`native` is the provider default and matches upstream guidance for PCIe GPU
sets. `nccl` remains an explicit operator choice for suitable interconnects.

An explicit `gpu_split` must contain exactly one positive value per
Worker-owned GPU.

## Residency policies

### vram_only

Supported, but model-size evidence is required.

If the known model estimate already exceeds total Worker VRAM, compatibility
fails before startup.

A model-size fit does not prove enough space remains for KV cache and runtime
overhead. AstrumWeaver therefore emits an advisory rather than inventing an
exact overhead formula. Operators should express hard requirements through the
generic VRAM minimum fields.

### prefer_vram

Supported and is the general-purpose ExLlamaV3 path.

### cpu_gpu_hybrid

Outside the v0.x ExLlamaV3 provider scope.

For RAM-heavy execution, AstrumWeaver should surface providers whose declared
role matches the request, especially llama.cpp or FreeToken. Explicit
ExLlamaV3 selection still fails rather than silently substituting another
runtime.

## TabbyAPI configuration

The RuntimeSetupIntent contains the deterministic config consumed by the
deployment driver. Important settings include:

```text
network.host = 127.0.0.1
network.disable_auth = true
network.allowed_origins = []
model.backend = exllamav3
model.inline_model_loading = false
model.tensor_parallel = ...
model.tensor_parallel_backend = ...
model.gpu_split_auto = ...
model.gpu_split = ...
model.cache_mode = ...
```

Optional provider inputs include:

- cache mode
- maximum sequence length
- maximum batch size
- autosplit reserve per visible GPU
- explicit GPU split
- tensor-parallel backend

## JobExecutor

The executor advertises:

```text
llm.chat
text.generate
```

Requests are bound to the model chosen during setup. A per-job `model` value
must match either the original model reference or the served TabbyAPI model
name.

Streaming is forced off at the current generic JobExecutor boundary.

Token usage fields are copied into generic metrics when present.

## Cancellation

Each request is represented by one asyncio HTTP task. `cancel(job_id)` cancels
that local request.

Durable cancellation and fencing remain Worker/Control responsibilities.

## Health and lifecycle

Managed startup:

1. rejects a healthy external process on the configured loopback endpoint
2. starts the reviewed TabbyAPI process
3. waits for `/health` to report `healthy`
4. verifies the configured model through `/v1/models`
5. stops the process it started if validation fails

Stop and release are idempotent.

## Residency reporting

The provider reports reviewed placement metadata:

- residency policy
- GPU topology/count
- exact GPU UUID set
- autosplit vs tensor parallel mode
- tensor-parallel backend
- explicit GPU split
- cache mode

TabbyAPI does not expose a provider-neutral measurement that proves exact model
accelerator residency for this contract, so AstrumWeaver does not invent
`accelerator_memory_bytes`.
