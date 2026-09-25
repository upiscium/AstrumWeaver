# FreeToken Runtime Provider

AstrumWeaver supports FreeToken as the first-class provider for RAM-heavy,
VRAM-constrained Mixture-of-Experts workloads on a single NVIDIA GPU.

This is intentionally an AstrumWeaver v0.x product scope, not a claim about
the complete upstream FreeToken feature set. Upstream FreeToken can also serve
dense models; AstrumWeaver currently reserves its FreeToken provider for the
MoE/offload role so runtime selection remains predictable.

## Scope

The provider owns:

- FreeToken package/setup intent
- one local \`ft serve\` process
- exact Worker GPU UUID selection
- MoE strategy and cache/offload launch policy
- health and served-model verification
- OpenAI-compatible \`llm.chat\` and \`text.generate\`
- cancellation of in-flight HTTP requests
- runtime residency/status reporting

It does not own:

- VM/LXC creation
- GPU passthrough
- NVIDIA driver installation
- CUDA toolkit or driver replacement
- multi-GPU FreeToken execution
- hidden runtime substitution
- provider-specific Control scheduling logic

## Current upstream prerequisites

The current upstream FreeToken package targets Linux x86_64 and NVIDIA GPUs.
Its accelerated installation is published as:

\`\`\`text
freetoken[accel]
\`\`\`

AstrumWeaver therefore rejects non-x86_64 hosts in this provider.

Driver/toolkit suitability remains part of host/runtime preflight. The
provider does not mutate the host NVIDIA driver to satisfy FreeToken.

## Provider identity

\`\`\`text
provider_id = freetoken
\`\`\`

The provider advertises:

\`\`\`text
llm.chat
text.generate
\`\`\`

The executor uses the FreeToken OpenAI-compatible endpoints:

\`\`\`text
/v1/chat/completions
/v1/completions
\`\`\`

and verifies runtime state through:

\`\`\`text
/health
/v1/models
/v1/stats
\`\`\`

Streaming is disabled at the generic AstrumWeaver JobExecutor boundary.

## Model topology

AstrumWeaver v0.x requires:

\`\`\`text
model_topology = moe
\`\`\`

A dense-model demand is rejected with:

\`\`\`text
model-topology-outside-provider-scope
\`\`\`

The diagnostic explicitly states that this is an AstrumWeaver provider scope
restriction, not an upstream FreeToken limitation.

## Model formats

The first provider accepts:

\`\`\`text
huggingface
hf
safetensors
ftw
\`\`\`

Hugging Face-style remote model references produce a reviewed SetupPlan
download action.

Local filesystem paths use reference verification.

FTW is treated as a local FreeToken checkpoint format, so the v0.x provider
requires an explicit local FTW path rather than interpreting an arbitrary
remote string as an FTW checkpoint.

GGUF is outside this provider contract.

## GPU topology

The first provider requires exactly one scheduler-owned GPU:

\`\`\`text
gpu_topology = single_gpu
Worker GPU count = 1
Worker GPU UUID count = 1
\`\`\`

The exact Worker UUID is passed directly to:

\`\`\`text
ft serve --gpu <GPU UUID>
\`\`\`

The provider does not rewrite the physical identity into an ordinal and does
not silently select only one device from a multi-GPU Worker.

A multi-GPU Worker therefore fails compatibility with:

\`\`\`text
single-gpu-required
\`\`\`

This preserves the AstrumWeaver invariant that a Worker is one exclusive
scheduler-visible compute unit.

## Residency policy

### cpu_gpu_hybrid

This is the primary FreeToken path.

The default MoE strategy is:

\`\`\`text
auto
\`\`\`

Upstream FreeToken currently resolves MoE auto mode toward offload, with hybrid
selection available when its bandwidth profile recommends it.

AstrumWeaver does not reproduce FreeToken's hardware auto-tuning algorithm.

### prefer_vram

Supported for MoE workloads.

FreeToken retains its expert-cache/offload behavior while the configured
memory ratio controls the GPU memory budget. This is not treated as
full-VRAM residency.

### vram_only

Not claimed by the v0.x FreeToken provider.

A request fails with:

\`\`\`text
vram-only-unsupported
\`\`\`

This keeps FreeToken focused on the RAM-heavy / VRAM-constrained role and
avoids pretending that model-size evidence alone proves all runtime memory
requirements.

## Host RAM semantics

ExecutionDemand remains authoritative for explicit resource requirements:

- \`min_host_ram_mb\` is blocking
- \`preferred_host_ram_mb\` is advisory
- total/single-device VRAM requirements use the generic planner

The FreeToken provider does not derive an exact required host-RAM value from
\`estimated_size_mb\`.

MoE expert banks, quantization, KV/cache pools, pinned buffers, CPU execution
and model-specific structures make such a simple formula unreliable.

Instead the provider emits a non-blocking advisory that usable model size and
performance depend on:

- host RAM capacity
- host memory bandwidth
- CPU-GPU interconnect bandwidth
- chosen MoE strategy
- model/expert format
- cache sizing

Operators who require a hard RAM floor should express it through
ExecutionDemand.

## Launch policy

The provider renders a deterministic \`ft serve\` command using:

\`\`\`text
ft serve
  --model <model>
  --host 127.0.0.1
  --port 1919
  --gpu <exact Worker GPU UUID>
  --served-model-name astrumweaver
  --moe-strategy <auto|offload|cpu|hybrid>
  --memory-ratio <ratio>
  --max-running-requests <count>
\`\`\`

Optional provider-local settings include:

- \`moe_cache_size\`
- \`moe_cache_rate\`
- \`moe_cpu_threads\`
- \`moe_cpu_layers\`
- \`moe_hybrid_max_fetch\`

\`moe_cache_size\` and \`moe_cache_rate\` are mutually exclusive.

If neither is supplied, the provider leaves cache sizing to FreeToken rather
than duplicating its auto-sizing logic.

## SetupPlan integration

The RuntimeSetupIntent records:

- package reference
- loopback endpoint
- executable
- model reference
- served model alias
- exact GPU UUID
- MoE strategy
- memory ratio
- optional cache/CPU/hybrid controls
- request concurrency
- executor capabilities

Remote model references use the shared reviewed download action.

Local paths use the shared reference-verification action.

No provider-specific setup mutation path exists.

## Process ownership

The ManagedRuntime owns one local FreeToken child process.

Startup:

1. refuses to adopt a healthy external server on the configured endpoint
2. starts \`ft serve\` if the provider does not already own a process
3. waits until \`/health\` reports \`status=ok\`
4. verifies the configured served-model alias through \`/v1/models\`
5. stops the process it started if startup validation fails

Stop is idempotent.

Release closes the provider API client idempotently. Higher-level lifecycle
management remains responsible for stop-before-release ordering.

## Health

A reachable \`/health\` endpoint is not enough.

FreeToken reports lifecycle states such as loading, ok and error, so the HTTP
adapter considers the runtime ready only when:

\`\`\`text
status = ok
\`\`\`

The ManagedRuntime also requires the expected model alias to appear in
\`/v1/models\`.

A healthy endpoint without an owned child process is reported as a failed
external-process collision rather than silently adopted.

## Job execution

Each request is bound to the model selected during setup.

A job may omit \`model\`, or specify either the original model reference or the
served alias. Any other per-job model value fails closed.

The executor forces:

\`\`\`text
stream = false
\`\`\`

Token usage from OpenAI-compatible responses is copied into generic metrics
when present:

- prompt tokens
- completion tokens
- total tokens

Raw provider responses remain available in \`JobResult.outputs\`.

## Cancellation

The executor tracks each in-flight HTTP request task by AstrumWeaver job ID.

\`cancel(job_id)\` cancels that local request task.

Durable cancellation, fencing, retry and stale-result handling remain
Worker/Control responsibilities.

## Residency and stats

The provider always reports the reviewed placement contract as metadata:

- residency policy
- GPU topology
- exact GPU UUID
- host RAM fact
- MoE strategy
- memory ratio
- cache/CPU/hybrid options

It does not synthesize accelerator byte counts.

When FreeToken \`/v1/stats\` provides a numeric \`vram_bytes\` measurement, that
observed value is exposed as \`accelerator_memory_bytes\`.

If the runtime does not report the value, the field remains unset.

The provider may also record observed model identity and the upstream MoE flag
from \`/v1/stats\`.

## Current limitations

The first provider deliberately does not claim:

- multi-GPU FreeToken execution
- dense-model execution through AstrumWeaver
- \`vram_only\` execution
- automatic derivation of required host RAM from checkpoint size
- GPU-driver installation or replacement
- CUDA-toolkit installation
- exact host-memory residency bytes
- runtime cache rebuild control through \`/v1/cache/rebuild\`
- multimodal capability routing
- Anthropic or Responses API as separate AstrumWeaver capabilities

These can be added later as provider-local extensions without changing the
Control-plane scheduling model.
