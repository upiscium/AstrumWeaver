# vLLM Runtime Provider

AstrumWeaver supports vLLM as the primary GPU-resident/high-throughput RuntimeProvider.

Its role is deliberately narrower than llama.cpp:

- single-GPU GPU-resident serving
- homogeneous multi-GPU tensor parallelism
- MoE expert parallelism where applicable
- optional explicit CPU weight offload
- OpenAI-compatible high-throughput serving

For heterogeneous GPUs or fine-grained CPU/GPU layer placement, llama.cpp remains the preferred provider.

## Scope

The provider owns:

- vLLM package/setup intent
- local `vllm serve` process lifecycle
- exact Worker GPU selection
- tensor/expert parallel launch policy
- GPU-memory utilization policy
- optional explicit CPU weight offload
- health/readiness
- OpenAI-compatible JobExecutor integration
- cancellation of in-flight HTTP requests
- generic residency/status metadata

The provider does not own:

- VM/LXC creation
- GPU passthrough
- NVIDIA driver installation
- multi-node vLLM clusters
- Ray placement groups
- arbitrary model conversion
- hidden runtime substitution

## Model references

The first provider accepts model formats:

```text
huggingface
hf
safetensors
vllm
```

A local absolute/relative path uses `reference_only` setup preparation.

A non-path model reference such as:

```text
Qwen/Qwen3-8B
```

is treated as an explicitly selected remote model and produces a reviewed model-download action in SetupPlan.

AstrumWeaver does not download it merely because vLLM is compatible.

## Local server

The provider owns a loopback vLLM OpenAI-compatible server, normally:

```text
http://127.0.0.1:8000
```

Startup waits for:

```text
/health
```

and verifies the configured alias through:

```text
/v1/models
```

The ManagedRuntime refuses to attach to a healthy external vLLM server it did not start.

## GPU identity

Current vLLM supports:

```text
--device-ids <physical GPU IDs or UUIDs>
```

AstrumWeaver therefore passes the exact Worker GPU UUID set directly to vLLM.

Unlike the llama.cpp provider, it does not rewrite these GPUs into local CUDA ordinals and does not set `CUDA_VISIBLE_DEVICES`.

This preserves physical topology visibility while still restricting vLLM execution to the scheduler-owned GPU set.

## Single GPU

A `single_gpu` execution demand uses:

```text
--device-ids <Worker GPU UUID>
tensor_parallel_size = 1
```

An explicit provider configuration requesting TP > 1 is rejected.

## Multi GPU

For `multi_gpu`, v0.x requires all Worker GPUs to participate.

The default is:

```text
--device-ids <all Worker GPU UUIDs>
--tensor-parallel-size <Worker GPU count>
--distributed-executor-backend mp
```

The provider rejects a TP size that uses only part of the Worker GPU set.

This matches the AstrumWeaver Worker model: a multi-GPU Worker is one exclusive compute unit rather than a bag of independently schedulable devices.

## Homogeneous GPU requirement

The first vLLM provider requires:

```text
total_vram_mb == max_single_gpu_vram_mb * gpu_count
```

for multi-GPU Workers.

This proves equal VRAM capacity from the current Worker resource shape.

It does not claim that two cards with equal capacity are identical in architecture/performance. Runtime startup remains the final compatibility check.

Heterogeneous-VRAM Worker shapes are rejected for vLLM and are intended to use llama.cpp instead.

## Tensor parallelism

Dense multi-GPU models use tensor parallelism across the complete Worker GPU set.

The provider does not expose pipeline parallel or multi-node distributed execution in this first implementation.

## Expert parallelism

For a declared MoE model on a multi-GPU Worker, the provider enables:

```text
--enable-expert-parallel
```

by default.

This uses vLLM expert parallelism for MoE layers while the configured tensor-parallel group still spans the Worker GPU set.

Expert parallelism can be explicitly disabled by provider configuration.

Forcing expert parallelism on a dense model or a single-GPU Worker fails compatibility.

## GPU memory utilization

Provider configuration exposes:

```text
gpu_memory_utilization
```

with the upstream-style default:

```text
0.92
```

This maps to:

```text
--gpu-memory-utilization
```

The compatibility planner uses this fraction as the approximate model-executor VRAM budget.

This is a preflight estimate, not an exact OOM predictor; vLLM still performs its own runtime memory checks for weights, KV cache, graph capture, and other allocations.

## vram_only

For `vram_only`:

- model-size evidence is required
- CPU weight offload must be zero
- the estimated model size must fit the configured vLLM VRAM budget

vLLM startup itself remains the final fail-closed memory test.

AstrumWeaver does not silently enable CPU offload when a `vram_only` model fails to fit.

## prefer_vram

`prefer_vram` uses the same GPU-memory utilization budget.

CPU offload may be configured explicitly, but it is not enabled automatically.

If an estimated model size is known to exceed the configured effective VRAM/offload budget, compatibility fails before setup.

## cpu_gpu_hybrid

The first vLLM hybrid mode supports explicit UVA weight offload through:

```text
--cpu-offload-gb <GiB per GPU>
```

A `cpu_gpu_hybrid` demand is incompatible unless a positive offload budget is explicitly configured.

vLLM documents this value as CPU offload capacity **per GPU**, so the AstrumWeaver preflight treats total host RAM demand as approximately:

```text
cpu_offload_gb * gpu_count
```

and rejects obvious host-RAM overcommit.

The provider also returns a non-blocking advisory that this path is CPU-GPU interconnect sensitive because model data is accessed from CPU memory during forward execution.

This mode exists for operators who explicitly choose vLLM offload; llama.cpp remains the primary RAM-rich/VRAM-constrained runtime.

## Selective/prefetch offload

Current vLLM also provides parameter-selective and prefetch-style offload controls.

They are intentionally **not** first-class in #27.

The initial provider exposes only `cpu_offload_gb`, keeping the contract portable and easy to review.

More advanced offload can be added later as provider-local configuration without changing Control.

## SetupPlan integration

The provider setup intent contains:

- package `vllm`
- model reference
- local API endpoint
- exact GPU UUID list
- TP size
- EP enablement
- GPU-memory utilization
- explicit CPU-offload budget
- served model alias
- generation-config policy
- trust-remote-code/enforce-eager options
- generic capabilities

Remote model references create a reviewed model-download action.

Local model paths create a reference-verification action.

## Deterministic generation configuration

The provider defaults to:

```text
--generation-config vllm
```

This prevents a model repository's `generation_config.json` from silently changing server-wide sampling defaults.

The operator can override this provider option explicitly.

## OpenAI-compatible executor

`VllmExecutor` advertises:

```text
llm.chat
text.generate
```

It uses:

```text
/v1/chat/completions
/v1/completions
```

Streaming is disabled at the generic terminal JobExecutor boundary.

The Worker-bound served-model alias cannot be silently changed per job.

## Results

Raw OpenAI-compatible response data stays in `JobResult.outputs`.

Where present, token usage is copied into generic metrics:

- prompt tokens
- completion tokens
- total tokens

The generated assistant/completion text is exposed through `JobResult.text`.

## Cancellation

The executor tracks the in-flight HTTP request task by job ID.

`cancel(job_id)` cancels the local request.

Durable cancellation, lease fencing, retry, and stale-result handling remain Worker/Control responsibilities.

## Residency/status

vLLM does not expose one simple standard endpoint equivalent to Ollama `/api/ps` that directly states model-weight VRAM bytes per served model.

The provider therefore does not invent observed accelerator-memory values.

Generic ResidencyItem metadata records the reviewed runtime placement contract:

- residency policy
- GPU topology/count
- TP size
- EP state
- GPU-memory utilization
- CPU-offload GiB per GPU
- exact selected device UUIDs

Runtime health and model presence are independently checked through `/health` and `/v1/models`.

## Process ownership

The initial provider owns one local subprocess:

```text
vllm serve <model>
```

If a healthy external vLLM server is already using the endpoint, startup fails.

If a process started by the current attempt never becomes healthy or exposes the wrong model alias, that process is stopped before startup returns failure.

#31 may replace this subprocess controller with a NixOS/systemd service controller while preserving the ManagedRuntime contract.

## Current limitations

The first implementation deliberately does not claim:

- heterogeneous-VRAM tensor parallelism
- pipeline parallelism
- multi-node vLLM
- Ray execution
- data parallel serving
- selective CPU parameter offload
- prefetch/group offload
- KV-cache CPU offload
- LoRA lifecycle management
- multimodal runtime policy
- speculative decoding configuration
- observed per-GPU memory accounting

These remain provider-local extensions for later work.
