# Runtime Providers and Execution Demand

AstrumWeaver separates **what the operator wants to run** from **which runtime implements it**.

The runtime layer lives entirely on the Worker side:

```text
Control
  │
  │ generic capability/resource scheduling
  ▼
Worker
  │
  ├─ ExecutionDemand
  ├─ RuntimeSelection
  ├─ RuntimeProvider
  │      ↓
  │   ManagedRuntime
  │      ↓
  └─ JobExecutor
```

Control does not import or branch on Ollama, llama.cpp, vLLM, FreeToken, ExLlamaV3, or future runtime implementations.

## Operator authority

Runtime choice belongs to the operator.

Two selection modes exist:

### `explicit`

The selected provider ID is authoritative.

Example intent:

```text
provider = "llama-cpp"
```

If llama.cpp is incompatible with the selected Worker/model/execution policy, AstrumWeaver fails with structured incompatibility reasons.

It does **not** silently replace llama.cpp with vLLM, Ollama, or another provider.

### `recommend`

AstrumWeaver evaluates installed providers and returns compatible candidates plus incompatibility reasons.

It does not finalize a provider.

The TUI may present these candidates, but the user still makes the final choice.

A later `auto` feature may build on recommendation logic, but it must remain opt-in and must not change an explicit runtime selection.

## Execution demand

`ExecutionDemand` describes provider-neutral execution intent.

### Residency policy

```text
vram_only
prefer_vram
cpu_gpu_hybrid
```

### GPU topology

```text
none
single_gpu
multi_gpu
```

For v0.x GPU model serving, `single_gpu` means the selected Worker owns exactly one GPU and `multi_gpu` means it owns at least two.

This keeps the scheduler-visible exclusive compute unit aligned with the runtime topology and avoids silently wasting or partially using a multi-GPU Worker.

### Model topology

```text
dense
moe
```

This lets providers such as FreeToken reject unsupported dense-model requests without encoding that rule in Control.

### Model format

The format is an extensible string rather than a closed enum.

Expected initial values include:

- `safetensors`
- `gguf`
- ExLlama-compatible quantization identifiers
- provider-native formats where unavoidable

Each provider owns its exact compatibility rules.

### Resource demand

ExecutionDemand may declare:

- minimum GPU count
- minimum total VRAM
- minimum single-device VRAM
- minimum host RAM
- preferred host RAM

Minimum values are blocking.

Preferred host RAM is advisory and produces a non-blocking compatibility reason.

The total/single-device distinction remains explicit: two 12 GiB GPUs do not satisfy a 20 GiB single-device requirement.

## Host facts

Runtime planning also receives `RuntimeHostFacts`.

v0.x contains:

- CPU count
- host RAM
- architecture
- generic labels

Host discovery belongs to the setup backend (#24), not Control.

## Compatibility result

Every provider returns structured `CompatibilityReason` values.

A reason contains:

- stable code
- human-readable message
- blocking/non-blocking flag

Examples:

```text
gpu-topology-mismatch
insufficient-total-vram
insufficient-single-gpu-vram
insufficient-host-ram
below-preferred-host-ram
model-topology-unsupported
model-format-unsupported
```

Provider-specific compatibility reasons are merged with generic Worker/host resource checks.

## RuntimeProvider

The provider-neutral contract is:

```text
RuntimeProvider
  ├─ info
  ├─ compatibility(context)
  ├─ setup_intent(context)
  └─ create_runtime(context, setup)
```

### `compatibility`

Evaluates provider-specific constraints.

Examples:

- FreeToken: MoE/model-format compatibility
- llama.cpp: GGUF/offload/split compatibility
- vLLM: model format and GPU topology
- ExLlamaV3: supported quantization/model topology
- Ollama: runtime/model support

### `setup_intent`

Returns a declarative `RuntimeSetupIntent`.

It may state:

- package references
- runtime configuration
- model preparation policy
- model reference
- whether privileged actions will be required

Constructing setup intent does not execute anything.

Issue #24 converts this into a deterministic reviewed `SetupPlan`.

### `create_runtime`

Creates a `ManagedRuntime` after setup has materialized the provider.

## ManagedRuntime

The lifecycle boundary is:

```text
start()
stop()
health()
residency()
executor()
release()
```

The returned executor must satisfy the existing generic `JobExecutor` contract.

Runtime process management therefore remains separate from durable Control/Worker job ownership.

## Model preparation policy

The initial generic policies are:

```text
reference_only
download
convert
```

A provider may request one of these in setup intent, but **download/convert is still subject to SetupPlan review and explicit application**.

AstrumWeaver must not download arbitrary models merely because a provider supports them.

## Intended first-class providers

| Provider | Primary role |
| --- | --- |
| Ollama | easy/general local serving |
| llama.cpp | GGUF, CPU/GPU hybrid, heterogeneous multi-GPU |
| vLLM | GPU-resident/high-throughput, tensor/expert parallel |
| FreeToken | RAM-heavy / VRAM-constrained MoE |
| ExLlamaV3 | quantized consumer-NVIDIA VRAM-resident serving |

This table is product intent, not hard-coded planner logic.

Each concrete provider must prove its own compatibility rules through provider tests and the runtime acceptance matrix.

## Setup TUI relationship

The future TUI (#30) is a frontend over these contracts.

Expected flow:

```text
host discovery
    ↓
ExecutionDemand
    ↓
recommend compatibility reports
    ↓
display candidates and reasons
    ↓
USER chooses provider
    ↓
RuntimeSelection(EXPLICIT)
    ↓
RuntimeSetupIntent
    ↓
SetupPlan review
    ↓
explicit apply
```

The TUI must not call provider-specific shell commands directly.

This makes TUI and non-interactive automation share exactly the same planning/apply behavior.

## Infrastructure boundary

Nothing in RuntimeProvider changes the Host Prerequisite Contract.

Before provider setup, the node still must already provide:

- Linux
- GPU/device exposure
- NVIDIA driver where required
- functional `nvidia-smi`
- exact guest-visible GPU identity

RuntimeProvider begins **after** that boundary.

It may install/manage Ollama, llama.cpp, vLLM, FreeToken, ExLlamaV3, model-runtime configuration, runtime processes, and executor adapters.

It must not create/mutate the VM/LXC, passthrough configuration, IOMMU/VFIO, or host GPU driver.
