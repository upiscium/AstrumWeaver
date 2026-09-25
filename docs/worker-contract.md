# Worker and Resource Contract

## 1. Worker meaning

A Worker is the smallest scheduler-visible **exclusive compute unit**.

A worker is not synonymous with one GPU, one host, one VM, or one LXC.

Examples:

- one runtime owning one GPU
- one runtime owning two GPUs as one scheduling unit
- a CPU-only worker
- a future accelerator-backed worker represented by a stable resource identity

The infrastructure placement is intentionally outside this contract.

## 2. Worker class

`worker_class` is an operator-selected execution/topology category.

Typical values may include:

- `modern-single`
- `multi-gpu`
- `legacy-single`
- `cpu`

These names are not application roles.

Avoid classes such as:

- `tts`
- `jev`
- `image`
- `chat`

Those belong in capabilities.

The class namespace is intentionally not a hard-coded enum in the domain core.

## 3. Capabilities

Capabilities answer:

> What kinds of jobs can this worker's configured executor perform?

Examples:

- `llm.chat`
- `code.review`
- `decision.system_one`
- `speech.tts`
- `image.generate`

Capabilities are opaque strings to the scheduler. Matching is generic set inclusion; the control plane does not contain workload-specific branches.

Sites and executor packages may define additional capability names.

## 4. Labels

Labels carry exact key/value traits useful for scheduling or policy.

Examples:

- `runtime_family=modern`
- `accelerator_generation=ada`
- `availability=borrowable`

Labels must not duplicate resource quantities that belong in the resource shape.

## 5. GPU resource shape

v0.1 distinguishes:

- `gpu_count`
- `total_vram_mb`
- `max_single_gpu_vram_mb`

### Why both VRAM values exist

These workers have equal total VRAM but are not interchangeable:

```text
Worker A
2 x 12 GiB
total_vram_mb          = 24576
max_single_gpu_vram_mb = 12288

Worker B
1 x 24 GiB
total_vram_mb          = 24576
max_single_gpu_vram_mb = 24576
```

A job that can shard across devices may request only total capacity.

A job that needs one contiguous 20 GiB device must request:

```text
min_single_gpu_vram_mb >= 20480
```

Worker A must not match that job.

## 6. GPU identity

For v0.1 NVIDIA GPU workers, the number of configured GPU UUIDs must equal `gpu_count`.

GPU UUIDs identify devices. CUDA ordinals and `/dev/nvidiaN` names do not.

A job may optionally require a particular UUID, but ordinary portable scheduling should prefer resource/capability constraints over hard-pinning a device.

## 7. Per-device accelerator facts

Aggregate resource shape is not sufficient to validate every runtime topology.

`WorkerSpec.accelerators` may therefore preserve one ordered
`AcceleratorDevice` record per owned GPU:

- UUID
- VRAM capacity
- compute capability when known, canonicalized as numeric `major.minor`
- device class/model identity when known

When present, the accelerator tuple must exactly match `gpu_uuids` order and
the aggregate VRAM totals in `ResourceShape`. Compute capability evidence is
validated at `AcceleratorDevice` construction; malformed strings are rejected
instead of becoming compatibility evidence.

These facts are durable Worker state, not TUI-only metadata. Runtime providers
may require them before making claims such as "homogeneous multi-GPU".

In particular, vLLM v0.x multi-GPU compatibility fails closed when per-device
facts are absent or when VRAM, compute capability, or device class differs
across the selected devices.

## 8. Job requirements

The initial generic requirements are:

- optional `worker_class`
- `required_capabilities`
- `required_labels`
- optional `required_gpu_uuids`
- `min_gpu_count`
- `min_total_vram_mb`
- `min_single_gpu_vram_mb`

All requirements are conjunctive.

The matcher returns both a boolean result and generic mismatch reasons for diagnostics.

## 9. Separation from application semantics

The following is intentionally invalid scheduler design:

```python
if job.kind == "tts":
    choose(worker_a)
elif job.kind == "image":
    choose(worker_b)
```

The intended design is:

```text
job requirements
      +
worker advertised capabilities/resources
      ↓
generic matcher
```

This keeps the control plane usable for workload types that did not exist when AstrumWeaver was first implemented.
