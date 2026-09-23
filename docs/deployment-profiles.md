# Deployment Profiles and v0.1 Acceptance

AstrumWeaver does not provision infrastructure. These profiles describe useful starting points for an already-provisioned Linux node; they are not hidden scheduler requirements.

The canonical machine-readable examples live under `profiles/v0.1/`. The v0.1 acceptance manifest lives at `acceptance/v0.1.toml`.

## Terminology

### Minimum

**Minimum** means the smallest resource envelope this project is prepared to document as a practical starting point for that role.

It is operational guidance, not an automatic admission gate unless the same property is also represented by an explicit AstrumWeaver contract such as:

- `gpu_count`
- `total_vram_mb`
- `max_single_gpu_vram_mb`
- required capabilities
- required labels
- exact accelerator identity

A host below a CPU/RAM/disk recommendation is not rejected merely because it is below the profile.

### Recommended

**Recommended** means a default deployment target with enough headroom for the role itself and ordinary maintenance/observability overhead.

Recommendations are deliberately conservative and workload-neutral. Executor-specific models, artifacts, caches, and datasets may require substantially more memory or storage.

### Validated

**Validated** means a specific shape or deployment property has evidence attached to it.

v0.1 distinguishes validation scope:

- `ci-contract`: the AstrumWeaver software contract/behavior is exercised in public CI using synthetic/example identities and resources.
- `hardware-e2e`: the complete profile has been exercised on physical/virtual hardware and the evidence is intentionally publishable.

A `ci-contract` validation is **not** a performance benchmark, CUDA compatibility certification, or proof that every executor will run on that hardware shape.

The public v0.1 profiles currently claim CI-contract validation only. Real private site inventory is intentionally not published.

## General sizing rule

CPU, RAM, and disk are host recommendations.

GPU topology/capacity is scheduler-visible only when declared by the Worker and can be constrained by a job.

For example:

```text
recommended host RAM = 8 GiB
```

does not cause Control to reject a 6 GiB host.

By contrast:

```text
min_single_gpu_vram_mb = 20480
```

is an explicit job resource constraint and is enforced by the scheduler.

## Control Plane profile

Role: durable scheduling/API plane. No GPU required.

### Minimum

- 2 vCPU
- 2 GiB RAM
- 10 GiB local persistent disk for service/runtime state
- reachable PostgreSQL
- stable network connectivity to participating Workers

The PostgreSQL data volume is capacity-planned separately from the Control host disk when the database is remote.

### Recommended

- 4 vCPU
- 4 GiB RAM
- 20 GiB local persistent disk
- PostgreSQL on durable storage with backup/monitoring appropriate to the deployment
- TLS or another protected transport boundary when traffic leaves a trusted network

These values are intended for small v0.1 deployments, not as a scale ceiling.

### Validated

Public CI validates:

- Python 3.12 runtime
- PostgreSQL 17 service-container integration
- migration application
- job durability/fencing/recovery
- Control/Worker HTTP transport
- Nix package/module evaluation

The CI runner CPU/RAM allocation is not treated as a validated sizing claim.

## `modern-single` profile

Role: one current/modern NVIDIA GPU owned exclusively by one Worker.

There is no universal minimum VRAM for the class; executor capability and job requirements determine whether a workload fits.

### Minimum

- 2 vCPU
- 4 GiB system RAM
- 10 GiB local disk plus executor/model/artifact needs
- exactly one declared GPU UUID
- a functional guest-visible `nvidia-smi`
- driver/runtime compatibility required by the configured executor

### Recommended

- 4+ vCPU
- 8+ GiB system RAM
- local cache capacity at least as large as the largest expected resident artifact/model, plus operational headroom
- current driver/runtime stack supported by the chosen executor
- explicit capability advertisement rather than assuming all modern GPUs support every workload

### Validated example shape

CI-contract example:

```text
gpu_count               = 1
total_vram_mb            = 16384
max_single_gpu_vram_mb   = 16384
```

This is an illustrative resource shape used for configuration/module validation, not a claim that 16 GiB is universally required or sufficient.

## `multi-gpu-large` profile

Role: one Worker whose executor owns multiple GPUs as one scheduler-visible compute unit.

### Minimum

- 4 vCPU
- 8 GiB system RAM
- 20 GiB local disk plus executor/model/artifact needs
- at least two declared GPU UUIDs
- executor/runtime that explicitly supports the intended multi-GPU strategy
- all advertised GPUs exclusively owned by that Worker while ONLINE/DRAINING

### Recommended

- 8+ vCPU
- 16+ GiB system RAM
- local cache sized for the largest expected multi-device workload
- homogeneous GPUs when the executor depends on symmetric sharding; heterogeneous devices are allowed only when the executor supports them
- workload requirements expressed with both total and per-device constraints where needed

### Validated example shape

CI-contract example:

```text
gpu_count               = 2
total_vram_mb            = 24576
max_single_gpu_vram_mb   = 12288
```

The acceptance suite specifically proves that this 2 × 12 GiB shape does **not** satisfy a job requiring 20 GiB on one device.

## `legacy-single` profile

Role: one older NVIDIA GPU used only for explicitly compatible workloads.

This profile exists to describe compatibility constraints, not to pretend legacy hardware is equivalent to a modern CUDA/runtime environment.

### Minimum

- 2 vCPU
- 4 GiB system RAM
- 10 GiB local disk plus workload assets
- exactly one declared GPU UUID
- a driver/userspace combination that still supports the device
- an executor that explicitly advertises only capabilities known to work on that runtime

### Recommended

- 4+ vCPU
- 8+ GiB system RAM
- pin the driver/runtime/executor versions known to work together
- keep workload capability labels narrow
- avoid assigning jobs solely because total VRAM appears sufficient

### Validated example shape

CI-contract example:

```text
gpu_count               = 1
total_vram_mb            = 8192
max_single_gpu_vram_mb   = 8192
```

Only generic scheduler semantics are validated for this example. No claim is made that a particular CUDA, inference, image, or media runtime supports every 8 GiB legacy GPU.

## Profile comparison

| Profile | Minimum host starting point | GPU shape | Scheduler meaning |
| --- | --- | --- | --- |
| Control Plane | 2 vCPU / 2 GiB RAM / 10 GiB disk | none | durable scheduling/API |
| `modern-single` | 2 vCPU / 4 GiB RAM / 10 GiB + assets | 1 GPU | one exclusive modern single-device Worker |
| `multi-gpu-large` | 4 vCPU / 8 GiB RAM / 20 GiB + assets | 2+ GPUs | one exclusive multi-device Worker |
| `legacy-single` | 2 vCPU / 4 GiB RAM / 10 GiB + assets | 1 GPU | one compatibility-limited Worker |

The table is deployment guidance, not admission logic.

## v0.1 acceptance matrix

The canonical matrix is `acceptance/v0.1.toml`; this table is its human-readable view.

| Requirement | Evidence | Scope |
| --- | --- | --- |
| single-GPU Worker registration | Control repository tests | CI contract |
| multi-GPU Worker registration | multi-GPU resource/claim tests | CI contract |
| exact GPU UUID preflight | setup/preflight tests | CI contract |
| 2 × 12 GiB does not satisfy 20 GiB single-device requirement | scheduler resource-shape test | CI contract |
| capability matching | generic scheduler capability test | CI contract |
| DRAINING/OFFLINE lifecycle | Control + borrowable lifecycle tests | CI contract |
| existing-node install | staged setup tests | CI contract |
| NixOS module deployment | `checks.x86_64-linux.module-eval` | CI contract |
| versioned Control/Worker transport | transport tests | CI contract |
| borrowable GPU ownership handoff | borrowable lifecycle tests | CI contract |

## v0.1 acceptance boundary

Passing the software acceptance matrix means the v0.1 contracts and supported deployment paths are internally consistent.

It does **not** mean:

- every NVIDIA generation is supported
- every executor supports every GPU
- a recommendation is a hard minimum
- AstrumWeaver has provisioned or certified the VM/LXC
- a private Home Lab topology has been disclosed or tested by public CI
- throughput/latency targets have been benchmarked

A site may record its own hardware-e2e validation without changing the generic public profiles.

## Infrastructure boundary

Every profile assumes the node already exists.

AstrumWeaver does not own:

- VM/LXC creation
- Proxmox VMID/CTID allocation
- bridge/VLAN configuration
- storage-pool creation
- IOMMU/VFIO/passthrough setup
- operating-system installation
- host NVIDIA driver installation

A Proxmox VM/LXC, another hypervisor guest, cloud VM, or bare-metal Linux node is acceptable when it satisfies the same Host Prerequisite Contract.
