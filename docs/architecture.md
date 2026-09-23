# AstrumWeaver Architecture

## 1. Purpose

AstrumWeaver is a deployment-agnostic compute fabric. It schedules capability-based jobs across pre-existing heterogeneous compute nodes while keeping resource identity, job ownership, and failure recovery explicit.

The project begins GPU-first, but the core contracts must remain usable for future accelerator or CPU-backed workers.

## 2. Trust and responsibility boundary

AstrumWeaver begins **inside an already-provisioned Linux node**.

The infrastructure layer is responsible for creating that node and making the required resources visible to it.

```text
┌────────────────────────────────────────────┐
│ Infrastructure layer                      │
│                                            │
│ Proxmox / other hypervisor / bare metal   │
│ OS install                                │
│ networking                                │
│ storage                                   │
│ IOMMU / PCI passthrough / device mapping  │
│ host GPU driver where applicable          │
└─────────────────────┬──────────────────────┘
                      │
                      │ node satisfies host contract
                      ▼
┌────────────────────────────────────────────┐
│ AstrumWeaver                               │
│                                            │
│ service install/configuration              │
│ worker registration                       │
│ exact resource identity verification       │
│ capability/resource scheduling             │
│ job lifecycle and lease fencing            │
│ executor lifecycle                         │
└────────────────────────────────────────────┘
```

AstrumWeaver must not need credentials for the hypervisor merely to operate a worker.

## 3. Core logical roles

### Control Plane

The Control Plane owns durable fabric state and scheduling policy.

Expected responsibilities include:

- durable job state
- worker registration and heartbeat state
- worker drain/offline lifecycle
- capability and resource matching
- atomic job claims
- lease/fencing tokens
- retry/recovery decisions
- artifact/result references

The Control Plane does not need a GPU.

### Worker

A Worker is the smallest **scheduler-visible exclusive compute unit**.

A worker is not synonymous with a physical GPU.

Valid examples include:

- one single-GPU runtime
- one runtime that exclusively owns two GPUs
- a future CPU-only executor
- a future accelerator type represented by a stable resource identity

A worker advertises:

- topology/class
- capabilities
- labels
- resource shape
- stable accelerator identities, when present
- health/readiness
- active/resident runtime information where useful

### Executor

An Executor performs a job on behalf of a Worker.

Executors may implement, for example:

- text/LLM inference
- code review
- System-One/decision inference
- TTS
- image generation

The Control Plane schedules by generic capabilities and resource constraints. It must not contain workload-specific branches such as "if TTS, choose worker X".

### Runtime Provider

A Runtime Provider manages a concrete local model-serving runtime such as Ollama, llama.cpp, vLLM, FreeToken, or ExLlamaV3.

It is Worker-local and sits above the generic JobExecutor boundary.

Runtime selection is an operator decision. AstrumWeaver may report compatibility and recommendations, but an explicit provider choice must never be silently substituted.

Runtime Providers own runtime package/process/model preparation and expose a generic JobExecutor to the Worker. Control remains runtime-agnostic.

See [Runtime Providers and Execution Demand](runtime-providers.md).

### Optional edge/artifact services

Gateway, artifact, dataset, or capture services may exist where the use case requires them, but they are not part of the virtualization/provisioning boundary.

## 4. Deployment model

AstrumWeaver supports deployment onto a node that already exists.

The deployment mechanism may differ by host:

- NixOS module
- Nix package plus systemd integration
- systemd-oriented setup script on another supported Linux distribution

The deployment path may install AstrumWeaver software and service configuration. It must not create the VM/LXC/host itself.

## 5. Proxmox compatibility model

Proxmox is treated as one possible infrastructure provider.

### VM

A VM may participate when the required GPU/device has already been passed through and the guest satisfies the same worker host contract as any other Linux node.

AstrumWeaver does not configure:

- IOMMU
- VFIO binding
- `hostpci` entries
- guest firmware/machine type
- Proxmox networking

### LXC

An LXC may participate when the host/operator has already exposed the required device nodes and userspace/driver compatibility is correct.

AstrumWeaver does not configure:

- Proxmox cgroup/device passthrough
- LXC mount entries
- host NVIDIA driver installation
- container privilege model

From AstrumWeaver's perspective, the acceptance criterion is the guest-visible host contract, not how Proxmox achieved it.

### No Proxmox identity in the scheduling model

VMIDs, CTIDs, Proxmox node names, bridges, and storage IDs are infrastructure inventory and must not become scheduler identity.

Worker identity and accelerator identity must remain portable if the same Linux workload is recreated elsewhere.

## 6. Resource identity

Accelerators must be identified by stable vendor/device identities where available.

For NVIDIA GPUs, AstrumWeaver uses the GPU UUID reported by the guest-visible NVIDIA stack. Device ordinals such as `/dev/nvidia0` are not stable identity.

A pinned worker must fail closed if its observed accelerator identity differs from its configured identity.

## 7. Public repository hygiene

The public repository must contain only generic examples.

Do not commit:

- production IP addresses
- internal DNS names
- actual GPU UUIDs
- real VM/LXC IDs
- private certificates or tokens
- site topology
- private model credentials

Documentation should use placeholders or standards-reserved example values.

## 8. Non-goals

AstrumWeaver v0.1 is not:

- a Proxmox provisioner
- a Kubernetes replacement
- an infrastructure-as-code system for hosts/networks/storage
- a GPU virtualization or partitioning layer
- an automatic PCI passthrough manager
- a model-specific orchestration framework
- an LLM-only platform

Keeping these responsibilities outside the core prevents the compute fabric from becoming coupled to one site or one application family.
