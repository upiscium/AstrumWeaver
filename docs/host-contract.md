# Host Prerequisite Contract

This document defines what an existing node must provide before AstrumWeaver may install or start a service on it.

The contract is intentionally virtualization-agnostic. A compliant node may be a Proxmox VM, a Proxmox LXC, another Linux VM, a cloud instance, or bare metal.

## 1. Contract levels

- **MUST**: absence is a correctness or safety blocker.
- **SHOULD**: strongly recommended, but not a universal startup blocker.
- **MAY**: optional integration.

Recommended sizing is not a hidden hard requirement. Resource-specific jobs are rejected through explicit job/worker constraints instead.

## 2. Common node requirements

A node running an AstrumWeaver service MUST provide:

- a supported Linux userspace
- a functioning service manager for the selected deployment path; v0.1 systemd integrations require systemd
- reliable local filesystem semantics for service state
- network connectivity to every AstrumWeaver service endpoint required by that role
- reasonably synchronized system time
- protected delivery of credentials/secrets outside the public repository
- a stable hostname or operator-selected AstrumWeaver service identity; it need not match hypervisor inventory

A node SHOULD provide:

- persistent local state storage appropriate to its role
- log persistence/collection
- sufficient free disk for package/runtime updates
- DNS resolution where configured endpoints use DNS names

AstrumWeaver MUST NOT infer site configuration from:

- default routes
- interface names
- Proxmox bridge names
- VM/LXC numeric IDs
- storage pool names

## 3. Control Plane requirements

A Control Plane node MUST provide:

- no GPU requirement
- connectivity to its durable state backend
- persistent state for any local control-plane data
- an identity/credential configuration accepted by AstrumWeaver

Where PostgreSQL is the selected durable backend, the database may be co-located or remote. AstrumWeaver must not require the Control Plane setup path to create the VM/container hosting that database.

A Control Plane SHOULD have enough CPU/RAM to handle scheduler/API concurrency without swapping under normal load. Concrete minimum/recommended/validated profiles belong in deployment-profile documentation rather than this safety contract.

## 4. Worker requirements

Every Worker MUST provide:

- connectivity to the Control Plane
- a configured worker identity
- an explicit capability set
- an explicit resource shape
- exclusive ownership of the resources it advertises for the duration of its ONLINE/ready state
- a reliable way to stop/drain the executor before those resources are reassigned

A Worker MUST NOT advertise a resource that is concurrently owned by an unrelated workload unless a future explicit sharing/partitioning contract supports that mode.

## 5. NVIDIA GPU Worker requirements

An NVIDIA GPU Worker MUST provide, before registration:

- NVIDIA device(s) already visible inside the worker node
- a functional `nvidia-smi`
- stable GPU UUID discovery from the guest-visible NVIDIA stack
- explicit configured expected GPU UUID(s)
- exact set equality between expected and observed UUIDs for an exclusively pinned worker
- a driver/runtime combination compatible with the configured executor
- sufficient device access for the service account/runtime

The following are unsafe identity mechanisms and MUST NOT be used as canonical identity:

- CUDA ordinal alone
- `/dev/nvidia0`, `/dev/nvidia1`, ... alone
- PCI slot ordering alone

Those values may be recorded as diagnostics, but GPU UUID is the canonical NVIDIA accelerator identity in v0.1.

### Fail-closed behavior

If:

```text
expected UUID set != observed UUID set
```

the worker MUST refuse ONLINE/readiness and MUST NOT claim jobs.

An unexpected additional GPU is also a mismatch for an exclusive pinned worker.

## 6. Proxmox deployment notes

These notes are guidance only. They are not a dependency of AstrumWeaver.

### Proxmox VM

Before AstrumWeaver setup:

- the VM already exists
- required PCI passthrough is already configured by the operator/infrastructure layer
- the guest OS already boots normally
- the guest NVIDIA driver is already installed where required
- `nvidia-smi` inside the guest reports exactly the intended device set

AstrumWeaver does not inspect or mutate Proxmox `hostpci`, IOMMU, machine, firmware, bridge, or storage configuration.

### Proxmox LXC

Before AstrumWeaver setup:

- the container already exists
- the operator has already configured the required device passthrough/mounts
- host/container driver/userspace compatibility is already resolved
- `nvidia-smi` inside the container reports exactly the intended device set
- the AstrumWeaver service account can access the required GPU device nodes

AstrumWeaver does not change LXC privilege mode, device cgroup rules, mount entries, or the Proxmox host driver.

### Portability requirement

The same AstrumWeaver Worker configuration model should remain valid if an operator later recreates the node on another hypervisor, provided the worker resource identities and capabilities are intentionally revalidated.

## 7. Secret and configuration requirements

Real credentials MUST NOT be embedded in:

- repository-tracked configuration
- Nix source committed to the public repository
- example setup commands
- public host profiles

Deployment mechanisms should separate non-secret declarative configuration from protected secret material.

## 8. Enrollment sequence

A generic Worker enrollment flow is:

```text
existing Linux node
    ↓
host prerequisites verified
    ↓
AstrumWeaver package installed
    ↓
non-secret config + protected credentials installed
    ↓
accelerator identity preflight
    ↓
executor preflight
    ↓
worker registration
    ↓
ONLINE / ready
```

VM/LXC creation, OS installation, networking, PCI passthrough, and GPU driver setup all occur before this sequence and remain outside AstrumWeaver.

## 9. Out of scope for this contract

This contract does not prescribe:

- a specific Proxmox version
- a specific Linux distribution for all roles
- specific CPU/RAM sizing
- a specific NVIDIA model
- a specific model-serving runtime

Those choices are documented as validated/recommended deployment profiles and executor-specific compatibility constraints.
