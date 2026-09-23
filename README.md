# AstrumWeaver

AstrumWeaver is a deployment-agnostic compute fabric for scheduling capability-based workloads across heterogeneous compute nodes.

It is GPU-first, but its control-plane contracts are intentionally not tied to LLMs, a particular accelerator generation, Proxmox, or a one-GPU-per-worker topology.

> **Project status:** early v0.1 architecture/bootstrap. Interfaces may change until the first stable worker/control contract is accepted.

## Scope

AstrumWeaver owns the compute-fabric layer:

- durable job/control state
- worker registration, health, drain, and lifecycle state
- capability- and resource-aware scheduling
- job leases, fencing, retries, and recovery
- accelerator identity verification
- executor lifecycle
- deployment of AstrumWeaver services onto an existing Linux node
- job result/artifact metadata needed by the fabric

AstrumWeaver does **not** provision the infrastructure underneath a node. In particular, it does not create or configure:

- virtual machines or containers
- bare-metal operating systems
- Proxmox VM/LXC IDs
- bridges, VLANs, or site routing
- IOMMU or PCI passthrough
- host storage pools
- NVIDIA host drivers

A node enters AstrumWeaver only after the surrounding infrastructure already satisfies the [Host Prerequisite Contract](docs/host-contract.md).

## Architecture boundary

```text
Infrastructure / virtualization layer
(Proxmox, another hypervisor, bare metal, cloud, ...)
                    │
                    │ already-provisioned Linux node
                    ▼
        ┌────────────────────────┐
        │      AstrumWeaver      │
        │                        │
        │  Control ─── Workers   │
        │             │          │
        │             └ Executors│
        └────────────────────────┘
```

Proxmox is a supported deployment environment, not an AstrumWeaver dependency. A Proxmox VM, a Proxmox LXC, a bare-metal Linux host, or another Linux VM may all become workers when they satisfy the same host contract.

## Design principles

- **Worker != GPU.** A worker is the smallest scheduler-visible exclusive compute unit.
- One worker may own one or several accelerator UUIDs.
- Hardware topology and workload capability are separate concepts.
- LLM, System-One/decision, TTS, image generation, and future workload types are executor capabilities, not control-plane special cases.
- Public configuration examples contain placeholders only; real site inventory, addresses, credentials, and GPU UUIDs stay outside the repository.
- Existing development machines may later participate as drainable/borrowable workers without reprovisioning.

## Documentation

- [Architecture](docs/architecture.md)
- [Host Prerequisite Contract](docs/host-contract.md)
- [Worker and Resource Contract](docs/worker-contract.md)
- [Job Execution Contract](docs/executor-contract.md)
- [Durable Control Plane](docs/control-plane.md)

## License

See [LICENSE](LICENSE).
