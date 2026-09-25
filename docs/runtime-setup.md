# Runtime Setup Backend

AstrumWeaver uses one deterministic setup backend for both automation and the interactive TUI.

Concrete RuntimeProviders do **not** execute package-manager, filesystem, service-manager, or model-download commands directly.

The flow is:

```text
RuntimeProvider
    ↓ setup_intent()
RuntimeSetupIntent
    ↓
build_runtime_setup_plan()
    ↓
SetupPlan
    ↓
preview / dry-run
    ↓
human or automation reviews exact digest
    ↓
SetupApproval
    ↓
apply_plan()
    ↓
SetupActionDriver
    ↓
deployment-specific mutation
```

The deployment-specific driver is intentionally deferred to #31. NixOS and generic systemd Linux will implement the same SetupAction contract through different drivers.

## Canonical namespace

The setup backend is exported from:

```python
astrumweaver.setup
```

Runtime selection/provider contracts remain under:

```python
astrumweaver.runtime
```

This separation is intentional:

- `astrumweaver.runtime`: what runtime is compatible and how it behaves
- `astrumweaver.setup`: how reviewed runtime intent becomes host mutations

## Host discovery

`discover_local_host()` returns a `SetupHostSnapshot` containing only generic local facts required for setup planning:

- deployment path: NixOS or generic systemd
- OS ID/version
- CPU count
- host RAM
- architecture
- available relevant commands
- package manager
- service manager
- privilege mode

Discovery deliberately does not inspect:

- hostname
- IP addresses
- default routes
- hypervisor inventory
- VMID/CTID
- bridge/VLAN names
- private site topology

GPU identity remains part of the existing Worker/GPU preflight contract.

`discover_local_gpus()` provides the interactive and non-interactive setup
frontends with exact NVIDIA GPU UUIDs, VRAM capacity, and best-effort compute
capability. These facts remain separate from the public-safe
`SetupHostSnapshot`.

## SetupPlan

A SetupPlan is immutable and serializable.

It contains:

- schema version
- selected provider ID
- deployment path
- activation/release goal
- ordered SetupActions
- non-secret planning metadata

The plan digest is SHA-256 over canonical JSON.

Generating the same plan from identical inputs produces the same digest.

This digest is the review boundary used by `SetupApproval`.

## SetupAction

Initial action kinds include:

- ensure directory
- ensure package
- render configuration
- verify model reference
- download model
- convert model
- runtime preflight
- runtime start
- health check
- runtime stop
- runtime release

Every action explicitly states whether it:

- requires privilege
- requires network access
- requires explicit confirmation
- is reversible

The TUI can therefore show mutation/authority scope before anything changes.

## Secret references

SetupPlan may carry `SecretReference` values.

Example:

```python
SecretReference("HF_TOKEN")
```

Serialization contains only:

```json
{"$secret_ref": "HF_TOKEN"}
```

The actual secret value must remain in protected local secret/environment storage and must never enter SetupPlan, plan digest, logs, or public evidence.

Concrete providers should use references rather than embedding credential values in provider configuration.

## Model preparation

`RuntimeSetupIntent.model_preparation` maps to distinct reviewed actions:

### reference_only

The model is already available or externally managed.

The plan verifies the reference but does not transfer model data.

### download

The plan contains a network + confirmation-gated download action.

Applying it requires explicit approval for:

- network access
- confirmation-gated actions
- model download

### convert

The plan contains a confirmation-gated conversion action.

No download or conversion happens merely because a RuntimeProvider is compatible.

## Dry-run / preview

`dry_run_plan()` / `preview_plan()` calls `SetupActionDriver.inspect()` for every action and returns:

- SATISFIED
- NEEDS_APPLY
- BLOCKED

No mutation occurs.

The preview exposes the same privilege/network/confirmation flags as the reviewed plan.

## Exact approval

Applying a plan requires `SetupApproval` bound to the exact SetupPlan digest.

The approval independently controls:

- privileged actions
- network actions
- explicit-confirmation actions
- model download
- model conversion

A stale approval for another plan digest fails before the first mutation.

This is the same contract used by the interactive TUI.

## Apply and idempotency

Before each mutation, the apply engine repeats `driver.inspect(action)`.

If already satisfied, the action is skipped.

This makes repeated application convergent when a deployment driver correctly implements inspect/apply semantics.

The apply result contains structured per-action:

- status
- changed flag
- public-safe detail
- structured evidence

`apply_plan(..., on_result=...)` can stream each structured action/rollback
result to an interactive frontend without moving mutation logic out of the
shared backend.

Deployment drivers must keep secrets out of evidence.

## Rollback

Actions declare whether they are reversible.

If a later action fails and rollback is enabled, the engine walks previously changed reversible actions in reverse order.

Rollback is best-effort and its own result is recorded:

- ROLLED_BACK
- ROLLBACK_FAILED

Non-reversible actions such as package/model acquisition are not falsely claimed to have been undone.

## Deployment drivers

`SetupActionDriver` is the mutation boundary:

```python
inspect(action) -> ActionInspection
apply(action) -> ActionReceipt
rollback(action, receipt) -> ActionReceipt
```

#31 will provide real drivers for:

- NixOS
- generic systemd Linux

The shared planner/apply engine itself does not invoke apt/dnf/pacman/nix/systemctl directly.

This keeps the TUI, CLI automation, tests, and future remote setup frontends on one deterministic backend.

## Shared ManagedRuntime lifecycle

Provider implementations also share `RuntimeLifecycleManager`.

It provides idempotent coordination around:

- ensure ready
- ensure stopped
- optional release
- restart

The provider still owns the concrete start/stop/health/release implementation.

The lifecycle helper does not move durable job ownership out of the Worker/Control layer.

## TUI relationship

The interactive TUI is implemented by
[`astrumweaver-setup-tui`](setup-tui.md).

It performs host/GPU/Worker discovery, execution-demand editing, compatibility
explanation, explicit runtime selection, provider-option editing, exact
SetupPlan review, dry-run, digest-bound approval and structured apply progress.

Without a deployment driver it stops in planning-only mode. #31 supplies the
NixOS/systemd mutation driver and Worker service integration.

The TUI does not execute ad-hoc provider shell commands outside this backend.

## Infrastructure boundary

The setup backend still begins after the Host Prerequisite Contract.

It must not plan or apply:

- VM/LXC creation
- Proxmox mutation
- IOMMU/VFIO setup
- PCI passthrough
- NVIDIA host-driver installation/replacement
- arbitrary model download without explicit approval
