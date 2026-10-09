# Runtime Deployment GPU Isolation Acceptance

Issue #31 requires more than recording a selected GPU UUID set.

A valid subset deployment must prove one of two outcomes:

- the Worker can access exactly its configured physical GPUs and may start, or
- the environment cannot enforce that subset and TSUMGI fails closed
  before starting the Worker.

A host-visible GPU superset is never accepted from configuration text alone.

## Isolation contract

For a GPU subset Worker, TSUMGI uses two independent checks:

```text
host-visible GPU superset
        ↓
UUID → /dev/nvidiaN mapping verification
        ↓
systemd DevicePolicy=closed
+ exact selected physical DeviceAllow entries
+ required shared NVIDIA control/UVM nodes
        ↓
Worker service cgroup
        ↓
explicit isolated-access gpu-preflight
(shell ExecStartPre + Python daemon)
        ↓
selected physical nodes open
unselected physical nodes denied
        ↓
ManagedRuntime start/readiness
        ↓
Worker registration
```

The host-level mapping verifier runs outside the restricted Worker cgroup.
For an isolated subset, both the service `ExecStartPre` helper and the Python
Worker daemon run the same isolated-access semantics inside the final Worker
service context.

For a non-isolated Worker, the original exact-visible rule remains unchanged:
raw NVIDIA-visible UUIDs must exactly equal the Worker contract. TSUMGI
does not reinterpret a raw superset as safe unless the explicit isolated-access
contract is present and actual device access proves the boundary.

`CUDA_VISIBLE_DEVICES` is also set to the selected UUID sequence so CUDA
enumeration preserves Worker GPU order, but it is not treated as the security
boundary. The systemd device allow-list is the device-access boundary.

## Generic systemd Linux

`setup-gpu-worker.sh` supports:

```text
--gpu-isolation auto|on|off
--gpu-device UUID=/dev/nvidiaN
```

The default is `auto`.

On a live host:

- when the host-visible set already equals the Worker set, ordinary exact-set preflight remains sufficient
- when the host has extra GPUs, setup discovers the selected UUID/minor mapping but does **not** trust `DevicePolicy=closed` or `DeviceAllow=` configuration text by itself
- before any persistent subset-isolation map, unit, or drop-in is written, setup runs a transient service with the proposed device policy and verifies that selected physical GPU nodes can be opened while every unselected physical GPU node is denied
- if an unselected GPU remains openable (for example inside an LXC where the parent did not delegate an effective device-cgroup boundary), setup fails closed and tells the operator to narrow guest-visible GPU exposure at the VM/container/hypervisor boundary
- if the mapping or enforcement capability cannot be proven, setup fails closed
- if the Worker is already active, stop it before applying GPU isolation; setup refuses to claim an isolation change on a running process

A successful transient capability probe is only authorization to materialize the
subset policy. Worker startup still needs an in-service ownership preflight;
configuration text alone is never accepted as proof of isolation.

For staged `--root` installs there is no live GPU discovery. To stage an
isolated subset, pass one reviewed `--gpu-device` mapping per selected UUID.

Example:

```sh
sudo astrumweaver-setup-gpu-worker \
  --config ./worker.toml \
  --environment-file ./worker.env \
  --runtime-manifest ./runtime-deployment.json \
  --gpu-isolation auto
```

The selected UUID remains only in the local `worker.toml` Worker contract and must not be copied into public evidence.

## NixOS

The Worker module provides an explicit declarative mapping:

```nix
services.astrumweaver.worker = {
  gpuUuids = [ "GPU-REDACTED-A" ];

  gpuIsolation = {
    enable = true;
    deviceMap."GPU-REDACTED-A" = "/dev/nvidia0";

    auxiliaryDeviceNodes = [
      "/dev/nvidiactl"
      "/dev/nvidia-modeset"
      "/dev/nvidia-uvm"
      "/dev/nvidia-uvm-tools"
    ];
  };
};
```

The NixOS module creates a separate host-level mapping preflight service before
the Worker service, applies `DevicePolicy=closed`, and explicitly selects
`isolated-access` for the Worker startup preflight. The immutable Nix-store
Worker config, reviewed map, and canonical mapper paths are passed into both
the shell preflight and Python daemon.

The `deviceMap` keys must exactly match `gpuUuids`. Runtime startup still
fails if the effective cgroup does not deny access to an unselected physical
GPU.

## Real-host acceptance

The acceptance schema is `runtime-deployment-v2` and supports two successful
outcomes.

### Enforced subset

Run this only after a subset deployment has been materialized and while the
Worker is inactive:

```sh
sudo astrumweaver-runtime-deployment-accept \
  --gpu-uuid GPU-REDACTED-A \
  --revision <reviewed-git-sha> \
  --deployment-path systemd
```

The checker validates the effective systemd policy, requires the explicit
`isolated-access` environment contract, starts the Worker, waits for local
ready+registered state, and stops it again.

A successful evidence record contains:

```text
Outcome = ENFORCED_SUBSET
Isolation enforcement = PASS
Worker start attempted = YES
Overall = PASS
```

### Expected fail closed

On an environment where the subset cannot be enforced, use a temporary Worker
config describing only the proposed subset and keep the authoritative Worker
service inactive:

```sh
sudo astrumweaver-runtime-deployment-accept \
  --gpu-uuid GPU-REDACTED-A \
  --revision <reviewed-git-sha> \
  --deployment-path systemd \
  --worker-config /root/private-worker-subset.toml \
  --expect-isolation-unavailable
```

This mode runs the reviewed transient isolation probe only. It does not start
the Worker. It succeeds only when the probe classifies the environment as
unable to deny an unselected physical GPU.

A successful evidence record contains:

```text
Outcome = FAIL_CLOSED
Isolation enforcement = UNAVAILABLE
Fail closed = PASS
Worker start attempted = NO
Overall = PASS
```

This is the expected result for an LXC/container where the parent boundary does
not delegate effective device-cgroup enforcement. The operator must narrow
guest-visible GPU exposure at the VM/LXC/hypervisor layer before using a smaller
Worker GPU contract.

For a NixOS deployment use `--deployment-path nixos`.

The acceptance command resolves reviewed packaged
`astrumweaver-gpu-device-map` and `astrumweaver-gpu-isolation-probe`
helpers without trusting ambient PATH. Explicit helper overrides must be
absolute executable paths.

The default evidence path is:

```text
validation/runtime-deployment/gpu-subset-e2e.md
```

## Public evidence boundary

The generated evidence contains only:

- date
- public TSUMGI revision
- deployment path
- outcome (`ENFORCED_SUBSET` or `FAIL_CLOSED`)
- host-visible GPU count
- selected GPU count
- public-safe PASS / UNAVAILABLE / NOT_APPLICABLE contract gates

It intentionally cannot contain:

- hostname
- IP address
- Worker ID
- GPU UUID
- `/dev/nvidiaN` ordinal
- GPU model
- Control URL
- credentials

An `overall=PASS` proves that the requested acceptance outcome was
satisfied:

- `ENFORCED_SUBSET`: actual in-service device access was enforced and the
  Worker reached ready+registered state.
- `FAIL_CLOSED`: enforcement was unavailable, the Worker was never started,
  and TSUMGI correctly required an external visibility boundary.

## Current v0.x boundary

This contract targets whole physical NVIDIA GPUs.

MIG device-instance isolation and NVIDIA capability-node policy are not claimed
by this acceptance and require a separate compatibility contract.
