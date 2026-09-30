# Runtime Deployment GPU Isolation Acceptance

Issue #31 requires more than recording a selected GPU UUID set.

A valid subset deployment must prove that a Worker started on a host with
additional GPUs can access exactly its configured physical GPUs, while the
existing exact-set startup gate remains meaningful.

## Isolation contract

For a GPU subset Worker, AstrumWeaver uses two independent checks:

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
existing gpu-preflight / require_exact_gpu_set()
        ↓
ManagedRuntime start/readiness
        ↓
Worker registration
```

The host-level mapping verifier runs outside the restricted Worker cgroup.
The existing exact-set preflight then runs inside the restricted Worker
service cgroup.

This separation is intentional. AstrumWeaver does not weaken
`require_exact_gpu_set()` into accepting a host superset.

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
the Worker service and applies `DevicePolicy=closed` to the Worker cgroup.

The `deviceMap` keys must exactly match `gpuUuids`.

## Real-host acceptance

Run the acceptance only on an idle Worker service. The command intentionally
requires the service to be inactive first, starts it, waits for local
ready/registered state, and stops it again.

The host must expose more GPUs than the selected Worker set.
The checker reads the effective Worker properties from `systemctl show`, so
later drop-ins that override the isolation settings are included in the
acceptance decision.

```sh
sudo astrumweaver-runtime-deployment-accept \
  --gpu-uuid GPU-REDACTED-A \
  --revision <reviewed-git-sha> \
  --deployment-path systemd
```

For a NixOS deployment use `--deployment-path nixos`.

The acceptance command invokes the packaged `astrumweaver-gpu-device-map`
helper, which uses `nvidia-smi` for current visibility and the
`uuid,minor_number` query for the primary UUID-to-minor mapping. It falls back
to `/proc` only when `nvidia-smi` explicitly reports that the field is
unsupported. Before using a mapping, it verifies that each `/dev/nvidiaN`
character device's major/minor matches the NVIDIA character-device registry
and reported minor. If the helper is not on
`/usr/local/libexec/astrumweaver/gpu-device-map` or beside the invoked
packaged acceptance command, pass its absolute executable path with
`--gpu-device-map`.

The default evidence path is:

```text
validation/runtime-deployment/gpu-subset-e2e.md
```

## Public evidence boundary

The generated evidence contains only:

- date
- public AstrumWeaver revision
- deployment path
- host-visible GPU count
- selected GPU count
- PASS/FAIL contract gates

It intentionally cannot contain:

- hostname
- IP address
- Worker ID
- GPU UUID
- `/dev/nvidiaN` ordinal
- GPU model
- Control URL
- credentials

A PASS proves:

- the host really had a GPU superset
- the Worker configuration matched the selected set
- exact-set preflight was enabled
- UUID/device mapping was valid
- `DevicePolicy=closed` was active
- the only physical `DeviceAllow` entries were the selected GPUs
- CUDA visible-device order matched Worker order
- the in-service exact-set gate remained present
- the isolated Worker reached local ready + registered
- the service stopped cleanly after acceptance

## Current v0.x boundary

This contract targets whole physical NVIDIA GPUs.

MIG device-instance isolation and NVIDIA capability-node policy are not claimed
by this acceptance and require a separate compatibility contract.
