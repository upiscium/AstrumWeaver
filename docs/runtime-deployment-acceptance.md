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
- when the host has extra GPUs, setup discovers the selected UUID/minor mapping and enables the systemd device-cgroup isolation drop-in
- if the mapping cannot be proven, setup fails closed

For staged `--root` installs there is no live GPU discovery. To stage an
isolated subset, pass one reviewed `--gpu-device` mapping per selected UUID.

Example:

```sh
sudo astrumweaver-setup-gpu-worker \
  --config ./worker.toml \
  --environment-file ./worker.env \
  --runtime-manifest ./runtime-deployment.json \
  --gpu-uuid GPU-REDACTED-A \
  --gpu-isolation auto
```

The real UUID remains local operator input and must not be copied into public evidence.

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

```sh
sudo astrumweaver-runtime-deployment-accept \
  --gpu-uuid GPU-REDACTED-A \
  --revision <reviewed-git-sha> \
  --deployment-path systemd
```

For a NixOS deployment use `--deployment-path nixos`.

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
