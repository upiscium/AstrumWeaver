# Runtime Deployment GPU Isolation Acceptance

Issue #31 requires more than recording a selected GPU UUID set.

A valid subset deployment must prove that a Worker started on a host with
additional GPUs can access exactly its configured physical GPUs. AstrumWeaver
uses an explicit GPU preflight mode rather than treating raw NVIDIA inventory
visibility and effective device access as the same thing.

## Isolation contract

AstrumWeaver has two explicit Worker GPU preflight modes:

```text
exact-visible
  raw nvidia-smi UUID set == Worker gpu_uuids

isolated-access
  reviewed UUID → /dev/nvidiaN map
  + proposed systemd DevicePolicy/DeviceAllow
  + ordered CUDA_VISIBLE_DEVICES
  + selected physical devices open successfully
  + every unselected visible physical device is denied
```

For a GPU subset Worker, the deployment path is:

```text
host-visible GPU superset
        ↓
canonical UUID → /dev/nvidiaN mapping
        ↓
transient service with proposed DevicePolicy/DeviceAllow
        ↓
actual selected-open + unselected-denied access proof
        ↓
persist reviewed map + systemd isolation
        ↓
Worker service cgroup
        ↓
verify-isolated-access ExecStartPre
        ↓
Worker daemon independently repeats isolated-access proof
        ↓
ManagedRuntime start/readiness
        ↓
Worker registration
```

A host where raw visibility is already exact continues to use
`exact-visible`. AstrumWeaver never turns an `exact-visible` mismatch into
an accepted superset merely because `CUDA_VISIBLE_DEVICES` is present.

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

A successful transient capability probe is only authorization to materialize
the subset policy. The generated Worker contract uses
`gpu_preflight_mode = "isolated-access"` and an explicit reviewed
`gpu_device_map`. Worker startup repeats the effective access proof both in
`ExecStartPre` and in the Worker daemon before registration/runtime startup.
Configuration text alone is never accepted as proof of isolation.

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
When `gpuIsolation.enable = true`, the generated Worker configuration uses
`gpu_preflight_mode = "isolated-access"`, records the immutable reviewed map,
and uses the same effective-access verifier in `ExecStartPre`.

The `deviceMap` keys must exactly match `gpuUuids`. A NixOS evaluation that
contains a device policy is not, by itself, evidence that the surrounding
container/VM actually enforces that policy.

## Real-host acceptance

There are two canonical outcomes for a host-visible GPU superset.

### Enforceable subset

Run this only after the isolated Worker configuration has been installed and
while the Worker service is inactive. The checker validates effective systemd
properties, requires the in-service `isolated-access` gate, starts the Worker,
waits for local ready/registered state, and stops it again.

```sh
sudo astrumweaver-runtime-deployment-accept \
  --gpu-uuid GPU-REDACTED-A \
  --revision <reviewed-git-sha> \
  --deployment-path systemd
```

For a NixOS deployment use `--deployment-path nixos`.

### Isolation unavailable / fail closed

When a VM/LXC exposes more GPUs than the Worker selects but its delegated
device boundary cannot deny an unselected physical GPU, the correct result is
not a partially isolated Worker. Keep the installed Worker inactive and pass a
temporary candidate Worker configuration that declares
`gpu_preflight_mode = "isolated-access"`:

```sh
sudo astrumweaver-runtime-deployment-accept \
  --candidate-worker-config ./candidate-worker.toml \
  --expect-isolation-unavailable \
  --revision <reviewed-git-sha> \
  --deployment-path systemd
```

This route never starts the candidate Worker. It requires a real host-visible
superset, reruns the transient effective-access probe, and succeeds only when
the probe returns the reviewed `ISOLATION_UNAVAILABLE` result. The resulting
redacted evidence records `isolation_enforcement=UNAVAILABLE`,
`worker_started=NO`, `fail_closed=PASS`, and `overall=PASS`.

The operator must then narrow guest-visible GPU exposure at the
VM/LXC/hypervisor boundary. AstrumWeaver does not mutate that infrastructure.

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

For an enforceable subset, a PASS proves:

- the host really had a GPU superset
- the Worker configuration matched the selected set
- GPU ownership preflight was enabled in `isolated-access` mode
- UUID/device mapping was valid
- `DevicePolicy=closed` was active
- the only physical `DeviceAllow` entries were the selected GPUs
- CUDA visible-device order matched Worker order
- the in-service isolated-access ownership gate remained present
- effective device isolation allowed the Worker to start
- the isolated Worker reached local ready + registered
- the service stopped cleanly after acceptance

For an unavailable subset, a PASS proves:

- the host really had a GPU superset
- the candidate contract requested isolated-access
- the effective-access probe could not deny an unselected GPU
- no candidate Worker was started
- fail-closed behavior was preserved

## Current v0.x boundary

This contract targets whole physical NVIDIA GPUs.

MIG device-instance isolation and NVIDIA capability-node policy are not claimed
by this acceptance and require a separate compatibility contract.
