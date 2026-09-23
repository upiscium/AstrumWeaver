# Existing-Node Deployment

AstrumWeaver deploys onto an already-created Linux node. VM/LXC/bare-metal creation and GPU passthrough remain outside the project boundary.

## Separation of responsibilities

Before setup:

```text
infrastructure owner
  ├─ creates VM/LXC/bare metal
  ├─ configures networking/storage
  ├─ configures PCI/GPU passthrough when needed
  ├─ installs the operating system
  └─ makes the required NVIDIA GPU visible
           ↓
AstrumWeaver setup
  ├─ validates prerequisites
  ├─ installs role configuration
  ├─ installs systemd integration
  ├─ validates exact GPU identity for workers
  └─ optionally enables/starts the role service
```

AstrumWeaver setup never invokes `qm create`, `pct create`, or equivalent infrastructure creation.

## Package/runtime prerequisite

Issue #6 owns **host integration**, while the Nix/release package supplies the daemon executable.

The packaged role executable must already be available on the target node:

- `astrumweaver-control`
- `astrumweaver-worker`

The Nix flake provides immutable AstrumWeaver Control/Worker daemon packages and integration assets; see [Nix Packaging and NixOS Modules](nix.md).

Keeping these concerns separate allows the same setup contract to work with a Nix package, a release artifact, or another reviewed packaging mechanism.

## Control Plane

Example:

```sh
sudo ./setup/setup-control-plane.sh \
  --config ./control.toml \
  --environment-file ./control.env \
  --start
```

The script:

- requires an existing Linux/systemd host
- installs configuration under `/etc/astrumweaver/`
- creates the `astrumweaver` service account when needed
- installs `astrumweaver-control.service`
- creates/uses `/var/lib/astrumweaver/`
- optionally enables and starts the service

It does not provision PostgreSQL itself. The configured Control Plane must point at an already-available durable database.

Before first Control startup, apply the packaged schema explicitly:

```sh
ASTRUMWEAVER_DATABASE_URL='postgresql://...' astrumweaver-migrate
```

NixOS deployments may instead opt in to `services.astrumweaver.control.migrateOnStart = true`; the default remains `false` because schema mutation is an explicit authority.

## GPU Worker

Example:

```sh
sudo ./setup/setup-gpu-worker.sh \
  --config ./worker.toml \
  --environment-file ./worker.env \
  --gpu-uuid GPU-example-a \
  --gpu-uuid GPU-example-b \
  --start
```

The expected UUID list is explicit and may contain one or several GPUs.

The script installs:

- `/etc/astrumweaver/worker.toml`
- optional `/etc/astrumweaver/worker.env`
- `/etc/astrumweaver/gpu-uuids`
- `/usr/local/libexec/astrumweaver/gpu-preflight`
- `/etc/systemd/system/astrumweaver-worker.service`

For a live install it verifies `nvidia-smi` before the worker can be started.

The systemd unit repeats exact UUID preflight as `ExecStartPre`, so reboot/service restart cannot silently start a worker against a changed GPU set.

## Exact GPU identity

The worker preflight compares sets, not ordinals.

These are equivalent:

```text
expected:
GPU-a
GPU-b

observed:
GPU-b
GPU-a
```

This is rejected because an unexpected GPU is also unsafe:

```text
expected:
GPU-a
GPU-b

observed:
GPU-a
GPU-b
GPU-c
```

`/dev/nvidia0` ordering is never used as identity.

## Conflict-safe idempotency

Setup scripts are convergent for reviewed identical inputs.

Running the same command again:

- preserves identical configuration
- corrects expected file modes
- regenerates an identical systemd unit
- succeeds without replacing unrelated state

If an existing managed file differs, setup **fails** instead of silently overwriting it.

To change configuration, the operator must deliberately replace/remove the old managed file through a reviewed change procedure before rerunning setup.

This policy prevents a generic bootstrap command from unexpectedly changing worker identity or service authority.

## Staged installs

Both setup scripts support:

```text
--root DIR
```

This writes the target filesystem layout under a staging directory without:

- creating system users
- calling systemd
- touching GPUs
- requiring root

It is used by CI and can also be used by packaging/integration tooling.

`--start` is intentionally rejected with staged installs.

## Existing Proxmox VM

A Proxmox VM is eligible when, before setup:

- the VM already exists
- GPU passthrough is already configured if required
- the guest OS is operational
- the role executable is installed
- for a GPU worker, guest `nvidia-smi` reports exactly the intended UUID set

No Proxmox API access is required by AstrumWeaver.

## Existing Proxmox LXC

A Proxmox LXC is eligible when, before setup:

- the container already exists
- NVIDIA device mapping is already configured by the operator
- driver/userspace compatibility is already resolved
- `nvidia-smi` works inside the container
- the AstrumWeaver service account can access the devices

AstrumWeaver does not alter LXC privilege mode, cgroup device rules, bind mounts, or host drivers.

## Secrets

Environment files may contain deployment secrets and therefore:

- must not be committed to the public repository
- are installed mode `0640`
- are readable by root and the AstrumWeaver service account

Non-secret role configuration belongs in the TOML configuration file.

## systemd lifecycle

Setup defaults to installing/reloading without starting the service.

Use `--start` only after configuration and credentials have been reviewed.

GPU Worker ordering is:

```text
systemd start
    ↓
exact UUID preflight (privileged ExecStartPre)
    ↓
worker process starts
    ↓
worker may register with Control
```

Therefore worker registration cannot occur before the local GPU identity gate passes.
