# Installation

This page answers one question: **how do I put AstrumWeaver on a machine?**

If this is your first installation, read [Getting Started](getting-started.md)
after this page. It walks through a minimal Control + Worker deployment and a
real `debug.echo` job.

> AstrumWeaver is still pre-v1. The supported installation paths are currently
> intentionally narrow.

## Supported installation paths

| Host | Installation path | Status |
| --- | --- | --- |
| NixOS x86_64-linux | Flake + NixOS module | Recommended / first-class |
| Other systemd Linux x86_64 | `#installer` Nix profile + first-run TUI | Recommended / supported |
| pip-only deployment | Python package only | Not a complete host installation |
| Docker/Kubernetes | — | Not currently provided |

AstrumWeaver does not install an operating system, PostgreSQL server, NVIDIA
host driver, VM/LXC, PCI passthrough, or hypervisor configuration.

## Before installing

Decide which role the machine will run:

- **Control** — durable scheduler/API; requires PostgreSQL.
- **Worker** — executes jobs; connects to Control.
- **Control + Worker** — valid for a small/smoke deployment.

For a GPU Worker, the host must already have a functioning NVIDIA driver and:

```sh
nvidia-smi
```

must work before AstrumWeaver installation.

For Control, prepare:

- a PostgreSQL database
- a client authority token
- a Worker authority token

The two tokens must be different.

A convenient way to create tokens is:

```sh
openssl rand -hex 32
openssl rand -hex 32
```

Keep them outside the repository.

## PostgreSQL prerequisite

**For AstrumWeaver v0.1, use PostgreSQL 17.x.** The repository CI currently
runs the durable Control integration tests against PostgreSQL 17, so PostgreSQL
17 is the validated and recommended major version. Within that major version,
use the latest available PostgreSQL 17 minor release.

Other PostgreSQL major versions may work, but they are not currently covered by
AstrumWeaver CI and should be treated as unvalidated rather than supported.

AstrumWeaver does not provision PostgreSQL. You may use an existing local,
remote, or managed PostgreSQL 17 instance.

For a simple local PostgreSQL installation where you have the usual
`postgres` administrator account, one possible bootstrap is:

```sh
sudo -u postgres createuser --pwprompt astrumweaver
sudo -u postgres createdb --owner astrumweaver astrumweaver
```

Then the Control connection string has the general shape:

```text
postgresql://astrumweaver:PASSWORD@DB_HOST:5432/astrumweaver
```

This is only a database bootstrap example. PostgreSQL authentication,
backups, TLS, HA, and network policy remain deployment/operator
responsibilities.

AstrumWeaver schema creation/upgrades are separate: run
`astrumweaver-migrate`, or explicitly enable the NixOS
`migrateOnStart` option described below.

## Where configuration files live

The final paths differ by deployment method.

### NixOS

With the NixOS modules, **do not create `control.toml` or `worker.toml`
manually**. Non-secret configuration is declared in Nix under
`services.astrumweaver.control` / `services.astrumweaver.worker`; the module
generates the TOML in the Nix store and points the systemd service at it.

Secret environment files are ordinary host files outside the Nix store. The
examples in this guide use:

```text
/etc/astrumweaver/control.env
/etc/astrumweaver/worker.env
```

and reference those exact paths through `environmentFile`.

### Generic systemd Linux

For the setup wrappers, `control.toml`, `control.env`, `worker.toml`, and
`worker.env` are **input/source files**. They may initially live anywhere
convenient, for example a temporary setup directory in your home directory:

```text
~/astrumweaver-setup/control.toml
~/astrumweaver-setup/control.env
~/astrumweaver-setup/worker.toml
~/astrumweaver-setup/worker.env
```

Pass those paths to the setup wrapper with `--config` and
`--environment-file`. The wrapper copies them into the canonical runtime
locations:

```text
/etc/astrumweaver/control.toml
/etc/astrumweaver/control.env
/etc/astrumweaver/worker.toml
/etc/astrumweaver/worker.env
```

A RuntimeProvider deployment manifest, when used, is installed as:

```text
/etc/astrumweaver/runtime-deployment.json
```

After a successful generic-systemd install, systemd reads the canonical
`/etc/astrumweaver/` copies, not the original setup-directory files.

Keep `*.env` files out of Git. They contain authority/database credentials.

## Option A — NixOS

Add AstrumWeaver as a flake input:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    astrumweaver.url = "github:upiscium/AstrumWeaver";
  };

  outputs = { self, nixpkgs, astrumweaver, ... }: {
    nixosConfigurations.my-host = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";

      modules = [
        astrumweaver.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

Then configure the role in `configuration.nix`.

### Control

Minimal Control module:

```nix
{
  services.astrumweaver.control = {
    enable = true;

    settings.control = {
      host = "127.0.0.1";
      port = 9000;
    };

    environmentFile = "/etc/astrumweaver/control.env";

    # Convenient for a first deployment. Production operators may prefer
    # explicit migration execution instead.
    migrateOnStart = true;
  };
}
```

Create the local protected configuration directory if this is the first
AstrumWeaver service on the host:

```sh
sudo install -d -m 0750 /etc/astrumweaver
```

Then create `/etc/astrumweaver/control.env` outside the Nix store:

```text
ASTRUMWEAVER_DATABASE_URL=postgresql://USER:PASSWORD@DB_HOST:5432/astrumweaver
ASTRUMWEAVER_CLIENT_TOKEN=REPLACE_WITH_CLIENT_TOKEN
ASTRUMWEAVER_WORKER_TOKEN=REPLACE_WITH_WORKER_TOKEN
```

Protect it:

```sh
sudo chown root:root /etc/astrumweaver/control.env
sudo chmod 600 /etc/astrumweaver/control.env
```

### Minimal Worker

For the first smoke test, use the built-in `debug.echo` executor. It avoids
mixing AstrumWeaver installation problems with model-runtime problems.

```nix
{
  services.astrumweaver.worker = {
    enable = true;

    workerId = "worker-smoke";
    workerClass = "cpu";
    controlUrl = "http://127.0.0.1:9000";

    capabilities = [ "debug.echo" ];

    gpuUuids = [ ];
    totalVramMb = 0;
    maxSingleGpuVramMb = 0;

    executorFactory =
      "astrumweaver.executors.structured_echo:create_executor";

    environmentFile = "/etc/astrumweaver/worker.env";
  };
}
```

Create `/etc/astrumweaver/worker.env`:

```text
ASTRUMWEAVER_WORKER_TOKEN=REPLACE_WITH_THE_SAME_WORKER_TOKEN_USED_BY_CONTROL
```

Then rebuild:

```sh
sudo nixos-rebuild switch --flake .#my-host
```

Continue with [Getting Started](getting-started.md#verify-the-installation).

### GPU + RuntimeProvider Worker

After the smoke Worker succeeds, replace the manual `executorFactory` path
with `runtime.enable = true`.

RuntimeProvider deployment is documented in:

- [Nix Packaging and NixOS Modules](nix.md#first-class-runtimeprovider-execution)
- [Runtime Providers and Execution Demand](runtime-providers.md)
- [Runtime Deployment GPU Isolation Acceptance](runtime-deployment-acceptance.md)

Do not configure both `executorFactory` and `runtime.enable = true`.

## Option B — generic systemd Linux with Nix

This path does **not** require a source checkout.

It assumes the host already has Nix with the `nix-command` and `flakes`
features available. AstrumWeaver does not bootstrap Nix itself.

### Recommended: first-run TUI

Install the combined first-run package:

```sh
sudo nix profile add \
  --profile /nix/var/nix/profiles/astrumweaver-installer \
  github:upiscium/AstrumWeaver#installer
```

Then run:

```sh
sudo /nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-setup-tui
```

The default TUI mode is `first-run`. It can wrap the common bootstrap work
that would otherwise require several manual commands:

```text
choose Control / Worker / Control+Worker
        ↓
Control settings + hidden PostgreSQL URL input
        ↓
generate or enter distinct authority tokens
        ↓
write canonical Control config/env
        ↓
run astrumweaver-migrate
        ↓
install/start Control and wait for /v1/ready
        ↓
discover NVIDIA GPU UUID/VRAM/compute capability
        ↓
select Worker GPU ownership
        ↓
generate Worker config/env
        ↓
install Worker systemd integration
        ↓
smoke debug.echo OR reviewed RuntimeProvider setup
        ↓
wait for Worker registration/readiness
```

Secret prompts use no-echo input. Secret values are omitted from the displayed
review/digest and are written only through the protected environment-file path.

The TUI does **not** provision PostgreSQL itself, install/replace NVIDIA host
drivers, configure PCI passthrough, or mutate hypervisor configuration.

For an already-installed Worker where you only want RuntimeProvider setup, use:

```sh
sudo /nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-setup-tui \
  --mode runtime \
  --driver astrumweaver.setup.systemd:create_systemd_driver
```

The manual commands below remain documented for troubleshooting,
non-interactive deployment, and understanding exactly what the TUI wraps.

### Manual Control package


```sh
sudo nix profile add \
  --profile /nix/var/nix/profiles/astrumweaver-control \
  github:upiscium/AstrumWeaver#control
```

The profile contains:

- `astrumweaver-control`
- `astrumweaver-migrate`
- `astrumweaver-setup-control-plane`
- host-integration assets

Check it:

```sh
/nix/var/nix/profiles/astrumweaver-control/bin/astrumweaver-control --help
```

Prepare a non-secret `control.toml`:

```toml
[control]
host = "127.0.0.1"
port = 9000
worker_ttl_seconds = 60
lease_seconds = 300
maintenance_interval_seconds = 5
access_log = false
```

Prepare `control.env`:

```text
ASTRUMWEAVER_DATABASE_URL=postgresql://USER:PASSWORD@DB_HOST:5432/astrumweaver
ASTRUMWEAVER_CLIENT_TOKEN=REPLACE_WITH_CLIENT_TOKEN
ASTRUMWEAVER_WORKER_TOKEN=REPLACE_WITH_WORKER_TOKEN
```

Apply database migrations before first startup:

```sh
sudo env \
  ASTRUMWEAVER_DATABASE_URL='postgresql://USER:PASSWORD@DB_HOST:5432/astrumweaver' \
  /nix/var/nix/profiles/astrumweaver-control/bin/astrumweaver-migrate
```

Install and start the service:

```sh
sudo /nix/var/nix/profiles/astrumweaver-control/bin/astrumweaver-setup-control-plane \
  --config ./control.toml \
  --environment-file ./control.env \
  --executable /nix/var/nix/profiles/astrumweaver-control/bin/astrumweaver-control \
  --start
```

### Manual Worker package

Install the Worker package:

```sh
sudo nix profile add \
  --profile /nix/var/nix/profiles/astrumweaver-worker \
  github:upiscium/AstrumWeaver#worker
```

Check it:

```sh
/nix/var/nix/profiles/astrumweaver-worker/bin/astrumweaver-worker --help
```

The current packaged generic setup wrapper is GPU-worker oriented. Confirm the
available GPUs:

```sh
nvidia-smi --query-gpu=uuid,memory.total,name --format=csv,noheader,nounits
```

Create `worker.toml` using the selected GPU facts:

```toml
[worker]
id = "worker-gpu-smoke"
class = "gpu-single"
control_url = "http://CONTROL_HOST:9000"
capabilities = ["debug.echo"]
gpu_uuids = ["GPU-REPLACE-ME"]
gpu_count = 1
total_vram_mb = 24576
max_single_gpu_vram_mb = 24576
max_concurrency = 1
health_host = "127.0.0.1"
health_port = 9100

[executor]
factory = "astrumweaver.executors.structured_echo:create_executor"

[executor.settings]
```

Create `worker.env`:

```text
ASTRUMWEAVER_WORKER_TOKEN=REPLACE_WITH_THE_SAME_WORKER_TOKEN_USED_BY_CONTROL
```

Install the service:

```sh
sudo /nix/var/nix/profiles/astrumweaver-worker/bin/astrumweaver-setup-gpu-worker \
  --config ./worker.toml \
  --environment-file ./worker.env \
  --executable /nix/var/nix/profiles/astrumweaver-worker/bin/astrumweaver-worker \
  --start
```

The setup wrapper reads the selected GPU UUID sequence only from
`worker.toml`'s `[worker].gpu_uuids`. There is intentionally no separate
`--gpu-uuid` setup argument; `/etc/astrumweaver/gpu-uuids` is generated from
that canonical Worker configuration for systemd preflight/isolation.

`--gpu-isolation auto` is the default. If the host exposes additional GPUs,
the setup path attempts to isolate the selected physical GPU set through the
Worker systemd cgroup. If it cannot prove the mapping/isolation, setup fails
closed.

For a RuntimeProvider-backed Worker, first establish the base Worker service,
then use the reviewed RuntimeProvider setup path described in
[Interactive Worker/runtime Setup TUI](setup-tui.md) and
[Existing-Node Deployment](deployment.md).

## Upgrading

### NixOS

Update the flake input and rebuild:

```sh
nix flake update astrumweaver
sudo nixos-rebuild switch --flake .#my-host
```

Review release/migration changes before enabling a newer Control binary.

### Generic systemd + Nix profile

For the TUI-first installation, upgrade the combined installer profile:

```sh
sudo nix profile upgrade \
  --profile /nix/var/nix/profiles/astrumweaver-installer \
  --all
```

If you intentionally installed separate role profiles, upgrade them instead:

```sh
sudo nix profile upgrade \
  --profile /nix/var/nix/profiles/astrumweaver-control \
  --all

sudo nix profile upgrade \
  --profile /nix/var/nix/profiles/astrumweaver-worker \
  --all
```

If the new Control binary contains new database migrations, run
`astrumweaver-migrate` before considering Control ready.

## What is not an installation method yet

### pip

The Python package can be useful for development, but a pip-only installation
does not currently constitute a complete supported host deployment because the
systemd/setup integration is packaged separately.

### copying setup scripts from the repository

This works for development but is not the preferred source-free deployment
path. Use the Nix `control` / `worker` packages, which include the reviewed
integration assets.

## Next

Once the binaries/services are installed, continue with:

[Getting Started — verify the installation](getting-started.md#verify-the-installation)
