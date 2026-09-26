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
| Other systemd Linux x86_64 | Nix package + packaged setup wrapper | Supported |
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

AstrumWeaver does not provision PostgreSQL. You may use an existing local,
remote, or managed PostgreSQL instance.

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

Create `/etc/astrumweaver/control.env` outside the Nix store:

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

Install the immutable role package into a dedicated system profile.

### Control package

```sh
sudo nix profile install \
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

### Worker package

Install the Worker package:

```sh
sudo nix profile install \
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
  --gpu-uuid GPU-REPLACE-ME \
  --start
```

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

Upgrade the dedicated profile:

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
