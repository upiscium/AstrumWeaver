# Nix Packaging and NixOS Modules

AstrumWeaver provides a pinned Nix flake for reproducible packaging and NixOS host integration.

## Current runtime boundary

As of this v0.1 bootstrap, AstrumWeaver contains the durable Control Plane domain/repository, Worker/resource contracts, generic JobExecutor boundary, setup scripts, and systemd integration.

It does **not yet** contain the long-running Control and Worker daemon entrypoints.

That operational runtime is tracked in #15.

For this reason, the current `control` and `worker` flake packages are **runtime support closures**, not fake service daemons, and the NixOS modules require an explicit `command` while #15 is open.

## Flake outputs

For `x86_64-linux`:

```text
packages.x86_64-linux.astrumweaver
packages.x86_64-linux.control
packages.x86_64-linux.worker
packages.x86_64-linux.integration
packages.x86_64-linux.default
```

### astrumweaver

The immutable Python package containing the provider-neutral AstrumWeaver domain/runtime library.

### control

Control Plane support closure containing:

- AstrumWeaver Python package
- psycopg
- PostgreSQL migration assets
- existing-node integration/setup assets

It currently does not claim to contain `astrumweaver-control`; #15 will add the real daemon.

### worker

Worker support closure containing:

- AstrumWeaver Python package
- GPU preflight/setup integration assets

It currently does not claim to contain `astrumweaver-worker`; #15 will add the real daemon.

### integration

Host-integration package containing immutable copies/wrappers for:

- `astrumweaver-setup-control-plane`
- `astrumweaver-setup-gpu-worker`
- `astrumweaver-gpu-preflight`
- systemd templates

This can be used on a system where Nix supplies package artifacts while the non-NixOS setup scripts own host integration.

## Reproducibility

`flake.lock` is committed.

CI runs:

```sh
nix flake check --no-update-lock-file --print-build-logs
```

so an unreviewed nixpkgs update cannot silently enter a build.

Intentional dependency updates should update `flake.lock` in a reviewed change.

## NixOS modules

The flake exposes:

```text
nixosModules.control
nixosModules.worker
nixosModules.default
```

The default module imports both role modules.

A consuming flake may import it with:

```nix
{
  inputs.astrumweaver.url = "github:upiscium/AstrumWeaver";

  outputs = { nixpkgs, astrumweaver, ... }: {
    nixosConfigurations.example = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        astrumweaver.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

## Worker module

Until #15 provides the real daemon, the module requires the reviewed Worker executable explicitly.

Example shape:

```nix
{ config, inputs, pkgs, ... }:

{
  services.astrumweaver.worker = {
    enable = true;

    package = inputs.astrumweaver.packages.${pkgs.system}.worker;

    # Temporary explicit boundary until #15 provides the packaged daemon.
    command = "/absolute/path/to/astrumweaver-worker";

    gpuUuids = [
      "GPU-example-a"
      "GPU-example-b"
    ];

    nvidiaSmiPackage = config.hardware.nvidia.package;

    supplementaryGroups = [
      "video"
      "render"
    ];

    settings = {
      worker = {
        class = "multi-gpu";
      };
    };

    # Secret values belong outside the Nix store.
    environmentFile = "/run/secrets/astrumweaver-worker.env";
  };
}
```

The module manages:

- system user/group
- generated non-secret TOML config
- systemd service
- runtime/state directories
- restart policy
- service hardening
- exact GPU UUID preflight before Worker startup

`gpuUuids` must be nonempty and unique.

The module deliberately does not infer the GPU set from PCI ordinals or Proxmox configuration.

## Control module

Example shape:

```nix
{ inputs, pkgs, ... }:

{
  services.astrumweaver.control = {
    enable = true;

    package = inputs.astrumweaver.packages.${pkgs.system}.control;

    # Temporary explicit boundary until #15 provides the packaged daemon.
    command = "/absolute/path/to/astrumweaver-control";

    settings = {
      control = {
        listen = "127.0.0.1:9000";
      };
    };

    environmentFile = "/run/secrets/astrumweaver-control.env";
  };
}
```

The module owns service wiring, not PostgreSQL provisioning. The configured durable database must already exist according to the Host Prerequisite Contract.

## Secret handling

Do not put credentials, tokens, private keys, database passwords, or private model credentials in `settings`.

Nix-generated configuration is stored in `/nix/store` and must be treated as public to local users.

Use `environmentFile` or another protected secret-management mechanism for sensitive values.

## Non-NixOS with Nix packages

The setup assets can be built independently:

```sh
nix build .#integration
```

Then an existing systemd Linux node can use the packaged wrappers, for example:

```sh
sudo ./result/bin/astrumweaver-setup-gpu-worker \
  --config ./worker.toml \
  --environment-file ./worker.env \
  --executable /absolute/path/to/astrumweaver-worker \
  --gpu-uuid GPU-example-a
```

The setup layer still does not create a VM/LXC, configure Proxmox, configure IOMMU, or install host GPU drivers.

## Relationship to #15

When #15 lands:

- `control` will include the real `astrumweaver-control` entrypoint
- `worker` will include the real `astrumweaver-worker` entrypoint
- the NixOS modules can default `command` to their packaged executables
- the existing module/service/preflight contracts remain reusable

Keeping that work separate prevents packaging from defining runtime semantics that do not yet exist.
