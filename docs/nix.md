# Nix Packaging and NixOS Modules

AstrumWeaver provides a pinned Nix flake for reproducible packaging and NixOS host integration.

## Runtime boundary

AstrumWeaver now ships real long-running entrypoints:

- `astrumweaver-control`
- `astrumweaver-worker`
- `astrumweaver-migrate`

The `control` and `worker` flake packages include these packaged entrypoints together with their role-specific runtime closure.

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

It includes the packaged `astrumweaver-control` and `astrumweaver-migrate` entrypoints.

### worker

Worker support closure containing:

- AstrumWeaver Python package
- GPU preflight/setup integration assets

It includes the packaged `astrumweaver-worker` entrypoint.

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

    workerId = "worker-example";
    workerClass = "multi-gpu";
    controlUrl = "https://control.example.invalid";
    capabilities = [ "llm.chat" ];

    gpuUuids = [
      "GPU-example-a"
      "GPU-example-b"
    ];
    totalVramMb = 24576;
    maxSingleGpuVramMb = 12288;
    nvidiaSmiPackage = config.hardware.nvidia.package;

    executorFactory = "my_executor:create_executor";

    supplementaryGroups = [
      "video"
      "render"
    ];

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

## Runtime entrypoint overrides

Both NixOS modules default to their packaged daemon entrypoint.

The optional `command` setting remains available only for reviewed development/testing overrides.

Control also exposes `migrateOnStart`, defaulting to `false`, for deployments that explicitly authorize schema migration as part of service startup.
