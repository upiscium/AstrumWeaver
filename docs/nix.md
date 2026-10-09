# Nix Packaging and NixOS Modules

TSUMGI provides a pinned Nix flake for reproducible packaging and NixOS host integration.

## Runtime boundary

TSUMGI now ships real long-running entrypoints:

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
packages.x86_64-linux.installer
packages.x86_64-linux.integration
packages.x86_64-linux.default
```

### astrumweaver

The immutable Python package containing the provider-neutral TSUMGI domain/runtime library.

### control

Control Plane support closure containing:

- TSUMGI Python package
- psycopg
- PostgreSQL migration assets
- existing-node integration/setup assets

It includes the packaged `astrumweaver-control` and `astrumweaver-migrate` entrypoints.

### worker

Worker support closure containing:

- TSUMGI Python package
- GPU preflight/setup integration assets

It includes the packaged `astrumweaver-worker` entrypoint.

### installer

Combined generic-systemd first-run closure containing:

- TSUMGI Control + Worker entrypoints
- psycopg and PostgreSQL migration assets
- host integration/setup wrappers
- `astrumweaver-setup-tui`

Install this package when you want one profile that can bootstrap Control,
Worker, or both without a source checkout.

### integration

Host-integration package containing immutable copies/wrappers for:

- `astrumweaver-setup-control-plane`
- `astrumweaver-setup-gpu-worker`
- `astrumweaver-gpu-preflight`
- `astrumweaver-gpu-device-map`
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
  inputs.astrumweaver.url = "github:upiscium/TSUMGI";

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

The module defaults to the packaged `astrumweaver-worker` daemon. `command` remains available only as an explicit development/testing override.

Example GPU Worker:

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

`gpuUuids` must be unique. Leave it empty for a CPU/non-GPU Worker; when it is non-empty, `nvidiaSmiPackage`, `totalVramMb`, and `maxSingleGpuVramMb` must describe the GPU resource shape consistently.

The module deliberately does not infer the GPU set from PCI ordinals or Proxmox configuration.

### First-class RuntimeProvider execution

A Worker can use a selected RuntimeProvider instead of a manually configured
`executorFactory`. The two execution paths are mutually exclusive.

Example:

```nix
services.astrumweaver.worker = {
  enable = true;

  workerId = "worker-runtime";
  workerClass = "modern-single";
  controlUrl = "https://control.example.invalid";
  capabilities = [ "llm.chat" "text.generate" ];

  gpuUuids = [ "GPU-example-a" ];
  accelerators = [
    {
      uuid = "GPU-example-a";
      memory_mb = 24576;
      compute_capability = "8.6";
      device_class = "NVIDIA example class";
    }
  ];
  totalVramMb = 24576;
  maxSingleGpuVramMb = 24576;
  nvidiaSmiPackage = config.hardware.nvidia.package;

  runtime = {
    enable = true;
    provider = "vllm";

    # Runtime packages are explicit and enter only this service closure.
    packages = [ pkgs.vllm ];

    modelRef = "/srv/models/example";
    modelFormat = "safetensors";
    modelTopology = "dense";
    residencyPolicy = "vram_only";
    estimatedModelSizeMb = 16000;

    providerConfig = {
      gpu_memory_utilization = 0.90;
    };
  };

  environmentFile = "/run/secrets/astrumweaver-worker.env";
};
```

The module serializes provider identity, provider configuration and
`ExecutionDemand` to an immutable `RuntimeDeploymentSpec`. At service start
the Worker re-checks compatibility against its current host/Worker facts,
starts the ManagedRuntime, waits for runtime readiness, then registers with
Control. On Worker shutdown it stops/releases the ManagedRuntime.

Nix evaluation does not implicitly download models or run network installers.
For model formats/providers that require acquisition, provision the reviewed
model declaratively or use an already-present local reference. A missing model
fails runtime startup rather than triggering an unreviewed download.

`providerConfig`, model metadata, and demand metadata are stored in the Nix
store and therefore must remain non-secret. Credentials stay in
`environmentFile` or another protected secret mechanism.

The exact-set GPU preflight remains authoritative. For a host-visible GPU
superset, enable `gpuIsolation` and declare the exact UUID→`/dev/nvidiaN`
mapping:

```nix
gpuIsolation = {
  enable = true;
  deviceMap."GPU-example-a" = "/dev/nvidia0";
};
```

The module verifies that mapping in a separate host-level oneshot service, then
runs the Worker under `DevicePolicy=closed` with only the selected physical
GPU nodes plus configured shared NVIDIA control/UVM nodes. The existing
`gpu-preflight` still runs inside the restricted Worker cgroup.

See [Runtime Deployment GPU Isolation Acceptance](runtime-deployment-acceptance.md)
for the real-host acceptance contract.

## Control module

Example shape:

```nix
{ inputs, pkgs, ... }:

{
  services.astrumweaver.control = {
    enable = true;

    package = inputs.astrumweaver.packages.${pkgs.system}.control;

    # null/omitted preserves the secure daemon default: bearer.
    clientAuth = "bearer";

    settings.control = {
      host = "127.0.0.1";
      port = 9000;
    };

    environmentFile = "/run/secrets/astrumweaver-control.env";

    # Optional explicit schema authority:
    # migrateOnStart = true;
  };
}
```

The module owns service wiring, not PostgreSQL provisioning. The configured durable database must already exist according to the Host Prerequisite Contract.

`clientAuth` accepts `"bearer"`, `"none"`, or `null`. The module default is
`null`, which leaves the TOML setting absent and therefore preserves the
Control daemon's secure `bearer` default. Set `clientAuth = "none";` only
when the deployment boundary deliberately protects Client job
submit/read/cancel access. In that mode the protected Control environment file
does not need `ASTRUMWEAVER_CLIENT_TOKEN`; it still requires
`ASTRUMWEAVER_DATABASE_URL` and `ASTRUMWEAVER_WORKER_TOKEN`.


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
  --environment-file ./worker.env
```

The setup wrapper derives the expected GPU UUID set from
`worker.toml` and does not accept a second GPU ownership argument.

The setup layer still does not create a VM/LXC, configure Proxmox, configure IOMMU, or install host GPU drivers.

## Runtime entrypoint overrides

Both NixOS modules default to their packaged daemon entrypoint.

The optional `command` setting remains available only for reviewed development/testing overrides.

Control also exposes `migrateOnStart`, defaulting to `false`, for deployments that explicitly authorize schema migration as part of service startup.


## Borrowable development GPU

A GPU Worker hosted on a development machine may enable the ownership-switch wrapper:

```nix
services.astrumweaver.worker.borrowable.enable = true;
```

The module then installs `astrumweaver-gpu-mode`, preconfigured from the Worker's declared GPU UUID set and local health endpoint.

```sh
sudo astrumweaver-gpu-mode development
sudo astrumweaver-gpu-mode astrumweaver
sudo astrumweaver-gpu-mode status
```

`borrowable.drainTimeoutSeconds = 0` is the default and means wait indefinitely for the active job rather than forcing it. See [Borrowable GPU Worker](borrowable-worker.md) for the complete handoff contract.

## Generated first-run module scope and validation

The first-run TUI emits a module with the explicit argument pattern
`{ config, pkgs, ... }:`. Runtime package input is a **simple attribute path
rooted in `pkgs`**, for example `pkgs.ollama`, `pkgs.vllm`, or
`pkgs.llama-cpp`. Each path component must be a non-keyword Nix identifier.
The TUI checks this grammar before review/write; Nix checks that the attribute
exists and denotes a package in the operator's package set.

An unbound root such as `myPkgs` or `inputs` is rejected. For a custom runtime,
expose the package through an ordinary `nixpkgs.overlays` entry and select its
`pkgs.<name>` path (nested paths are allowed). The generated snippet does not
import an arbitrary flake or discover custom lexical variables from the caller.
General expressions, function calls and interpolated attribute names are not
accepted. Literal `${...}` in model/configuration data stays literal; it is not
Nix code. Secrets remain in the separately protected environment files, never
in the generated module or Nix store.

A generated snippet must be imported together with the TSUMGI NixOS
module. It is not a complete host configuration: retain the documented host,
network, PostgreSQL, NVIDIA and GPU-exposure prerequisites and your normal
`nixos-rebuild` review/apply boundary.

The repository's generated-module regression runs with:

```sh
nix run --no-update-lock-file .#check-generated-nixos
```

Run this on a checkout of the candidate revision. The app supplies its pinned
Python dependencies and Nix; the evaluator uses this checkout's locked nixpkgs
and TSUMGI modules, without updating the lock file. It writes disposable
fixtures through the **production first-run renderer**, then evaluates the
actual generated files for Control (default bearer and explicit none), smoke
CPU/GPU Worker, RuntimeProvider Worker, and combined roles. It forces relevant
module assertions, service commands/environment-file bindings, runtime demand
and metadata, and package paths, including nested overlay references and
literal interpolation-like input. A negative control removes only the `pkgs`
argument from an emitted runtime module and must fail with an undefined
variable error.

CI runs this as an additional Nix step alongside `nix flake check`. No service
is started, no host configuration is applied, and model/driver/runtime packages
are not built or executed by the evaluation step. Import-from-derivation is
disabled. This is configuration-evaluation evidence, **not** GPU/provider
hardware acceptance or the operator-led documentation-only Real Smoke gate.
