# Interactive Setup TUI

AstrumWeaver provides a keyboard-first terminal setup wizard.

For the recommended generic-systemd installation using the dedicated installer
profile, run it with the profile path directly:

```sh
sudo /nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-setup-tui
```

A bare `astrumweaver-setup-tui` command is also valid when the installer
profile's `bin` directory is already on `PATH`. The first-run implementation
does not require you to modify `PATH` when using the absolute profile path.

The default mode is now **first-run**. It wraps the common Control/Worker host
bootstrap and then reuses the existing deterministic RuntimeProvider setup
backend. Client API auth defaults to `bearer` for secure compatibility. Choosing
`none` is explicit and means the TUI omits the Client token from protected
environment material while leaving Worker bearer authentication mandatory.

backend when requested.

## First-run flow

```text
local host discovery
    ↓
choose role: Control / Worker / Control+Worker
    ↓
Control phase (when selected)
  - Control bind settings
  - explicit Client API auth choice: bearer or none
  - hidden PostgreSQL URL input
  - bearer mode: generate or enter distinct Client/Worker authority tokens
  - none mode: generate or enter only the Worker authority token
  - write canonical config/env
  - run migration
  - install/start Control
  - wait for /v1/ready
    ↓
Worker phase (when selected)
  - NVIDIA GPU discovery
  - choose GPU ownership
  - Worker identity/class
  - generate canonical Worker config/env
  - install systemd integration
    ↓
choose execution mode
  - smoke: built-in debug.echo
  - runtime: existing RuntimeProvider wizard
    ↓
health/readiness summary
```

For generic systemd first-run apply, run the TUI as root. Secret values are
collected with no-echo input and are never placed in the first-run digest,
Runtime SetupPlan, progress output, or public evidence.

## Runtime-only flow

Runtime-only setup remains available:

```sh
sudo /nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-setup-tui \
  --mode runtime
```

On generic systemd, runtime-only mode is an **existing Worker migration**.
The TUI reads the installed non-secret `/etc/astrumweaver/worker.toml` and
canonical Worker unit, verifies their execution authority, and uses that
installed contract as the immutable source for Worker identity, Control URL,
GPU ownership/order, resource shape, accelerator facts, labels, concurrency,
GPU preflight and health settings. It never reads `worker.env` or asks the
operator to reconstruct those values.

The supported source state for the first migration is the canonical smoke
Worker (`debug.echo` + the built-in structured-echo executor). An already
canonical RuntimeProvider Worker is accepted for idempotent reruns. Mixed,
custom-executor, store-pinned, provider-switch, or otherwise unrecognized
Worker/unit states fail closed before mutation.

The generic-systemd flow is:

```text
load + validate installed Worker contract
    ↓
NVIDIA GPU discovery verifies preserved ownership facts
    ↓
model + execution demand
    ↓
all runtime compatibility reports
    ↓
USER selects runtime
    ↓
optional provider-specific settings
    ↓
deterministic SetupPlan
  - runtime prerequisites/config
  - stop existing Worker
  - reconcile Worker config + unit execution authority
  - runtime preflight
  - restart + health
    ↓
dry-run / exact plan confirmation / apply
```

On NixOS, runtime mode remains a planning workflow over explicitly selected
Worker/GPU facts; declarative `services.astrumweaver.worker.runtime` remains
the normal mutation boundary.

## Runtime authority

Every installed first-class provider remains visible in the chooser, including
incompatible providers.

Blocking and advisory reasons are displayed separately. Selecting an
incompatible provider does not cause a fallback. The user can instead:

- choose another runtime
- edit the execution demand
- edit the selected provider's options
- quit

The final setup always uses:

```text
RuntimeSelection(mode=explicit, provider_id=<user choice>)
```

so the TUI cannot silently substitute another runtime.

## GPU ownership

GPU discovery uses the shared setup discovery backend, not UI-local parsing.

The backend queries:

- exact NVIDIA GPU UUID
- total device VRAM
- compute capability when the installed driver exposes it

If compute capability is unavailable, UUID/VRAM discovery remains usable and
providers that require architecture evidence return an advisory. `nvidia-smi`
optional-field sentinels such as `N/A` are normalized to unknown evidence
rather than passed into strict accelerator-fact validation.

During first-run (and non-migrating planning flows), the user explicitly
chooses which locally visible GPUs form the Worker. The resulting WorkerSpec
preserves that UUID order and calculates:

- GPU count
- total VRAM
- maximum single-device VRAM
- minimum known compute capability

During generic-systemd runtime migration, those values are not re-entered.
Discovery instead verifies that every installed Worker UUID and preserved
per-device/resource fact still matches the local host. Extra host GPUs are
allowed only as host-superset facts; they are not silently added to Worker
ownership.

The compute capability is represented as the generic Worker label:

```text
gpu.compute_capability.min
```

GPU UUIDs are not copied into SetupHostSnapshot metadata.

## Execution demand editor

The wizard collects provider-neutral model/execution facts:

- model reference
- model format
- dense vs MoE
- residency policy
- estimated model size when known
- minimum total VRAM
- minimum single-device VRAM
- minimum host RAM
- preferred host RAM

GPU topology is derived from the GPU set assigned to the Worker rather than
being separately entered, preventing an internally contradictory Worker shape.

## Runtime-specific options

After the operator picks a runtime, the TUI exposes a curated set of safe
provider configuration fields.

Current examples include:

- Ollama: keep-alive
- llama.cpp: context size, GPU layers, split mode, tensor split, CPU MoE
- vLLM: GPU utilization, TP/EP, CPU offload
- FreeToken: MoE strategy/cache/CPU controls
- ExLlamaV3: autosplit/tensor-parallel mode, GPU split, cache/context/batch

These values construct a new provider configuration and the provider
compatibility check is rerun before any SetupPlan is generated.

The TUI does not translate those settings into provider-specific mutation
commands. Provider setup still enters the shared RuntimeSetupIntent →
SetupPlan path.

## Plan review and approval

The complete SetupPlan is displayed before mutation.

The plan includes a SHA-256 digest. Applying requires typing:

```text
APPLY <first 12 hex characters of reviewed digest>
```

The wizard then requests explicit authorization for each action category that
the exact plan requires:

- privileged actions
- network access
- model download
- model conversion
- other confirmation-gated actions

A denied permission cancels application before mutation.

SetupPlan serialization stores secret references rather than secret values.
The TUI does not request or print secret material.

## Progress and recovery

When a deployment driver is connected, the TUI first runs
`dry_run_plan()`. Blocked actions are displayed before apply.

`apply_plan()` exposes a result callback used by the TUI to show action
progress without implementing a second mutation loop.

Final output shows:

- applied/skipped/failed/blocked actions
- rollback results
- concise recovery guidance

Driver evidence payloads are intentionally not dumped by the TUI.

## Planning-only and deployment modes

The TUI remains usable without deployment authority. When the installer
profile's `bin` directory is already on `PATH`, running without a driver:

```sh
astrumweaver-setup-tui --mode runtime
```

loads and validates the installed Worker contract, completes demand/runtime
selection and produces the exact reviewed SetupPlan, then stops without
mutation. The protected Worker EnvironmentFile is not read.

On an already-integrated generic systemd Worker host, the first-party driver
can be connected with:

```sh
sudo /nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-setup-tui \
  --mode runtime \
  --driver astrumweaver.setup.systemd:create_systemd_driver
```

The Worker service account, Worker TOML, protected token EnvironmentFile, and
systemd unit must already exist. The runtime driver validates the installed
non-secret Worker TOML/unit, owns reviewed runtime package/config/model actions,
stops the smoke Worker, atomically reconciles Worker execution authority,
reloads systemd, and restarts the same Worker. The token EnvironmentFile is
preserved by reference and never read; Control credentials, network identity,
Worker identity and GPU ownership are not invented or reconstructed.

Provider package/model commands are opt-in argv maps supplied through protected
deployment environment, for example a locally reviewed wrapper:

```sh
export ASTRUMWEAVER_RUNTIME_INSTALLERS_JSON='{"vllm":["/usr/local/sbin/install-reviewed-vllm"]}'
export ASTRUMWEAVER_RUNTIME_DOWNLOADERS_JSON='{"vllm":["/usr/local/sbin/fetch-reviewed-vllm-model"]}'
```

The driver appends the reviewed package reference or model reference as the
final argument. It does not invoke a shell or guess `apt`, `pip`, `curl`,
or another installer.

NixOS normally uses `services.astrumweaver.worker.runtime` instead. The
runtime provider and demand are persisted as an immutable Nix-store deployment
manifest and the runtime package is explicitly supplied in the service closure.

## SSH/tmux/mobile use

The interface intentionally uses a simple line-oriented terminal wizard instead
of a full-screen framework dependency.

This keeps it practical in:

- SSH
- tmux
- serial/limited terminals
- mobile terminal clients

All operations are keyboard-only and the package adds no TUI framework
dependency.


## NixOS first-run boundary

On NixOS, first-run mode does not rewrite an existing flake or
`configuration.nix`. It renders a deterministic module snippet for Control,
smoke Worker, RuntimeProvider Worker, or combined roles for review/import,
while preserving `nixos-rebuild` as the operator-owned mutation boundary.

RuntimeProvider-first generation requires the reviewed provider/demand and an
explicit runtime package attribute path supplied by the operator. It never
invents a package expression. The generated module must be imported together
with the AstrumWeaver NixOS module; host prerequisites and applying the reviewed
configuration remain operator responsibilities.

### NixOS runtime package input

For first-run RuntimeProvider setup on NixOS, enter a package attribute path
rooted in `pkgs`, such as `pkgs.ollama`, `pkgs.vllm`, or `pkgs.llama-cpp`.
Unknown roots (including `myPkgs`/`inputs`) and general Nix expressions are
rejected before the review/write confirmation. Custom packages can be exposed
under `pkgs` using an operator-maintained nixpkgs overlay. See
[generated module scope and validation](nix.md#generated-first-run-module-scope-and-validation)
for the supported grammar and the generated-module regression command.
