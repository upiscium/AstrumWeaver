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
backend when requested.

## First-run flow

```text
local host discovery
    ↓
choose role: Control / Worker / Control+Worker
    ↓
Control phase (when selected)
  - Control bind settings
  - hidden PostgreSQL URL input
  - generate or enter client/Worker authority tokens
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

The previous Worker/runtime wizard remains available:

```sh
sudo /nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-setup-tui \
  --mode runtime
```

Its flow remains:

```text
NVIDIA GPU discovery
    ↓
choose Worker GPU ownership
    ↓
Worker identity/class
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
    ↓
dry-run / exact plan confirmation / apply
```

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

The user explicitly chooses which locally visible GPUs form the Worker. The
resulting WorkerSpec preserves that UUID order and calculates:

- GPU count
- total VRAM
- maximum single-device VRAM
- minimum known compute capability

The compute capability is represented as the generic Worker label:

```text
gpu.compute_capability.min
```

GPU UUIDs are not copied into SetupHostSnapshot metadata.

When the user selects fewer GPUs than the local NVIDIA inventory exposes, the
generic-systemd first-run Worker config is rendered with:

```toml
gpu_preflight = true
gpu_preflight_mode = "isolated-access"
gpu_device_map = "/etc/astrumweaver/gpu-device-map"
```

Setup then performs the transient effective-access probe before the candidate
Worker config is persisted. If an unselected physical GPU remains openable,
the wizard fails closed and instructs the operator to narrow guest-visible GPU
exposure at the VM/LXC/hypervisor boundary. It does not fall back to
`CUDA_VISIBLE_DEVICES`-only isolation and does not silently expand the Worker
GPU set.


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

completes host/Worker/demand/runtime selection and produces the exact reviewed
SetupPlan, then stops without mutation.

On an already-integrated generic systemd Worker host, the first-party driver
can be connected with:

```sh
sudo /nix/var/nix/profiles/astrumweaver-installer/bin/astrumweaver-setup-tui \
  --mode runtime \
  --driver astrumweaver.setup.systemd:create_systemd_driver
```

The Worker service account, Worker TOML, protected token EnvironmentFile, and
systemd unit must already exist. The runtime driver owns reviewed runtime
package/config/model actions and starts the existing Worker service; it does not
invent Control credentials or network identity.

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
`configuration.nix`. For smoke-mode Control/Worker bootstrap it can render a
deterministic module snippet for review/import while preserving
`nixos-rebuild` as the operator-owned mutation boundary.

RuntimeProvider-first snippet generation currently fails closed rather than
inventing a Nix package expression for the selected runtime. Configure the
reviewed `services.astrumweaver.worker.runtime` block declaratively after the
base snippet, or use runtime mode for planning.
