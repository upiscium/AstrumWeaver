# Interactive Worker/runtime Setup TUI

AstrumWeaver provides a keyboard-first terminal setup wizard:

```sh
astrumweaver-setup-tui
```

The TUI is a frontend over the same provider-neutral runtime contracts and
deterministic SetupPlan backend used by non-interactive automation. It does not
execute runtime-specific shell commands itself.

## Flow

```text
local host discovery
    ↓
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
compatibility re-check
    ↓
deterministic SetupPlan
    ↓
dry-run through SetupActionDriver
    ↓
exact plan confirmation
    ↓
shared apply_plan()
    ↓
progress/result/recovery guidance
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

The TUI remains usable without deployment authority. Without a driver, running:

```sh
astrumweaver-setup-tui
```

completes host/Worker/demand/runtime selection and produces the exact reviewed
SetupPlan, then stops without mutation.

On an already-integrated generic systemd Worker host, the first-party driver
can be connected with:

```sh
sudo astrumweaver-setup-tui \
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
