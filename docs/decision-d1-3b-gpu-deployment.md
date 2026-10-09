# Reproducible Liquid AI d1-3B CUDA System-One GPU test deployment

Tracking: #91, #114, #116, #117. This is **post-v0.1, test-only**.
It does not replace operator-led v0.1 Real Smoke #80, authorize any
production Worker reconfiguration, or qualify decision quality.

The native text-only decision head remains experimental: scores are
`choice_set_probability`, `uncalibrated`, and recommendation-only in
`shadow` mode. The historical OpenJev identity is never replaced.

## Exact reproducibility boundary

The locked Nix flake provides **two independent opt-in packages**:

| Nix package | Target | Initial observed topology |
| --- | --- | --- |
| `decision-llama-cuda-sm61` | CUDA 12.9, compute 6.1 | single GTX 1080 Ti |
| `decision-llama-cuda-sm86` | CUDA 12.9, compute 8.6 | two RTX 3060s |

Both package upstream llama.cpp at exact commit
`bd4eeaa047006cb1fe71999fbd11134b5836e167`, pinned by
NAR hash `sha256-R8eTkbeh47PDk/LMZUctWnrkivWCp7J0BczN6cLCBxY=`.
They do **not** use the incompatible llama.cpp 0.4.1 decision-head build,
manual host CMake overlays, mutable tags or an unreviewed nixpkgs revision.
CUDA unfree packages are enabled explicitly in the opt-in Nix scope.

Each runtime contains:

- `bin/llama-server` — Nix-built CUDA binary with pinned libraries and ELF loader;
- `bin/astrumweaver-llama-server` — the preferred managed launcher. It
  discovers host `libcuda.so.1`, creates a private *driver-only* loader
  directory, forwards termination to its owned child and removes the
  directory on exit. Host glibc/libstdc++ are never inserted;
- `share/astrumweaver/{llama-cpp-revision,cuda-sm,cuda-toolkit}` — immutable
  source, SM architecture and toolkit provenance verified by the CLI.

No package bundles or downloads a model, starts services, registers
deployments, or acquires GPUs.

## Test-host preflight

Run only on an authorized test GPU host or in a reviewed maintenance
window. Do **not** stop or replace the existing Worker/model endpoint.
Inspect exact GPU identity, memory and service health:

~~~sh
nvidia-smi --query-gpu=uuid,compute_cap,memory.total,memory.used --format=csv,noheader
systemctl is-active astrumweaver-worker.service
~~~

Use an operator-selected **free** loopback port. The probe refuses port
collisions, model hash mismatches, unsupported GPU counts/architectures,
and existing GPU load over 128 MiB unless explicit
`--acknowledge-existing-gpu-load` is given. The flag permits a carefully
reviewed **test**, not GPU reassignment or production resource sharing.

The NVIDIA host driver and GPU passthrough must already be working.
The test host must provide `nvidia-smi`, `ldconfig`, and Linux
`ss` (from `iproute2`) for fail-closed GPU and loopback-process-ownership
checks. AstrumWeaver does not install drivers, virtual machines or hypervisors.

## Pin the existing model separately

Prepare or locate the following **existing local GGUF**:

- Repo: `LiquidAI/d1-3B-GGUF`
- Exact revision: `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`
- File: `d1-3B-Q4_K_M.gguf`
- SHA-256: `16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402`

Do not silently substitute a different model or quantization.

~~~sh
export MODEL_FILE=/private/existing/d1-3B-Q4_K_M.gguf
printf '%s  %s\n' \
  16aff27ea2eefdc32b9897f43854a5d3170c1dc8dccb9c756905af30a4e22402 \
  "$MODEL_FILE" | sha256sum --check --status
~~~

The model remains outside the package and is never implicitly copied
to the Nix store.

## Build and run a disposable native test

From the reviewed AstrumWeaver source checkout carrying this package:

~~~sh
# On a single SM61 GPU test host:
RUNTIME=$(nix build --no-link --print-out-paths \
  .#decision-llama-cuda-sm61)

# OR on a two-GPU SM86 test host:
RUNTIME=$(nix build --no-link --print-out-paths \
  .#decision-llama-cuda-sm86)

test -x "$RUNTIME/bin/llama-server"
test -x "$RUNTIME/bin/astrumweaver-llama-server"
"$RUNTIME/bin/astrumweaver-llama-server" --version
~~~

This builds only the package; **no model or Worker is started**.
The probe is available from the packaged AstrumWeaver CLI
`astrumweaver-d1-gpu-smoke`, or via the checkout's Python module
`astrumweaver.validation.d1_gpu_smoke`.

~~~sh
export EVIDENCE_DIR=/private/isolated-evidence
install -d -m 700 "$EVIDENCE_DIR"

astrumweaver-d1-gpu-smoke \
  --package "$RUNTIME" \
  --model "$MODEL_FILE" \
  --gpu-count 1 \
  --port 18211 \
  --evidence "$EVIDENCE_DIR/d1-native-sm61.json"
~~~

For two GPUs use `--gpu-count 2` and a new evidence path and free port.
The port above is an *example*, not reserved. If an existing endpoint is
using the GPU, add `--acknowledge-existing-gpu-load` only after confirming
that coexistence is safe and sufficient VRAM remains. This native probe
does not coordinate with the Worker scheduler or reserve GPU ownership.

The probe performs:

1. GPU count/capability/VRAM preflight and exact pinned package provenance;
   the package **must resolve to a top-level `/nix/store` output**. A
   writable directory containing forged metadata is rejected;
2. streaming GGUF SHA-256 verification **before starting a subprocess**;
3. temporary loopback llama-server (`--offline --no-webui`, 4096 context,
   concurrency 1, all GPU layers, `--fit off`);
4. confirmation that **all** loopback listeners on the selected port
   belong to the owned subprocess group before any HTTP request, then `/health` and
   `/v1/models` text-input/native-decisions output validation;
   the native HTTP client ignores ambient proxies, rejects redirects
   and limits JSON responses to 1 MiB;
5. measured incremental VRAM on **every** selected GPU, five synthetic
   choice requests and a reversed-choice ordering probe, finite normalized
   scores and zero output tokens;
6. bounded termination of the entire owned process group, including
   cases where the launcher parent exits before the GPU child, and
   port-release checks even on error.

The subprocess inherits only a minimal environment plus the selected GPU
IDs and driver-library path; ambient Worker/Client bearer tokens and cloud
secrets are never forwarded into the smoke-owned inference process.

Result is a **new** mode-0600 JSON file. It contains only aggregate
measurements and public artifact digests, not GPU UUIDs, URLs, paths,
hostnames, credentials, prompts, or server logs.

A protocol PASS is **not** Control/Worker acceptance, model correctness,
or calibration. Earlier five-case evidence was only 2/5 correct, and
the deliberate unknown was not abstained on (0/1). Keep model quality
`OBSERVED_ONLY`.

## Worker integration: operator-reviewed separate step

Only after native success, separately approve immutable
deployment/profile/contract/semantics identities and reserve the selected
GPUs. For a NixOS Worker, include the exact package under
`runtime.packages` and point `providerConfig.executable` explicitly to the
**driver-only wrapper**:

~~~nix
let
  d1Runtime = inputs.astrumweaver.packages.${pkgs.system}.decision-llama-cuda-sm61;
in
{
  # Fragment within the reviewed services.astrumweaver.worker.runtime:
  packages = [ d1Runtime ];
  providerConfig.executable =
    "${d1Runtime}/bin/astrumweaver-llama-server";
}
~~~

For generic systemd, put the same pinned Nix package in the approved
runtime closure and use the wrapper path in a **new** reviewed
`RuntimeDeploymentSpec`. Use the existing `astrumweaver-setup-tui` /
`SetupPlan` approval path for any host mutations. The launcher respects
the Worker-scoped `CUDA_VISIBLE_DEVICES` and never grants GPU ownership.

The dedicated launcher also strips ambient process credentials before
spawning llama-server. This is essential when the Worker Provider passes
its own environment to a managed runtime: the child receives only
the GPU visibility, driver search path, and necessary basic OS variables,
**not** Worker/Client bearer tokens or cloud API credentials.

The driver shim prefers the standard NixOS
`/run/opengl-driver/lib/libcuda.so.1` and then well-known generic Linux
driver paths. For a different verified host, set
`ASTRUMWEAVER_LIBCUDA_SO=/approved/driver/libcuda.so.1` in the runtime
environment; never use CUDA toolkit `stubs/`. That is a host driver
**runtime input**, not a build-time patch or permission escalation.

Do not reuse a test-only single-/multi-GPU profile digest, mutate a
historical model revision, or cut over the existing Worker. Run any
PostgreSQL-backed shadow acceptance against a disposable, separately
owned Control, database, Worker and ports before a human-reviewed cutover.

## Limitations and failure handling

Only CUDA 12.9 and explicitly pinned SM61/SM86 have packages. This
does not qualify arbitrary drivers, multi-tenant scheduling, vision/mmproj,
production decision authority, cost savings or #97 direct-remote coding
comparison. On failures, **leave existing services intact** and collect
private diagnostic evidence rather than disabling the decision-head check
or silently falling back to CPU.
