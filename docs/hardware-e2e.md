# Hardware E2E Acceptance

This runbook completes the only v0.1 Epic criterion that public CI cannot prove:

> one real existing GPU node can enroll, execute work, drain/release the GPU, return ONLINE, and execute work again.

The validation is intentionally designed so private site details remain local.

## What is private

The command needs local deployment values such as:

- Control URL
- Client token
- actual GPU UUIDs
- systemd/NVIDIA command paths when non-default

These values are used during validation but are **not written into the generated evidence**.

Do not paste them into a public Issue, PR, or chat.

## What is public-safe

The generated evidence contains only:

- date
- AstrumWeaver git revision
- deployment path: `nixos` or `systemd`
- generic profile class
- GPU count
- PASS/FAIL contract gates
- zero/nonzero process-context count at the release boundary
- confirmation that private values were omitted

It does not contain:

- hostname
- IP address
- VMID/CTID
- bridge/VLAN
- actual GPU UUID
- credentials/token
- private Control URL
- GPU model
- model/data path

## Validation Worker

Use one real existing Linux GPU node.

For the cleanest acceptance run, configure the Worker with a harmless capability such as the built-in structured echo executor:

```toml
[worker]
id = "local-private-worker-id"
class = "modern-single"
control_url = "https://private-control.example"
capabilities = ["debug.echo"]
gpu_uuids = ["GPU-private-uuid"]
gpu_count = 1
total_vram_mb = 16384
max_single_gpu_vram_mb = 16384
max_concurrency = 1

[executor]
factory = "astrumweaver.executors.structured_echo:create_executor"

[executor.settings]
```

The values above are examples. Do not copy a private identity into the repository.

The validation job is pinned by the locally supplied GPU UUID set, so another Worker cannot satisfy the job accidentally.

## Preconditions

Before running:

- the Linux node already exists
- GPU passthrough/device exposure is already configured outside AstrumWeaver
- `nvidia-smi` works
- AstrumWeaver Worker package/service is installed
- Control + PostgreSQL are already operational
- Worker token is configured in the Worker service environment
- the chosen validation capability is advertised by this Worker
- no development GPU process is running when beginning the test

No Proxmox API access is required.

## Run

Set the **Client** authority locally:

```sh
export ASTRUMWEAVER_CLIENT_TOKEN='...'
```

Then run:

```sh
sudo --preserve-env=ASTRUMWEAVER_CLIENT_TOKEN \
  astrumweaver-hardware-accept \
  --control-url 'https://private-control.example' \
  --gpu-uuid 'GPU-private-uuid' \
  --revision '<public-git-commit-sha>' \
  --deployment-path nixos \
  --profile-class modern-single \
  --capability debug.echo \
  --evidence ./hardware-e2e.md
```

For a multi-GPU Worker, repeat `--gpu-uuid` once per GPU.

For generic non-NixOS systemd deployment:

```text
--deployment-path systemd
```

## What the harness does

The harness performs:

```text
exact local GPU UUID preflight
        ↓
Worker ONLINE/ready
        ↓
generic job pinned to this GPU set → SUCCEEDED
        ↓
SIGUSR1 drain
        ↓
DRAINING observed
        ↓
submit second pinned job
        ↓
verify it remains QUEUED while DRAINING
        ↓
cancel probe job
        ↓
wait current job empty
        ↓
stop Worker service
        ↓
verify GPU process contexts = 0
        ↓
development ownership reached
        ↓
exact UUID + no conflicting process
        ↓
start Worker
        ↓
ONLINE/ready
        ↓
second generic pinned job → SUCCEEDED
        ↓
write redacted evidence
```

The harness does not force-kill a long-running job. A finite drain timeout may fail the acceptance run, but does not convert that into forced termination.

## Evidence handling

Review the generated Markdown before committing it.

A successful file looks structurally like:

```md
# AstrumWeaver v0.1 Hardware E2E Evidence

| Field | Result |
| --- | --- |
| Evidence version | v0.1 |
| Date (UTC) | 2026-... |
| AstrumWeaver revision | ... |
| Deployment path | nixos |
| Profile class | modern-single |
| GPU count | 1 |
| Private values omitted | true |
| Exact UUID preflight | PASS |
| Worker registration / readiness | PASS |
| First generic job round-trip | PASS |
| DRAINING observed | PASS |
| No new claim while DRAINING | PASS |
| Worker service stopped after drain | PASS |
| GPU process contexts after release | 0 |
| Development ownership reached | PASS |
| Worker returned ONLINE | PASS |
| Second generic job round-trip | PASS |
| Overall | PASS |
```

Only this redacted evidence should be committed for #20.

Do not commit command history, raw `nvidia-smi` output, environment files, systemd environment dumps, or HTTP traces.

## Failure behavior

The harness fails closed when it cannot verify:

- exact GPU identity
- Control API behavior
- Worker readiness
- no-new-claim during drain
- service stop
- zero GPU process contexts
- successful return to ONLINE

Failures are reported generically and do not print tokens or private URLs.

## Acceptance completion

After one real node produces redacted PASS evidence:

1. commit only the generated evidence
2. attach it to #20
3. mark #20 complete
4. mark the final Epic #1 criterion complete
5. close Epic #1 as v0.1 architecture accepted
