# Borrowable GPU Worker

AstrumWeaver can use a GPU inside an existing development Linux node without permanently dedicating that node to the compute fabric.

The ownership model is explicit:

```text
development
    ↕
draining
    ↕
astrumweaver
```

This is an **exclusive ownership handoff**, not concurrent GPU sharing.

## Safety model

AstrumWeaver does not attempt to partition VRAM or coordinate arbitrary development processes.

The mode switch succeeds only when the relevant transition can be verified:

### AstrumWeaver → development

1. request `DRAINING` without terminating the Worker
2. stop local job claiming immediately
3. allow the active job to finish normally
4. wait until the Worker reports no active job
5. stop the Worker service
6. verify the configured GPU UUID set is still exactly visible
7. verify no NVIDIA C/G process context remains on those GPUs
8. return success to the operator

The default drain timeout is **unbounded**. AstrumWeaver therefore does not force a long-running job merely because an operator requested development mode.

A finite drain timeout may be configured. Expiry fails the mode switch and leaves the Worker service running/draining; it does not forcibly stop the job.

### development → AstrumWeaver

1. verify the exact configured GPU UUID set
2. reject the transition if an existing development/unrelated GPU process is present
3. start the Worker service
4. repeat the normal Worker GPU UUID preflight
5. register with Control
6. wait for local Worker readiness
7. return success to the operator

The Worker returns ONLINE through its normal registration path. There is no special scheduler-side re-registration workaround.

## Drain signal

The Worker daemon accepts:

```text
SIGUSR1
```

as a non-terminating drain request.

The signal immediately prevents new local claims and asynchronously mirrors the Worker to durable `DRAINING` state in Control.

This differs from `SIGTERM`:

- `SIGUSR1`: drain, remain alive
- `SIGTERM`: graceful process shutdown

The borrowable mode controller uses `SIGUSR1` first and only stops the service after the active job has disappeared.

## GPU release verification

The controller uses two NVIDIA checks.

First, exact identity is revalidated using GPU UUIDs.

Second, `nvidia-smi pmon -c 1` is inspected for process contexts on the GPU indexes corresponding to the configured UUID set.

C, G, and combined process types are therefore treated as ownership blockers.

If process inspection itself fails, the transition fails closed.

Persistence/driver management state is not treated as a development workload merely because the driver remains loaded; the gate is concerned with active GPU process contexts.

## Generic CLI

The packaged command is:

```text
astrumweaver-worker-mode
```

Example:

```sh
sudo astrumweaver-worker-mode development \
  --gpu-uuid GPU-example-a \
  --service astrumweaver-worker.service
```

Return the GPU to AstrumWeaver:

```sh
sudo astrumweaver-worker-mode astrumweaver \
  --gpu-uuid GPU-example-a \
  --service astrumweaver-worker.service
```

Inspect local mode:

```sh
sudo astrumweaver-worker-mode status \
  --gpu-uuid GPU-example-a
```

The command prints a JSON report containing:

- inferred local mode
- systemd Worker activity
- Worker readiness/draining state
- active job ID when present
- active GPU process contexts

The generic CLI intentionally requires explicit GPU UUIDs so it remains independent of site inventory.

## NixOS integration

For a development node:

```nix
services.astrumweaver.worker = {
  enable = true;

  workerId = "worker-example";
  workerClass = "modern-single";
  controlUrl = "https://control.example.invalid";
  capabilities = [ "llm.chat" ];

  gpuUuids = [ "GPU-example" ];
  totalVramMb = 16384;
  maxSingleGpuVramMb = 16384;
  nvidiaSmiPackage = config.hardware.nvidia.package;

  executorFactory = "my_executor:create_executor";

  borrowable.enable = true;

  # 0 = wait indefinitely for the current job.
  borrowable.drainTimeoutSeconds = 0;
  borrowable.startTimeoutSeconds = 60;

  environmentFile = "/run/secrets/astrumweaver-worker.env";
};
```

The module adds:

```text
astrumweaver-gpu-mode
```

with the configured service name, GPU UUIDs, NVIDIA tooling, and local health endpoint already bound into the wrapper.

Usage becomes:

```sh
sudo astrumweaver-gpu-mode development
sudo astrumweaver-gpu-mode astrumweaver
sudo astrumweaver-gpu-mode status
```

Borrowable mode requires the Worker health endpoint to remain local (`127.0.0.1` or `localhost`).

## Development-mode contract

After a successful transition to `development`:

- `astrumweaver-worker.service` is inactive
- no AstrumWeaver job can be claimed on that Worker
- exact GPU identity was observed
- no GPU process context was present at the handoff boundary

After the command returns, the development environment may start using the GPU.

AstrumWeaver cannot prevent an unrelated local process from racing the handoff after verification. Operators should therefore treat `astrumweaver-gpu-mode` as the ownership boundary and avoid launching development GPU processes concurrently with a transition.

## Non-goals

Borrowable Worker mode does not provide:

- concurrent AstrumWeaver/development GPU sharing
- MIG or VRAM partitioning
- forced termination of arbitrary GPU processes
- automatic killing of development workloads
- Proxmox passthrough changes
- VM/LXC lifecycle management
- a distributed host lock for uncooperative local processes

The goal is a safe, observable handoff on a node where both modes are under the operator's control.
