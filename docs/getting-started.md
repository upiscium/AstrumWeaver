# Getting Started

This guide takes a new AstrumWeaver installation from **nothing running** to a
verified Control + Worker + completed job.

If you have not installed the packages/services yet, start with
[Installation](installation.md).

## What you are building

The smallest useful deployment is:

```text
PostgreSQL
    ↑
Control :9000
    ↑
Worker :9100 local health
    ↑
built-in debug.echo executor
```

For the first successful deployment, use `debug.echo` rather than an LLM
RuntimeProvider. Once this path works, add Ollama, llama.cpp, vLLM, FreeToken,
or ExLlamaV3 separately.

This gives a clean answer to:

> Is AstrumWeaver itself installed and able to schedule work?

## 1. Check prerequisites

Control requires:

- PostgreSQL reachable through `ASTRUMWEAVER_DATABASE_URL`
- `ASTRUMWEAVER_CLIENT_TOKEN`
- `ASTRUMWEAVER_WORKER_TOKEN`

Worker requires:

- network access to Control
- the same `ASTRUMWEAVER_WORKER_TOKEN`

The client and Worker authority tokens must be different.

For a GPU Worker, `nvidia-smi` must already work.

## 2. Start Control

How Control is installed depends on the host:

- NixOS: `services.astrumweaver.control.enable = true`
- generic systemd: `astrumweaver-setup-control-plane`

See [Installation](installation.md).

After startup:

```sh
systemctl status astrumweaver-control.service
```

Check process health:

```sh
curl -fsS http://127.0.0.1:9000/v1/health
```

Expected shape:

```json
{"status":"ok","protocol_version":"v1"}
```

Then check readiness:

```sh
curl -fsS http://127.0.0.1:9000/v1/ready
```

Expected shape:

```json
{"ready":true,"protocol_version":"v1"}
```

If `/v1/health` works but `/v1/ready` returns HTTP 503, check:

1. PostgreSQL is reachable.
2. `ASTRUMWEAVER_DATABASE_URL` is correct.
3. all packaged migrations have been applied.

For an explicit migration:

```sh
ASTRUMWEAVER_DATABASE_URL='postgresql://...' astrumweaver-migrate
```

## 3. Start one Worker

For the first test, advertise only:

```text
debug.echo
```

and use:

```text
astrumweaver.executors.structured_echo:create_executor
```

The NixOS and generic-systemd examples are in
[Installation](installation.md).

Check the service:

```sh
systemctl status astrumweaver-worker.service
```

Check Worker health on the Worker host:

```sh
curl -fsS http://127.0.0.1:9100/health
```

A successful Worker should eventually report:

```json
{
  "status": "ok",
  "registered": true,
  "ready": true,
  "draining": false
}
```

The real response also contains `worker_id` and `active_job_id`.

If the Worker is running but `registered` is false, check:

- Worker `control_url`
- `ASTRUMWEAVER_WORKER_TOKEN`
- Control readiness
- network/TLS reachability

For a GPU Worker, startup also fails closed when its configured GPU ownership
does not match the deployment/isolation contract.

## Verify the installation

At this point:

- Control is ready
- Worker is registered and ready
- Worker advertises `debug.echo`

Submit a real job.

Set the Control URL and client token locally:

```sh
export ASTRUMWEAVER_CONTROL_URL='http://127.0.0.1:9000'
export ASTRUMWEAVER_CLIENT_TOKEN='REPLACE_WITH_CLIENT_TOKEN'
```

Submit:

```sh
response="$(
  curl -fsS \
    -H "Authorization: Bearer $ASTRUMWEAVER_CLIENT_TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{
      "protocol_version": "v1",
      "capability": "debug.echo",
      "payload": {
        "message": "hello from AstrumWeaver"
      },
      "requirements": {},
      "priority": 0,
      "max_attempts": 1
    }' \
    "$ASTRUMWEAVER_CONTROL_URL/v1/jobs"
)"

printf '%s\n' "$response"
```

With `jq`:

```sh
job_id="$(printf '%s' "$response" | jq -r .job_id)"
```

Inspect the job:

```sh
curl -fsS \
  -H "Authorization: Bearer $ASTRUMWEAVER_CLIENT_TOKEN" \
  "$ASTRUMWEAVER_CONTROL_URL/v1/jobs/$job_id" | jq
```

After the Worker claims it, the terminal state should become:

```text
succeeded
```

and the result should contain the submitted payload under the structured echo
output.

If this succeeds, the basic AstrumWeaver installation is working end to end.

## 4. Move from smoke test to a RuntimeProvider

Only after the `debug.echo` path succeeds, choose the model/runtime policy.

AstrumWeaver currently has first-class providers for:

- Ollama
- llama.cpp
- vLLM
- FreeToken
- ExLlamaV3

Read:

- [Runtime Providers and Execution Demand](runtime-providers.md)
- [Interactive Worker/runtime Setup TUI](setup-tui.md)
- [Nix Packaging and NixOS Modules](nix.md#first-class-runtimeprovider-execution)

The important rule is:

> Runtime selection is explicit. AstrumWeaver does not silently substitute a
> different provider when your selected provider is incompatible.

### NixOS

Use:

```nix
services.astrumweaver.worker.runtime = {
  enable = true;
  provider = "ollama"; # example
  packages = [ ... ];

  modelRef = "...";
  modelFormat = "ollama";
  modelTopology = "dense";
  residencyPolicy = "prefer_vram";
};
```

Remove the smoke-test `executorFactory`; the RuntimeProvider path and manual
executor path are mutually exclusive.

### Generic systemd

The interactive setup frontend is:

```sh
sudo astrumweaver-setup-tui \
  --driver astrumweaver.setup.systemd:create_systemd_driver
```

The TUI shows compatibility results, asks you to explicitly choose a provider,
builds a deterministic SetupPlan, shows the exact digest, performs a dry run,
and applies only explicitly authorized actions.

Package/model installation commands are not guessed automatically. See
[Interactive Worker/runtime Setup TUI](setup-tui.md) for the required
deployment-driver environment when an action needs an installer/downloader.

## 5. GPU subset deployments

If a Worker owns only some GPUs visible on the host, do not disable the
exact-set checks.

AstrumWeaver supports a reviewed systemd device-cgroup isolation path and then
runs the existing exact GPU-set check inside the restricted Worker service.

Read:

[Runtime Deployment GPU Isolation Acceptance](runtime-deployment-acceptance.md)

For #31 acceptance, the repository also includes:

```sh
astrumweaver-runtime-deployment-accept --help
```

## Troubleshooting order

When the first deployment fails, check in this order:

1. `systemctl status astrumweaver-control`
2. Control `/v1/health`
3. Control `/v1/ready`
4. `systemctl status astrumweaver-worker`
5. Worker `/health`
6. Worker token/control URL
7. GPU preflight/isolation, if applicable
8. RuntimeProvider health, only after the base Worker path is known-good

Useful logs:

```sh
journalctl -u astrumweaver-control.service -b
journalctl -u astrumweaver-worker.service -b
```

## Where to go next

- install/update packages: [Installation](installation.md)
- host/systemd details: [Existing-Node Deployment](deployment.md)
- NixOS reference: [Nix Packaging and NixOS Modules](nix.md)
- API/auth/protocol: [Control / Worker Protocol v1](protocol-v1.md)
- runtime selection: [Runtime Providers and Execution Demand](runtime-providers.md)
- interactive runtime setup: [Interactive Worker/runtime Setup TUI](setup-tui.md)
- GPU subset acceptance: [Runtime Deployment GPU Isolation Acceptance](runtime-deployment-acceptance.md)
