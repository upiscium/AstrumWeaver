from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from astrumweaver import AcceleratorDevice, ResourceShape, WorkerSpec
from astrumweaver.setup.first_run import (
    ControlBootstrapSpec,
    FirstRunExecutionMode,
    FirstRunRole,
    FirstRunSecrets,
    SystemdFirstRunInstaller,
    generate_authority_tokens,
    render_control_env,
    render_control_toml,
    render_nixos_bootstrap_snippet,
    render_worker_env,
    render_worker_toml,
)


def worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="worker-test",
        worker_class="gpu-single",
        resources=ResourceShape(
            gpu_count=1,
            total_vram_mb=24576,
            max_single_gpu_vram_mb=24576,
        ),
        gpu_uuids=("GPU-private-a",),
        accelerators=(
            AcceleratorDevice(
                uuid="GPU-private-a",
                memory_mb=24576,
                compute_capability="8.6",
                device_class="NVIDIA Test GPU",
            ),
        ),
        capabilities=frozenset({"llm.chat", "text.generate"}),
        labels={"gpu.compute_capability.min": "8.6"},
    )


def test_generated_authority_tokens_are_distinct_and_high_entropy() -> None:
    client, worker_token = generate_authority_tokens()

    assert client != worker_token
    assert len(client) == 64
    assert len(worker_token) == 64
    int(client, 16)
    int(worker_token, 16)


def test_control_rendering_keeps_secret_values_out_of_toml() -> None:
    spec = ControlBootstrapSpec(bind_host="0.0.0.0", port=9000)
    secrets = FirstRunSecrets(
        database_url="postgresql://secret-user:secret-pass@db/astrumweaver",
        client_token="client-secret",
        worker_token="worker-secret",
    )

    toml_text = render_control_toml(spec)
    env_text = render_control_env(secrets)

    parsed = tomllib.loads(toml_text)
    assert parsed["control"]["host"] == "0.0.0.0"
    assert parsed["control"]["port"] == 9000
    assert "secret-pass" not in toml_text
    assert "client-secret" not in toml_text
    assert "worker-secret" not in toml_text

    assert "ASTRUMWEAVER_DATABASE_URL=" in env_text
    assert "ASTRUMWEAVER_CLIENT_TOKEN=client-secret" in env_text
    assert "ASTRUMWEAVER_WORKER_TOKEN=worker-secret" in env_text


def test_worker_toml_is_generated_from_discovered_worker_contract() -> None:
    text = render_worker_toml(
        worker(),
        control_url="http://control.internal:9000",
        execution_mode=FirstRunExecutionMode.RUNTIME,
    )
    parsed = tomllib.loads(text)

    assert parsed["worker"]["id"] == "worker-test"
    assert parsed["worker"]["gpu_uuids"] == ["GPU-private-a"]
    assert parsed["worker"]["gpu_count"] == 1
    assert parsed["worker"]["accelerators"][0]["memory_mb"] == 24576
    assert (
        parsed["worker"]["labels"]["gpu.compute_capability.min"]
        == "8.6"
    )
    assert parsed["runtime"]["manifest"] == (
        "/etc/astrumweaver/runtime-deployment.json"
    )
    assert "executor" not in parsed


def test_smoke_worker_toml_uses_built_in_echo_executor() -> None:
    text = render_worker_toml(
        worker(),
        control_url="http://control.internal:9000",
        execution_mode=FirstRunExecutionMode.SMOKE,
    )
    parsed = tomllib.loads(text)

    assert parsed["executor"]["factory"] == (
        "astrumweaver.executors.structured_echo:create_executor"
    )
    assert "runtime" not in parsed


def test_nixos_runtime_snippet_embeds_reviewed_provider_demand_and_package() -> None:
    deployment = {
        "provider_id": "vllm",
        "provider_config": {
            "gpu_memory_utilization": 0.9,
        },
        "demand": {
            "model": {
                "model_ref": "/srv/models/example",
                "model_format": "safetensors",
                "topology": "dense",
                "estimated_size_mb": 16000,
                "metadata": {},
            },
            "residency_policy": "vram_only",
            "gpu_topology": "single_gpu",
            "min_gpu_count": 0,
            "min_total_vram_mb": 12000,
            "min_single_gpu_vram_mb": 12000,
            "min_host_ram_mb": 0,
            "preferred_host_ram_mb": 0,
            "metadata": {},
        },
    }

    snippet = render_nixos_bootstrap_snippet(
        role=FirstRunRole.WORKER,
        worker=worker(),
        control_url="http://control.internal:9000",
        execution_mode=FirstRunExecutionMode.RUNTIME,
        runtime_deployment=deployment,
        runtime_package_expression="pkgs.vllm",
    )

    assert "runtime = {" in snippet
    assert 'provider = "vllm";' in snippet
    assert "packages = [ pkgs.vllm ];" in snippet
    assert 'modelRef = "/srv/models/example";' in snippet
    assert 'residencyPolicy = "vram_only";' in snippet
    assert "providerConfig = builtins.fromJSON" in snippet
    assert "executorFactory" not in snippet


def test_nixos_snippet_uses_nix_lists_without_json_commas() -> None:
    snippet = render_nixos_bootstrap_snippet(
        role=FirstRunRole.WORKER,
        worker=worker(),
        control_url="http://control.internal:9000",
    )

    assert 'capabilities = [ "llm.chat" "text.generate" ];' in snippet
    assert 'gpuUuids = [ "GPU-private-a" ];' in snippet
    assert '["llm.chat", "text.generate"]' not in snippet
    assert "executorFactory" in snippet


@dataclass
class RunCall:
    args: list[str]
    env: dict[str, str] | None


class RecordingInstaller(SystemdFirstRunInstaller):
    def __init__(self) -> None:
        super().__init__(tool_dir=None)
        self.calls: list[RunCall] = []
        self.ready_urls: list[tuple[str, bool]] = []

    def _require_root(self) -> None:
        return None

    def _resolve_tool(self, name: str) -> str:
        return f"/tools/{name}"

    def _run(self, args, *, env=None) -> None:
        self.calls.append(
            RunCall(
                args=list(args),
                env=None if env is None else dict(env),
            )
        )

    def _wait_json_ready(
        self,
        url: str,
        *,
        require_registered: bool = False,
    ) -> bool:
        self.ready_urls.append((url, require_registered))
        return True


def test_resolve_tool_prefers_invoking_profile_bin_without_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_bin = tmp_path / "profile" / "bin"
    profile_bin.mkdir(parents=True)
    tui = profile_bin / "astrumweaver-setup-tui"
    tui.write_text("#!/bin/sh\n", encoding="utf-8")
    tui.chmod(0o755)

    helper_names = (
        "astrumweaver-setup-control-plane",
        "astrumweaver-setup-gpu-worker",
        "astrumweaver-control",
        "astrumweaver-worker",
        "astrumweaver-migrate",
    )
    for name in helper_names:
        helper = profile_bin / name
        helper.write_text("#!/bin/sh\n", encoding="utf-8")
        helper.chmod(0o755)

    monkeypatch.setattr(sys, "argv", [str(tui)])
    monkeypatch.setenv("PATH", "")

    installer = SystemdFirstRunInstaller(tool_dir=None)

    for name in helper_names:
        assert installer._resolve_tool(name) == str(profile_bin / name)


def test_control_bootstrap_orders_install_migration_start_readiness() -> None:
    installer = RecordingInstaller()
    secrets = FirstRunSecrets(
        database_url="postgresql://secret@db/astrumweaver",
        client_token="client",
        worker_token="worker",
    )

    result = installer.install_control(
        spec=ControlBootstrapSpec(bind_host="0.0.0.0", port=9000),
        secrets=secrets,
    )

    assert result.control_installed
    assert result.migration_applied
    assert result.control_ready

    assert installer.calls[0].args[0] == (
        "/tools/astrumweaver-setup-control-plane"
    )
    assert installer.calls[1].args == ["/tools/astrumweaver-migrate"]
    assert installer.calls[1].env is not None
    assert installer.calls[1].env["ASTRUMWEAVER_DATABASE_URL"] == (
        "postgresql://secret@db/astrumweaver"
    )
    assert installer.calls[2].args == [
        "systemctl",
        "enable",
        "--now",
        "astrumweaver-control.service",
    ]
    assert installer.ready_urls == [
        ("http://127.0.0.1:9000/v1/ready", False)
    ]


def test_worker_bootstrap_passes_discovered_gpu_and_runtime_manifest() -> None:
    installer = RecordingInstaller()

    result = installer.install_worker(
        worker_toml=render_worker_toml(
            worker(),
            control_url="http://control:9000",
            execution_mode=FirstRunExecutionMode.RUNTIME,
        ),
        worker_token="worker-secret",
        runtime_manifest_json='{"schema_version":"v1"}\n',
        start=False,
    )

    assert result.worker_installed
    assert not result.worker_ready
    assert len(installer.calls) == 1
    args = installer.calls[0].args
    assert args[0] == "/tools/astrumweaver-setup-gpu-worker"
    assert "--gpu-uuid" not in args
    assert "--runtime-manifest" in args
    assert "--start" not in args
    assert installer.ready_urls == []


def test_worker_smoke_bootstrap_starts_and_waits_for_registered_health() -> None:
    installer = RecordingInstaller()

    result = installer.install_worker(
        worker_toml=render_worker_toml(
            worker(),
            control_url="http://control:9000",
            execution_mode=FirstRunExecutionMode.SMOKE,
        ),
        worker_token="worker-secret",
        start=True,
    )

    assert result.worker_installed
    assert result.worker_ready
    args = installer.calls[0].args
    assert "--start" in args
    assert installer.ready_urls == [
        ("http://127.0.0.1:9100/health", True)
    ]


def test_first_run_secrets_repr_does_not_expose_values() -> None:
    secrets = FirstRunSecrets(
        database_url="postgresql://private",
        client_token="client-private",
        worker_token="worker-private",
    )

    rendered = repr(secrets)
    assert "postgresql://private" not in rendered
    assert "client-private" not in rendered
    assert "worker-private" not in rendered


def test_worker_env_contains_only_worker_authority() -> None:
    env = render_worker_env("worker-secret")
    assert env == "ASTRUMWEAVER_WORKER_TOKEN=worker-secret\n"
    assert "CLIENT" not in env
    assert "DATABASE" not in env
