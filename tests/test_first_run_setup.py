from __future__ import annotations

import os
import sys
import tomllib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from astrumweaver import AcceleratorDevice, ResourceShape, WorkerSpec
from astrumweaver.control import ClientAuthMode
from astrumweaver.setup.first_run import (
    ControlBootstrapSpec,
    control_url_for_bind_host,
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


def worker(*, smoke: bool = False) -> WorkerSpec:
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
        capabilities=frozenset({"debug.echo"} if smoke else {"llm.chat", "text.generate"}),
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
    assert 'ASTRUMWEAVER_CLIENT_TOKEN="client-secret"' in env_text
    assert 'ASTRUMWEAVER_WORKER_TOKEN="worker-secret"' in env_text


def test_control_client_auth_none_omits_client_secret_from_env() -> None:
    spec = ControlBootstrapSpec(
        client_auth=ClientAuthMode.NONE,
    )
    secrets = FirstRunSecrets(
        database_url="postgresql://secret@db/astrumweaver",
        worker_token="worker-secret",
    )

    toml_text = render_control_toml(spec)
    env_text = render_control_env(
        secrets,
        client_auth=spec.client_auth,
    )

    parsed = tomllib.loads(toml_text)
    assert parsed["control"]["client_auth"] == "none"
    assert "ASTRUMWEAVER_CLIENT_TOKEN" not in env_text
    assert 'ASTRUMWEAVER_WORKER_TOKEN="worker-secret"' in env_text


def test_control_bearer_is_secure_first_run_default() -> None:
    spec = ControlBootstrapSpec()
    parsed = tomllib.loads(render_control_toml(spec))

    assert spec.client_auth is ClientAuthMode.BEARER
    assert parsed["control"]["client_auth"] == "bearer"

    with pytest.raises(ValueError, match="Client secret"):
        render_control_env(
            FirstRunSecrets(
                database_url="postgresql://secret@db/astrumweaver",
                worker_token="worker-secret",
            ),
            client_auth=spec.client_auth,
        )


def test_systemd_environment_values_are_quoted_and_control_characters_rejected() -> None:
    env = render_worker_env('token"with\\slashes')
    assert env == 'ASTRUMWEAVER_WORKER_TOKEN="token\\"with\\\\slashes"\n'

    with pytest.raises(ValueError, match="control characters"):
        render_worker_env("token\nInjected=value")


@pytest.mark.parametrize(
    ("bind_host", "expected"),
    (
        ("192.0.2.10", "http://192.0.2.10:9000"),
        ("0.0.0.0", "http://127.0.0.1:9000"),
        ("::1", "http://[::1]:9000"),
        ("[::1]", "http://[::1]:9000"),
    ),
)
def test_control_url_for_bind_host_formats_local_and_ipv6_addresses(
    bind_host: str, expected: str
) -> None:
    assert control_url_for_bind_host(bind_host, 9000) == expected


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
        worker(smoke=True),
        control_url="http://control.internal:9000",
        execution_mode=FirstRunExecutionMode.SMOKE,
    )
    parsed = tomllib.loads(text)

    assert parsed["executor"]["factory"] == (
        "astrumweaver.executors.structured_echo:create_executor"
    )
    assert "runtime" not in parsed


def test_nixos_control_snippet_carries_explicit_client_auth() -> None:
    snippet = render_nixos_bootstrap_snippet(
        role=FirstRunRole.CONTROL,
        control=ControlBootstrapSpec(
            client_auth=ClientAuthMode.NONE,
        ),
    )

    assert 'clientAuth = "none";' in snippet
    assert "environmentFile" in snippet


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
        worker=worker(smoke=True),
        control_url="http://control.internal:9000",
    )

    assert 'capabilities = [ "debug.echo" ];' in snippet
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


def test_explicit_packaged_tool_dir_is_the_only_resolution_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_bin = tmp_path / "profile" / "bin"
    store_bin = tmp_path / "store" / "bin"
    path_bin = tmp_path / "path" / "bin"
    for directory in (profile_bin, store_bin, path_bin):
        directory.mkdir(parents=True)

    name = "astrumweaver-worker"
    for directory, marker in (
        (profile_bin, "profile"),
        (store_bin, "store"),
        (path_bin, "path"),
    ):
        executable = directory / name
        executable.write_text(f"#!/bin/sh\n# {marker}\n", encoding="utf-8")
        executable.chmod(0o755)

    monkeypatch.setattr(sys, "argv", [str(store_bin / "astrumweaver-setup-tui")])
    monkeypatch.setenv("PATH", str(path_bin))

    installer = SystemdFirstRunInstaller(tool_dir=profile_bin)

    assert installer._resolve_tool(name) == str(profile_bin / name)

    (profile_bin / name).unlink()
    with pytest.raises(RuntimeError, match="explicit packaged tool authority"):
        installer._resolve_tool(name)


def test_packaging_check_exposes_only_the_explicit_tool_authority(
    tmp_path: Path,
    capsys,
) -> None:
    from astrumweaver.setup.tui import _check_packaging

    profile_bin = tmp_path / "profile" / "bin"
    profile_bin.mkdir(parents=True)
    for name in (
        "astrumweaver-setup-control-plane",
        "astrumweaver-setup-gpu-worker",
        "astrumweaver-control",
        "astrumweaver-worker",
        "astrumweaver-migrate",
    ):
        executable = profile_bin / name
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)

    assert _check_packaging(profile_bin) == 0
    output = capsys.readouterr().out
    assert f"packaged tool authority: {profile_bin}" in output
    assert f"astrumweaver-worker: {profile_bin / 'astrumweaver-worker'}" in output


def test_persistent_daemon_resolution_rejects_immutable_store_path(
    monkeypatch,
) -> None:
    installer = SystemdFirstRunInstaller()
    monkeypatch.setattr(
        installer,
        "_resolve_tool",
        lambda name: f"/nix/store/current/bin/{name}",
    )

    with pytest.raises(RuntimeError, match="immutable Nix store path"):
        installer._resolve_persistent_daemon("astrumweaver-worker")


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
            worker(smoke=True),
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
    assert env == 'ASTRUMWEAVER_WORKER_TOKEN="worker-secret"\n'
    assert "CLIENT" not in env
    assert "DATABASE" not in env


@pytest.mark.parametrize("role", ["control", "worker"])
@pytest.mark.parametrize("failure", ["timeout", "command"])
def test_failure_diagnostics_are_actionable_and_private(role, failure, monkeypatch):
    installer = RecordingInstaller()
    monkeypatch.setattr(installer, "_wait_json_ready", lambda *a, **kw: False)
    if failure == "command":
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, ["private-secret-command"])
        monkeypatch.setattr(installer, "_run", fail)

    with pytest.raises(RuntimeError) as error:
        if role == "control":
            installer.install_control(
                spec=ControlBootstrapSpec(port=9001),
                secrets=FirstRunSecrets(
                    database_url="postgresql://private-db-secret",
                    client_token="private-client-secret",
                    worker_token="private-worker-secret",
                ),
            )
        else:
            installer.install_worker(
                worker_toml=render_worker_toml(
                    worker(smoke=True), control_url="http://private-control:9000",
                    execution_mode=FirstRunExecutionMode.SMOKE,
                ),
                worker_token="private-worker-secret", start=True,
            )
    message = str(error.value)
    assert f"systemctl status astrumweaver-{role}.service" in message
    assert f"journalctl -u astrumweaver-{role}.service -b" in message
    endpoint = "9001/v1/ready" if role == "control" else "9100/health"
    assert f"curl http://127.0.0.1:{endpoint}" in message
    assert "private" not in message
    assert "GPU-" not in message


def test_both_bootstraps_resolve_helpers_through_symlink_profile(tmp_path, monkeypatch):
    store = tmp_path / "store"
    python_bin = store / "python" / "bin"
    integration_bin = store / "integration" / "bin"
    installer_bin = store / "installer" / "bin"
    for directory in (python_bin, integration_bin, installer_bin):
        directory.mkdir(parents=True)
    names = (
        "astrumweaver-setup-tui", "astrumweaver-setup-control-plane",
        "astrumweaver-setup-gpu-worker", "astrumweaver-control",
        "astrumweaver-worker", "astrumweaver-migrate",
    )
    for name in names:
        directory = integration_bin if name.endswith(("control-plane", "gpu-worker")) else python_bin
        executable = directory / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        (installer_bin / name).symlink_to(executable)
    profile = tmp_path / "installer-profile"
    profile.symlink_to(installer_bin.parent, target_is_directory=True)
    profile_bin = profile / "bin"
    monkeypatch.setattr(sys, "argv", [str(profile_bin / "astrumweaver-setup-tui")])
    monkeypatch.setenv("PATH", "")
    installer = SystemdFirstRunInstaller()
    monkeypatch.setattr(installer, "_require_root", lambda: None)
    monkeypatch.setattr(installer, "_wait_json_ready", lambda *a, **kw: True)
    calls = []
    monkeypatch.setattr(installer, "_run", lambda args, **kw: calls.append(args))
    for name in names:
        assert installer._resolve_tool(name) == str(profile_bin / name)
    installer.install_control(
        spec=ControlBootstrapSpec(),
        secrets=FirstRunSecrets(database_url="postgresql://test", client_token="client", worker_token="worker"),
    )
    installer.install_worker(worker_toml="[worker]\n", worker_token="worker", start=True)
    assert calls[0][0] == str(profile_bin / "astrumweaver-setup-control-plane")
    assert calls[0][-1] == str(profile_bin / "astrumweaver-control")
    assert calls[1] == [str(profile_bin / "astrumweaver-migrate")]
    assert calls[3][0] == str(profile_bin / "astrumweaver-setup-gpu-worker")
    assert calls[3][-2] == str(profile_bin / "astrumweaver-worker")


def test_smoke_renderers_reject_inconsistent_worker_spec():
    with pytest.raises(ValueError, match="only debug.echo"):
        render_worker_toml(
            worker(), control_url="http://localhost:9000",
            execution_mode=FirstRunExecutionMode.SMOKE,
        )
    with pytest.raises(ValueError, match="only debug.echo"):
        render_nixos_bootstrap_snippet(
            role=FirstRunRole.WORKER, worker=worker(),
            control_url="http://localhost:9000",
        )


def test_explicit_dot_slash_invocation_resolves_siblings(tmp_path, monkeypatch):
    helper = tmp_path / "astrumweaver-setup-gpu-worker"
    helper.write_text("#!/bin/sh\n")
    helper.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["./astrumweaver-setup-tui"])
    monkeypatch.setenv("PATH", "")
    assert SystemdFirstRunInstaller()._resolve_tool(helper.name) == str(helper)
