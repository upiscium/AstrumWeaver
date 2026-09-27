"""First-run host bootstrap helpers for the interactive setup wizard.

This module owns generic host-level materialization for Control/Worker bootstrap.
RuntimeProvider planning remains in setup.tui/setup.planner.
"""

from __future__ import annotations

import json
import os
import secrets as secrets_module
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Mapping

from ..contracts import WorkerSpec


class FirstRunRole(StrEnum):
    CONTROL = "control"
    WORKER = "worker"
    BOTH = "both"


class FirstRunExecutionMode(StrEnum):
    SMOKE = "smoke"
    RUNTIME = "runtime"


@dataclass(frozen=True, slots=True)
class ControlBootstrapSpec:
    bind_host: str = "127.0.0.1"
    port: int = 9000
    worker_ttl_seconds: int = 60
    lease_seconds: int = 300
    maintenance_interval_seconds: float = 5.0
    access_log: bool = False

    def __post_init__(self) -> None:
        if not self.bind_host.strip():
            raise ValueError("Control bind host must not be blank")
        if not 1 <= self.port <= 65535:
            raise ValueError("Control port must be between 1 and 65535")
        if self.worker_ttl_seconds <= 0 or self.lease_seconds <= 0:
            raise ValueError("Control TTL/lease must be positive")
        if self.maintenance_interval_seconds <= 0:
            raise ValueError("maintenance interval must be positive")


@dataclass(frozen=True, slots=True)
class FirstRunSecrets:
    database_url: str | None = field(default=None, repr=False)
    client_token: str | None = field(default=None, repr=False)
    worker_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("database_url", "client_token", "worker_token"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be blank when supplied")
        if (
            self.client_token is not None
            and self.worker_token is not None
            and self.client_token == self.worker_token
        ):
            raise ValueError("client and Worker authority tokens must differ")


def generate_authority_tokens() -> tuple[str, str]:
    client = secrets_module.token_hex(32)
    worker = secrets_module.token_hex(32)
    while worker == client:
        worker = secrets_module.token_hex(32)
    return client, worker


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_string_list(values: tuple[str, ...] | frozenset[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def render_control_toml(spec: ControlBootstrapSpec) -> str:
    access_log = "true" if spec.access_log else "false"
    return (
        "[control]\n"
        f"host = {_toml_string(spec.bind_host)}\n"
        f"port = {spec.port}\n"
        f"worker_ttl_seconds = {spec.worker_ttl_seconds}\n"
        f"lease_seconds = {spec.lease_seconds}\n"
        f"maintenance_interval_seconds = {spec.maintenance_interval_seconds}\n"
        f"access_log = {access_log}\n"
    )


def render_control_env(secrets: FirstRunSecrets) -> str:
    if not all(
        (
            secrets.database_url,
            secrets.client_token,
            secrets.worker_token,
        )
    ):
        raise ValueError("Control bootstrap requires database/client/Worker secrets")
    return (
        f"ASTRUMWEAVER_DATABASE_URL={secrets.database_url}\n"
        f"ASTRUMWEAVER_CLIENT_TOKEN={secrets.client_token}\n"
        f"ASTRUMWEAVER_WORKER_TOKEN={secrets.worker_token}\n"
    )


def render_worker_env(worker_token: str) -> str:
    if not worker_token.strip():
        raise ValueError("Worker token must not be blank")
    return f"ASTRUMWEAVER_WORKER_TOKEN={worker_token}\n"


def render_worker_toml(
    worker: WorkerSpec,
    *,
    control_url: str,
    execution_mode: FirstRunExecutionMode,
    runtime_manifest: str = "/etc/astrumweaver/runtime-deployment.json",
    health_host: str = "127.0.0.1",
    health_port: int = 9100,
) -> str:
    if not control_url.strip():
        raise ValueError("Control URL must not be blank")
    if not 1 <= health_port <= 65535:
        raise ValueError("Worker health port must be between 1 and 65535")

    capabilities = tuple(sorted(worker.capabilities))
    gpu_uuids = tuple(worker.gpu_uuids)
    lines = [
        "[worker]",
        f"id = {_toml_string(worker.worker_id)}",
        f"class = {_toml_string(worker.worker_class)}",
        f"control_url = {_toml_string(control_url)}",
        f"capabilities = {_toml_string_list(capabilities)}",
        f"gpu_uuids = {_toml_string_list(gpu_uuids)}",
        f"gpu_count = {worker.resources.gpu_count}",
        f"total_vram_mb = {worker.resources.total_vram_mb}",
        f"max_single_gpu_vram_mb = {worker.resources.max_single_gpu_vram_mb}",
        "max_concurrency = 1",
        f"gpu_preflight = {'true' if gpu_uuids else 'false'}",
        f"health_host = {_toml_string(health_host)}",
        f"health_port = {health_port}",
    ]
    if worker.labels:
        lines.append("")
        lines.append("[worker.labels]")
        for key, value in sorted(worker.labels.items()):
            lines.append(f"{_toml_string(key)} = {_toml_string(value)}")

    for accelerator in worker.accelerators:
        lines.extend(
            [
                "",
                "[[worker.accelerators]]",
                f"uuid = {_toml_string(accelerator.uuid)}",
                f"memory_mb = {accelerator.memory_mb}",
            ]
        )
        if accelerator.compute_capability is not None:
            lines.append(
                "compute_capability = "
                + _toml_string(accelerator.compute_capability)
            )
        if accelerator.device_class is not None:
            lines.append(
                "device_class = " + _toml_string(accelerator.device_class)
            )

    lines.append("")
    if execution_mode is FirstRunExecutionMode.SMOKE:
        lines.extend(
            [
                "[executor]",
                "factory = "
                + _toml_string(
                    "astrumweaver.executors.structured_echo:create_executor"
                ),
                "",
                "[executor.settings]",
            ]
        )
    else:
        lines.extend(
            [
                "[runtime]",
                f"manifest = {_toml_string(runtime_manifest)}",
                "startup_timeout_seconds = 600",
                "shutdown_timeout_seconds = 60",
            ]
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class SystemdBootstrapResult:
    control_installed: bool = False
    control_ready: bool = False
    migration_applied: bool = False
    worker_installed: bool = False
    worker_ready: bool = False


class SystemdFirstRunInstaller:
    """Use packaged integration wrappers to perform existing-node bootstrap."""

    def __init__(
        self,
        *,
        tool_dir: Path | None = None,
        systemctl: str = "systemctl",
        poll_interval_seconds: float = 0.5,
        ready_timeout_seconds: float = 60.0,
    ) -> None:
        self.tool_dir = tool_dir
        self.systemctl = systemctl
        self.poll_interval_seconds = poll_interval_seconds
        self.ready_timeout_seconds = ready_timeout_seconds

    def _require_root(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            raise PermissionError(
                "generic systemd first-run apply must run as root "
                "(invoke astrumweaver-setup-tui with sudo)"
            )

    def _resolve_tool(self, name: str) -> str:
        candidates: list[Path] = []
        if self.tool_dir is not None:
            candidates.append(self.tool_dir / name)
        argv0 = Path(sys.argv[0])
        if argv0.parent != Path("."):
            candidates.append(argv0.resolve().parent / name)
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        raise RuntimeError(
            f"required packaged command is unavailable: {name}"
        )

    @contextmanager
    def _temporary_file(
        self,
        content: str,
        *,
        suffix: str,
    ) -> Iterator[Path]:
        fd, raw_path = tempfile.mkstemp(
            prefix="astrumweaver-first-run-",
            suffix=suffix,
        )
        path = Path(raw_path)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            yield path
        finally:
            path.unlink(missing_ok=True)

    def _run(
        self,
        args: list[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        subprocess.run(
            args,
            check=True,
            env=None if env is None else dict(env),
        )

    def _wait_json_ready(
        self,
        url: str,
        *,
        require_registered: bool = False,
    ) -> bool:
        deadline = time.monotonic() + self.ready_timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    url,
                    timeout=3.0,
                ) as response:
                    if response.status != 200:
                        raise RuntimeError("not ready")
                    payload = json.loads(
                        response.read().decode("utf-8")
                    )
                if not isinstance(payload, dict):
                    raise RuntimeError("invalid readiness payload")
                if payload.get("ready") is not True:
                    raise RuntimeError("not ready")
                if (
                    require_registered
                    and payload.get("registered") is not True
                ):
                    raise RuntimeError("Worker is not registered")
                return True
            except (
                OSError,
                RuntimeError,
                urllib.error.URLError,
                json.JSONDecodeError,
            ):
                time.sleep(self.poll_interval_seconds)
        return False

    def install_control(
        self,
        *,
        spec: ControlBootstrapSpec,
        secrets: FirstRunSecrets,
    ) -> SystemdBootstrapResult:
        self._require_root()
        setup = self._resolve_tool("astrumweaver-setup-control-plane")
        executable = self._resolve_tool("astrumweaver-control")
        migrate = self._resolve_tool("astrumweaver-migrate")

        with self._temporary_file(
            render_control_toml(spec),
            suffix=".toml",
        ) as config_path, self._temporary_file(
            render_control_env(secrets),
            suffix=".env",
        ) as env_path:
            self._run(
                [
                    setup,
                    "--config",
                    str(config_path),
                    "--environment-file",
                    str(env_path),
                    "--executable",
                    executable,
                ]
            )

        migration_env = dict(os.environ)
        assert secrets.database_url is not None
        migration_env["ASTRUMWEAVER_DATABASE_URL"] = (
            secrets.database_url
        )
        self._run([migrate], env=migration_env)
        self._run(
            [
                self.systemctl,
                "enable",
                "--now",
                "astrumweaver-control.service",
            ]
        )

        host = spec.bind_host
        if host in {"0.0.0.0", "::", "[::]"}:
            host = "127.0.0.1"
        ready = self._wait_json_ready(
            f"http://{host}:{spec.port}/v1/ready"
        )
        if not ready:
            raise RuntimeError(
                "Control service did not become ready after migration/start"
            )
        return SystemdBootstrapResult(
            control_installed=True,
            control_ready=True,
            migration_applied=True,
        )

    def install_worker(
        self,
        *,
        worker_toml: str,
        worker_token: str,
        gpu_uuids: tuple[str, ...],
        runtime_manifest_json: str | None = None,
        start: bool,
    ) -> SystemdBootstrapResult:
        self._require_root()
        if not gpu_uuids:
            raise RuntimeError(
                "generic systemd first-run Worker integration currently "
                "requires at least one selected NVIDIA GPU"
            )
        setup = self._resolve_tool("astrumweaver-setup-gpu-worker")
        executable = self._resolve_tool("astrumweaver-worker")

        with self._temporary_file(
            worker_toml,
            suffix=".toml",
        ) as config_path, self._temporary_file(
            render_worker_env(worker_token),
            suffix=".env",
        ) as env_path:
            args = [
                setup,
                "--config",
                str(config_path),
                "--environment-file",
                str(env_path),
                "--executable",
                executable,
            ]
            for uuid in gpu_uuids:
                args.extend(["--gpu-uuid", uuid])

            if runtime_manifest_json is not None:
                with self._temporary_file(
                    runtime_manifest_json,
                    suffix=".json",
                ) as manifest_path:
                    args.extend(
                        [
                            "--runtime-manifest",
                            str(manifest_path),
                        ]
                    )
                    if start:
                        args.append("--start")
                    self._run(args)
            else:
                if start:
                    args.append("--start")
                self._run(args)

        ready = False
        if start:
            ready = self._wait_json_ready(
                "http://127.0.0.1:9100/health",
                require_registered=True,
            )
            if not ready:
                raise RuntimeError(
                    "Worker service did not become ready/registered"
                )
        return SystemdBootstrapResult(
            worker_installed=True,
            worker_ready=ready,
        )


def render_nixos_bootstrap_snippet(
    *,
    role: FirstRunRole,
    control: ControlBootstrapSpec | None = None,
    worker: WorkerSpec | None = None,
    control_url: str | None = None,
    execution_mode: FirstRunExecutionMode = FirstRunExecutionMode.SMOKE,
    control_env_path: str = "/etc/astrumweaver/control.env",
    worker_env_path: str = "/etc/astrumweaver/worker.env",
) -> str:
    """Render deterministic Nix config without mutating operator Nix sources."""

    lines = ["{ config, ... }:", "", "{"]
    if role in {FirstRunRole.CONTROL, FirstRunRole.BOTH}:
        if control is None:
            raise ValueError("Control role requires ControlBootstrapSpec")
        lines.extend(
            [
                "  services.astrumweaver.control = {",
                "    enable = true;",
                "    settings.control = {",
                f"      host = {_toml_string(control.bind_host)};",
                f"      port = {control.port};",
                f"      worker_ttl_seconds = {control.worker_ttl_seconds};",
                f"      lease_seconds = {control.lease_seconds};",
                "      maintenance_interval_seconds = "
                f"{control.maintenance_interval_seconds};",
                "      access_log = "
                + ("true;" if control.access_log else "false;"),
                "    };",
                f"    environmentFile = {_toml_string(control_env_path)};",
                "    migrateOnStart = true;",
                "  };",
                "",
            ]
        )

    if role in {FirstRunRole.WORKER, FirstRunRole.BOTH}:
        if worker is None or control_url is None:
            raise ValueError("Worker role requires WorkerSpec/control URL")
        lines.extend(
            [
                "  services.astrumweaver.worker = {",
                "    enable = true;",
                f"    workerId = {_toml_string(worker.worker_id)};",
                f"    workerClass = {_toml_string(worker.worker_class)};",
                f"    controlUrl = {_toml_string(control_url)};",
                "    capabilities = "
                + _toml_string_list(tuple(sorted(worker.capabilities)))
                + ";",
                "    gpuUuids = "
                + _toml_string_list(tuple(worker.gpu_uuids))
                + ";",
                f"    totalVramMb = {worker.resources.total_vram_mb};",
                "    maxSingleGpuVramMb = "
                f"{worker.resources.max_single_gpu_vram_mb};",
                "    nvidiaSmiPackage = config.hardware.nvidia.package;",
                f"    environmentFile = {_toml_string(worker_env_path)};",
            ]
        )
        if execution_mode is FirstRunExecutionMode.SMOKE:
            lines.extend(
                [
                    "    executorFactory = "
                    + _toml_string(
                        "astrumweaver.executors.structured_echo:create_executor"
                    )
                    + ";",
                ]
            )
        else:
            lines.extend(
                [
                    "    # RuntimeProvider settings are generated/reviewed by",
                    "    # the runtime phase of astrumweaver-setup-tui.",
                    "    # Do not set executorFactory when runtime.enable = true.",
                ]
            )
        lines.extend(["  };", ""])

    lines.append("}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ControlBootstrapSpec",
    "FirstRunExecutionMode",
    "FirstRunRole",
    "FirstRunSecrets",
    "SystemdBootstrapResult",
    "SystemdFirstRunInstaller",
    "generate_authority_tokens",
    "render_control_env",
    "render_control_toml",
    "render_nixos_bootstrap_snippet",
    "render_worker_env",
    "render_worker_toml",
]
