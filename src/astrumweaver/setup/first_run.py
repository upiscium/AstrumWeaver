"""First-run host bootstrap helpers for the interactive setup wizard.

This module owns generic host-level materialization for Control/Worker bootstrap.
RuntimeProvider planning remains in setup.tui/setup.planner.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
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
from ..control.auth import ClientAuthMode


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
    client_auth: ClientAuthMode = ClientAuthMode.BEARER
    worker_ttl_seconds: int = 60
    lease_seconds: int = 300
    maintenance_interval_seconds: float = 5.0
    access_log: bool = False

    def __post_init__(self) -> None:
        if not self.bind_host.strip():
            raise ValueError("Control bind host must not be blank")
        if not 1 <= self.port <= 65535:
            raise ValueError("Control port must be between 1 and 65535")
        object.__setattr__(
            self,
            "client_auth",
            ClientAuthMode(self.client_auth),
        )
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


def generate_worker_token() -> str:
    return secrets_module.token_hex(32)


def generate_authority_tokens() -> tuple[str, str]:
    client = secrets_module.token_hex(32)
    worker = generate_worker_token()
    while worker == client:
        worker = generate_worker_token()
    return client, worker


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_string_list(values: tuple[str, ...] | frozenset[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _nix_string(value: str) -> str:
    # JSON and Nix share quote/backslash escapes, but only Nix interpolates
    # ${...}. Operator/model data must remain literal, never acquire scope.
    return json.dumps(value, ensure_ascii=False).replace("${", r"\${")


def validate_nixos_runtime_package_expression(value: str) -> str:
    """Accept only a package attribute path rooted in the bound module pkgs.

    Custom packages belong in a nixpkgs overlay, not an undeclared myPkgs or
    inputs root. This is deliberately not a general-purpose Nix expression
    parser; attribute existence/package type is checked by Nix evaluation.
    """
    expression = value.strip()
    if not re.fullmatch(r"pkgs(?:\.[A-Za-z_][A-Za-z0-9_'-]*)+", expression) or any(
        part in {"if", "then", "else", "assert", "with", "let", "in", "rec", "inherit"}
        for part in expression.split(".")[1:]
    ):
        raise ValueError(
            "runtime package expression must be a simple attribute path rooted "
            "in pkgs, such as pkgs.ollama or pkgs.vllm; expose custom packages "
            "through a nixpkgs overlay"
        )
    return expression


def _nix_string_list(values: tuple[str, ...] | frozenset[str]) -> str:
    return "[ " + " ".join(_nix_string(value) for value in values) + " ]"


def render_control_toml(spec: ControlBootstrapSpec) -> str:
    access_log = "true" if spec.access_log else "false"
    return (
        "[control]\n"
        f"host = {_toml_string(spec.bind_host)}\n"
        f"port = {spec.port}\n"
        f"client_auth = {_toml_string(spec.client_auth.value)}\n"
        f"worker_ttl_seconds = {spec.worker_ttl_seconds}\n"
        f"lease_seconds = {spec.lease_seconds}\n"
        f"maintenance_interval_seconds = {spec.maintenance_interval_seconds}\n"
        f"access_log = {access_log}\n"
    )


def render_control_env(
    secrets: FirstRunSecrets,
    *,
    client_auth: ClientAuthMode | str = ClientAuthMode.BEARER,
) -> str:
    mode = ClientAuthMode(client_auth)
    if not secrets.database_url or not secrets.worker_token:
        raise ValueError(
            "Control bootstrap requires database and Worker secrets"
        )
    if mode is ClientAuthMode.BEARER and not secrets.client_token:
        raise ValueError(
            "Control bearer client auth requires a Client secret"
        )

    lines = [
        _render_systemd_environment_line(
            "ASTRUMWEAVER_DATABASE_URL", secrets.database_url
        ),
        _render_systemd_environment_line(
            "ASTRUMWEAVER_WORKER_TOKEN", secrets.worker_token
        ),
    ]
    if mode is ClientAuthMode.BEARER:
        assert secrets.client_token is not None
        lines.insert(
            1,
            _render_systemd_environment_line(
                "ASTRUMWEAVER_CLIENT_TOKEN", secrets.client_token
            ),
        )
    return "".join(lines)


def render_worker_env(worker_token: str) -> str:
    if not worker_token.strip():
        raise ValueError("Worker token must not be blank")
    return _render_systemd_environment_line(
        "ASTRUMWEAVER_WORKER_TOKEN", worker_token
    )


def _render_systemd_environment_line(name: str, value: str) -> str:
    """Render one safely quoted systemd EnvironmentFile assignment."""

    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{name} must not contain control characters")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{escaped}"\n'


def control_url_for_bind_host(bind_host: str, port: int) -> str:
    """Build a local Worker/readiness URL for a Control bind address."""

    host = bind_host.strip()
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    elif host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        formatted_host = host
    else:
        formatted_host = f"[{host}]" if address.version == 6 else host
    return f"http://{formatted_host}:{port}"


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
    _validate_smoke_capabilities(worker, execution_mode)

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


def _validate_smoke_capabilities(
    worker: WorkerSpec, execution_mode: FirstRunExecutionMode,
) -> None:
    if (
        execution_mode is FirstRunExecutionMode.SMOKE
        and worker.capabilities != frozenset({"debug.echo"})
    ):
        raise ValueError("smoke WorkerSpec must advertise only debug.echo")


def write_protected_file(path: Path, content: str) -> None:
    """Write a root-owned secret/config input without silently replacing drift."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != content:
            raise RuntimeError(
                f"refusing to replace existing protected file with different content: {path}"
            )
        os.chmod(path, 0o600)
        return

    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    tmp = Path(raw_tmp)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SystemdBootstrapResult:
    control_installed: bool = False
    control_ready: bool = False
    migration_applied: bool = False
    worker_installed: bool = False
    worker_ready: bool = False


class SystemdFirstRunInstaller:
    """Use packaged integration wrappers to perform existing-node bootstrap.

    ``tool_dir`` is an explicit packaging authority.  When supplied, every
    helper and daemon executable is resolved lexically from that directory;
    neither ``sys.argv[0]`` nor the ambient ``PATH`` may select a different
    closure.  This is important because setup helpers are transient commands,
    while the daemon path is persisted in a systemd unit.
    """

    def __init__(
        self,
        *,
        tool_dir: Path | None = None,
        systemctl: str = "systemctl",
        poll_interval_seconds: float = 0.5,
        ready_timeout_seconds: float = 60.0,
    ) -> None:
        if tool_dir is not None and not tool_dir.is_absolute():
            raise ValueError(
                f"packaged tool directory must be absolute: {tool_dir}"
            )
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
        if self.tool_dir is not None:
            candidate = self.tool_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
            raise RuntimeError(
                "required packaged command is unavailable from the explicit "
                f"packaged tool authority: {name}"
            )

        candidates: list[Path] = []
        argv0 = Path(sys.argv[0])
        if os.sep in sys.argv[0]:
            # Keep the invocation path before resolving symlinks. Dedicated
            # Nix profiles expose a combined bin/ directory whose setup
            # helpers are siblings of astrumweaver-setup-tui; resolving the
            # console-script symlink first would jump into the Python store
            # path and lose those integration helpers.
            candidates.append(argv0.absolute().parent / name)
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

    def _resolve_packaged_tool(self, name: str) -> str:
        """Resolve a transient helper from the explicit package authority."""

        return self._resolve_tool(name)

    def _resolve_persistent_daemon(self, name: str) -> str:
        """Resolve the executable path that will be persisted in ExecStart."""

        resolved = self._resolve_tool(name)
        if self.tool_dir is None and resolved.startswith("/nix/store/"):
            raise RuntimeError(
                "persistent daemon executable resolved to an immutable Nix "
                "store path; invoke the packaged TUI or supply an explicit "
                "profile tool authority"
            )
        return resolved

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

    @staticmethod
    def _diagnostics(role: str, port: int) -> str:
        endpoint = "v1/ready" if role == "control" else "health"
        # Only fixed unit names and a validated, non-secret port are emitted.
        # Never include subprocess argv/env, DB URLs, tokens or GPU identity.
        return (
            "Inspect:\n"
            f"  systemctl status astrumweaver-{role}.service\n"
            f"  journalctl -u astrumweaver-{role}.service -b\n"
            f"  curl http://127.0.0.1:{port}/{endpoint}\n"
            "Use the configured local bind address if loopback is not enabled."
        )

    def _run_service(
        self,
        args: list[str],
        *,
        role: str,
        port: int,
        env: Mapping[str, str] | None = None,
    ) -> None:
        try:
            self._run(args, env=env)
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError(
                f"{role.capitalize()} setup/migration/start command failed.\n"
                + self._diagnostics(role, port)
            ) from None

    def install_control(
        self,
        *,
        spec: ControlBootstrapSpec,
        secrets: FirstRunSecrets,
    ) -> SystemdBootstrapResult:
        self._require_root()
        setup = self._resolve_packaged_tool("astrumweaver-setup-control-plane")
        executable = self._resolve_persistent_daemon("astrumweaver-control")
        migrate = self._resolve_packaged_tool("astrumweaver-migrate")

        with self._temporary_file(
            render_control_toml(spec),
            suffix=".toml",
        ) as config_path, self._temporary_file(
            render_control_env(
                secrets,
                client_auth=spec.client_auth,
            ),
            suffix=".env",
        ) as env_path:
            self._run_service(
                [
                    setup,
                    "--config",
                    str(config_path),
                    "--environment-file",
                    str(env_path),
                    "--executable",
                    executable,
                ],
                role="control",
                port=spec.port,
            )

        migration_env = dict(os.environ)
        assert secrets.database_url is not None
        migration_env["ASTRUMWEAVER_DATABASE_URL"] = (
            secrets.database_url
        )
        self._run_service(
            [migrate], env=migration_env, role="control", port=spec.port
        )
        self._run_service(
            [
                self.systemctl,
                "enable",
                "--now",
                "astrumweaver-control.service",
            ],
            role="control",
            port=spec.port,
        )

        ready = self._wait_json_ready(
            control_url_for_bind_host(spec.bind_host, spec.port) + "/v1/ready"
        )
        if not ready:
            raise RuntimeError(
                "Control service did not become ready after migration/start.\n"
                + self._diagnostics("control", spec.port)
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
        runtime_manifest_json: str | None = None,
        start: bool,
    ) -> SystemdBootstrapResult:
        self._require_root()
        setup = self._resolve_packaged_tool("astrumweaver-setup-gpu-worker")
        executable = self._resolve_persistent_daemon("astrumweaver-worker")

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
                    self._run_service(args, role="worker", port=9100)
            else:
                if start:
                    args.append("--start")
                self._run_service(args, role="worker", port=9100)

        ready = False
        if start:
            ready = self._wait_json_ready(
                "http://127.0.0.1:9100/health",
                require_registered=True,
            )
            if not ready:
                raise RuntimeError(
                    "Worker service did not become ready/registered.\n"
                    + self._diagnostics("worker", 9100)
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
    runtime_deployment: Mapping[str, object] | None = None,
    runtime_package_expression: str | None = None,
) -> str:
    """Render deterministic Nix config without mutating operator Nix sources."""

    lines = ["{ config, pkgs, ... }:", "", "{"]
    if role in {FirstRunRole.CONTROL, FirstRunRole.BOTH}:
        if control is None:
            raise ValueError("Control role requires ControlBootstrapSpec")
        lines.extend(
            [
                "  services.astrumweaver.control = {",
                "    enable = true;",
                f"    clientAuth = {_nix_string(control.client_auth.value)};",
                "    settings.control = {",
                f"      host = {_nix_string(control.bind_host)};",
                f"      port = {control.port};",
                f"      worker_ttl_seconds = {control.worker_ttl_seconds};",
                f"      lease_seconds = {control.lease_seconds};",
                "      maintenance_interval_seconds = "
                f"{control.maintenance_interval_seconds};",
                "      access_log = "
                + ("true;" if control.access_log else "false;"),
                "    };",
                f"    environmentFile = {_nix_string(control_env_path)};",
                "    migrateOnStart = true;",
                "  };",
                "",
            ]
        )

    if role in {FirstRunRole.WORKER, FirstRunRole.BOTH}:
        if worker is None or control_url is None:
            raise ValueError("Worker role requires WorkerSpec/control URL")
        _validate_smoke_capabilities(worker, execution_mode)
        lines.extend(
            [
                "  services.astrumweaver.worker = {",
                "    enable = true;",
                f"    workerId = {_nix_string(worker.worker_id)};",
                f"    workerClass = {_nix_string(worker.worker_class)};",
                f"    controlUrl = {_nix_string(control_url)};",
                "    capabilities = "
                + _nix_string_list(tuple(sorted(worker.capabilities)))
                + ";",
                "    gpuUuids = "
                + _nix_string_list(tuple(worker.gpu_uuids))
                + ";",
                f"    totalVramMb = {worker.resources.total_vram_mb};",
                "    maxSingleGpuVramMb = "
                f"{worker.resources.max_single_gpu_vram_mb};",
                "    nvidiaSmiPackage = config.hardware.nvidia.package;",
                f"    environmentFile = {_nix_string(worker_env_path)};",
            ]
        )
        if execution_mode is FirstRunExecutionMode.SMOKE:
            lines.extend(
                [
                    "    executorFactory = "
                    + _nix_string(
                        "astrumweaver.executors.structured_echo:create_executor"
                    )
                    + ";",
                ]
            )
        else:
            if runtime_deployment is None or runtime_package_expression is None:
                raise ValueError(
                    "RuntimeProvider Nix rendering requires reviewed deployment "
                    "data and an explicit Nix package expression"
                )
            package_expr = validate_nixos_runtime_package_expression(
                runtime_package_expression
            )
            provider_id = str(runtime_deployment["provider_id"])
            provider_config = dict(
                runtime_deployment.get("provider_config") or {}
            )
            demand = dict(runtime_deployment["demand"])
            model = dict(demand["model"])
            lines.extend(
                [
                    "    runtime = {",
                    "      enable = true;",
                    f"      provider = {_nix_string(provider_id)};",
                    f"      packages = [ {package_expr} ];",
                    f"      modelRef = {_nix_string(str(model['model_ref']))};",
                    f"      modelFormat = {_nix_string(str(model['model_format']))};",
                    f"      modelTopology = {_nix_string(str(model['topology']))};",
                    f"      residencyPolicy = {_nix_string(str(demand['residency_policy']))};",
                ]
            )
            if model.get("estimated_size_mb") is not None:
                lines.append(
                    "      estimatedModelSizeMb = "
                    f"{int(model['estimated_size_mb'])};"
                )
            if demand.get("gpu_topology") is not None:
                lines.append(
                    "      gpuTopology = "
                    + _nix_string(str(demand["gpu_topology"]))
                    + ";"
                )
            for nix_name, demand_name in (
                ("minGpuCount", "min_gpu_count"),
                ("minTotalVramMb", "min_total_vram_mb"),
                ("minSingleGpuVramMb", "min_single_gpu_vram_mb"),
                ("minHostRamMb", "min_host_ram_mb"),
                ("preferredHostRamMb", "preferred_host_ram_mb"),
            ):
                lines.append(
                    f"      {nix_name} = {int(demand.get(demand_name, 0))};"
                )
            provider_json = json.dumps(
                provider_config,
                sort_keys=True,
                separators=(",", ":"),
            )
            model_metadata_json = json.dumps(
                dict(model.get("metadata") or {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            demand_metadata_json = json.dumps(
                dict(demand.get("metadata") or {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            lines.extend(
                [
                    "      providerConfig = builtins.fromJSON "
                    + _nix_string(provider_json)
                    + ";",
                    "      modelMetadata = builtins.fromJSON "
                    + _nix_string(model_metadata_json)
                    + ";",
                    "      demandMetadata = builtins.fromJSON "
                    + _nix_string(demand_metadata_json)
                    + ";",
                    "    };",
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
    "control_url_for_bind_host",
    "generate_authority_tokens",
    "generate_worker_token",
    "render_control_env",
    "render_control_toml",
    "render_nixos_bootstrap_snippet",
    "render_worker_env",
    "render_worker_toml",
    "validate_nixos_runtime_package_expression",
    "write_protected_file",
]
