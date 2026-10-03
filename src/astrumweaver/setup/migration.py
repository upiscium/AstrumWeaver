"""Fail-closed contracts for migrating an installed systemd Worker to runtime execution."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from typing import Any

from ..contracts import AcceleratorDevice, ResourceShape, WorkerSpec


SMOKE_EXECUTOR_FACTORY = "astrumweaver.executors.structured_echo:create_executor"
RUNTIME_CAPABILITIES = frozenset({"llm.chat", "text.generate"})
DEFAULT_RUNTIME_MANIFEST = "/etc/astrumweaver/runtime-deployment.json"
DEFAULT_RUNTIME_STARTUP_TIMEOUT_SECONDS = 600
DEFAULT_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS = 60
NVIDIA_DRIVER_BRIDGE_DIRECTORY = (
    "/var/lib/astrumweaver/runtime/nvidia-driver"
)
_RUNTIME_DRIVER_ENV_LINE = (
    f"Environment=LD_LIBRARY_PATH={NVIDIA_DRIVER_BRIDGE_DIRECTORY}"
)

_SMOKE_SUFFIX = (
    "\n[executor]\n"
    'factory = "astrumweaver.executors.structured_echo:create_executor"\n'
    "\n[executor.settings]\n"
)
_RUNTIME_SUFFIX_TEMPLATE = (
    "\n[runtime]\n"
    'manifest = "{manifest}"\n'
    "startup_timeout_seconds = 600\n"
    "shutdown_timeout_seconds = 60\n"
)
_SMOKE_CAPABILITIES_LINE = 'capabilities = ["debug.echo"]'
_RUNTIME_CAPABILITIES_LINE = 'capabilities = ["llm.chat", "text.generate"]'

_UNIT_PREFIX = """[Unit]
Description=AstrumWeaver GPU Worker
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
"""
_UNIT_SUFFIX = """Restart=on-failure
RestartSec=5s
RuntimeDirectory=astrumweaver-worker
RuntimeDirectoryMode=0750
StateDirectory=astrumweaver
StateDirectoryMode=0750
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
"""
_EXEC_START_RE = re.compile(
    r"^ExecStart=(?P<executable>/[^\s]+) --config "
    r"/etc/astrumweaver/worker\.toml"
    r"(?P<runtime_arg> --runtime-manifest "
    r"/etc/astrumweaver/runtime-deployment\.json)?$"
)


@dataclass(frozen=True, slots=True)
class InstalledWorkerContract:
    spec: WorkerSpec
    control_url: str
    max_concurrency: int
    gpu_preflight: bool
    health_host: str
    health_port: int
    execution_mode: str
    runtime_manifest: str | None = None


@dataclass(frozen=True, slots=True)
class InstalledWorkerUnit:
    service_user: str
    executable: str
    execution_mode: str


_REQUIRED_WORKER_KEYS = frozenset(
    {
        "id",
        "class",
        "control_url",
        "capabilities",
        "gpu_uuids",
        "gpu_count",
        "total_vram_mb",
        "max_single_gpu_vram_mb",
    }
)
_OPTIONAL_WORKER_KEYS = frozenset(
    {
        "labels",
        "accelerators",
        "max_concurrency",
        "gpu_preflight",
        "nvidia_smi_command",
        "request_timeout_seconds",
        "poll_interval_seconds",
        "heartbeat_interval_seconds",
        "health_host",
        "health_port",
        "shutdown_grace_seconds",
    }
)
_ACCELERATOR_KEYS = frozenset(
    {"uuid", "memory_mb", "compute_capability", "device_class"}
)


def _validate_worker_shape(worker: dict[str, Any]) -> None:
    keys = set(worker)
    missing = _REQUIRED_WORKER_KEYS - keys
    unknown = keys - _REQUIRED_WORKER_KEYS - _OPTIONAL_WORKER_KEYS
    if missing:
        raise RuntimeError(
            "installed Worker TOML is missing canonical worker fields"
        )
    if unknown:
        raise RuntimeError(
            "installed Worker TOML contains unrecognized worker fields"
        )

    for name in ("id", "class", "control_url"):
        if not isinstance(worker.get(name), str):
            raise RuntimeError(
                f"installed Worker worker.{name} must be a string"
            )
    for name in ("health_host", "nvidia_smi_command"):
        value = worker.get(name)
        if value is not None and not isinstance(value, str):
            raise RuntimeError(
                f"installed Worker worker.{name} must be a string"
            )
    for name in (
        "gpu_count",
        "total_vram_mb",
        "max_single_gpu_vram_mb",
    ):
        value = worker.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"installed Worker worker.{name} must be an integer"
            )
    for name in ("max_concurrency", "health_port"):
        value = worker.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise RuntimeError(
                f"installed Worker worker.{name} must be an integer"
            )
    gpu_preflight = worker.get("gpu_preflight")
    if gpu_preflight is not None and not isinstance(gpu_preflight, bool):
        raise RuntimeError(
            "installed Worker worker.gpu_preflight must be boolean"
        )
    for name in (
        "request_timeout_seconds",
        "poll_interval_seconds",
        "heartbeat_interval_seconds",
        "shutdown_grace_seconds",
    ):
        value = worker.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise RuntimeError(
                f"installed Worker worker.{name} must be numeric"
            )
    for name in ("capabilities", "gpu_uuids"):
        value = worker.get(name)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise RuntimeError(
                f"installed Worker worker.{name} must be a string array"
            )

    labels = worker.get("labels")
    if labels is not None and (
        not isinstance(labels, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in labels.items()
        )
    ):
        raise RuntimeError(
            "installed Worker worker.labels must be a string table"
        )

    accelerators = worker.get("accelerators")
    if accelerators is not None:
        if not isinstance(accelerators, list):
            raise RuntimeError(
                "installed Worker worker.accelerators must be an array of tables"
            )
        for accelerator in accelerators:
            if not isinstance(accelerator, dict):
                raise RuntimeError(
                    "installed Worker accelerator entry must be a table"
                )
            keys = set(accelerator)
            if not {"uuid", "memory_mb"} <= keys or not keys <= _ACCELERATOR_KEYS:
                raise RuntimeError(
                    "installed Worker accelerator entry is not canonical"
                )
            if not isinstance(accelerator["uuid"], str):
                raise RuntimeError(
                    "installed Worker accelerator uuid must be a string"
                )
            memory_mb = accelerator["memory_mb"]
            if isinstance(memory_mb, bool) or not isinstance(memory_mb, int):
                raise RuntimeError(
                    "installed Worker accelerator memory_mb must be an integer"
                )
            for name in ("compute_capability", "device_class"):
                value = accelerator.get(name)
                if value is not None and not isinstance(value, str):
                    raise RuntimeError(
                        f"installed Worker accelerator {name} must be a string"
                    )


def _worker_spec(worker: dict[str, Any]) -> WorkerSpec:
    gpu_uuids = tuple(str(item) for item in worker.get("gpu_uuids", ()))
    accelerators = tuple(
        AcceleratorDevice(
            uuid=str(item["uuid"]),
            memory_mb=int(item["memory_mb"]),
            compute_capability=item.get("compute_capability"),
            device_class=item.get("device_class"),
        )
        for item in worker.get("accelerators", ())
    )
    return WorkerSpec(
        worker_id=str(worker["id"]),
        worker_class=str(worker["class"]),
        resources=ResourceShape(
            gpu_count=int(worker.get("gpu_count", len(gpu_uuids))),
            total_vram_mb=int(worker.get("total_vram_mb", 0)),
            max_single_gpu_vram_mb=int(
                worker.get("max_single_gpu_vram_mb", 0)
            ),
        ),
        gpu_uuids=gpu_uuids,
        accelerators=accelerators,
        capabilities=frozenset(
            str(item) for item in worker.get("capabilities", ())
        ),
        labels={
            str(key): str(value)
            for key, value in dict(worker.get("labels") or {}).items()
        },
    )


def parse_installed_worker_toml(text: str) -> InstalledWorkerContract:
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError("installed Worker TOML is invalid") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("installed Worker TOML must contain a table")

    raw_worker = parsed.get("worker")
    if not isinstance(raw_worker, dict):
        raise RuntimeError("installed Worker TOML lacks a canonical worker table")
    worker = dict(raw_worker)
    _validate_worker_shape(worker)
    for key in ("id", "class", "control_url"):
        if not str(worker.get(key) or "").strip():
            raise RuntimeError(
                f"installed Worker TOML is missing worker.{key}"
            )

    try:
        spec = _worker_spec(worker)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "installed Worker resource/accelerator contract is invalid"
        ) from exc

    max_concurrency = int(worker.get("max_concurrency", 1))
    if max_concurrency <= 0:
        raise RuntimeError("installed Worker max_concurrency must be positive")
    gpu_preflight = worker.get("gpu_preflight", True)
    health_host = str(worker.get("health_host", "127.0.0.1")).strip()
    if not health_host:
        raise RuntimeError("installed Worker health_host must not be blank")
    health_port = int(worker.get("health_port", 9100))
    if not 1 <= health_port <= 65535:
        raise RuntimeError("installed Worker health_port is invalid")

    top_level = set(parsed)
    executor = dict(parsed.get("executor") or {})
    runtime = dict(parsed.get("runtime") or {})

    if executor and runtime:
        raise RuntimeError(
            "installed Worker mixes executor and runtime execution authority"
        )

    if executor:
        if top_level != {"worker", "executor"}:
            raise RuntimeError(
                "installed smoke Worker contains unrecognized top-level configuration"
            )
        if set(executor) != {"factory", "settings"}:
            raise RuntimeError(
                "installed smoke executor differs from the canonical contract"
            )
        if executor.get("factory") != SMOKE_EXECUTOR_FACTORY:
            raise RuntimeError(
                "installed Worker uses an unrecognized executor factory"
            )
        if dict(executor.get("settings") or {}):
            raise RuntimeError(
                "installed smoke executor settings are not canonical"
            )
        if spec.capabilities != frozenset({"debug.echo"}):
            raise RuntimeError(
                "installed smoke Worker capabilities are not canonical"
            )
        if not text.endswith(_SMOKE_SUFFIX):
            raise RuntimeError(
                "installed smoke Worker TOML differs from the canonical rendered form"
            )
        if text.count(_SMOKE_CAPABILITIES_LINE) != 1:
            raise RuntimeError(
                "installed smoke Worker capability line is not canonical"
            )
        return InstalledWorkerContract(
            spec=spec,
            control_url=str(worker["control_url"]),
            max_concurrency=max_concurrency,
            gpu_preflight=gpu_preflight,
            health_host=health_host,
            health_port=health_port,
            execution_mode="smoke",
        )

    if runtime:
        if top_level != {"worker", "runtime"}:
            raise RuntimeError(
                "installed runtime Worker contains unrecognized top-level configuration"
            )
        expected_runtime_keys = {
            "manifest",
            "startup_timeout_seconds",
            "shutdown_timeout_seconds",
        }
        if set(runtime) != expected_runtime_keys:
            raise RuntimeError(
                "installed runtime section differs from the canonical contract"
            )
        manifest = str(runtime.get("manifest") or "").strip()
        if manifest != DEFAULT_RUNTIME_MANIFEST:
            raise RuntimeError(
                "installed runtime manifest path is not canonical"
            )
        if int(runtime.get("startup_timeout_seconds", 0)) != (
            DEFAULT_RUNTIME_STARTUP_TIMEOUT_SECONDS
        ):
            raise RuntimeError(
                "installed runtime startup timeout is not canonical"
            )
        if int(runtime.get("shutdown_timeout_seconds", 0)) != (
            DEFAULT_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS
        ):
            raise RuntimeError(
                "installed runtime shutdown timeout is not canonical"
            )
        if spec.capabilities != RUNTIME_CAPABILITIES:
            raise RuntimeError(
                "installed runtime Worker capabilities are not canonical"
            )
        expected_suffix = _RUNTIME_SUFFIX_TEMPLATE.format(manifest=manifest)
        if not text.endswith(expected_suffix):
            raise RuntimeError(
                "installed runtime Worker TOML differs from the canonical rendered form"
            )
        if text.count(_RUNTIME_CAPABILITIES_LINE) != 1:
            raise RuntimeError(
                "installed runtime Worker capability line is not canonical"
            )
        return InstalledWorkerContract(
            spec=spec,
            control_url=str(worker["control_url"]),
            max_concurrency=max_concurrency,
            gpu_preflight=gpu_preflight,
            health_host=health_host,
            health_port=health_port,
            execution_mode="runtime",
            runtime_manifest=manifest,
        )

    raise RuntimeError(
        "installed Worker has neither canonical smoke nor runtime execution authority"
    )


def render_runtime_worker_toml(
    text: str,
    *,
    runtime_manifest: str = DEFAULT_RUNTIME_MANIFEST,
) -> str:
    contract = parse_installed_worker_toml(text)
    if runtime_manifest != DEFAULT_RUNTIME_MANIFEST:
        raise RuntimeError("runtime manifest path is not canonical")

    if contract.execution_mode == "runtime":
        if contract.runtime_manifest != runtime_manifest:
            raise RuntimeError(
                "installed runtime Worker points at a different manifest"
            )
        return text

    prefix = text[: -len(_SMOKE_SUFFIX)]
    if prefix.count(_SMOKE_CAPABILITIES_LINE) != 1:
        raise RuntimeError("canonical smoke capability line is missing")
    prefix = prefix.replace(
        _SMOKE_CAPABILITIES_LINE,
        _RUNTIME_CAPABILITIES_LINE,
        1,
    )
    rendered = prefix + _RUNTIME_SUFFIX_TEMPLATE.format(
        manifest=runtime_manifest
    )
    migrated = parse_installed_worker_toml(rendered)
    if migrated.execution_mode != "runtime":
        raise RuntimeError("rendered runtime Worker contract is invalid")
    return rendered


def parse_installed_worker_unit(text: str) -> InstalledWorkerUnit:
    if not text.startswith(_UNIT_PREFIX) or not text.endswith(_UNIT_SUFFIX):
        raise RuntimeError(
            "installed Worker unit differs from the canonical template"
        )

    lines = text.splitlines()
    user_lines = [line for line in lines if line.startswith("User=")]
    group_lines = [line for line in lines if line.startswith("Group=")]
    supplementary = [
        line
        for line in lines
        if line.startswith("SupplementaryGroups=")
    ]
    env_lines = [line for line in lines if line.startswith("EnvironmentFile=")]
    runtime_env_lines = [
        line for line in lines if line.startswith("Environment=")
    ]
    preflight_lines = [line for line in lines if line.startswith("ExecStartPre=")]
    exec_lines = [line for line in lines if line.startswith("ExecStart=")]
    if not (
        len(user_lines)
        == len(group_lines)
        == len(supplementary)
        == len(env_lines)
        == len(preflight_lines)
        == len(exec_lines)
        == 1
    ):
        raise RuntimeError("installed Worker unit has non-canonical directives")

    service_user = user_lines[0].partition("=")[2].strip()
    service_group = group_lines[0].partition("=")[2].strip()
    if not service_user or service_group != service_user:
        raise RuntimeError("installed Worker unit user/group is not canonical")
    if supplementary[0] != "SupplementaryGroups=astrumweaver-config":
        raise RuntimeError(
            "installed Worker unit supplementary config group is not canonical"
        )
    if env_lines[0] != "EnvironmentFile=-/etc/astrumweaver/worker.env":
        raise RuntimeError(
            "installed Worker unit environment reference is not canonical"
        )
    if preflight_lines[0] != (
        "ExecStartPre=+/usr/local/libexec/astrumweaver/gpu-preflight "
        "/etc/astrumweaver/gpu-uuids"
    ):
        raise RuntimeError(
            "installed Worker unit GPU preflight is not canonical"
        )

    match = _EXEC_START_RE.fullmatch(exec_lines[0])
    if match is None:
        raise RuntimeError(
            "installed Worker unit ExecStart is not a recognized canonical form"
        )
    executable = match.group("executable")
    if executable.startswith("/nix/store/"):
        raise RuntimeError(
            "installed Worker unit is store-pinned rather than using a stable executable path"
        )
    execution_mode = (
        "runtime" if match.group("runtime_arg") is not None else "smoke"
    )
    expected_runtime_env = (
        [_RUNTIME_DRIVER_ENV_LINE]
        if execution_mode == "runtime"
        else []
    )
    if runtime_env_lines != expected_runtime_env:
        raise RuntimeError(
            "installed Worker unit runtime driver environment is not canonical"
        )
    runtime_arg = (
        " --runtime-manifest /etc/astrumweaver/runtime-deployment.json"
        if execution_mode == "runtime"
        else ""
    )
    expected = (
        _UNIT_PREFIX
        + f"User={service_user}\n"
        + f"Group={service_user}\n"
        + "SupplementaryGroups=astrumweaver-config\n"
        + "EnvironmentFile=-/etc/astrumweaver/worker.env\n"
        + (
            _RUNTIME_DRIVER_ENV_LINE + "\n"
            if execution_mode == "runtime"
            else ""
        )
        + (
            "ExecStartPre=+/usr/local/libexec/astrumweaver/gpu-preflight "
            "/etc/astrumweaver/gpu-uuids\n"
        )
        + (
            f"ExecStart={executable} --config "
            f"/etc/astrumweaver/worker.toml{runtime_arg}\n"
        )
        + _UNIT_SUFFIX
    )
    if text != expected:
        raise RuntimeError(
            "installed Worker unit contains unrelated or reordered modifications"
        )
    return InstalledWorkerUnit(
        service_user=service_user,
        executable=executable,
        execution_mode=execution_mode,
    )


def render_runtime_worker_unit(text: str) -> str:
    unit = parse_installed_worker_unit(text)
    if unit.execution_mode == "runtime":
        return text
    old = (
        f"ExecStart={unit.executable} --config "
        "/etc/astrumweaver/worker.toml"
    )
    new = old + (
        " --runtime-manifest "
        "/etc/astrumweaver/runtime-deployment.json"
    )
    if text.count(old) != 1:
        raise RuntimeError("canonical Worker ExecStart is ambiguous")
    environment_anchor = (
        "EnvironmentFile=-/etc/astrumweaver/worker.env\n"
    )
    if text.count(environment_anchor) != 1:
        raise RuntimeError(
            "canonical Worker EnvironmentFile directive is ambiguous"
        )
    rendered = text.replace(
        environment_anchor,
        environment_anchor + _RUNTIME_DRIVER_ENV_LINE + "\n",
        1,
    )
    rendered = rendered.replace(old, new, 1)
    migrated = parse_installed_worker_unit(rendered)
    if migrated.execution_mode != "runtime":
        raise RuntimeError("rendered Worker unit is invalid")
    if migrated.executable != unit.executable:
        raise RuntimeError("Worker executable path changed during migration")
    return rendered


__all__ = [
    "DEFAULT_RUNTIME_MANIFEST",
    "NVIDIA_DRIVER_BRIDGE_DIRECTORY",
    "InstalledWorkerContract",
    "InstalledWorkerUnit",
    "RUNTIME_CAPABILITIES",
    "SMOKE_EXECUTOR_FACTORY",
    "parse_installed_worker_toml",
    "parse_installed_worker_unit",
    "render_runtime_worker_toml",
    "render_runtime_worker_unit",
]
