"""Private-safe real-host acceptance for RuntimeProvider GPU subset isolation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol

import httpx

from .hardware import REVISION_PATTERN


class RuntimeDeploymentAcceptanceError(RuntimeError):
    pass


def resolve_packaged_gpu_device_map(
    *,
    argv0: str | None = None,
    installed_path: Path = Path(
        "/usr/local/libexec/astrumweaver/gpu-device-map"
    ),
) -> str:
    """Resolve the reviewed mapper without consulting ambient ``PATH``."""

    candidates = [installed_path]
    invocation = argv0 if argv0 is not None else sys.argv[0]
    if os.sep in invocation:
        candidates.append(
            Path(invocation).absolute().parent
            / "astrumweaver-gpu-device-map"
        )
        candidates.append(
            Path(invocation).resolve().parent
            / "astrumweaver-gpu-device-map"
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeDeploymentAcceptanceError(
        "reviewed packaged GPU mapper is unavailable; supply "
        "--gpu-device-map explicitly"
    )


class RuntimeDeploymentHost(Protocol):
    def host_gpu_device_map(self) -> Mapping[str, str]: ...

    def worker_gpu_contract(self) -> tuple[tuple[str, ...], bool]: ...

    def worker_unit_properties(self) -> Mapping[str, str]: ...

    def probe_isolation_enforcement(
        self,
        host_map: Mapping[str, str],
        expected_gpu_uuids: tuple[str, ...],
    ) -> bool: ...

    def service_active(self) -> bool: ...

    def start_service(self) -> None: ...

    def stop_service(self) -> None: ...

    def readiness(self) -> Mapping[str, object] | None: ...


class SystemdRuntimeDeploymentHost:
    def __init__(
        self,
        *,
        service: str,
        worker_config: Path,
        systemctl: str = "systemctl",
        systemd_run: str = "systemd-run",
        nvidia_smi: str = "nvidia-smi",
        gpu_device_map: str | None = None,
        health_url: str = "http://127.0.0.1:9100/health",
    ) -> None:
        self.service = service
        self.worker_config = worker_config
        self.systemctl = systemctl
        self.systemd_run = systemd_run
        self.nvidia_smi = nvidia_smi
        if gpu_device_map is not None:
            mapper_path = Path(gpu_device_map)
            if not mapper_path.is_absolute():
                raise RuntimeDeploymentAcceptanceError(
                    "--gpu-device-map must be an absolute executable path"
                )
            if not mapper_path.is_file() or not os.access(mapper_path, os.X_OK):
                raise RuntimeDeploymentAcceptanceError(
                    "--gpu-device-map must name an existing executable file"
                )
            self.gpu_device_map = gpu_device_map
        else:
            self.gpu_device_map = resolve_packaged_gpu_device_map()
        self.health_url = health_url.rstrip("/")

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                check=check,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeDeploymentAcceptanceError(
                "required host command failed"
            ) from exc

    def host_gpu_device_map(self) -> Mapping[str, str]:
        completed = self._run(
            [
                self.gpu_device_map,
                "--nvidia-smi",
                self.nvidia_smi,
                "discover-visible",
            ]
        )
        result: dict[str, str] = {}
        for raw_line in completed.stdout.splitlines():
            fields = [item.strip() for item in raw_line.split("=", 1)]
            if (
                len(fields) != 2
                or not fields[0]
                or not _GPU_DEVICE_PATH_RE.fullmatch(fields[1])
            ):
                raise RuntimeDeploymentAcceptanceError(
                    "host GPU UUID/device mapping is unavailable"
                )
            uuid, device_path = fields
            if uuid in result:
                raise RuntimeDeploymentAcceptanceError(
                    "host reported duplicate GPU UUID"
                )
            if device_path in result.values():
                raise RuntimeDeploymentAcceptanceError(
                    "host reported duplicate GPU device minor"
                )
            result[uuid] = device_path
        if not result:
            raise RuntimeDeploymentAcceptanceError(
                "host reported no NVIDIA GPUs"
            )
        return result

    def worker_gpu_contract(self) -> tuple[tuple[str, ...], bool]:
        try:
            with self.worker_config.open("rb") as handle:
                config = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise RuntimeDeploymentAcceptanceError(
                "Worker configuration is unavailable"
            ) from exc
        worker = dict(config.get("worker") or {})
        uuids = tuple(str(item) for item in worker.get("gpu_uuids", ()))
        preflight = bool(worker.get("gpu_preflight", True))
        return uuids, preflight

    def worker_unit_properties(self) -> Mapping[str, str]:
        completed = self._run(
            [
                self.systemctl,
                "show",
                "--no-pager",
                "--all",
                "--property=DevicePolicy,DeviceAllow,Environment,EnvironmentFiles,UnsetEnvironment,ExecStartPre,Requires,After",
                self.service,
            ]
        )
        return parse_systemd_show_properties(completed.stdout)

    def probe_isolation_enforcement(
        self,
        host_map: Mapping[str, str],
        expected_gpu_uuids: tuple[str, ...],
    ) -> bool:
        if len(host_map) <= len(expected_gpu_uuids):
            raise RuntimeDeploymentAcceptanceError(
                "isolation capability probe requires a GPU superset"
            )
        selected_paths = []
        for uuid in expected_gpu_uuids:
            path = host_map.get(uuid)
            if path is None:
                raise RuntimeDeploymentAcceptanceError(
                    "selected GPU UUID is not visible on the host"
                )
            selected_paths.append(path)

        with tempfile.TemporaryDirectory(
            prefix="astrumweaver-gpu-isolation-"
        ) as raw_tmp:
            tmp = Path(raw_tmp)
            expected_path = tmp / "expected-uuids"
            visible_map_path = tmp / "visible-map"
            expected_path.write_text(
                "".join(f"{uuid}\n" for uuid in expected_gpu_uuids),
                encoding="utf-8",
            )
            visible_map_path.write_text(
                "".join(
                    f"{uuid}={path}\n"
                    for uuid, path in sorted(host_map.items())
                ),
                encoding="utf-8",
            )

            command = [
                self.systemd_run,
                "--quiet",
                "--wait",
                "--pipe",
                "--collect",
                "--property=Type=oneshot",
                "--property=DevicePolicy=closed",
            ]
            command.extend(
                f"--property=DeviceAllow={path} rw"
                for path in selected_paths
            )
            for path in _NVIDIA_AUXILIARY_DEVICE_PATHS:
                if Path(path).exists():
                    command.append(f"--property=DeviceAllow={path} rw")
            command.extend(
                [
                    self.gpu_device_map,
                    "probe-access",
                    str(expected_path),
                    str(visible_map_path),
                ]
            )
            completed = self._run(command, check=False)
            if completed.returncode == 0:
                return True
            if completed.returncode == 3:
                return False
            raise RuntimeDeploymentAcceptanceError(
                "GPU isolation capability probe failed"
            )

    def service_active(self) -> bool:
        completed = self._run(
            [self.systemctl, "is-active", "--quiet", self.service],
            check=False,
        )
        if completed.returncode == 0:
            return True
        if completed.returncode == 3:
            return False
        raise RuntimeDeploymentAcceptanceError(
            "cannot determine Worker service state"
        )

    def start_service(self) -> None:
        self._run([self.systemctl, "start", self.service])

    def stop_service(self) -> None:
        self._run([self.systemctl, "stop", self.service])

    def readiness(self) -> Mapping[str, object] | None:
        try:
            response = httpx.get(self.health_url, timeout=3.0)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None


@dataclass(frozen=True, slots=True)
class RuntimeDeploymentAcceptanceEvidence:
    evidence_version: str
    date_utc: str
    astrumweaver_revision: str
    deployment_path: str
    host_gpu_count: int
    selected_gpu_count: int
    private_values_omitted: bool
    host_gpu_superset: str
    worker_contract_exact: str
    worker_exact_set_preflight_enabled: str
    uuid_device_mapping_verified: str
    device_policy_closed: str
    selected_device_allow_exact: str
    cuda_visible_devices_exact: str
    in_service_exact_set_gate_present: str
    worker_started_ready: str
    worker_registered: str
    service_stopped_after_acceptance: str
    isolation_enforcement: str
    fail_closed: str
    overall: str


_GPU_DEVICE_PATH_RE = re.compile(r"^/dev/nvidia[0-9]+$")
_GPU_ALLOWED_DEVICE_PATH_RE = re.compile(
    r"^/dev/nvidia(?:[0-9]+|[-a-zA-Z0-9_/]+)$"
)
_NVIDIA_AUXILIARY_DEVICE_PATHS = (
    "/dev/nvidiactl",
    "/dev/nvidia-modeset",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-nvswitchctl",
)
_NIX_STORE_HASH = r"[a-z0-9]{32}"
_NIX_GPU_PREFLIGHT_RE = re.compile(
    rf"^/nix/store/{_NIX_STORE_HASH}-astrumweaver-gpu-preflight/"
    r"bin/astrumweaver-gpu-preflight$"
)
_NIX_GPU_UUIDS_RE = re.compile(
    rf"^/nix/store/{_NIX_STORE_HASH}-astrumweaver-gpu-uuids$"
)
_NIX_GPU_DEVICE_MAP_RE = re.compile(
    rf"^/nix/store/{_NIX_STORE_HASH}-astrumweaver-gpu-device-map/"
    r"bin/astrumweaver-gpu-device-map$"
)
_GPU_ISOLATION_VISIBLE_MAP = (
    "/run/astrumweaver-worker-gpu-isolation/gpu-visible-map"
)
_SYSTEMD_ACCEPTANCE_PROPERTIES = frozenset(
    {
        "DevicePolicy",
        "DeviceAllow",
        "Environment",
        "EnvironmentFiles",
        "UnsetEnvironment",
        "ExecStartPre",
        "Requires",
        "After",
    }
)


def parse_systemd_show_properties(output: str) -> Mapping[str, str]:
    """Parse selected effective properties emitted by ``systemctl show``."""

    properties: dict[str, str] = {}
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        if not separator or name not in _SYSTEMD_ACCEPTANCE_PROPERTIES:
            continue
        if name in properties:
            if name == "DevicePolicy":
                raise RuntimeDeploymentAcceptanceError(
                    "systemd returned duplicate DevicePolicy properties"
                )
            properties[name] = f"{properties[name]} {value}"
        else:
            properties[name] = value

    missing = _SYSTEMD_ACCEPTANCE_PROPERTIES - properties.keys()
    if missing:
        raise RuntimeDeploymentAcceptanceError(
            "systemd effective Worker properties are unavailable"
        )
    return properties


def _split_systemd_list(value: str, property_name: str) -> list[str]:
    try:
        return shlex.split(value)
    except ValueError as exc:
        raise RuntimeDeploymentAcceptanceError(
            f"systemd returned malformed {property_name} property"
        ) from exc


def _parse_device_allow(value: str) -> list[tuple[str, str]]:
    fields = _split_systemd_list(value, "DeviceAllow")
    if len(fields) % 2:
        raise RuntimeDeploymentAcceptanceError(
            "systemd returned malformed DeviceAllow property"
        )
    return list(zip(fields[::2], fields[1::2], strict=True))


def _has_exact_set_preflight(value: str) -> bool:
    for record in re.findall(r"\{([^{}]*)\}", value):
        properties = {
            name.strip(): field_value.strip()
            for field in record.split(";")
            for name, separator, field_value in [field.partition("=")]
            if separator
        }
        executable = properties.get("path", "")
        argv_text = properties.get("argv[]", "")
        if properties.get("ignore_errors") != "no":
            continue
        try:
            argv = shlex.split(argv_text)
        except ValueError:
            continue
        if not argv or argv[0] != executable:
            continue

        if (
            len(argv) == 2
            and executable == "/usr/local/libexec/astrumweaver/gpu-preflight"
            and argv[1] == "/etc/astrumweaver/gpu-uuids"
        ):
            return True
        if (
            len(argv) == 2
            and _NIX_GPU_PREFLIGHT_RE.fullmatch(executable)
            and _NIX_GPU_UUIDS_RE.fullmatch(argv[1])
        ):
            return True

        generic_mapper = (
            executable == "/usr/local/libexec/astrumweaver/gpu-device-map"
        )
        nix_mapper = _NIX_GPU_DEVICE_MAP_RE.fullmatch(executable) is not None
        if (
            len(argv) == 4
            and (generic_mapper or nix_mapper)
            and argv[1] == "probe-access"
            and (
                argv[2] == "/etc/astrumweaver/gpu-uuids"
                or _NIX_GPU_UUIDS_RE.fullmatch(argv[2])
            )
            and argv[3] == _GPU_ISOLATION_VISIBLE_MAP
        ):
            return True
    return False


def _reject_effective_cuda_environment_override(
    properties: Mapping[str, str],
) -> None:
    unset_environment = _split_systemd_list(
        properties.get("UnsetEnvironment", ""), "UnsetEnvironment"
    )
    reserved_environment = {
        "CUDA_VISIBLE_DEVICES",
        "ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP",
    }
    if any(
        item.partition("=")[0] in reserved_environment
        for item in unset_environment
    ):
        raise RuntimeDeploymentAcceptanceError(
            "Worker GPU isolation environment is unset by effective systemd policy"
        )

    environment_files = _split_systemd_list(
        properties.get("EnvironmentFiles", ""), "EnvironmentFiles"
    )
    for item in environment_files:
        if item.startswith("(") and item.endswith(")"):
            continue
        if item.startswith("-/"):
            path = Path(item[1:])
        elif item.startswith("/"):
            path = Path(item)
        else:
            raise RuntimeDeploymentAcceptanceError(
                "systemd returned malformed EnvironmentFiles property"
            )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as exc:
            raise RuntimeDeploymentAcceptanceError(
                "Worker EnvironmentFile could not be checked for GPU overrides"
            ) from exc
        if any(
            re.match(
                r"^\s*(?:CUDA_VISIBLE_DEVICES|"
                r"ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP)\s*=",
                line,
            )
            for line in lines
            if not line.lstrip().startswith(("#", ";"))
        ):
            raise RuntimeDeploymentAcceptanceError(
                "Worker GPU isolation environment may be overridden by an EnvironmentFile"
            )


def validate_effective_worker_gpu_isolation(
    properties: Mapping[str, str],
    *,
    expected_gpu_uuids: tuple[str, ...],
    expected_device_paths: set[str],
) -> None:
    """Require effective systemd properties to retain the GPU isolation gate."""

    if properties.get("DevicePolicy") != "closed":
        raise RuntimeDeploymentAcceptanceError(
            "Worker service does not use DevicePolicy=closed"
        )

    device_allow = _parse_device_allow(properties.get("DeviceAllow", ""))
    if any(
        not _GPU_ALLOWED_DEVICE_PATH_RE.fullmatch(path)
        or permissions != "rw"
        for path, permissions in device_allow
    ):
        raise RuntimeDeploymentAcceptanceError(
            "Worker DeviceAllow includes a broad or unsupported device grant"
        )
    allowed_physical = [
        (path, permissions)
        for path, permissions in device_allow
        if _GPU_DEVICE_PATH_RE.fullmatch(path)
    ]
    expected_physical = [(path, "rw") for path in sorted(expected_device_paths)]
    if sorted(allowed_physical) != expected_physical:
        raise RuntimeDeploymentAcceptanceError(
            "Worker DeviceAllow physical GPU set is not exact"
        )

    environment = _split_systemd_list(
        properties.get("Environment", ""), "Environment"
    )
    visible_matches = [
        assignment.partition("=")[2]
        for assignment in environment
        if assignment.startswith("CUDA_VISIBLE_DEVICES=")
    ]
    if visible_matches != [",".join(expected_gpu_uuids)]:
        raise RuntimeDeploymentAcceptanceError(
            "Worker CUDA_VISIBLE_DEVICES does not preserve selected GPU order"
        )
    isolation_map_matches = [
        assignment.partition("=")[2]
        for assignment in environment
        if assignment.startswith(
            "ASTRUMWEAVER_GPU_ISOLATION_VISIBLE_MAP="
        )
    ]
    if isolation_map_matches != [_GPU_ISOLATION_VISIBLE_MAP]:
        raise RuntimeDeploymentAcceptanceError(
            "Worker isolated GPU map authority is missing or ambiguous"
        )
    _reject_effective_cuda_environment_override(properties)

    if not _has_exact_set_preflight(properties.get("ExecStartPre", "")):
        raise RuntimeDeploymentAcceptanceError(
            "Worker service lacks in-cgroup GPU access preflight"
        )

    isolation_preflight = "astrumweaver-worker-gpu-isolation-preflight.service"
    for property_name in ("Requires", "After"):
        dependencies = _split_systemd_list(
            properties.get(property_name, ""), property_name
        )
        if isolation_preflight not in dependencies:
            raise RuntimeDeploymentAcceptanceError(
                f"Worker service lacks effective {property_name} isolation preflight"
            )


class RuntimeDeploymentAcceptanceRunner:
    def __init__(
        self,
        *,
        host: RuntimeDeploymentHost,
        expected_gpu_uuids: tuple[str, ...],
        revision: str,
        deployment_path: str,
        start_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not expected_gpu_uuids:
            raise ValueError("at least one expected GPU UUID is required")
        if len(set(expected_gpu_uuids)) != len(expected_gpu_uuids):
            raise ValueError("expected GPU UUIDs must be unique")
        if not REVISION_PATTERN.fullmatch(revision.strip()):
            raise ValueError("revision must be a public git commit SHA")
        if deployment_path not in {"nixos", "systemd"}:
            raise ValueError("deployment_path must be nixos or systemd")
        if start_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("acceptance timeouts must be positive")
        self.host = host
        self.expected_gpu_uuids = expected_gpu_uuids
        self.revision = revision.strip()
        self.deployment_path = deployment_path
        self.start_timeout_seconds = start_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    def _wait_ready(self) -> Mapping[str, object]:
        deadline = self._monotonic() + self.start_timeout_seconds
        while True:
            payload = self.host.readiness()
            if (
                payload is not None
                and payload.get("ready") is True
                and payload.get("registered") is True
            ):
                return payload
            if self._monotonic() >= deadline:
                raise RuntimeDeploymentAcceptanceError(
                    "isolated Worker did not become ready"
                )
            self._sleep(self.poll_interval_seconds)

    def _ensure_service_stopped(self) -> None:
        last_stop_error: Exception | None = None
        for _ in range(2):
            try:
                self.host.stop_service()
                last_stop_error = None
            except Exception as exc:
                last_stop_error = exc
            try:
                if not self.host.service_active():
                    return
            except Exception as exc:
                raise RuntimeDeploymentAcceptanceError(
                    "could not verify Worker service stopped after acceptance"
                ) from exc

        raise RuntimeDeploymentAcceptanceError(
            "Worker service remained active after acceptance cleanup"
        ) from last_stop_error

    def run(self) -> RuntimeDeploymentAcceptanceEvidence:
        host_map = dict(self.host.host_gpu_device_map())
        expected = self.expected_gpu_uuids
        if len(host_map) <= len(expected):
            raise RuntimeDeploymentAcceptanceError(
                "acceptance requires a host-visible GPU superset"
            )
        if any(uuid not in host_map for uuid in expected):
            raise RuntimeDeploymentAcceptanceError(
                "selected GPU UUID is not visible on the host"
            )

        if self.host.service_active():
            raise RuntimeDeploymentAcceptanceError(
                "acceptance requires the Worker service to start from inactive state"
            )

        if not self.host.probe_isolation_enforcement(host_map, expected):
            return RuntimeDeploymentAcceptanceEvidence(
                evidence_version="runtime-deployment-v2",
                date_utc=datetime.now(UTC).date().isoformat(),
                astrumweaver_revision=self.revision,
                deployment_path=self.deployment_path,
                host_gpu_count=len(host_map),
                selected_gpu_count=len(expected),
                private_values_omitted=True,
                host_gpu_superset="PASS",
                worker_contract_exact="NOT_RUN",
                worker_exact_set_preflight_enabled="NOT_RUN",
                uuid_device_mapping_verified="PASS",
                device_policy_closed="NOT_RUN",
                selected_device_allow_exact="NOT_RUN",
                cuda_visible_devices_exact="NOT_RUN",
                in_service_exact_set_gate_present="NOT_RUN",
                worker_started_ready="NOT_RUN",
                worker_registered="NOT_RUN",
                service_stopped_after_acceptance="PASS",
                isolation_enforcement="UNAVAILABLE",
                fail_closed="PASS",
                overall="PASS",
            )

        configured_uuids, gpu_preflight = self.host.worker_gpu_contract()
        if configured_uuids != expected:
            raise RuntimeDeploymentAcceptanceError(
                "Worker GPU UUID order/set does not match acceptance selection"
            )
        if not gpu_preflight:
            raise RuntimeDeploymentAcceptanceError(
                "Worker exact GPU preflight is disabled"
            )

        expected_paths = {host_map[uuid] for uuid in expected}
        validate_effective_worker_gpu_isolation(
            self.host.worker_unit_properties(),
            expected_gpu_uuids=expected,
            expected_device_paths=expected_paths,
        )

        start_attempted = False
        try:
            start_attempted = True
            self.host.start_service()
            ready = self._wait_ready()
            if ready.get("ready") is not True:
                raise RuntimeDeploymentAcceptanceError(
                    "Worker readiness did not prove exact-set startup"
                )
            if ready.get("registered") is not True:
                raise RuntimeDeploymentAcceptanceError(
                    "Worker did not register after isolated startup"
                )
        finally:
            if start_attempted:
                self._ensure_service_stopped()

        if self.host.service_active():
            raise RuntimeDeploymentAcceptanceError(
                "Worker service remained active after acceptance"
            )

        return RuntimeDeploymentAcceptanceEvidence(
            evidence_version="runtime-deployment-v2",
            date_utc=datetime.now(UTC).date().isoformat(),
            astrumweaver_revision=self.revision,
            deployment_path=self.deployment_path,
            host_gpu_count=len(host_map),
            selected_gpu_count=len(expected),
            private_values_omitted=True,
            host_gpu_superset="PASS",
            worker_contract_exact="PASS",
            worker_exact_set_preflight_enabled="PASS",
            uuid_device_mapping_verified="PASS",
            device_policy_closed="PASS",
            selected_device_allow_exact="PASS",
            cuda_visible_devices_exact="PASS",
            in_service_exact_set_gate_present="PASS",
            worker_started_ready="PASS",
            worker_registered="PASS",
            service_stopped_after_acceptance="PASS",
            isolation_enforcement="PASS",
            fail_closed="N/A",
            overall="PASS",
        )


def render_runtime_deployment_markdown(
    evidence: RuntimeDeploymentAcceptanceEvidence,
) -> str:
    fields = [
        ("Evidence version", evidence.evidence_version),
        ("Date (UTC)", evidence.date_utc),
        ("AstrumWeaver revision", evidence.astrumweaver_revision),
        ("Deployment path", evidence.deployment_path),
        ("Host-visible GPU count", str(evidence.host_gpu_count)),
        ("Selected GPU count", str(evidence.selected_gpu_count)),
        ("Private values omitted", str(evidence.private_values_omitted).lower()),
        ("Host-visible GPU superset", evidence.host_gpu_superset),
        ("Worker GPU contract exact", evidence.worker_contract_exact),
        (
            "Worker exact-set preflight enabled",
            evidence.worker_exact_set_preflight_enabled,
        ),
        ("UUID/device mapping verified", evidence.uuid_device_mapping_verified),
        ("DevicePolicy closed", evidence.device_policy_closed),
        ("Selected DeviceAllow exact", evidence.selected_device_allow_exact),
        ("CUDA visible-device order exact", evidence.cuda_visible_devices_exact),
        (
            "In-service exact-set gate present",
            evidence.in_service_exact_set_gate_present,
        ),
        ("Worker started ready", evidence.worker_started_ready),
        ("Worker registered", evidence.worker_registered),
        (
            "Service stopped after acceptance",
            evidence.service_stopped_after_acceptance,
        ),
        ("Isolation enforcement", evidence.isolation_enforcement),
        ("Fail closed", evidence.fail_closed),
        ("Overall", evidence.overall),
    ]
    rows = "\n".join(f"| {key} | {value} |" for key, value in fields)
    return (
        "# AstrumWeaver Runtime Deployment GPU Isolation Evidence\n\n"
        "This file is intentionally redacted. It contains no hostname, IP "
        "address, Worker ID, GPU UUID, device ordinal, credential, private "
        "URL, or GPU model.\n\n"
        "| Field | Result |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )


def write_runtime_deployment_evidence(
    path: Path,
    evidence: RuntimeDeploymentAcceptanceEvidence,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_runtime_deployment_markdown(evidence),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astrumweaver-runtime-deployment-accept"
    )
    parser.add_argument(
        "--gpu-uuid",
        action="append",
        default=[],
        dest="gpu_uuids",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--deployment-path",
        choices=("nixos", "systemd"),
        required=True,
    )
    parser.add_argument(
        "--worker-config",
        type=Path,
        default=Path("/etc/astrumweaver/worker.toml"),
    )
    parser.add_argument(
        "--service",
        default="astrumweaver-worker.service",
    )
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--systemd-run", default="systemd-run")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--gpu-device-map",
        default=None,
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:9100/health",
    )
    parser.add_argument(
        "--start-timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(
            "validation/runtime-deployment/gpu-subset-e2e.md"
        ),
    )
    args = parser.parse_args()

    gpu_uuids = tuple(str(value).strip() for value in args.gpu_uuids)
    if not gpu_uuids or any(not value for value in gpu_uuids):
        parser.exit(
            2,
            "astrumweaver-runtime-deployment-accept: "
            "at least one --gpu-uuid is required\n",
        )

    try:
        host = SystemdRuntimeDeploymentHost(
            service=args.service,
            worker_config=args.worker_config,
            systemctl=args.systemctl,
            systemd_run=args.systemd_run,
            nvidia_smi=args.nvidia_smi,
            gpu_device_map=args.gpu_device_map,
            health_url=args.health_url,
        )
        runner = RuntimeDeploymentAcceptanceRunner(
            host=host,
            expected_gpu_uuids=gpu_uuids,
            revision=args.revision,
            deployment_path=args.deployment_path,
            start_timeout_seconds=args.start_timeout_seconds,
        )
        evidence = runner.run()
        write_runtime_deployment_evidence(args.evidence, evidence)
    except (
        RuntimeDeploymentAcceptanceError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(
            1,
            f"astrumweaver-runtime-deployment-accept: {exc}\n",
        )

    print(json.dumps(asdict(evidence), sort_keys=True))


if __name__ == "__main__":
    main()
