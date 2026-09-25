"""Local host discovery inputs for runtime setup planning."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..runtime import RuntimeHostFacts
from .contracts import DeploymentPath, PrivilegeMode, SetupHostSnapshot


class HostDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DiscoveredGpu:
    uuid: str
    memory_mb: int
    compute_capability: str | None = None

    def __post_init__(self) -> None:
        normalized_uuid = self.uuid.strip()
        if not normalized_uuid:
            raise ValueError("GPU UUID must not be blank")
        if self.memory_mb <= 0:
            raise ValueError("GPU memory must be positive")
        object.__setattr__(self, "uuid", normalized_uuid)
        if self.compute_capability is not None:
            value = self.compute_capability.strip()
            object.__setattr__(
                self,
                "compute_capability",
                value or None,
            )


def _run_nvidia_query(
    command: str,
    fields: tuple[str, ...],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[tuple[str, ...], ...]:
    try:
        completed = run(
            [
                command,
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HostDiscoveryError("nvidia-smi GPU discovery failed") from exc

    rows: list[tuple[str, ...]] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        values = tuple(value.strip() for value in raw_line.split(","))
        if len(values) != len(fields) or any(not value for value in values):
            raise HostDiscoveryError("nvidia-smi returned malformed GPU discovery data")
        rows.append(values)
    return tuple(rows)


def discover_local_gpus(
    *,
    command: str = "nvidia-smi",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[DiscoveredGpu, ...]:
    """Discover local NVIDIA GPU facts without exposing them in host metadata.

    UUID/device ordering follows nvidia-smi so a user-selected multi-GPU order
    can be preserved for runtime placement. Compute capability is best-effort:
    older drivers that do not expose the query still provide UUID/VRAM facts.
    """

    try:
        rows = _run_nvidia_query(
            command,
            ("uuid", "memory.total", "compute_cap"),
            run=run,
        )
    except HostDiscoveryError:
        rows = _run_nvidia_query(
            command,
            ("uuid", "memory.total"),
            run=run,
        )
        if not rows:
            return ()
        parsed: list[DiscoveredGpu] = []
        for uuid, memory_raw in rows:
            try:
                memory_mb = int(memory_raw)
            except ValueError as exc:
                raise HostDiscoveryError(
                    "nvidia-smi returned invalid GPU memory"
                ) from exc
            parsed.append(DiscoveredGpu(uuid=uuid, memory_mb=memory_mb))
        return tuple(parsed)

    parsed = []
    for uuid, memory_raw, compute_capability in rows:
        try:
            memory_mb = int(memory_raw)
        except ValueError as exc:
            raise HostDiscoveryError(
                "nvidia-smi returned invalid GPU memory"
            ) from exc
        parsed.append(
            DiscoveredGpu(
                uuid=uuid,
                memory_mb=memory_mb,
                compute_capability=compute_capability,
            )
        )
    return tuple(parsed)


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HostDiscoveryError(f"cannot read host metadata: {path}") from exc

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        value = raw_value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _read_host_ram_mb(path: Path) -> int:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HostDiscoveryError("cannot read host RAM information") from exc

    for raw_line in content.splitlines():
        if not raw_line.startswith("MemTotal:"):
            continue
        fields = raw_line.split()
        if len(fields) < 2:
            break
        try:
            kib = int(fields[1])
        except ValueError as exc:
            raise HostDiscoveryError("invalid MemTotal value") from exc
        mib = kib // 1024
        if mib <= 0:
            raise HostDiscoveryError("host RAM must be positive")
        return mib

    raise HostDiscoveryError("MemTotal was not found")


def discover_local_host(
    *,
    os_release_path: Path = Path("/etc/os-release"),
    meminfo_path: Path = Path("/proc/meminfo"),
    nixos_marker_path: Path = Path("/etc/NIXOS"),
    which: Callable[[str], str | None] = shutil.which,
    cpu_count: Callable[[], int | None] = os.cpu_count,
    geteuid: Callable[[], int] = os.geteuid,
    machine: Callable[[], str] = platform.machine,
) -> SetupHostSnapshot:
    """Discover non-network host facts used by setup planning.

    Hostname, addresses, routes and hypervisor inventory are intentionally not
    inspected here.
    """

    release = _read_key_value_file(os_release_path)
    os_id = (release.get("ID") or "linux").strip().lower()
    os_version = (release.get("VERSION_ID") or "").strip()

    count = cpu_count()
    if count is None or count <= 0:
        raise HostDiscoveryError("CPU count is unavailable")

    ram_mb = _read_host_ram_mb(meminfo_path)
    architecture = machine().strip()
    if not architecture:
        raise HostDiscoveryError("architecture is unavailable")

    command_candidates = (
        "systemctl",
        "nix",
        "nvidia-smi",
        "sudo",
        "apt-get",
        "dnf",
        "pacman",
        "zypper",
    )
    available = frozenset(
        command
        for command in command_candidates
        if which(command) is not None
    )

    is_nixos = os_id == "nixos" or nixos_marker_path.exists()
    if is_nixos:
        deployment_path = DeploymentPath.NIXOS
        package_manager = "nix" if "nix" in available else None
    else:
        if "systemctl" not in available:
            raise HostDiscoveryError(
                "v0.x generic Linux setup requires systemd/systemctl"
            )
        deployment_path = DeploymentPath.SYSTEMD
        package_manager = next(
            (
                command
                for command in ("apt-get", "dnf", "pacman", "zypper")
                if command in available
            ),
            None,
        )

    service_manager = "systemd" if "systemctl" in available else None

    if geteuid() == 0:
        privilege_mode = PrivilegeMode.ROOT
    elif "sudo" in available:
        privilege_mode = PrivilegeMode.SUDO
    else:
        privilege_mode = PrivilegeMode.UNAVAILABLE

    return SetupHostSnapshot(
        runtime_host=RuntimeHostFacts(
            cpu_count=count,
            host_ram_mb=ram_mb,
            architecture=architecture,
        ),
        deployment_path=deployment_path,
        os_id=os_id,
        os_version=os_version,
        service_manager=service_manager,
        package_manager=package_manager,
        available_commands=available,
        privilege_mode=privilege_mode,
        metadata={},
    )
