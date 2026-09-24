"""Local host discovery inputs for runtime setup planning."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Callable

from ..runtime import RuntimeHostFacts
from .contracts import DeploymentPath, PrivilegeMode, SetupHostSnapshot


class HostDiscoveryError(RuntimeError):
    pass


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
