"""Local discovery inputs for runtime setup planning."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .contracts import RuntimeHostFacts


@dataclass(frozen=True, slots=True)
class SetupDiscoverySnapshot:
    host: RuntimeHostFacts
    observed_gpu_uuids: tuple[str, ...] = ()
    installed_provider_ids: frozenset[str] = frozenset()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.host, RuntimeHostFacts):
            raise TypeError("host must be RuntimeHostFacts")

        gpu_uuids = tuple(
            str(value).strip()
            for value in self.observed_gpu_uuids
        )
        if any(not value for value in gpu_uuids):
            raise ValueError("observed_gpu_uuids must not contain blanks")
        if len(gpu_uuids) != len(set(gpu_uuids)):
            raise ValueError("observed_gpu_uuids must not contain duplicates")
        object.__setattr__(self, "observed_gpu_uuids", gpu_uuids)

        providers = frozenset(
            str(value).strip()
            for value in self.installed_provider_ids
        )
        if any(not value for value in providers):
            raise ValueError("installed_provider_ids must not contain blanks")
        object.__setattr__(self, "installed_provider_ids", providers)

        normalized = {
            str(key).strip(): str(value).strip()
            for key, value in dict(self.metadata).items()
        }
        if any(not key or not value for key, value in normalized.items()):
            raise ValueError("metadata keys/values must not be blank")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized),
        )


def _linux_mem_total_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return max(int(fields[1]) // 1024, 1)
    except (OSError, ValueError):
        pass

    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return max((int(page_size) * int(page_count)) // (1024 * 1024), 1)


def discover_local_host_facts() -> RuntimeHostFacts:
    """Discover only generic host facts; no hostname/site topology is read."""

    return RuntimeHostFacts(
        cpu_count=max(os.cpu_count() or 1, 1),
        host_ram_mb=_linux_mem_total_mb(),
        architecture=platform.machine() or "unknown",
    )
