"""Provider-neutral worker and job resource contracts.

These contracts intentionally describe scheduler-visible facts rather than
application-specific workload behavior. A worker may own one or multiple
GPUs, and a job describes what resource shape and capabilities it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _require_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _unique_strings(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = tuple(values or ())
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        item = str(value).strip()
        if not item:
            raise ValueError(f"{field_name} must not contain blank values")
        if item in seen:
            raise ValueError(f"{field_name} must not contain duplicates")
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalize_labels(labels: Mapping[str, str] | None) -> Mapping[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in dict(labels or {}).items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key or not value:
            raise ValueError("labels must use non-blank keys and values")
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ResourceShape:
    """Scheduler-visible GPU resource shape.

    total_vram_mb is the sum across GPUs owned by the worker.
    max_single_gpu_vram_mb is the largest contiguous per-device capacity.

    Keeping both values prevents a 2 x 12 GiB worker from masquerading as a
    1 x 24 GiB worker for jobs that require one large device.
    """

    gpu_count: int = 0
    total_vram_mb: int = 0
    max_single_gpu_vram_mb: int = 0

    def __post_init__(self) -> None:
        for field_name in ("gpu_count", "total_vram_mb", "max_single_gpu_vram_mb"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")

        if self.gpu_count == 0:
            if self.total_vram_mb != 0 or self.max_single_gpu_vram_mb != 0:
                raise ValueError("GPU VRAM must be zero when gpu_count is zero")
            return

        if self.total_vram_mb <= 0:
            raise ValueError("total_vram_mb must be positive when GPUs are present")
        if self.max_single_gpu_vram_mb <= 0:
            raise ValueError("max_single_gpu_vram_mb must be positive when GPUs are present")
        if self.max_single_gpu_vram_mb > self.total_vram_mb:
            raise ValueError("max_single_gpu_vram_mb must not exceed total_vram_mb")


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """One scheduler-visible exclusive compute unit."""

    worker_id: str
    worker_class: str
    resources: ResourceShape = field(default_factory=ResourceShape)
    gpu_uuids: tuple[str, ...] = ()
    capabilities: frozenset[str] = field(default_factory=frozenset)
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _require_nonblank(self.worker_id, "worker_id"))
        object.__setattr__(
            self,
            "worker_class",
            _require_nonblank(self.worker_class, "worker_class"),
        )

        gpu_uuids = _unique_strings(self.gpu_uuids, "gpu_uuids")
        object.__setattr__(self, "gpu_uuids", gpu_uuids)

        capabilities = frozenset(_unique_strings(self.capabilities, "capabilities"))
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "labels", _normalize_labels(self.labels))

        if len(gpu_uuids) != self.resources.gpu_count:
            raise ValueError(
                "gpu_uuids count must equal resources.gpu_count for a GPU worker"
            )


@dataclass(frozen=True, slots=True)
class JobRequirements:
    """Generic worker constraints attached to a schedulable job."""

    worker_class: str | None = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    required_labels: Mapping[str, str] = field(default_factory=dict)
    required_gpu_uuids: frozenset[str] = field(default_factory=frozenset)
    min_gpu_count: int = 0
    min_total_vram_mb: int = 0
    min_single_gpu_vram_mb: int = 0

    def __post_init__(self) -> None:
        if self.worker_class is not None:
            object.__setattr__(
                self,
                "worker_class",
                _require_nonblank(self.worker_class, "worker_class"),
            )

        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(_unique_strings(self.required_capabilities, "required_capabilities")),
        )
        object.__setattr__(
            self,
            "required_gpu_uuids",
            frozenset(_unique_strings(self.required_gpu_uuids, "required_gpu_uuids")),
        )
        object.__setattr__(self, "required_labels", _normalize_labels(self.required_labels))

        for field_name in (
            "min_gpu_count",
            "min_total_vram_mb",
            "min_single_gpu_vram_mb",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")
