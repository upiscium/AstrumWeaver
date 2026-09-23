"""Generic execution contracts for AstrumWeaver workers.

The control plane schedules jobs by capability and resources. Executors are
the worker-local boundary that turns one generic JobRequest into a JobResult.
Neither contract assumes text generation, models, or a particular accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


def _nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a durable or externally retrievable job artifact."""

    uri: str
    media_type: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _nonblank(self.uri, "uri"))
        if self.media_type is not None:
            object.__setattr__(self, "media_type", _nonblank(self.media_type, "media_type"))
        if self.digest is not None:
            object.__setattr__(self, "digest", _nonblank(self.digest, "digest"))
        if self.name is not None:
            object.__setattr__(self, "name", _nonblank(self.name, "name"))
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class JobRequest:
    """Executor-facing job payload.

    Lease/fencing state belongs to the worker/control lifecycle, not to the
    application payload interpreted by an executor.
    """

    job_id: str
    capability: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _nonblank(self.job_id, "job_id"))
        object.__setattr__(self, "capability", _nonblank(self.capability, "capability"))
        object.__setattr__(self, "payload", _mapping(self.payload))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class JobResult:
    """Provider-neutral result returned by an executor."""

    outputs: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: Mapping[str, int | float] = field(default_factory=dict)
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        artifacts = tuple(self.artifacts)
        if not all(isinstance(artifact, ArtifactRef) for artifact in artifacts):
            raise TypeError("artifacts must contain only ArtifactRef values")

        object.__setattr__(self, "outputs", _mapping(self.outputs))
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or None")
        for key, value in self.metrics.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metrics must use non-blank string keys")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("metric values must be int or float")


@dataclass(frozen=True, slots=True)
class ResidencyItem:
    """One worker-local resource currently kept resident by an executor."""

    name: str
    kind: str
    accelerator_memory_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonblank(self.name, "name"))
        object.__setattr__(self, "kind", _nonblank(self.kind, "kind"))
        if self.accelerator_memory_bytes is not None and self.accelerator_memory_bytes < 0:
            raise ValueError("accelerator_memory_bytes must not be negative")
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ResidencyReport:
    """Point-in-time executor residency snapshot."""

    items: tuple[ResidencyItem, ...] = ()
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not all(isinstance(item, ResidencyItem) for item in items):
            raise TypeError("items must contain only ResidencyItem values")
        if not isinstance(self.observed_at, datetime):
            raise TypeError("observed_at must be a datetime")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def accelerator_memory_bytes(self) -> int:
        return sum(item.accelerator_memory_bytes or 0 for item in self.items)


@runtime_checkable
class JobExecutor(Protocol):
    """Worker-local execution boundary."""

    async def execute(self, job: JobRequest) -> JobResult:
        """Execute one claimed job."""
        ...

    async def cancel(self, job_id: str) -> None:
        """Request cancellation of an in-flight job."""
        ...

    async def residency(self) -> ResidencyReport:
        """Report resources currently kept resident by the executor."""
        ...
