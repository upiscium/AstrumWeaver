"""Durable control-plane domain models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from ..contracts import JobRequirements, WorkerSpec
from ..execution import JobResult
from ..serving import ServingJobBinding, WorkerServingAdvertisement


def utc_now() -> datetime:
    return datetime.now(UTC)


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerState(StrEnum):
    ONLINE = "online"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class WorkerRegistration:
    spec: WorkerSpec
    max_concurrency: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    serving: WorkerServingAdvertisement | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.serving is not None:
            if not isinstance(self.serving, WorkerServingAdvertisement):
                raise TypeError("serving must be WorkerServingAdvertisement")
            advertised = frozenset(
                contract.capability for contract in self.serving.contracts
            )
            if not advertised.issubset(self.spec.capabilities):
                raise ValueError(
                    "serving contract capability must be advertised by WorkerSpec"
                )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class WorkerHeartbeat:
    state: WorkerState | None = None
    active_job_id: str | None = None
    lease_token: str | None = None
    runtime_instance_epoch: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.runtime_instance_epoch is not None:
            value = self.runtime_instance_epoch.strip()
            if not value:
                raise ValueError("runtime_instance_epoch must not be blank")
            object.__setattr__(self, "runtime_instance_epoch", value)
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    spec: WorkerSpec
    max_concurrency: int
    state: WorkerState
    active_jobs: int
    registered_at: datetime
    last_seen_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)
    serving: WorkerServingAdvertisement | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.active_jobs < 0:
            raise ValueError("active_jobs must not be negative")
        if self.serving is not None and not isinstance(
            self.serving, WorkerServingAdvertisement
        ):
            raise TypeError("serving must be WorkerServingAdvertisement")
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def worker_id(self) -> str:
        return self.spec.worker_id


@dataclass(frozen=True, slots=True)
class JobSubmission:
    capability: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    requirements: JobRequirements = field(default_factory=JobRequirements)
    priority: int = 0
    max_attempts: int = 3
    idempotency_key: str | None = None
    available_at: datetime | None = None
    deadline_at: datetime | None = None
    serving: ServingJobBinding | None = None

    def __post_init__(self) -> None:
        capability = self.capability.strip()
        if not capability:
            raise ValueError("capability must not be blank")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be blank")
        if self.available_at is not None and self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone-aware")
        if (
            self.available_at is not None
            and self.deadline_at is not None
            and self.available_at >= self.deadline_at
        ):
            raise ValueError("available_at must be earlier than deadline_at")
        if self.deadline_at is not None and self.serving is None:
            raise ValueError("deadline_at is only supported for serving jobs")
        if self.serving is not None:
            if not isinstance(self.serving, ServingJobBinding):
                raise TypeError("serving must be ServingJobBinding")
            if self.serving.capability != capability:
                raise ValueError("serving capability must match job capability")
            if self.deadline_at is None:
                raise ValueError("serving jobs require deadline_at")
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "payload", _mapping(self.payload))


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    capability: str
    payload: Mapping[str, Any]
    requirements: JobRequirements
    priority: int
    sequence: int
    status: JobStatus
    attempts: int
    max_attempts: int
    idempotency_key: str | None
    created_at: datetime
    available_at: datetime
    updated_at: datetime
    deadline_at: datetime | None = None
    serving: ServingJobBinding | None = None
    assigned_worker_id: str | None = None
    lease_token: str | None = None
    lease_expires_at: datetime | None = None
    claimed_deployment_revision: str | None = None
    claimed_serving_contract_revision: str | None = None
    claimed_runtime_instance_epoch: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: JobResult | None = None
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone-aware")
        if self.serving is not None and not isinstance(self.serving, ServingJobBinding):
            raise TypeError("serving must be ServingJobBinding")
        claimed = (
            self.claimed_deployment_revision,
            self.claimed_serving_contract_revision,
            self.claimed_runtime_instance_epoch,
        )
        if any(value is not None for value in claimed) and not all(
            value is not None for value in claimed
        ):
            raise ValueError("claimed serving identity fields must be all set or all unset")
        object.__setattr__(self, "payload", _mapping(self.payload))
        if self.error is not None:
            object.__setattr__(self, "error", _mapping(self.error))

    def with_updates(self, **changes: Any) -> "JobRecord":
        return replace(self, **changes)


__all__ = [
    "JobRecord",
    "JobStatus",
    "JobSubmission",
    "WorkerHeartbeat",
    "WorkerRecord",
    "WorkerRegistration",
    "WorkerState",
    "utc_now",
]
