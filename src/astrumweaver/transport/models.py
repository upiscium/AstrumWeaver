"""Versioned HTTP transport models.

These DTOs are intentionally workload-neutral.  They adapt JSON payloads to
AstrumWeaver domain contracts without teaching the transport about LLMs, TTS,
images, or other executor-specific semantics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import JobRequirements, ResourceShape, WorkerSpec
from ..control.models import (
    JobRecord,
    JobStatus,
    JobSubmission,
    WorkerHeartbeat,
    WorkerRecord,
    WorkerRegistration,
    WorkerState,
)
from ..control.serde import job_result_from_dict, job_result_to_dict
from ..execution import JobRequest, JobResult


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResourceShapeDTO(StrictModel):
    gpu_count: int = Field(default=0, ge=0)
    total_vram_mb: int = Field(default=0, ge=0)
    max_single_gpu_vram_mb: int = Field(default=0, ge=0)

    def to_domain(self) -> ResourceShape:
        return ResourceShape(**self.model_dump())

    @classmethod
    def from_domain(cls, value: ResourceShape) -> "ResourceShapeDTO":
        return cls(
            gpu_count=value.gpu_count,
            total_vram_mb=value.total_vram_mb,
            max_single_gpu_vram_mb=value.max_single_gpu_vram_mb,
        )


class WorkerSpecDTO(StrictModel):
    worker_id: str
    worker_class: str
    resources: ResourceShapeDTO = Field(default_factory=ResourceShapeDTO)
    gpu_uuids: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)

    def to_domain(self) -> WorkerSpec:
        return WorkerSpec(
            worker_id=self.worker_id,
            worker_class=self.worker_class,
            resources=self.resources.to_domain(),
            gpu_uuids=tuple(self.gpu_uuids),
            capabilities=frozenset(self.capabilities),
            labels=self.labels,
        )

    @classmethod
    def from_domain(cls, value: WorkerSpec) -> "WorkerSpecDTO":
        return cls(
            worker_id=value.worker_id,
            worker_class=value.worker_class,
            resources=ResourceShapeDTO.from_domain(value.resources),
            gpu_uuids=list(value.gpu_uuids),
            capabilities=sorted(value.capabilities),
            labels=dict(value.labels),
        )


class WorkerRegistrationDTO(StrictModel):
    spec: WorkerSpecDTO
    max_concurrency: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> WorkerRegistration:
        return WorkerRegistration(
            spec=self.spec.to_domain(),
            max_concurrency=self.max_concurrency,
            metadata=self.metadata,
        )


class WorkerHeartbeatDTO(StrictModel):
    state: WorkerState | None = None
    active_job_id: str | None = None
    lease_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> WorkerHeartbeat:
        return WorkerHeartbeat(
            state=self.state,
            active_job_id=self.active_job_id,
            lease_token=self.lease_token,
            metadata=self.metadata,
        )


class WorkerRecordDTO(StrictModel):
    spec: WorkerSpecDTO
    max_concurrency: int
    state: WorkerState
    active_jobs: int
    registered_at: datetime
    last_seen_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, value: WorkerRecord) -> "WorkerRecordDTO":
        return cls(
            spec=WorkerSpecDTO.from_domain(value.spec),
            max_concurrency=value.max_concurrency,
            state=value.state,
            active_jobs=value.active_jobs,
            registered_at=value.registered_at,
            last_seen_at=value.last_seen_at,
            metadata=dict(value.metadata),
        )


class WorkerStateUpdateDTO(StrictModel):
    state: WorkerState


class JobRequirementsDTO(StrictModel):
    worker_class: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    required_labels: dict[str, str] = Field(default_factory=dict)
    required_gpu_uuids: list[str] = Field(default_factory=list)
    min_gpu_count: int = Field(default=0, ge=0)
    min_total_vram_mb: int = Field(default=0, ge=0)
    min_single_gpu_vram_mb: int = Field(default=0, ge=0)

    def to_domain(self) -> JobRequirements:
        return JobRequirements(
            worker_class=self.worker_class,
            required_capabilities=frozenset(self.required_capabilities),
            required_labels=self.required_labels,
            required_gpu_uuids=frozenset(self.required_gpu_uuids),
            min_gpu_count=self.min_gpu_count,
            min_total_vram_mb=self.min_total_vram_mb,
            min_single_gpu_vram_mb=self.min_single_gpu_vram_mb,
        )

    @classmethod
    def from_domain(cls, value: JobRequirements) -> "JobRequirementsDTO":
        return cls(
            worker_class=value.worker_class,
            required_capabilities=sorted(value.required_capabilities),
            required_labels=dict(value.required_labels),
            required_gpu_uuids=sorted(value.required_gpu_uuids),
            min_gpu_count=value.min_gpu_count,
            min_total_vram_mb=value.min_total_vram_mb,
            min_single_gpu_vram_mb=value.min_single_gpu_vram_mb,
        )


class JobSubmissionDTO(StrictModel):
    capability: str
    payload: dict[str, Any] = Field(default_factory=dict)
    requirements: JobRequirementsDTO = Field(default_factory=JobRequirementsDTO)
    priority: int = 0
    max_attempts: int = Field(default=3, ge=1)
    idempotency_key: str | None = None
    available_at: datetime | None = None

    def to_domain(self) -> JobSubmission:
        return JobSubmission(
            capability=self.capability,
            payload=self.payload,
            requirements=self.requirements.to_domain(),
            priority=self.priority,
            max_attempts=self.max_attempts,
            idempotency_key=self.idempotency_key,
            available_at=self.available_at,
        )


class JobResultDTO(StrictModel):
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, int | float] = Field(default_factory=dict)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> JobResult:
        result = job_result_from_dict(self.model_dump())
        if result is None:  # pragma: no cover - defensive
            raise ValueError("job result cannot be null")
        return result

    @classmethod
    def from_domain(cls, value: JobResult) -> "JobResultDTO":
        return cls.model_validate(job_result_to_dict(value))


class JobViewDTO(StrictModel):
    job_id: str
    capability: str
    payload: dict[str, Any]
    requirements: JobRequirementsDTO
    priority: int
    sequence: int
    status: JobStatus
    attempts: int
    max_attempts: int
    idempotency_key: str | None
    assigned_worker_id: str | None
    result: JobResultDTO | None
    error: dict[str, Any] | None
    created_at: datetime
    available_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime

    @classmethod
    def from_domain(cls, value: JobRecord) -> "JobViewDTO":
        return cls(
            job_id=value.job_id,
            capability=value.capability,
            payload=dict(value.payload),
            requirements=JobRequirementsDTO.from_domain(value.requirements),
            priority=value.priority,
            sequence=value.sequence,
            status=value.status,
            attempts=value.attempts,
            max_attempts=value.max_attempts,
            idempotency_key=value.idempotency_key,
            assigned_worker_id=value.assigned_worker_id,
            result=JobResultDTO.from_domain(value.result) if value.result else None,
            error=dict(value.error) if value.error is not None else None,
            created_at=value.created_at,
            available_at=value.available_at,
            started_at=value.started_at,
            finished_at=value.finished_at,
            updated_at=value.updated_at,
        )


class ClaimedJobDTO(StrictModel):
    job_id: str
    capability: str
    payload: dict[str, Any]
    lease_token: str
    lease_expires_at: datetime
    attempts: int

    @classmethod
    def from_domain(cls, value: JobRecord) -> "ClaimedJobDTO":
        if value.lease_token is None or value.lease_expires_at is None:
            raise ValueError("claimed job is missing lease state")
        return cls(
            job_id=value.job_id,
            capability=value.capability,
            payload=dict(value.payload),
            lease_token=value.lease_token,
            lease_expires_at=value.lease_expires_at,
            attempts=value.attempts,
        )

    def to_job_request(self) -> JobRequest:
        return JobRequest(
            job_id=self.job_id,
            capability=self.capability,
            payload=self.payload,
        )


class JobStatusDTO(StrictModel):
    job_id: str
    status: JobStatus
    updated_at: datetime

    @classmethod
    def from_domain(cls, value: JobRecord) -> "JobStatusDTO":
        return cls(job_id=value.job_id, status=value.status, updated_at=value.updated_at)


class JobCompletionDTO(StrictModel):
    worker_id: str
    lease_token: str
    result: JobResultDTO


class JobFailureDTO(StrictModel):
    worker_id: str
    lease_token: str
    error: str | dict[str, Any]
    retryable: bool = True


class HealthDTO(StrictModel):
    status: str
    protocol_version: str = "v1"
