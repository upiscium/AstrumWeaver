"""AstrumWeaver v1 transport contracts."""

from .auth import AuthConfig
from .models import (
    ClaimedJobDTO,
    HealthDTO,
    JobCompletionDTO,
    JobFailureDTO,
    JobRequirementsDTO,
    JobResultDTO,
    JobStatusDTO,
    JobSubmissionDTO,
    JobViewDTO,
    ResourceShapeDTO,
    WorkerHeartbeatDTO,
    WorkerRecordDTO,
    WorkerRegistrationDTO,
    WorkerSpecDTO,
    WorkerStateUpdateDTO,
)

PROTOCOL_VERSION = "v1"

__all__ = [
    "AuthConfig",
    "ClaimedJobDTO",
    "HealthDTO",
    "JobCompletionDTO",
    "JobFailureDTO",
    "JobRequirementsDTO",
    "JobResultDTO",
    "JobStatusDTO",
    "JobSubmissionDTO",
    "JobViewDTO",
    "PROTOCOL_VERSION",
    "ResourceShapeDTO",
    "WorkerHeartbeatDTO",
    "WorkerRecordDTO",
    "WorkerRegistrationDTO",
    "WorkerSpecDTO",
    "WorkerStateUpdateDTO",
]
