"""AstrumWeaver domain contracts."""

from .contracts import AcceleratorDevice, JobRequirements, ResourceShape, WorkerSpec
from .execution import (
    ArtifactRef,
    JobEvent,
    JobEventSink,
    JobExecutionError,
    JobExecutor,
    JobRequest,
    JobResult,
    ResidencyItem,
    ResidencyReport,
    StreamingJobExecutor,
)
from .scheduling import MatchResult, match_worker, worker_matches

__all__ = [
    "AcceleratorDevice",
    "ArtifactRef",
    "JobEvent",
    "JobEventSink",
    "JobExecutionError",
    "JobExecutor",
    "JobRequest",
    "JobRequirements",
    "JobResult",
    "MatchResult",
    "ResidencyItem",
    "ResidencyReport",
    "StreamingJobExecutor",
    "ResourceShape",
    "WorkerSpec",
    "match_worker",
    "worker_matches",
]
