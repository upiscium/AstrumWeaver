"""AstrumWeaver domain contracts."""

from .contracts import JobRequirements, ResourceShape, WorkerSpec
from .execution import (
    ArtifactRef,
    JobExecutor,
    JobRequest,
    JobResult,
    ResidencyItem,
    ResidencyReport,
)
from .scheduling import MatchResult, match_worker, worker_matches

__all__ = [
    "ArtifactRef",
    "JobExecutor",
    "JobRequest",
    "JobRequirements",
    "JobResult",
    "MatchResult",
    "ResidencyItem",
    "ResidencyReport",
    "ResourceShape",
    "WorkerSpec",
    "match_worker",
    "worker_matches",
]
