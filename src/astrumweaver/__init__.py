"""AstrumWeaver domain contracts."""

from .contracts import JobRequirements, ResourceShape, WorkerSpec
from .scheduling import MatchResult, match_worker, worker_matches

__all__ = [
    "JobRequirements",
    "MatchResult",
    "ResourceShape",
    "WorkerSpec",
    "match_worker",
    "worker_matches",
]
