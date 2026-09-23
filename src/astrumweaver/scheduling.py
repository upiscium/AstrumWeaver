"""Pure worker eligibility matching.

The matcher is deliberately unaware of LLMs, TTS, image generation, or any
other application family. It only evaluates declared capabilities, labels,
identity constraints, and resource shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import JobRequirements, WorkerSpec


@dataclass(frozen=True, slots=True)
class MatchResult:
    matched: bool
    reasons: tuple[str, ...] = ()


def match_worker(worker: WorkerSpec, requirements: JobRequirements) -> MatchResult:
    """Evaluate whether worker satisfies every job requirement."""

    reasons: list[str] = []

    if requirements.worker_class and worker.worker_class != requirements.worker_class:
        reasons.append("worker_class")

    if not requirements.required_capabilities.issubset(worker.capabilities):
        reasons.append("capabilities")

    for key, value in requirements.required_labels.items():
        if worker.labels.get(key) != value:
            reasons.append(f"label:{key}")

    if not requirements.required_gpu_uuids.issubset(set(worker.gpu_uuids)):
        reasons.append("gpu_uuids")

    resources = worker.resources
    if resources.gpu_count < requirements.min_gpu_count:
        reasons.append("gpu_count")
    if resources.total_vram_mb < requirements.min_total_vram_mb:
        reasons.append("total_vram_mb")
    if resources.max_single_gpu_vram_mb < requirements.min_single_gpu_vram_mb:
        reasons.append("max_single_gpu_vram_mb")

    return MatchResult(matched=not reasons, reasons=tuple(reasons))


def worker_matches(worker: WorkerSpec, requirements: JobRequirements) -> bool:
    """Boolean convenience wrapper around match_worker."""

    return match_worker(worker, requirements).matched
