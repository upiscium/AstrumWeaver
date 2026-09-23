"""JSON-compatible serialization helpers for durable control state."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import JobRequirements, ResourceShape, WorkerSpec
from ..execution import ArtifactRef, JobResult


def requirements_to_dict(value: JobRequirements) -> dict[str, Any]:
    return {
        "worker_class": value.worker_class,
        "required_capabilities": sorted(value.required_capabilities),
        "required_labels": dict(value.required_labels),
        "required_gpu_uuids": sorted(value.required_gpu_uuids),
        "min_gpu_count": value.min_gpu_count,
        "min_total_vram_mb": value.min_total_vram_mb,
        "min_single_gpu_vram_mb": value.min_single_gpu_vram_mb,
    }


def requirements_from_dict(value: Mapping[str, Any] | None) -> JobRequirements:
    data = dict(value or {})
    return JobRequirements(
        worker_class=data.get("worker_class"),
        required_capabilities=frozenset(data.get("required_capabilities") or ()),
        required_labels=data.get("required_labels") or {},
        required_gpu_uuids=frozenset(data.get("required_gpu_uuids") or ()),
        min_gpu_count=int(data.get("min_gpu_count") or 0),
        min_total_vram_mb=int(data.get("min_total_vram_mb") or 0),
        min_single_gpu_vram_mb=int(data.get("min_single_gpu_vram_mb") or 0),
    )


def worker_spec_to_dict(value: WorkerSpec) -> dict[str, Any]:
    return {
        "worker_id": value.worker_id,
        "worker_class": value.worker_class,
        "gpu_uuids": list(value.gpu_uuids),
        "capabilities": sorted(value.capabilities),
        "labels": dict(value.labels),
        "resources": {
            "gpu_count": value.resources.gpu_count,
            "total_vram_mb": value.resources.total_vram_mb,
            "max_single_gpu_vram_mb": value.resources.max_single_gpu_vram_mb,
        },
    }


def worker_spec_from_dict(value: Mapping[str, Any]) -> WorkerSpec:
    data = dict(value)
    resource = dict(data.get("resources") or {})
    return WorkerSpec(
        worker_id=str(data["worker_id"]),
        worker_class=str(data["worker_class"]),
        gpu_uuids=tuple(data.get("gpu_uuids") or ()),
        capabilities=frozenset(data.get("capabilities") or ()),
        labels=data.get("labels") or {},
        resources=ResourceShape(
            gpu_count=int(resource.get("gpu_count") or 0),
            total_vram_mb=int(resource.get("total_vram_mb") or 0),
            max_single_gpu_vram_mb=int(resource.get("max_single_gpu_vram_mb") or 0),
        ),
    )


def job_result_to_dict(value: JobResult) -> dict[str, Any]:
    return {
        "outputs": dict(value.outputs),
        "artifacts": [
            {
                "uri": artifact.uri,
                "media_type": artifact.media_type,
                "digest": artifact.digest,
                "size_bytes": artifact.size_bytes,
                "name": artifact.name,
                "metadata": dict(artifact.metadata),
            }
            for artifact in value.artifacts
        ],
        "metrics": dict(value.metrics),
        "text": value.text,
        "metadata": dict(value.metadata),
    }


def job_result_from_dict(value: Mapping[str, Any] | None) -> JobResult | None:
    if value is None:
        return None
    data = dict(value)
    artifacts = tuple(
        ArtifactRef(
            uri=str(item["uri"]),
            media_type=item.get("media_type"),
            digest=item.get("digest"),
            size_bytes=item.get("size_bytes"),
            name=item.get("name"),
            metadata=item.get("metadata") or {},
        )
        for item in data.get("artifacts") or ()
    )
    return JobResult(
        outputs=data.get("outputs") or {},
        artifacts=artifacts,
        metrics=data.get("metrics") or {},
        text=data.get("text"),
        metadata=data.get("metadata") or {},
    )
