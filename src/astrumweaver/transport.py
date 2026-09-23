"""Versioned JSON transport helpers for AstrumWeaver v1."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .contracts import JobRequirements, ResourceShape, WorkerSpec
from .control.models import (
    JobRecord,
    JobStatus,
    JobSubmission,
    WorkerHeartbeat,
    WorkerRecord,
    WorkerRegistration,
    WorkerState,
)
from .control.serde import (
    job_result_from_dict,
    job_result_to_dict,
    requirements_from_dict,
    requirements_to_dict,
    worker_spec_from_dict,
    worker_spec_to_dict,
)

PROTOCOL_VERSION = "v1"


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def worker_record_to_dict(value: WorkerRecord) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "spec": worker_spec_to_dict(value.spec),
        "max_concurrency": value.max_concurrency,
        "state": value.state.value,
        "active_jobs": value.active_jobs,
        "registered_at": value.registered_at.isoformat(),
        "last_seen_at": value.last_seen_at.isoformat(),
        "metadata": dict(value.metadata),
    }


def job_record_to_dict(value: JobRecord, *, include_payload: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "job_id": value.job_id,
        "capability": value.capability,
        "requirements": requirements_to_dict(value.requirements),
        "priority": value.priority,
        "sequence": value.sequence,
        "status": value.status.value,
        "attempts": value.attempts,
        "max_attempts": value.max_attempts,
        "idempotency_key": value.idempotency_key,
        "assigned_worker_id": value.assigned_worker_id,
        "lease_token": value.lease_token,
        "created_at": value.created_at.isoformat(),
        "available_at": value.available_at.isoformat(),
        "started_at": _iso(value.started_at),
        "finished_at": _iso(value.finished_at),
        "lease_expires_at": _iso(value.lease_expires_at),
        "updated_at": value.updated_at.isoformat(),
        "result": None if value.result is None else job_result_to_dict(value.result),
        "error": None if value.error is None else dict(value.error),
    }
    if include_payload:
        result["payload"] = dict(value.payload)
    return result


def worker_registration_from_dict(value: Mapping[str, Any]) -> WorkerRegistration:
    data = dict(value)
    return WorkerRegistration(
        spec=worker_spec_from_dict(data["spec"]),
        max_concurrency=int(data.get("max_concurrency", 1)),
        metadata=data.get("metadata") or {},
    )


def worker_heartbeat_from_dict(value: Mapping[str, Any]) -> WorkerHeartbeat:
    data = dict(value)
    state = data.get("state")
    return WorkerHeartbeat(
        state=None if state is None else WorkerState(str(state)),
        active_job_id=data.get("active_job_id"),
        lease_token=data.get("lease_token"),
        metadata=data.get("metadata") or {},
    )


def job_submission_from_dict(value: Mapping[str, Any]) -> JobSubmission:
    data = dict(value)
    available_at = data.get("available_at")
    return JobSubmission(
        capability=str(data["capability"]),
        payload=data.get("payload") or {},
        requirements=requirements_from_dict(data.get("requirements")),
        priority=int(data.get("priority", 0)),
        max_attempts=int(data.get("max_attempts", 3)),
        idempotency_key=data.get("idempotency_key"),
        available_at=None if available_at is None else datetime.fromisoformat(str(available_at)),
    )


def job_request_from_record(value: JobRecord):
    from .execution import JobRequest

    return JobRequest(
        job_id=value.job_id,
        capability=value.capability,
        payload=value.payload,
        metadata={
            "attempt": value.attempts,
            "lease_expires_at": _iso(value.lease_expires_at),
        },
    )


__all__ = [
    "PROTOCOL_VERSION",
    "job_record_to_dict",
    "job_request_from_record",
    "job_submission_from_dict",
    "worker_heartbeat_from_dict",
    "worker_record_to_dict",
    "worker_registration_from_dict",
]
