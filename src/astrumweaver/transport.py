"""Versioned JSON transport helpers for AstrumWeaver v1."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .contracts import JobRequirements, ResourceShape, WorkerSpec
from .control.models import (
    JobEventRecord,
    JobRecord,
    JobStatus,
    JobSubmission,
    WorkerHeartbeat,
    WorkerRecord,
    WorkerRegistration,
    WorkerState,
)
from .serving import ServingJobBinding, WorkerServingAdvertisement
from .control.serde import (
    job_result_from_dict,
    job_result_to_dict,
    requirements_from_dict,
    requirements_to_dict,
    worker_spec_from_dict,
    worker_spec_to_dict,
)

PROTOCOL_VERSION = "v1"
SERVING_EXTENSION = "serving-bindings-v1"
JOB_EVENTS_EXTENSION = "job-events-v1"
_SUPPORTED_EXTENSIONS = frozenset({SERVING_EXTENSION, JOB_EVENTS_EXTENSION})


def _extensions(value: Mapping[str, Any]) -> frozenset[str]:
    raw = value.get("extensions", ())
    if not isinstance(raw, (list, tuple)):
        raise TypeError("extensions must be an array")
    items = tuple(str(item) for item in raw)
    if len(set(items)) != len(items):
        raise ValueError("extensions must not contain duplicates")
    unknown = set(items) - _SUPPORTED_EXTENSIONS
    if unknown:
        raise ValueError("unsupported protocol extension")
    return frozenset(items)


def _require_serving_extension(value: Mapping[str, Any]) -> None:
    if SERVING_EXTENSION not in _extensions(value):
        raise ValueError("serving-bindings-v1 extension is required")


def _require_job_events_extension(value: Mapping[str, Any]) -> None:
    if JOB_EVENTS_EXTENSION not in _extensions(value):
        raise ValueError("job-events-v1 extension is required")


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def job_event_record_to_dict(value: JobEventRecord) -> dict[str, Any]:
    extensions = [JOB_EVENTS_EXTENSION]
    if value.runtime_instance_epoch is not None:
        extensions.append(SERVING_EXTENSION)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "extensions": extensions,
        "job_id": value.job_id,
        "attempt": value.attempt,
        "sequence": value.sequence,
        "worker_id": value.worker_id,
        "runtime_instance_epoch": value.runtime_instance_epoch,
        "kind": value.kind,
        "payload": dict(value.payload),
        "created_at": value.created_at.isoformat(),
    }


def worker_record_to_dict(value: WorkerRecord) -> dict[str, Any]:
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "spec": worker_spec_to_dict(value.spec),
        "max_concurrency": value.max_concurrency,
        "state": value.state.value,
        "active_jobs": value.active_jobs,
        "registered_at": value.registered_at.isoformat(),
        "last_seen_at": value.last_seen_at.isoformat(),
        "metadata": dict(value.metadata),
    }
    if value.serving is not None:
        result["serving"] = value.serving.to_dict()
        result["extensions"] = [SERVING_EXTENSION]
    return result


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
    if value.serving is not None or value.claimed_runtime_instance_epoch is not None:
        result.update(
            {
                "deadline_at": _iso(value.deadline_at),
                "serving": None if value.serving is None else value.serving.to_dict(),
                "claimed_deployment_revision": value.claimed_deployment_revision,
                "claimed_serving_contract_revision": value.claimed_serving_contract_revision,
                "claimed_runtime_instance_epoch": value.claimed_runtime_instance_epoch,
                "extensions": [SERVING_EXTENSION],
            }
        )
    return result


def worker_registration_from_dict(value: Mapping[str, Any]) -> WorkerRegistration:
    data = dict(value)
    raw_serving = data.get("serving")
    _extensions(data)
    if raw_serving is not None:
        _require_serving_extension(data)
    return WorkerRegistration(
        spec=worker_spec_from_dict(data["spec"]),
        max_concurrency=int(data.get("max_concurrency", 1)),
        metadata=data.get("metadata") or {},
        serving=(
            None
            if raw_serving is None
            else WorkerServingAdvertisement.from_dict(raw_serving)
        ),
    )


def worker_heartbeat_from_dict(value: Mapping[str, Any]) -> WorkerHeartbeat:
    data = dict(value)
    state = data.get("state")
    _extensions(data)
    if data.get("runtime_instance_epoch") is not None:
        _require_serving_extension(data)
    return WorkerHeartbeat(
        state=None if state is None else WorkerState(str(state)),
        active_job_id=data.get("active_job_id"),
        lease_token=data.get("lease_token"),
        runtime_instance_epoch=data.get("runtime_instance_epoch"),
        metadata=data.get("metadata") or {},
    )


def job_submission_from_dict(value: Mapping[str, Any]) -> JobSubmission:
    data = dict(value)
    available_at = data.get("available_at")
    deadline_at = data.get("deadline_at")
    raw_serving = data.get("serving")
    _extensions(data)
    if raw_serving is not None or deadline_at is not None:
        _require_serving_extension(data)
    return JobSubmission(
        capability=str(data["capability"]),
        payload=data.get("payload") or {},
        requirements=requirements_from_dict(data.get("requirements")),
        priority=int(data.get("priority", 0)),
        max_attempts=int(data.get("max_attempts", 3)),
        idempotency_key=data.get("idempotency_key"),
        available_at=None if available_at is None else datetime.fromisoformat(str(available_at)),
        deadline_at=None if deadline_at is None else datetime.fromisoformat(str(deadline_at)),
        serving=None if raw_serving is None else ServingJobBinding.from_dict(raw_serving),
    )


def job_request_from_record(value: JobRecord):
    from .execution import JobRequest

    metadata: dict[str, Any] = {
        "attempt": value.attempts,
        "lease_expires_at": _iso(value.lease_expires_at),
    }
    if value.serving is not None:
        metadata.update(
            {
                "deadline_at": _iso(value.deadline_at),
                "serving": value.serving.to_dict(),
                "claimed_deployment_revision": value.claimed_deployment_revision,
                "claimed_serving_contract_revision": value.claimed_serving_contract_revision,
                "claimed_runtime_instance_epoch": value.claimed_runtime_instance_epoch,
            }
        )
    return JobRequest(
        job_id=value.job_id,
        capability=value.capability,
        payload=value.payload,
        metadata=metadata,
    )


__all__ = [
    "JOB_EVENTS_EXTENSION",
    "PROTOCOL_VERSION",
    "SERVING_EXTENSION",
    "_require_job_events_extension",
    "_require_serving_extension",
    "job_event_record_to_dict",
    "job_record_to_dict",
    "job_request_from_record",
    "job_submission_from_dict",
    "worker_heartbeat_from_dict",
    "worker_record_to_dict",
    "worker_registration_from_dict",
]
