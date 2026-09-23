"""Async Worker client for the AstrumWeaver v1 Control protocol."""

from __future__ import annotations

from typing import Any

import httpx

from ..control.models import WorkerHeartbeat, WorkerRegistration, WorkerState
from ..execution import JobResult
from ..transport import (
    ClaimedJobDTO,
    JobCompletionDTO,
    JobFailureDTO,
    JobResultDTO,
    JobStatusDTO,
    WorkerHeartbeatDTO,
    WorkerRecordDTO,
    WorkerRegistrationDTO,
    WorkerSpecDTO,
)


class ControlClientError(RuntimeError):
    """Base Worker-to-Control transport error."""


class ControlUnauthorized(ControlClientError):
    pass


class ControlNotFound(ControlClientError):
    pass


class ControlConflict(ControlClientError):
    pass


class ControlUnavailable(ControlClientError):
    pass


class ControlClient:
    def __init__(
        self,
        *,
        base_url: str,
        worker_token: str,
        timeout_seconds: float = 15.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not worker_token:
            raise ValueError("worker_token must not be empty")
        self._headers = {"authorization": f"Bearer {worker_token}"}
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.request(
                method,
                path,
                headers=self._headers,
                json=json,
            )
        except httpx.HTTPError as exc:
            raise ControlUnavailable("control request failed") from exc

        if response.status_code == 401:
            raise ControlUnauthorized("control rejected Worker credentials")
        if response.status_code == 404:
            raise ControlNotFound("control resource not found")
        if response.status_code == 409:
            raise ControlConflict("control rejected stale/conflicting state")
        if response.status_code >= 500:
            raise ControlUnavailable("control is unavailable")
        if response.status_code >= 400:
            raise ControlClientError(
                f"control request failed with status {response.status_code}"
            )
        return response

    async def register(self, registration: WorkerRegistration) -> WorkerRecordDTO:
        payload = WorkerRegistrationDTO(
            spec=WorkerSpecDTO.from_domain(registration.spec),
            max_concurrency=registration.max_concurrency,
            metadata=dict(registration.metadata),
        )
        response = await self._request(
            "POST",
            "/v1/workers/register",
            json=payload.model_dump(mode="json"),
        )
        return WorkerRecordDTO.model_validate(response.json())

    async def heartbeat(
        self,
        worker_id: str,
        heartbeat: WorkerHeartbeat | None = None,
    ) -> WorkerRecordDTO:
        value = heartbeat or WorkerHeartbeat()
        payload = WorkerHeartbeatDTO(
            state=value.state,
            active_job_id=value.active_job_id,
            lease_token=value.lease_token,
            metadata=dict(value.metadata),
        )
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/heartbeat",
            json=payload.model_dump(mode="json"),
        )
        return WorkerRecordDTO.model_validate(response.json())

    async def claim(self, worker_id: str) -> ClaimedJobDTO | None:
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/jobs/claim",
        )
        if response.status_code == 204:
            return None
        return ClaimedJobDTO.model_validate(response.json())

    async def job_status(self, worker_id: str, job_id: str) -> JobStatusDTO:
        response = await self._request(
            "GET",
            f"/v1/workers/{worker_id}/jobs/{job_id}",
        )
        return JobStatusDTO.model_validate(response.json())

    async def complete(
        self,
        *,
        worker_id: str,
        job_id: str,
        lease_token: str,
        result: JobResult,
    ) -> None:
        payload = JobCompletionDTO(
            worker_id=worker_id,
            lease_token=lease_token,
            result=JobResultDTO.from_domain(result),
        )
        await self._request(
            "POST",
            f"/v1/jobs/{job_id}/complete",
            json=payload.model_dump(mode="json"),
        )

    async def fail(
        self,
        *,
        worker_id: str,
        job_id: str,
        lease_token: str,
        error: str | dict[str, Any],
        retryable: bool,
    ) -> None:
        payload = JobFailureDTO(
            worker_id=worker_id,
            lease_token=lease_token,
            error=error,
            retryable=retryable,
        )
        await self._request(
            "POST",
            f"/v1/jobs/{job_id}/fail",
            json=payload.model_dump(mode="json"),
        )
