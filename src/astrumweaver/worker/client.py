"""Async client for the AstrumWeaver v1 Control protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from ..control.models import WorkerState
from ..control.serde import job_result_to_dict, worker_spec_to_dict
from ..execution import JobRequest, JobResult
from ..transport import PROTOCOL_VERSION


class ControlTransportError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"control transport error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    request: JobRequest
    lease_token: str
    lease_expires_at: str | None
    attempts: int


class ControlClient:
    def __init__(
        self,
        base_url: str,
        worker_token: str,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not worker_token:
            raise ValueError("worker_token is required")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"authorization": f"Bearer {worker_token}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> "ControlClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ControlTransportError(0, "control unavailable") from exc
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("detail", "request failed"))
            except Exception:
                detail = "request failed"
            raise ControlTransportError(response.status_code, detail)
        return response

    async def register(
        self,
        *,
        spec,
        max_concurrency: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            "/v1/workers/register",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "spec": worker_spec_to_dict(spec),
                "max_concurrency": max_concurrency,
                "metadata": dict(metadata or {}),
            },
        )
        return response.json()

    async def heartbeat(
        self,
        worker_id: str,
        *,
        active_job_id: str | None = None,
        lease_token: str | None = None,
        state: WorkerState | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/heartbeat",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "active_job_id": active_job_id,
                "lease_token": lease_token,
                "state": None if state is None else state.value,
                "metadata": dict(metadata or {}),
            },
        )
        return response.json()

    async def set_state(self, worker_id: str, state: WorkerState) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/state",
            json={"protocol_version": PROTOCOL_VERSION, "state": state.value},
        )
        return response.json()

    async def claim(self, worker_id: str) -> ClaimedJob | None:
        response = await self._request("POST", f"/v1/workers/{worker_id}/jobs/claim")
        if response.status_code == 204:
            return None
        body = response.json()
        lease_token = body.get("lease_token")
        if not isinstance(lease_token, str) or not lease_token:
            raise ControlTransportError(502, "claim response lacks lease token")
        return ClaimedJob(
            request=JobRequest(
                job_id=str(body["job_id"]),
                capability=str(body["capability"]),
                payload=body.get("payload") or {},
                metadata={
                    "attempt": int(body.get("attempts", 0)),
                    "lease_expires_at": body.get("lease_expires_at"),
                },
            ),
            lease_token=lease_token,
            lease_expires_at=body.get("lease_expires_at"),
            attempts=int(body.get("attempts", 0)),
        )

    async def inspect_job(self, worker_id: str, job_id: str) -> dict[str, Any]:
        response = await self._request(
            "GET", f"/v1/workers/{worker_id}/jobs/{job_id}"
        )
        return response.json()

    async def complete(
        self,
        worker_id: str,
        job_id: str,
        lease_token: str,
        result: JobResult,
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/jobs/{job_id}/complete",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": lease_token,
                "result": job_result_to_dict(result),
            },
        )
        return response.json()

    async def fail(
        self,
        worker_id: str,
        job_id: str,
        lease_token: str,
        *,
        error: str | Mapping[str, Any],
        retryable: bool,
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/jobs/{job_id}/fail",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": lease_token,
                "error": dict(error) if isinstance(error, Mapping) else error,
                "retryable": retryable,
            },
        )
        return response.json()
