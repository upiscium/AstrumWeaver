"""Async client for the AstrumWeaver v1 Control protocol."""

from __future__ import annotations

import asyncio
import math

from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from ..control.models import WorkerState
from ..control.serde import job_result_to_dict, worker_spec_to_dict
from ..execution import JobRequest, JobResult
from ..serving import WorkerServingAdvertisement
from ..transport import JOB_EVENTS_EXTENSION, PROTOCOL_VERSION, SERVING_EXTENSION


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
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Control request timeout must be finite and positive")
        self.timeout_seconds = timeout_seconds
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
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._client.request(method, path, **kwargs)
        except (httpx.HTTPError, TimeoutError) as exc:
            raise ControlTransportError(0, "control unavailable") from exc
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("detail", "request failed"))
            except Exception:
                detail = "request failed"
            raise ControlTransportError(response.status_code, detail)
        return response

    @staticmethod
    def _object(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise ControlTransportError(502, "Control returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise ControlTransportError(502, "Control response must be an object")
        raw_extensions = body.get("extensions") or ()
        if (
            isinstance(raw_extensions, str)
            or not isinstance(raw_extensions, (list, tuple, set, frozenset))
        ):
            raise ControlTransportError(502, "Control response extensions are invalid")
        extensions = frozenset(str(item) for item in raw_extensions)
        if extensions - {SERVING_EXTENSION, JOB_EVENTS_EXTENSION}:
            raise ControlTransportError(
                502, "Control response uses an unsupported extension"
            )
        serving_fields = (
            "serving",
            "claimed_deployment_revision",
            "claimed_serving_contract_revision",
            "claimed_runtime_instance_epoch",
            "runtime_instance_epoch",
        )
        if (
            any(body.get(field) is not None for field in serving_fields)
            and SERVING_EXTENSION not in extensions
        ):
            raise ControlTransportError(
                502, "Control response omitted the serving extension"
            )
        return body

    async def require_extension(self, extension: str) -> None:
        response = await self._request("GET", "/v1/ready")
        body = self._object(response)
        if body.get("protocol_version") != PROTOCOL_VERSION:
            raise ControlTransportError(502, "Control protocol version is invalid")
        extensions = frozenset(str(item) for item in body.get("extensions") or ())
        if extension not in extensions:
            raise ControlTransportError(
                409, f"Control does not support required extension: {extension}"
            )

    async def register(
        self,
        *,
        spec,
        max_concurrency: int,
        metadata: Mapping[str, Any] | None = None,
        serving: WorkerServingAdvertisement | None = None,
    ) -> dict[str, Any]:
        if serving is not None:
            await self.require_extension(SERVING_EXTENSION)
        response = await self._request(
            "POST",
            "/v1/workers/register",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "spec": worker_spec_to_dict(spec),
                "max_concurrency": max_concurrency,
                "metadata": dict(metadata or {}),
                "serving": None if serving is None else serving.to_dict(),
                "extensions": [] if serving is None else [SERVING_EXTENSION],
            },
        )
        return self._object(response)

    async def heartbeat(
        self,
        worker_id: str,
        *,
        active_job_id: str | None = None,
        lease_token: str | None = None,
        state: WorkerState | None = None,
        runtime_instance_epoch: str | None = None,
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
                "runtime_instance_epoch": runtime_instance_epoch,
                "extensions": (
                    [] if runtime_instance_epoch is None else [SERVING_EXTENSION]
                ),
                "metadata": dict(metadata or {}),
            },
        )
        return self._object(response)

    async def set_state(
        self,
        worker_id: str,
        state: WorkerState,
        *,
        runtime_instance_epoch: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "state": state.value,
        }
        if runtime_instance_epoch is not None:
            body["runtime_instance_epoch"] = runtime_instance_epoch
            body["extensions"] = [SERVING_EXTENSION]
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/state",
            json=body,
        )
        return self._object(response)

    async def claim(
        self,
        worker_id: str,
        *,
        runtime_instance_epoch: str | None = None,
    ) -> ClaimedJob | None:
        kwargs: dict[str, Any] = {}
        if runtime_instance_epoch is not None:
            kwargs["json"] = {
                "protocol_version": PROTOCOL_VERSION,
                "runtime_instance_epoch": runtime_instance_epoch,
                "extensions": [SERVING_EXTENSION],
            }
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/jobs/claim",
            **kwargs,
        )
        if response.status_code == 204:
            return None
        body = self._object(response)
        lease_token = body.get("lease_token")
        if not isinstance(lease_token, str) or not lease_token:
            raise ControlTransportError(502, "claim response lacks lease token")
        metadata: dict[str, Any] = {
            "attempt": int(body.get("attempts", 0)),
            "lease_expires_at": body.get("lease_expires_at"),
        }
        if body.get("serving") is not None:
            if SERVING_EXTENSION not in set(body.get("extensions") or ()):
                raise ControlTransportError(
                    502, "claim response lacks serving extension marker"
                )
            metadata.update(
                {
                    "deadline_at": body.get("deadline_at"),
                    "serving": body.get("serving"),
                    "claimed_deployment_revision": body.get("claimed_deployment_revision"),
                    "claimed_serving_contract_revision": body.get("claimed_serving_contract_revision"),
                    "claimed_runtime_instance_epoch": body.get("claimed_runtime_instance_epoch"),
                }
            )
        return ClaimedJob(
            request=JobRequest(
                job_id=str(body["job_id"]),
                capability=str(body["capability"]),
                payload=body.get("payload") or {},
                metadata=metadata,
            ),
            lease_token=lease_token,
            lease_expires_at=body.get("lease_expires_at"),
            attempts=int(body.get("attempts", 0)),
        )

    async def publish_event(
        self,
        worker_id: str,
        job_id: str,
        lease_token: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
        runtime_instance_epoch: str | None = None,
    ) -> dict[str, Any]:
        extensions = [JOB_EVENTS_EXTENSION]
        if runtime_instance_epoch is not None:
            extensions.append(SERVING_EXTENSION)
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/jobs/{job_id}/events",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "extensions": extensions,
                "lease_token": lease_token,
                "runtime_instance_epoch": runtime_instance_epoch,
                "kind": kind,
                "payload": dict(payload),
            },
        )
        return self._object(response)

    async def inspect_job(self, worker_id: str, job_id: str) -> dict[str, Any]:
        response = await self._request(
            "GET", f"/v1/workers/{worker_id}/jobs/{job_id}"
        )
        return self._object(response)

    async def complete(
        self,
        worker_id: str,
        job_id: str,
        lease_token: str,
        result: JobResult,
        *,
        runtime_instance_epoch: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/jobs/{job_id}/complete",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": lease_token,
                "runtime_instance_epoch": runtime_instance_epoch,
                "extensions": (
                    [] if runtime_instance_epoch is None else [SERVING_EXTENSION]
                ),
                "result": job_result_to_dict(result),
            },
        )
        return self._object(response)

    async def fail(
        self,
        worker_id: str,
        job_id: str,
        lease_token: str,
        *,
        error: str | Mapping[str, Any],
        retryable: bool,
        runtime_instance_epoch: str | None = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"/v1/workers/{worker_id}/jobs/{job_id}/fail",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": lease_token,
                "runtime_instance_epoch": runtime_instance_epoch,
                "extensions": (
                    [] if runtime_instance_epoch is None else [SERVING_EXTENSION]
                ),
                "error": dict(error) if isinstance(error, Mapping) else error,
                "retryable": retryable,
            },
        )
        return self._object(response)
