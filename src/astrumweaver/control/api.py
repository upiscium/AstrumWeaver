"""FastAPI transport for the AstrumWeaver Control Plane."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from ..control.models import WorkerState
from ..control.repository import (
    ConflictError,
    ControlRepository,
    NotFoundError,
    RepositoryError,
    StorageUnavailable,
)
from ..control.serde import job_result_from_dict
from ..transport import (
    PROTOCOL_VERSION,
    job_record_to_dict,
    job_submission_from_dict,
    worker_heartbeat_from_dict,
    worker_record_to_dict,
    worker_registration_from_dict,
)


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _require_token(request: Request, expected: str, authority: str) -> None:
    supplied = _bearer_token(request)
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{authority} authorization required",
        )


async def _json_v1(request: Request) -> dict[str, Any]:
    body = await _json_v1(request)
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="JSON object is required")
    if body.get("protocol_version") != PROTOCOL_VERSION:
        raise HTTPException(
            status_code=409,
            detail=f"unsupported protocol version; expected {PROTOCOL_VERSION}",
        )
    return body


def create_app(
    repository: ControlRepository,
    *,
    client_token: str,
    worker_token: str,
    maintenance_interval_seconds: float = 5.0,
) -> FastAPI:
    if not client_token or not worker_token:
        raise ValueError("client_token and worker_token are required")
    if secrets.compare_digest(client_token, worker_token):
        raise ValueError("client_token and worker_token must be distinct")
    if maintenance_interval_seconds <= 0:
        raise ValueError("maintenance_interval_seconds must be positive")

    async def maintenance_loop() -> None:
        while True:
            await asyncio.sleep(maintenance_interval_seconds)
            try:
                await asyncio.to_thread(repository.expire_stale_workers)
                await asyncio.to_thread(repository.recover_expired_jobs)
            except Exception:
                # Repository/API calls still expose current storage failures.
                # Maintenance must not terminate the server process.
                continue

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(maintenance_loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="AstrumWeaver Control API",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )

    def require_client(request: Request) -> None:
        _require_token(request, client_token, "client")

    def require_worker(request: Request) -> None:
        _require_token(request, worker_token, "worker")

    def require_either(request: Request) -> None:
        supplied = _bearer_token(request)
        if supplied and (
            secrets.compare_digest(supplied, client_token)
            or secrets.compare_digest(supplied, worker_token)
        ):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authorization required",
        )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_handler(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(StorageUnavailable)
    async def storage_handler(_: Request, __: StorageUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "storage unavailable"})

    @app.exception_handler(RepositoryError)
    async def repository_handler(_: Request, __: RepositoryError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "repository error"})

    @app.exception_handler(Exception)
    async def unexpected_handler(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "protocol_version": PROTOCOL_VERSION}

    @app.get("/v1/ready")
    async def ready() -> dict[str, Any]:
        await asyncio.to_thread(repository.check_storage)
        return {"ready": True, "protocol_version": PROTOCOL_VERSION}

    @app.post("/v1/jobs", status_code=201, dependencies=[Depends(require_client)])
    async def submit_job(request: Request) -> dict[str, Any]:
        try:
            submission = job_submission_from_dict(await _json_v1(request))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = await asyncio.to_thread(repository.submit_job, submission)
        return job_record_to_dict(record)

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_either)])
    async def get_job(job_id: str) -> dict[str, Any]:
        record = await asyncio.to_thread(repository.get_job, job_id)
        return job_record_to_dict(record)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(require_client)])
    async def cancel_job(job_id: str) -> dict[str, Any]:
        record = await asyncio.to_thread(repository.cancel_job, job_id)
        return job_record_to_dict(record)

    @app.post("/v1/workers/register", status_code=201, dependencies=[Depends(require_worker)])
    async def register_worker(request: Request) -> dict[str, Any]:
        try:
            registration = worker_registration_from_dict(await _json_v1(request))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = await asyncio.to_thread(repository.register_worker, registration)
        return worker_record_to_dict(record)

    @app.post("/v1/workers/{worker_id}/heartbeat", dependencies=[Depends(require_worker)])
    async def heartbeat_worker(worker_id: str, request: Request) -> dict[str, Any]:
        try:
            heartbeat = worker_heartbeat_from_dict(await _json_v1(request))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = await asyncio.to_thread(
            repository.heartbeat_worker, worker_id, heartbeat
        )
        return worker_record_to_dict(record)

    @app.post("/v1/workers/{worker_id}/state", dependencies=[Depends(require_worker)])
    async def set_worker_state(worker_id: str, request: Request) -> dict[str, Any]:
        try:
            body = await _json_v1(request)
            state_value = body["state"]
            state = WorkerState(str(state_value))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = await asyncio.to_thread(repository.set_worker_state, worker_id, state)
        return worker_record_to_dict(record)

    @app.post("/v1/workers/{worker_id}/jobs/claim", dependencies=[Depends(require_worker)])
    async def claim_job(worker_id: str) -> Response:
        record = await asyncio.to_thread(repository.claim_next_job, worker_id)
        if record is None:
            return Response(status_code=204)
        return JSONResponse(status_code=200, content=job_record_to_dict(record))

    @app.get(
        "/v1/workers/{worker_id}/jobs/{job_id}",
        dependencies=[Depends(require_worker)],
    )
    async def inspect_worker_job(worker_id: str, job_id: str) -> dict[str, Any]:
        record = await asyncio.to_thread(repository.get_job, job_id)
        visible = {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": record.job_id,
            "status": record.status.value,
            "assigned_worker_id": record.assigned_worker_id,
            "lease_expires_at": (
                None if record.lease_expires_at is None else record.lease_expires_at.isoformat()
            ),
            "updated_at": record.updated_at.isoformat(),
        }
        if record.assigned_worker_id not in (None, worker_id):
            raise HTTPException(status_code=404, detail="job not visible to this worker")
        return visible

    @app.post(
        "/v1/workers/{worker_id}/jobs/{job_id}/complete",
        dependencies=[Depends(require_worker)],
    )
    async def complete_job(worker_id: str, job_id: str, request: Request) -> dict[str, Any]:
        try:
            body = await _json_v1(request)
            lease_token = str(body["lease_token"])
            result = job_result_from_dict(body["result"])
            if result is None:
                raise ValueError("result is required")
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = await asyncio.to_thread(
            repository.complete_job,
            job_id,
            result,
            worker_id=worker_id,
            lease_token=lease_token,
        )
        return job_record_to_dict(record)

    @app.post(
        "/v1/workers/{worker_id}/jobs/{job_id}/fail",
        dependencies=[Depends(require_worker)],
    )
    async def fail_job(worker_id: str, job_id: str, request: Request) -> dict[str, Any]:
        try:
            body = await _json_v1(request)
            lease_token = str(body["lease_token"])
            error = body.get("error", "executor failed")
            if not isinstance(error, (str, dict)):
                raise TypeError("error must be a string or object")
            retryable = bool(body.get("retryable", False))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = await asyncio.to_thread(
            repository.fail_job,
            job_id,
            error,
            retryable=retryable,
            worker_id=worker_id,
            lease_token=lease_token,
        )
        return job_record_to_dict(record)

    return app


__all__ = ["create_app"]
