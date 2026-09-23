"""FastAPI service exposing the AstrumWeaver v1 Control protocol."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Any, Awaitable, Callable, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse

from ..transport import (
    AuthConfig,
    ClaimedJobDTO,
    HealthDTO,
    JobCompletionDTO,
    JobFailureDTO,
    JobStatusDTO,
    JobSubmissionDTO,
    JobViewDTO,
    PROTOCOL_VERSION,
    WorkerHeartbeatDTO,
    WorkerRecordDTO,
    WorkerRegistrationDTO,
    WorkerStateUpdateDTO,
)
from ..transport.auth import bearer_guard
from .repository import (
    ConflictError,
    ControlRepository,
    NotFoundError,
    RepositoryError,
    StorageUnavailable,
)

T = TypeVar("T")


async def _call_sync(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(function, *args, **kwargs)


def _unprocessable(exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=str(exc),
    )


async def _maintenance_loop(
    repository: ControlRepository,
    interval_seconds: float,
) -> None:
    while True:
        try:
            await _call_sync(repository.expire_stale_workers)
            await _call_sync(repository.recover_expired_jobs)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Readiness exposes storage failure.  The maintenance loop must not
            # crash the API process or leak backend exception details.
            pass
        await asyncio.sleep(interval_seconds)


def create_app(
    *,
    repository: ControlRepository,
    auth: AuthConfig,
    maintenance_interval_seconds: float = 5.0,
) -> FastAPI:
    if maintenance_interval_seconds <= 0:
        raise ValueError("maintenance_interval_seconds must be positive")

    client_guard = bearer_guard(auth.client_token)
    worker_guard = bearer_guard(auth.worker_token)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(
            _maintenance_loop(repository, maintenance_interval_seconds),
            name="astrumweaver-control-maintenance",
        )
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="AstrumWeaver Control",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_handler(_, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(StorageUnavailable)
    async def storage_handler(_, __: StorageUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "control storage unavailable"},
        )

    @app.exception_handler(RepositoryError)
    async def repository_handler(_, __: RepositoryError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "control repository error"},
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(_, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"},
        )

    @app.get("/v1/healthz", response_model=HealthDTO)
    async def health() -> HealthDTO:
        return HealthDTO(status="ok")

    @app.get("/v1/readyz", response_model=HealthDTO)
    async def ready() -> HealthDTO:
        await _call_sync(repository.health)
        return HealthDTO(status="ready")

    @app.post(
        "/v1/jobs",
        response_model=JobViewDTO,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(client_guard)],
    )
    async def submit_job(request: JobSubmissionDTO) -> JobViewDTO:
        try:
            record = await _call_sync(repository.submit_job, request.to_domain())
        except ValueError as exc:
            raise _unprocessable(exc) from exc
        return JobViewDTO.from_domain(record)

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobViewDTO,
        dependencies=[Depends(client_guard)],
    )
    async def get_job(job_id: str) -> JobViewDTO:
        record = await _call_sync(repository.get_job, job_id)
        return JobViewDTO.from_domain(record)

    @app.post(
        "/v1/jobs/{job_id}/cancel",
        response_model=JobViewDTO,
        dependencies=[Depends(client_guard)],
    )
    async def cancel_job(job_id: str) -> JobViewDTO:
        record = await _call_sync(repository.cancel_job, job_id)
        return JobViewDTO.from_domain(record)

    @app.post(
        "/v1/workers/register",
        response_model=WorkerRecordDTO,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(worker_guard)],
    )
    async def register_worker(request: WorkerRegistrationDTO) -> WorkerRecordDTO:
        try:
            record = await _call_sync(repository.register_worker, request.to_domain())
        except ValueError as exc:
            raise _unprocessable(exc) from exc
        return WorkerRecordDTO.from_domain(record)

    @app.post(
        "/v1/workers/{worker_id}/heartbeat",
        response_model=WorkerRecordDTO,
        dependencies=[Depends(worker_guard)],
    )
    async def heartbeat_worker(
        worker_id: str,
        request: WorkerHeartbeatDTO,
    ) -> WorkerRecordDTO:
        try:
            record = await _call_sync(
                repository.heartbeat_worker,
                worker_id,
                request.to_domain(),
            )
        except ValueError as exc:
            raise _unprocessable(exc) from exc
        return WorkerRecordDTO.from_domain(record)

    @app.put(
        "/v1/workers/{worker_id}/state",
        response_model=WorkerRecordDTO,
        dependencies=[Depends(client_guard)],
    )
    async def set_worker_state(
        worker_id: str,
        request: WorkerStateUpdateDTO,
    ) -> WorkerRecordDTO:
        record = await _call_sync(
            repository.set_worker_state,
            worker_id,
            request.state,
        )
        return WorkerRecordDTO.from_domain(record)

    @app.post(
        "/v1/workers/{worker_id}/jobs/claim",
        response_model=ClaimedJobDTO,
        dependencies=[Depends(worker_guard)],
        responses={204: {"description": "No eligible job"}},
    )
    async def claim_job(worker_id: str) -> ClaimedJobDTO | Response:
        record = await _call_sync(repository.claim_next_job, worker_id)
        if record is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return ClaimedJobDTO.from_domain(record)

    @app.get(
        "/v1/workers/{worker_id}/jobs/{job_id}",
        response_model=JobStatusDTO,
        dependencies=[Depends(worker_guard)],
    )
    async def worker_job_status(worker_id: str, job_id: str) -> JobStatusDTO:
        # Worker auth is role-level in v1. worker_id is retained in the URL for
        # an evolution path to per-worker credentials without changing routes.
        _ = worker_id
        record = await _call_sync(repository.get_job, job_id)
        return JobStatusDTO.from_domain(record)

    @app.post(
        "/v1/jobs/{job_id}/complete",
        response_model=JobViewDTO,
        dependencies=[Depends(worker_guard)],
    )
    async def complete_job(job_id: str, request: JobCompletionDTO) -> JobViewDTO:
        record = await _call_sync(
            repository.complete_job,
            job_id,
            request.result.to_domain(),
            worker_id=request.worker_id,
            lease_token=request.lease_token,
        )
        return JobViewDTO.from_domain(record)

    @app.post(
        "/v1/jobs/{job_id}/fail",
        response_model=JobViewDTO,
        dependencies=[Depends(worker_guard)],
    )
    async def fail_job(job_id: str, request: JobFailureDTO) -> JobViewDTO:
        record = await _call_sync(
            repository.fail_job,
            job_id,
            request.error,
            retryable=request.retryable,
            worker_id=request.worker_id,
            lease_token=request.lease_token,
        )
        return JobViewDTO.from_domain(record)

    return app
