"""FastAPI edge router for bounded immutable-space embedding serving."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..control.auth import ClientAuthMode
from ..control.models import JobStatus, JobSubmission, utc_now
from ..control.repository import (
    ConflictError,
    ControlRepository,
    DeadlineExceededError,
    NoCompatibleDeployment,
    OverloadedError,
)
from .api import _error, _require_client
from .embedding import (
    CompiledEmbeddingRequest,
    EmbeddingGatewayError,
    EmbeddingGatewayProfile,
    EmbeddingProfileCatalog,
    compile_embedding_request,
    normalize_embedding_response,
)


def load_embedding_catalog(path: str) -> EmbeddingProfileCatalog:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "cannot load embedding gateway profile catalog"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            "embedding gateway profile catalog must contain an object"
        )
    try:
        return EmbeddingProfileCatalog.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "embedding gateway profile catalog is invalid"
        ) from exc


class EmbeddingGatewayService:
    def __init__(
        self,
        repository: ControlRepository,
        catalog: EmbeddingProfileCatalog,
        *,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError(
                "poll_interval_seconds must be finite and positive"
            )
        self.repository = repository
        self.catalog = catalog
        self.poll_interval_seconds = float(poll_interval_seconds)

    async def _cancel(self, job_id: str) -> None:
        try:
            await asyncio.to_thread(self.repository.cancel_job, job_id)
        except Exception:
            return

    @staticmethod
    def _terminal(
        record,
        *,
        compiled: CompiledEmbeddingRequest,
    ) -> dict[str, object]:
        if record.status is JobStatus.SUCCEEDED:
            if record.result is None:
                raise EmbeddingGatewayError(
                    "invalid_provider_response",
                    "embedding backend succeeded without a result",
                    status_code=502,
                )
            return normalize_embedding_response(
                record.result.outputs,
                profile=compiled.profile,
                input_type=compiled.input_type,
                item_count=compiled.item_count,
            )

        if record.status is JobStatus.CANCELLED:
            raise EmbeddingGatewayError(
                "request_cancelled",
                "embedding request was cancelled",
                status_code=409,
            )
        if record.status is not JobStatus.FAILED:
            raise RuntimeError(
                "terminal conversion called for a non-terminal embedding job"
            )

        error = dict(record.error or {})
        code = error.get("code")
        error_type = error.get("type")
        if code in {
            "embedding_batch_exceeded",
            "embedding_input_exceeded",
            "invalid_embedding_job",
            "unsupported_embedding_adapter",
            "unsupported_embedding_encoding",
        }:
            message = error.get("message")
            raise EmbeddingGatewayError(
                str(code),
                str(message)
                if isinstance(message, str)
                else "embedding request rejected",
                status_code=400,
            )
        if error_type == "deadline_expired":
            raise EmbeddingGatewayError(
                "deadline_exceeded",
                "embedding request deadline expired",
                status_code=408,
            )
        raise EmbeddingGatewayError(
            "backend_failure",
            "embedding backend failed before producing a valid vector batch",
            status_code=502,
        )

    async def complete(
        self,
        request: Request,
        body: dict[str, object],
        *,
        request_size_bytes: int,
    ) -> dict[str, object]:
        model = body.get("model")
        if not isinstance(model, str):
            raise EmbeddingGatewayError(
                "invalid_request",
                "model must be a configured embedding profile ID",
            )
        profile = self.catalog.get(model)
        compiled = compile_embedding_request(
            body,
            profile,
            request_size_bytes=request_size_bytes,
        )
        now = utc_now()
        deadline_at = now + timedelta(
            seconds=profile.request_timeout_seconds
        )
        try:
            record = await asyncio.to_thread(
                self.repository.submit_job,
                JobSubmission(
                    capability="text.embed",
                    payload=compiled.payload,
                    max_attempts=profile.max_attempts,
                    deadline_at=deadline_at,
                    serving=profile.binding,
                ),
                now=now,
            )
        except DeadlineExceededError as exc:
            raise EmbeddingGatewayError(
                "deadline_exceeded",
                "embedding request deadline expired before admission",
                status_code=408,
            ) from exc
        except OverloadedError as exc:
            raise EmbeddingGatewayError(
                "overloaded",
                "all compatible embedding replicas are busy",
                status_code=429,
            ) from exc
        except NoCompatibleDeployment as exc:
            raise EmbeddingGatewayError(
                "no_compatible_deployment",
                "no compatible embedding deployment is available",
                status_code=503,
            ) from exc
        except ConflictError as exc:
            raise EmbeddingGatewayError(
                "admission_conflict",
                "embedding request conflicts with durable admission state",
                status_code=409,
            ) from exc

        loop = asyncio.get_running_loop()
        stop_at = loop.time() + profile.request_timeout_seconds
        job_id = record.job_id
        try:
            while True:
                current = await asyncio.to_thread(
                    self.repository.get_job,
                    job_id,
                )
                if current.status in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    return self._terminal(
                        current,
                        compiled=compiled,
                    )

                if await request.is_disconnected():
                    await self._cancel(job_id)
                    raise EmbeddingGatewayError(
                        "client_disconnected",
                        "embedding client disconnected",
                        status_code=499,
                    )

                remaining = stop_at - loop.time()
                if remaining <= 0:
                    await self._cancel(job_id)
                    raise EmbeddingGatewayError(
                        "gateway_timeout",
                        "embedding request exceeded the bounded gateway deadline",
                        status_code=504,
                    )
                await asyncio.sleep(
                    min(self.poll_interval_seconds, remaining)
                )
        except asyncio.CancelledError:
            await self._cancel(job_id)
            raise


def _space_object(profile: EmbeddingGatewayProfile) -> dict[str, object]:
    return {
        "id": profile.profile_id,
        "object": "astrumweaver.embedding_space",
        "x_astrumweaver": {
            "embedding_space_id": profile.embedding_space_id,
            "profile_revision": profile.resolved.profile_revision,
            "deployment_revision": profile.resolved.deployment.revision,
            "serving_contract_revision": profile.resolved.contract.revision,
            "dimensions": profile.space.dimensions,
            "pooling": profile.space.pooling,
            "normalization": profile.space.normalization,
            "input_types": ["query", "document"],
            "effective_limits": dict(profile.resolved.effective_limits),
        },
    }


def create_embedding_router(
    repository: ControlRepository,
    catalog: EmbeddingProfileCatalog,
    *,
    client_auth: ClientAuthMode | str,
    client_token: str | None,
    poll_interval_seconds: float = 0.05,
) -> APIRouter:
    mode = ClientAuthMode(client_auth)
    if mode is ClientAuthMode.BEARER and not client_token:
        raise ValueError(
            "client_token is required when embedding gateway uses bearer auth"
        )
    service = EmbeddingGatewayService(
        repository,
        catalog,
        poll_interval_seconds=poll_interval_seconds,
    )
    router = APIRouter()

    def require_client(request: Request) -> None:
        _require_client(request, mode=mode, token=client_token)

    @router.get(
        "/v1/embedding-spaces",
        dependencies=[Depends(require_client)],
    )
    async def embedding_spaces() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                _space_object(profile)
                for profile in catalog.profiles
            ],
        }

    @router.post(
        "/v1/embeddings",
        dependencies=[Depends(require_client)],
    )
    async def embeddings(request: Request):
        raw = await request.body()
        maximum = max(
            int(profile.resolved.effective_limits["request_bytes"])
            for profile in catalog.profiles
        )
        if len(raw) > maximum:
            return _error(
                413,
                code="request_too_large",
                message=(
                    "embedding request exceeds every configured "
                    "profile byte limit"
                ),
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _error(
                400,
                code="invalid_json",
                message="embedding request body must be valid JSON",
            )
        if not isinstance(value, dict):
            return _error(
                400,
                code="invalid_request",
                message="embedding request body must be a JSON object",
            )
        try:
            result = await service.complete(
                request,
                value,
                request_size_bytes=len(raw),
            )
        except EmbeddingGatewayError as exc:
            return _error(
                exc.status_code,
                code=exc.code,
                message=str(exc),
                error_type=(
                    "server_error"
                    if exc.status_code >= 500
                    else "invalid_request_error"
                ),
            )
        return JSONResponse(status_code=200, content=result)

    return router


__all__ = [
    "EmbeddingGatewayService",
    "create_embedding_router",
    "load_embedding_catalog",
]
