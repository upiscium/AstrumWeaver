"""FastAPI edge router for the bounded Stage A chat gateway."""

from __future__ import annotations

import asyncio
import json
import math
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
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
from .chat import (
    ChatGatewayError,
    ChatProfileCatalog,
    compile_chat_request,
    normalize_chat_completion,
)


def load_chat_catalog(path: str) -> ChatProfileCatalog:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot load chat gateway profile catalog") from exc
    if not isinstance(value, dict):
        raise RuntimeError("chat gateway profile catalog must contain an object")
    try:
        return ChatProfileCatalog.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("chat gateway profile catalog is invalid") from exc


def _error(
    status_code: int,
    *,
    code: str,
    message: str,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def _require_client(
    request: Request,
    *,
    mode: ClientAuthMode,
    token: str | None,
) -> None:
    if mode is ClientAuthMode.NONE:
        return
    assert token is not None
    header = request.headers.get("authorization", "")
    scheme, _, supplied = header.partition(" ")
    if (
        scheme.lower() != "bearer"
        or not supplied
        or not secrets.compare_digest(supplied, token)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="client authorization required",
        )


class ChatGatewayService:
    def __init__(
        self,
        repository: ControlRepository,
        catalog: ChatProfileCatalog,
        *,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be finite and positive")
        self.repository = repository
        self.catalog = catalog
        self.poll_interval_seconds = float(poll_interval_seconds)

    async def _cancel(self, job_id: str) -> None:
        try:
            await asyncio.to_thread(self.repository.cancel_job, job_id)
        except Exception:
            # Cancellation is best-effort at this edge. Durable Worker fencing
            # remains authoritative even if the acknowledgement is lost.
            return

    @staticmethod
    def _terminal(record, *, profile_id: str) -> dict[str, object]:
        if record.status is JobStatus.SUCCEEDED:
            if record.result is None:
                raise ChatGatewayError(
                    "invalid_provider_response",
                    "chat backend succeeded without a result",
                    status_code=502,
                )
            return normalize_chat_completion(
                record.result.outputs,
                profile_id=profile_id,
                job_id=record.job_id,
                created=int(record.created_at.timestamp()),
            )

        if record.status is JobStatus.CANCELLED:
            raise ChatGatewayError(
                "request_cancelled",
                "chat request was cancelled",
                status_code=409,
            )

        if record.status is not JobStatus.FAILED:
            raise RuntimeError("terminal conversion called for a non-terminal job")

        error = dict(record.error or {})
        code = error.get("code")
        error_type = error.get("type")
        if code in {
            "context_length_exceeded",
            "invalid_chat_job",
            "output_limit_exceeded",
            "streaming_not_supported",
            "unsupported_chat_adapter",
        }:
            message = error.get("message")
            raise ChatGatewayError(
                str(code),
                str(message) if isinstance(message, str) else "chat request rejected",
                status_code=400,
            )
        if error_type == "deadline_expired":
            raise ChatGatewayError(
                "deadline_exceeded",
                "chat request deadline expired",
                status_code=408,
            )
        raise ChatGatewayError(
            "backend_failure",
            "chat backend failed before producing a response",
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
            raise ChatGatewayError(
                "invalid_request",
                "model must be a configured profile ID",
            )
        profile = self.catalog.get(model)
        compiled = compile_chat_request(
            body,
            profile,
            request_size_bytes=request_size_bytes,
        )

        now = utc_now()
        deadline_at = now + timedelta(seconds=profile.request_timeout_seconds)
        try:
            record = await asyncio.to_thread(
                self.repository.submit_job,
                JobSubmission(
                    capability="llm.chat",
                    payload=compiled.payload,
                    max_attempts=profile.max_attempts,
                    deadline_at=deadline_at,
                    serving=profile.binding,
                ),
                now=now,
            )
        except DeadlineExceededError as exc:
            raise ChatGatewayError(
                "deadline_exceeded",
                "chat request deadline expired before admission",
                status_code=408,
            ) from exc
        except OverloadedError as exc:
            raise ChatGatewayError(
                "overloaded",
                "all compatible chat replicas are busy",
                status_code=429,
            ) from exc
        except NoCompatibleDeployment as exc:
            raise ChatGatewayError(
                "no_compatible_deployment",
                "no compatible chat deployment is available",
                status_code=503,
            ) from exc
        except ConflictError as exc:
            raise ChatGatewayError(
                "admission_conflict",
                "chat request conflicts with durable admission state",
                status_code=409,
            ) from exc

        job_id = record.job_id
        loop = asyncio.get_running_loop()
        stop_at = loop.time() + profile.request_timeout_seconds
        try:
            while True:
                current = await asyncio.to_thread(self.repository.get_job, job_id)
                if current.status in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    return self._terminal(current, profile_id=profile.profile_id)

                if await request.is_disconnected():
                    await self._cancel(job_id)
                    raise ChatGatewayError(
                        "client_disconnected",
                        "chat client disconnected",
                        status_code=499,
                    )

                remaining = stop_at - loop.time()
                if remaining <= 0:
                    await self._cancel(job_id)
                    raise ChatGatewayError(
                        "gateway_timeout",
                        "chat request exceeded the bounded gateway deadline",
                        status_code=504,
                    )
                await asyncio.sleep(min(self.poll_interval_seconds, remaining))
        except asyncio.CancelledError:
            await self._cancel(job_id)
            raise


def create_chat_router(
    repository: ControlRepository,
    catalog: ChatProfileCatalog,
    *,
    client_auth: ClientAuthMode | str,
    client_token: str | None,
    poll_interval_seconds: float = 0.05,
) -> APIRouter:
    mode = ClientAuthMode(client_auth)
    if mode is ClientAuthMode.BEARER and not client_token:
        raise ValueError("client_token is required when chat gateway uses bearer auth")

    service = ChatGatewayService(
        repository,
        catalog,
        poll_interval_seconds=poll_interval_seconds,
    )
    router = APIRouter()

    def require_client(request: Request) -> None:
        _require_client(request, mode=mode, token=client_token)

    @router.get("/v1/models", dependencies=[Depends(require_client)])
    async def models() -> dict[str, object]:
        return catalog.models_response()

    @router.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_client)],
    )
    async def chat_completions(request: Request):
        raw = await request.body()
        maximum = max(
            int(profile.resolved.effective_limits["request_bytes"])
            for profile in catalog.profiles
        )
        if len(raw) > maximum:
            return _error(
                413,
                code="request_too_large",
                message="chat request exceeds every configured profile byte limit",
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _error(
                400,
                code="invalid_json",
                message="chat request body must be valid JSON",
            )
        if not isinstance(value, dict):
            return _error(
                400,
                code="invalid_request",
                message="chat request body must be a JSON object",
            )
        try:
            result = await service.complete(
                request,
                value,
                request_size_bytes=len(raw),
            )
        except ChatGatewayError as exc:
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
    "ChatGatewayService",
    "create_chat_router",
    "load_chat_catalog",
]
