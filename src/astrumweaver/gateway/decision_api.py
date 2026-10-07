"""FastAPI edge router for bounded shadow-mode decisions."""

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
from .decision import (
    CompiledDecisionRequest,
    DecisionGatewayError,
    DecisionGatewayProfile,
    DecisionProfileCatalog,
    compile_decision_request,
    normalize_decision_response,
)


def load_decision_catalog(path: str) -> DecisionProfileCatalog:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "cannot load decision gateway profile catalog"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            "decision gateway profile catalog must contain an object"
        )
    try:
        return DecisionProfileCatalog.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "decision gateway profile catalog is invalid"
        ) from exc


class DecisionGatewayService:
    def __init__(
        self,
        repository: ControlRepository,
        catalog: DecisionProfileCatalog,
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
        compiled: CompiledDecisionRequest,
    ) -> dict[str, object]:
        if record.status is JobStatus.SUCCEEDED:
            if record.result is None:
                raise DecisionGatewayError(
                    "invalid_provider_response",
                    "decision backend succeeded without a result",
                    status_code=502,
                )
            response = normalize_decision_response(
                record.result.outputs,
                compiled=compiled,
            )
            extension = response["x_astrumweaver"]
            timing: dict[str, float] = {}
            if record.started_at is not None:
                timing["queue_wait_ms"] = max(
                    0.0,
                    (
                        record.started_at - record.created_at
                    ).total_seconds()
                    * 1000.0,
                )
            if (
                record.started_at is not None
                and record.finished_at is not None
            ):
                timing["execution_ms"] = max(
                    0.0,
                    (
                        record.finished_at - record.started_at
                    ).total_seconds()
                    * 1000.0,
                )
            if record.finished_at is not None:
                timing["durable_job_ms"] = max(
                    0.0,
                    (
                        record.finished_at - record.created_at
                    ).total_seconds()
                    * 1000.0,
                )
            extension["timing"] = timing
            return response

        if record.status is JobStatus.CANCELLED:
            raise DecisionGatewayError(
                "request_cancelled",
                "decision request was cancelled",
                status_code=409,
            )
        if record.status is not JobStatus.FAILED:
            raise RuntimeError(
                "terminal conversion called for a non-terminal decision job"
            )

        error = dict(record.error or {})
        code = error.get("code")
        error_type = error.get("type")
        if code in {
            "decision_input_exceeded",
            "invalid_decision_job",
            "unsupported_decision_adapter",
            "decision_semantics_mismatch",
            "invalid_provider_response",
        }:
            message = error.get("message")
            raise DecisionGatewayError(
                str(code),
                str(message)
                if isinstance(message, str)
                else "decision request rejected",
                status_code=(
                    502 if code == "invalid_provider_response" else 400
                ),
            )
        if error_type == "deadline_expired":
            raise DecisionGatewayError(
                "deadline_exceeded",
                "decision request deadline expired",
                status_code=408,
            )
        raise DecisionGatewayError(
            "backend_failure",
            "decision backend failed before producing a valid answer",
            status_code=502,
        )

    async def complete(
        self,
        request: Request,
        body: dict[str, object],
        *,
        request_size_bytes: int,
    ) -> dict[str, object]:
        profile_id = body.get("profile")
        if not isinstance(profile_id, str):
            raise DecisionGatewayError(
                "invalid_request",
                "profile must be a configured decision profile ID",
            )
        profile = self.catalog.get(profile_id)
        compiled = compile_decision_request(
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
                    capability="decision.system_one",
                    payload=compiled.payload,
                    max_attempts=profile.max_attempts,
                    deadline_at=deadline_at,
                    serving=profile.binding,
                ),
                now=now,
            )
        except DeadlineExceededError as exc:
            raise DecisionGatewayError(
                "deadline_exceeded",
                "decision request deadline expired before admission",
                status_code=408,
            ) from exc
        except OverloadedError as exc:
            raise DecisionGatewayError(
                "overloaded",
                "all compatible decision replicas are busy",
                status_code=429,
            ) from exc
        except NoCompatibleDeployment as exc:
            raise DecisionGatewayError(
                "no_compatible_deployment",
                "no compatible decision deployment is available",
                status_code=503,
            ) from exc
        except ConflictError as exc:
            raise DecisionGatewayError(
                "admission_conflict",
                "decision request conflicts with durable admission state",
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
                    raise DecisionGatewayError(
                        "client_disconnected",
                        "decision client disconnected",
                        status_code=499,
                    )

                remaining = stop_at - loop.time()
                if remaining <= 0:
                    await self._cancel(job_id)
                    raise DecisionGatewayError(
                        "gateway_timeout",
                        "decision request exceeded the bounded gateway deadline",
                        status_code=504,
                    )
                await asyncio.sleep(
                    min(self.poll_interval_seconds, remaining)
                )
        except asyncio.CancelledError:
            await self._cancel(job_id)
            raise


def _profile_object(profile: DecisionGatewayProfile) -> dict[str, object]:
    semantics = profile.semantics
    return {
        "id": profile.profile_id,
        "object": "astrumweaver.decision_profile",
        "x_astrumweaver": {
            "profile_revision": profile.resolved.profile_revision,
            "deployment_revision": profile.resolved.deployment.revision,
            "serving_contract_revision": profile.resolved.contract.revision,
            "decision_semantics_id": semantics.decision_semantics_id,
            "score_kind": semantics.score_kind,
            "provider_score_semantics": semantics.provider_score_semantics,
            "calibration_status": semantics.calibration_status,
            "calibration_reference_sha256": (
                semantics.calibration_reference_sha256
            ),
            "abstain_below": float(semantics.abstain_below),
            "mode": semantics.mode,
            "authority": "recommendation-only",
            "effective_limits": dict(profile.resolved.effective_limits),
        },
    }


def create_decision_router(
    repository: ControlRepository,
    catalog: DecisionProfileCatalog,
    *,
    client_auth: ClientAuthMode | str,
    client_token: str | None,
    poll_interval_seconds: float = 0.05,
) -> APIRouter:
    mode = ClientAuthMode(client_auth)
    if mode is ClientAuthMode.BEARER and not client_token:
        raise ValueError(
            "client_token is required when decision gateway uses bearer auth"
        )
    service = DecisionGatewayService(
        repository,
        catalog,
        poll_interval_seconds=poll_interval_seconds,
    )
    router = APIRouter()

    def require_client(request: Request) -> None:
        _require_client(request, mode=mode, token=client_token)

    @router.get(
        "/v1/decision-profiles",
        dependencies=[Depends(require_client)],
    )
    async def decision_profiles() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                _profile_object(profile)
                for profile in catalog.profiles
            ],
        }

    @router.post(
        "/v1/decisions",
        dependencies=[Depends(require_client)],
    )
    async def decisions(request: Request):
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
                    "decision request exceeds every configured "
                    "profile byte limit"
                ),
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _error(
                400,
                code="invalid_json",
                message="decision request body must be valid JSON",
            )
        if not isinstance(value, dict):
            return _error(
                400,
                code="invalid_request",
                message="decision request body must be a JSON object",
            )
        try:
            result = await service.complete(
                request,
                value,
                request_size_bytes=len(raw),
            )
        except DecisionGatewayError as exc:
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
    "DecisionGatewayService",
    "create_decision_router",
    "load_decision_catalog",
]