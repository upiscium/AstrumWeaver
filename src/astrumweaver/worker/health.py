"""Local Worker health/readiness API."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Response

from .runtime import WorkerRuntime


def create_health_app(runtime: WorkerRuntime) -> FastAPI:
    app = FastAPI(title="AstrumWeaver Worker Local Health", version="v1")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "worker_id": runtime.spec.worker_id,
            "active_job_id": runtime.active_job_id,
            "draining": runtime.draining,
        }

    @app.get("/ready")
    async def ready(response: Response) -> dict[str, Any]:
        is_ready = runtime.ready and runtime.registered and not runtime.draining
        if not is_ready:
            response.status_code = 503
        return {
            "ready": is_ready,
            "worker_id": runtime.spec.worker_id,
            "active_job_id": runtime.active_job_id,
            "draining": runtime.draining,
        }

    return app
