"""Small built-in executor useful for transport and acceptance validation."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from ..execution import JobRequest, JobResult, ResidencyReport


class StructuredEchoExecutor:
    capabilities = frozenset({"debug.echo"})

    async def execute(self, job: JobRequest) -> JobResult:
        delay = job.payload.get("_debug_delay_seconds", 0)
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise TypeError("_debug_delay_seconds must be numeric")
        delay_seconds = float(delay)
        if delay_seconds < 0 or delay_seconds > 30:
            raise ValueError("_debug_delay_seconds must be between 0 and 30")
        if delay_seconds:
            await asyncio.sleep(delay_seconds)

        return JobResult(
            outputs={
                "capability": job.capability,
                "payload": dict(job.payload),
            },
            metadata={"executor": "structured-echo"},
        )

    async def cancel(self, job_id: str) -> None:
        return None

    async def residency(self) -> ResidencyReport:
        return ResidencyReport()


def create_executor(_: Mapping[str, Any] | None = None) -> StructuredEchoExecutor:
    return StructuredEchoExecutor()
