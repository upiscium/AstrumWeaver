"""Small built-in executor useful for transport validation and smoke tests."""

from __future__ import annotations

from typing import Any, Mapping

from ..execution import JobRequest, JobResult, ResidencyReport


class StructuredEchoExecutor:
    capabilities = frozenset({"debug.echo"})

    async def execute(self, job: JobRequest) -> JobResult:
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
