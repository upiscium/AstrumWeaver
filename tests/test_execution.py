from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from astrumweaver import (
    ArtifactRef,
    JobExecutor,
    JobRequest,
    JobResult,
    ResidencyItem,
    ResidencyReport,
)
from astrumweaver.executors import TextGenerationExecutor


def test_non_text_result_supports_structured_output_and_artifact() -> None:
    artifact = ArtifactRef(
        uri="artifact://job-1/image.png",
        media_type="image/png",
        digest="sha256:example",
        size_bytes=1024,
    )
    result = JobResult(
        outputs={"width": 1024, "height": 1024, "seed": 42},
        artifacts=(artifact,),
        metrics={"elapsed_ms": 125.5},
    )

    assert result.text is None
    assert result.outputs["width"] == 1024
    assert result.artifacts == (artifact,)
    assert result.metrics["elapsed_ms"] == 125.5


def test_job_request_is_capability_oriented_not_application_typed() -> None:
    request = JobRequest(
        job_id="job-1",
        capability="custom.future.workload",
        payload={"opaque": {"value": 1}},
    )

    assert request.capability == "custom.future.workload"
    assert request.payload["opaque"] == {"value": 1}


def test_residency_report_is_not_model_specific() -> None:
    report = ResidencyReport(
        items=(
            ResidencyItem(
                name="runtime-cache",
                kind="executor-cache",
                accelerator_memory_bytes=256,
            ),
            ResidencyItem(
                name="weights",
                kind="model",
                accelerator_memory_bytes=1024,
            ),
        )
    )

    assert report.accelerator_memory_bytes == 1280


class FakeTextRuntime:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.payloads: list[dict[str, object]] = []
        self.report = ResidencyReport(
            items=(ResidencyItem(name="example-model", kind="model", accelerator_memory_bytes=4096),)
        )

    async def generate(self, payload):
        self.payloads.append(dict(payload))
        return {"response": "hello", "tokens": 3}

    async def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    async def residency(self) -> ResidencyReport:
        return self.report


def test_text_generation_is_only_an_executor_adapter() -> None:
    async def scenario() -> None:
        runtime = FakeTextRuntime()
        executor = TextGenerationExecutor(runtime)
        request = JobRequest(
            job_id="job-text",
            capability="llm.chat",
            payload={"model": "example", "prompt": "hi"},
        )

        result = await executor.execute(request)

        assert result.text == "hello"
        assert result.outputs["tokens"] == 3
        assert runtime.payloads == [{"model": "example", "prompt": "hi"}]
        await executor.cancel("job-text")
        assert runtime.cancelled == ["job-text"]
        assert await executor.residency() == runtime.report

    asyncio.run(scenario())


def test_text_executor_satisfies_generic_protocol() -> None:
    executor = TextGenerationExecutor(FakeTextRuntime())

    assert isinstance(executor, JobExecutor)


def test_artifact_size_and_metric_types_are_validated() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        ArtifactRef(uri="artifact://x", size_bytes=-1)

    with pytest.raises(TypeError, match="metric values"):
        JobResult(metrics={"ok": True})
