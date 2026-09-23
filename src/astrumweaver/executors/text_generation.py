"""Compatibility adapter for text-generation runtimes.

Text generation is one executor implementation, not part of the AstrumWeaver
control-plane contract.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Protocol

from ..execution import JobExecutor, JobRequest, JobResult, ResidencyReport


class TextGenerationRuntime(Protocol):
    async def generate(self, payload: Mapping[str, Any]) -> Any:
        ...

    async def cancel(self, job_id: str) -> None:
        ...

    async def residency(self) -> ResidencyReport:
        ...


class TextGenerationExecutor(JobExecutor):
    """Wrap a text runtime behind the generic JobExecutor boundary."""

    def __init__(
        self,
        runtime: TextGenerationRuntime,
        *,
        capabilities: frozenset[str] | None = None,
    ) -> None:
        self.runtime = runtime
        runtime_capabilities = getattr(runtime, "capabilities", None)
        configured = capabilities if capabilities is not None else runtime_capabilities
        self.capabilities = frozenset(configured or {"text.generate"})

    async def execute(self, job: JobRequest) -> JobResult:
        value = self.runtime.generate(job.payload)
        if inspect.isawaitable(value):
            value = await value
        return _coerce_text_result(value)

    async def cancel(self, job_id: str) -> None:
        value = self.runtime.cancel(job_id)
        if inspect.isawaitable(value):
            await value

    async def residency(self) -> ResidencyReport:
        value = self.runtime.residency()
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, ResidencyReport):
            raise TypeError("text runtime residency() must return ResidencyReport")
        return value


def _coerce_text_result(value: Any) -> JobResult:
    if isinstance(value, JobResult):
        return value

    if isinstance(value, str):
        return JobResult(outputs={"text": value}, text=value)

    if isinstance(value, Mapping):
        outputs = dict(value)
        text = _extract_text(outputs)
        return JobResult(outputs=outputs, text=text)

    text = getattr(value, "text", None)
    if isinstance(text, str):
        raw = getattr(value, "raw", None)
        outputs = dict(raw) if isinstance(raw, Mapping) else {"text": text}
        outputs.setdefault("text", text)
        return JobResult(outputs=outputs, text=text)

    raise TypeError("text runtime returned an unsupported result type")


def _extract_text(value: Mapping[str, Any]) -> str | None:
    for key in ("text", "response", "content"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
    return None
