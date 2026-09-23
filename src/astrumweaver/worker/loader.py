"""Configuration-driven JobExecutor loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from ..execution import JobExecutor


class ExecutorLoadError(RuntimeError):
    """Configured executor factory cannot be loaded safely."""


def load_executor(
    factory_spec: str,
    settings: Mapping[str, Any],
) -> JobExecutor:
    module_name, separator, attribute_name = factory_spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ExecutorLoadError(
            "executor.factory must use 'python.module:factory' syntax"
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ExecutorLoadError(f"cannot import executor module: {module_name}") from exc

    factory = getattr(module, attribute_name, None)
    if factory is None or not callable(factory):
        raise ExecutorLoadError(
            f"executor factory is not callable: {factory_spec}"
        )

    try:
        executor = factory(dict(settings))
    except Exception as exc:
        raise ExecutorLoadError(
            f"executor factory failed: {factory_spec}"
        ) from exc

    if not isinstance(executor, JobExecutor):
        raise ExecutorLoadError(
            "executor factory result does not implement JobExecutor"
        )
    return executor
