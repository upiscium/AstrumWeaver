"""Shared lifecycle coordinator for ManagedRuntime implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .contracts import (
    ManagedRuntime,
    RuntimeHealth,
    RuntimeHealthState,
)


class RuntimeLifecycleError(RuntimeError):
    pass


@dataclass(slots=True)
class RuntimeLifecycleManager:
    runtime: ManagedRuntime
    poll_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._released = False

    async def ensure_ready(
        self,
        *,
        timeout_seconds: float = 60.0,
    ) -> RuntimeHealth:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        health = await self.runtime.health()
        if health.ready:
            self._released = False
            return health
        if health.state is RuntimeHealthState.FAILED:
            raise RuntimeLifecycleError(
                "runtime is failed before start; inspect provider diagnostics"
            )

        if health.state is RuntimeHealthState.STOPPED:
            await self.runtime.start()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            health = await self.runtime.health()
            if health.ready:
                self._released = False
                return health
            if health.state is RuntimeHealthState.FAILED:
                raise RuntimeLifecycleError(
                    "runtime entered failed state while starting"
                )
            if loop.time() >= deadline:
                raise RuntimeLifecycleError(
                    "runtime did not become ready before timeout"
                )
            await asyncio.sleep(self.poll_interval_seconds)

    async def ensure_stopped(
        self,
        *,
        timeout_seconds: float = 60.0,
        release: bool = False,
    ) -> RuntimeHealth:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        health = await self.runtime.health()
        if health.state is not RuntimeHealthState.STOPPED:
            await self.runtime.stop()

            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_seconds
            while True:
                health = await self.runtime.health()
                if health.state is RuntimeHealthState.STOPPED:
                    break
                if health.state is RuntimeHealthState.FAILED:
                    raise RuntimeLifecycleError(
                        "runtime entered failed state while stopping"
                    )
                if loop.time() >= deadline:
                    raise RuntimeLifecycleError(
                        "runtime did not stop before timeout"
                    )
                await asyncio.sleep(self.poll_interval_seconds)

        if release and not self._released:
            await self.runtime.release()
            self._released = True

        return health

    async def restart(
        self,
        *,
        timeout_seconds: float = 60.0,
    ) -> RuntimeHealth:
        await self.ensure_stopped(
            timeout_seconds=timeout_seconds,
            release=False,
        )
        return await self.ensure_ready(timeout_seconds=timeout_seconds)
