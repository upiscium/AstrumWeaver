"""Adversarial overlapping composite lifecycle calls (#123)."""
from __future__ import annotations

import asyncio

import pytest

from test_dual_llama_cpp import managed_pair
from astrumweaver.runtime.contracts import RuntimeHealthState


@pytest.mark.asyncio
async def test_concurrent_start_calls_never_enter_child_start_together(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    entered = asyncio.Event()
    allow = asyncio.Event()
    original_start = embedding.start
    active = 0
    peak = 0

    async def blocked_embedding_start():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        try:
            await allow.wait()
            await original_start()
        finally:
            active -= 1

    embedding.start = blocked_embedding_start
    first = asyncio.create_task(managed.start())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(managed.start())
        await asyncio.sleep(0.03)
        assert peak == 1, "a second start entered the native child concurrently"
        allow.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        await managed.stop()
        assert not (await managed.health()).ready
    finally:
        allow.set()
        first.cancel()
        if second is not None:
            second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
        else:
            await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_cancels_and_joins_inflight_start_before_return(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    entered = asyncio.Event()
    never = asyncio.Event()

    async def stalled_native_start():
        entered.set()
        await never.wait()

    embedding.start = stalled_native_start
    starting = asyncio.create_task(managed.start())
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        stopping = asyncio.create_task(managed.stop())
        await asyncio.wait_for(stopping, 2)
        assert starting.done(), "stop returned while native startup is still alive"
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert embedding.health_state is RuntimeHealthState.STOPPED
        assert decision.health_state is RuntimeHealthState.STOPPED
        assert embedding.stops >= 1 and decision.stops >= 1
    finally:
        starting.cancel()
        if stopping is not None:
            stopping.cancel()
            await asyncio.gather(starting, stopping, return_exceptions=True)
        else:
            await asyncio.gather(starting, return_exceptions=True)


@pytest.mark.asyncio
async def test_release_joins_inflight_start_and_stops_both(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    entered = asyncio.Event()
    never = asyncio.Event()

    async def stalled_native_start():
        entered.set()
        await never.wait()

    embedding.start = stalled_native_start
    starting = asyncio.create_task(managed.start())
    releasing = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        releasing = asyncio.create_task(managed.release())
        await asyncio.wait_for(releasing, 2)
        assert starting.done(), "release returned before an owned child finished starting"
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert embedding.stops >= 1 and decision.stops >= 1
        assert embedding.releases == 1 and decision.releases == 1
        with pytest.raises(RuntimeError, match="released"):
            await managed.start()
    finally:
        starting.cancel()
        if releasing is not None:
            releasing.cancel()
            await asyncio.gather(starting, releasing, return_exceptions=True)
        else:
            await asyncio.gather(starting, return_exceptions=True)


@pytest.mark.asyncio
async def test_release_ready_runtime_stops_owned_children(tmp_path):
    managed, embedding, decision = managed_pair(tmp_path)
    await managed.start()
    assert (await managed.health()).ready
    await managed.release()
    assert embedding.health_state is RuntimeHealthState.STOPPED
    assert decision.health_state is RuntimeHealthState.STOPPED
    assert embedding.stops == 1 and decision.stops == 1
    assert embedding.releases == 1 and decision.releases == 1
