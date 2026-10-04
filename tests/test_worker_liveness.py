"""Liveness uses real Worker/Control code; timing and races are bounded locally."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import time

import httpx
import pytest

from astrumweaver import JobRequirements, JobResult, ResidencyReport, ResourceShape, WorkerSpec
from astrumweaver.control import (
    ConflictError, JobStatus, JobSubmission, WorkerRegistration, WorkerState, utc_now,
)
from astrumweaver.control.api import create_app
from astrumweaver.control.repository import InMemoryControlRepository
from astrumweaver.worker import ControlClient, ControlTransportError, WorkerRuntime
from astrumweaver.worker.health import create_health_app


SPEC = WorkerSpec(
    worker_id="liveness-worker", worker_class="test",
    resources=ResourceShape(gpu_count=1, total_vram_mb=1, max_single_gpu_vram_mb=1),
    gpu_uuids=("GPU-test-liveness",), capabilities=frozenset({"test.work"}),
)


class Executor:
    capabilities = SPEC.capabilities

    def __init__(self, *, delay=0.002, gate=None):
        self.delay = delay
        self.gate = gate
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.executions = []
        self.cancelled = []

    async def execute(self, job):
        self.executions.append(job.job_id)
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            else:
                await asyncio.sleep(self.delay)
            return JobResult(text="ok")
        finally:
            self.finished.set()

    async def cancel(self, job_id):
        self.cancelled.append(job_id)

    async def residency(self):
        return ResidencyReport()


class ObservedClient(ControlClient):
    def __init__(self, repository):
        app = create_app(
            repository, client_token=None, worker_token="worker-test-token",
            client_auth="none",
        )
        super().__init__(
            "http://control", "worker-test-token", timeout_seconds=0.3,
            transport=httpx.ASGITransport(app=app),
        )
        self.calls = []
        self.after = {}
        self.faults = {}
        self.in_flight = 0
        self.max_in_flight = 0

    async def _request(self, method, path, **kwargs):
        self.calls.append((path.rsplit("/", 1)[-1], time.monotonic(), kwargs.get("json")))
        kind = path.rsplit("/", 1)[-1]
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if kind in self.faults:
                raise ControlTransportError(self.faults[kind], "test transport failure")
            response = await super()._request(method, path, **kwargs)
            if hook := self.after.get(kind):
                await hook(response)
            return response
        finally:
            self.in_flight -= 1

    def count(self, kind):
        return sum(name == kind for name, _, _ in self.calls)


class BlockOnce:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, _response):
        if not self.entered.is_set():
            self.entered.set()
            await asyncio.wait_for(self.release.wait(), 2)


async def eventually(predicate, timeout=1.5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.003)


def submit(repo, count=1):
    return [
        repo.submit_job(JobSubmission(capability="test.work", payload={}, max_attempts=2))
        for _ in range(count)
    ]


@asynccontextmanager
async def running(repo=None, executor=None, *, poll=0.01, heartbeat=0.025, request_timeout=0.3):
    repo = repo or InMemoryControlRepository(worker_ttl_seconds=1, lease_seconds=1)
    executor = executor or Executor()
    async with ObservedClient(repo) as client:
        client.timeout_seconds = request_timeout
        runtime = WorkerRuntime(
            spec=SPEC, max_concurrency=1, executor=executor, client=client,
            heartbeat_interval_seconds=heartbeat, poll_interval_seconds=poll,
        )
        task = asyncio.create_task(runtime.run_forever(), name="test-worker-loop")
        try:
            await eventually(lambda: runtime.registered or task.done())
            if task.done():
                await task
                raise AssertionError("Worker exited during registration")
            yield repo, executor, client, runtime, task
        finally:
            # Test barriers cannot keep teardown blocked.
            for hook in client.after.values():
                if isinstance(hook, BlockOnce):
                    hook.release.set()
            if executor.gate is not None:
                executor.gate.set()
            runtime.request_stop()
            await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_continuous_short_job_backlog_survives_more_than_two_ttl_windows():
    repo = InMemoryControlRepository(worker_ttl_seconds=1, lease_seconds=1)
    submit(repo, 1200)
    async with running(repo) as (_, executor, client, runtime, task):
        start = time.monotonic()
        ages = []
        while time.monotonic() - start < 2.15:
            assert not task.done()
            assert not repo.expire_stale_workers()
            ages.append((utc_now() - repo.get_worker(SPEC.worker_id).last_seen_at).total_seconds())
            assert runtime.ready
            await asyncio.sleep(0.01)
        assert len(executor.executions) > 10
        assert any(j.status is JobStatus.QUEUED for j in repo.list_jobs()), "backlog exhausted"
        assert client.count("heartbeat") > 10
        assert max(ages) < 0.5
        assert repo.get_worker(SPEC.worker_id).state is WorkerState.ONLINE
        # Acknowledgements, not just a local heartbeat invocation, stayed fresh.
        times = [at for kind, at, _ in client.calls if kind == "heartbeat"]
        assert max(b - a for a, b in zip(times, times[1:])) < 0.5
        assert client.max_in_flight == 1


@pytest.mark.asyncio
async def test_negative_control_per_job_deadline_reset_reproduces_starvation():
    """Deliberately reintroduce the old postponement; this harness detects it."""
    class ResettingExecutor(Executor):
        runtime = None

        async def execute(self, job):
            result = await super().execute(job)
            self.runtime._next_heartbeat_at = time.monotonic() + 0.5
            return result

    executor = ResettingExecutor()
    repo = InMemoryControlRepository(worker_ttl_seconds=1, lease_seconds=1)
    async with running(repo, executor, heartbeat=0.5) as (_, _, client, runtime, _):
        executor.runtime = runtime
        submit(repo, 1200)
        # The original bug looks locally ONLINE until a heartbeat notices expiry.
        def expired():
            repo.expire_stale_workers()
            return repo.get_worker(SPEC.worker_id).state is WorkerState.OFFLINE
        await eventually(expired, timeout=1.4)
        assert client.count("heartbeat") == 0
        assert len(executor.executions) > 10
        assert any(j.status is JobStatus.QUEUED for j in repo.list_jobs())


@pytest.mark.asyncio
@pytest.mark.parametrize("drained", [False, True])
async def test_idle_and_draining_heartbeat_independent_of_long_poll_interval(drained):
    async with running(poll=3) as (repo, _, client, runtime, _):
        if drained:
            await runtime.drain()
        await asyncio.sleep(1.1)
        assert not repo.expire_stale_workers()
        assert client.count("heartbeat") > 10
        expected = WorkerState.DRAINING if drained else WorkerState.ONLINE
        assert repo.get_worker(SPEC.worker_id).state is expected
        assert runtime.ready is not drained


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [WorkerState.OFFLINE, WorkerState.DRAINING])
async def test_observed_non_online_state_blocks_claims_until_explicit_control_transition(state):
    async with running() as (repo, _, client, runtime, _):
        repo.set_worker_state(SPEC.worker_id, state)
        await eventually(lambda: runtime.control_state == state.value)
        before = client.count("claim")
        jobs = submit(repo)
        await asyncio.sleep(0.08)
        assert client.count("claim") == before
        assert not runtime.ready
        assert repo.get_job(jobs[0].job_id).attempts == 0
        # Runtime never attempts to promote/re-register itself.
        assert client.count("register") == 1
        assert client.count("state") == 0
        repo.set_worker_state(SPEC.worker_id, WorkerState.ONLINE)
        await eventually(lambda: repo.get_job(jobs[0].job_id).status is JobStatus.SUCCEEDED)
        assert runtime.ready


@pytest.mark.asyncio
async def test_offline_recovery_cannot_bypass_gpu_ownership_conflict():
    async with running() as (repo, _, client, runtime, _):
        repo.set_worker_state(SPEC.worker_id, WorkerState.OFFLINE)
        await eventually(lambda: runtime.control_state == "offline")
        competitor = replace(SPEC, worker_id="competing-owner")
        repo.register_worker(WorkerRegistration(spec=competitor))
        with pytest.raises(ConflictError):
            repo.set_worker_state(SPEC.worker_id, WorkerState.ONLINE)
        await runtime.drain()  # Local drain cannot reacquire released GPUs either.
        await asyncio.sleep(0.05)
        assert repo.get_worker(SPEC.worker_id).state is WorkerState.OFFLINE
        assert not runtime.ready
        assert client.count("register") == 1
        assert client.count("state") == 0


@pytest.mark.asyncio
async def test_local_drain_racing_offline_falls_back_to_stateless_heartbeat():
    async with running(heartbeat=0.03) as (repo, _, client, runtime, _):
        repo.set_worker_state(SPEC.worker_id, WorkerState.OFFLINE)
        # Last locally known state is ONLINE. Control must reject promotion.
        with pytest.raises(ControlTransportError) as error:
            await runtime.drain()
        assert error.value.status_code == 409
        await eventually(lambda: runtime.control_state == "offline")
        assert runtime.draining and not runtime.ready
        assert repo.get_worker(SPEC.worker_id).state is WorkerState.OFFLINE
        assert client.count("state") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 401, 403, 404, 409, 503])
async def test_control_errors_withdraw_ready_and_retry_without_new_claim_or_registration(code):
    async with running() as (repo, _, client, runtime, _):
        client.faults["heartbeat"] = code
        await eventually(lambda: not runtime.control_available)
        claims = client.count("claim")
        heartbeats = client.count("heartbeat")
        jobs = submit(repo)
        await asyncio.sleep(0.09)
        assert not runtime.ready
        assert client.count("claim") == claims
        assert 1 <= client.count("heartbeat") - heartbeats <= 6
        assert repo.get_job(jobs[0].job_id).attempts == 0
        assert client.count("register") == 1
        if code in (401, 403, 404):
            assert not runtime.registered
        del client.faults["heartbeat"]
        await eventually(lambda: repo.get_job(jobs[0].job_id).status is JobStatus.SUCCEEDED)
        assert runtime.ready


@pytest.mark.asyncio
async def test_missing_registration_stays_not_ready_without_blind_reenrollment():
    async with running() as (repo, _, client, runtime, _):
        with repo._lock:
            del repo._workers[SPEC.worker_id]
        await eventually(lambda: not runtime.registered)
        count = client.count("claim")
        await asyncio.sleep(0.08)
        assert not runtime.ready
        assert client.count("claim") == count
        assert client.count("register") == 1


@pytest.mark.asyncio
async def test_long_job_renews_only_current_lease():
    executor = Executor(delay=1.2)
    async with running(executor=executor) as (repo, _, client, runtime, _):
        job = submit(repo)[0]
        await asyncio.wait_for(executor.started.wait(), 1)
        lease = repo.get_job(job.job_id).lease_token
        await eventually(lambda: repo.get_job(job.job_id).status is JobStatus.SUCCEEDED, 1.8)
        renewals = [body for kind, _, body in client.calls if kind == "heartbeat" and body["active_job_id"]]
        assert len(renewals) > 10
        assert all(body["active_job_id"] == job.job_id and body["lease_token"] == lease for body in renewals)
        assert repo.get_job(job.job_id).attempts == 1
        assert runtime.ready


@pytest.mark.asyncio
async def test_heartbeat_finishes_before_terminal_write_and_waiting_drain_uses_no_stale_lease():
    gate = asyncio.Event()
    executor = Executor(gate=gate)
    async with running(executor=executor, heartbeat=0.05) as (repo, _, client, runtime, _):
        block = BlockOnce()
        client.after["heartbeat"] = block
        job = submit(repo)[0]
        await asyncio.wait_for(executor.started.wait(), 1)
        await asyncio.wait_for(block.entered.wait(), 1)
        gate.set()
        await asyncio.wait_for(executor.finished.wait(), 1)
        await asyncio.sleep(0.01)
        assert client.count("complete") == 0
        assert not repo.expire_stale_workers()
        block.release.set()
        await eventually(lambda: repo.get_job(job.job_id).status is JobStatus.SUCCEEDED)
        await runtime.drain()
        assert client.max_in_flight == 1
        completion = next(i for i, (kind, _, _) in enumerate(client.calls) if kind == "complete")
        assert all(body["active_job_id"] is None for kind, _, body in client.calls[completion+1:] if kind == "heartbeat")


@pytest.mark.asyncio
async def test_terminal_response_in_flight_serializes_signal_drain():
    async with running(heartbeat=0.04) as (repo, _, client, runtime, _):
        block = BlockOnce()
        client.after["complete"] = block
        job = submit(repo)[0]
        await asyncio.wait_for(block.entered.wait(), 1)
        assert repo.get_job(job.job_id).status is JobStatus.SUCCEEDED
        pending_drain = asyncio.create_task(runtime.drain())
        try:
            await asyncio.sleep(0.02)
            assert not pending_drain.done()
            assert not runtime.ready
            block.release.set()
            await asyncio.wait_for(pending_drain, 1)
        finally:
            block.release.set()
            await pending_drain
        assert client.max_in_flight == 1
        assert client.calls[-1][0] == "heartbeat"
        assert client.calls[-1][2]["active_job_id"] is None


@pytest.mark.asyncio
async def test_drain_during_pending_claim_finishes_only_that_admitted_attempt():
    async with running(heartbeat=0.05) as (repo, executor, client, runtime, _):
        block = BlockOnce()
        client.after["claim"] = block
        # A no-job claim might already be in flight. Inspect the first claimed
        # response so this barrier always belongs to the submitted work.
        async def after_claim(response):
            if response.status_code == 200:
                await block(response)
        client.after["claim"] = after_claim
        jobs = submit(repo, 2)
        await asyncio.wait_for(block.entered.wait(), 1)
        runtime.request_drain()
        drain = asyncio.create_task(runtime.drain())
        try:
            assert not runtime.ready
            block.release.set()
            await asyncio.wait_for(drain, 1)
            await eventually(lambda: repo.get_job(jobs[0].job_id).status is JobStatus.SUCCEEDED)
            await asyncio.sleep(0.08)
            assert repo.get_job(jobs[1].job_id).attempts == 0
            assert executor.executions == [jobs[0].job_id]
            assert client.max_in_flight == 1
        finally:
            block.release.set()
            await drain


@pytest.mark.asyncio
async def test_delayed_online_response_cannot_clear_local_stop_or_drain():
    async with running(heartbeat=0.04) as (repo, _, client, runtime, task):
        block = BlockOnce()
        client.after["heartbeat"] = block
        await asyncio.wait_for(block.entered.wait(), 1)
        runtime.request_stop()
        assert not runtime.ready
        block.release.set()
        await asyncio.wait_for(task, 1)
        assert not runtime.ready and not runtime.registered
        assert repo.get_worker(SPEC.worker_id).state is WorkerState.OFFLINE


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "expired", "transport"])
async def test_failed_active_lease_stops_executor_without_stale_terminal_write(failure):
    executor = Executor(gate=asyncio.Event())
    async with running(executor=executor) as (repo, _, client, runtime, _):
        job = submit(repo)[0]
        await asyncio.wait_for(executor.started.wait(), 1)
        if failure == "cancel":
            repo.cancel_job(job.job_id)
        elif failure == "expired":
            with repo._lock:
                repo._jobs[job.job_id] = repo.get_job(job.job_id).with_updates(
                    lease_expires_at=utc_now() - timedelta(seconds=1),
                )
        else:
            client.faults["heartbeat"] = 503
        await asyncio.wait_for(executor.finished.wait(), 1)
        await eventually(lambda: runtime.active_job_id is None)
        assert executor.cancelled == [job.job_id]
        assert client.count("complete") == 0 and client.count("fail") == 0
        assert repo.get_job(job.job_id).attempts == 1


@pytest.mark.asyncio
async def test_health_reports_observed_offline_and_control_loss():
    async with running() as (repo, _, client, runtime, _):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_health_app(runtime)), base_url="http://worker"
        ) as probe:
            repo.set_worker_state(SPEC.worker_id, WorkerState.OFFLINE)
            await eventually(lambda: runtime.control_state == "offline")
            assert (await probe.get("/ready")).status_code == 503
            body = (await probe.get("/health")).json()
            assert body["control_state"] == "offline" and body["registered"]
            client.faults["heartbeat"] = 503
            await eventually(lambda: not runtime.control_available)
            body = (await probe.get("/health")).json()
            assert not body["ready"] and not body["control_available"]


@pytest.mark.asyncio
@pytest.mark.parametrize("record", [{}, [], {"state": "online"}, {"protocol_version": "v1", "state": "mystery", "spec": {}}])
async def test_invalid_worker_acknowledgement_fails_closed(record):
    async with running() as (_, _, client, runtime, _):
        async def corrupt(response):
            response._content = __import__("json").dumps(record).encode()
        client.after["heartbeat"] = corrupt
        await eventually(lambda: not runtime.control_available)
        assert not runtime.ready
        assert client.count("register") == 1


@pytest.mark.asyncio
async def test_control_request_total_timeout_is_bounded_even_for_slow_custom_transport():
    class NeverReturns(httpx.AsyncBaseTransport):
        cancelled = False

        async def handle_async_request(self, request):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    transport = NeverReturns()
    async with ControlClient("http://control", "worker", transport=transport, timeout_seconds=0.02) as client:
        with pytest.raises(ControlTransportError) as error:
            await asyncio.wait_for(client.heartbeat("test"), 0.5)
        assert error.value.status_code == 0
        assert transport.cancelled


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_nonfinite_or_nonpositive_liveness_settings_rejected(value):
    with pytest.raises(ValueError):
        ControlClient("http://control", "worker", timeout_seconds=value)
    for key in ("heartbeat_interval_seconds", "poll_interval_seconds"):
        with pytest.raises(ValueError):
            WorkerRuntime(spec=SPEC, max_concurrency=1, executor=Executor(), client=None, **{key: value})


@pytest.mark.asyncio
async def test_readiness_ages_out_during_late_heartbeat_response():
    async with running(heartbeat=0.025, request_timeout=0.04) as (_, _, client, runtime, _):
        block = BlockOnce()
        client.after["heartbeat"] = block
        await asyncio.wait_for(block.entered.wait(), 1)
        await asyncio.sleep(0.09)
        assert not runtime.ready
        block.release.set()
        # The old acknowledgement may be received but is not fresh authority.
        await eventually(lambda: runtime._next_heartbeat_at > time.monotonic())
        assert not runtime.ready
        del client.after["heartbeat"]
        await eventually(lambda: runtime.ready)


@pytest.mark.asyncio
async def test_cancel_run_loop_cancels_active_executor_and_withdraws_registration():
    executor = Executor(gate=asyncio.Event())
    repo = InMemoryControlRepository()
    async with ObservedClient(repo) as client:
        runtime = WorkerRuntime(spec=SPEC, max_concurrency=1, executor=executor, client=client)
        task = asyncio.create_task(runtime.run_forever())
        try:
            await eventually(lambda: runtime.registered)
            job = submit(repo)[0]
            await asyncio.wait_for(executor.started.wait(), 1.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert executor.finished.is_set()
            assert executor.cancelled == [job.job_id]
            assert not runtime.ready and not runtime.registered
            assert repo.get_worker(SPEC.worker_id).state is WorkerState.OFFLINE
            assert not any(t.get_name() == "astrumweaver-job-execution" for t in asyncio.all_tasks())
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_daemon_failure_cleans_up_sibling_health_and_signal_tasks(monkeypatch):
    from astrumweaver.worker import daemon

    stopped = asyncio.Event()
    health_started = asyncio.Event()
    config = {
        "worker": {"id": "test", "class": "cpu", "control_url": "http://control", "capabilities": ["test.work"]},
        "executor": {"factory": "test:factory"},
    }

    class FailingWorker:
        registered = False

        def __init__(self, **kwargs):
            pass

        async def run_forever(self):
            await health_started.wait()
            raise RuntimeError("test worker failure")

        def request_stop(self):
            stopped.set()

    health_stopped = asyncio.Event()

    class HealthServer:
        should_exit = False

        def __init__(self, config):
            pass

        async def serve(self):
            health_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                health_stopped.set()

    monkeypatch.setenv("ASTRUMWEAVER_WORKER_TOKEN", "worker-test-token")
    monkeypatch.setattr(daemon, "_load_toml", lambda _: config)
    monkeypatch.setattr(daemon, "load_executor", lambda *_: Executor())
    monkeypatch.setattr(daemon, "WorkerRuntime", FailingWorker)
    monkeypatch.setattr(daemon, "create_health_app", lambda _: object())
    monkeypatch.setattr(daemon.uvicorn, "Config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(daemon.uvicorn, "Server", HealthServer)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *_: None)
    with pytest.raises(RuntimeError, match="test worker failure"):
        await asyncio.wait_for(daemon.run_worker("unused"), 1)
    assert stopped.is_set() and health_stopped.is_set()
    assert not any(t.get_name().startswith("astrumweaver-worker") for t in asyncio.all_tasks())
