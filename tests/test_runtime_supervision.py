"""Bounded post-start supervision with real Worker/Control and subprocesses."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import importlib
import sys
import time

import httpx
import pytest

from astrumweaver.control import ConflictError, JobStatus, WorkerRegistration, WorkerState, utc_now
from astrumweaver.runtime import RuntimeHealth, RuntimeHealthState, RuntimeLifecycleManager, RuntimeLifecycleError
from astrumweaver.worker import WorkerRuntime
from astrumweaver.worker.health import create_health_app
from astrumweaver.worker.supervision import RuntimeHealthSupervisor
from test_worker_liveness import SPEC, Executor, ObservedClient, BlockOnce, eventually, submit
from astrumweaver.control.repository import InMemoryControlRepository


READY = RuntimeHealth(state=RuntimeHealthState.READY, ready=True)


class Managed:
    provider_id = "test-runtime"

    def __init__(self, executor=None):
        self.value = READY
        self.block = None
        self.probes = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.stops = 0
        self.releases = 0
        self._executor = executor or Executor()

    async def start(self):
        self.value = READY

    async def stop(self):
        self.stops += 1
        self.value = RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False)

    async def release(self):
        self.releases += 1

    async def health(self):
        self.probes += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.block is not None:
                await self.block.wait()
            if isinstance(self.value, Exception):
                raise self.value
            return self.value
        finally:
            self.in_flight -= 1

    def executor(self):
        return self._executor


@asynccontextmanager
async def supervised(managed=None, repo=None, *, interval=0.02, timeout=0.2):
    managed = managed or Managed()
    repo = repo or InMemoryControlRepository()
    supervisor = RuntimeHealthSupervisor(managed, interval_seconds=interval, timeout_seconds=timeout)
    async with ObservedClient(repo) as client:
        worker = WorkerRuntime(
            spec=SPEC, max_concurrency=1, executor=managed.executor(), client=client,
            poll_interval_seconds=0.005, heartbeat_interval_seconds=0.025,
            runtime_supervisor=supervisor,
        )
        task = asyncio.create_task(worker.run_forever())
        try:
            await eventually(lambda: worker.registered or supervisor.failed.is_set() or task.done())
            if task.done():
                await task
            yield repo, managed, supervisor, client, worker, task
        finally:
            for hook in client.after.values():
                if isinstance(hook, BlockOnce):
                    hook.release.set()
            if managed.block is not None:
                managed.block.set()
            if managed.executor().gate is not None:
                managed.executor().gate.set()
            worker.request_stop()
            await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False),
    RuntimeHealth(state=RuntimeHealthState.STARTING, ready=False),
    RuntimeHealth(state=RuntimeHealthState.DEGRADED, ready=False),
    RuntimeHealth(state=RuntimeHealthState.FAILED, ready=False),
    RuntimeHealth(state=RuntimeHealthState.FAILED, ready=True),
    RuntimeHealth(state=RuntimeHealthState.READY, ready=1),
    RuntimeHealth(state=RuntimeHealthState.READY, ready=False, detail="private-sentinel", metadata={"token": "private-sentinel"}),
    None,
    RuntimeError("private-sentinel"),
    "hang",
])
async def test_post_ready_fault_latches_readiness_and_preserves_queued_attempts(fault, caplog):
    async with supervised() as (repo, managed, supervisor, client, worker, task):
        assert worker.ready
        if fault == "hang":
            managed.block = asyncio.Event()
        else:
            managed.value = fault
        await asyncio.wait_for(supervisor.failed.wait(), 1)
        await eventually(lambda: client.in_flight == 0 and worker.active_job_id is None)
        claims = client.count("claim")
        beats = client.count("heartbeat")
        jobs = submit(repo, 3)
        # Returning a healthy endpoint/ONLINE state cannot clear quarantine.
        managed.value = READY
        if managed.block is not None:
            managed.block.set()
        repo.set_worker_state(SPEC.worker_id, WorkerState.ONLINE)
        await asyncio.sleep(0.09)
        assert not task.done()
        assert client.count("claim") == claims
        assert client.count("heartbeat") > beats
        assert all(repo.get_job(job.job_id).attempts == 0 for job in jobs)
        assert not worker.ready and worker.registered and worker.draining
        assert client.count("register") == 1
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_health_app(worker)), base_url="http://worker") as probe:
            assert (await probe.get("/ready")).status_code == 503
            health = (await probe.get("/health")).json()
            assert health["runtime_state"] == "failed"
            assert health["runtime_available"] is False
            assert "private-sentinel" not in str(health)
        assert "private-sentinel" not in caplog.text


@pytest.mark.asyncio
async def test_initial_post_start_probe_failure_prevents_registration():
    managed = Managed()
    managed.value = RuntimeHealth(state=RuntimeHealthState.FAILED, ready=False)
    async with supervised(managed) as (_, _, _, client, worker, task):
        assert not worker.ready and not worker.registered
        assert client.count("register") == 0 and client.count("claim") == 0
        assert not task.done()  # no automatic systemd failure/restart loop


@pytest.mark.asyncio
async def test_failed_active_runtime_cancels_once_and_leaves_terminal_fencing_to_control():
    managed = Managed(Executor(gate=asyncio.Event()))
    async with supervised(managed) as (repo, _, supervisor, client, worker, _):
        first, second = submit(repo, 2)
        await asyncio.wait_for(managed.executor().started.wait(), 1)
        claimed = repo.get_job(first.job_id)
        managed.value = RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False)
        await asyncio.wait_for(supervisor.failed.wait(), 1)
        await asyncio.wait_for(managed.executor().finished.wait(), 1)
        await eventually(lambda: worker.active_job_id is None)
        assert managed.executor().cancelled == [first.job_id]
        assert client.count("complete") == client.count("fail") == 0
        assert repo.get_job(second.job_id).attempts == 0
        repo.recover_expired_jobs(now=claimed.lease_expires_at + timedelta(seconds=1))
        with pytest.raises(ConflictError):
            repo.complete_job(first.job_id, __import__('astrumweaver').JobResult(text="stale"), worker_id=SPEC.worker_id, lease_token=claimed.lease_token)
        await asyncio.sleep(0.05)
        assert repo.get_job(second.job_id).attempts == 0
        assert not any(t.get_name() in {"astrumweaver-job-execution", "astrumweaver-runtime-failure"} for t in asyncio.all_tasks())


@pytest.mark.asyncio
async def test_fault_during_admitted_claim_does_not_execute_or_claim_another_job():
    async with supervised() as (repo, managed, supervisor, client, worker, _):
        block = BlockOnce()
        async def after_claim(response):
            if response.status_code == 200:
                await block(response)
        client.after["claim"] = after_claim
        first, second = submit(repo, 2)
        try:
            await asyncio.wait_for(block.entered.wait(), 1)
            managed.value = RuntimeHealth(state=RuntimeHealthState.FAILED, ready=False)
            await asyncio.wait_for(supervisor.failed.wait(), 1)
            block.release.set()
            await eventually(lambda: worker.active_job_id is None)
            await asyncio.sleep(0.05)
            assert not managed.executor().executions
            assert repo.get_job(first.job_id).attempts == 1  # already admitted, not undone
            assert repo.get_job(second.job_id).attempts == 0
            assert client.count("complete") == client.count("fail") == 0
        finally:
            block.release.set()


@pytest.mark.asyncio
async def test_fault_before_terminal_write_and_pending_drain_never_reuse_terminal_lease():
    managed = Managed(Executor(gate=asyncio.Event()))
    async with supervised(managed) as (repo, _, supervisor, client, worker, _):
        block = BlockOnce()
        client.after["heartbeat"] = block
        job = submit(repo)[0]
        await asyncio.wait_for(managed.executor().started.wait(), 1)
        await asyncio.wait_for(block.entered.wait(), 1)
        managed.executor().gate.set()
        await asyncio.wait_for(managed.executor().finished.wait(), 1)
        managed.value = RuntimeHealth(state=RuntimeHealthState.FAILED, ready=False)
        await asyncio.wait_for(supervisor.failed.wait(), 1)
        block.release.set()
        await eventually(lambda: worker.active_job_id is None)
        await worker.drain()
        assert client.count("complete") == client.count("fail") == 0
        assert client.max_in_flight == 1
        assert repo.get_job(job.job_id).status is JobStatus.RUNNING


@pytest.mark.asyncio
async def test_already_committed_terminal_result_is_not_undone_by_later_runtime_fault():
    async with supervised() as (repo, managed, supervisor, client, worker, _):
        block = BlockOnce()
        client.after["complete"] = block
        first, second = submit(repo, 2)
        await asyncio.wait_for(block.entered.wait(), 1)
        assert repo.get_job(first.job_id).status is JobStatus.SUCCEEDED
        managed.value = RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False)
        await asyncio.wait_for(supervisor.failed.wait(), 1)
        block.release.set()
        await eventually(lambda: worker.active_job_id is None)
        await asyncio.sleep(0.05)
        assert repo.get_job(first.job_id).status is JobStatus.SUCCEEDED
        assert repo.get_job(second.job_id).attempts == 0
        assert client.count("complete") == 1 and client.count("fail") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("dead", [False, True])
async def test_executor_error_checks_health_before_considering_more_work(dead):
    managed = Managed()
    async def error(_request):
        if dead:
            managed.value = RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False)
        raise ValueError("test invalid request")
    managed._executor.execute = error
    async with supervised(managed, interval=0.5) as (repo, _, supervisor, client, worker, _):
        jobs = submit(repo, 2)
        if dead:
            await asyncio.wait_for(supervisor.failed.wait(), 1)
            assert client.count("fail") == 0
            assert repo.get_job(jobs[1].job_id).attempts == 0
            assert not worker.ready
        else:
            await eventually(lambda: all(repo.get_job(j.job_id).status is JobStatus.FAILED for j in jobs))
            assert worker.ready and not supervisor.failed.is_set()
        assert managed.max_in_flight == 1


@pytest.mark.asyncio
async def test_supervisor_serializes_forced_and_periodic_probes():
    managed = Managed()
    supervisor = RuntimeHealthSupervisor(managed, interval_seconds=0.01, timeout_seconds=0.2)
    await supervisor.start()
    try:
        managed.block = asyncio.Event()
        checks = [asyncio.create_task(supervisor.check()) for _ in range(3)]
        try:
            await asyncio.sleep(0.02)
            assert managed.max_in_flight == 1
            managed.block.set()
            assert all(await asyncio.gather(*checks))
        finally:
            managed.block.set()
            await asyncio.gather(*checks, return_exceptions=True)
    finally:
        await supervisor.close()
    assert managed.in_flight == 0


@pytest.mark.asyncio
async def test_expired_health_cannot_be_refreshed_by_a_delayed_success():
    managed = Managed()
    supervisor = RuntimeHealthSupervisor(managed, interval_seconds=0.02, timeout_seconds=0.2)
    assert await supervisor.check()
    managed.block = asyncio.Event()
    pending = asyncio.create_task(supervisor.check())
    try:
        await eventually(lambda: managed.in_flight == 1)
        supervisor._valid_until = time.monotonic() - 1
        assert not supervisor.available
        managed.block.set()
        assert not await pending
        assert supervisor.failed.is_set() and supervisor.failure_reason == "health_stale"
    finally:
        managed.block.set()
        await pending
        await supervisor.close()


@pytest.mark.asyncio
async def test_unexpected_monitor_cancellation_fails_closed():
    supervisor = RuntimeHealthSupervisor(Managed(), interval_seconds=0.01, timeout_seconds=0.1)
    await supervisor.start()
    supervisor._task.cancel()
    await asyncio.gather(supervisor._task, return_exceptions=True)
    assert supervisor.failed.is_set() and not supervisor.available
    await supervisor.close()


@pytest.mark.asyncio
async def test_normal_worker_drain_and_healthy_generation_preserve_supervision():
    async with supervised() as (repo, managed, supervisor, client, worker, _):
        job = submit(repo)[0]
        await eventually(lambda: repo.get_job(job.job_id).status is JobStatus.SUCCEEDED)
        await worker.drain()
        probes = managed.probes
        await asyncio.sleep(0.08)
        assert not worker.ready and worker.runtime_available
        assert not supervisor.failed.is_set()
        assert managed.probes > probes
        assert client.count("heartbeat") > 1
    assert supervisor.state == "stopped" and managed.in_flight == 0


@pytest.mark.asyncio
async def test_new_invocation_requires_runtime_health_and_gpu_ownership_again():
    repo = InMemoryControlRepository()
    managed = Managed()
    async with supervised(managed, repo) as (_, _, supervisor, _, worker, _):
        managed.value = RuntimeHealth(state=RuntimeHealthState.FAILED, ready=False)
        await asyncio.wait_for(supervisor.failed.wait(), 1)
        assert not worker.ready
    # Ownership conflict is not bypassed by a restarted Worker.
    competitor = replace(SPEC, worker_id="replacement-owner")
    repo.register_worker(WorkerRegistration(spec=competitor))
    managed = Managed()
    with pytest.raises(Exception) as raised:
        async with supervised(managed, repo):
            pass
    from astrumweaver.worker import ControlTransportError
    assert isinstance(raised.value, ControlTransportError) and raised.value.status_code == 409
    repo.set_worker_state(competitor.worker_id, WorkerState.OFFLINE)
    managed = Managed()
    async with supervised(managed, repo) as (_, _, _, _, worker, _):
        job = submit(repo)[0]
        await eventually(lambda: repo.get_job(job.job_id).status is JobStatus.SUCCEEDED)
        assert worker.ready


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_supervision_settings_rejected(value):
    for option in ("interval_seconds", "timeout_seconds"):
        with pytest.raises(ValueError):
            RuntimeHealthSupervisor(Managed(), **{option: value})


@pytest.mark.asyncio
async def test_shutdown_targets_owned_process_without_health_rpc():
    managed = Managed()
    managed.block = asyncio.Event()
    lifecycle = RuntimeLifecycleManager(managed)
    await lifecycle.shutdown_owned(timeout_seconds=0.1)
    await lifecycle.shutdown_owned(timeout_seconds=0.1)
    assert managed.stops == managed.releases == 1
    assert managed.probes == 0


@pytest.mark.asyncio
async def test_owned_shutdown_budget_and_error_redaction():
    managed = Managed()
    async def hang():
        await asyncio.Event().wait()
    managed.stop = hang
    with pytest.raises(RuntimeLifecycleError, match="owned runtime shutdown failed"):
        await asyncio.wait_for(RuntimeLifecycleManager(managed).shutdown_owned(timeout_seconds=0.02), 0.5)
    async def error():
        raise RuntimeError("private-sentinel")
    managed.stop = error
    with pytest.raises(RuntimeLifecycleError) as raised:
        await RuntimeLifecycleManager(managed).shutdown_owned(timeout_seconds=0.1)
    assert "private-sentinel" not in str(raised.value)


PROVIDERS = [
    ("ollama", "OllamaProvider"), ("llama_cpp", "LlamaCppProvider"),
    ("vllm", "VllmProvider"), ("freetoken", "FreeTokenProvider"),
    ("exllamav3", "ExLlamaV3Provider"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,provider_name", PROVIDERS)
async def test_all_providers_reject_owned_process_exit_during_inventory(name, provider_name):
    fixtures = importlib.import_module(f"test_{name}_provider")
    api = fixtures.FakeApi(reachable=False)
    process = fixtures.FakeProcess(api)
    provider = getattr(fixtures, provider_name)(
        api_factory=lambda *_args, **_kwargs: api,
        process_factory=lambda **_kwargs: process,
    )
    ctx = fixtures.context()
    runtime = provider.create_runtime(ctx, provider.setup_intent(ctx))
    await runtime.start()
    try:
        assert (await runtime.health()).ready
        # Ollama checks its list once for reachability and once for inventory.
        method = "list_models" if name == "ollama" else "models"
        original = getattr(api, method)
        calls = 0
        async def dies_during_inventory():
            nonlocal calls
            calls += 1
            models = await original()
            if name != "ollama" or calls == 2:
                process._running = False
            return models
        setattr(api, method, dies_during_inventory)
        health = await runtime.health()
        assert not health.ready and health.state is RuntimeHealthState.STOPPED
    finally:
        await runtime.stop()
        await runtime.release()


@pytest.mark.asyncio
async def test_real_owned_child_exit_reaches_worker_readiness_gate():
    """Real OS child lifetime, not a GPU or model-performance acceptance."""
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(30)")
    managed = Managed()
    original = managed.health
    async def child_health():
        if process.returncode is not None:
            return RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False)
        return await original()
    managed.health = child_health
    try:
        async with supervised(managed) as (repo, _, supervisor, client, worker, _):
            process.terminate()
            await asyncio.wait_for(process.wait(), 1)
            await asyncio.wait_for(supervisor.failed.wait(), 1)
            await eventually(lambda: client.in_flight == 0 and worker.active_job_id is None)
            count = client.count("claim")
            jobs = submit(repo, 2)
            await asyncio.sleep(0.06)
            assert not worker.ready and client.count("claim") == count
            assert all(repo.get_job(job.job_id).attempts == 0 for job in jobs)
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


@pytest.mark.asyncio
async def test_managed_daemon_quarantines_without_restart_and_shutdown_joins_tasks(monkeypatch):
    from astrumweaver.worker import daemon

    repo = InMemoryControlRepository()
    managed = Managed(Executor(gate=asyncio.Event()))
    handlers = {}
    clients = []
    config = {
        "worker": {"id": SPEC.worker_id, "class": "test", "control_url": "http://control", "gpu_preflight": False,
                   "capabilities": list(SPEC.capabilities), "heartbeat_interval_seconds": 0.02,
                   "poll_interval_seconds": 0.005, "shutdown_grace_seconds": 0},
        "runtime": {"manifest": "test-manifest", "shutdown_timeout_seconds": 0.2},
    }
    health_apps = []
    class HealthServer:
        should_exit = False
        def __init__(self, config):
            health_apps.append(config)
        async def serve(self):
            while not self.should_exit:
                await asyncio.sleep(0.005)
    def make_client(*_args, **_kwargs):
        client = ObservedClient(repo)
        clients.append(client)
        return client

    monkeypatch.setenv("ASTRUMWEAVER_WORKER_TOKEN", "test-worker-only")
    monkeypatch.setattr(daemon, "_load_toml", lambda _: config)
    monkeypatch.setattr(daemon, "_build_spec", lambda _: SPEC)
    monkeypatch.setattr(daemon, "_load_runtime_deployment", lambda _: object())
    monkeypatch.setattr(daemon, "managed_runtime_from_deployment", lambda *_args, **_kwargs: managed)
    monkeypatch.setattr(daemon, "discover_runtime_host_facts", lambda: object())
    monkeypatch.setattr(daemon, "RuntimeHealthSupervisor", lambda r, **kwargs: RuntimeHealthSupervisor(r, interval_seconds=0.01, timeout_seconds=0.1, **kwargs))
    monkeypatch.setattr(daemon, "ControlClient", make_client)
    monkeypatch.setattr(daemon.uvicorn, "Config", lambda app, **_kwargs: app)
    monkeypatch.setattr(daemon.uvicorn, "Server", HealthServer)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda name, fn: handlers.__setitem__(name, fn))
    first, second = submit(repo, 2)
    task = asyncio.create_task(daemon.run_worker("unused"))
    try:
        await asyncio.wait_for(managed.executor().started.wait(), 1)
        managed.value = RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False)
        await asyncio.wait_for(managed.executor().finished.wait(), 1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=health_apps[0]), base_url="http://worker") as probe:
            assert (await probe.get("/ready")).status_code == 503
            assert (await probe.get("/health")).json()["runtime_state"] == "failed"
        await asyncio.sleep(0.05)
        assert not task.done()
        assert repo.get_job(second.job_id).attempts == 0
        assert clients[0].count("register") == 1
        # Even a subsequently hung diagnostic endpoint cannot block stop().
        managed.block = asyncio.Event()
        handlers[daemon.signal.SIGTERM]()
        await asyncio.wait_for(task, 1)
        assert managed.executor().cancelled == [first.job_id]
        assert managed.stops == managed.releases == 1
        assert clients[0]._client.is_closed
        assert repo.get_worker(SPEC.worker_id).state is WorkerState.OFFLINE
        assert not any(t.get_name().startswith("astrumweaver-") for t in asyncio.all_tasks())
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_ollama_release_does_not_unload_model_on_foreign_endpoint():
    from test_ollama_provider import FakeApi, FakeProcess, context, OllamaManagedRuntime
    api = FakeApi(reachable=True)
    runtime = OllamaManagedRuntime(api=api, process=FakeProcess(api, running=False),
                                   context=context(), model=api.model, keep_alive="5m", startup_timeout_seconds=1)
    await runtime.release()
    assert api.closed and api.unloaded == []


@pytest.mark.asyncio
async def test_shutdown_keeps_control_reservation_until_owned_runtime_stops():
    async with supervised() as (repo, managed, _, _, worker, task):
        entered = asyncio.Event()
        release = asyncio.Event()
        original = managed.stop
        async def blocked_stop():
            entered.set()
            await release.wait()
            await original()
        managed.stop = blocked_stop
        worker.request_stop()
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert not worker.ready
            assert repo.get_worker(SPEC.worker_id).state is not WorkerState.OFFLINE
            with pytest.raises(ConflictError):
                repo.register_worker(WorkerRegistration(spec=replace(SPEC, worker_id="competitor")))
        finally:
            release.set()
        await asyncio.wait_for(task, 1)
        assert repo.get_worker(SPEC.worker_id).state is WorkerState.OFFLINE
        assert managed.stops == managed.releases == 1


@pytest.mark.asyncio
async def test_failed_owned_cleanup_does_not_report_control_offline():
    repo = InMemoryControlRepository()
    managed = Managed()
    async def broken_stop():
        raise RuntimeError("test stop failed")
    managed.stop = broken_stop
    with pytest.raises(RuntimeLifecycleError):
        async with supervised(managed, repo) as (_, _, _, _, worker, _):
            assert worker.ready
    assert repo.get_worker(SPEC.worker_id).state is not WorkerState.OFFLINE
