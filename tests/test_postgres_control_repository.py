from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

import httpx
import pytest

psycopg = pytest.importorskip("psycopg")

from astrumweaver import JobRequirements, JobResult, ResourceShape, WorkerSpec
from astrumweaver.control.api import create_app
from astrumweaver.control.migrate import apply_migrations
from astrumweaver.transport import PROTOCOL_VERSION

from astrumweaver.control import (
    ConflictError,
    JobStatus,
    JobSubmission,
    PostgresControlRepository,
    StorageUnavailable,
    WorkerRegistration,
    WorkerHeartbeat,
    WorkerState,
    utc_now,
)


DATABASE_URL = os.environ.get("ASTRUMWEAVER_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="ASTRUMWEAVER_TEST_DATABASE_URL is not configured",
)


@pytest.fixture(autouse=True)
def reset_database() -> None:
    assert DATABASE_URL is not None
    apply_migrations(DATABASE_URL)
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("TRUNCATE TABLE jobs, workers RESTART IDENTITY CASCADE")


def worker(
    worker_id: str,
    *,
    single_vram_mb: int,
    gpu_count: int = 1,
    total_vram_mb: int | None = None,
) -> WorkerRegistration:
    if total_vram_mb is None:
        total_vram_mb = single_vram_mb * gpu_count
    return WorkerRegistration(
        spec=WorkerSpec(
            worker_id=worker_id,
            worker_class="multi-gpu" if gpu_count > 1 else "modern-single",
            resources=ResourceShape(
                gpu_count=gpu_count,
                total_vram_mb=total_vram_mb,
                max_single_gpu_vram_mb=single_vram_mb,
            ),
            gpu_uuids=tuple(f"GPU-{worker_id}-{index}" for index in range(gpu_count)),
            capabilities=frozenset({"llm.chat", "code.review"}),
            labels={"runtime_family": "modern"},
        )
    )


def test_postgres_priority_fifo_and_resource_matching() -> None:
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL)
    registered = repo.register_worker(
        worker(
            "multi-worker",
            single_vram_mb=12_288,
            gpu_count=2,
            total_vram_mb=24_576,
        ),
        now=now,
    )

    blocked = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            priority=100,
            payload={"name": "needs-one-large-gpu"},
            requirements=JobRequirements(min_single_gpu_vram_mb=20_000),
        ),
        now=now,
    )
    first = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            priority=10,
            payload={"name": "first"},
            requirements=JobRequirements(
                min_gpu_count=2,
                min_total_vram_mb=20_000,
            ),
        ),
        now=now,
    )
    second = repo.submit_job(
        JobSubmission(
            capability="llm.chat",
            priority=10,
            payload={"name": "second"},
            requirements=JobRequirements(
                min_gpu_count=2,
                min_total_vram_mb=20_000,
            ),
        ),
        now=now,
    )

    claimed = repo.claim_next_job(registered.worker_id, now=now)

    assert claimed is not None
    assert claimed.job_id == first.job_id
    assert first.sequence < second.sequence
    assert repo.get_job(blocked.job_id).status is JobStatus.QUEUED


def test_postgres_lease_recovery_fences_stale_attempt() -> None:
    assert DATABASE_URL is not None
    now = utc_now()
    repo = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    registered = repo.register_worker(
        worker("single-worker", single_vram_mb=24_576),
        now=now,
    )
    job = repo.submit_job(JobSubmission(capability="llm.chat"), now=now)
    first = repo.claim_next_job(registered.worker_id, now=now)
    assert first is not None and first.lease_token

    expired = now + timedelta(seconds=2)
    with pytest.raises(ConflictError, match="lease has expired"):
        repo.complete_job(
            job.job_id,
            JobResult(outputs={"stale": True}),
            worker_id=registered.worker_id,
            lease_token=first.lease_token,
            now=expired,
        )

    recovered = repo.recover_expired_jobs(now=expired)
    assert [item.job_id for item in recovered] == [job.job_id]

    second = repo.claim_next_job(registered.worker_id, now=expired)
    assert second is not None and second.lease_token != first.lease_token

    with pytest.raises(ConflictError):
        repo.complete_job(
            job.job_id,
            JobResult(outputs={"stale": True}),
            worker_id=registered.worker_id,
            lease_token=first.lease_token,
            now=expired,
        )


def test_postgres_idempotency_is_durable() -> None:
    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL)
    submission = JobSubmission(
        capability="code.review",
        payload={"repository": "example/repository"},
        idempotency_key="durable-request-1",
    )

    first = repo.submit_job(submission)
    replay = repo.submit_job(submission)

    assert replay.job_id == first.job_id

    with pytest.raises(ConflictError, match="different job request"):
        repo.submit_job(
            JobSubmission(
                capability="code.review",
                payload={"repository": "different/repository"},
                idempotency_key="durable-request-1",
            )
        )


@pytest.mark.asyncio
async def test_postgres_transport_fencing_and_non_text_result() -> None:
    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL, lease_seconds=1)
    app = create_app(
        repo,
        client_token="client-secret",
        worker_token="worker-secret",
        maintenance_interval_seconds=60.0,
    )
    transport = httpx.ASGITransport(app=app)
    client_headers = {"authorization": "Bearer client-secret"}
    worker_headers = {"authorization": "Bearer worker-secret"}

    registration = {
        "protocol_version": PROTOCOL_VERSION,
        "spec": {
            "worker_id": "transport-worker",
            "worker_class": "cpu-test",
            "gpu_uuids": [],
            "capabilities": ["decision.system_one"],
            "labels": {},
            "resources": {
                "gpu_count": 0,
                "total_vram_mb": 0,
                "max_single_gpu_vram_mb": 0,
            },
        },
        "max_concurrency": 1,
        "metadata": {},
    }
    submission = {
        "protocol_version": PROTOCOL_VERSION,
        "capability": "decision.system_one",
        "payload": {"question": "route"},
        "requirements": {},
        "priority": 0,
        "max_attempts": 3,
    }

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        registered = await client.post(
            "/v1/workers/register",
            headers=worker_headers,
            json=registration,
        )
        assert registered.status_code == 201

        created = await client.post(
            "/v1/jobs",
            headers=client_headers,
            json=submission,
        )
        assert created.status_code == 201
        job_id = created.json()["job_id"]

        first = await client.post(
            "/v1/workers/transport-worker/jobs/claim",
            headers=worker_headers,
        )
        assert first.status_code == 200
        old_token = first.json()["lease_token"]

        await asyncio.sleep(1.05)
        expired = await client.post(
            f"/v1/workers/transport-worker/jobs/{job_id}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": old_token,
                "result": {"outputs": {"stale": True}},
            },
        )
        assert expired.status_code == 409

        repo.recover_expired_jobs()

        second = await client.post(
            "/v1/workers/transport-worker/jobs/claim",
            headers=worker_headers,
        )
        assert second.status_code == 200
        new_token = second.json()["lease_token"]
        assert new_token != old_token

        stale = await client.post(
            f"/v1/workers/transport-worker/jobs/{job_id}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": old_token,
                "result": {"outputs": {"stale": True}},
            },
        )
        assert stale.status_code == 409

        completed = await client.post(
            f"/v1/workers/transport-worker/jobs/{job_id}/complete",
            headers=worker_headers,
            json={
                "protocol_version": PROTOCOL_VERSION,
                "lease_token": new_token,
                "result": {
                    "outputs": {
                        "choice": "local",
                        "probabilities": {"local": 0.8, "remote": 0.2},
                    },
                    "artifacts": [],
                    "metrics": {"entropy": 0.5},
                    "text": None,
                    "metadata": {"executor": "decision-test"},
                },
            },
        )
        assert completed.status_code == 200

        fetched = await client.get(
            f"/v1/jobs/{job_id}",
            headers=client_headers,
        )

    result = fetched.json()["result"]
    assert result["text"] is None
    assert result["outputs"]["choice"] == "local"
    assert result["outputs"]["probabilities"]["remote"] == 0.2


def test_check_storage_rejects_legacy_001_only_schema_until_migrated() -> None:
    assert DATABASE_URL is not None
    migration_001 = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "001_control_plane.sql"
    ).read_text(encoding="utf-8")

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("DROP TABLE IF EXISTS jobs CASCADE")
        connection.execute("DROP TABLE IF EXISTS workers CASCADE")
        connection.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
        connection.execute(migration_001)

    repo = PostgresControlRepository(DATABASE_URL)
    with pytest.raises(StorageUnavailable, match="schema"):
        repo.check_storage()

    applied = apply_migrations(DATABASE_URL)

    assert applied == [
        "000_schema_migrations.sql",
        "001_control_plane.sql",
        "002_worker_accelerators.sql",
        "003_serving_bindings.sql",
    ]
    repo.check_storage()

    with psycopg.connect(DATABASE_URL) as connection:
        columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'workers'
                """
            ).fetchall()
        }
        recorded = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM schema_migrations"
            ).fetchall()
        }

    assert "accelerators" in columns
    assert set(applied) <= recorded


@pytest.mark.asyncio
async def test_ready_rejects_legacy_001_only_schema_until_all_migrations_apply() -> None:
    assert DATABASE_URL is not None
    migration_001 = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "001_control_plane.sql"
    ).read_text(encoding="utf-8")

    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("DROP TABLE IF EXISTS jobs CASCADE")
        connection.execute("DROP TABLE IF EXISTS workers CASCADE")
        connection.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
        connection.execute(migration_001)

    repo = PostgresControlRepository(DATABASE_URL)
    app = create_app(
        repo,
        client_token="client-secret",
        worker_token="worker-secret",
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://control",
    ) as client:
        before = await client.get("/v1/ready")
        assert before.status_code == 503
        assert before.json() == {"detail": "storage unavailable"}

        apply_migrations(DATABASE_URL)

        after = await client.get("/v1/ready")
        assert after.status_code == 200
        assert after.json()["ready"] is True


def test_packaged_migration_entrypoint_is_idempotent() -> None:
    assert DATABASE_URL is not None
    applied = apply_migrations(DATABASE_URL)

    assert "000_schema_migrations.sql" in applied
    assert "001_control_plane.sql" in applied
    assert "002_worker_accelerators.sql" in applied
    assert "003_serving_bindings.sql" in applied

    repo = PostgresControlRepository(DATABASE_URL)
    repo.check_storage()


# F2 regressions use real connections and a controlled pause *after* the first
# ownership predicate, before INSERT/UPDATE. The competitor must wait on the
# ownership advisory lock, not slip through the absent-row gap. No GPU needed.



def shared_worker(name, gpus=("GPU-shared",)):
    original = worker(name, single_vram_mb=1024, gpu_count=len(gpus))
    return replace(original, spec=replace(original.spec, gpu_uuids=gpus))


def bounded_repo(cls=PostgresControlRepository, **kwargs):
    name = "aw-ownership-test-" + uuid4().hex
    dsn = psycopg.conninfo.make_conninfo(
        DATABASE_URL, application_name=name, connect_timeout=5,
        options="-c lock_timeout=4000 -c statement_timeout=8000",
    )
    return cls(dsn, **kwargs), name


class PausedOwnershipRepository(PostgresControlRepository):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked = Event()
        self.release = Event()

    def _gpu_overlap(self, connection, registration):
        result = super()._gpu_overlap(connection, registration)
        self.checked.set()
        if not self.release.wait(8):
            raise AssertionError("test ownership pause timed out")
        return result


def wait_for_advisory_waiter(application_name, future):
    deadline = monotonic() + 3
    with psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=5) as observer:
        observer.execute("SET statement_timeout = '2s'")
        while monotonic() < deadline:
            if future.done():
                pytest.fail("competing ownership mutation did not wait for the held lock")
            row = observer.execute(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE application_name = %s AND wait_event_type = 'Lock' "
                "AND wait_event = 'advisory'", (application_name,),
            ).fetchone()
            if row[0]:
                return
            sleep(0.01)
    pytest.fail("competitor did not reach the database advisory lock")


def run_ownership_race(first, first_call, second, second_call, *, conflict=True):
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(first_call)
        try:
            assert first.checked.wait(3), "first request did not reach its overlap check"
            competitor = pool.submit(second_call)
            wait_for_advisory_waiter(second, competitor)
        finally:
            first.release.set()
        winner.result(timeout=10)
        if conflict:
            with pytest.raises(ConflictError, match="overlaps"):
                competitor.result(timeout=10)
        else:
            competitor.result(timeout=10)


@pytest.mark.parametrize("a,b", [
    (("GPU-shared",), ("GPU-shared",)),
    (("GPU-a", "GPU-shared"), ("GPU-shared", "GPU-c")),
    (("GPU-a", "GPU-b"), ("GPU-b", "GPU-a")),
])
def test_postgres_concurrent_new_workers_cannot_reserve_overlapping_gpus(a, b):
    first, _ = bounded_repo(PausedOwnershipRepository)
    second, name = bounded_repo()
    run_ownership_race(
        first, lambda: first.register_worker(shared_worker("first", a)),
        name, lambda: second.register_worker(shared_worker("second", b)),
    )
    with psycopg.connect(DATABASE_URL) as connection:
        assert connection.execute("SELECT count(*) FROM workers").fetchone()[0] == 1
    # The rejected transaction released *all* locks; a different GPU can enroll.
    assert second.register_worker(shared_worker("second", ("GPU-other",))).state is WorkerState.ONLINE


def test_postgres_disjoint_gpu_registration_is_not_globally_serialized():
    first, _ = bounded_repo(PausedOwnershipRepository)
    second, _ = bounded_repo()
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first.register_worker, shared_worker("first", ("GPU-a",)))
        try:
            assert first.checked.wait(3)
            two = pool.submit(second.register_worker, shared_worker("second", ("GPU-b",)))
            assert two.result(timeout=3).state is WorkerState.ONLINE
        finally:
            first.release.set()
        one.result(timeout=10)


def test_postgres_same_new_worker_identity_serializes_topology_replacement():
    first, _ = bounded_repo(PausedOwnershipRepository)
    second, name = bounded_repo()
    run_ownership_race(
        first, lambda: first.register_worker(shared_worker("same", ("GPU-a",))),
        name, lambda: second.register_worker(shared_worker("same", ("GPU-b",))),
        conflict=False,
    )
    assert second.get_worker("same").spec.gpu_uuids == ("GPU-b",)
    assert second.register_worker(shared_worker("reuse", ("GPU-a",))).state is WorkerState.ONLINE


@pytest.mark.parametrize("target", [WorkerState.ONLINE, WorkerState.DRAINING])
@pytest.mark.parametrize("first_reacquires", [False, True])
def test_postgres_reacquisition_competes_with_registration(target, first_reacquires):
    base, _ = bounded_repo()
    base.register_worker(shared_worker("old"))
    base.set_worker_state("old", WorkerState.OFFLINE)
    first, _ = bounded_repo(PausedOwnershipRepository)
    second, name = bounded_repo()
    if first_reacquires:
        first_call = lambda: first.set_worker_state("old", target)
        second_call = lambda: second.register_worker(shared_worker("new"))
    else:
        first_call = lambda: first.register_worker(shared_worker("new"))
        second_call = lambda: second.set_worker_state("old", target)
    run_ownership_race(first, first_call, name, second_call)
    if not first_reacquires:
        assert base.get_worker("old").state is WorkerState.OFFLINE


@pytest.mark.parametrize("target", [WorkerState.ONLINE, WorkerState.DRAINING])
def test_postgres_offline_heartbeat_cannot_reacquire_gpus(target):
    repo, _ = bounded_repo()
    repo.register_worker(shared_worker("old"))
    repo.set_worker_state("old", WorkerState.OFFLINE)
    repo.register_worker(shared_worker("new"))
    with pytest.raises(ConflictError, match="explicitly reacquire"):
        repo.heartbeat_worker("old", WorkerHeartbeat(state=target))
    assert repo.get_worker("old").state is WorkerState.OFFLINE
    assert repo.heartbeat_worker("old").state is WorkerState.OFFLINE


@pytest.mark.parametrize("release", ["complete", "fail", "cancel", "recover"])
@pytest.mark.parametrize("offline_reason", ["state", "ttl"])
def test_postgres_offline_busy_worker_keeps_gpus_until_fenced_release(release, offline_reason):
    repo, _ = bounded_repo(lease_seconds=300, worker_ttl_seconds=1)
    now = utc_now()
    repo.register_worker(shared_worker("old"), now=now)
    job = repo.submit_job(JobSubmission(capability="llm.chat"), now=now)
    claim = repo.claim_next_job("old", now=now)
    assert claim is not None
    if offline_reason == "ttl":
        now += timedelta(seconds=2)
        assert repo.expire_stale_workers(now=now)
    else:
        repo.set_worker_state("old", WorkerState.OFFLINE, now=now)
    with pytest.raises(ConflictError, match="overlaps"):
        repo.register_worker(shared_worker("new"), now=now)
    with pytest.raises(ConflictError, match="topology"):
        repo.register_worker(shared_worker("old", ("GPU-other",)), now=now)
    if release == "complete":
        repo.complete_job(job.job_id, JobResult(text="ok"), worker_id="old", lease_token=claim.lease_token, now=now)
    elif release == "fail":
        repo.fail_job(job.job_id, "test", worker_id="old", lease_token=claim.lease_token, retryable=False, now=now)
    elif release == "cancel":
        repo.cancel_job(job.job_id, now=now)
    else:
        repo.recover_expired_jobs(now=now + timedelta(seconds=301))
    assert repo.get_worker("old").active_jobs == 0
    assert repo.register_worker(shared_worker("new"), now=now).state is WorkerState.ONLINE
    with pytest.raises(ConflictError):
        repo.complete_job(job.job_id, JobResult(text="stale"), worker_id="old", lease_token=claim.lease_token, now=now)


def test_postgres_enrollment_pins_read_committed_even_if_database_default_differs():
    dsn = psycopg.conninfo.make_conninfo(
        DATABASE_URL, options="-c default_transaction_isolation=repeatable\\ read",
    )
    repo = PostgresControlRepository(dsn)
    with repo._transaction() as connection:
        assert connection.execute("SHOW transaction_isolation").fetchone()["transaction_isolation"] == "read committed"


def test_postgres_ownership_race_harness_detects_missing_serialization():
    # Negative control: deliberately remove ONLY the new acquisition protocol
    # in this test double. Both absent-owner predicates can then pass; the
    # controlled schedule reproduces F2 without a production escape hatch.
    class Unserialized(PausedOwnershipRepository):
        def _lock_worker_ownership(self, connection, worker_id, requested_gpu_uuids=()):
            return connection.execute(
                "SELECT * FROM workers WHERE id = %s FOR UPDATE", (worker_id,)
            ).fetchone()

    first, _ = bounded_repo(Unserialized)
    second, _ = bounded_repo(Unserialized)
    second.release.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first.register_worker, shared_worker("first"))
        try:
            assert first.checked.wait(3)
            two = pool.submit(second.register_worker, shared_worker("second"))
            assert two.result(timeout=3).state is WorkerState.ONLINE
        finally:
            first.release.set()
        assert one.result(timeout=10).state is WorkerState.ONLINE
    with psycopg.connect(DATABASE_URL) as connection:
        assert connection.execute("SELECT count(*) FROM workers WHERE state = 'online'").fetchone()[0] == 2


@pytest.mark.asyncio
async def test_postgres_worker_heartbeat_survives_continuous_short_job_backlog():
    """Exercise real Worker -> HTTP API -> PostgreSQL for more than two TTLs."""
    from astrumweaver.worker import ControlClient, WorkerRuntime
    from astrumweaver import ResidencyReport

    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL, worker_ttl_seconds=1, lease_seconds=1)
    spec = worker("heartbeat-backlog", single_vram_mb=1).spec
    job_ids = [
        repo.submit_job(JobSubmission(capability="llm.chat", payload={}, max_attempts=1)).job_id
        for _ in range(400)
    ]

    class ShortExecutor:
        executed = 0

        async def execute(self, request):
            self.executed += 1
            await asyncio.sleep(0.01)
            return JobResult(text="ok")

        async def cancel(self, job_id):
            pass

        async def residency(self):
            return ResidencyReport()

    app = create_app(repo, client_token=None, worker_token="test-worker", client_auth="none")
    async with ControlClient(
        "http://control", "test-worker", transport=httpx.ASGITransport(app=app), timeout_seconds=0.5
    ) as client:
        executor = ShortExecutor()
        runtime = WorkerRuntime(
            spec=spec, max_concurrency=1, executor=executor, client=client,
            heartbeat_interval_seconds=0.04, poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runtime.run_forever())
        try:
            async with asyncio.timeout(3):
                while not runtime.registered:
                    if task.done():
                        await task
                    await asyncio.sleep(0.005)
            start = monotonic()
            ages = []
            while monotonic() - start < 2.2:
                assert not task.done()
                expired = await asyncio.to_thread(repo.expire_stale_workers)
                assert not expired
                current = await asyncio.to_thread(repo.get_worker, spec.worker_id)
                assert current.state is WorkerState.ONLINE
                assert runtime.ready
                ages.append((utc_now() - current.last_seen_at).total_seconds())
                await asyncio.sleep(0.02)
            assert executor.executed > 5
            assert max(ages) < 0.8
            assert (await asyncio.to_thread(repo.get_job, job_ids[-1])).status is JobStatus.QUEUED, "backlog exhausted"
        finally:
            runtime.request_stop()
            await asyncio.wait_for(task, 3)
        assert repo.get_worker(spec.worker_id).state is WorkerState.OFFLINE


@pytest.mark.asyncio
async def test_postgres_runtime_observes_offline_without_implicit_gpu_reacquisition():
    from astrumweaver.worker import ControlClient, WorkerRuntime
    from astrumweaver.executors.structured_echo import StructuredEchoExecutor

    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL)
    spec = replace(worker("offline-liveness", single_vram_mb=1).spec, capabilities=frozenset({"debug.echo"}))
    app = create_app(repo, client_token=None, worker_token="test-worker", client_auth="none")
    async with ControlClient("http://control", "test-worker", transport=httpx.ASGITransport(app=app)) as client:
        runtime = WorkerRuntime(
            spec=spec, max_concurrency=1, executor=StructuredEchoExecutor(), client=client,
            heartbeat_interval_seconds=0.02, poll_interval_seconds=0.01,
        )
        task = asyncio.create_task(runtime.run_forever())
        try:
            async with asyncio.timeout(3):
                while not runtime.registered:
                    await asyncio.sleep(0.005)
            await asyncio.to_thread(repo.set_worker_state, spec.worker_id, WorkerState.OFFLINE)
            async with asyncio.timeout(3):
                while runtime.control_state != "offline":
                    await asyncio.sleep(0.005)
            competitor = replace(spec, worker_id="replacement-owner")
            await asyncio.to_thread(repo.register_worker, WorkerRegistration(spec=competitor))
            with pytest.raises(ConflictError):
                await asyncio.to_thread(repo.set_worker_state, spec.worker_id, WorkerState.ONLINE)
            await runtime.drain()
            await asyncio.sleep(0.08)
            assert not runtime.ready
            assert repo.get_worker(spec.worker_id).state is WorkerState.OFFLINE
            assert repo.get_worker(competitor.worker_id).state is WorkerState.ONLINE
        finally:
            runtime.request_stop()
            await asyncio.wait_for(task, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True])
async def test_postgres_post_start_runtime_failure_quarantines_worker(active):
    """Real Worker/API/PostgreSQL; provider health is a bounded test double."""
    from test_runtime_supervision import Managed, supervised
    from test_worker_liveness import SPEC, Executor, eventually, submit
    from astrumweaver.runtime import RuntimeHealth, RuntimeHealthState

    assert DATABASE_URL is not None
    repo = PostgresControlRepository(DATABASE_URL)
    managed = Managed(Executor(gate=asyncio.Event()) if active else Executor())
    async with supervised(managed, repo, interval=0.05, timeout=0.8) as (_, _, supervisor, client, runtime, task):
        claimed = None
        if active:
            first = (await asyncio.to_thread(submit, repo))[0]
            await asyncio.wait_for(managed.executor().started.wait(), 2)
            claimed = await asyncio.to_thread(repo.get_job, first.job_id)
        managed.value = RuntimeHealth(state=RuntimeHealthState.STOPPED, ready=False)
        await asyncio.wait_for(supervisor.failed.wait(), 2)
        await eventually(lambda: runtime.active_job_id is None and client.in_flight == 0)
        count = client.count("claim")
        second = (await asyncio.to_thread(submit, repo))[0]
        await asyncio.sleep(0.15)
        assert not runtime.ready and not task.done()
        assert client.count("claim") == count
        assert (await asyncio.to_thread(repo.get_job, second.job_id)).attempts == 0
        assert client.count("register") == 1
        if claimed is not None:
            assert managed.executor().cancelled == [claimed.job_id]
            assert client.count("complete") == client.count("fail") == 0
            await asyncio.to_thread(repo.recover_expired_jobs, now=claimed.lease_expires_at + timedelta(seconds=1))
            with pytest.raises(ConflictError):
                await asyncio.to_thread(repo.complete_job, claimed.job_id, JobResult(text="stale"),
                                        worker_id=SPEC.worker_id, lease_token=claimed.lease_token)
