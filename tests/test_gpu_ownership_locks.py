"""Lock ordering and reference-repository ownership contracts."""
import pytest

from astrumweaver import ResourceShape, WorkerSpec
from astrumweaver.control import (
    ConflictError, InMemoryControlRepository, WorkerHeartbeat,
    WorkerRegistration, WorkerState,
)
from astrumweaver.control import postgres


class Connection:
    def __init__(self):
        self.calls = []
        self.row = {"gpu_uuids": ["GPU-old"]}

    def execute(self, sql, args=()):
        self.calls.append((sql, args))
        return self

    def fetchone(self):
        return self.row


def test_ownership_locks_precede_row_lock_and_order_old_and_new_gpu_keys():
    repo = postgres.PostgresControlRepository("unused")
    first, second = Connection(), Connection()
    repo._lock_worker_ownership(first, "worker", ("GPU-b", "GPU-a"))
    repo._lock_worker_ownership(second, "worker", ("GPU-a", "GPU-b"))
    assert first.calls == second.calls
    advisory = [(sql, args) for sql, args in first.calls if "pg_advisory" in sql]
    assert advisory[0][1][0] == postgres._WORKER_OWNERSHIP_LOCK
    keys = sorted({postgres._ownership_lock_key(g) for g in ("GPU-old", "GPU-a", "GPU-b")})
    assert [args for _, args in advisory[1:]] == [
        (postgres._GPU_OWNERSHIP_LOCK, key) for key in keys
    ]
    assert "FOR UPDATE" in first.calls[-1][0]
    assert not any("FOR UPDATE" in sql for sql, _ in first.calls[:-1])
    assert postgres._WORKER_OWNERSHIP_LOCK != postgres._GPU_OWNERSHIP_LOCK


def test_digest_collisions_only_collapse_locks_within_their_namespace(monkeypatch):
    monkeypatch.setattr(postgres, "_ownership_lock_key", lambda _: 7)
    connection = Connection()
    postgres.PostgresControlRepository("unused")._lock_worker_ownership(
        connection, "worker", ("GPU-a", "GPU-b")
    )
    locks = [args for sql, args in connection.calls if "pg_advisory" in sql]
    assert locks == [(postgres._WORKER_OWNERSHIP_LOCK, 7), (postgres._GPU_OWNERSHIP_LOCK, 7)]


def registration(name):
    return WorkerRegistration(spec=WorkerSpec(
        worker_id=name, worker_class="test", gpu_uuids=("GPU-shared",),
        resources=ResourceShape(gpu_count=1, total_vram_mb=1024, max_single_gpu_vram_mb=1024),
        capabilities=frozenset({"debug.echo"}),
    ))


@pytest.mark.parametrize("target", [WorkerState.ONLINE, WorkerState.DRAINING])
def test_reference_repository_reacquisition_and_heartbeat_cannot_bypass_ownership(target):
    repo = InMemoryControlRepository()
    repo.register_worker(registration("old"))
    repo.set_worker_state("old", WorkerState.OFFLINE)
    repo.register_worker(registration("new"))
    with pytest.raises(ConflictError, match="overlaps"):
        repo.set_worker_state("old", target)
    with pytest.raises(ConflictError, match="explicitly reacquire"):
        repo.heartbeat_worker("old", WorkerHeartbeat(state=target))
    assert repo.heartbeat_worker("old").state is WorkerState.OFFLINE
    repo.set_worker_state("new", WorkerState.OFFLINE)
    assert repo.set_worker_state("old", target).state is target
