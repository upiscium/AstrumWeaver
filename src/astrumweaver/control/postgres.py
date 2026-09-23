"""PostgreSQL implementation of AstrumWeaver durable control state."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ..contracts import ResourceShape, WorkerSpec
from ..execution import JobResult
from ..scheduling import worker_matches
from .models import (
    JobRecord,
    JobStatus,
    JobSubmission,
    WorkerHeartbeat,
    WorkerRecord,
    WorkerRegistration,
    WorkerState,
    utc_now,
)
from .repository import (
    ConflictError,
    NotFoundError,
    RepositoryError,
    StorageUnavailable,
    _aware,
    _failure_payload,
    _submission_matches,
)
from .serde import (
    job_result_from_dict,
    job_result_to_dict,
    requirements_from_dict,
    requirements_to_dict,
)

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - optional dependency path
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    Jsonb = None  # type: ignore[assignment,misc]


def _json(value: Any) -> Any:
    return Jsonb(value) if Jsonb is not None else value


class PostgresControlRepository:
    """Transactional PostgreSQL control-plane repository."""

    def check_storage(self) -> None:
        with self._connection() as connection:
            connection.execute("SELECT 1").fetchone()


    def __init__(
        self,
        database_url: str,
        *,
        worker_ttl_seconds: int = 60,
        lease_seconds: int = 300,
    ) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if worker_ttl_seconds < 1:
            raise ValueError("worker_ttl_seconds must be positive")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        self.database_url = database_url
        self.worker_ttl_seconds = worker_ttl_seconds
        self.lease_seconds = lease_seconds

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        if psycopg is None:
            raise StorageUnavailable(
                "PostgreSQL support requires the astrumweaver postgres dependency"
            )
        try:
            connection = psycopg.connect(self.database_url, row_factory=dict_row)
        except Exception as exc:
            raise StorageUnavailable("postgresql is unavailable") from exc
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        with self._connection() as connection:
            try:
                with connection.transaction():
                    yield connection
            except RepositoryError:
                raise
            except Exception as exc:
                raise StorageUnavailable("postgresql transaction failed") from exc

    @staticmethod
    def _worker(row: dict[str, Any]) -> WorkerRecord:
        spec = WorkerSpec(
            worker_id=row["id"],
            worker_class=row["worker_class"],
            gpu_uuids=tuple(row.get("gpu_uuids") or ()),
            capabilities=frozenset(row.get("capabilities") or ()),
            labels=row.get("labels") or {},
            resources=ResourceShape(
                gpu_count=int(row.get("gpu_count") or 0),
                total_vram_mb=int(row.get("total_vram_mb") or 0),
                max_single_gpu_vram_mb=int(row.get("max_single_gpu_vram_mb") or 0),
            ),
        )
        return WorkerRecord(
            spec=spec,
            max_concurrency=int(row["max_concurrency"]),
            state=WorkerState(row["state"]),
            active_jobs=int(row.get("active_jobs") or 0),
            registered_at=row["registered_at"],
            last_seen_at=row["last_seen_at"],
            metadata=row.get("metadata") or {},
        )

    @staticmethod
    def _job(row: dict[str, Any]) -> JobRecord:
        return JobRecord(
            job_id=str(row["id"]),
            capability=row["capability"],
            payload=row.get("payload") or {},
            requirements=requirements_from_dict(row.get("requirements")),
            priority=int(row["priority"]),
            sequence=int(row["sequence"]),
            status=JobStatus(row["status"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            idempotency_key=row.get("idempotency_key"),
            assigned_worker_id=row.get("assigned_worker_id"),
            lease_token=row.get("lease_token"),
            result=job_result_from_dict(row.get("result")),
            error=row.get("error"),
            created_at=row["created_at"],
            available_at=row["available_at"],
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            lease_expires_at=row.get("lease_expires_at"),
            updated_at=row["updated_at"],
        )

    def _gpu_overlap(
        self,
        connection: Any,
        registration: WorkerRegistration,
    ) -> str | None:
        gpu_uuids = list(registration.spec.gpu_uuids)
        if not gpu_uuids:
            return None
        row = connection.execute(
            """
            SELECT id
            FROM workers
            WHERE id <> %s
              AND (state <> 'offline' OR active_jobs > 0)
              AND EXISTS (
                  SELECT 1
                  FROM jsonb_array_elements_text(%s::jsonb) AS requested(uuid)
                  WHERE gpu_uuids ? requested.uuid
              )
            ORDER BY id
            LIMIT 1
            FOR UPDATE
            """,
            (registration.spec.worker_id, _json(gpu_uuids)),
        ).fetchone()
        return None if row is None else str(row["id"])

    def register_worker(
        self, registration: WorkerRegistration, *, now: datetime | None = None
    ) -> WorkerRecord:
        timestamp = _aware(now)
        spec = registration.spec
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM workers WHERE id = %s FOR UPDATE",
                (spec.worker_id,),
            ).fetchone()
            if existing is not None:
                current = self._worker(existing)
                if current.active_jobs > 0:
                    if registration.spec != current.spec:
                        raise ConflictError(
                            "worker resource/topology cannot change while jobs are active"
                        )
                    if registration.max_concurrency < current.active_jobs:
                        raise ConflictError(
                            "max_concurrency cannot be lower than active job count"
                        )

            conflicting = self._gpu_overlap(connection, registration)
            if conflicting:
                raise ConflictError(
                    f"GPU identity overlaps with worker {conflicting}"
                )

            registered_at = existing["registered_at"] if existing else timestamp
            active_jobs = connection.execute(
                """
                SELECT count(*)::integer AS count
                FROM jobs
                WHERE status = 'running' AND assigned_worker_id = %s
                """,
                (spec.worker_id,),
            ).fetchone()["count"]

            row = connection.execute(
                """
                INSERT INTO workers (
                    id, worker_class, capabilities, labels, gpu_uuids,
                    gpu_count, total_vram_mb, max_single_gpu_vram_mb,
                    max_concurrency, state, metadata, registered_at,
                    last_seen_at, active_jobs, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, 'online', %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    worker_class = EXCLUDED.worker_class,
                    capabilities = EXCLUDED.capabilities,
                    labels = EXCLUDED.labels,
                    gpu_uuids = EXCLUDED.gpu_uuids,
                    gpu_count = EXCLUDED.gpu_count,
                    total_vram_mb = EXCLUDED.total_vram_mb,
                    max_single_gpu_vram_mb = EXCLUDED.max_single_gpu_vram_mb,
                    max_concurrency = EXCLUDED.max_concurrency,
                    state = 'online',
                    metadata = EXCLUDED.metadata,
                    last_seen_at = EXCLUDED.last_seen_at,
                    active_jobs = EXCLUDED.active_jobs,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
                """,
                (
                    spec.worker_id,
                    spec.worker_class,
                    _json(sorted(spec.capabilities)),
                    _json(dict(spec.labels)),
                    _json(list(spec.gpu_uuids)),
                    spec.resources.gpu_count,
                    spec.resources.total_vram_mb,
                    spec.resources.max_single_gpu_vram_mb,
                    registration.max_concurrency,
                    _json(dict(registration.metadata)),
                    registered_at,
                    timestamp,
                    active_jobs,
                    timestamp,
                ),
            ).fetchone()
            return self._worker(row)

    def get_worker(self, worker_id: str) -> WorkerRecord:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM workers WHERE id = %s", (worker_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"worker not found: {worker_id}")
        return self._worker(row)

    def set_worker_state(
        self,
        worker_id: str,
        state: WorkerState,
        *,
        now: datetime | None = None,
    ) -> WorkerRecord:
        timestamp = _aware(now)
        with self._transaction() as connection:
            current_row = connection.execute(
                "SELECT * FROM workers WHERE id = %s FOR UPDATE", (worker_id,)
            ).fetchone()
            if current_row is None:
                raise NotFoundError(f"worker not found: {worker_id}")
            current = self._worker(current_row)
            if state is WorkerState.ONLINE and current.state is WorkerState.OFFLINE:
                conflict = self._gpu_overlap(
                    connection,
                    WorkerRegistration(
                        spec=current.spec,
                        max_concurrency=current.max_concurrency,
                        metadata=current.metadata,
                    ),
                )
                if conflict:
                    raise ConflictError(
                        f"GPU identity overlaps with worker {conflict}"
                    )
            row = connection.execute(
                """
                UPDATE workers
                SET state = %s, last_seen_at = %s, updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (state.value, timestamp, timestamp, worker_id),
            ).fetchone()
            return self._worker(row)

    def heartbeat_worker(
        self,
        worker_id: str,
        heartbeat: WorkerHeartbeat | None = None,
        *,
        now: datetime | None = None,
    ) -> WorkerRecord:
        timestamp = _aware(now)
        heartbeat = heartbeat or WorkerHeartbeat()
        with self._transaction() as connection:
            # Keep the global lock order job -> worker whenever a job is
            # involved. Claim/completion/recovery use the same order.
            if heartbeat.active_job_id is not None or heartbeat.lease_token is not None:
                if not heartbeat.active_job_id or not heartbeat.lease_token:
                    raise ConflictError(
                        "active_job_id and lease_token must be supplied together"
                    )
                job_row = connection.execute(
                    "SELECT * FROM jobs WHERE id::text = %s FOR UPDATE",
                    (heartbeat.active_job_id,),
                ).fetchone()
                if job_row is None:
                    raise NotFoundError(
                        f"job not found: {heartbeat.active_job_id}"
                    )
                self._assert_running_row(
                    job_row,
                    worker_id=worker_id,
                    lease_token=heartbeat.lease_token,
                    now=timestamp,
                )
                connection.execute(
                    """
                    UPDATE jobs
                    SET lease_expires_at = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (
                        timestamp + timedelta(seconds=self.lease_seconds),
                        timestamp,
                        job_row["id"],
                    ),
                )

            worker_row = connection.execute(
                "SELECT * FROM workers WHERE id = %s FOR UPDATE", (worker_id,)
            ).fetchone()
            if worker_row is None:
                raise NotFoundError(f"worker not found: {worker_id}")
            current = self._worker(worker_row)
            if current.state is WorkerState.OFFLINE and heartbeat.state is WorkerState.ONLINE:
                raise ConflictError("offline worker must be explicitly returned online")

            active_jobs = connection.execute(
                """
                SELECT count(*)::integer AS count
                FROM jobs
                WHERE status = 'running' AND assigned_worker_id = %s
                """,
                (worker_id,),
            ).fetchone()["count"]
            state = (heartbeat.state or current.state).value
            metadata = {**dict(current.metadata), **dict(heartbeat.metadata)}
            row = connection.execute(
                """
                UPDATE workers
                SET state = %s, last_seen_at = %s, active_jobs = %s,
                    metadata = %s, updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (
                    state,
                    timestamp,
                    active_jobs,
                    _json(metadata),
                    timestamp,
                    worker_id,
                ),
            ).fetchone()
            return self._worker(row)

    def expire_stale_workers(
        self, *, now: datetime | None = None
    ) -> list[WorkerRecord]:
        timestamp = _aware(now)
        cutoff = timestamp - timedelta(seconds=self.worker_ttl_seconds)
        with self._transaction() as connection:
            rows = connection.execute(
                """
                UPDATE workers
                SET state = 'offline', updated_at = %s
                WHERE state <> 'offline' AND last_seen_at < %s
                RETURNING *
                """,
                (timestamp, cutoff),
            ).fetchall()
        return [self._worker(row) for row in rows]

    def submit_job(
        self, submission: JobSubmission, *, now: datetime | None = None
    ) -> JobRecord:
        timestamp = _aware(now)
        available_at = (
            _aware(submission.available_at)
            if submission.available_at is not None
            else timestamp
        )
        job_id = str(uuid4())
        requirements = requirements_to_dict(submission.requirements)

        with self._transaction() as connection:
            if submission.idempotency_key:
                existing = connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE idempotency_key = %s
                    FOR UPDATE
                    """,
                    (submission.idempotency_key,),
                ).fetchone()
                if existing is not None:
                    record = self._job(existing)
                    if not _submission_matches(record, submission):
                        raise ConflictError(
                            "idempotency key already belongs to a different job request"
                        )
                    return record

            row = connection.execute(
                """
                INSERT INTO jobs (
                    id, capability, payload, requirements, priority,
                    attempts, max_attempts, idempotency_key,
                    created_at, available_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    0, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING *
                """,
                (
                    job_id,
                    submission.capability,
                    _json(dict(submission.payload)),
                    _json(requirements),
                    submission.priority,
                    submission.max_attempts,
                    submission.idempotency_key,
                    timestamp,
                    available_at,
                    timestamp,
                ),
            ).fetchone()
            if row is not None:
                return self._job(row)

            existing = connection.execute(
                """
                SELECT * FROM jobs
                WHERE idempotency_key = %s
                FOR UPDATE
                """,
                (submission.idempotency_key,),
            ).fetchone()
            if existing is None:
                raise StorageUnavailable("idempotent job insert did not persist")
            record = self._job(existing)
            if not _submission_matches(record, submission):
                raise ConflictError(
                    "idempotency key already belongs to a different job request"
                )
            return record

    def get_job(self, job_id: str) -> JobRecord:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id::text = %s", (job_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"job not found: {job_id}")
        return self._job(row)

    def claim_next_job(
        self, worker_id: str, *, now: datetime | None = None
    ) -> JobRecord | None:
        timestamp = _aware(now)
        cutoff = timestamp - timedelta(seconds=self.worker_ttl_seconds)
        with self._transaction() as connection:
            worker_row = connection.execute(
                "SELECT * FROM workers WHERE id = %s", (worker_id,)
            ).fetchone()
            if worker_row is None:
                raise NotFoundError(f"worker not found: {worker_id}")
            worker = self._worker(worker_row)
            if worker.last_seen_at < cutoff:
                connection.execute(
                    """
                    UPDATE workers
                    SET state = 'offline', updated_at = %s
                    WHERE id = %s
                    """,
                    (timestamp, worker_id),
                )
                return None
            if worker.state is not WorkerState.ONLINE:
                return None

            row = connection.execute(
                """
                SELECT j.*
                FROM jobs AS j
                WHERE j.status = 'queued'
                  AND j.available_at <= %s
                  AND j.attempts < j.max_attempts
                  AND %s::jsonb ? j.capability
                  AND (
                      j.requirements->>'worker_class' IS NULL
                      OR j.requirements->>'worker_class' = %s
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM jsonb_array_elements_text(
                          COALESCE(j.requirements->'required_capabilities', '[]'::jsonb)
                      ) AS required(capability)
                      WHERE NOT (%s::jsonb ? required.capability)
                  )
                  AND COALESCE(j.requirements->'required_labels', '{}'::jsonb) <@ %s::jsonb
                  AND NOT EXISTS (
                      SELECT 1
                      FROM jsonb_array_elements_text(
                          COALESCE(j.requirements->'required_gpu_uuids', '[]'::jsonb)
                      ) AS required(uuid)
                      WHERE NOT (%s::jsonb ? required.uuid)
                  )
                  AND %s >= COALESCE(
                      (j.requirements->>'min_gpu_count')::integer, 0
                  )
                  AND %s >= COALESCE(
                      (j.requirements->>'min_total_vram_mb')::integer, 0
                  )
                  AND %s >= COALESCE(
                      (j.requirements->>'min_single_gpu_vram_mb')::integer, 0
                  )
                  AND (
                      SELECT w.active_jobs < w.max_concurrency
                      FROM workers AS w
                      WHERE w.id = %s
                  )
                ORDER BY j.priority DESC, j.sequence ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (
                    timestamp,
                    _json(sorted(worker.spec.capabilities)),
                    worker.spec.worker_class,
                    _json(sorted(worker.spec.capabilities)),
                    _json(dict(worker.spec.labels)),
                    _json(list(worker.spec.gpu_uuids)),
                    worker.spec.resources.gpu_count,
                    worker.spec.resources.total_vram_mb,
                    worker.spec.resources.max_single_gpu_vram_mb,
                    worker_id,
                ),
            ).fetchone()
            if row is None:
                return None

            locked_worker_row = connection.execute(
                "SELECT * FROM workers WHERE id = %s FOR UPDATE", (worker_id,)
            ).fetchone()
            if locked_worker_row is None:
                raise NotFoundError(f"worker not found: {worker_id}")
            locked_worker = self._worker(locked_worker_row)
            if (
                locked_worker.last_seen_at < cutoff
                or locked_worker.state is not WorkerState.ONLINE
                or locked_worker.active_jobs >= locked_worker.max_concurrency
            ):
                return None

            selected = self._job(row)
            if (
                selected.capability not in locked_worker.spec.capabilities
                or not worker_matches(locked_worker.spec, selected.requirements)
            ):
                return None

            lease_token = str(uuid4())
            claimed_row = connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    assigned_worker_id = %s,
                    lease_token = %s,
                    lease_expires_at = %s,
                    started_at = %s,
                    finished_at = NULL,
                    error = NULL,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (
                    worker_id,
                    lease_token,
                    timestamp + timedelta(seconds=self.lease_seconds),
                    timestamp,
                    timestamp,
                    row["id"],
                ),
            ).fetchone()
            connection.execute(
                """
                UPDATE workers
                SET active_jobs = active_jobs + 1, updated_at = %s
                WHERE id = %s
                """,
                (timestamp, worker_id),
            )
            return self._job(claimed_row)

    @staticmethod
    def _assert_running_row(
        row: dict[str, Any],
        *,
        worker_id: str,
        lease_token: str,
        now: datetime,
    ) -> None:
        if row["status"] != JobStatus.RUNNING.value:
            raise ConflictError(f"job {row['id']} is not running")
        if row.get("assigned_worker_id") != worker_id:
            raise ConflictError(f"job {row['id']} is assigned to another worker")
        if not lease_token or row.get("lease_token") != lease_token:
            raise ConflictError(f"job {row['id']} lease token is stale")
        expires_at = row.get("lease_expires_at")
        if expires_at is None or expires_at <= now:
            raise ConflictError(f"job {row['id']} lease has expired")

    @staticmethod
    def _decrement_worker(
        connection: Any, worker_id: str | None, timestamp: datetime
    ) -> None:
        if worker_id is None:
            return
        connection.execute(
            """
            UPDATE workers
            SET active_jobs = GREATEST(active_jobs - 1, 0), updated_at = %s
            WHERE id = %s
            """,
            (timestamp, worker_id),
        )

    def complete_job(
        self,
        job_id: str,
        result: JobResult,
        *,
        worker_id: str,
        lease_token: str,
        now: datetime | None = None,
    ) -> JobRecord:
        if not isinstance(result, JobResult):
            raise TypeError("result must be JobResult")
        timestamp = _aware(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id::text = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"job not found: {job_id}")
            self._assert_running_row(
                row,
                worker_id=worker_id,
                lease_token=lease_token,
                now=timestamp,
            )
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    result = %s,
                    error = NULL,
                    assigned_worker_id = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    finished_at = %s,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (
                    _json(job_result_to_dict(result)),
                    timestamp,
                    timestamp,
                    row["id"],
                ),
            ).fetchone()
            self._decrement_worker(connection, worker_id, timestamp)
            return self._job(updated)

    def fail_job(
        self,
        job_id: str,
        error: str | dict[str, object],
        *,
        retryable: bool,
        worker_id: str,
        lease_token: str,
        now: datetime | None = None,
    ) -> JobRecord:
        timestamp = _aware(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id::text = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"job not found: {job_id}")
            self._assert_running_row(
                row,
                worker_id=worker_id,
                lease_token=lease_token,
                now=timestamp,
            )
            should_retry = retryable and int(row["attempts"]) < int(row["max_attempts"])
            status = JobStatus.QUEUED if should_retry else JobStatus.FAILED
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = %s,
                    error = %s,
                    assigned_worker_id = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    started_at = CASE WHEN %s THEN NULL ELSE started_at END,
                    finished_at = CASE WHEN %s THEN NULL ELSE %s END,
                    available_at = CASE WHEN %s THEN %s ELSE available_at END,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (
                    status.value,
                    _json(_failure_payload(error, retryable=should_retry)),
                    should_retry,
                    should_retry,
                    timestamp,
                    should_retry,
                    timestamp,
                    timestamp,
                    row["id"],
                ),
            ).fetchone()
            self._decrement_worker(connection, worker_id, timestamp)
            return self._job(updated)

    def cancel_job(
        self, job_id: str, *, now: datetime | None = None
    ) -> JobRecord:
        timestamp = _aware(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id::text = %s FOR UPDATE", (job_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"job not found: {job_id}")
            job = self._job(row)
            if job.status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                return job
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'cancelled',
                    assigned_worker_id = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    finished_at = %s,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (timestamp, timestamp, row["id"]),
            ).fetchone()
            if job.status is JobStatus.RUNNING:
                self._decrement_worker(
                    connection, job.assigned_worker_id, timestamp
                )
            return self._job(updated)

    def recover_expired_jobs(
        self, *, now: datetime | None = None
    ) -> list[JobRecord]:
        timestamp = _aware(now)
        recovered: list[JobRecord] = []
        while True:
            with self._transaction() as connection:
                row = connection.execute(
                    """
                    SELECT *
                    FROM jobs
                    WHERE status = 'running'
                      AND lease_expires_at <= %s
                    ORDER BY sequence ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """,
                    (timestamp,),
                ).fetchone()
                if row is None:
                    return recovered

                should_retry = int(row["attempts"]) < int(row["max_attempts"])
                status = JobStatus.QUEUED if should_retry else JobStatus.FAILED
                updated = connection.execute(
                    """
                    UPDATE jobs
                    SET status = %s,
                        error = %s,
                        assigned_worker_id = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        started_at = CASE WHEN %s THEN NULL ELSE started_at END,
                        finished_at = CASE WHEN %s THEN NULL ELSE %s END,
                        available_at = CASE WHEN %s THEN %s ELSE available_at END,
                        updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        status.value,
                        _json(
                            {
                                "message": "worker lease expired",
                                "retryable": should_retry,
                            }
                        ),
                        should_retry,
                        should_retry,
                        timestamp,
                        should_retry,
                        timestamp,
                        timestamp,
                        row["id"],
                    ),
                ).fetchone()
                self._decrement_worker(
                    connection, row.get("assigned_worker_id"), timestamp
                )
                recovered.append(self._job(updated))


__all__ = ["PostgresControlRepository"]
