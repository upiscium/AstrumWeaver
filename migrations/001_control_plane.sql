-- AstrumWeaver v0.1 durable control-plane state.
-- Application-generated UUID job IDs avoid requiring PostgreSQL extensions.

BEGIN;

CREATE TABLE IF NOT EXISTS workers (
    id text PRIMARY KEY,
    worker_class text NOT NULL,
    capabilities jsonb NOT NULL DEFAULT '[]'::jsonb,
    labels jsonb NOT NULL DEFAULT '{}'::jsonb,
    gpu_uuids jsonb NOT NULL DEFAULT '[]'::jsonb,
    gpu_count integer NOT NULL DEFAULT 0 CHECK (gpu_count >= 0),
    total_vram_mb integer NOT NULL DEFAULT 0 CHECK (total_vram_mb >= 0),
    max_single_gpu_vram_mb integer NOT NULL DEFAULT 0
        CHECK (max_single_gpu_vram_mb >= 0),
    max_concurrency integer NOT NULL DEFAULT 1 CHECK (max_concurrency > 0),
    state text NOT NULL DEFAULT 'online'
        CHECK (state IN ('online', 'draining', 'offline')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    registered_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    active_jobs integer NOT NULL DEFAULT 0 CHECK (active_jobs >= 0),
    updated_at timestamptz NOT NULL,
    CHECK (
        (gpu_count = 0 AND total_vram_mb = 0 AND max_single_gpu_vram_mb = 0)
        OR
        (
            gpu_count > 0
            AND total_vram_mb > 0
            AND max_single_gpu_vram_mb > 0
            AND max_single_gpu_vram_mb <= total_vram_mb
        )
    )
);

CREATE INDEX IF NOT EXISTS workers_schedulable_idx
    ON workers (state, worker_class, last_seen_at);

CREATE TABLE IF NOT EXISTS jobs (
    id uuid PRIMARY KEY,
    sequence bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
    capability text NOT NULL,
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    requirements jsonb NOT NULL DEFAULT '{}'::jsonb,
    priority integer NOT NULL DEFAULT 0,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    idempotency_key text UNIQUE,
    assigned_worker_id text REFERENCES workers(id) ON DELETE RESTRICT,
    lease_token text,
    result jsonb,
    error jsonb,
    created_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    started_at timestamptz,
    finished_at timestamptz,
    lease_expires_at timestamptz,
    updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS jobs_claim_idx
    ON jobs (status, priority DESC, sequence ASC, available_at);
CREATE INDEX IF NOT EXISTS jobs_worker_idx
    ON jobs (assigned_worker_id, status);
CREATE INDEX IF NOT EXISTS jobs_lease_idx
    ON jobs (status, lease_expires_at)
    WHERE status = 'running';

COMMIT;
