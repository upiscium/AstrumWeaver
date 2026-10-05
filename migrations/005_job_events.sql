-- Bounded per-attempt event channel for fenced live output.
-- Event payloads remain opaque to Control; Worker lease/runtime fencing is
-- enforced by repository writes before rows are inserted.

BEGIN;

CREATE TABLE IF NOT EXISTS job_events (
    job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    sequence bigint NOT NULL CHECK (sequence > 0),
    attempt integer NOT NULL CHECK (attempt > 0),
    worker_id text NOT NULL REFERENCES workers(id) ON DELETE RESTRICT,
    runtime_instance_epoch text,
    kind text NOT NULL CHECK (btrim(kind) <> ''),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    payload_bytes integer NOT NULL CHECK (payload_bytes >= 0),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (job_id, sequence)
);

CREATE INDEX IF NOT EXISTS job_events_job_sequence_idx
    ON job_events (job_id, sequence);

COMMIT;
