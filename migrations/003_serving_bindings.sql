-- Deployment-bound serving identity and bounded job deadlines.

BEGIN;

ALTER TABLE workers
    ADD COLUMN IF NOT EXISTS serving jsonb;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS serving_binding jsonb,
    ADD COLUMN IF NOT EXISTS deadline_at timestamptz,
    ADD COLUMN IF NOT EXISTS attempt_runtime_instance_epoch text;

CREATE INDEX IF NOT EXISTS jobs_serving_deadline_idx
    ON jobs (deadline_at)
    WHERE status IN ('queued', 'running') AND deadline_at IS NOT NULL;

COMMIT;
