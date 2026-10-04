-- Deployment-bound serving identity and bounded job deadlines.

BEGIN;

ALTER TABLE workers
    ADD COLUMN IF NOT EXISTS serving jsonb;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS serving_binding jsonb,
    ADD COLUMN IF NOT EXISTS deadline_at timestamptz,
    ADD COLUMN IF NOT EXISTS attempt_runtime_instance_epoch text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_serving_deadline_check'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_serving_deadline_check
            CHECK (serving_binding IS NULL OR deadline_at IS NOT NULL);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_serving_epoch_scope_check'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_serving_epoch_scope_check
            CHECK (
                attempt_runtime_instance_epoch IS NULL
                OR (serving_binding IS NOT NULL AND status = 'running')
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_serving_running_epoch_check'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_serving_running_epoch_check
            CHECK (
                serving_binding IS NULL
                OR status <> 'running'
                OR attempt_runtime_instance_epoch IS NOT NULL
            );
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS jobs_serving_deadline_idx
    ON jobs (deadline_at)
    WHERE status IN ('queued', 'running') AND deadline_at IS NOT NULL;

COMMIT;
