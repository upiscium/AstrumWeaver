-- Deployment-bound serving metadata and attempt fencing.
-- Existing v0.1 Workers/Jobs remain valid with NULL serving fields.

BEGIN;

ALTER TABLE workers
    ADD COLUMN IF NOT EXISTS serving jsonb;

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS serving jsonb,
    ADD COLUMN IF NOT EXISTS deadline_at timestamptz,
    ADD COLUMN IF NOT EXISTS claimed_deployment_revision text,
    ADD COLUMN IF NOT EXISTS claimed_serving_contract_revision text,
    ADD COLUMN IF NOT EXISTS claimed_runtime_instance_epoch text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_serving_requires_deadline'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_serving_requires_deadline
            CHECK (serving IS NULL OR deadline_at IS NOT NULL);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_claimed_serving_identity_complete'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_claimed_serving_identity_complete
            CHECK (
                (
                    claimed_deployment_revision IS NULL
                    AND claimed_serving_contract_revision IS NULL
                    AND claimed_runtime_instance_epoch IS NULL
                )
                OR
                (
                    claimed_deployment_revision IS NOT NULL
                    AND claimed_serving_contract_revision IS NOT NULL
                    AND claimed_runtime_instance_epoch IS NOT NULL
                )
            );
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS jobs_deadline_idx
    ON jobs (deadline_at)
    WHERE status IN ('queued', 'running') AND deadline_at IS NOT NULL;

COMMIT;
