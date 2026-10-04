-- Fail closed if durable serving bindings and claimed attempt identity diverge.
-- This is additive rather than modifying migration 003 so databases that
-- already recorded 003 still receive the stronger invariants.

BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_deadline_matches_serving_binding'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_deadline_matches_serving_binding
            CHECK ((serving IS NULL) = (deadline_at IS NULL));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_serving_binding_matches_capability'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_serving_binding_matches_capability
            CHECK (
                serving IS NULL
                OR (
                    serving ? 'capability'
                    AND serving->>'capability' = capability
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_claimed_identity_matches_serving_binding'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_claimed_identity_matches_serving_binding
            CHECK (
                claimed_deployment_revision IS NULL
                OR (
                    serving IS NOT NULL
                    AND serving ? 'deployment_revision'
                    AND serving ? 'serving_contract_revision'
                    AND claimed_deployment_revision
                        = serving->>'deployment_revision'
                    AND claimed_serving_contract_revision
                        = serving->>'serving_contract_revision'
                    AND btrim(claimed_runtime_instance_epoch) <> ''
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'jobs'::regclass
          AND conname = 'jobs_running_serving_has_claimed_identity'
    ) THEN
        ALTER TABLE jobs
            ADD CONSTRAINT jobs_running_serving_has_claimed_identity
            CHECK (
                status <> 'running'
                OR serving IS NULL
                OR claimed_deployment_revision IS NOT NULL
            );
    END IF;
END
$$;

COMMIT;
