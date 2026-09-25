-- Preserve per-device accelerator facts required for runtime compatibility.

BEGIN;

ALTER TABLE workers
    ADD COLUMN IF NOT EXISTS accelerators jsonb NOT NULL DEFAULT '[]'::jsonb;

COMMIT;
