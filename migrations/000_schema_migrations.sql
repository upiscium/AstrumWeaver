-- Track canonical AstrumWeaver database migrations.

BEGIN;

CREATE TABLE IF NOT EXISTS schema_migrations (
    name text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

COMMIT;
