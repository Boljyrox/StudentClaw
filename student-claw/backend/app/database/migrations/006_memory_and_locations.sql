-- 006: long-term group memory + member home locations for /meetpoint.
--
-- Memory replaces the old "clear the whole cache" model with discrete facts the
-- group can list, edit and delete one at a time. Entries older than one month
-- are pruned by the application (see MEMORY_RETENTION_DAYS).

CREATE TABLE IF NOT EXISTS group_memories (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id         UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    content            TEXT NOT NULL,
    source             VARCHAR(16) NOT NULL DEFAULT 'user',
    created_by_user_id BIGINT,
    created_by_name    VARCHAR(100),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_group_memories_project_id
    ON group_memories (project_id);

CREATE TABLE IF NOT EXISTS member_locations (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id        UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    display_name      VARCHAR(100) NOT NULL,
    telegram_user_id  BIGINT,
    telegram_username VARCHAR(50),
    raw_input         VARCHAR(255) NOT NULL,
    address           VARCHAR(255),
    postal_code       VARCHAR(12),
    latitude          DOUBLE PRECISION,
    longitude         DOUBLE PRECISION,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_member_location_name UNIQUE (project_id, display_name)
);

CREATE INDEX IF NOT EXISTS ix_member_locations_project_id
    ON member_locations (project_id);
