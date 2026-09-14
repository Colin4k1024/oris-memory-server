//! PostgreSQL schema definition and migration runner.
//!
//! The DDL matches the architecture document (`docs/CONTROL_PLANE_ARCHITECTURE.md` §3).
//! All tables use `memory_type` as a field, not a separate database.

use sqlx::{PgPool, Row};

/// DDL for the full memory service schema.
pub const SCHEMA_DDL: &str = r#"
-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";     -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "vector";        -- pgvector

-- ──────────────────────── memory_item ────────────────────────
CREATE TABLE IF NOT EXISTS memory_item (
    memory_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id          TEXT NOT NULL,
    memory_type        TEXT NOT NULL,
    scope              TEXT NOT NULL,
    subject_type       TEXT,
    subject_id         TEXT,
    entity_refs        JSONB DEFAULT '[]',
    content            TEXT,
    structured_payload JSONB,
    embedding         vector(1536),
    source_type        TEXT NOT NULL,
    source_reference   TEXT,
    evidence_refs      JSONB DEFAULT '[]',
    confidence         REAL DEFAULT 0.5,
    authority_level    TEXT NOT NULL,
    importance         REAL DEFAULT 0.5,
    observed_at        TIMESTAMPTZ,
    valid_from         TIMESTAMPTZ,
    valid_to           TIMESTAMPTZ,
    privacy_class      TEXT NOT NULL,
    acl                JSONB DEFAULT '{}',
    retention_policy   TEXT,
    status             TEXT DEFAULT 'active',
    version            INTEGER DEFAULT 1,
    derived_from       JSONB DEFAULT '[]',
    created_by_user    TEXT,
    created_by_agent   TEXT,
    last_verified_at   TIMESTAMPTZ,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_filter
    ON memory_item (tenant_id, scope, memory_type, status, valid_to);
CREATE INDEX IF NOT EXISTS idx_memory_subject
    ON memory_item (subject_type, subject_id);
CREATE INDEX IF NOT EXISTS idx_memory_tenant_created
    ON memory_item (tenant_id, created_at DESC);

-- HNSW vector index (created separately — may need tuning)
CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON memory_item USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ──────────────────── canonical_user_profile ──────────────────
CREATE TABLE IF NOT EXISTS canonical_user_profile (
    user_id          TEXT PRIMARY KEY,
    organization_id  TEXT NOT NULL,
    factory_id       TEXT,
    identity_links   JSONB DEFAULT '[]',
    role             TEXT,
    position         TEXT,
    language         TEXT,
    timezone         TEXT,
    preferences      JSONB DEFAULT '{}',
    common_entities  JSONB DEFAULT '[]',
    active_projects  JSONB DEFAULT '[]',
    consent_scope    JSONB DEFAULT '{}',
    privacy_class    TEXT DEFAULT 'internal',
    source           TEXT NOT NULL,
    authority_level  TEXT DEFAULT 'L1_authoritative',
    version          INTEGER DEFAULT 1,
    valid_from       TIMESTAMPTZ,
    valid_to         TIMESTAMPTZ,
    last_verified_at TIMESTAMPTZ,
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_org
    ON canonical_user_profile (organization_id);

-- ─────────────────────── shared_task_context ──────────────────
CREATE TABLE IF NOT EXISTS shared_task_context (
    task_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_task_id    UUID,
    initiator_user_id TEXT NOT NULL,
    organization_scope TEXT NOT NULL,
    goal              TEXT NOT NULL,
    constraints        JSONB DEFAULT '[]',
    success_criteria   JSONB DEFAULT '[]',
    entities           JSONB DEFAULT '[]',
    business_refs      JSONB DEFAULT '[]',
    current_findings   JSONB DEFAULT '[]',
    evidence_refs      JSONB DEFAULT '[]',
    decisions          JSONB DEFAULT '[]',
    assumptions        JSONB DEFAULT '[]',
    completed_steps    JSONB DEFAULT '[]',
    pending_steps      JSONB DEFAULT '[]',
    current_owner_agent TEXT,
    participant_agents  JSONB DEFAULT '[]',
    artifact_refs       JSONB DEFAULT '[]',
    source_system_refs  JSONB DEFAULT '[]',
    status              TEXT DEFAULT 'active',
    version             INTEGER DEFAULT 1,
    expires_at          TIMESTAMPTZ,
    acl                 JSONB DEFAULT '{}',
    privacy_class       TEXT DEFAULT 'internal',
    audit_ref           TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_org
    ON shared_task_context (organization_scope, status);

-- ──────────────────── decision_record ─────────────────────────
CREATE TABLE IF NOT EXISTS decision_record (
    decision_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id        UUID REFERENCES shared_task_context(task_id),
    situation     TEXT NOT NULL,
    options       JSONB DEFAULT '[]',
    recommendation TEXT,
    human_decision TEXT,
    reason         TEXT,
    action         TEXT,
    outcome        TEXT,
    evidence_refs  JSONB DEFAULT '[]',
    decided_at     TIMESTAMPTZ,
    outcome_at     TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────── entity / entity_relation ────────────────
CREATE TABLE IF NOT EXISTS entity (
    entity_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    name        TEXT NOT NULL,
    attributes  JSONB DEFAULT '{}',
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS entity_relation (
    relation_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_entity   UUID REFERENCES entity(entity_id),
    to_entity     UUID REFERENCES entity(entity_id),
    relation_type TEXT NOT NULL,
    attributes    JSONB DEFAULT '{}',
    valid_from    TIMESTAMPTZ,
    valid_to      TIMESTAMPTZ,
    source        TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_tenant_type
    ON entity (tenant_id, entity_type);

-- ──────────────────── memory_access_audit ─────────────────────
CREATE TABLE IF NOT EXISTS memory_access_audit (
    audit_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id   UUID,
    accessor_user  TEXT,
    accessor_agent TEXT,
    action      TEXT NOT NULL,
    purpose     TEXT,
    task_id     UUID,
    trace_id    TEXT,
    policy_decision_id TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────── memory_version ─────────────────────────
CREATE TABLE IF NOT EXISTS memory_version (
    version_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id   UUID NOT NULL,
    version     INTEGER NOT NULL,
    payload     JSONB NOT NULL,
    changed_by  TEXT NOT NULL,
    change_reason TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(memory_id, version)
);

-- ──────────────────── outbox_event ───────────────────────────
CREATE TABLE IF NOT EXISTS outbox_event (
    event_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type  TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    payload     JSONB NOT NULL,
    status      TEXT DEFAULT 'pending',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_outbox_status_created
    ON outbox_event (status, created_at);

-- ──────────────────── approval_request ──────────────────────
CREATE TABLE IF NOT EXISTS approval_request (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id     UUID NOT NULL,
    target_scope  TEXT NOT NULL,
    requester     TEXT NOT NULL,
    justification TEXT,
    approver_role TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at    TIMESTAMPTZ,
    decided_by    TEXT,
    expires_at    TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_approval_memory_scope
    ON approval_request (memory_id, target_scope, status);
CREATE INDEX IF NOT EXISTS idx_approval_status
    ON approval_request (status, expires_at);
"#;

/// RLS policy DDL.
pub const RLS_DDL: &str = r#"
ALTER TABLE memory_item ENABLE ROW LEVEL SECURITY;
ALTER TABLE canonical_user_profile ENABLE ROW LEVEL SECURITY;
ALTER TABLE shared_task_context ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS tenant_isolation ON memory_item
    USING (tenant_id = current_setting('app.tenant_id', true));
CREATE POLICY IF NOT EXISTS tenant_isolation_user ON canonical_user_profile
    USING (organization_id = current_setting('app.tenant_id', true));
CREATE POLICY IF NOT EXISTS tenant_isolation_task ON shared_task_context
    USING (organization_scope = current_setting('app.tenant_id', true));
"#;

/// Schema manager — runs DDL and migrations against a PostgreSQL pool.
pub struct PostgresSchema;

impl PostgresSchema {
    /// Apply the full schema DDL (idempotent — safe to call on every startup).
    pub async fn apply(pool: &PgPool) -> Result<(), sqlx::Error> {
        sqlx::query(SCHEMA_DDL).execute(pool).await?;
        tracing::info!("schema DDL applied");
        Ok(())
    }

    /// Apply Row Level Security policies.
    pub async fn apply_rls(pool: &PgPool) -> Result<(), sqlx::Error> {
        sqlx::query(RLS_DDL).execute(pool).await?;
        tracing::info!("RLS policies applied");
        Ok(())
    }

    /// Set the tenant context for RLS on the current connection.
    pub async fn set_tenant(pool: &PgPool, tenant_id: &str) -> Result<(), sqlx::Error> {
        sqlx::query("SELECT set_config('app.tenant_id', $1, true)")
            .bind(tenant_id)
            .execute(pool)
            .await?;
        Ok(())
    }

    /// Check if the pgvector extension is installed.
    pub async fn has_vector_extension(pool: &PgPool) -> Result<bool, sqlx::Error> {
        let row = sqlx::query(
            "SELECT EXISTS(
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            ) as exists",
        )
        .fetch_one(pool)
        .await?;
        Ok(row.try_get::<bool, _>("exists").unwrap_or(false))
    }

    /// Count tables in the schema (for health checks).
    pub async fn table_count(pool: &PgPool) -> Result<i64, sqlx::Error> {
        let row = sqlx::query(
            "SELECT COUNT(*) as cnt FROM information_schema.tables
             WHERE table_schema = 'public'",
        )
        .fetch_one(pool)
        .await?;
        Ok(row.try_get::<i64, _>("cnt").unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddl_is_non_empty() {
        assert!(SCHEMA_DDL.contains("CREATE TABLE"));
        assert!(SCHEMA_DDL.contains("memory_item"));
        assert!(SCHEMA_DDL.contains("canonical_user_profile"));
        assert!(SCHEMA_DDL.contains("shared_task_context"));
        assert!(SCHEMA_DDL.contains("outbox_event"));
        assert!(SCHEMA_DDL.contains("approval_request"));
        assert!(SCHEMA_DDL.contains("vector(1536)"));
        assert!(SCHEMA_DDL.contains("hnsw"));
    }

    #[test]
    fn rls_ddl_is_non_empty() {
        assert!(RLS_DDL.contains("ENABLE ROW LEVEL SECURITY"));
        assert!(RLS_DDL.contains("tenant_isolation"));
    }
}
