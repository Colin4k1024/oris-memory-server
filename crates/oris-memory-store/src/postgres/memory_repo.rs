//! CRUD operations for the `memory_item` table.

use chrono::Utc;
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::memory_types::{
    AuthorityLevel, MemoryCandidate, MemoryItem, MemoryStatus, MemoryType, PrivacyClass, Scope,
    SourceType,
};

/// Repository for `memory_item` records.
pub struct MemoryRepo {
    pool: PgPool,
}

impl MemoryRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert a new memory item from a candidate. Returns the created item's ID.
    pub async fn insert(&self, candidate: &MemoryCandidate) -> Result<Uuid, MemoryRepoError> {
        let memory_id = Uuid::new_v4();

        sqlx::query(
            r#"INSERT INTO memory_item (
                memory_id, tenant_id, memory_type, scope,
                subject_type, subject_id, content, structured_payload,
                source_type, source_reference, confidence, authority_level,
                privacy_class, status, version, created_by_user, created_by_agent,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8,
                $9, $10, $11, $12, $13, 'candidate', 1, $14, $15,
                NOW(), NOW()
            )"#,
        )
        .bind(memory_id)
        .bind(&candidate.tenant_id)
        .bind(candidate.memory_type.as_str())
        .bind(candidate.scope.as_str())
        .bind(&candidate.subject_type)
        .bind(&candidate.subject_id)
        .bind(&candidate.content)
        .bind(&candidate.structured_payload)
        .bind(candidate.source_type.as_str())
        .bind(&candidate.source_reference)
        .bind(candidate.confidence)
        .bind(candidate.authority_level.as_str())
        .bind(candidate.privacy_class.as_str())
        .bind(&candidate.created_by_user)
        .bind(&candidate.created_by_agent)
        .execute(&self.pool)
        .await?;

        Ok(memory_id)
    }

    /// Fetch a single memory item by ID.
    pub async fn get_by_id(&self, memory_id: Uuid) -> Result<Option<MemoryItem>, MemoryRepoError> {
        let row = sqlx::query(r#"SELECT * FROM memory_item WHERE memory_id = $1"#)
            .bind(memory_id)
            .fetch_optional(&self.pool)
            .await?;

        row.map(|r| map_row_to_memory_item(&r)).transpose()
    }

    /// Update the status of a memory item (e.g., candidate → active).
    pub async fn update_status(
        &self,
        memory_id: Uuid,
        new_status: MemoryStatus,
    ) -> Result<(), MemoryRepoError> {
        let result = sqlx::query(
            r#"UPDATE memory_item SET status = $2, updated_at = NOW()
               WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .bind(new_status.as_str())
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(MemoryRepoError::NotFound);
        }
        Ok(())
    }

    /// Soft-delete: set status to 'archived'.
    pub async fn archive(&self, memory_id: Uuid) -> Result<(), MemoryRepoError> {
        self.update_status(memory_id, MemoryStatus::Archived).await
    }

    /// Revoke a memory item (set status to 'revoked').
    pub async fn revoke(&self, memory_id: Uuid) -> Result<(), MemoryRepoError> {
        self.update_status(memory_id, MemoryStatus::Revoked).await
    }

    /// Quarantine a memory item (set status to 'quarantined').
    pub async fn quarantine(&self, memory_id: Uuid) -> Result<(), MemoryRepoError> {
        self.update_status(memory_id, MemoryStatus::Quarantined)
            .await
    }

    /// Promote a candidate to active status.
    pub async fn promote(&self, memory_id: Uuid) -> Result<(), MemoryRepoError> {
        self.update_status(memory_id, MemoryStatus::Active).await
    }

    /// List memories for a tenant, optionally filtered by status.
    pub async fn list_by_tenant(
        &self,
        tenant_id: &str,
        status_filter: Option<MemoryStatus>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<MemoryItem>, MemoryRepoError> {
        let rows = if let Some(status) = status_filter {
            sqlx::query(
                r#"SELECT * FROM memory_item
                   WHERE tenant_id = $1 AND status = $2
                   ORDER BY created_at DESC
                   LIMIT $3 OFFSET $4"#,
            )
            .bind(tenant_id)
            .bind(status.as_str())
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                r#"SELECT * FROM memory_item
                   WHERE tenant_id = $1
                   ORDER BY created_at DESC
                   LIMIT $2 OFFSET $3"#,
            )
            .bind(tenant_id)
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await?
        };

        rows.iter().map(|r| map_row_to_memory_item(r)).collect()
    }

    /// List memories for a specific subject (user, entity, etc.).
    pub async fn list_by_subject(
        &self,
        tenant_id: &str,
        subject_type: &str,
        subject_id: &str,
        limit: i64,
    ) -> Result<Vec<MemoryItem>, MemoryRepoError> {
        let rows = sqlx::query(
            r#"SELECT * FROM memory_item
               WHERE tenant_id = $1 AND subject_type = $2 AND subject_id = $3
                 AND status = 'active'
               ORDER BY created_at DESC
               LIMIT $4"#,
        )
        .bind(tenant_id)
        .bind(subject_type)
        .bind(subject_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(|r| map_row_to_memory_item(r)).collect()
    }

    /// Update the embedding for a memory item.
    pub async fn set_embedding(
        &self,
        memory_id: Uuid,
        embedding: &[f32],
    ) -> Result<(), MemoryRepoError> {
        let vec_str = encode_vector(embedding);
        let result = sqlx::query(
            r#"UPDATE memory_item SET embedding = $2::vector, updated_at = NOW()
               WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .bind(&vec_str)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(MemoryRepoError::NotFound);
        }
        Ok(())
    }

    /// Record a version snapshot before an update (for audit trail).
    pub async fn record_version(
        &self,
        memory_id: Uuid,
        version: i32,
        payload: &serde_json::Value,
        changed_by: &str,
        change_reason: Option<&str>,
    ) -> Result<(), MemoryRepoError> {
        sqlx::query(
            r#"INSERT INTO memory_version (version_id, memory_id, version, payload, changed_by, change_reason)
               VALUES (gen_random_uuid(), $1, $2, $3, $4, $5)"#,
        )
        .bind(memory_id)
        .bind(version)
        .bind(payload)
        .bind(changed_by)
        .bind(change_reason)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Bump version counter (call after a content update).
    pub async fn bump_version(&self, memory_id: Uuid) -> Result<i32, MemoryRepoError> {
        let row = sqlx::query(
            r#"UPDATE memory_item
               SET version = version + 1, updated_at = NOW()
               WHERE memory_id = $1
               RETURNING version"#,
        )
        .bind(memory_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(row.get::<i32, _>("version"))
    }
}

// ──────────────────────── Row Mapping ────────────────────────

fn map_row_to_memory_item(row: &sqlx::postgres::PgRow) -> Result<MemoryItem, MemoryRepoError> {
    // Embedding is not decoded here — it requires a text cast in the query.
    // Use set_embedding() for writes and SearchRepo for vector reads.
    let embedding: Option<Vec<f32>> = None;
    Ok(MemoryItem {
        memory_id: row.try_get("memory_id")?,
        tenant_id: row.try_get("tenant_id")?,
        memory_type: parse_memory_type(row.try_get("memory_type")?),
        scope: parse_scope(row.try_get("scope")?),
        subject_type: row.try_get("subject_type")?,
        subject_id: row.try_get("subject_id")?,
        entity_refs: row
            .try_get::<serde_json::Value, _>("entity_refs")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        content: row.try_get("content")?,
        structured_payload: row.try_get("structured_payload")?,
        embedding: embedding.map(|v| v.to_vec()),
        source_type: parse_source_type(row.try_get("source_type")?),
        source_reference: row.try_get("source_reference")?,
        evidence_refs: row
            .try_get::<serde_json::Value, _>("evidence_refs")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        confidence: row.try_get("confidence")?,
        authority_level: parse_authority_level(row.try_get("authority_level")?),
        importance: row.try_get("importance")?,
        observed_at: row.try_get("observed_at")?,
        valid_from: row.try_get("valid_from")?,
        valid_to: row.try_get("valid_to")?,
        privacy_class: parse_privacy_class(row.try_get("privacy_class")?),
        acl: row
            .try_get::<serde_json::Value, _>("acl")
            .unwrap_or_default(),
        retention_policy: row.try_get("retention_policy")?,
        status: parse_memory_status(row.try_get("status")?),
        version: row.try_get("version")?,
        derived_from: row
            .try_get::<serde_json::Value, _>("derived_from")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        created_by_user: row.try_get("created_by_user")?,
        created_by_agent: row.try_get("created_by_agent")?,
        last_verified_at: row.try_get("last_verified_at")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn parse_memory_type(s: &str) -> MemoryType {
    match s {
        "semantic" => MemoryType::Semantic,
        "episodic" => MemoryType::Episodic,
        "decision" => MemoryType::Decision,
        "experience" => MemoryType::Experience,
        "user_preference" => MemoryType::UserPreference,
        _ => MemoryType::Semantic,
    }
}

fn parse_scope(s: &str) -> Scope {
    match s {
        "personal" => Scope::Personal,
        "agent" => Scope::Agent,
        "task" => Scope::Task,
        "team" => Scope::Team,
        "process" => Scope::Process,
        "factory" => Scope::Factory,
        "enterprise" => Scope::Enterprise,
        _ => Scope::Personal,
    }
}

fn parse_source_type(s: &str) -> SourceType {
    match s {
        "iam" => SourceType::Iam,
        "hr" => SourceType::Hr,
        "user_explicit" => SourceType::UserExplicit,
        "agent_inferred" => SourceType::AgentInferred,
        "tool_result" => SourceType::ToolResult,
        "business_event" => SourceType::BusinessEvent,
        _ => SourceType::AgentInferred,
    }
}

fn parse_authority_level(s: &str) -> AuthorityLevel {
    match s {
        "L0_source_of_truth" => AuthorityLevel::L0SourceOfTruth,
        "L1_authoritative" => AuthorityLevel::L1Authoritative,
        "L2_verified" => AuthorityLevel::L2Verified,
        "L3_inferred" => AuthorityLevel::L3Inferred,
        _ => AuthorityLevel::L3Inferred,
    }
}

fn parse_privacy_class(s: &str) -> PrivacyClass {
    match s {
        "public" => PrivacyClass::Public,
        "internal" => PrivacyClass::Internal,
        "confidential" => PrivacyClass::Confidential,
        "restricted" => PrivacyClass::Restricted,
        _ => PrivacyClass::Internal,
    }
}

fn parse_memory_status(s: &str) -> MemoryStatus {
    match s {
        "candidate" => MemoryStatus::Candidate,
        "active" => MemoryStatus::Active,
        "archived" => MemoryStatus::Archived,
        "revoked" => MemoryStatus::Revoked,
        "quarantined" => MemoryStatus::Quarantined,
        _ => MemoryStatus::Active,
    }
}

// ──────────────────────── Errors ────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum MemoryRepoError {
    #[error("memory item not found")]
    NotFound,

    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}

/// Encode a float slice as a PostgreSQL vector string literal: "[0.1,0.2,0.3]".
fn encode_vector(v: &[f32]) -> String {
    let inner: Vec<String> = v.iter().map(|f| f.to_string()).collect();
    format!("[{}]", inner.join(","))
}

/// Parse a PostgreSQL vector text representation back to floats.
fn parse_vector(s: &str) -> Vec<f32> {
    s.trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|p| p.trim().parse::<f32>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips() {
        for mt in [
            MemoryType::Semantic,
            MemoryType::Episodic,
            MemoryType::Decision,
            MemoryType::Experience,
            MemoryType::UserPreference,
        ] {
            assert_eq!(mt.as_str(), parse_memory_type(mt.as_str()).as_str());
        }

        for al in [
            AuthorityLevel::L0SourceOfTruth,
            AuthorityLevel::L1Authoritative,
            AuthorityLevel::L2Verified,
            AuthorityLevel::L3Inferred,
        ] {
            assert_eq!(al.as_str(), parse_authority_level(al.as_str()).as_str());
        }
    }
}
