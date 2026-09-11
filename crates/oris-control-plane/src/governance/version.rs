//! Version tracking — snapshots `memory_item` state into the
//! `memory_version` table for audit trails and point-in-time recovery.
//!
//! The [`VersionManager`] provides create / list / get / rollback operations.
//! Each version snapshot stores the full [`MemoryItem`] serialised as a
//! JSONB `payload` in the `memory_version` table (the table is created by the
//! shared schema DDL and already exists).

use chrono::{DateTime, Utc};
use oris_memory_store::memory_types::MemoryItem;
use oris_memory_store::postgres::{MemoryRepo, MemoryRepoError, Pool};
use serde_json::Value;
use sqlx::Row;
use uuid::Uuid;

/// A historical snapshot of a `memory_item` row.
///
/// The `payload` field contains the full [`MemoryItem`] serialised as JSONB.
/// Use [`MemoryVersion::to_memory_item`] to deserialise it back into a
/// [`MemoryItem`].
///
/// **Note:** the `embedding` vector column is not captured in version
/// snapshots (it is always `None` in the deserialised [`MemoryItem`]),
/// consistent with [`MemoryRepo::get_by_id`].
#[derive(Debug, Clone)]
pub struct MemoryVersion {
    pub version_id: Uuid,
    pub memory_id: Uuid,
    pub version_number: i32,
    pub payload: Value,
    pub changed_by: String,
    pub change_reason: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl MemoryVersion {
    /// Deserialise the `payload` back into a [`MemoryItem`].
    pub fn to_memory_item(&self) -> Result<MemoryItem, VersionError> {
        serde_json::from_value(self.payload.clone()).map_err(VersionError::from)
    }
}

/// Creates, lists, and rolls back memory version snapshots.
pub struct VersionManager {
    pool: Pool,
}

impl VersionManager {
    /// Create a new manager backed by the given connection pool.
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Snapshot the current `memory_item` row into the `memory_version` table.
    ///
    /// The snapshot is a full JSONB copy of the memory item (serialised via
    /// serde so that enum variants round-trip correctly).  The `embedding`
    /// vector column is not included.
    pub async fn create_version(
        &self,
        memory_id: Uuid,
        changed_by: &str,
        change_reason: Option<&str>,
    ) -> Result<MemoryVersion, VersionError> {
        // 1. Fetch the current memory item (uses the store's row mapper so
        //    enum columns are parsed from their DB text representation).
        let repo = MemoryRepo::new(self.pool.clone());
        let item = repo
            .get_by_id(memory_id)
            .await
            .map_err(convert_repo_error(memory_id))?
            .ok_or(VersionError::NotFound(memory_id))?;

        // 2. Serialise the full item as the version payload.
        let payload = serde_json::to_value(&item)?;

        // 3. Insert the version snapshot.
        let row = sqlx::query(
            r#"INSERT INTO memory_version (version_id, memory_id, version, payload, changed_by, change_reason)
               VALUES (gen_random_uuid(), $1, $2, $3, $4, $5)
               RETURNING version_id, memory_id, version, payload, changed_by, change_reason, created_at"#,
        )
        .bind(memory_id)
        .bind(item.version)
        .bind(&payload)
        .bind(changed_by)
        .bind(change_reason)
        .fetch_one(&self.pool)
        .await?;

        map_row_to_version(&row)
    }

    /// List all version snapshots for a memory, newest first.
    pub async fn list_versions(&self, memory_id: Uuid) -> Result<Vec<MemoryVersion>, VersionError> {
        let rows = sqlx::query(
            r#"SELECT version_id, memory_id, version, payload, changed_by, change_reason, created_at
               FROM memory_version
               WHERE memory_id = $1
               ORDER BY created_at DESC"#,
        )
        .bind(memory_id)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(map_row_to_version).collect()
    }

    /// Fetch a specific version snapshot by version number.
    pub async fn get_version(
        &self,
        memory_id: Uuid,
        version_num: i32,
    ) -> Result<MemoryVersion, VersionError> {
        let row = sqlx::query(
            r#"SELECT version_id, memory_id, version, payload, changed_by, change_reason, created_at
               FROM memory_version
               WHERE memory_id = $1 AND version = $2"#,
        )
        .bind(memory_id)
        .bind(version_num)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(r) => map_row_to_version(&r),
            None => Err(VersionError::VersionNotFound {
                memory_id,
                version: version_num,
            }),
        }
    }

    /// Restore a `memory_item` to a previous version.
    ///
    /// 1. A new version snapshot of the **current** state is created first
    ///    (so the pre-rollback state is never lost).
    /// 2. The target version's payload is deserialised into a [`MemoryItem`].
    /// 3. All `memory_item` columns are restored from the target version,
    ///    and the `version` counter is bumped.
    ///
    /// The `embedding` column is left untouched (it is not part of version
    /// snapshots).  The returned [`MemoryItem`] has `embedding = None`.
    pub async fn rollback_to_version(
        &self,
        memory_id: Uuid,
        version_num: i32,
        changed_by: &str,
    ) -> Result<MemoryItem, VersionError> {
        // 1. Snapshot the current state before rolling back.
        self.create_version(memory_id, changed_by, Some("pre_rollback_snapshot"))
            .await?;

        // 2. Fetch the target version.
        let target = self.get_version(memory_id, version_num).await?;

        // 3. Deserialise the payload into a MemoryItem.
        let item = target.to_memory_item()?;

        // 4. Restore the memory_item from the old version's data, bump version.
        let row = sqlx::query(
            r#"UPDATE memory_item SET
                tenant_id = $2, memory_type = $3, scope = $4,
                subject_type = $5, subject_id = $6,
                entity_refs = $7, content = $8, structured_payload = $9,
                source_type = $10, source_reference = $11, evidence_refs = $12,
                confidence = $13, authority_level = $14, importance = $15,
                observed_at = $16, valid_from = $17, valid_to = $18,
                privacy_class = $19, acl = $20, retention_policy = $21,
                status = $22, derived_from = $23,
                created_by_user = $24, created_by_agent = $25,
                last_verified_at = $26,
                version = version + 1, updated_at = NOW()
               WHERE memory_id = $1
               RETURNING version, updated_at"#,
        )
        .bind(memory_id)
        .bind(&item.tenant_id)
        .bind(item.memory_type.as_str())
        .bind(item.scope.as_str())
        .bind(&item.subject_type)
        .bind(&item.subject_id)
        .bind(Value::Array(item.entity_refs.clone()))
        .bind(&item.content)
        .bind(&item.structured_payload)
        .bind(item.source_type.as_str())
        .bind(&item.source_reference)
        .bind(Value::Array(item.evidence_refs.clone()))
        .bind(item.confidence)
        .bind(item.authority_level.as_str())
        .bind(item.importance)
        .bind(item.observed_at)
        .bind(item.valid_from)
        .bind(item.valid_to)
        .bind(item.privacy_class.as_str())
        .bind(&item.acl)
        .bind(&item.retention_policy)
        .bind(item.status.as_str())
        .bind(Value::Array(item.derived_from.clone()))
        .bind(&item.created_by_user)
        .bind(&item.created_by_agent)
        .bind(item.last_verified_at)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(r) => {
                let new_version: i32 = r.try_get("version")?;
                let updated_at: DateTime<Utc> = r.try_get("updated_at")?;
                let mut restored = item;
                restored.version = new_version;
                restored.updated_at = updated_at;
                Ok(restored)
            }
            None => Err(VersionError::NotFound(memory_id)),
        }
    }
}

// ──────────────────────── Helpers ────────────────────────

/// Map a database row to a [`MemoryVersion`].
fn map_row_to_version(row: &sqlx::postgres::PgRow) -> Result<MemoryVersion, VersionError> {
    Ok(MemoryVersion {
        version_id: row.try_get("version_id")?,
        memory_id: row.try_get("memory_id")?,
        version_number: row.try_get("version")?,
        payload: row.try_get("payload")?,
        changed_by: row.try_get("changed_by")?,
        change_reason: row.try_get("change_reason")?,
        created_at: row.try_get("created_at")?,
    })
}

/// Convert a [`MemoryRepoError`] into a [`VersionError`], preserving the
/// `NotFound` context with the memory id.
fn convert_repo_error(memory_id: Uuid) -> impl Fn(MemoryRepoError) -> VersionError {
    move |e| match e {
        MemoryRepoError::NotFound => VersionError::NotFound(memory_id),
        MemoryRepoError::Database(db) => VersionError::Database(db),
    }
}

// ──────────────────────── Errors ────────────────────────

/// Errors returned by [`VersionManager`] operations.
#[derive(Debug, thiserror::Error)]
pub enum VersionError {
    /// Wraps a database error from `sqlx`.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    /// The memory item does not exist.
    #[error("memory item not found: {0}")]
    NotFound(Uuid),

    /// The requested version number does not exist for the given memory.
    #[error("version {version} not found for memory {memory_id}")]
    VersionNotFound { memory_id: Uuid, version: i32 },

    /// Failed to (de)serialise the version payload.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use oris_memory_store::{
        AuthorityLevel, MemoryStatus, MemoryType, PrivacyClass, Scope, SourceType,
    };
    use serde_json::json;

    /// Build a `MemoryItem` with test values for round-trip tests.
    fn test_memory_item() -> MemoryItem {
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "tenant-1".to_string(),
            memory_type: MemoryType::Semantic,
            scope: Scope::Personal,
            subject_type: Some("user".to_string()),
            subject_id: Some("user-123".to_string()),
            entity_refs: vec![json!({"type": "person", "id": "user-123"})],
            content: Some("test content".to_string()),
            structured_payload: Some(json!({"key": "value"})),
            embedding: None,
            source_type: SourceType::UserExplicit,
            source_reference: Some("ref-1".to_string()),
            evidence_refs: vec![json!({"ref": "evidence-1"})],
            confidence: 0.85,
            authority_level: AuthorityLevel::L1Authoritative,
            importance: 0.9,
            observed_at: Some(Utc::now()),
            valid_from: Some(Utc::now()),
            valid_to: None,
            privacy_class: PrivacyClass::Internal,
            acl: json!({"read": ["user-123"]}),
            retention_policy: Some("default".to_string()),
            status: MemoryStatus::Active,
            version: 3,
            derived_from: vec![json!("parent-uuid")],
            created_by_user: Some("user-123".to_string()),
            created_by_agent: None,
            last_verified_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn version_payload_roundtrip() {
        let item = test_memory_item();
        let payload = serde_json::to_value(&item).unwrap();
        let restored: MemoryItem = serde_json::from_value(payload).unwrap();

        assert_eq!(restored.memory_id, item.memory_id);
        assert_eq!(restored.tenant_id, item.tenant_id);
        assert_eq!(restored.memory_type, item.memory_type);
        assert_eq!(restored.scope, item.scope);
        assert_eq!(restored.subject_type, item.subject_type);
        assert_eq!(restored.content, item.content);
        assert_eq!(restored.confidence, item.confidence);
        assert_eq!(restored.authority_level, item.authority_level);
        assert_eq!(restored.importance, item.importance);
        assert_eq!(restored.privacy_class, item.privacy_class);
        assert_eq!(restored.status, item.status);
        assert_eq!(restored.version, item.version);
        assert_eq!(restored.created_by_user, item.created_by_user);
        assert_eq!(restored.derived_from, item.derived_from);
    }

    #[test]
    fn memory_version_to_memory_item() {
        let item = test_memory_item();
        let payload = serde_json::to_value(&item).unwrap();

        let version = MemoryVersion {
            version_id: Uuid::new_v4(),
            memory_id: item.memory_id,
            version_number: item.version,
            payload,
            changed_by: "admin".to_string(),
            change_reason: Some("test".to_string()),
            created_at: Utc::now(),
        };

        let restored = version.to_memory_item().unwrap();
        assert_eq!(restored.memory_id, item.memory_id);
        assert_eq!(restored.tenant_id, item.tenant_id);
        assert_eq!(restored.version, item.version);
        assert_eq!(restored.content, item.content);
        assert_eq!(restored.status, item.status);
        assert_eq!(restored.authority_level, item.authority_level);
    }

    #[test]
    fn to_memory_item_invalid_payload() {
        let version = MemoryVersion {
            version_id: Uuid::new_v4(),
            memory_id: Uuid::new_v4(),
            version_number: 1,
            payload: json!("not an object"),
            changed_by: "admin".to_string(),
            change_reason: None,
            created_at: Utc::now(),
        };
        assert!(version.to_memory_item().is_err());
    }

    #[test]
    fn version_error_display() {
        let id = Uuid::new_v4();

        let e = VersionError::NotFound(id);
        assert!(format!("{e}").contains("not found"));

        let e = VersionError::VersionNotFound {
            memory_id: id,
            version: 5,
        };
        let msg = format!("{e}");
        assert!(msg.contains("version 5"));
        assert!(msg.contains(&id.to_string()));
    }

    #[test]
    fn version_error_from_serde_json() {
        let serde_err = serde_json::from_str::<MemoryItem>("{}").unwrap_err();
        let v_err: VersionError = serde_err.into();
        assert!(format!("{v_err}").contains("serialization"));
    }
}
