//! Access audit logger — persists every access decision to the
//! `memory_access_audit` table.
//!
//! Each [`AuditEntry`] captures *who* accessed *what* memory, *why* (purpose),
//! under which trace, and whether the policy decision was allow or deny.
//! This provides the full audit trail required for compliance and forensic
//! analysis.

use chrono::{DateTime, Utc};
use oris_memory_store::postgres::Pool;
use sqlx::Row;
use uuid::Uuid;

/// A single audit-record row.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub audit_id: Uuid,
    pub memory_id: Option<Uuid>,
    pub accessor_user: String,
    pub accessor_agent: Option<String>,
    pub action: String,
    pub purpose: String,
    pub task_id: Option<Uuid>,
    pub trace_id: String,
    /// `"allow"` or `"deny"`.
    pub policy_decision: String,
    pub created_at: DateTime<Utc>,
}

impl AuditEntry {
    /// Create a builder with sensible defaults (new UUID, current timestamp).
    pub fn new(
        accessor_user: impl Into<String>,
        action: impl Into<String>,
        decision: &str,
    ) -> Self {
        Self {
            audit_id: Uuid::new_v4(),
            memory_id: None,
            accessor_user: accessor_user.into(),
            accessor_agent: None,
            action: action.into(),
            purpose: String::new(),
            task_id: None,
            trace_id: String::new(),
            policy_decision: decision.to_string(),
            created_at: Utc::now(),
        }
    }
}

/// Writes and reads audit entries from the `memory_access_audit` table.
pub struct AuditLogger {
    pool: Pool,
}

impl AuditLogger {
    /// Create a new logger backed by the given connection pool.
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Persist an audit entry.
    ///
    /// The `policy_decision` column in the table is `policy_decision_id`; the
    /// struct field `policy_decision` maps to that column.
    pub async fn record(&self, entry: &AuditEntry) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"INSERT INTO memory_access_audit (
                audit_id, memory_id, accessor_user, accessor_agent,
                action, purpose, task_id, trace_id, policy_decision_id, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"#,
        )
        .bind(entry.audit_id)
        .bind(entry.memory_id)
        .bind(&entry.accessor_user)
        .bind(&entry.accessor_agent)
        .bind(&entry.action)
        .bind(&entry.purpose)
        .bind(entry.task_id)
        .bind(&entry.trace_id)
        .bind(&entry.policy_decision)
        .bind(entry.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// List all audit entries for a specific memory, newest first.
    pub async fn list_by_memory(&self, memory_id: Uuid) -> Result<Vec<AuditEntry>, sqlx::Error> {
        let rows = sqlx::query(
            r#"SELECT audit_id, memory_id, accessor_user, accessor_agent,
                      action, purpose, task_id, trace_id, policy_decision_id, created_at
               FROM memory_access_audit
               WHERE memory_id = $1
               ORDER BY created_at DESC"#,
        )
        .bind(memory_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(map_row).collect())
    }

    /// List the most recent audit entries for a specific user.
    pub async fn list_by_user(
        &self,
        user_id: &str,
        limit: i64,
    ) -> Result<Vec<AuditEntry>, sqlx::Error> {
        let rows = sqlx::query(
            r#"SELECT audit_id, memory_id, accessor_user, accessor_agent,
                      action, purpose, task_id, trace_id, policy_decision_id, created_at
               FROM memory_access_audit
               WHERE accessor_user = $1
               ORDER BY created_at DESC
               LIMIT $2"#,
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(map_row).collect())
    }
}

/// Map a database row to an [`AuditEntry`].
///
/// `accessor_user` and `accessor_agent` are nullable in the schema; we coerce
/// `NULL` to an empty string / `None` respectively.
fn map_row(row: &sqlx::postgres::PgRow) -> AuditEntry {
    let accessor_user: Option<String> = row.try_get("accessor_user").unwrap_or(None);
    let purpose: Option<String> = row.try_get("purpose").unwrap_or(None);
    let trace_id: Option<String> = row.try_get("trace_id").unwrap_or(None);
    AuditEntry {
        audit_id: row.try_get("audit_id").unwrap_or_default(),
        memory_id: row.try_get("memory_id").unwrap_or(None),
        accessor_user: accessor_user.unwrap_or_default(),
        accessor_agent: row.try_get("accessor_agent").unwrap_or(None),
        action: row.try_get("action").unwrap_or_default(),
        purpose: purpose.unwrap_or_default(),
        task_id: row.try_get("task_id").unwrap_or(None),
        trace_id: trace_id.unwrap_or_default(),
        policy_decision: row
            .try_get::<Option<String>, _>("policy_decision_id")
            .unwrap_or(None)
            .unwrap_or_default(),
        created_at: row.try_get("created_at").unwrap_or_else(|_| Utc::now()),
    }
}
