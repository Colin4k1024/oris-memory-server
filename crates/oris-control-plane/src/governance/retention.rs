//! Lifecycle management — decay → archive → delete.
//!
//! The [`RetentionManager`] scans the `memory_item` table for records that
//! have reached their archive or delete thresholds (as defined by the
//! [`PolicyEngine`](crate::governance::policy::PolicyEngine)) and performs
//! the corresponding state transition.
//!
//! Lifecycle order:
//! 1. **Decay** — gradually reduce the `importance` score based on age.
//! 2. **Archive** — set `status = 'archived'` when `archive_after_days` is
//!    reached.
//! 3. **Delete** — hard-delete when `delete_after_days` is reached (or when
//!    the memory was revoked / forgotten).

use oris_memory_store::postgres::Pool;
use sqlx::Row;
use uuid::Uuid;

use super::policy::PolicyEngine;

/// Manages the retention lifecycle for memory items.
pub struct RetentionManager {
    pool: Pool,
    policy_engine: PolicyEngine,
}

impl RetentionManager {
    /// Create a new manager backed by the given pool and policy engine.
    pub fn new(pool: Pool, policy_engine: PolicyEngine) -> Self {
        Self {
            pool,
            policy_engine,
        }
    }

    /// Scan for active memories whose `archive_after_days` threshold has been
    /// reached, across all retention policies.
    ///
    /// Returns a de-duplicated list of memory IDs.
    pub async fn find_archivable(&self, limit: i64) -> Result<Vec<Uuid>, sqlx::Error> {
        let mut results = Vec::new();
        let mut remaining = limit;

        for policy in self.policy_engine.policies() {
            if remaining <= 0 {
                break;
            }
            if let Some(days) = policy.archive_after_days {
                let rows = sqlx::query(
                    r#"SELECT memory_id FROM memory_item
                       WHERE status = 'active'
                         AND scope = $1
                         AND memory_type = $2
                         AND created_at < NOW() - (INTERVAL '1 day' * $3)
                       LIMIT $4"#,
                )
                .bind(policy.scope.as_str())
                .bind(policy.memory_type.as_str())
                .bind(days as i64)
                .bind(remaining)
                .fetch_all(&self.pool)
                .await?;

                for row in &rows {
                    let id: Uuid = row.try_get("memory_id").unwrap_or_default();
                    if !results.contains(&id) {
                        results.push(id);
                        remaining -= 1;
                        if remaining <= 0 {
                            break;
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Scan for memories whose `delete_after_days` threshold has been reached,
    /// across all retention policies.  Includes active, archived, and revoked
    /// records.
    pub async fn find_deletable(&self, limit: i64) -> Result<Vec<Uuid>, sqlx::Error> {
        let mut results = Vec::new();
        let mut remaining = limit;

        for policy in self.policy_engine.policies() {
            if remaining <= 0 {
                break;
            }
            if let Some(days) = policy.delete_after_days {
                let rows = sqlx::query(
                    r#"SELECT memory_id FROM memory_item
                       WHERE status IN ('active', 'archived', 'revoked')
                         AND scope = $1
                         AND memory_type = $2
                         AND created_at < NOW() - (INTERVAL '1 day' * $3)
                       LIMIT $4"#,
                )
                .bind(policy.scope.as_str())
                .bind(policy.memory_type.as_str())
                .bind(days as i64)
                .bind(remaining)
                .fetch_all(&self.pool)
                .await?;

                for row in &rows {
                    let id: Uuid = row.try_get("memory_id").unwrap_or_default();
                    if !results.contains(&id) {
                        results.push(id);
                        remaining -= 1;
                        if remaining <= 0 {
                            break;
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Archive a memory by setting `status = 'archived'`.
    pub async fn archive(&self, memory_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"UPDATE memory_item
               SET status = 'archived', updated_at = NOW()
               WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Hard-delete a memory row.  Only call for expired or forgotten items.
    pub async fn delete(&self, memory_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query(r#"DELETE FROM memory_item WHERE memory_id = $1"#)
            .bind(memory_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Decay: multiply the `importance` score by `factor` (0.0–1.0).
    pub async fn decay(&self, memory_id: Uuid, factor: f32) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"UPDATE memory_item
               SET importance = importance * $2, updated_at = NOW()
               WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .bind(factor)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
