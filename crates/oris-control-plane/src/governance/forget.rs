//! Forget / delete operations — GDPR right-to-be-forgotten, soft & hard
//! delete, and cascade deletion of derived memories.
//!
//! - [`ForgetManager::soft_delete`] — mark a memory as `revoked`, log to
//!   audit, and enqueue an outbox `MemoryRevoked` event.
//! - [`ForgetManager::hard_delete`] — permanently remove a memory and all
//!   related data (version entries, outbox events, and the memory row
//!   itself which carries the HNSW vector index entry).
//! - [`ForgetManager::forget_user_data`] — bulk soft-delete of every memory
//!   created by a given user within a tenant.
//! - [`ForgetManager::cascade_forget`] — recursively soft-delete a memory and
//!   all memories whose `derived_from` array references it, depth-limited to
//!   [`MAX_CASCADE_DEPTH`] to prevent infinite loops in circular chains.

use std::collections::HashSet;

use oris_memory_store::memory_types::EventType;
use oris_memory_store::postgres::{OutboxError, OutboxRepo, Pool};
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use super::audit::{AuditEntry, AuditLogger};

/// Maximum recursion depth for [`ForgetManager::cascade_forget`].
///
/// The root memory is depth 0; its direct descendants are depth 1, and so
/// on.  Processing stops when `depth >= MAX_CASCADE_DEPTH`, giving five
/// levels of cascade (depths 0–4).
const MAX_CASCADE_DEPTH: u32 = 5;

/// Manages forget/delete operations on memory items.
pub struct ForgetManager {
    pool: Pool,
}

impl ForgetManager {
    /// Create a new manager backed by the given connection pool.
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Soft-delete: mark a memory as `revoked`, log to audit, and enqueue a
    /// `MemoryRevoked` outbox event.
    ///
    /// The memory row is preserved (only `status` changes) so it can be
    /// recovered or audited later.
    pub async fn soft_delete(
        &self,
        memory_id: Uuid,
        requested_by: &str,
    ) -> Result<(), ForgetError> {
        // 1. Revoke the memory item.
        let result = sqlx::query(
            r#"UPDATE memory_item SET status = 'revoked', updated_at = NOW()
               WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(ForgetError::NotFound(memory_id));
        }

        // 2. Log to audit (best-effort — failure surfaces as an error but the
        //    memory is already revoked).
        let entry = build_forget_audit_entry(memory_id, requested_by, "soft_delete");
        let logger = AuditLogger::new(self.pool.clone());
        logger.record(&entry).await?;

        // 3. Enqueue outbox event.
        let outbox = OutboxRepo::new(self.pool.clone());
        let aggregate_id = memory_id.to_string();
        let payload = build_forget_outbox_payload(memory_id, requested_by);
        let _ = outbox
            .enqueue(EventType::MemoryRevoked, &aggregate_id, &payload)
            .await?;

        Ok(())
    }

    /// Hard-delete: permanently remove a memory and all related data.
    ///
    /// Deletes `memory_version` entries, `outbox_event` entries, and the
    /// `memory_item` row itself (which also removes its HNSW vector index
    /// entry).  The audit log entry is written *before* deletion so the
    /// trail survives even if the memory is gone.
    pub async fn hard_delete(
        &self,
        memory_id: Uuid,
        requested_by: &str,
    ) -> Result<(), ForgetError> {
        // 1. Log to audit first (before any deletion).
        let entry = build_forget_audit_entry(memory_id, requested_by, "hard_delete");
        let logger = AuditLogger::new(self.pool.clone());
        logger.record(&entry).await?;

        // 2. Delete version entries.
        sqlx::query(r#"DELETE FROM memory_version WHERE memory_id = $1"#)
            .bind(memory_id)
            .execute(&self.pool)
            .await?;

        // 3. Delete outbox events for this aggregate.
        let aggregate_id = memory_id.to_string();
        sqlx::query(r#"DELETE FROM outbox_event WHERE aggregate_id = $1"#)
            .bind(aggregate_id)
            .execute(&self.pool)
            .await?;

        // 4. Delete the memory_item row (removes the HNSW index entry too).
        let result = sqlx::query(r#"DELETE FROM memory_item WHERE memory_id = $1"#)
            .bind(memory_id)
            .execute(&self.pool)
            .await?;

        if result.rows_affected() == 0 {
            return Err(ForgetError::NotFound(memory_id));
        }

        Ok(())
    }

    /// GDPR right-to-be-forgotten: soft-delete ALL memories created by a user
    /// within a tenant.
    ///
    /// Sets `status = 'revoked'` for every matching row.  Returns the number
    /// of affected rows.  Per-memory audit and outbox events are not emitted
    /// in this bulk path; callers that need them should iterate and call
    /// [`soft_delete`](Self::soft_delete) individually.
    pub async fn forget_user_data(
        &self,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<u64, ForgetError> {
        let result = sqlx::query(
            r#"UPDATE memory_item SET status = 'revoked', updated_at = NOW()
               WHERE created_by_user = $1 AND tenant_id = $2"#,
        )
        .bind(user_id)
        .bind(tenant_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    /// Cascade forget: soft-delete a memory and all memories whose
    /// `derived_from` array references it, recursively, up to
    /// [`MAX_CASCADE_DEPTH`] levels.
    ///
    /// Already-revoked memories are skipped.  A `visited` set prevents
    /// infinite loops in circular `derived_from` chains.  Returns the total
    /// number of memories soft-deleted.
    pub async fn cascade_forget(
        &self,
        memory_id: Uuid,
        requested_by: &str,
    ) -> Result<u64, ForgetError> {
        let mut count: u64 = 0;
        let mut visited = HashSet::new();
        let mut stack: Vec<(Uuid, u32)> = vec![(memory_id, 0)];

        while let Some((current_id, depth)) = stack.pop() {
            if !should_cascade_at_depth(depth) {
                continue;
            }
            if !visited.insert(current_id) {
                continue;
            }

            // Find derived memories before soft-deleting.
            let children = self.find_derived_memories(current_id).await?;

            // Soft-delete this memory (skip if already gone).
            match self.soft_delete(current_id, requested_by).await {
                Ok(()) => count += 1,
                Err(ForgetError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }

            for child_id in children {
                stack.push((child_id, depth + 1));
            }
        }

        Ok(count)
    }

    /// Find all non-revoked memory_ids whose `derived_from` JSONB array
    /// contains the given `memory_id` (as a string).
    async fn find_derived_memories(&self, memory_id: Uuid) -> Result<Vec<Uuid>, ForgetError> {
        let needle = json!([memory_id.to_string()]);
        let rows = sqlx::query(
            r#"SELECT memory_id FROM memory_item
               WHERE derived_from @> $1::jsonb
                 AND status != 'revoked'"#,
        )
        .bind(needle)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .filter_map(|r| r.try_get::<Uuid, _>("memory_id").ok())
            .collect())
    }
}

// ──────────────────────── Pure helpers ────────────────────────

/// Returns `true` if the cascade should process children at the given depth.
fn should_cascade_at_depth(depth: u32) -> bool {
    depth < MAX_CASCADE_DEPTH
}

/// Build an [`AuditEntry`] for a forget operation.
fn build_forget_audit_entry(memory_id: Uuid, requested_by: &str, action: &str) -> AuditEntry {
    let mut entry = AuditEntry::new(requested_by, action, "allow");
    entry.memory_id = Some(memory_id);
    entry.purpose = "right_to_be_forgotten".to_string();
    entry
}

/// Build the outbox event payload for a forget/revoke operation.
fn build_forget_outbox_payload(memory_id: Uuid, requested_by: &str) -> Value {
    json!({
        "memory_id": memory_id.to_string(),
        "action": "revoke",
        "requested_by": requested_by,
    })
}

// ──────────────────────── Errors ────────────────────────

/// Errors returned by [`ForgetManager`] operations.
#[derive(Debug, thiserror::Error)]
pub enum ForgetError {
    /// Wraps a database error from `sqlx`.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    /// The memory item does not exist.
    #[error("memory item not found: {0}")]
    NotFound(Uuid),

    /// An outbox enqueue failed.
    #[error("outbox enqueue failed: {0}")]
    Outbox(#[from] OutboxError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn should_cascade_at_depth_respects_limit() {
        assert!(should_cascade_at_depth(0));
        assert!(should_cascade_at_depth(1));
        assert!(should_cascade_at_depth(4));
        assert!(!should_cascade_at_depth(5));
        assert!(!should_cascade_at_depth(100));
    }

    /// Pure BFS that mirrors [`ForgetManager::cascade_forget`] but uses a
    /// static adjacency map instead of DB queries — lets us unit-test the
    /// depth-limit and cycle-handling logic without a live database.
    fn compute_cascade_set(
        root: Uuid,
        adjacency: &HashMap<Uuid, Vec<Uuid>>,
        max_depth: u32,
    ) -> Vec<Uuid> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack: Vec<(Uuid, u32)> = vec![(root, 0)];

        while let Some((id, depth)) = stack.pop() {
            if depth >= max_depth {
                continue;
            }
            if !visited.insert(id) {
                continue;
            }
            result.push(id);
            if let Some(children) = adjacency.get(&id) {
                for &child in children {
                    stack.push((child, depth + 1));
                }
            }
        }

        result
    }

    #[test]
    fn cascade_depth_limit_stops_at_five() {
        // Chain: a → b → c → d → e → f → g  (7 nodes, depth 0–6)
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let d = Uuid::new_v4();
        let e = Uuid::new_v4();
        let f = Uuid::new_v4();
        let g = Uuid::new_v4();

        let adjacency = HashMap::from([
            (a, vec![b]),
            (b, vec![c]),
            (c, vec![d]),
            (d, vec![e]),
            (e, vec![f]),
            (f, vec![g]),
            (g, vec![]),
        ]);

        let result = compute_cascade_set(a, &adjacency, MAX_CASCADE_DEPTH);

        // Depths 0–4 are processed (5 nodes); f (depth 5) and g (depth 6) excluded.
        assert_eq!(result.len(), 5);
        assert!(result.contains(&a));
        assert!(result.contains(&b));
        assert!(result.contains(&c));
        assert!(result.contains(&d));
        assert!(result.contains(&e));
        assert!(!result.contains(&f));
        assert!(!result.contains(&g));
    }

    #[test]
    fn cascade_handles_cycles() {
        // Cycle: a → b → c → a
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        let adjacency = HashMap::from([(a, vec![b]), (b, vec![c]), (c, vec![a])]);

        let result = compute_cascade_set(a, &adjacency, MAX_CASCADE_DEPTH);

        // Each node processed exactly once despite the cycle.
        assert_eq!(result.len(), 3);
        assert!(result.contains(&a));
        assert!(result.contains(&b));
        assert!(result.contains(&c));
    }

    #[test]
    fn cascade_handles_branching() {
        //       a
        //      / \
        //     b   c
        //    /     \
        //   d       e
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let d = Uuid::new_v4();
        let e = Uuid::new_v4();

        let adjacency = HashMap::from([
            (a, vec![b, c]),
            (b, vec![d]),
            (c, vec![e]),
            (d, vec![]),
            (e, vec![]),
        ]);

        let result = compute_cascade_set(a, &adjacency, MAX_CASCADE_DEPTH);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn build_forget_audit_entry_fields() {
        let memory_id = Uuid::new_v4();
        let entry = build_forget_audit_entry(memory_id, "admin", "hard_delete");

        assert_eq!(entry.memory_id, Some(memory_id));
        assert_eq!(entry.accessor_user, "admin");
        assert_eq!(entry.action, "hard_delete");
        assert_eq!(entry.policy_decision, "allow");
        assert_eq!(entry.purpose, "right_to_be_forgotten");
    }

    #[test]
    fn build_forget_outbox_payload_structure() {
        let memory_id = Uuid::new_v4();
        let payload = build_forget_outbox_payload(memory_id, "admin");

        assert_eq!(payload["memory_id"], memory_id.to_string());
        assert_eq!(payload["action"], "revoke");
        assert_eq!(payload["requested_by"], "admin");
    }

    #[test]
    fn forget_error_display() {
        let id = Uuid::new_v4();
        let e = ForgetError::NotFound(id);
        assert!(format!("{e}").contains("not found"));
        assert!(format!("{e}").contains(&id.to_string()));
    }
}
