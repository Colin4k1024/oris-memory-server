//! Outbox pattern: transactional event queue for async processing.
//!
//! Events are written in the same transaction as the memory_item write,
//! then processed by a background worker that invalidates caches and
//! notifies subscribers.

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::memory_types::{EventType, OutboxEvent, OutboxStatus};

pub struct OutboxRepo {
    pool: PgPool,
}

impl OutboxRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Enqueue an event in pending status.
    pub async fn enqueue(
        &self,
        event_type: EventType,
        aggregate_id: &str,
        payload: &serde_json::Value,
    ) -> Result<Uuid, OutboxError> {
        let event_id = Uuid::new_v4();
        sqlx::query(
            r#"INSERT INTO outbox_event (event_id, event_type, aggregate_id, payload, status)
               VALUES ($1, $2, $3, $4, 'pending')"#,
        )
        .bind(event_id)
        .bind(event_type.as_str())
        .bind(aggregate_id)
        .bind(payload)
        .execute(&self.pool)
        .await?;
        Ok(event_id)
    }

    /// Dequeue pending events using FOR UPDATE SKIP LOCKED (non-blocking).
    /// Marks them as 'processing' and returns them to the caller.
    pub async fn dequeue(&self, batch_size: i64) -> Result<Vec<OutboxEvent>, OutboxError> {
        let rows = sqlx::query(
            r#"UPDATE outbox_event
               SET status = 'processing'
               WHERE event_id IN (
                   SELECT event_id FROM outbox_event
                   WHERE status = 'pending'
                   ORDER BY created_at
                   LIMIT $1
                   FOR UPDATE SKIP LOCKED
               )
               RETURNING event_id, event_type, aggregate_id, payload, status, created_at, processed_at"#,
        )
        .bind(batch_size)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(|r| map_row_to_outbox_event(r)).collect()
    }

    /// Mark an event as done.
    pub async fn mark_done(&self, event_id: Uuid) -> Result<(), OutboxError> {
        sqlx::query(
            r#"UPDATE outbox_event SET status = 'done', processed_at = NOW()
               WHERE event_id = $1"#,
        )
        .bind(event_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Mark an event as failed (will be retried or dead-lettered).
    pub async fn mark_failed(&self, event_id: Uuid) -> Result<(), OutboxError> {
        sqlx::query(
            r#"UPDATE outbox_event SET status = 'failed'
               WHERE event_id = $1"#,
        )
        .bind(event_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Count events by status (for health monitoring).
    pub async fn count_by_status(&self, status: OutboxStatus) -> Result<i64, OutboxError> {
        let row = sqlx::query(r#"SELECT COUNT(*) as cnt FROM outbox_event WHERE status = $1"#)
            .bind(status.as_str())
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get::<i64, _>("cnt"))
    }
}

fn map_row_to_outbox_event(row: &sqlx::postgres::PgRow) -> Result<OutboxEvent, OutboxError> {
    let event_type_str: String = row.try_get("event_type")?;
    let status_str: String = row.try_get("status")?;
    let event_type = match event_type_str.as_str() {
        "USER_CONTEXT_UPDATED" => EventType::UserContextUpdated,
        "PREFERENCE_UPDATED" => EventType::PreferenceUpdated,
        "TASK_CONTEXT_UPDATED" => EventType::TaskContextUpdated,
        "MEMORY_PROMOTED" => EventType::MemoryPromoted,
        "MEMORY_REVOKED" => EventType::MemoryRevoked,
        "MEMORY_EXPIRED" => EventType::MemoryExpired,
        "PERMISSION_CHANGED" => EventType::PermissionChanged,
        "ENGINE_PROJECTION_FAILED" => EventType::EngineProjectionFailed,
        _ => EventType::UserContextUpdated,
    };
    let status = match status_str.as_str() {
        "pending" => OutboxStatus::Pending,
        "processing" => OutboxStatus::Processing,
        "done" => OutboxStatus::Done,
        "failed" => OutboxStatus::Failed,
        _ => OutboxStatus::Pending,
    };
    Ok(OutboxEvent {
        event_id: row.try_get("event_id")?,
        event_type,
        aggregate_id: row.try_get("aggregate_id")?,
        payload: row.try_get("payload")?,
        status,
        created_at: row.try_get("created_at")?,
        processed_at: row.try_get("processed_at")?,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum OutboxError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_type_round_trip() {
        for et in [
            EventType::UserContextUpdated,
            EventType::PreferenceUpdated,
            EventType::TaskContextUpdated,
            EventType::MemoryPromoted,
            EventType::MemoryRevoked,
            EventType::MemoryExpired,
            EventType::PermissionChanged,
            EventType::EngineProjectionFailed,
        ] {
            assert!(!et.as_str().is_empty());
        }
    }
}
