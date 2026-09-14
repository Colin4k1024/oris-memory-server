//! Oris Outbox Worker — background event processor.
//!
//! Polls the `outbox_event` table for pending events and processes them:
//! - `MEMORY_PROPOSED` — trigger async embedding generation
//! - `MEMORY_PROMOTED` — cache invalidation
//! - `MEMORY_REVOKED` — cache invalidation + cascade cleanup
//! - `TASK_CONTEXT_UPDATED` — hot context refresh
//! - `USER_CONTEXT_UPDATED` — hot context refresh
//!
//! Usage:
//! ```bash
//! DATABASE_URL=postgres://localhost/oris_memory cargo run --example outbox_worker
//! ```
//!
//! The worker runs in a loop with a configurable poll interval (default 2s).

use std::sync::Arc;
use std::time::Duration;

use oris_memory_contract::memory_types::{EventType, OutboxEvent};
use oris_memory_store::postgres::{OutboxRepo, Pool};
use tracing::{info, warn};
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost/oris_memory".into());

    let pool: Pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;

    let repo = OutboxRepo::new(pool.clone());
    let poll_interval = Duration::from_secs(
        std::env::var("OUTBOX_POLL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2),
    );
    let batch_size = std::env::var("OUTBOX_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    info!(?poll_interval, batch_size, "Outbox worker started");

    loop {
        match repo.dequeue(batch_size).await {
            Ok(events) if events.is_empty() => {
                tokio::time::sleep(poll_interval).await;
            }
            Ok(events) => {
                info!(count = events.len(), "processing outbox batch");
                for event in events {
                    if let Err(e) = process_event(&repo, &event).await {
                        warn!(event_id = %event.event_id, error = %e, "failed to process event");
                        let _ = repo.mark_failed(event.event_id).await;
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "dequeue failed — backing off");
                tokio::time::sleep(poll_interval).await;
            }
        }
    }
}

async fn process_event(repo: &OutboxRepo, event: &OutboxEvent) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(
        event_id = %event.event_id,
        event_type = event.event_type.as_str(),
        aggregate_id = %event.aggregate_id,
        "processing event"
    );

    match event.event_type {
        EventType::MemoryProposed => {
            // In production: trigger embedding generation via an embedding service.
            // For now: log and mark done.
            info!(
                memory_id = %event.aggregate_id,
                "MEMORY_PROPOSED — would trigger embedding generation"
            );
        }
        EventType::MemoryPromoted => {
            info!(
                memory_id = %event.aggregate_id,
                "MEMORY_PROMOTED — would invalidate hot context cache"
            );
        }
        EventType::MemoryRevoked => {
            info!(
                memory_id = %event.aggregate_id,
                "MEMORY_REVOKED — would invalidate cache + cleanup derived"
            );
        }
        EventType::MemoryExpired => {
            info!(
                memory_id = %event.aggregate_id,
                "MEMORY_EXPIRED — would mark as expired"
            );
        }
        EventType::TaskContextUpdated => {
            info!(
                task_id = %event.aggregate_id,
                "TASK_CONTEXT_UPDATED — would refresh hot context"
            );
        }
        EventType::UserContextUpdated | EventType::PreferenceUpdated => {
            info!(
                user_id = %event.aggregate_id,
                "USER_CONTEXT_UPDATED — would refresh user hot context"
            );
        }
        EventType::PermissionChanged => {
            info!(
                entity = %event.aggregate_id,
                "PERMISSION_CHANGED — would refresh permission snapshot"
            );
        }
        EventType::EngineProjectionFailed => {
            warn!(
                aggregate = %event.aggregate_id,
                "ENGINE_PROJECTION_FAILED — circuit breaker may be open"
            );
        }
        _ => {
            info!(event_type = event.event_type.as_str(), "unhandled event type — marking done");
        }
    }

    repo.mark_done(event.event_id).await?;
    Ok(())
}
