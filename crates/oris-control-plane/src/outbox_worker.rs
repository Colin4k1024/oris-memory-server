//! Outbox Pattern Worker — async event processing for the control plane.
//!
//! Consumes events from the `outbox_event` table (written transactionally
//! alongside memory writes) and performs side-effects: Redis cache
//! invalidation and subscriber notification.
//!
//! The worker runs a poll-dequeue-process loop until a shutdown signal is
//! received.  Failed events are retried up to `max_retries` times before
//! being permanently marked as failed (dead-lettered).
//!
//! # Event Processing
//!
//! | Event Type             | Action                                          |
//! |------------------------|-------------------------------------------------|
//! | UserContextUpdated     | Invalidate `oris:hotctx:{tenant}:{user}`       |
//! | PreferenceUpdated      | Invalidate hot context + cache `*user*` pattern |
//! | TaskContextUpdated     | Invalidate `oris:task:{task_id}`                |
//! | MemoryPromoted         | Invalidate cache pattern for tenant scope        |
//! | MemoryRevoked          | Invalidate hot context for affected user        |
//! | MemoryExpired          | Same as revoked                                 |
//! | PermissionChanged      | Invalidate `oris:perm:{tenant}:{user}`         |
//! | EngineProjectionFailed | Log warning, no cache invalidation             |

use std::time::Duration;

use oris_memory_store::memory_types::{EventType, OutboxEvent};
use oris_memory_store::postgres::outbox::{OutboxError, OutboxRepo};
use oris_memory_store::postgres::Pool;
use oris_memory_store::redis::cache::CacheRepo;
use oris_memory_store::redis::hot_context::HotContextRepo;
use serde::de::Error as _;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Default batch size for each dequeue call.
const DEFAULT_BATCH_SIZE: i64 = 50;
/// Default poll interval between batches when the queue is empty.
const DEFAULT_POLL_INTERVAL_SECS: u64 = 2;
/// Default maximum retry attempts before dead-lettering.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Errors that can occur during outbox event processing.
#[derive(Debug, Error)]
pub enum OutboxWorkerError {
    /// Database error from the outbox repository.
    #[error("database error: {0}")]
    DbError(#[from] sqlx::Error),
    /// Redis error during cache invalidation.
    #[error("redis error: {0}")]
    RedisError(String),
    /// Serialization/deserialization error for event payloads.
    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

impl From<OutboxError> for OutboxWorkerError {
    fn from(e: OutboxError) -> Self {
        match e {
            OutboxError::Database(e) => OutboxWorkerError::DbError(e),
        }
    }
}

/// A cache invalidation action derived from an outbox event.
///
/// This is the pure dispatch result — it describes *what* should be
/// invalidated, without touching Redis.  Separating computation from
/// execution makes the dispatch logic unit-testable without a live
/// database or Redis connection.
#[derive(Debug, Clone, PartialEq, Eq)]
enum InvalidationAction {
    /// Invalidate `oris:hotctx:{tenant}:{user}` in Redis hot context.
    InvalidateUserContext { tenant_id: String, user_id: String },
    /// Invalidate `oris:task:{task_id}` in Redis hot context.
    InvalidateTaskContext { task_id: String },
    /// Invalidate `oris:perm:{tenant}:{user}`.  Implemented via user-context
    /// invalidation because `HotContextRepo` does not expose a dedicated
    /// permissions delete — the permission snapshot is rebuilt on the next
    /// read from PostgreSQL.
    InvalidatePermissions { tenant_id: String, user_id: String },
    /// Invalidate all cache entries matching a glob pattern (under the
    /// `oris:cache:` namespace).
    InvalidateCachePattern { pattern: String },
    /// Log a warning only — no cache invalidation needed.
    LogWarning,
}

/// Result of processing a single outbox event.
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub event_id: Uuid,
    pub success: bool,
    pub error: Option<String>,
}

/// Handles outbox events by invalidating caches and notifying subscribers.
///
/// The worker polls the `outbox_event` table for pending events, processes
/// them in batches, and marks each as done or failed.  Failed events are
/// retried up to `max_retries` times before being dead-lettered.
pub struct OutboxWorker {
    outbox: OutboxRepo,
    hot_context: Option<HotContextRepo>,
    cache: Option<CacheRepo>,
    batch_size: i64,
    poll_interval: Duration,
    max_retries: u32,
}

impl OutboxWorker {
    /// Create a new `OutboxWorker` from a PostgreSQL connection pool.
    pub fn new(pool: Pool) -> Self {
        Self {
            outbox: OutboxRepo::new(pool),
            hot_context: None,
            cache: None,
            batch_size: DEFAULT_BATCH_SIZE,
            poll_interval: Duration::from_secs(DEFAULT_POLL_INTERVAL_SECS),
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Attach a Redis hot context repo for cache invalidation.
    pub fn with_hot_context(mut self, repo: HotContextRepo) -> Self {
        self.hot_context = Some(repo);
        self
    }

    /// Attach a Redis cache repo for pattern-based invalidation.
    pub fn with_cache(mut self, repo: CacheRepo) -> Self {
        self.cache = Some(repo);
        self
    }

    /// Process a single event based on its type.
    ///
    /// Computes the invalidation actions from the event (pure dispatch),
    /// then executes them against the attached Redis repos.
    async fn process_event(&self, event: &OutboxEvent) -> Result<(), OutboxWorkerError> {
        let actions = compute_invalidation_actions(event)?;
        for action in actions {
            match action {
                InvalidationAction::InvalidateUserContext { tenant_id, user_id } => {
                    if let Some(hc) = &self.hot_context {
                        hc.invalidate_user_context(&user_id, &tenant_id)
                            .await
                            .map_err(|e| OutboxWorkerError::RedisError(e.to_string()))?;
                    } else {
                        debug!(
                            event_id = %event.event_id,
                            "no hot_context repo configured, skipping user context invalidation"
                        );
                    }
                }
                InvalidationAction::InvalidateTaskContext { task_id } => {
                    if let Some(hc) = &self.hot_context {
                        hc.invalidate_task_context(&task_id)
                            .await
                            .map_err(|e| OutboxWorkerError::RedisError(e.to_string()))?;
                    } else {
                        debug!(
                            event_id = %event.event_id,
                            "no hot_context repo configured, skipping task context invalidation"
                        );
                    }
                }
                InvalidationAction::InvalidatePermissions { tenant_id, user_id } => {
                    if let Some(hc) = &self.hot_context {
                        // HotContextRepo does not expose a dedicated permissions
                        // delete; invalidating the user context forces a rebuild
                        // with the updated permission snapshot on next read.
                        hc.invalidate_user_context(&user_id, &tenant_id)
                            .await
                            .map_err(|e| OutboxWorkerError::RedisError(e.to_string()))?;
                    } else {
                        debug!(
                            event_id = %event.event_id,
                            "no hot_context repo configured, skipping permission invalidation"
                        );
                    }
                }
                InvalidationAction::InvalidateCachePattern { pattern } => {
                    if let Some(cache) = &self.cache {
                        cache
                            .invalidate_pattern(&pattern)
                            .await
                            .map_err(|e| OutboxWorkerError::RedisError(e.to_string()))?;
                    } else {
                        debug!(
                            event_id = %event.event_id,
                            pattern = %pattern,
                            "no cache repo configured, skipping pattern invalidation"
                        );
                    }
                }
                InvalidationAction::LogWarning => {
                    warn!(
                        event_id = %event.event_id,
                        "engine projection failed — no cache invalidation required"
                    );
                }
            }
        }
        Ok(())
    }

    /// Process one batch of events.
    ///
    /// Dequeues up to `batch_size` pending events, processes each, and marks
    /// them done or failed.  Returns the result for each processed event.
    /// Useful for testing and one-shot processing.
    pub async fn process_batch(&self) -> Vec<ProcessingResult> {
        let events = match self.outbox.dequeue(self.batch_size).await {
            Ok(events) => events,
            Err(e) => {
                error!("failed to dequeue outbox events: {e}");
                return Vec::new();
            }
        };

        let mut results = Vec::with_capacity(events.len());
        for event in &events {
            let result = match self.process_event(event).await {
                Ok(()) => match self.outbox.mark_done(event.event_id).await {
                    Ok(()) => ProcessingResult {
                        event_id: event.event_id,
                        success: true,
                        error: None,
                    },
                    Err(e) => {
                        let msg = format!("mark_done failed: {e}");
                        error!(event_id = %event.event_id, "{msg}");
                        ProcessingResult {
                            event_id: event.event_id,
                            success: false,
                            error: Some(msg),
                        }
                    }
                },
                Err(e) => {
                    let error_msg = e.to_string();
                    if let Err(handle_err) = self.handle_failure(event, &error_msg).await {
                        warn!(
                            event_id = %event.event_id,
                            error = %handle_err,
                            "failed to handle event failure"
                        );
                    }
                    ProcessingResult {
                        event_id: event.event_id,
                        success: false,
                        error: Some(error_msg),
                    }
                }
            };
            results.push(result);
        }
        results
    }

    /// Handle a failed event: retry or dead-letter.
    ///
    /// Reads `retry_count` from the event payload (default 0).  If the count
    /// is below `max_retries`, re-enqueues the event with an incremented
    /// count and marks the original as done.  Otherwise, marks the event as
    /// permanently failed.
    async fn handle_failure(
        &self,
        event: &OutboxEvent,
        error: &str,
    ) -> Result<(), OutboxWorkerError> {
        let retry_count = extract_retry_count(&event.payload);

        if retry_count < self.max_retries {
            let new_payload = augment_retry_count(&event.payload, retry_count + 1);
            warn!(
                event_id = %event.event_id,
                retry_count = retry_count,
                error = %error,
                "re-enqueuing event for retry"
            );
            self.outbox
                .enqueue(event.event_type, &event.aggregate_id, &new_payload)
                .await?;
            self.outbox.mark_done(event.event_id).await?;
        } else {
            warn!(
                event_id = %event.event_id,
                retry_count = retry_count,
                max_retries = self.max_retries,
                error = %error,
                "event permanently failed after max retries"
            );
            self.outbox.mark_failed(event.event_id).await?;
        }
        Ok(())
    }

    /// Main processing loop.
    ///
    /// Polls the outbox table, processes batches, and sleeps between polls
    /// when the queue is empty.  Runs until the shutdown signal (`true` on
    /// the watch channel) is received or the sender is dropped.
    pub async fn run(&self, mut shutdown: watch::Receiver<bool>) {
        info!(
            batch_size = self.batch_size,
            poll_interval = ?self.poll_interval,
            max_retries = self.max_retries,
            "outbox worker started"
        );

        loop {
            if *shutdown.borrow() {
                info!("shutdown signal received, stopping outbox worker");
                break;
            }

            let results = self.process_batch().await;
            if !results.is_empty() {
                let success_count = results.iter().filter(|r| r.success).count();
                let fail_count = results.len() - success_count;
                debug!(
                    total = results.len(),
                    success = success_count,
                    failed = fail_count,
                    "batch processed"
                );
            }

            // If we processed events, loop immediately for the next batch.
            // Only sleep when the queue was empty to avoid busy-looping.
            if results.is_empty() {
                tokio::select! {
                    _ = tokio::time::sleep(self.poll_interval) => {}
                    res = shutdown.changed() => {
                        match res {
                            Ok(()) => {
                                if *shutdown.borrow() {
                                    info!(
                                        "shutdown signal received during sleep, \
                                         stopping outbox worker"
                                    );
                                    break;
                                }
                            }
                            Err(_) => {
                                info!("shutdown sender dropped, stopping outbox worker");
                                break;
                            }
                        }
                    }
                }
            }
        }

        info!("outbox worker stopped");
    }
}

// ─────────────────────── Pure helper functions ───────────────────────

/// Extract a string field from a JSON payload.
///
/// Returns `SerializationError` if the field is missing or not a string.
fn payload_str(payload: &Value, field: &str) -> Result<String, OutboxWorkerError> {
    payload
        .get(field)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| {
            OutboxWorkerError::SerializationError(serde_json::Error::custom(format!(
                "missing or invalid field: {field}"
            )))
        })
}

/// Extract the retry count from an event payload.
///
/// Reads the `retry_count` field as a u64; defaults to 0 if absent.
fn extract_retry_count(payload: &Value) -> u32 {
    payload
        .get("retry_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

/// Return a new payload JSON with `retry_count` set to the given value.
///
/// If the original payload is a JSON object, the field is inserted (or
/// replaced).  If the payload is not an object, it is wrapped under an
/// `original_payload` key alongside `retry_count`.
fn augment_retry_count(payload: &Value, new_count: u32) -> Value {
    match payload {
        Value::Object(map) => {
            let mut new_map = map.clone();
            new_map.insert("retry_count".into(), serde_json::Value::from(new_count));
            Value::Object(new_map)
        }
        _ => {
            let mut wrapper = serde_json::Map::new();
            wrapper.insert("original_payload".into(), payload.clone());
            wrapper.insert("retry_count".into(), serde_json::Value::from(new_count));
            Value::Object(wrapper)
        }
    }
}

/// Compute the cache invalidation actions for an outbox event.
///
/// This is the pure dispatch logic — it maps an event type and payload to
/// a list of [`InvalidationAction`]s without touching Redis or the
/// database.  Unit-testable in isolation.
fn compute_invalidation_actions(
    event: &OutboxEvent,
) -> Result<Vec<InvalidationAction>, OutboxWorkerError> {
    let mut actions = Vec::new();
    match event.event_type {
        EventType::UserContextUpdated => {
            let tenant_id = payload_str(&event.payload, "tenant_id")?;
            let user_id = payload_str(&event.payload, "user_id")?;
            actions.push(InvalidationAction::InvalidateUserContext { tenant_id, user_id });
        }
        EventType::PreferenceUpdated => {
            let tenant_id = payload_str(&event.payload, "tenant_id")?;
            let user_id = payload_str(&event.payload, "user_id")?;
            let pattern = format!("*{user_id}*");
            actions.push(InvalidationAction::InvalidateUserContext { tenant_id, user_id });
            actions.push(InvalidationAction::InvalidateCachePattern { pattern });
        }
        EventType::TaskContextUpdated => {
            let task_id = payload_str(&event.payload, "task_id")?;
            actions.push(InvalidationAction::InvalidateTaskContext { task_id });
        }
        EventType::MemoryPromoted => {
            let tenant_id = payload_str(&event.payload, "tenant_id")?;
            actions.push(InvalidationAction::InvalidateCachePattern {
                pattern: format!("tenant:{tenant_id}*"),
            });
        }
        EventType::MemoryRevoked | EventType::MemoryExpired => {
            let tenant_id = payload_str(&event.payload, "tenant_id")?;
            let user_id = payload_str(&event.payload, "user_id")?;
            actions.push(InvalidationAction::InvalidateUserContext { tenant_id, user_id });
        }
        EventType::PermissionChanged => {
            let tenant_id = payload_str(&event.payload, "tenant_id")?;
            let user_id = payload_str(&event.payload, "user_id")?;
            actions.push(InvalidationAction::InvalidatePermissions { tenant_id, user_id });
        }
        EventType::EngineProjectionFailed => {
            actions.push(InvalidationAction::LogWarning);
        }
        EventType::MemoryProposed
        | EventType::MemoryQuarantined
        | EventType::ExperienceOutcomeRecorded => {
            let tenant_id =
                payload_str(&event.payload, "tenant_id").unwrap_or_else(|_| "default".to_string());
            actions.push(InvalidationAction::InvalidateCachePattern {
                pattern: format!("tenant:{tenant_id}*"),
            });
        }
    }
    Ok(actions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use oris_memory_store::memory_types::OutboxStatus;
    use uuid::Uuid;

    fn make_event(event_type: EventType, payload: &Value) -> OutboxEvent {
        OutboxEvent {
            event_id: Uuid::new_v4(),
            event_type,
            aggregate_id: "test-aggregate".to_string(),
            payload: payload.clone(),
            status: OutboxStatus::Processing,
            created_at: Utc::now(),
            processed_at: None,
        }
    }

    #[test]
    fn user_context_updated_invalidates_hot_context() {
        let payload = serde_json::json!({
            "tenant_id": "acme",
            "user_id": "user-42",
        });
        let event = make_event(EventType::UserContextUpdated, &payload);
        let actions = compute_invalidation_actions(&event).unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            InvalidationAction::InvalidateUserContext {
                tenant_id: "acme".into(),
                user_id: "user-42".into(),
            }
        );
    }

    #[test]
    fn preference_updated_invalidates_hot_context_and_cache() {
        let payload = serde_json::json!({
            "tenant_id": "acme",
            "user_id": "user-42",
        });
        let event = make_event(EventType::PreferenceUpdated, &payload);
        let actions = compute_invalidation_actions(&event).unwrap();

        assert_eq!(actions.len(), 2);
        assert!(
            actions.contains(&InvalidationAction::InvalidateUserContext {
                tenant_id: "acme".into(),
                user_id: "user-42".into(),
            })
        );
        assert!(
            actions.contains(&InvalidationAction::InvalidateCachePattern {
                pattern: "*user-42*".into(),
            })
        );
    }

    #[test]
    fn task_context_updated_invalidates_task() {
        let payload = serde_json::json!({
            "task_id": "task-99",
        });
        let event = make_event(EventType::TaskContextUpdated, &payload);
        let actions = compute_invalidation_actions(&event).unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            InvalidationAction::InvalidateTaskContext {
                task_id: "task-99".into(),
            }
        );
    }

    #[test]
    fn engine_projection_failed_logs_warning_only() {
        let payload = serde_json::json!({
            "engine_id": "vector-engine",
            "reason": "embedding model unavailable",
        });
        let event = make_event(EventType::EngineProjectionFailed, &payload);
        let actions = compute_invalidation_actions(&event).unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], InvalidationAction::LogWarning);
    }

    #[test]
    fn permission_changed_invalidates_permissions() {
        let payload = serde_json::json!({
            "tenant_id": "acme",
            "user_id": "user-42",
        });
        let event = make_event(EventType::PermissionChanged, &payload);
        let actions = compute_invalidation_actions(&event).unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            InvalidationAction::InvalidatePermissions {
                tenant_id: "acme".into(),
                user_id: "user-42".into(),
            }
        );
    }

    #[test]
    fn memory_promoted_invalidates_tenant_cache_pattern() {
        let payload = serde_json::json!({
            "tenant_id": "acme",
            "memory_id": "mem-1",
        });
        let event = make_event(EventType::MemoryPromoted, &payload);
        let actions = compute_invalidation_actions(&event).unwrap();

        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            InvalidationAction::InvalidateCachePattern {
                pattern: "tenant:acme*".into(),
            }
        );
    }

    #[test]
    fn revoked_and_expired_both_invalidate_user_context() {
        let payload = serde_json::json!({
            "tenant_id": "acme",
            "user_id": "user-42",
            "memory_id": "mem-1",
        });

        for et in [EventType::MemoryRevoked, EventType::MemoryExpired] {
            let event = make_event(et, &payload);
            let actions = compute_invalidation_actions(&event).unwrap();
            assert_eq!(actions.len(), 1);
            assert_eq!(
                actions[0],
                InvalidationAction::InvalidateUserContext {
                    tenant_id: "acme".into(),
                    user_id: "user-42".into(),
                }
            );
        }
    }

    #[test]
    fn missing_field_returns_serialization_error() {
        let payload = serde_json::json!({ "user_id": "user-42" }); // no tenant_id
        let event = make_event(EventType::UserContextUpdated, &payload);
        let result = compute_invalidation_actions(&event);
        assert!(matches!(
            result,
            Err(OutboxWorkerError::SerializationError(_))
        ));
    }

    #[test]
    fn retry_count_extraction_defaults_to_zero() {
        let payload = serde_json::json!({ "tenant_id": "acme", "user_id": "u1" });
        assert_eq!(extract_retry_count(&payload), 0);

        let payload_with_retry = serde_json::json!({
            "tenant_id": "acme",
            "retry_count": 2,
        });
        assert_eq!(extract_retry_count(&payload_with_retry), 2);
    }

    #[test]
    fn augment_retry_count_preserves_payload_fields() {
        let payload = serde_json::json!({
            "tenant_id": "acme",
            "user_id": "u1",
        });
        let augmented = augment_retry_count(&payload, 1);
        assert_eq!(augmented["retry_count"], serde_json::json!(1));
        assert_eq!(augmented["tenant_id"], serde_json::json!("acme"));
        assert_eq!(augmented["user_id"], serde_json::json!("u1"));

        // Second augmentation increments further.
        let augmented2 = augment_retry_count(&augmented, 2);
        assert_eq!(augmented2["retry_count"], serde_json::json!(2));
    }
}
