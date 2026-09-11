//! Resilience policy layer — per-engine-type timeout policy, SLO event
//! recording, and health-aware routing integration.
//!
//! Wraps the [`crate::engine::EngineRegistry`] concept with configurable
//! per-engine-type timeouts (§8.5.8), SLO event recording for downstream
//! monitoring (#19), and health-aware routing so the
//! [`crate::context_router::ContextRouter`] can skip unhealthy engines.
//!
//! ## Architecture references
//!
//! - §8.5.8 — Standard recall 150-250ms timeout; per-engine independent
//!   timeouts so a slow engine never blocks the PostgreSQL baseline.
//! - §8.4 — Latency budget per stage.

use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, instrument, warn};

// ─────────────────────────────────────────────────────────────────────
// Engine Timeout Configuration (§8.5.8)
// ─────────────────────────────────────────────────────────────────────

/// Per-engine-type timeout configuration (§8.5.8).
///
/// Each engine type gets an independent timeout so that a slow engine
/// never blocks the canonical PostgreSQL baseline recall path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineTimeoutConfig {
    /// Standard recall timeout (PostgreSQL baseline): 150-250ms.
    pub standard_recall_ms: u64,
    /// Mem0 engine timeout: 50-150ms.
    pub mem0_ms: u64,
    /// Cognee engine timeout: 200-800ms.
    pub cognee_ms: u64,
    /// Graphiti engine timeout: 200-800ms.
    pub graphiti_ms: u64,
    /// Failure threshold before circuit opens.
    pub failure_threshold: u32,
    /// Cooldown before half-open probe (seconds).
    pub cooldown_secs: u64,
}

impl Default for EngineTimeoutConfig {
    fn default() -> Self {
        Self {
            standard_recall_ms: 200,
            mem0_ms: 100,
            cognee_ms: 500,
            graphiti_ms: 500,
            failure_threshold: 5,
            cooldown_secs: 30,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// SLO Event Recording
// ─────────────────────────────────────────────────────────────────────

/// The kind of SLO-impacting event recorded by the resilience layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SloEventType {
    /// An engine call exceeded its timeout.
    Timeout,
    /// Circuit breaker transitioned to Open.
    CircuitOpen,
    /// Circuit breaker transitioned to HalfOpen (probe).
    CircuitHalfOpen,
    /// Circuit breaker transitioned back to Closed.
    CircuitClosed,
    /// A query completed but exceeded the slow-query threshold.
    SlowQuery,
    /// An engine recovered from a degraded/unreachable state.
    EngineRecovered,
}

/// An event recorded for SLO monitoring. Issue #19 will consume these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloEvent {
    pub event_type: SloEventType,
    pub engine_name: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration_ms: Option<u64>,
    pub detail: String,
}

/// Trait for SLO event sinks (allows mock testing without a real monitoring system).
#[async_trait]
pub trait SloEventSink: Send + Sync {
    async fn record(&self, event: SloEvent);
}

/// No-op sink for testing.
pub struct NoopSloSink;

#[async_trait]
impl SloEventSink for NoopSloSink {
    async fn record(&self, _event: SloEvent) {}
}

// ─────────────────────────────────────────────────────────────────────
// Engine Health Status
// ─────────────────────────────────────────────────────────────────────

/// Health status tracked by the resilience manager for routing decisions.
///
/// This mirrors [`crate::engine::EngineHealth`] but is owned by the
/// resilience layer so that health-aware routing decisions are decoupled
/// from individual engine self-reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineHealthStatus {
    Healthy,
    Degraded,
    Unreachable,
}

// ─────────────────────────────────────────────────────────────────────
// Resilience Manager
// ─────────────────────────────────────────────────────────────────────

/// Wraps EngineRegistry with configurable timeouts, SLO recording, and
/// health-aware routing.
///
/// The manager does **not** own the registry itself — it is a policy layer
/// that the registry (or a caller orchestrating the registry) consults to
/// determine per-engine timeouts, circuit-breaker parameters, and which
/// engines are safe to route to.
pub struct ResilienceManager {
    config: EngineTimeoutConfig,
    slo_sink: Arc<dyn SloEventSink>,
    /// Track engine health for routing decisions.
    engine_health: RwLock<HashMap<String, EngineHealthStatus>>,
}

impl ResilienceManager {
    /// Create a new resilience manager.
    pub fn new(config: EngineTimeoutConfig, slo_sink: Arc<dyn SloEventSink>) -> Self {
        Self {
            config,
            slo_sink,
            engine_health: RwLock::new(HashMap::new()),
        }
    }

    /// Get the timeout for a specific engine name based on its type.
    ///
    /// Matching is case-insensitive against known patterns:
    /// - Contains "mem0"     → `mem0_ms`
    /// - Contains "cognee"   → `cognee_ms`
    /// - Contains "graphiti" → `graphiti_ms`
    /// - Default (PostgreSQL baseline) → `standard_recall_ms`
    pub fn timeout_for(&self, engine_name: &str) -> Duration {
        let lower = engine_name.to_lowercase();
        if lower.contains("mem0") {
            Duration::from_millis(self.config.mem0_ms)
        } else if lower.contains("cognee") {
            Duration::from_millis(self.config.cognee_ms)
        } else if lower.contains("graphiti") {
            Duration::from_millis(self.config.graphiti_ms)
        } else {
            Duration::from_millis(self.config.standard_recall_ms)
        }
    }

    /// Get the circuit breaker config for a specific engine.
    ///
    /// Returns `(failure_threshold, cooldown, timeout)` — the three
    /// parameters needed to construct a
    /// [`crate::engine::CircuitBreaker`].
    pub fn breaker_config(&self, engine_name: &str) -> (u32, Duration, Duration) {
        (
            self.config.failure_threshold,
            Duration::from_secs(self.config.cooldown_secs),
            self.timeout_for(engine_name),
        )
    }

    /// Record a timeout event and update engine health to `Degraded`.
    #[instrument(skip(self, engine_name))]
    pub async fn record_timeout(&self, engine_name: &str, duration: Duration) {
        {
            let mut health = self.engine_health.write().await;
            health.insert(engine_name.to_string(), EngineHealthStatus::Degraded);
        }
        warn!(
            engine = engine_name,
            duration_ms = duration.as_millis() as u64,
            "engine timeout recorded — marking degraded"
        );
        self.slo_sink
            .record(SloEvent {
                event_type: SloEventType::Timeout,
                engine_name: Some(engine_name.to_string()),
                timestamp: Utc::now(),
                duration_ms: Some(duration.as_millis() as u64),
                detail: format!("engine '{}' timed out after {:?}", engine_name, duration),
            })
            .await;
    }

    /// Record a circuit-open event and mark the engine `Unreachable`.
    #[instrument(skip(self, engine_name))]
    pub async fn record_circuit_open(&self, engine_name: &str) {
        {
            let mut health = self.engine_health.write().await;
            health.insert(engine_name.to_string(), EngineHealthStatus::Unreachable);
        }
        warn!(
            engine = engine_name,
            "circuit breaker open — marking unreachable"
        );
        self.slo_sink
            .record(SloEvent {
                event_type: SloEventType::CircuitOpen,
                engine_name: Some(engine_name.to_string()),
                timestamp: Utc::now(),
                duration_ms: None,
                detail: format!("circuit breaker opened for engine '{}'", engine_name),
            })
            .await;
    }

    /// Record a circuit recovery event and mark the engine `Healthy`.
    #[instrument(skip(self, engine_name))]
    pub async fn record_circuit_recovered(&self, engine_name: &str) {
        {
            let mut health = self.engine_health.write().await;
            health.insert(engine_name.to_string(), EngineHealthStatus::Healthy);
        }
        info!(engine = engine_name, "engine recovered — marking healthy");
        self.slo_sink
            .record(SloEvent {
                event_type: SloEventType::EngineRecovered,
                engine_name: Some(engine_name.to_string()),
                timestamp: Utc::now(),
                duration_ms: None,
                detail: format!("engine '{}' recovered — circuit closed", engine_name),
            })
            .await;
    }

    /// Get the list of healthy engines (for `ContextRouter` to use).
    pub async fn healthy_engines(&self) -> Vec<String> {
        self.engine_health
            .read()
            .await
            .iter()
            .filter(|(_, &status)| status == EngineHealthStatus::Healthy)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get health status of all tracked engines.
    pub async fn engine_health_all(&self) -> HashMap<String, EngineHealthStatus> {
        self.engine_health.read().await.clone()
    }

    /// Check if an engine should be skipped (circuit open or unhealthy).
    ///
    /// Returns `true` only when the engine is known to be `Unreachable`.
    /// Unknown engines (never observed) return `false` — they get a
    /// chance to prove themselves.
    pub async fn should_skip(&self, engine_name: &str) -> bool {
        self.engine_health
            .read()
            .await
            .get(engine_name)
            .map(|&status| status == EngineHealthStatus::Unreachable)
            .unwrap_or(false)
    }

    /// Record a slow query (completed but above threshold).
    #[instrument(skip(self, engine_name))]
    pub async fn record_slow_query(
        &self,
        engine_name: &str,
        duration: Duration,
        threshold: Duration,
    ) {
        warn!(
            engine = engine_name,
            duration_ms = duration.as_millis() as u64,
            threshold_ms = threshold.as_millis() as u64,
            "slow query detected"
        );
        self.slo_sink
            .record(SloEvent {
                event_type: SloEventType::SlowQuery,
                engine_name: Some(engine_name.to_string()),
                timestamp: Utc::now(),
                duration_ms: Some(duration.as_millis() as u64),
                detail: format!(
                    "engine '{}' slow query: {:?} exceeded threshold {:?}",
                    engine_name, duration, threshold
                ),
            })
            .await;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    // ── Test helpers ──

    /// SLO sink that counts every `record` call.
    struct CountingSloSink {
        count: AtomicU64,
    }

    impl CountingSloSink {
        fn new() -> Self {
            Self {
                count: AtomicU64::new(0),
            }
        }
        fn count(&self) -> u64 {
            self.count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl SloEventSink for CountingSloSink {
        async fn record(&self, _event: SloEvent) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// SLO sink that captures every event for assertion.
    struct CapturingSloSink {
        events: Mutex<Vec<SloEvent>>,
    }

    impl CapturingSloSink {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }
        fn events(&self) -> Vec<SloEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl SloEventSink for CapturingSloSink {
        async fn record(&self, event: SloEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    fn manager_with_noop() -> ResilienceManager {
        ResilienceManager::new(EngineTimeoutConfig::default(), Arc::new(NoopSloSink))
    }

    fn manager_with_counting() -> (ResilienceManager, Arc<CountingSloSink>) {
        let sink = Arc::new(CountingSloSink::new());
        let mgr = ResilienceManager::new(EngineTimeoutConfig::default(), sink.clone());
        (mgr, sink)
    }

    fn manager_with_capturing() -> (ResilienceManager, Arc<CapturingSloSink>) {
        let sink = Arc::new(CapturingSloSink::new());
        let mgr = ResilienceManager::new(EngineTimeoutConfig::default(), sink.clone());
        (mgr, sink)
    }

    // ── 1. Default config values within spec ranges ──

    #[test]
    fn default_config_within_spec_ranges() {
        let cfg = EngineTimeoutConfig::default();
        assert!((150..=250).contains(&cfg.standard_recall_ms));
        assert!((50..=150).contains(&cfg.mem0_ms));
        assert!((200..=800).contains(&cfg.cognee_ms));
        assert!((200..=800).contains(&cfg.graphiti_ms));
        assert_eq!(cfg.failure_threshold, 5);
        assert_eq!(cfg.cooldown_secs, 30);
    }

    // ── 2-5. timeout_for matching ──

    #[test]
    fn timeout_for_mem0() {
        let mgr = manager_with_noop();
        assert_eq!(mgr.timeout_for("mem0-engine"), Duration::from_millis(100));
    }

    #[test]
    fn timeout_for_cognee() {
        let mgr = manager_with_noop();
        assert_eq!(mgr.timeout_for("cognee-engine"), Duration::from_millis(500));
    }

    #[test]
    fn timeout_for_graphiti() {
        let mgr = manager_with_noop();
        assert_eq!(
            mgr.timeout_for("graphiti-engine"),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn timeout_for_unknown_defaults_to_standard_recall() {
        let mgr = manager_with_noop();
        assert_eq!(
            mgr.timeout_for("postgres-baseline"),
            Duration::from_millis(200)
        );
    }

    #[test]
    fn timeout_for_case_insensitive() {
        let mgr = manager_with_noop();
        assert_eq!(mgr.timeout_for("MEM0"), Duration::from_millis(100));
        assert_eq!(mgr.timeout_for("Cognee"), Duration::from_millis(500));
        assert_eq!(mgr.timeout_for("GRAPHITI"), Duration::from_millis(500));
    }

    // ── 6. breaker_config ──

    #[test]
    fn breaker_config_per_engine_type() {
        let mgr = manager_with_noop();
        let (threshold, cooldown, timeout) = mgr.breaker_config("mem0-engine");
        assert_eq!(threshold, 5);
        assert_eq!(cooldown, Duration::from_secs(30));
        assert_eq!(timeout, Duration::from_millis(100));

        let (threshold, cooldown, timeout) = mgr.breaker_config("cognee");
        assert_eq!(threshold, 5);
        assert_eq!(cooldown, Duration::from_secs(30));
        assert_eq!(timeout, Duration::from_millis(500));

        let (threshold, cooldown, timeout) = mgr.breaker_config("unknown");
        assert_eq!(threshold, 5);
        assert_eq!(cooldown, Duration::from_secs(30));
        assert_eq!(timeout, Duration::from_millis(200));
    }

    // ── 7. record_timeout marks Degraded ──

    #[tokio::test]
    async fn record_timeout_marks_degraded() {
        let mgr = manager_with_noop();
        mgr.record_timeout("mem0-engine", Duration::from_millis(150))
            .await;
        let health = mgr.engine_health_all().await;
        assert_eq!(health["mem0-engine"], EngineHealthStatus::Degraded);
    }

    // ── 8. record_circuit_open marks Unreachable ──

    #[tokio::test]
    async fn record_circuit_open_marks_unreachable() {
        let mgr = manager_with_noop();
        mgr.record_circuit_open("cognee-engine").await;
        let health = mgr.engine_health_all().await;
        assert_eq!(health["cognee-engine"], EngineHealthStatus::Unreachable);
    }

    // ── 9. record_circuit_recovered marks Healthy ──

    #[tokio::test]
    async fn record_circuit_recovered_marks_healthy() {
        let mgr = manager_with_noop();
        mgr.record_circuit_open("graphiti-engine").await;
        assert_eq!(
            mgr.engine_health_all().await["graphiti-engine"],
            EngineHealthStatus::Unreachable
        );
        mgr.record_circuit_recovered("graphiti-engine").await;
        assert_eq!(
            mgr.engine_health_all().await["graphiti-engine"],
            EngineHealthStatus::Healthy
        );
    }

    // ── 10. should_skip true for Unreachable ──

    #[tokio::test]
    async fn should_skip_true_for_unreachable() {
        let mgr = manager_with_noop();
        mgr.record_circuit_open("bad-engine").await;
        assert!(mgr.should_skip("bad-engine").await);
    }

    // ── 11. should_skip false for Healthy ──

    #[tokio::test]
    async fn should_skip_false_for_healthy() {
        let mgr = manager_with_noop();
        mgr.record_circuit_recovered("good-engine").await;
        assert!(!mgr.should_skip("good-engine").await);
    }

    #[tokio::test]
    async fn should_skip_false_for_unknown_engine() {
        let mgr = manager_with_noop();
        assert!(!mgr.should_skip("never-seen").await);
    }

    // ── 12. healthy_engines returns only healthy ──

    #[tokio::test]
    async fn healthy_engines_returns_only_healthy() {
        let mgr = manager_with_noop();
        mgr.record_circuit_recovered("engine-a").await;
        mgr.record_timeout("engine-b", Duration::from_millis(100))
            .await;
        mgr.record_circuit_open("engine-c").await;

        let healthy = mgr.healthy_engines().await;
        assert_eq!(healthy.len(), 1);
        assert!(healthy.contains(&"engine-a".to_string()));
    }

    // ── 13. SLO events recorded via counting sink ──

    #[tokio::test]
    async fn slo_events_recorded_via_sink() {
        let (mgr, sink) = manager_with_counting();
        mgr.record_timeout("engine-x", Duration::from_millis(50))
            .await;
        mgr.record_circuit_open("engine-x").await;
        mgr.record_circuit_recovered("engine-x").await;
        assert_eq!(sink.count(), 3);
    }

    // ── 14. record_slow_query generates SlowQuery event ──

    #[tokio::test]
    async fn record_slow_query_generates_event() {
        let (mgr, sink) = manager_with_capturing();
        mgr.record_slow_query(
            "cognee-engine",
            Duration::from_millis(600),
            Duration::from_millis(400),
        )
        .await;
        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SloEventType::SlowQuery);
        assert_eq!(events[0].engine_name.as_deref(), Some("cognee-engine"));
        assert_eq!(events[0].duration_ms, Some(600));
    }

    #[tokio::test]
    async fn record_timeout_generates_timeout_event() {
        let (mgr, sink) = manager_with_capturing();
        mgr.record_timeout("mem0-engine", Duration::from_millis(120))
            .await;
        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SloEventType::Timeout);
        assert_eq!(events[0].duration_ms, Some(120));
    }

    #[tokio::test]
    async fn record_circuit_open_generates_circuit_open_event() {
        let (mgr, sink) = manager_with_capturing();
        mgr.record_circuit_open("graphiti-engine").await;
        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SloEventType::CircuitOpen);
    }

    #[tokio::test]
    async fn record_circuit_recovered_generates_recovered_event() {
        let (mgr, sink) = manager_with_capturing();
        mgr.record_circuit_recovered("engine-y").await;
        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, SloEventType::EngineRecovered);
    }

    // ── 15. Custom config overrides defaults ──

    #[test]
    fn custom_config_overrides_defaults() {
        let custom = EngineTimeoutConfig {
            standard_recall_ms: 250,
            mem0_ms: 150,
            cognee_ms: 800,
            graphiti_ms: 200,
            failure_threshold: 10,
            cooldown_secs: 60,
        };
        let mgr = ResilienceManager::new(custom, Arc::new(NoopSloSink));

        assert_eq!(mgr.timeout_for("baseline"), Duration::from_millis(250));
        assert_eq!(mgr.timeout_for("mem0-x"), Duration::from_millis(150));
        assert_eq!(mgr.timeout_for("cognee-x"), Duration::from_millis(800));
        assert_eq!(mgr.timeout_for("graphiti-x"), Duration::from_millis(200));

        let (threshold, cooldown, _) = mgr.breaker_config("baseline");
        assert_eq!(threshold, 10);
        assert_eq!(cooldown, Duration::from_secs(60));
    }

    // ── Bonus: full lifecycle ──

    #[tokio::test]
    async fn full_health_lifecycle() {
        let mgr = manager_with_noop();

        // Start: unknown engine not skipped
        assert!(!mgr.should_skip("engine-z").await);
        assert!(mgr.healthy_engines().await.is_empty());

        // Timeout → degraded
        mgr.record_timeout("engine-z", Duration::from_millis(200))
            .await;
        assert_eq!(
            mgr.engine_health_all().await["engine-z"],
            EngineHealthStatus::Degraded
        );
        // Degraded is not skipped (still usable)
        assert!(!mgr.should_skip("engine-z").await);

        // Circuit open → unreachable
        mgr.record_circuit_open("engine-z").await;
        assert_eq!(
            mgr.engine_health_all().await["engine-z"],
            EngineHealthStatus::Unreachable
        );
        assert!(mgr.should_skip("engine-z").await);
        assert!(!mgr
            .healthy_engines()
            .await
            .contains(&"engine-z".to_string()));

        // Recover → healthy
        mgr.record_circuit_recovered("engine-z").await;
        assert_eq!(
            mgr.engine_health_all().await["engine-z"],
            EngineHealthStatus::Healthy
        );
        assert!(!mgr.should_skip("engine-z").await);
        assert!(mgr
            .healthy_engines()
            .await
            .contains(&"engine-z".to_string()));
    }
}
