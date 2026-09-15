//! Pluggable Memory Engine registry, circuit breaker, and NoopEngine.
//!
//! The engine contract types (`MemoryEngine`, `EngineQuery`, `EngineResult`,
//! `EngineWriteItem`, `EngineCapabilities`, `EngineError`, `EngineHealth`)
//! are defined once in [`oris_memory_contract::engine_contract`] and
//! re-exported here for convenience.
//!
//! This module adds the control-plane-specific infrastructure:
//! - [`CircuitBreaker`] — per-engine failure isolation
//! - [`EngineRegistry`] — registry with circuit breakers and **parallel**
//!   multi-engine search (fixes #49 — `search_all` now uses
//!   `futures_util::future::join_all` instead of sequential `for` loop)
//! - [`NoopEngine`] — testing placeholder

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

// Re-export all contract types so downstream code that does
// `use crate::engine::{EngineQuery, MemoryEngine, ...}` still works.
pub use oris_memory_contract::engine_contract::{
    EngineCapabilities, EngineError, EngineHealth, EngineQuery, EngineResult, EngineWriteItem,
    MemoryEngine,
};

// ─────────────────────────────────────────────────────────────────────
// Circuit Breaker
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: BreakerState,
    consecutive_failures: u32,
    failure_threshold: u32,
    cooldown: Duration,
    last_failure: Option<Instant>,
    timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown: Duration, timeout: Duration) -> Self {
        Self {
            state: BreakerState::Closed,
            consecutive_failures: 0,
            failure_threshold,
            cooldown,
            last_failure: None,
            timeout,
        }
    }

    pub fn allow(&mut self) -> Result<Duration, EngineError> {
        match self.state {
            BreakerState::Closed => Ok(self.timeout),
            BreakerState::HalfOpen => {
                self.state = BreakerState::Open;
                self.last_failure = Some(Instant::now());
                Ok(self.timeout)
            }
            BreakerState::Open => {
                if let Some(last) = self.last_failure {
                    if last.elapsed() >= self.cooldown {
                        self.state = BreakerState::HalfOpen;
                        return Ok(self.timeout);
                    }
                }
                Err(EngineError::CircuitOpen("engine".into()))
            }
        }
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = BreakerState::Closed;
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());
        if self.consecutive_failures >= self.failure_threshold {
            self.state = BreakerState::Open;
        }
    }

    pub fn state(&self) -> BreakerState {
        self.state
    }

    pub fn is_open(&self) -> bool {
        self.state == BreakerState::Open
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30), Duration::from_millis(200))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Engine Registry
// ─────────────────────────────────────────────────────────────────────

pub struct EngineRegistry {
    engines: RwLock<HashMap<String, Arc<dyn MemoryEngine>>>,
    breakers: RwLock<HashMap<String, CircuitBreaker>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: RwLock::new(HashMap::new()),
            breakers: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register(&self, engine: Arc<dyn MemoryEngine>, breaker: CircuitBreaker) {
        let name = engine.name().to_string();
        self.engines.write().await.insert(name.clone(), engine);
        self.breakers.write().await.insert(name, breaker);
    }

    pub async fn register_default(&self, engine: Arc<dyn MemoryEngine>) {
        let name = engine.name().to_string();
        self.breakers
            .write()
            .await
            .insert(name.clone(), CircuitBreaker::default());
        self.engines.write().await.insert(name, engine);
    }

    pub async fn unregister(&self, name: &str) {
        self.engines.write().await.remove(name);
        self.breakers.write().await.remove(name);
    }

    pub async fn engine_names(&self) -> Vec<String> {
        self.engines.read().await.keys().cloned().collect()
    }

    /// Search a specific engine, respecting circuit breaker and timeout.
    pub async fn search(
        &self,
        engine_name: &str,
        query: &EngineQuery,
    ) -> Result<Vec<EngineResult>, EngineError> {
        // Clone the Arc before dropping the read guard to avoid borrow issues.
        let engine = {
            let engines = self.engines.read().await;
            engines
                .get(engine_name)
                .cloned()
                .ok_or_else(|| EngineError::NotFound(engine_name.into()))?
        };

        let timeout = {
            let mut breakers = self.breakers.write().await;
            let breaker = breakers
                .entry(engine_name.to_string())
                .or_insert_with(CircuitBreaker::default);
            breaker.allow()?
        };

        match tokio::time::timeout(timeout, engine.search(query)).await {
            Ok(Ok(results)) => {
                self.breakers
                    .write()
                    .await
                    .entry(engine_name.to_string())
                    .or_insert_with(CircuitBreaker::default)
                    .record_success();
                Ok(results)
            }
            Ok(Err(e)) => {
                self.breakers
                    .write()
                    .await
                    .entry(engine_name.to_string())
                    .or_insert_with(CircuitBreaker::default)
                    .record_failure();
                Err(e)
            }
            Err(_) => {
                self.breakers
                    .write()
                    .await
                    .entry(engine_name.to_string())
                    .or_insert_with(CircuitBreaker::default)
                    .record_failure();
                Err(EngineError::Timeout(timeout))
            }
        }
    }

    /// Search all healthy engines **in parallel** and merge results.
    ///
    /// Uses `futures_util::future::join_all` so that the total latency is the
    /// maximum of individual engine latencies, not the sum. Each engine is
    /// still isolated by its own circuit breaker and timeout.
    pub async fn search_all(&self, query: &EngineQuery) -> Vec<EngineResult> {
        let names = self.engine_names().await;

        // Collect futures for each engine search. We clone `self` as `Arc`
        // is not needed — we capture `&self` which lives long enough.
        let futures: Vec<_> = names
            .iter()
            .map(|name| self.search(name, query))
            .collect();

        // Run all searches concurrently.
        let results = futures_util::future::join_all(futures).await;

        results
            .into_iter()
            .flatten()
            .collect()
    }

    /// Search all healthy engines **in parallel** (alias for `search_all`).
    ///
    /// This method is kept for API compatibility; it delegates to
    /// [`search_all`](Self::search_all).
    pub async fn search_all_parallel(&self, query: &EngineQuery) -> Vec<EngineResult> {
        self.search_all(query).await
    }

    pub async fn health_all(&self) -> HashMap<String, EngineHealth> {
        let engines = self.engines.read().await;
        let mut health_map = HashMap::new();
        for (name, engine) in engines.iter() {
            let is_open = self
                .breakers
                .read()
                .await
                .get(name)
                .map(|b| b.is_open())
                .unwrap_or(false);
            if is_open {
                health_map.insert(name.clone(), EngineHealth::Unreachable);
                continue;
            }
            let health = engine.health().await.unwrap_or(EngineHealth::Unreachable);
            health_map.insert(name.clone(), health);
        }
        health_map
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// No-op engine (for testing and as a placeholder)
// ─────────────────────────────────────────────────────────────────────

pub struct NoopEngine {
    name: String,
}

impl NoopEngine {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl MemoryEngine for NoopEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities::default()
    }

    async fn search(&self, _query: &EngineQuery) -> Result<Vec<EngineResult>, EngineError> {
        Ok(vec![])
    }

    async fn write(&self, _item: &EngineWriteItem) -> Result<(), EngineError> {
        Ok(())
    }

    async fn delete(&self, _id: &str) -> Result<(), EngineError> {
        Ok(())
    }

    async fn health(&self) -> Result<EngineHealth, EngineError> {
        Ok(EngineHealth::Healthy)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(!cb.is_open());
    }

    #[test]
    fn circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(5), Duration::from_millis(100));
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());
        cb.record_failure();
        assert!(cb.is_open());
        assert!(cb.allow().is_err());
    }

    #[test]
    fn circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(5), Duration::from_millis(100));
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());
        cb.record_success();
        assert!(!cb.is_open());
        assert_eq!(cb.state(), BreakerState::Closed);
    }

    #[tokio::test]
    async fn registry_search_noop_engine() {
        let registry = EngineRegistry::new();
        registry
            .register_default(Arc::new(NoopEngine::new("test")))
            .await;

        let query = EngineQuery {
            text: "test query".into(),
            ..Default::default()
        };
        let results = registry.search("test", &query).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn registry_search_unknown_engine_errors() {
        let registry = EngineRegistry::new();
        let query = EngineQuery::default();
        let result = registry.search("nonexistent", &query).await;
        assert!(matches!(result, Err(EngineError::NotFound(_))));
    }

    #[tokio::test]
    async fn registry_health_all_reports_healthy() {
        let registry = EngineRegistry::new();
        registry
            .register_default(Arc::new(NoopEngine::new("engine-a")))
            .await;
        registry
            .register_default(Arc::new(NoopEngine::new("engine-b")))
            .await;

        let health = registry.health_all().await;
        assert_eq!(health.len(), 2);
        assert_eq!(health["engine-a"], EngineHealth::Healthy);
        assert_eq!(health["engine-b"], EngineHealth::Healthy);
    }

    #[tokio::test]
    async fn registry_search_all_skips_open_breakers() {
        let registry = EngineRegistry::new();
        registry
            .register(
                Arc::new(NoopEngine::new("noop")),
                CircuitBreaker::new(1, Duration::from_secs(60), Duration::from_millis(1)),
            )
            .await;

        {
            let mut breakers = registry.breakers.write().await;
            let cb = breakers.get_mut("noop").unwrap();
            cb.record_failure();
        }

        let query = EngineQuery::default();
        let results = registry.search_all(&query).await;
        assert!(results.is_empty());
    }

    #[test]
    fn engine_capabilities_default() {
        let caps = EngineCapabilities::default();
        assert!(!caps.semantic_search);
        assert!(!caps.graph_search);
    }

    /// Verify that `search_all` runs engines in parallel (not sequentially).
    ///
    /// Each engine sleeps for 50ms. If run sequentially, total time ≥ 100ms.
    /// If run in parallel, total time < 100ms.
    #[tokio::test]
    async fn search_all_runs_in_parallel() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Instant;

        struct SlowEngine {
            name: String,
            counter: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl MemoryEngine for SlowEngine {
            fn name(&self) -> &str {
                &self.name
            }
            fn capabilities(&self) -> EngineCapabilities {
                EngineCapabilities::default()
            }
            async fn search(&self, _q: &EngineQuery) -> Result<Vec<EngineResult>, EngineError> {
                self.counter.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(vec![EngineResult {
                    engine_name: self.name.clone(),
                    memory_id: None,
                    content: "result".into(),
                    score: 1.0,
                    metadata: serde_json::Value::Null,
                    evidence_refs: vec![],
                }])
            }
            async fn write(&self, _i: &EngineWriteItem) -> Result<(), EngineError> {
                Ok(())
            }
            async fn delete(&self, _id: &str) -> Result<(), EngineError> {
                Ok(())
            }
            async fn health(&self) -> Result<EngineHealth, EngineError> {
                Ok(EngineHealth::Healthy)
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let registry = EngineRegistry::new();
        registry
            .register(
                Arc::new(SlowEngine {
                    name: "slow-a".into(),
                    counter: counter.clone(),
                }),
                CircuitBreaker::new(100, Duration::from_secs(60), Duration::from_secs(2)),
            )
            .await;
        registry
            .register(
                Arc::new(SlowEngine {
                    name: "slow-b".into(),
                    counter: counter.clone(),
                }),
                CircuitBreaker::new(100, Duration::from_secs(60), Duration::from_secs(2)),
            )
            .await;

        let query = EngineQuery::default();
        let start = Instant::now();
        let results = registry.search_all(&query).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        // Parallel: total should be well under 100ms (sequential would be ≥100ms).
        assert!(
            elapsed < Duration::from_millis(100),
            "search_all took {:?}, expected < 100ms (parallel)",
            elapsed
        );
    }
}
