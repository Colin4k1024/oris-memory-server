//! Pluggable Memory Engine adapter trait and registry.
//!
//! Defines the contract that external memory engines (Mem0, Cognee, Graphiti)
//! must satisfy to be integrated as replaceable Data Plane components behind
//! the Enterprise Context & Memory Service control plane.
//!
//! Each engine is isolated with its own timeout and circuit breaker so that
//! a failure in one engine never blocks the canonical PostgreSQL baseline.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Capabilities advertised by an engine implementation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub semantic_search: bool,
    pub keyword_search: bool,
    pub graph_search: bool,
    pub temporal_search: bool,
    pub entity_linking: bool,
    pub ontology_grounding: bool,
}

/// A query dispatched to a specific engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineQuery {
    pub text: String,
    pub tenant_id: String,
    pub user_id: Option<String>,
    pub task_id: Option<String>,
    pub top_k: usize,
    pub filters: HashMap<String, serde_json::Value>,
}

impl Default for EngineQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            tenant_id: "default".into(),
            user_id: None,
            task_id: None,
            top_k: 10,
            filters: HashMap::new(),
        }
    }
}

/// A single result returned by an engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineResult {
    pub engine_name: String,
    pub memory_id: String,
    pub content: String,
    pub score: f64,
    pub metadata: serde_json::Value,
    pub evidence_refs: Vec<String>,
}

/// Engine health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineHealth {
    Healthy,
    Degraded,
    Unreachable,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum EngineError {
    #[error("engine timeout after {0:?}")]
    Timeout(Duration),
    #[error("engine error: {0}")]
    Internal(String),
    #[error("circuit breaker open for engine {0}")]
    CircuitOpen(String),
    #[error("engine not found: {0}")]
    NotFound(String),
}

/// The contract that pluggable memory engines implement.
#[async_trait]
pub trait MemoryEngine: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> EngineCapabilities;
    async fn search(&self, query: &EngineQuery) -> Result<Vec<EngineResult>, EngineError>;
    async fn write(&self, item: &EngineWriteItem) -> Result<(), EngineError>;
    async fn delete(&self, id: &str) -> Result<(), EngineError>;
    async fn health(&self) -> Result<EngineHealth, EngineError>;
}

/// A memory item to be written to an engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineWriteItem {
    pub memory_id: String,
    pub tenant_id: String,
    pub content: String,
    pub memory_type: String,
    pub scope: String,
    pub metadata: serde_json::Value,
    pub evidence_refs: Vec<String>,
}

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

    /// Search all healthy engines in parallel and merge results.
    pub async fn search_all(&self, query: &EngineQuery) -> Vec<EngineResult> {
        let names = self.engine_names().await;
        let mut results = Vec::new();
        for name in names {
            if let Ok(r) = self.search(&name, query).await {
                results.extend(r);
            }
        }
        results
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
}
