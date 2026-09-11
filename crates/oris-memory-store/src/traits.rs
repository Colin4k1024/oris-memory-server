//! Pluggable memory engine adapter traits.
//!
//! These traits define the contract between the control plane and pluggable
//! memory engines (Mem0, Cognee, Graphiti). Engines are optional — the core
//! Canonical User Memory and Shared Task Memory capabilities must function
//! without any pluggable engine.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Errors returned by memory engines.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine unavailable: {0}")]
    Unavailable(String),

    #[error("engine timeout after {ms}ms")]
    Timeout { ms: u64 },

    #[error("engine internal error: {0}")]
    Internal(String),

    #[error("circuit breaker open — engine temporarily disabled")]
    CircuitOpen,

    #[error("invalid query: {0}")]
    InvalidQuery(String),
}

/// Capabilities advertised by a memory engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub supports_write: bool,
    pub supports_vector_search: bool,
    pub supports_keyword_search: bool,
    pub supports_graph_traversal: bool,
    pub supports_temporal_queries: bool,
    pub supports_entity_linking: bool,
    pub max_embedding_dimensions: Option<usize>,
}

/// A write request to a memory engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineWriteItem {
    pub user_id: String,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub content: String,
    pub metadata: Value,
}

/// A query to a memory engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineQuery {
    pub user_id: String,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub query: String,
    pub limit: usize,
}

/// A result from a memory engine query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineResult {
    pub content: String,
    pub score: f64,
    pub metadata: Value,
}

/// Trait for pluggable memory engines.
///
/// Implementations include Mem0 (personal long-term memory), Cognee
/// (enterprise semantic/graph memory), and Graphiti (temporal graph memory).
/// All are optional — the control plane must degrade gracefully when an
/// engine is unavailable.
#[async_trait]
pub trait MemoryEngine: Send + Sync {
    /// Human-readable name (e.g. "mem0", "cognee", "graphiti").
    fn name(&self) -> &str;

    /// Advertised capabilities.
    fn capabilities(&self) -> &EngineCapabilities;

    /// Write a memory item to the engine. May fail silently if the engine
    /// is unavailable — the control plane logs and continues.
    async fn write(&self, item: &EngineWriteItem) -> Result<(), EngineError>;

    /// Query the engine for relevant memories.
    async fn search(&self, query: &EngineQuery) -> Result<Vec<EngineResult>, EngineError>;

    /// Health check — used by the circuit breaker to probe recovery.
    async fn health(&self) -> Result<(), EngineError>;
}
