//! Pluggable memory engine contract — shared types for engine adapters.
//!
//! These types define the contract between the control plane and pluggable
//! memory engines (Mem0, Cognee, Graphiti). They live in the contract crate
//! so that both `oris-memory-store` and `oris-control-plane` can reference
//! them without circular dependencies.
//!
//! Engines are optional — the core Canonical User Memory and Shared Task
//! Memory capabilities must function without any pluggable engine.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ──────────────────────────── Error ────────────────────────────

/// Errors returned by memory engines.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EngineError {
    #[error("engine unavailable: {0}")]
    Unavailable(String),

    #[error("engine timeout after {0:?}")]
    Timeout(std::time::Duration),

    #[error("engine internal error: {0}")]
    Internal(String),

    #[error("circuit breaker open for engine {0}")]
    CircuitOpen(String),

    #[error("engine not found: {0}")]
    NotFound(String),

    #[error("invalid query: {0}")]
    InvalidQuery(String),
}

// ──────────────────────────── Health ────────────────────────────

/// Engine health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineHealth {
    Healthy,
    Degraded,
    Unreachable,
}

// ──────────────────────────── Capabilities ────────────────────────────

/// Capabilities advertised by a memory engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub semantic_search: bool,
    pub keyword_search: bool,
    pub graph_search: bool,
    pub temporal_search: bool,
    pub entity_linking: bool,
    pub ontology_grounding: bool,
    pub supports_write: bool,
    pub supports_vector_search: bool,
    pub supports_graph_traversal: bool,
    pub supports_temporal_queries: bool,
    pub supports_entity_linking: bool,
    pub max_embedding_dimensions: Option<usize>,
}

// ──────────────────────────── Query ────────────────────────────

/// A query dispatched to a specific engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineQuery {
    /// Natural-language or keyword query text.
    pub text: String,
    /// Tenant / organization ID — enforces data isolation.
    pub tenant_id: String,
    /// User ID for personal-memory scoping (Mem0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Agent ID for agent-scoped queries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Run ID for run-scoped queries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Task ID for task-scoped queries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Additional filter key-value pairs.
    #[serde(default)]
    pub filters: HashMap<String, Value>,
}

impl Default for EngineQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            tenant_id: "default".into(),
            user_id: None,
            agent_id: None,
            run_id: None,
            task_id: None,
            top_k: 10,
            filters: HashMap::new(),
        }
    }
}

// ──────────────────────────── Write ────────────────────────────

/// A memory item to be written to an engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineWriteItem {
    /// Canonical memory ID (UUID as string).
    pub memory_id: String,
    /// Tenant / organization ID.
    pub tenant_id: String,
    /// User ID for personal-memory scoping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Agent ID that produced this memory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Run ID that produced this memory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Raw content text of the memory.
    pub content: String,
    /// Memory type label (e.g. "semantic", "episodic").
    pub memory_type: String,
    /// Organisational scope label (e.g. "personal", "enterprise").
    pub scope: String,
    /// Additional structured metadata.
    #[serde(default)]
    pub metadata: Value,
    /// Evidence reference IDs.
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

// ──────────────────────────── Result ────────────────────────────

/// A single result returned by an engine query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineResult {
    /// Name of the engine that produced this result.
    pub engine_name: String,
    /// Canonical memory ID (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<String>,
    /// Content text of the result.
    pub content: String,
    /// Relevance score (0.0–1.0, higher = more relevant).
    pub score: f64,
    /// Additional structured metadata.
    #[serde(default)]
    pub metadata: Value,
    /// Evidence reference IDs.
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

// ──────────────────────────── Trait ────────────────────────────

/// The contract that pluggable memory engines implement.
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
    fn capabilities(&self) -> EngineCapabilities;

    /// Query the engine for relevant memories.
    async fn search(&self, query: &EngineQuery) -> Result<Vec<EngineResult>, EngineError>;

    /// Write a memory item to the engine. May fail silently if the engine
    /// is unavailable — the control plane logs and continues.
    async fn write(&self, item: &EngineWriteItem) -> Result<(), EngineError>;

    /// Delete a memory item from the engine by ID.
    async fn delete(&self, id: &str) -> Result<(), EngineError>;

    /// Health check — used by the circuit breaker to probe recovery.
    async fn health(&self) -> Result<EngineHealth, EngineError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_query_default() {
        let q = EngineQuery::default();
        assert_eq!(q.tenant_id, "default");
        assert_eq!(q.top_k, 10);
        assert!(q.user_id.is_none());
        assert!(q.filters.is_empty());
    }

    #[test]
    fn engine_capabilities_default_all_false() {
        let caps = EngineCapabilities::default();
        assert!(!caps.semantic_search);
        assert!(!caps.supports_write);
        assert!(caps.max_embedding_dimensions.is_none());
    }

    #[test]
    fn engine_health_serde() {
        let h = EngineHealth::Healthy;
        let s = serde_json::to_string(&h).unwrap();
        assert_eq!(s, "\"healthy\"");
    }
}
