//! oris-memory-store
//!
//! Storage layer for Oris Enterprise Context & Memory Service.
//!
//! Contains both the legacy SQLite GeneStore and the new PostgreSQL + pgvector
//! and Redis implementations.
//!
//! Why SQLite instead of JSONL?
//! - Indexed reads: O(log n) by id, tag, confidence — vs O(n) JSONL scan
//! - Atomic writes: WAL mode prevents corruption on crash
//! - Concurrent readers: multiple Oris worker processes can read simultaneously
//! - Schema migrations: ALTER TABLE is far safer than rewriting JSONL files
//! - Aggregate queries: confidence stats, success-rate histograms — free with SQL
//!
//! # Example
//!
//! ```rust,no_run
//! use oris_memory_store::{SqliteGeneStore, GeneQuery};
//!
//! let store = SqliteGeneStore::open(":memory:").unwrap();
//! let query = GeneQuery { required_tags: vec!["test_failure".into()], ..Default::default() };
//! ```

pub mod memory_types;
pub mod migrate;
pub mod partition;
pub mod postgres;
pub mod redis;
pub mod replay_hook;
pub mod store;
pub mod traits;
pub mod types;

pub use replay_hook::{ReplayCandidate, ReplayConfig, ReplayFeedbackHook, ReplayMetrics};
pub use store::{GeneStore, SqliteGeneStore};
pub use types::{Capsule, Gene, GeneMatch, GeneQuery};

// Memory domain types
pub use memory_types::{
    AuthorityLevel, CanonicalUserProfile, EventType, MatchType, MemoryCandidate, MemoryItem,
    MemoryStatus, MemoryType, OutboxEvent, OutboxStatus, PrivacyClass, Scope, SearchParams,
    SearchResult, SharedTaskContext, SourceType,
};

// Engine traits
pub use traits::{
    EngineCapabilities, EngineError, EngineQuery, EngineResult, EngineWriteItem, MemoryEngine,
};
