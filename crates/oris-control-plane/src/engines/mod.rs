//! Pluggable memory engine implementations — Mem0 (§3.6), Cognee (§3.7),
//! and Graphiti (§3.8).
//!
//! These are concrete `MemoryEngine` implementations that wrap external
//! memory system REST APIs. Each engine is isolated behind the
//! [`EngineRegistry`](crate::engine::EngineRegistry) with its own circuit
//! breaker and timeout, so a failure in one engine never blocks the
//! canonical PostgreSQL baseline.
//!
//! ## Architecture references
//!
//! - §3.6 — Mem0: personal long-term inference memory; conflict merge,
//!   delete propagation, idempotency, Chinese recall quality.
//! - §3.7 — Cognee: enterprise semantic relationship graph; multi-hop
//!   accuracy, entity duplication, relationship correctness, source tracing.
//! - §3.8 — Graphiti: temporal (bi-temporal) context graph; fact validity
//!   windows, source provenance, dual-temporal queries. Optional, Phase 2+.
//! - [`crate::engine`] — defines the `MemoryEngine` trait, `EngineRegistry`,
//!   and `CircuitBreaker` infrastructure.

pub mod cognee;
pub mod graphiti;
pub mod mem0;

pub use cognee::CogneeEngine;
pub use graphiti::GraphitiEngine;
pub use mem0::Mem0Engine;
