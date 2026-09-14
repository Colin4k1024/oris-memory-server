//! Pluggable memory engine implementations — Mem0 (§3.6) and Cognee (§3.7).
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
//! - [`crate::engine`] — defines the `MemoryEngine` trait, `EngineRegistry`,
//!   and `CircuitBreaker` infrastructure.

pub mod cognee;
pub mod mem0;

pub use cognee::CogneeEngine;
pub use mem0::Mem0Engine;
