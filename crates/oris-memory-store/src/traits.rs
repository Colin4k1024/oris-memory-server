//! Pluggable memory engine adapter traits — re-exported from contract.
//!
//! The canonical engine contract types now live in
//! [`oris_memory_contract::engine_contract`] so that both `oris-memory-store`
//! and `oris-control-plane` reference a single definition without circular
//! dependencies.
//!
//! This module preserves the previous public API (`traits::MemoryEngine`,
//! `traits::EngineQuery`, etc.) as re-exports for downstream code that
//! imports through `oris_memory_store::traits`.

pub use oris_memory_contract::engine_contract::*;
