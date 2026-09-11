//! Redis hot-context, cache, and session implementations.
//!
//! Redis serves as a **non-authoritative** layer for hot context
//! materialization, result caching, and distributed locks. All data in Redis
//! can be lost and rebuilt from PostgreSQL.

pub mod cache;
pub mod hot_context;
pub mod session;
