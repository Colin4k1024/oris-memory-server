//! Governance module — RBAC + ABAC permission engine, retention policies,
//! access audit logging, memory lifecycle management, version tracking, and
//! forget/delete operations.
//!
//! This module provides the control-plane governance layer:
//! - [`acl`] — Role-based and attribute-based access control
//! - [`policy`] — Retention policy engine and privacy classification
//! - [`audit`] — Access audit logging to the `memory_access_audit` table
//! - [`retention`] — Lifecycle management (decay → archive → delete)
//! - [`version`] — Memory version tracking and rollback
//! - [`forget`] — Soft/hard delete, GDPR right-to-be-forgotten, cascade forget

pub mod acl;
pub mod audit;
pub mod forget;
pub mod policy;
pub mod retention;
pub mod version;

pub use acl::{AccessDecision, AccessRequest, AclEngine, Action, PermissionSet};
pub use audit::{AuditEntry, AuditLogger};
pub use forget::{ForgetError, ForgetManager};
pub use policy::{PolicyEngine, RetentionPolicy};
pub use retention::RetentionManager;
pub use version::{MemoryVersion, VersionError, VersionManager};

/// Unified error type for governance operations.
#[derive(Debug, thiserror::Error)]
pub enum GovernanceError {
    /// Wraps a database error from `sqlx`.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    /// No retention policy matched the given scope and memory type.
    #[error("no retention policy for scope {scope}, memory type {mem_type}")]
    PolicyNotFound { scope: String, mem_type: String },
}
