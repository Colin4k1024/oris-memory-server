//! Degradation Strategy — graceful degradation when subsystems fail.
//!
//! Implements the degradation principles defined in architecture §8.5.9.
//! When pluggable engines or infrastructure components become unavailable,
//! the system degrades gracefully rather than failing outright. The only
//! exception is the permission and identity subsystems — when either is
//! unavailable, the request is **denied** (never fail-open).
//!
//! # Degradation Ladder (most → least restrictive)
//!
//! 1. `Deny` — permission/identity unavailable → reject the request
//! 2. `MinimalContext` — full recall unavailable → hot context + identity only
//! 3. `KeywordFallback` — PG vector search unavailable → keyword + structured
//! 4. `NoHotContext` — Redis unavailable → direct PostgreSQL (slower)
//! 5. `PgBaselineOnly` — pluggable engines unavailable → PostgreSQL baseline
//! 6. `Normal` — all systems operational
//!
//! When multiple subsystems are degraded simultaneously the most restrictive
//! (highest) level wins. `Deny` always overrides every other level.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, instrument, warn};

// ──────────────────────────── DegradationLevel ────────────────────────────

/// The current degradation level of the system.
///
/// Each variant represents a progressively less capable operating mode.
/// See the module-level docs for the full degradation ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradationLevel {
    /// All systems operational — no degradation.
    Normal,
    /// Pluggable engines (Mem0/Cognee/Graphiti) unavailable — PostgreSQL baseline only.
    PgBaselineOnly,
    /// PostgreSQL vector search unavailable — keyword + structured search only.
    KeywordFallback,
    /// Redis hot context unavailable — direct PostgreSQL queries (slower but functional).
    NoHotContext,
    /// Full recall unavailable — return hot context + identity only (minimal viable).
    MinimalContext,
    /// Permission or identity system unavailable — DENY (never fail-open).
    Deny,
}

// ──────────────────────────── Subsystem ────────────────────────────

/// Identifiable subsystems that can be tracked for degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Subsystem {
    /// Redis hot-context cache.
    Redis,
    /// PostgreSQL pgvector / vector search.
    PostgresVector,
    /// PostgreSQL full-text / keyword search.
    PostgresKeyword,
    /// PostgreSQL structured (JSONB / relational) search.
    PostgresStructured,
    /// Mem0 pluggable engine.
    Mem0,
    /// Cognee pluggable engine.
    Cognee,
    /// Graphiti pluggable engine.
    Graphiti,
    /// Permission / ACL engine.
    PermissionEngine,
    /// Identity resolver (SSO/IAM).
    IdentityResolver,
    /// Outbox worker (async, does not block the read path).
    OutboxWorker,
}

impl Subsystem {
    /// Stable lowercase label for tracing and metrics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Redis => "redis",
            Self::PostgresVector => "postgres_vector",
            Self::PostgresKeyword => "postgres_keyword",
            Self::PostgresStructured => "postgres_structured",
            Self::Mem0 => "mem0",
            Self::Cognee => "cognee",
            Self::Graphiti => "graphiti",
            Self::PermissionEngine => "permission_engine",
            Self::IdentityResolver => "identity_resolver",
            Self::OutboxWorker => "outbox_worker",
        }
    }
}

// ──────────────────────────── DegradationManager ────────────────────────────

/// Tracks subsystem health and computes the effective degradation level.
///
/// The manager is designed to be wrapped in `Arc` and shared across the
/// control plane. All state mutations go through [`report_failure`] and
/// [`report_recovery`]; readers use [`current_level`], [`is_available`], etc.
///
/// [`report_failure`]: DegradationManager::report_failure
/// [`report_recovery`]: DegradationManager::report_recovery
/// [`current_level`]: DegradationManager::current_level
/// [`is_available`]: DegradationManager::is_available
pub struct DegradationManager {
    /// Current degradation level (starts at Normal).
    current_level: RwLock<DegradationLevel>,
    /// Subsystems currently in a degraded state.
    degraded_subsystems: RwLock<HashSet<Subsystem>>,
    /// Whether the current request must be denied (permissions/identity failure).
    deny_request: RwLock<bool>,
}

impl DegradationManager {
    /// Create a new manager with all subsystems healthy.
    pub fn new() -> Self {
        Self {
            current_level: RwLock::new(DegradationLevel::Normal),
            degraded_subsystems: RwLock::new(HashSet::new()),
            deny_request: RwLock::new(false),
        }
    }

    /// Report that a subsystem has failed or become degraded.
    ///
    /// After recording the failure the effective level is recomputed.
    #[instrument(skip(self), fields(subsystem = %subsystem.as_str()))]
    pub async fn report_failure(&self, subsystem: Subsystem) {
        warn!(subsystem = %subsystem.as_str(), "subsystem reported as degraded");
        self.degraded_subsystems.write().await.insert(subsystem);
        self.recompute_level().await;
    }

    /// Report that a subsystem has recovered.
    ///
    /// After removing the subsystem the effective level is recomputed.
    #[instrument(skip(self), fields(subsystem = %subsystem.as_str()))]
    pub async fn report_recovery(&self, subsystem: Subsystem) {
        info!(subsystem = %subsystem.as_str(), "subsystem recovered");
        self.degraded_subsystems.write().await.remove(&subsystem);
        self.recompute_level().await;
    }

    /// Get the current effective degradation level.
    pub async fn current_level(&self) -> DegradationLevel {
        *self.current_level.read().await
    }

    /// Check whether a subsystem is available (not degraded).
    pub async fn is_available(&self, subsystem: Subsystem) -> bool {
        !self.degraded_subsystems.read().await.contains(&subsystem)
    }

    /// Human-readable degradation annotation for the response, or `None`
    /// when operating normally.
    pub async fn degradation_annotation(&self) -> Option<String> {
        match self.current_level().await {
            DegradationLevel::Normal => None,
            DegradationLevel::PgBaselineOnly => {
                Some("降级模式：可插拔引擎不可用，仅使用 PostgreSQL 基线".into())
            }
            DegradationLevel::KeywordFallback => {
                Some("降级模式：向量检索不可用，使用关键词+结构化检索".into())
            }
            DegradationLevel::NoHotContext => {
                Some("降级模式：热上下文不可用，直接查询 PostgreSQL".into())
            }
            DegradationLevel::MinimalContext => {
                Some("降级模式：完整召回不可用，仅返回热上下文+身份上下文".into())
            }
            DegradationLevel::Deny => {
                Some("拒绝请求：权限或身份服务不可用，不执行 fail-open".into())
            }
        }
    }

    /// Whether the request must be denied due to permission/identity failure.
    pub async fn must_deny(&self) -> bool {
        *self.deny_request.read().await
    }

    /// Recompute the effective degradation level from the current set of
    /// degraded subsystems.
    ///
    /// Degradation rules (§8.5.9):
    /// 1. Mem0/Cognee/Graphiti unavailable → `PgBaselineOnly`
    /// 2. PG vector search unavailable → `KeywordFallback`
    /// 3. Redis hot context unavailable → `NoHotContext`
    /// 4. Full recall unavailable → `MinimalContext`
    /// 5. PermissionEngine unavailable → `Deny` (never fail-open)
    /// 6. IdentityResolver unavailable → `Deny` (never fail-open)
    ///
    /// Priority: `Deny` > `MinimalContext` > `KeywordFallback` >
    /// `NoHotContext` > `PgBaselineOnly` > `Normal`.
    /// Multiple degradations yield the most restrictive (highest) level.
    #[instrument(skip(self))]
    async fn recompute_level(&self) {
        let (new_level, deny) = {
            let degraded = self.degraded_subsystems.read().await;

            // Rules 5 & 6 — critical subsystems: Deny always wins.
            let deny = degraded.contains(&Subsystem::PermissionEngine)
                || degraded.contains(&Subsystem::IdentityResolver);

            // Rule 4 — full recall = all three PostgreSQL recall paths down.
            let full_recall_down = degraded.contains(&Subsystem::PostgresVector)
                && degraded.contains(&Subsystem::PostgresKeyword)
                && degraded.contains(&Subsystem::PostgresStructured);

            // Rule 2 — PG vector search down.
            let vector_down = degraded.contains(&Subsystem::PostgresVector);

            // Rule 3 — Redis hot context down.
            let redis_down = degraded.contains(&Subsystem::Redis);

            // Rule 1 — pluggable engines down.
            let engines_down = degraded.contains(&Subsystem::Mem0)
                || degraded.contains(&Subsystem::Cognee)
                || degraded.contains(&Subsystem::Graphiti);

            let level = if deny {
                DegradationLevel::Deny
            } else if full_recall_down {
                DegradationLevel::MinimalContext
            } else if vector_down {
                DegradationLevel::KeywordFallback
            } else if redis_down {
                DegradationLevel::NoHotContext
            } else if engines_down {
                DegradationLevel::PgBaselineOnly
            } else {
                DegradationLevel::Normal
            };

            (level, deny)
        }; // read lock released

        *self.current_level.write().await = new_level;
        *self.deny_request.write().await = deny;
    }
}

impl Default for DegradationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Normal state ──

    #[tokio::test]
    async fn normal_state_no_degradation() {
        let mgr = DegradationManager::new();
        assert_eq!(mgr.current_level().await, DegradationLevel::Normal);
        assert!(!mgr.must_deny().await);
        assert_eq!(mgr.degradation_annotation().await, None);
    }

    #[tokio::test]
    async fn default_is_normal() {
        let mgr = DegradationManager::default();
        assert_eq!(mgr.current_level().await, DegradationLevel::Normal);
        assert!(!mgr.must_deny().await);
    }

    // ── Single subsystem failures ──

    #[tokio::test]
    async fn mem0_failure_pg_baseline_only() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Mem0).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::PgBaselineOnly);
        assert!(!mgr.must_deny().await);
    }

    #[tokio::test]
    async fn cognee_failure_pg_baseline_only() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Cognee).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::PgBaselineOnly);
    }

    #[tokio::test]
    async fn graphiti_failure_pg_baseline_only() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Graphiti).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::PgBaselineOnly);
    }

    #[tokio::test]
    async fn postgres_vector_failure_keyword_fallback() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::PostgresVector).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::KeywordFallback);
    }

    #[tokio::test]
    async fn redis_failure_no_hot_context() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Redis).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::NoHotContext);
    }

    #[tokio::test]
    async fn full_recall_failure_minimal_context() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::PostgresVector).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::KeywordFallback);
        mgr.report_failure(Subsystem::PostgresKeyword).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::KeywordFallback);
        mgr.report_failure(Subsystem::PostgresStructured).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::MinimalContext);
    }

    #[tokio::test]
    async fn permission_engine_failure_deny() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::PermissionEngine).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Deny);
        assert!(mgr.must_deny().await);
    }

    #[tokio::test]
    async fn identity_resolver_failure_deny() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::IdentityResolver).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Deny);
        assert!(mgr.must_deny().await);
    }

    #[tokio::test]
    async fn postgres_keyword_alone_no_degradation() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::PostgresKeyword).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Normal);
        assert!(!mgr.is_available(Subsystem::PostgresKeyword).await);
    }

    #[tokio::test]
    async fn postgres_structured_alone_no_degradation() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::PostgresStructured).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Normal);
        assert!(!mgr.is_available(Subsystem::PostgresStructured).await);
    }

    #[tokio::test]
    async fn outbox_worker_no_degradation() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::OutboxWorker).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Normal);
        assert!(!mgr.is_available(Subsystem::OutboxWorker).await);
    }

    // ── Multiple failures → most restrictive ──

    #[tokio::test]
    async fn multiple_failures_most_restrictive() {
        let mgr = DegradationManager::new();
        // Mem0 → PgBaselineOnly (1), Redis → NoHotContext (2)
        // Most restrictive: NoHotContext
        mgr.report_failure(Subsystem::Mem0).await;
        mgr.report_failure(Subsystem::Redis).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::NoHotContext);
    }

    #[tokio::test]
    async fn vector_and_redis_keyword_fallback_wins() {
        let mgr = DegradationManager::new();
        // PostgresVector → KeywordFallback (3), Redis → NoHotContext (2)
        // Most restrictive: KeywordFallback
        mgr.report_failure(Subsystem::PostgresVector).await;
        mgr.report_failure(Subsystem::Redis).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::KeywordFallback);
    }

    #[tokio::test]
    async fn all_engines_down_still_pg_baseline_only() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Mem0).await;
        mgr.report_failure(Subsystem::Cognee).await;
        mgr.report_failure(Subsystem::Graphiti).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::PgBaselineOnly);
    }

    // ── Deny always wins ──

    #[tokio::test]
    async fn deny_overrides_everything() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Mem0).await;
        mgr.report_failure(Subsystem::PostgresVector).await;
        mgr.report_failure(Subsystem::Redis).await;
        mgr.report_failure(Subsystem::PermissionEngine).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Deny);
        assert!(mgr.must_deny().await);
    }

    #[tokio::test]
    async fn all_subsystems_down_deny() {
        let mgr = DegradationManager::new();
        for subsystem in [
            Subsystem::Redis,
            Subsystem::PostgresVector,
            Subsystem::PostgresKeyword,
            Subsystem::PostgresStructured,
            Subsystem::Mem0,
            Subsystem::Cognee,
            Subsystem::Graphiti,
            Subsystem::PermissionEngine,
            Subsystem::IdentityResolver,
            Subsystem::OutboxWorker,
        ] {
            mgr.report_failure(subsystem).await;
        }
        assert_eq!(mgr.current_level().await, DegradationLevel::Deny);
        assert!(mgr.must_deny().await);
    }

    // ── Recovery ──

    #[tokio::test]
    async fn recovery_returns_to_normal() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Mem0).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::PgBaselineOnly);
        mgr.report_recovery(Subsystem::Mem0).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Normal);
        assert!(!mgr.must_deny().await);
        assert!(mgr.is_available(Subsystem::Mem0).await);
    }

    #[tokio::test]
    async fn partial_recovery_correct_level() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::Mem0).await;
        mgr.report_failure(Subsystem::Redis).await;
        // Level: NoHotContext (most restrictive)
        assert_eq!(mgr.current_level().await, DegradationLevel::NoHotContext);
        mgr.report_recovery(Subsystem::Redis).await;
        // Now only Mem0 is down → PgBaselineOnly
        assert_eq!(mgr.current_level().await, DegradationLevel::PgBaselineOnly);
    }

    #[tokio::test]
    async fn recovery_from_deny_when_critical_restored() {
        let mgr = DegradationManager::new();
        mgr.report_failure(Subsystem::PermissionEngine).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Deny);
        assert!(mgr.must_deny().await);
        mgr.report_recovery(Subsystem::PermissionEngine).await;
        assert_eq!(mgr.current_level().await, DegradationLevel::Normal);
        assert!(!mgr.must_deny().await);
    }

    // ── Degradation annotations ──

    #[tokio::test]
    async fn degradation_annotations() {
        let mgr = DegradationManager::new();

        // Normal → None
        assert_eq!(mgr.degradation_annotation().await, None);

        // PgBaselineOnly
        mgr.report_failure(Subsystem::Mem0).await;
        assert_eq!(
            mgr.degradation_annotation().await.as_deref(),
            Some("降级模式：可插拔引擎不可用，仅使用 PostgreSQL 基线")
        );
        mgr.report_recovery(Subsystem::Mem0).await;

        // KeywordFallback
        mgr.report_failure(Subsystem::PostgresVector).await;
        assert_eq!(
            mgr.degradation_annotation().await.as_deref(),
            Some("降级模式：向量检索不可用，使用关键词+结构化检索")
        );
        mgr.report_recovery(Subsystem::PostgresVector).await;

        // NoHotContext
        mgr.report_failure(Subsystem::Redis).await;
        assert_eq!(
            mgr.degradation_annotation().await.as_deref(),
            Some("降级模式：热上下文不可用，直接查询 PostgreSQL")
        );
        mgr.report_recovery(Subsystem::Redis).await;

        // MinimalContext
        mgr.report_failure(Subsystem::PostgresVector).await;
        mgr.report_failure(Subsystem::PostgresKeyword).await;
        mgr.report_failure(Subsystem::PostgresStructured).await;
        assert_eq!(
            mgr.degradation_annotation().await.as_deref(),
            Some("降级模式：完整召回不可用，仅返回热上下文+身份上下文")
        );
        mgr.report_recovery(Subsystem::PostgresVector).await;
        mgr.report_recovery(Subsystem::PostgresKeyword).await;
        mgr.report_recovery(Subsystem::PostgresStructured).await;

        // Deny
        mgr.report_failure(Subsystem::PermissionEngine).await;
        assert_eq!(
            mgr.degradation_annotation().await.as_deref(),
            Some("拒绝请求：权限或身份服务不可用，不执行 fail-open")
        );
    }

    // ── is_available for all 10 subsystems ──

    #[tokio::test]
    async fn is_available_all_subsystems() {
        let mgr = DegradationManager::new();

        let all = [
            Subsystem::Redis,
            Subsystem::PostgresVector,
            Subsystem::PostgresKeyword,
            Subsystem::PostgresStructured,
            Subsystem::Mem0,
            Subsystem::Cognee,
            Subsystem::Graphiti,
            Subsystem::PermissionEngine,
            Subsystem::IdentityResolver,
            Subsystem::OutboxWorker,
        ];

        // All available initially
        for &s in &all {
            assert!(mgr.is_available(s).await, "{:?} should be available", s);
        }

        // Mark each as failed and verify availability flips
        for &s in &all {
            mgr.report_failure(s).await;
            assert!(!mgr.is_available(s).await, "{:?} should be unavailable", s);
            mgr.report_recovery(s).await;
            assert!(
                mgr.is_available(s).await,
                "{:?} should be available again",
                s
            );
        }
    }
}
