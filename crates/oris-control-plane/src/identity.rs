//! Identity Resolver — SSO/IAM delegated identity resolution.
//!
//! Resolves an external token (SSO/IAM/HR) into a `ResolvedIdentity` that
//! the control plane uses for all permission, tenant, and trace decisions.
//! Permission snapshots are cached in Redis (TTL 60s) to keep identity
//! resolution within the 10–20ms latency budget defined in §6.
//!
//! # Flow
//!
//! 1. Extract tenant + user hint from token (without full verification —
//!    the IAM system is authoritative for authentication).
//! 2. Check Redis permission snapshot (`oris:perm:{tenant}:{user}`).
//! 3. On cache miss, call `IamClient::resolve_identity`.
//! 4. Cache the result in Redis.
//! 5. Return `ResolvedIdentity` with trace_id.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use thiserror::Error;
use tracing::{debug, instrument};
use uuid::Uuid;

use oris_memory_store::redis::hot_context::HotContextRepo;

// ──────────────────────────── Errors ────────────────────────────

/// Errors that can occur during identity resolution.
#[derive(Debug, Error)]
pub enum IdentityError {
    #[error("IAM client error: {0}")]
    IamClient(String),

    #[error("Redis cache error: {0}")]
    Redis(String),

    #[error("Token is invalid or expired")]
    InvalidToken,

    #[error("Identity not found for token")]
    NotFound,

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Agent {0} is not authorized for purpose {1}")]
    AgentNotAuthorized(String, String),
}

// ──────────────────────────── Types ────────────────────────────

/// Permission snapshot carried with the resolved identity.
///
/// This is the *resolved* permission set — the roles and explicit
/// permissions granted to this identity at resolution time. The
/// `governance::acl` module interprets these against policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSet {
    /// RBAC roles (e.g., "operator", "quality_manager", "admin").
    pub roles: Vec<String>,

    /// Explicit permission strings (e.g., "memory:read", "memory:promote").
    pub permissions: Vec<String>,

    /// Hard-deny list — always overrides any allow.
    #[serde(default)]
    pub denied: Vec<String>,
}

impl PermissionSet {
    pub fn new(roles: Vec<String>, permissions: Vec<String>) -> Self {
        Self {
            roles,
            permissions,
            denied: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.roles.is_empty() && self.permissions.is_empty()
    }

    pub fn has_permission(&self, perm: &str) -> bool {
        !self.denied.iter().any(|d| d == perm)
            && self.permissions.iter().any(|p| p == perm || p == "*")
    }
}

/// A fully resolved identity — the output of `IdentityResolver::resolve`.
///
/// This is the single source of truth for "who is asking" throughout the
/// control plane. Every memory read/write carries a `ResolvedIdentity`
/// for permission checks, tenant isolation, and audit tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedIdentity {
    /// Internal user ID (canonical).
    pub user_id: String,

    /// Tenant / organization ID — enforces data isolation.
    pub organization_id: String,

    /// Factory ID if the user belongs to a specific factory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub factory_id: Option<String>,

    /// RBAC roles.
    pub roles: Vec<String>,

    /// If this request is delegated by an agent on behalf of a user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delegated_agent_id: Option<String>,

    /// Resolved permission set.
    pub permissions: PermissionSet,

    /// The stated purpose of this access (for audit).
    pub purpose: String,

    /// Unique trace ID for the request lifecycle.
    pub trace_id: String,

    /// When this identity was resolved (for TTL checks).
    pub resolved_at: DateTime<Utc>,
}

impl ResolvedIdentity {
    /// Create a synthetic identity for testing or internal services.
    pub fn system(tenant: &str, trace_id: &str) -> Self {
        Self {
            user_id: "system".to_string(),
            organization_id: tenant.to_string(),
            factory_id: None,
            roles: vec!["system".to_string()],
            delegated_agent_id: None,
            permissions: PermissionSet::new(vec!["system".to_string()], vec!["*".to_string()]),
            purpose: "system".to_string(),
            trace_id: trace_id.to_string(),
            resolved_at: Utc::now(),
        }
    }

    /// Check if the identity has a specific role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// Check if the identity has a specific permission.
    pub fn has_permission(&self, perm: &str) -> bool {
        self.permissions.has_permission(perm)
    }

    /// Check if this identity belongs to a tenant.
    pub fn belongs_to_tenant(&self, tenant: &str) -> bool {
        self.organization_id == tenant
    }
}

/// Raw identity data returned by the IAM client before enrichment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IamIdentity {
    pub user_id: String,
    pub organization_id: String,
    pub factory_id: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub display_name: Option<String>,
    pub email: Option<String>,
}

// ──────────────────────────── IamClient Trait ────────────────────────────

/// Abstracts SSO/IAM/HR systems (Keycloak, Auth0, internal IdP, etc.).
///
/// The control plane never directly verifies tokens — it delegates to
/// the configured IAM client, which is authoritative for authentication.
/// The control plane only adds tenant isolation, permission caching,
/// and audit tracing on top.
#[async_trait]
pub trait IamClient: Send + Sync {
    /// Verify the token and return raw identity data.
    ///
    /// Returns `Err(IdentityError::InvalidToken)` for expired/invalid tokens,
    /// `Err(IdentityError::NotFound)` if the user doesn't exist.
    async fn verify_token(&self, token: &str) -> Result<IamIdentity, IdentityError>;

    /// Fetch updated permissions for a user (used on cache refresh).
    ///
    /// Default implementation re-verifies the token. Override for
    /// systems that support a separate permissions endpoint.
    async fn fetch_permissions(
        &self,
        _user_id: &str,
        _organization_id: &str,
    ) -> Result<PermissionSet, IdentityError> {
        Err(IdentityError::NotFound)
    }

    /// Name of the IAM provider (for logging/metrics).
    fn name(&self) -> &str {
        "iam"
    }
}

/// A no-op IAM client for testing and local development.
pub struct NoopIamClient {
    tenant: String,
}

impl NoopIamClient {
    pub fn new(tenant: impl Into<String>) -> Self {
        Self {
            tenant: tenant.into(),
        }
    }
}

#[async_trait]
impl IamClient for NoopIamClient {
    async fn verify_token(&self, token: &str) -> Result<IamIdentity, IdentityError> {
        if token.is_empty() {
            return Err(IdentityError::InvalidToken);
        }
        // Extract user_id from token (simulated: token == user_id for noop).
        Ok(IamIdentity {
            user_id: token.to_string(),
            organization_id: self.tenant.clone(),
            factory_id: None,
            roles: vec!["user".to_string()],
            permissions: vec!["memory:read".to_string()],
            display_name: Some(token.to_string()),
            email: None,
        })
    }

    fn name(&self) -> &str {
        "noop-iam"
    }
}

// ──────────────────────────── IdentityResolver ────────────────────────────

/// Resolves external tokens into `ResolvedIdentity` with Redis caching.
///
/// # Cache Strategy
///
/// - **Cache key**: `oris:perm:{tenant}:{user}`
/// - **TTL**: 60 seconds (permissions can change; short TTL balances
///   latency vs. staleness)
/// - **Cache miss**: call `IamClient`, store result
/// - **Cache hit**: deserialize and return (skip IAM round-trip)
///
/// # Latency Budget
///
/// - Cache hit: <5ms (Redis GET + deserialize)
/// - Cache miss: 10–20ms (IAM round-trip + Redis SET)
pub struct IdentityResolver {
    iam_client: Arc<dyn IamClient>,
    hot_context: Option<HotContextRepo>,
    cache_ttl_secs: u64,
}

/// Permission snapshot TTL in seconds.
const PERMISSION_CACHE_TTL_SECS: u64 = 60;

/// Cached permission entry stored in Redis.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedPermission {
    identity: IamIdentity,
    cached_at: DateTime<Utc>,
}

impl IdentityResolver {
    /// Create a new resolver with an IAM client (no Redis caching).
    pub fn new(iam_client: Arc<dyn IamClient>) -> Self {
        Self {
            iam_client,
            hot_context: None,
            cache_ttl_secs: PERMISSION_CACHE_TTL_SECS,
        }
    }

    /// Enable Redis permission caching.
    pub fn with_hot_context(mut self, repo: HotContextRepo) -> Self {
        self.hot_context = Some(repo);
        self
    }

    /// Override the default cache TTL.
    pub fn with_cache_ttl(mut self, ttl_secs: u64) -> Self {
        self.cache_ttl_secs = ttl_secs;
        self
    }

    /// Resolve an external token into a `ResolvedIdentity`.
    ///
    /// # Arguments
    /// - `token` — SSO/IAM token (opaque to the control plane)
    /// - `agent_id` — Agent making the request (None for direct user access)
    /// - `purpose` — Stated purpose (for audit, e.g. "task_execution", "memory_write")
    ///
    /// # Returns
    /// `ResolvedIdentity` with roles, permissions, and trace_id.
    #[instrument(skip(self, token), fields(iam = self.iam_client.name()))]
    pub async fn resolve(
        &self,
        token: &str,
        agent_id: Option<&str>,
        purpose: &str,
    ) -> Result<ResolvedIdentity, IdentityError> {
        let trace_id = Uuid::now_v7().to_string();

        // Step 1: Try Redis cache first.
        let cached = self.try_get_cached(token).await;

        let iam_identity = match cached {
            Some(identity) => {
                debug!(trace_id = %trace_id, "identity cache hit");
                identity
            }
            None => {
                debug!(trace_id = %trace_id, "identity cache miss — calling IAM");
                let identity = self.iam_client.verify_token(token).await?;

                // Cache the result.
                self.try_set_cached(&identity).await;

                identity
            }
        };

        // Step 2: Build resolved identity.
        let resolved = ResolvedIdentity {
            user_id: iam_identity.user_id,
            organization_id: iam_identity.organization_id.clone(),
            factory_id: iam_identity.factory_id,
            roles: iam_identity.roles.clone(),
            delegated_agent_id: agent_id.map(|s| s.to_string()),
            permissions: PermissionSet::new(iam_identity.roles, iam_identity.permissions),
            purpose: purpose.to_string(),
            trace_id: trace_id.clone(),
            resolved_at: Utc::now(),
        };

        Ok(resolved)
    }

    /// Invalidate the permission cache for a user (e.g., after role change).
    pub async fn invalidate_cache(
        &self,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<(), IdentityError> {
        if let Some(hc) = &self.hot_context {
            hc.invalidate_user_context(user_id, tenant_id)
                .await
                .map_err(|e| IdentityError::Redis(e.to_string()))?;
        }
        Ok(())
    }

    /// Try to get a cached identity from Redis.
    async fn try_get_cached(&self, token: &str) -> Option<IamIdentity> {
        let hc = self.hot_context.as_ref()?;

        // We don't know the tenant/user yet — we use the token itself
        // as a cache key component. In production, the token would be
        // a JWT with claims we could decode to get tenant/user.
        // For now, we use a hash of the token as the cache key.
        let cache_key = format!("oris:iam:{}", short_hash(token));

        match hc.get_user_context(&cache_key, "iam").await {
            Ok(Some(data)) => serde_json::from_slice::<CachedPermission>(&data)
                .map(|c| c.identity)
                .ok(),
            _ => None,
        }
    }

    /// Cache an identity in Redis.
    async fn try_set_cached(&self, identity: &IamIdentity) {
        if let Some(hc) = &self.hot_context {
            let cached = CachedPermission {
                identity: identity.clone(),
                cached_at: Utc::now(),
            };
            if let Ok(data) = serde_json::to_vec(&cached) {
                let cache_key = format!("oris:iam:{}", short_hash(&identity.user_id));
                let _ = hc
                    .set_user_context(&cache_key, "iam", &data, self.cache_ttl_secs)
                    .await;
            }
        }
    }
}

/// Short hash for cache keys (FNV-1a, 16 hex chars).
fn short_hash(s: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in s.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PermissionSet tests ──

    #[test]
    fn permission_set_has_permission_exact() {
        let ps = PermissionSet::new(
            vec!["operator".into()],
            vec!["memory:read".into(), "memory:write".into()],
        );
        assert!(ps.has_permission("memory:read"));
        assert!(ps.has_permission("memory:write"));
        assert!(!ps.has_permission("memory:delete"));
    }

    #[test]
    fn permission_set_wildcard_grants_all() {
        let ps = PermissionSet::new(vec!["admin".into()], vec!["*".into()]);
        assert!(ps.has_permission("memory:read"));
        assert!(ps.has_permission("memory:delete"));
        assert!(ps.has_permission("anything"));
    }

    #[test]
    fn permission_set_denied_overrides_allow() {
        let mut ps = PermissionSet::new(vec!["admin".into()], vec!["*".into()]);
        ps.denied = vec!["memory:delete".into()];
        assert!(ps.has_permission("memory:read"));
        assert!(!ps.has_permission("memory:delete"));
    }

    #[test]
    fn permission_set_is_empty_for_no_roles() {
        let ps = PermissionSet::new(vec![], vec![]);
        assert!(ps.is_empty());
    }

    // ── ResolvedIdentity tests ──

    #[test]
    fn resolved_identity_has_role() {
        let id = ResolvedIdentity {
            user_id: "u1".into(),
            organization_id: "acme".into(),
            factory_id: None,
            roles: vec!["operator".into(), "quality_manager".into()],
            delegated_agent_id: None,
            permissions: PermissionSet::new(vec![], vec!["memory:read".into()]),
            purpose: "test".into(),
            trace_id: "t1".into(),
            resolved_at: Utc::now(),
        };
        assert!(id.has_role("operator"));
        assert!(id.has_role("quality_manager"));
        assert!(!id.has_role("admin"));
    }

    #[test]
    fn resolved_identity_belongs_to_tenant() {
        let id = ResolvedIdentity {
            user_id: "u1".into(),
            organization_id: "acme".into(),
            factory_id: None,
            roles: vec![],
            delegated_agent_id: None,
            permissions: PermissionSet::new(vec![], vec![]),
            purpose: "test".into(),
            trace_id: "t1".into(),
            resolved_at: Utc::now(),
        };
        assert!(id.belongs_to_tenant("acme"));
        assert!(!id.belongs_to_tenant("other"));
    }

    #[test]
    fn resolved_identity_system_has_all_permissions() {
        let id = ResolvedIdentity::system("acme", "trace-1");
        assert!(id.has_role("system"));
        assert!(id.has_permission("memory:read"));
        assert!(id.has_permission("memory:delete"));
        assert!(id.has_permission("anything"));
        assert_eq!(id.organization_id, "acme");
    }

    #[test]
    fn resolved_identity_delegated_agent_set() {
        let id = ResolvedIdentity {
            user_id: "u1".into(),
            organization_id: "acme".into(),
            factory_id: Some("factory-a".into()),
            roles: vec!["operator".into()],
            delegated_agent_id: Some("agent-001".into()),
            permissions: PermissionSet::new(vec!["operator".into()], vec!["memory:read".into()]),
            purpose: "task_execution".into(),
            trace_id: "t1".into(),
            resolved_at: Utc::now(),
        };
        assert_eq!(id.delegated_agent_id.as_deref(), Some("agent-001"));
        assert_eq!(id.factory_id.as_deref(), Some("factory-a"));
    }

    // ── IamClient tests ──

    #[tokio::test]
    async fn noop_iam_client_verifies_non_empty_token() {
        let client = NoopIamClient::new("acme");
        let identity = client.verify_token("user-42").await.unwrap();
        assert_eq!(identity.user_id, "user-42");
        assert_eq!(identity.organization_id, "acme");
        assert!(identity.roles.contains(&"user".to_string()));
    }

    #[tokio::test]
    async fn noop_iam_client_rejects_empty_token() {
        let client = NoopIamClient::new("acme");
        let result = client.verify_token("").await;
        assert!(matches!(result, Err(IdentityError::InvalidToken)));
    }

    // ── IdentityResolver tests (without Redis) ──

    #[tokio::test]
    async fn resolver_without_redis_calls_iam_directly() {
        let client = Arc::new(NoopIamClient::new("acme"));
        let resolver = IdentityResolver::new(client);
        let resolved = resolver
            .resolve("user-42", Some("agent-1"), "memory_read")
            .await
            .unwrap();

        assert_eq!(resolved.user_id, "user-42");
        assert_eq!(resolved.organization_id, "acme");
        assert_eq!(resolved.delegated_agent_id.as_deref(), Some("agent-1"));
        assert_eq!(resolved.purpose, "memory_read");
        assert!(!resolved.trace_id.is_empty());
    }

    #[tokio::test]
    async fn resolver_generates_unique_trace_ids() {
        let client = Arc::new(NoopIamClient::new("acme"));
        let resolver = IdentityResolver::new(client);
        let r1 = resolver.resolve("u1", None, "test").await.unwrap();
        let r2 = resolver.resolve("u1", None, "test").await.unwrap();
        assert_ne!(r1.trace_id, r2.trace_id);
    }

    #[tokio::test]
    async fn resolver_propagates_iam_errors() {
        let client = Arc::new(NoopIamClient::new("acme"));
        let resolver = IdentityResolver::new(client);
        let result = resolver.resolve("", None, "test").await;
        assert!(matches!(result, Err(IdentityError::InvalidToken)));
    }

    #[tokio::test]
    async fn resolver_invalidate_cache_without_redis_is_noop() {
        let client = Arc::new(NoopIamClient::new("acme"));
        let resolver = IdentityResolver::new(client);
        let result = resolver.invalidate_cache("u1", "acme").await;
        assert!(result.is_ok());
    }

    // ── short_hash tests ──

    #[test]
    fn short_hash_is_deterministic() {
        let h1 = short_hash("test-token");
        let h2 = short_hash("test-token");
        assert_eq!(h1, h2);
    }

    #[test]
    fn short_hash_differs_for_different_inputs() {
        let h1 = short_hash("token-a");
        let h2 = short_hash("token-b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn short_hash_is_16_hex_chars() {
        let h = short_hash("anything");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
