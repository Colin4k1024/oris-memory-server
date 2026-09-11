//! RBAC + ABAC permission engine.
//!
//! Combines role-based access control (RBAC) with attribute-based access
//! control (ABAC).  The [`AclEngine`] is initialised with a mapping of role
//! names to permission strings (e.g. `"memory:read"`).  At check-time the
//! engine evaluates the [`AccessRequest`] through an ordered pipeline:
//!
//! 1. **Deny-list** — explicit denials always override allows.
//! 2. **Owner self-read** — owners can always read their own personal-scope
//!    memories.
//! 3. **Restricted privacy** — `Restricted` content requires an explicit
//!    permission grant, regardless of role.
//! 4. **Enterprise scope** — requires an enterprise-level role.
//! 5. **Task scope** — only task participants (owner, subject, or explicit
//!    grant) may access.
//! 6. **Explicit permission** — a direct allow in the request's permission set.
//! 7. **RBAC** — any of the user's roles grants the required permission.
//! 8. **Default deny** — if nothing matched, deny.

use std::collections::{HashMap, HashSet};

use oris_memory_store::memory_types::{PrivacyClass, Scope};
use serde::{Deserialize, Serialize};

/// The set of permissions resolved for a single identity.
///
/// Carries the user's roles together with any explicit allow/deny overrides
/// that were assigned out-of-band (e.g. by a policy administrator).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionSet {
    pub roles: Vec<String>,
    pub explicit_permissions: HashSet<String>,
    pub denied_permissions: HashSet<String>,
}

impl PermissionSet {
    /// Create a permission set from a list of role names.
    pub fn from_roles(roles: Vec<String>) -> Self {
        Self {
            roles,
            ..Default::default()
        }
    }

    /// Grant an explicit permission (builder style).
    pub fn allow(mut self, perm: &str) -> Self {
        self.explicit_permissions.insert(perm.to_string());
        self
    }

    /// Deny a permission (builder style).
    pub fn deny(mut self, perm: &str) -> Self {
        self.denied_permissions.insert(perm.to_string());
        self
    }
}

/// Actions that can be authorised against a memory item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Read,
    Write,
    Promote,
    Revoke,
    Forget,
    Verify,
    Share,
}

impl Action {
    /// The canonical permission string for this action (e.g. `"memory:read"`).
    pub fn as_permission(&self) -> &'static str {
        match self {
            Self::Read => "memory:read",
            Self::Write => "memory:write",
            Self::Promote => "memory:promote",
            Self::Revoke => "memory:revoke",
            Self::Forget => "memory:forget",
            Self::Verify => "memory:verify",
            Self::Share => "memory:share",
        }
    }

    /// Short display name (e.g. `"read"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Promote => "promote",
            Self::Revoke => "revoke",
            Self::Forget => "forget",
            Self::Verify => "verify",
            Self::Share => "share",
        }
    }
}

/// A request for access to a memory item, carrying all attributes needed for
/// both RBAC and ABAC evaluation.
#[derive(Debug, Clone)]
pub struct AccessRequest {
    pub tenant_id: String,
    pub user_id: String,
    pub roles: Vec<String>,
    pub action: Action,
    pub scope: Scope,
    pub privacy_class: PrivacyClass,
    pub memory_owner: Option<String>,
    pub subject_type: Option<String>,
    pub subject_id: Option<String>,
    /// Explicitly-granted permission strings (ABAC / out-of-band grants).
    pub explicit_permissions: HashSet<String>,
    /// Explicitly-denied permission strings.  Always overrides allow.
    pub denied_permissions: HashSet<String>,
}

/// The outcome of an access check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessDecision {
    /// Access is granted.
    Allow { reason: String },
    /// Access is refused.
    Deny { reason: String },
}

impl AccessDecision {
    /// Returns `true` if this is an `Allow` decision.
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow { .. })
    }

    /// Returns `true` if this is a `Deny` decision.
    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }
}

/// The permission engine combining RBAC and ABAC.
pub struct AclEngine {
    /// Role → list of permission strings granted by that role.
    role_permissions: HashMap<String, Vec<String>>,
}

impl AclEngine {
    /// Create a new engine from a role-permissions mapping.
    pub fn new(role_permissions: HashMap<String, Vec<String>>) -> Self {
        Self { role_permissions }
    }

    /// Build an [`AccessRequest`] from a [`PermissionSet`] and context fields.
    ///
    /// Convenience method for callers that resolve a user's `PermissionSet`
    /// before calling [`check`](Self::check).
    #[allow(clippy::too_many_arguments)]
    pub fn build_request(
        &self,
        perms: &PermissionSet,
        tenant_id: impl Into<String>,
        user_id: impl Into<String>,
        action: Action,
        scope: Scope,
        privacy_class: PrivacyClass,
        memory_owner: Option<String>,
        subject_type: Option<String>,
        subject_id: Option<String>,
    ) -> AccessRequest {
        AccessRequest {
            tenant_id: tenant_id.into(),
            user_id: user_id.into(),
            roles: perms.roles.clone(),
            action,
            scope,
            privacy_class,
            memory_owner,
            subject_type,
            subject_id,
            explicit_permissions: perms.explicit_permissions.clone(),
            denied_permissions: perms.denied_permissions.clone(),
        }
    }

    /// Evaluate an access request through the full RBAC + ABAC pipeline.
    ///
    /// See the module-level documentation for the evaluation order.
    pub fn check(&self, req: &AccessRequest) -> AccessDecision {
        let perm = req.action.as_permission();

        // ── 1. Deny-list overrides everything ──────────────────────
        if req.denied_permissions.contains(perm) {
            return AccessDecision::Deny {
                reason: format!("permission '{}' is explicitly denied", perm),
            };
        }

        // ── 2. Owner self-read on personal scope ───────────────────
        if req.action == Action::Read
            && req.scope == Scope::Personal
            && req.memory_owner.as_deref() == Some(&req.user_id)
        {
            return AccessDecision::Allow {
                reason: "owner reading own personal-scope memory".into(),
            };
        }

        // ── 3. Restricted privacy requires explicit permission ──────
        if req.privacy_class == PrivacyClass::Restricted && !req.explicit_permissions.contains(perm)
        {
            return AccessDecision::Deny {
                reason: "restricted privacy class requires explicit permission".into(),
            };
        }

        // ── 4. Enterprise scope requires an enterprise-level role ───
        if req.scope == Scope::Enterprise {
            let has_enterprise_role = req
                .roles
                .iter()
                .any(|r| r.to_lowercase().contains("enterprise"));
            if !has_enterprise_role {
                return AccessDecision::Deny {
                    reason: "enterprise scope requires an enterprise-level role".into(),
                };
            }
        }

        // ── 5. Task scope: only task participants ──────────────────
        //   Participants (owner, subject, or explicit grantee) are allowed;
        //   everyone else is denied — even with a role that would normally
        //   grant the permission.
        if req.scope == Scope::Task {
            let is_owner = req.memory_owner.as_deref() == Some(&req.user_id);
            let has_explicit = req.explicit_permissions.contains(perm);
            let is_subject = req.subject_id.as_deref() == Some(&req.user_id);
            if is_owner || has_explicit || is_subject {
                return AccessDecision::Allow {
                    reason: "task participant access granted".into(),
                };
            }
            return AccessDecision::Deny {
                reason: "task scope requires task participation".into(),
            };
        }

        // ── 6. Explicit permission ─────────────────────────────────
        if req.explicit_permissions.contains(perm) {
            return AccessDecision::Allow {
                reason: "explicit permission granted".into(),
            };
        }

        // ── 7. RBAC: does any role grant this permission? ──────────
        for role in &req.roles {
            if let Some(perms) = self.role_permissions.get(role) {
                if perms.iter().any(|p| p == perm) {
                    return AccessDecision::Allow {
                        reason: format!("role '{}' grants '{}'", role, perm),
                    };
                }
            }
        }

        // ── 8. Default deny ────────────────────────────────────────
        AccessDecision::Deny {
            reason: "no matching permission found".into(),
        }
    }
}

// ────────────────────────── Tests ──────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oris_memory_store::memory_types::{PrivacyClass, Scope};
    use std::collections::HashMap;

    /// Helper: build an engine with a few standard roles.
    fn test_engine() -> AclEngine {
        let mut map = HashMap::new();
        map.insert("viewer".into(), vec!["memory:read".into()]);
        map.insert(
            "editor".into(),
            vec!["memory:read".into(), "memory:write".into()],
        );
        map.insert(
            "admin".into(),
            vec![
                "memory:read".into(),
                "memory:write".into(),
                "memory:promote".into(),
                "memory:revoke".into(),
                "memory:forget".into(),
                "memory:verify".into(),
                "memory:share".into(),
            ],
        );
        map.insert(
            "enterprise_admin".into(),
            vec![
                "memory:read".into(),
                "memory:write".into(),
                "memory:promote".into(),
            ],
        );
        AclEngine::new(map)
    }

    /// Helper: build a basic request.
    fn req(
        user: &str,
        roles: &[&str],
        action: Action,
        scope: Scope,
        privacy: PrivacyClass,
        owner: Option<&str>,
    ) -> AccessRequest {
        AccessRequest {
            tenant_id: "acme".into(),
            user_id: user.into(),
            roles: roles.iter().map(|s| s.to_string()).collect(),
            action,
            scope,
            privacy_class: privacy,
            memory_owner: owner.map(String::from),
            subject_type: None,
            subject_id: None,
            explicit_permissions: HashSet::new(),
            denied_permissions: HashSet::new(),
        }
    }

    #[test]
    fn owner_can_read_own_personal_memory() {
        let engine = test_engine();
        let r = req(
            "alice",
            &[],
            Action::Read,
            Scope::Personal,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r).is_allow());
    }

    #[test]
    fn non_owner_denied_personal_without_role() {
        let engine = test_engine();
        let r = req(
            "bob",
            &[],
            Action::Read,
            Scope::Personal,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r).is_deny());
    }

    #[test]
    fn role_based_allow_for_read() {
        let engine = test_engine();
        let r = req(
            "bob",
            &["viewer"],
            Action::Read,
            Scope::Team,
            PrivacyClass::Internal,
            Some("alice"),
        );
        let decision = engine.check(&r);
        assert!(decision.is_allow());
        assert!(decision.clone().is_allow());
    }

    #[test]
    fn deny_list_overrides_role_allow() {
        let engine = test_engine();
        let mut r = req(
            "bob",
            &["viewer"],
            Action::Read,
            Scope::Team,
            PrivacyClass::Internal,
            Some("alice"),
        );
        r.denied_permissions.insert("memory:read".into());
        let decision = engine.check(&r);
        assert!(decision.is_deny());
    }

    #[test]
    fn restricted_requires_explicit_permission() {
        let engine = test_engine();
        // Even with admin role, restricted content needs explicit grant
        let r = req(
            "bob",
            &["admin"],
            Action::Read,
            Scope::Team,
            PrivacyClass::Restricted,
            Some("alice"),
        );
        assert!(engine.check(&r).is_deny());
    }

    #[test]
    fn restricted_with_explicit_permission_allowed() {
        let engine = test_engine();
        let mut r = req(
            "bob",
            &["admin"],
            Action::Read,
            Scope::Team,
            PrivacyClass::Restricted,
            Some("alice"),
        );
        r.explicit_permissions.insert("memory:read".into());
        assert!(engine.check(&r).is_allow());
    }

    #[test]
    fn enterprise_scope_requires_enterprise_role() {
        let engine = test_engine();
        // admin role grants the permission but lacks enterprise scope
        let r = req(
            "bob",
            &["admin"],
            Action::Read,
            Scope::Enterprise,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r).is_deny());
    }

    #[test]
    fn enterprise_scope_with_enterprise_role_allowed() {
        let engine = test_engine();
        let r = req(
            "bob",
            &["enterprise_admin"],
            Action::Read,
            Scope::Enterprise,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r).is_allow());
    }

    #[test]
    fn task_scope_denied_for_non_participant() {
        let engine = test_engine();
        // viewer has read permission but is not a task participant
        let r = req(
            "bob",
            &["viewer"],
            Action::Read,
            Scope::Task,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r).is_deny());
    }

    #[test]
    fn task_scope_allowed_for_owner() {
        let engine = test_engine();
        let r = req(
            "alice",
            &[],
            Action::Read,
            Scope::Task,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r).is_allow());
    }

    #[test]
    fn task_scope_allowed_for_subject_user() {
        let engine = test_engine();
        let mut r = req(
            "bob",
            &["viewer"],
            Action::Read,
            Scope::Task,
            PrivacyClass::Internal,
            Some("alice"),
        );
        r.subject_id = Some("bob".into());
        assert!(engine.check(&r).is_allow());
    }

    #[test]
    fn task_scope_allowed_with_explicit_permission() {
        let engine = test_engine();
        let mut r = req(
            "bob",
            &["viewer"],
            Action::Read,
            Scope::Task,
            PrivacyClass::Internal,
            Some("alice"),
        );
        r.explicit_permissions.insert("memory:read".into());
        assert!(engine.check(&r).is_allow());
    }

    #[test]
    fn write_requires_write_permission() {
        let engine = test_engine();
        // viewer only has read
        let r = req(
            "bob",
            &["viewer"],
            Action::Write,
            Scope::Team,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r).is_deny());
        // editor has write
        let r2 = req(
            "bob",
            &["editor"],
            Action::Write,
            Scope::Team,
            PrivacyClass::Internal,
            Some("alice"),
        );
        assert!(engine.check(&r2).is_allow());
    }

    #[test]
    fn owner_read_bypasses_restricted_for_personal_scope() {
        let engine = test_engine();
        // Owner reading their own personal restricted memory should be allowed
        let r = req(
            "alice",
            &[],
            Action::Read,
            Scope::Personal,
            PrivacyClass::Restricted,
            Some("alice"),
        );
        assert!(engine.check(&r).is_allow());
    }

    #[test]
    fn permission_set_builder_works() {
        let perms = PermissionSet::from_roles(vec!["viewer".into()])
            .allow("memory:promote")
            .deny("memory:forget");
        assert!(perms.explicit_permissions.contains("memory:promote"));
        assert!(perms.denied_permissions.contains("memory:forget"));
        assert_eq!(perms.roles, vec!["viewer".to_string()]);
    }
}
