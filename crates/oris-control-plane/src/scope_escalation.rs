//! Scope Escalation — explicit promotion of memories through widening scope
//! levels.
//!
//! Implements the promotion ladder from architecture doc §4.2: memories must
//! go through explicit promotion before wider sharing.  The upgrade path is:
//!
//! ```text
//! Personal → Task → Team → Factory → Enterprise
//! ```
//!
//! Higher scope levels require more authority, more evidence, and stricter
//! review.  Scope promotion is **orthogonal** to lifecycle promotion
//! (`candidate → stable`): a memory's scope describes *who can see it*, while
//! its lifecycle status describes *how mature it is*.
//!
//! Cross-factory and enterprise promotions require an explicit approval
//! workflow ([`ApprovalWorkflow`]) before the scope change is applied.
//!
//! # Design
//!
//! - [`EscalationPolicy`] is a pure rule set — fully unit-testable.
//! - [`ApprovalWorkflow`] is an in-memory state machine — fully unit-testable.
//! - [`ScopeEscalationManager`] orchestrates a [`MemoryScopeStore`] (an
//!   `async_trait`), the policy, and the workflow.  The default
//!   [`PgMemoryScopeStore`] hits PostgreSQL; tests supply an in-memory mock.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use oris_memory_store::memory_types::{MemoryItem, Scope};
use oris_memory_store::postgres::{MemoryRepo, MemoryRepoError, Pool};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;
use sqlx::Row;

// ─────────────────────── Scope ladder helpers ──────────────────

/// Rank of a scope on the escalation ladder.
///
/// Returns `None` for scopes that are not on the promotion path (`Agent`,
/// `Process`).  `Enterprise` is the top of the ladder.
fn ladder_rank(scope: Scope) -> Option<u8> {
    match scope {
        Scope::Personal => Some(0),
        Scope::Task => Some(1),
        Scope::Team => Some(2),
        Scope::Factory => Some(3),
        Scope::Enterprise => Some(4),
        Scope::Agent | Scope::Process => None,
    }
}

/// The next scope up the escalation ladder, or `None` if already at the top
/// (or off the ladder entirely).
fn next_scope(scope: Scope) -> Option<Scope> {
    match scope {
        Scope::Personal => Some(Scope::Task),
        Scope::Task => Some(Scope::Team),
        Scope::Team => Some(Scope::Factory),
        Scope::Factory => Some(Scope::Enterprise),
        Scope::Enterprise => None,
        Scope::Agent | Scope::Process => None,
    }
}

// ─────────────────────────── Errors ────────────────────────────

/// Errors arising during scope escalation.
#[derive(Debug, Error)]
pub enum EscalationError {
    #[error("memory item {0} not found")]
    MemoryNotFound(Uuid),

    #[error("scope {0:?} is not on the escalation ladder (Personal → Task → Team → Factory → Enterprise)")]
    ScopeNotOnLadder(Scope),

    #[error("scope {0:?} is already at the top of the escalation ladder")]
    AlreadyAtTop(Scope),

    #[error("cannot promote from {from:?} to {to:?}: promotions advance one level at a time")]
    NonAdjacentPromotion { from: Scope, to: Scope },

    #[error("no escalation rule defined for {from:?} → {to:?}")]
    NoRuleFound { from: Scope, to: Scope },

    #[error(
        "confidence {confidence:.3} is below the minimum {min_confidence:.3} required for {from:?} → {to:?}"
    )]
    InsufficientConfidence {
        from: Scope,
        to: Scope,
        confidence: f32,
        min_confidence: f32,
    },

    #[error(
        "evidence count {count} is below the minimum {min_evidence} required for {from:?} → {to:?}"
    )]
    InsufficientEvidence {
        from: Scope,
        to: Scope,
        count: usize,
        min_evidence: usize,
    },

    #[error("approval request {0} not found")]
    ApprovalNotFound(Uuid),

    #[error("approval request {id} is already {status:?}, cannot change")]
    ApprovalAlreadyDecided { id: Uuid, status: ApprovalStatus },

    #[error("approval request {0} has expired")]
    ApprovalExpired(Uuid),

    #[error("store error: {0}")]
    Store(String),

    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("repo error: {0}")]
    Repo(#[from] MemoryRepoError),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("invalid scope string: {0}")]
    InvalidScope(String),

    #[error("invalid approver role string: {0}")]
    InvalidApproverRole(String),

    #[error("invalid approval status string: {0}")]
    InvalidApprovalStatus(String),
}

// ───────────────────────── Approver role ──────────────────────

/// The role authorised to approve a given scope transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApproverRole {
    /// Self-service — no external approval required.
    SelfService,
    /// Team lead approves Task → Team.
    TeamLead,
    /// Factory manager approves Team → Factory.
    FactoryManager,
    /// Enterprise admin approves Factory → Enterprise.
    EnterpriseAdmin,
}

impl ApproverRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SelfService => "self_service",
            Self::TeamLead => "team_lead",
            Self::FactoryManager => "factory_manager",
            Self::EnterpriseAdmin => "enterprise_admin",
        }
    }

    /// Whether this role requires an external approval step.
    pub fn requires_approval(&self) -> bool {
        !matches!(self, Self::SelfService)
    }

    /// Parse an approver role from its `as_str` representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "self_service" => Some(Self::SelfService),
            "team_lead" => Some(Self::TeamLead),
            "factory_manager" => Some(Self::FactoryManager),
            "enterprise_admin" => Some(Self::EnterpriseAdmin),
            _ => None,
        }
    }
}

// ─────────────────────── Escalation policy ────────────────────

/// A single rule governing one scope transition.
#[derive(Debug, Clone)]
pub struct ScopeRule {
    pub from: Scope,
    pub to: Scope,
    pub min_confidence: f32,
    pub min_evidence_refs: usize,
    pub approver_role: ApproverRole,
    pub requires_audit_trail: bool,
}

/// The full set of rules governing scope promotions.
///
/// Defaults match architecture doc §4.2:
///
/// | Transition | Min confidence | Min evidence | Approver | Audit |
/// |------------|---------------|--------------|----------|-------|
/// | Personal→Task | 0.5 | 0 | self-service | no |
/// | Task→Team | 0.7 | 0 | team lead | no |
/// | Team→Factory | 0.8 | 3 | factory manager | no |
/// | Factory→Enterprise | 0.9 | 5 | enterprise admin | yes |
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    rules: Vec<ScopeRule>,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            rules: vec![
                ScopeRule {
                    from: Scope::Personal,
                    to: Scope::Task,
                    min_confidence: 0.5,
                    min_evidence_refs: 0,
                    approver_role: ApproverRole::SelfService,
                    requires_audit_trail: false,
                },
                ScopeRule {
                    from: Scope::Task,
                    to: Scope::Team,
                    min_confidence: 0.7,
                    min_evidence_refs: 0,
                    approver_role: ApproverRole::TeamLead,
                    requires_audit_trail: false,
                },
                ScopeRule {
                    from: Scope::Team,
                    to: Scope::Factory,
                    min_confidence: 0.8,
                    min_evidence_refs: 3,
                    approver_role: ApproverRole::FactoryManager,
                    requires_audit_trail: false,
                },
                ScopeRule {
                    from: Scope::Factory,
                    to: Scope::Enterprise,
                    min_confidence: 0.9,
                    min_evidence_refs: 5,
                    approver_role: ApproverRole::EnterpriseAdmin,
                    requires_audit_trail: true,
                },
            ],
        }
    }
}

/// The outcome of validating a promotion request against the policy.
#[derive(Debug, Clone)]
pub struct AssessmentOutcome {
    /// True when an external approval must be obtained before the scope
    /// change can be applied.
    pub approval_required: bool,
    pub approver_role: ApproverRole,
    pub requires_audit_trail: bool,
    pub rule: ScopeRule,
}

impl EscalationPolicy {
    /// Create a policy with custom rules.
    pub fn new(rules: Vec<ScopeRule>) -> Self {
        Self { rules }
    }

    /// Look up the rule for a specific transition.
    pub fn rule_for(&self, from: Scope, to: Scope) -> Option<&ScopeRule> {
        self.rules.iter().find(|r| r.from == from && r.to == to)
    }

    /// Validate a promotion request against the policy.
    ///
    /// Checks (in order):
    /// 1. The current scope is on the escalation ladder.
    /// 2. The target scope is the *next* level (one step at a time).
    /// 3. A rule exists for the transition.
    /// 4. The memory's confidence meets the minimum.
    /// 5. The memory has enough evidence references.
    ///
    /// Returns [`AssessmentOutcome`] on success (with `approval_required`
    /// indicating whether an approval must still be obtained) or an
    /// [`EscalationError`] describing the first failure.
    pub fn assess(
        &self,
        request: &ScopePromotionRequest,
        current_scope: Scope,
        confidence: f32,
        evidence_count: usize,
    ) -> Result<AssessmentOutcome, EscalationError> {
        // 1. Current scope must be on the ladder.
        ladder_rank(current_scope).ok_or(EscalationError::ScopeNotOnLadder(current_scope))?;

        // 2. Target must be the immediate next level.
        let expected =
            next_scope(current_scope).ok_or(EscalationError::AlreadyAtTop(current_scope))?;
        if request.target_scope != expected {
            return Err(EscalationError::NonAdjacentPromotion {
                from: current_scope,
                to: request.target_scope,
            });
        }

        // 3. A rule must exist for this transition.
        let rule = self
            .rule_for(current_scope, request.target_scope)
            .ok_or(EscalationError::NoRuleFound {
                from: current_scope,
                to: request.target_scope,
            })?
            .clone();

        // 4. Confidence threshold.
        if confidence < rule.min_confidence {
            return Err(EscalationError::InsufficientConfidence {
                from: current_scope,
                to: request.target_scope,
                confidence,
                min_confidence: rule.min_confidence,
            });
        }

        // 5. Evidence threshold.
        if evidence_count < rule.min_evidence_refs {
            return Err(EscalationError::InsufficientEvidence {
                from: current_scope,
                to: request.target_scope,
                count: evidence_count,
                min_evidence: rule.min_evidence_refs,
            });
        }

        Ok(AssessmentOutcome {
            approval_required: rule.approver_role.requires_approval(),
            approver_role: rule.approver_role,
            requires_audit_trail: rule.requires_audit_trail,
            rule,
        })
    }
}

// ─────────────────────── Approval workflow ─────────────────────

/// Lifecycle state of an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

impl ApprovalStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
            Self::Expired => "expired",
        }
    }

    /// Parse an approval status from its `as_str` representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "approved" => Some(Self::Approved),
            "rejected" => Some(Self::Rejected),
            "expired" => Some(Self::Expired),
            _ => None,
        }
    }
}

/// A single approval request for a cross-scope promotion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: Uuid,
    pub memory_id: Uuid,
    pub target_scope: Scope,
    pub requester: String,
    pub justification: String,
    pub approver_role: ApproverRole,
    pub status: ApprovalStatus,
    pub created_at: DateTime<Utc>,
    pub decided_at: Option<DateTime<Utc>>,
    pub decided_by: Option<String>,
    pub expires_at: DateTime<Utc>,
}

/// In-memory approval workflow for cross-factory / enterprise promotions.
///
/// Pending requests expire after a configurable TTL (default 24 h).  The
/// workflow is intentionally in-memory: in a production deployment the
/// [`ScopeEscalationManager`] could persist requests to a table, but the
/// state-machine semantics (submit → approve/reject → expire) remain the same.
pub struct ApprovalWorkflow {
    ttl: Duration,
    requests: Mutex<HashMap<Uuid, ApprovalRequest>>,
}

impl Default for ApprovalWorkflow {
    fn default() -> Self {
        Self::new(Duration::hours(24))
    }
}

impl ApprovalWorkflow {
    /// Create a workflow with a custom approval TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            requests: Mutex::new(HashMap::new()),
        }
    }

    /// Submit a new pending approval request.
    pub fn submit(
        &self,
        memory_id: Uuid,
        target_scope: Scope,
        requester: &str,
        justification: &str,
        approver_role: ApproverRole,
    ) -> ApprovalRequest {
        let now = Utc::now();
        let request = ApprovalRequest {
            id: Uuid::new_v4(),
            memory_id,
            target_scope,
            requester: requester.to_string(),
            justification: justification.to_string(),
            approver_role,
            status: ApprovalStatus::Pending,
            created_at: now,
            decided_at: None,
            decided_by: None,
            expires_at: now + self.ttl,
        };
        let mut requests = self.lock();
        requests.insert(request.id, request.clone());
        request
    }

    /// Fetch a request by ID.
    pub fn get(&self, id: Uuid) -> Option<ApprovalRequest> {
        self.lock().get(&id).cloned()
    }

    /// Find an *approved* request matching the memory + target scope.
    pub fn find_approved(&self, memory_id: Uuid, target_scope: Scope) -> Option<ApprovalRequest> {
        self.lock()
            .values()
            .find(|r| {
                r.memory_id == memory_id
                    && r.target_scope == target_scope
                    && r.status == ApprovalStatus::Approved
            })
            .cloned()
    }

    /// Approve a pending request.
    pub fn approve(&self, id: Uuid, approver: &str) -> Result<ApprovalRequest, EscalationError> {
        let mut requests = self.lock();
        let request = requests
            .get_mut(&id)
            .ok_or(EscalationError::ApprovalNotFound(id))?;
        if request.status == ApprovalStatus::Expired {
            return Err(EscalationError::ApprovalExpired(id));
        }
        if request.status != ApprovalStatus::Pending {
            return Err(EscalationError::ApprovalAlreadyDecided {
                id,
                status: request.status,
            });
        }
        request.status = ApprovalStatus::Approved;
        request.decided_at = Some(Utc::now());
        request.decided_by = Some(approver.to_string());
        Ok(request.clone())
    }

    /// Reject a pending request.
    pub fn reject(&self, id: Uuid, approver: &str) -> Result<ApprovalRequest, EscalationError> {
        let mut requests = self.lock();
        let request = requests
            .get_mut(&id)
            .ok_or(EscalationError::ApprovalNotFound(id))?;
        if request.status == ApprovalStatus::Expired {
            return Err(EscalationError::ApprovalExpired(id));
        }
        if request.status != ApprovalStatus::Pending {
            return Err(EscalationError::ApprovalAlreadyDecided {
                id,
                status: request.status,
            });
        }
        request.status = ApprovalStatus::Rejected;
        request.decided_at = Some(Utc::now());
        request.decided_by = Some(approver.to_string());
        Ok(request.clone())
    }

    /// Expire all pending requests whose TTL has elapsed.  Returns the newly
    /// expired requests.
    pub fn expire_pending(&self) -> Vec<ApprovalRequest> {
        let now = Utc::now();
        let mut requests = self.lock();
        let mut expired = Vec::new();
        for request in requests.values_mut() {
            if request.status == ApprovalStatus::Pending && request.expires_at <= now {
                request.status = ApprovalStatus::Expired;
                expired.push(request.clone());
            }
        }
        expired
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Uuid, ApprovalRequest>> {
        self.requests
            .lock()
            .expect("approval workflow mutex poisoned")
    }
}

// ─────────────────── Promotion request / result ────────────────

/// A request to promote a memory to a wider scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopePromotionRequest {
    pub memory_id: Uuid,
    pub target_scope: Scope,
    pub requester: String,
    pub justification: String,
    pub evidence_refs: Vec<serde_json::Value>,
}

/// The outcome of a [`ScopeEscalationManager::promote`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopePromotionResult {
    /// `true` when the scope change was actually applied.
    pub success: bool,
    /// The new scope, if the promotion succeeded.
    pub new_scope: Option<Scope>,
    /// `true` when an approval is required before the change can be applied.
    pub approval_required: bool,
    /// The ID of the (pending or approved) approval request, if any.
    pub approval_id: Option<Uuid>,
    pub message: String,
}

// ──────────────────── Memory scope store trait ─────────────────

/// A minimal view of a memory's scope-related fields, used by the escalation
/// manager to make policy decisions without pulling the full [`MemoryItem`].
#[derive(Debug, Clone)]
pub struct MemoryScopeInfo {
    pub memory_id: Uuid,
    pub scope: Scope,
    pub confidence: f32,
    pub evidence_refs: Vec<serde_json::Value>,
    pub tenant_id: String,
    pub created_by_user: Option<String>,
}

impl From<&MemoryItem> for MemoryScopeInfo {
    fn from(item: &MemoryItem) -> Self {
        Self {
            memory_id: item.memory_id,
            scope: item.scope,
            confidence: item.confidence,
            evidence_refs: item.evidence_refs.clone(),
            tenant_id: item.tenant_id.clone(),
            created_by_user: item.created_by_user.clone(),
        }
    }
}

/// Abstracts the persistence operations the escalation manager needs.
///
/// The default implementation [`PgMemoryScopeStore`] wraps a PostgreSQL pool;
/// tests supply an in-memory mock so [`ScopeEscalationManager::promote`] can
/// be exercised end-to-end without a database.
#[async_trait]
pub trait MemoryScopeStore: Send + Sync {
    /// Fetch the scope-related fields for a memory.
    async fn get_memory(&self, memory_id: Uuid)
        -> Result<Option<MemoryScopeInfo>, EscalationError>;
    /// Persist a new scope for a memory.
    async fn update_scope(&self, memory_id: Uuid, new_scope: Scope) -> Result<(), EscalationError>;
}

/// PostgreSQL-backed [`MemoryScopeStore`].
pub struct PgMemoryScopeStore {
    pool: Pool,
}

impl PgMemoryScopeStore {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl MemoryScopeStore for PgMemoryScopeStore {
    async fn get_memory(
        &self,
        memory_id: Uuid,
    ) -> Result<Option<MemoryScopeInfo>, EscalationError> {
        let repo = MemoryRepo::new(self.pool.clone());
        let item = repo.get_by_id(memory_id).await?;
        Ok(item.as_ref().map(MemoryScopeInfo::from))
    }

    async fn update_scope(&self, memory_id: Uuid, new_scope: Scope) -> Result<(), EscalationError> {
        let result = sqlx::query(
            r#"UPDATE memory_item
                  SET scope = $2, version = version + 1, updated_at = NOW()
                WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .bind(new_scope.as_str())
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(EscalationError::MemoryNotFound(memory_id));
        }
        Ok(())
    }
}

// ─────────────────────── ApprovalStore trait ──────────────────

/// Persists approval requests for scope promotions (§4.2, P1-5).
///
/// The default production implementation [`PgApprovalStore`] persists to
/// the `approval_request` PostgreSQL table.  Tests use
/// [`InMemoryApprovalStore`] so [`ScopeEscalationManager::promote`] can be
/// exercised without a database.
#[async_trait]
pub trait ApprovalStore: Send + Sync {
    /// Insert a new approval request.
    async fn insert(&self, request: &ApprovalRequest) -> Result<(), EscalationError>;

    /// Fetch a request by ID.
    async fn get(&self, id: Uuid) -> Result<Option<ApprovalRequest>, EscalationError>;

    /// Find an *approved* request matching the memory + target scope.
    async fn find_approved(
        &self,
        memory_id: Uuid,
        target_scope: Scope,
    ) -> Result<Option<ApprovalRequest>, EscalationError>;

    /// Update the status of a request (approve / reject / expire).
    async fn update_status(
        &self,
        id: Uuid,
        status: ApprovalStatus,
        decided_by: &str,
    ) -> Result<ApprovalRequest, EscalationError>;

    /// Expire all pending requests whose TTL has elapsed.  Returns the
    /// newly expired requests.
    async fn expire_pending(&self) -> Result<Vec<ApprovalRequest>, EscalationError>;
}

/// PostgreSQL-backed [`ApprovalStore`].
pub struct PgApprovalStore {
    pool: Pool,
}

impl PgApprovalStore {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    fn row_to_request(row: &sqlx::postgres::PgRow) -> Result<ApprovalRequest, EscalationError> {
        let status_str: String = row
            .try_get("status")
            .map_err(|e| EscalationError::Database(e))?;
        let scope_str: String = row
            .try_get("target_scope")
            .map_err(|e| EscalationError::Database(e))?;
        let role_str: String = row
            .try_get("approver_role")
            .map_err(|e| EscalationError::Database(e))?;

        let target_scope = Scope::from_str(&scope_str)
            .ok_or_else(|| EscalationError::InvalidScope(scope_str.clone()))?;
        let approver_role = ApproverRole::from_str(&role_str)
            .ok_or_else(|| EscalationError::InvalidApproverRole(role_str.clone()))?;
        let status = ApprovalStatus::from_str(&status_str)
            .ok_or_else(|| EscalationError::InvalidApprovalStatus(status_str.clone()))?;

        Ok(ApprovalRequest {
            id: row.try_get("id").map_err(|e| EscalationError::Database(e))?,
            memory_id: row
                .try_get("memory_id")
                .map_err(|e| EscalationError::Database(e))?,
            target_scope,
            requester: row
                .try_get("requester")
                .map_err(|e| EscalationError::Database(e))?,
            justification: row
                .try_get("justification")
                .unwrap_or_default(),
            approver_role,
            status,
            created_at: row
                .try_get("created_at")
                .map_err(|e| EscalationError::Database(e))?,
            decided_at: row
                .try_get("decided_at")
                .ok(),
            decided_by: row.try_get("decided_by").ok(),
            expires_at: row
                .try_get("expires_at")
                .map_err(|e| EscalationError::Database(e))?,
        })
    }
}

#[async_trait]
impl ApprovalStore for PgApprovalStore {
    async fn insert(&self, request: &ApprovalRequest) -> Result<(), EscalationError> {
        sqlx::query(
            r#"INSERT INTO approval_request
                  (id, memory_id, target_scope, requester, justification,
                   approver_role, status, created_at, decided_at, decided_by,
                   expires_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"#,
        )
        .bind(request.id)
        .bind(request.memory_id)
        .bind(request.target_scope.as_str())
        .bind(&request.requester)
        .bind(&request.justification)
        .bind(request.approver_role.as_str())
        .bind(request.status.as_str())
        .bind(request.created_at)
        .bind(request.decided_at)
        .bind(&request.decided_by)
        .bind(request.expires_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<ApprovalRequest>, EscalationError> {
        let row = sqlx::query(r#"SELECT * FROM approval_request WHERE id = $1"#)
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| Self::row_to_request(&r)).transpose()
    }

    async fn find_approved(
        &self,
        memory_id: Uuid,
        target_scope: Scope,
    ) -> Result<Option<ApprovalRequest>, EscalationError> {
        let row = sqlx::query(
            r#"SELECT * FROM approval_request
                WHERE memory_id = $1 AND target_scope = $2 AND status = 'approved'
                ORDER BY decided_at DESC LIMIT 1"#,
        )
        .bind(memory_id)
        .bind(target_scope.as_str())
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| Self::row_to_request(&r)).transpose()
    }

    async fn update_status(
        &self,
        id: Uuid,
        status: ApprovalStatus,
        decided_by: &str,
    ) -> Result<ApprovalRequest, EscalationError> {
        let row = sqlx::query(
            r#"UPDATE approval_request
                  SET status = $2, decided_at = NOW(), decided_by = $3
                WHERE id = $1
            RETURNING *"#,
        )
        .bind(id)
        .bind(status.as_str())
        .bind(decided_by)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|r| Self::row_to_request(&r))
            .transpose()?
            .ok_or(EscalationError::ApprovalNotFound(id))
    }

    async fn expire_pending(&self) -> Result<Vec<ApprovalRequest>, EscalationError> {
        let rows = sqlx::query(
            r#"UPDATE approval_request
                  SET status = 'expired'
                WHERE status = 'pending' AND expires_at <= NOW()
            RETURNING *"#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(Self::row_to_request).collect()
    }
}

/// In-memory [`ApprovalStore`] for testing.
pub struct InMemoryApprovalStore {
    requests: tokio::sync::Mutex<HashMap<Uuid, ApprovalRequest>>,
}

impl Default for InMemoryApprovalStore {
    fn default() -> Self {
        Self {
            requests: tokio::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl InMemoryApprovalStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ApprovalStore for InMemoryApprovalStore {
    async fn insert(&self, request: &ApprovalRequest) -> Result<(), EscalationError> {
        self.requests
            .lock()
            .await
            .insert(request.id, request.clone());
        Ok(())
    }

    async fn get(&self, id: Uuid) -> Result<Option<ApprovalRequest>, EscalationError> {
        Ok(self.requests.lock().await.get(&id).cloned())
    }

    async fn find_approved(
        &self,
        memory_id: Uuid,
        target_scope: Scope,
    ) -> Result<Option<ApprovalRequest>, EscalationError> {
        Ok(self
            .requests
            .lock()
            .await
            .values()
            .find(|r| {
                r.memory_id == memory_id
                    && r.target_scope == target_scope
                    && r.status == ApprovalStatus::Approved
            })
            .cloned())
    }

    async fn update_status(
        &self,
        id: Uuid,
        status: ApprovalStatus,
        decided_by: &str,
    ) -> Result<ApprovalRequest, EscalationError> {
        let mut requests = self.requests.lock().await;
        let request = requests
            .get_mut(&id)
            .ok_or(EscalationError::ApprovalNotFound(id))?;
        if request.status == ApprovalStatus::Expired {
            return Err(EscalationError::ApprovalExpired(id));
        }
        if request.status != ApprovalStatus::Pending {
            return Err(EscalationError::ApprovalAlreadyDecided {
                id,
                status: request.status,
            });
        }
        request.status = status;
        request.decided_at = Some(Utc::now());
        request.decided_by = Some(decided_by.to_string());
        Ok(request.clone())
    }

    async fn expire_pending(&self) -> Result<Vec<ApprovalRequest>, EscalationError> {
        let now = Utc::now();
        let mut requests = self.requests.lock().await;
        let mut expired = Vec::new();
        for request in requests.values_mut() {
            if request.status == ApprovalStatus::Pending && request.expires_at <= now {
                request.status = ApprovalStatus::Expired;
                expired.push(request.clone());
            }
        }
        Ok(expired)
    }
}

// ─────────────────────── Escalation manager ────────────────────

/// Orchestrates scope promotions: fetches the memory, validates the request
/// against [`EscalationPolicy`], routes cross-scope promotions through
/// [`ApprovalStore`], and persists the new scope via [`MemoryScopeStore`].
pub struct ScopeEscalationManager<S: MemoryScopeStore = PgMemoryScopeStore> {
    store: S,
    policy: EscalationPolicy,
    approval_store: Arc<dyn ApprovalStore>,
}

impl ScopeEscalationManager<PgMemoryScopeStore> {
    /// Create a manager backed by a PostgreSQL pool with default policy.
    pub fn new(pool: Pool) -> Self {
        let scope_store = PgMemoryScopeStore::new(pool.clone());
        let approval_store: Arc<dyn ApprovalStore> = Arc::new(PgApprovalStore::new(pool));
        Self::with_stores(scope_store, approval_store, EscalationPolicy::default())
    }
}

impl<S: MemoryScopeStore> ScopeEscalationManager<S> {
    /// Create a manager with a custom store and policy.
    /// Uses an in-memory approval store (suitable for tests).
    pub fn with_store(store: S, policy: EscalationPolicy) -> Self {
        Self {
            store,
            policy,
            approval_store: Arc::new(InMemoryApprovalStore::new()),
        }
    }

    /// Create a manager with explicit scope and approval stores.
    pub fn with_stores(
        store: S,
        approval_store: Arc<dyn ApprovalStore>,
        policy: EscalationPolicy,
    ) -> Self {
        Self {
            store,
            policy,
            approval_store,
        }
    }

    /// Replace the policy (builder-style).
    pub fn with_policy(mut self, policy: EscalationPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn policy(&self) -> &EscalationPolicy {
        &self.policy
    }

    /// Access the underlying approval store.
    pub fn approval_store(&self) -> &Arc<dyn ApprovalStore> {
        &self.approval_store
    }

    /// Promote a memory to a wider scope.
    ///
    /// - For **self-service** transitions (Personal → Task) the scope is
    ///   applied immediately and the result reports `success = true`.
    /// - For **cross-scope** transitions the manager first looks for an
    ///   already-approved request; if found, the scope is applied.  Otherwise
    ///   a pending approval request is created and the result reports
    ///   `approval_required = true` with the new `approval_id`.
    ///
    /// Hard validation failures (memory not found, insufficient confidence,
    /// non-adjacent promotion, …) are returned as [`EscalationError`].
    pub async fn promote(
        &self,
        request: &ScopePromotionRequest,
    ) -> Result<ScopePromotionResult, EscalationError> {
        let memory = self
            .store
            .get_memory(request.memory_id)
            .await?
            .ok_or(EscalationError::MemoryNotFound(request.memory_id))?;

        let outcome = self.policy.assess(
            request,
            memory.scope,
            memory.confidence,
            memory.evidence_refs.len(),
        )?;

        // Self-service transitions apply immediately.
        if !outcome.approval_required {
            self.store
                .update_scope(request.memory_id, request.target_scope)
                .await?;
            return Ok(ScopePromotionResult {
                success: true,
                new_scope: Some(request.target_scope),
                approval_required: false,
                approval_id: None,
                message: format!(
                    "scope promoted from {:?} to {:?}",
                    memory.scope, request.target_scope
                ),
            });
        }

        // Cross-scope: apply immediately if an approval is already on record.
        if let Some(approval) = self
            .approval_store
            .find_approved(request.memory_id, request.target_scope)
            .await?
        {
            self.store
                .update_scope(request.memory_id, request.target_scope)
                .await?;
            return Ok(ScopePromotionResult {
                success: true,
                new_scope: Some(request.target_scope),
                approval_required: true,
                approval_id: Some(approval.id),
                message: format!("scope promoted with approval {}", approval.id),
            });
        }

        // Otherwise create a pending approval request and wait.
        let now = Utc::now();
        let approval = ApprovalRequest {
            id: Uuid::new_v4(),
            memory_id: request.memory_id,
            target_scope: request.target_scope,
            requester: request.requester.clone(),
            justification: request.justification.clone(),
            approver_role: outcome.approver_role,
            status: ApprovalStatus::Pending,
            created_at: now,
            decided_at: None,
            decided_by: None,
            expires_at: now + Duration::hours(24),
        };
        self.approval_store.insert(&approval).await?;
        Ok(ScopePromotionResult {
            success: false,
            new_scope: None,
            approval_required: true,
            approval_id: Some(approval.id),
            message: format!(
                "approval required ({}): pending request {}",
                outcome.approver_role.as_str(),
                approval.id
            ),
        })
    }

    /// Approve a pending promotion request.
    pub async fn approve_promotion(
        &self,
        approval_id: Uuid,
        approver: &str,
    ) -> Result<ApprovalRequest, EscalationError> {
        self.approval_store
            .update_status(approval_id, ApprovalStatus::Approved, approver)
            .await
    }

    /// Reject a pending promotion request.
    pub async fn reject_promotion(
        &self,
        approval_id: Uuid,
        approver: &str,
    ) -> Result<ApprovalRequest, EscalationError> {
        self.approval_store
            .update_status(approval_id, ApprovalStatus::Rejected, approver)
            .await
    }

    /// Expire all pending approval requests whose TTL has elapsed.
    pub async fn expire_pending_promotions(&self) -> Result<Vec<ApprovalRequest>, EscalationError> {
        self.approval_store.expire_pending().await
    }
}

// ─────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────

    fn make_memory(scope: Scope, confidence: f32, evidence_count: usize) -> MemoryScopeInfo {
        MemoryScopeInfo {
            memory_id: Uuid::new_v4(),
            scope,
            confidence,
            evidence_refs: (0..evidence_count)
                .map(|i| serde_json::json!({ "ref": i }))
                .collect(),
            tenant_id: "acme".to_string(),
            created_by_user: Some("user1".to_string()),
        }
    }

    fn make_request(memory_id: Uuid, target: Scope) -> ScopePromotionRequest {
        ScopePromotionRequest {
            memory_id,
            target_scope: target,
            requester: "user1".to_string(),
            justification: "test justification".to_string(),
            evidence_refs: Vec::new(),
        }
    }

    /// In-memory [`MemoryScopeStore`] for exercising the manager end-to-end.
    struct MockStore {
        memories: Mutex<HashMap<Uuid, MemoryScopeInfo>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                memories: Mutex::new(HashMap::new()),
            }
        }
        fn insert(&self, info: MemoryScopeInfo) {
            self.memories
                .lock()
                .expect("mock mutex")
                .insert(info.memory_id, info);
        }
    }

    #[async_trait]
    impl MemoryScopeStore for MockStore {
        async fn get_memory(
            &self,
            memory_id: Uuid,
        ) -> Result<Option<MemoryScopeInfo>, EscalationError> {
            Ok(self
                .memories
                .lock()
                .expect("mock mutex")
                .get(&memory_id)
                .cloned())
        }

        async fn update_scope(
            &self,
            memory_id: Uuid,
            new_scope: Scope,
        ) -> Result<(), EscalationError> {
            let mut memories = self.memories.lock().expect("mock mutex");
            let memory = memories
                .get_mut(&memory_id)
                .ok_or(EscalationError::MemoryNotFound(memory_id))?;
            memory.scope = new_scope;
            Ok(())
        }
    }

    // ── ladder helpers ─────────────────────────────────────────

    #[test]
    fn ladder_rank_orders_escalation_path() {
        assert_eq!(ladder_rank(Scope::Personal), Some(0));
        assert_eq!(ladder_rank(Scope::Task), Some(1));
        assert_eq!(ladder_rank(Scope::Team), Some(2));
        assert_eq!(ladder_rank(Scope::Factory), Some(3));
        assert_eq!(ladder_rank(Scope::Enterprise), Some(4));
    }

    #[test]
    fn ladder_rank_rejects_off_ladder_scopes() {
        assert_eq!(ladder_rank(Scope::Agent), None);
        assert_eq!(ladder_rank(Scope::Process), None);
    }

    #[test]
    fn next_scope_walks_ladder() {
        assert_eq!(next_scope(Scope::Personal), Some(Scope::Task));
        assert_eq!(next_scope(Scope::Task), Some(Scope::Team));
        assert_eq!(next_scope(Scope::Team), Some(Scope::Factory));
        assert_eq!(next_scope(Scope::Factory), Some(Scope::Enterprise));
        assert_eq!(next_scope(Scope::Enterprise), None);
        assert_eq!(next_scope(Scope::Agent), None);
        assert_eq!(next_scope(Scope::Process), None);
    }

    // ── policy rules ───────────────────────────────────────────

    #[test]
    fn policy_default_rules_match_doc() {
        let policy = EscalationPolicy::default();

        let personal_task = policy.rule_for(Scope::Personal, Scope::Task).unwrap();
        assert_eq!(personal_task.min_confidence, 0.5);
        assert_eq!(personal_task.min_evidence_refs, 0);
        assert_eq!(personal_task.approver_role, ApproverRole::SelfService);
        assert!(!personal_task.requires_audit_trail);

        let task_team = policy.rule_for(Scope::Task, Scope::Team).unwrap();
        assert_eq!(task_team.min_confidence, 0.7);
        assert_eq!(task_team.approver_role, ApproverRole::TeamLead);

        let team_factory = policy.rule_for(Scope::Team, Scope::Factory).unwrap();
        assert_eq!(team_factory.min_confidence, 0.8);
        assert_eq!(team_factory.min_evidence_refs, 3);
        assert_eq!(team_factory.approver_role, ApproverRole::FactoryManager);

        let factory_ent = policy.rule_for(Scope::Factory, Scope::Enterprise).unwrap();
        assert_eq!(factory_ent.min_confidence, 0.9);
        assert_eq!(factory_ent.min_evidence_refs, 5);
        assert_eq!(factory_ent.approver_role, ApproverRole::EnterpriseAdmin);
        assert!(factory_ent.requires_audit_trail);
    }

    #[test]
    fn policy_rule_for_unknown_transition_returns_none() {
        let policy = EscalationPolicy::default();
        // Skipping a level is not a single-step rule.
        assert!(policy.rule_for(Scope::Personal, Scope::Team).is_none());
        // Backwards.
        assert!(policy.rule_for(Scope::Enterprise, Scope::Factory).is_none());
        // Off-ladder.
        assert!(policy.rule_for(Scope::Agent, Scope::Task).is_none());
    }

    // ── policy assess ──────────────────────────────────────────

    #[test]
    fn assess_self_service_personal_to_task_succeeds() {
        let policy = EscalationPolicy::default();
        let mem = make_memory(Scope::Personal, 0.6, 0);
        let req = make_request(mem.memory_id, Scope::Task);
        let outcome = policy
            .assess(&req, mem.scope, mem.confidence, 0)
            .expect("self-service promotion should pass");
        assert!(!outcome.approval_required);
        assert_eq!(outcome.approver_role, ApproverRole::SelfService);
    }

    #[test]
    fn assess_low_confidence_blocks_personal_to_task() {
        let policy = EscalationPolicy::default();
        let req = make_request(Uuid::new_v4(), Scope::Task);
        let err = policy.assess(&req, Scope::Personal, 0.3, 0).unwrap_err();
        assert!(matches!(
            err,
            EscalationError::InsufficientConfidence {
                min_confidence: 0.5,
                ..
            }
        ));
    }

    #[test]
    fn assess_non_adjacent_promotion_rejected() {
        let policy = EscalationPolicy::default();
        // Personal → Team skips Task.
        let req = make_request(Uuid::new_v4(), Scope::Team);
        let err = policy.assess(&req, Scope::Personal, 0.99, 10).unwrap_err();
        assert!(matches!(err, EscalationError::NonAdjacentPromotion { .. }));
    }

    #[test]
    fn assess_off_ladder_scope_rejected() {
        let policy = EscalationPolicy::default();
        let req = make_request(Uuid::new_v4(), Scope::Task);
        let err = policy.assess(&req, Scope::Agent, 0.99, 10).unwrap_err();
        assert!(matches!(err, EscalationError::ScopeNotOnLadder(_)));
    }

    #[test]
    fn assess_enterprise_is_already_at_top() {
        let policy = EscalationPolicy::default();
        let req = make_request(Uuid::new_v4(), Scope::Enterprise);
        let err = policy
            .assess(&req, Scope::Enterprise, 0.99, 10)
            .unwrap_err();
        assert!(matches!(err, EscalationError::AlreadyAtTop(_)));
    }

    #[test]
    fn assess_task_to_team_requires_approval() {
        let policy = EscalationPolicy::default();
        let req = make_request(Uuid::new_v4(), Scope::Team);
        let outcome = policy.assess(&req, Scope::Task, 0.75, 0).unwrap();
        assert!(outcome.approval_required);
        assert_eq!(outcome.approver_role, ApproverRole::TeamLead);
    }

    #[test]
    fn assess_team_to_factory_insufficient_evidence() {
        let policy = EscalationPolicy::default();
        let req = make_request(Uuid::new_v4(), Scope::Factory);
        let err = policy.assess(&req, Scope::Team, 0.85, 2).unwrap_err();
        assert!(matches!(
            err,
            EscalationError::InsufficientEvidence {
                count: 2,
                min_evidence: 3,
                ..
            }
        ));
    }

    #[test]
    fn assess_team_to_factory_succeeds_with_enough_evidence() {
        let policy = EscalationPolicy::default();
        let req = make_request(Uuid::new_v4(), Scope::Factory);
        let outcome = policy.assess(&req, Scope::Team, 0.85, 3).unwrap();
        assert!(outcome.approval_required);
        assert_eq!(outcome.approver_role, ApproverRole::FactoryManager);
    }

    #[test]
    fn assess_factory_to_enterprise_requires_audit_trail() {
        let policy = EscalationPolicy::default();
        let req = make_request(Uuid::new_v4(), Scope::Enterprise);
        let outcome = policy.assess(&req, Scope::Factory, 0.95, 5).unwrap();
        assert!(outcome.requires_audit_trail);
        assert_eq!(outcome.approver_role, ApproverRole::EnterpriseAdmin);
    }

    // ── approval workflow ──────────────────────────────────────

    #[test]
    fn workflow_submit_creates_pending_request() {
        let wf = ApprovalWorkflow::default();
        let req = wf.submit(
            Uuid::new_v4(),
            Scope::Team,
            "user1",
            "justification",
            ApproverRole::TeamLead,
        );
        assert_eq!(req.status, ApprovalStatus::Pending);
        assert!(wf.get(req.id).is_some());
    }

    #[test]
    fn workflow_approve_transitions_to_approved() {
        let wf = ApprovalWorkflow::default();
        let req = wf.submit(
            Uuid::new_v4(),
            Scope::Team,
            "user1",
            "j",
            ApproverRole::TeamLead,
        );
        let approved = wf.approve(req.id, "lead1").unwrap();
        assert_eq!(approved.status, ApprovalStatus::Approved);
        assert_eq!(approved.decided_by.as_deref(), Some("lead1"));
        assert!(approved.decided_at.is_some());
    }

    #[test]
    fn workflow_reject_transitions_to_rejected() {
        let wf = ApprovalWorkflow::default();
        let req = wf.submit(
            Uuid::new_v4(),
            Scope::Team,
            "user1",
            "j",
            ApproverRole::TeamLead,
        );
        let rejected = wf.reject(req.id, "lead1").unwrap();
        assert_eq!(rejected.status, ApprovalStatus::Rejected);
        assert_eq!(rejected.decided_by.as_deref(), Some("lead1"));
    }

    #[test]
    fn workflow_cannot_decide_already_decided() {
        let wf = ApprovalWorkflow::default();
        let req = wf.submit(
            Uuid::new_v4(),
            Scope::Team,
            "user1",
            "j",
            ApproverRole::TeamLead,
        );
        wf.approve(req.id, "lead1").unwrap();
        let err = wf.reject(req.id, "lead1").unwrap_err();
        assert!(matches!(
            err,
            EscalationError::ApprovalAlreadyDecided { .. }
        ));
    }

    #[test]
    fn workflow_approve_unknown_id_fails() {
        let wf = ApprovalWorkflow::default();
        let err = wf.approve(Uuid::new_v4(), "x").unwrap_err();
        assert!(matches!(err, EscalationError::ApprovalNotFound(_)));
    }

    #[test]
    fn workflow_expire_pending_expires_old_requests() {
        // Negative TTL → already expired at submission time.
        let wf = ApprovalWorkflow::new(Duration::seconds(-1));
        let req = wf.submit(
            Uuid::new_v4(),
            Scope::Team,
            "user1",
            "j",
            ApproverRole::TeamLead,
        );
        let expired = wf.expire_pending();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].status, ApprovalStatus::Expired);
        // An expired request can no longer be approved.
        let err = wf.approve(req.id, "lead1").unwrap_err();
        assert!(matches!(err, EscalationError::ApprovalExpired(_)));
    }

    // ── manager end-to-end (mock store) ────────────────────────

    #[tokio::test]
    async fn promote_self_service_applies_immediately() {
        let store = MockStore::new();
        let mem = make_memory(Scope::Personal, 0.6, 0);
        let id = mem.memory_id;
        store.insert(mem);
        let mgr = ScopeEscalationManager::with_store(store, EscalationPolicy::default());
        let req = make_request(id, Scope::Task);
        let result = mgr.promote(&req).await.unwrap();
        assert!(result.success);
        assert!(!result.approval_required);
        assert_eq!(result.new_scope, Some(Scope::Task));
        // Scope was actually persisted in the mock store.
        let updated = mgr.store.get_memory(id).await.unwrap().unwrap();
        assert_eq!(updated.scope, Scope::Task);
    }

    #[tokio::test]
    async fn promote_cross_scope_requires_approval_then_applies() {
        let store = MockStore::new();
        let mem = make_memory(Scope::Task, 0.75, 0);
        let id = mem.memory_id;
        store.insert(mem);
        let mgr = ScopeEscalationManager::with_store(store, EscalationPolicy::default());
        let req = make_request(id, Scope::Team);

        // First call: approval required, scope unchanged.
        let result = mgr.promote(&req).await.unwrap();
        assert!(!result.success);
        assert!(result.approval_required);
        let approval_id = result.approval_id.unwrap();

        // Approve, then promote again — scope is applied.
        let approved = mgr.approve_promotion(approval_id, "lead1").await.unwrap();
        assert_eq!(approved.status, ApprovalStatus::Approved);
        let result2 = mgr.promote(&req).await.unwrap();
        assert!(result2.success);
        assert_eq!(result2.new_scope, Some(Scope::Team));
        assert_eq!(result2.approval_id, Some(approval_id));

        let updated = mgr.store.get_memory(id).await.unwrap().unwrap();
        assert_eq!(updated.scope, Scope::Team);
    }

    #[tokio::test]
    async fn promote_rejected_approval_does_not_apply() {
        let store = MockStore::new();
        let mem = make_memory(Scope::Task, 0.75, 0);
        let id = mem.memory_id;
        store.insert(mem);
        let mgr = ScopeEscalationManager::with_store(store, EscalationPolicy::default());
        let req = make_request(id, Scope::Team);

        let result = mgr.promote(&req).await.unwrap();
        let approval_id = result.approval_id.unwrap();
        mgr.reject_promotion(approval_id, "lead1").await.unwrap();

        // Retrying creates a fresh pending request (rejected one is not "approved").
        let result2 = mgr.promote(&req).await.unwrap();
        assert!(!result2.success);
        assert!(result2.approval_required);
        assert_ne!(result2.approval_id, Some(approval_id));

        // Scope unchanged.
        let mem2 = mgr.store.get_memory(id).await.unwrap().unwrap();
        assert_eq!(mem2.scope, Scope::Task);
    }

    #[tokio::test]
    async fn promote_nonexistent_memory_fails() {
        let store = MockStore::new();
        let mgr = ScopeEscalationManager::with_store(store, EscalationPolicy::default());
        let req = make_request(Uuid::new_v4(), Scope::Task);
        let err = mgr.promote(&req).await.unwrap_err();
        assert!(matches!(err, EscalationError::MemoryNotFound(_)));
    }

    #[tokio::test]
    async fn promote_low_confidence_fails() {
        let store = MockStore::new();
        let mem = make_memory(Scope::Personal, 0.2, 0); // below 0.5
        let id = mem.memory_id;
        store.insert(mem);
        let mgr = ScopeEscalationManager::with_store(store, EscalationPolicy::default());
        let req = make_request(id, Scope::Task);
        let err = mgr.promote(&req).await.unwrap_err();
        assert!(matches!(
            err,
            EscalationError::InsufficientConfidence { .. }
        ));
    }

    // ── serde / misc ───────────────────────────────────────────

    #[test]
    fn approver_role_as_str_and_requires_approval() {
        assert_eq!(ApproverRole::SelfService.as_str(), "self_service");
        assert!(!ApproverRole::SelfService.requires_approval());
        assert_eq!(ApproverRole::TeamLead.as_str(), "team_lead");
        assert!(ApproverRole::TeamLead.requires_approval());
        assert_eq!(ApproverRole::FactoryManager.as_str(), "factory_manager");
        assert_eq!(ApproverRole::EnterpriseAdmin.as_str(), "enterprise_admin");
    }

    #[test]
    fn promotion_request_and_result_serde_round_trip() {
        let req = ScopePromotionRequest {
            memory_id: Uuid::new_v4(),
            target_scope: Scope::Team,
            requester: "user1".into(),
            justification: "because".into(),
            evidence_refs: vec![serde_json::json!({ "a": 1 })],
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ScopePromotionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.target_scope, Scope::Team);
        assert_eq!(back.requester, "user1");

        let res = ScopePromotionResult {
            success: true,
            new_scope: Some(Scope::Enterprise),
            approval_required: true,
            approval_id: Some(Uuid::new_v4()),
            message: "ok".into(),
        };
        let json = serde_json::to_string(&res).unwrap();
        let back: ScopePromotionResult = serde_json::from_str(&json).unwrap();
        assert!(back.success);
        assert_eq!(back.new_scope, Some(Scope::Enterprise));
    }

    #[test]
    fn approval_status_serde_round_trip() {
        for status in [
            ApprovalStatus::Pending,
            ApprovalStatus::Approved,
            ApprovalStatus::Rejected,
            ApprovalStatus::Expired,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: ApprovalStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }
}
