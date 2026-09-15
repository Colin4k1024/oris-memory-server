//! Memory domain types shared across PostgreSQL and Redis implementations.
//!
//! These types correspond to the schema defined in the architecture document
//! (`docs/CONTROL_PLANE_ARCHITECTURE.md` §3). Originally defined in
//! `oris-memory-store`, they have been promoted to this contract crate so
//! that all workspace members can share them without circular dependencies.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// ──────────────────────────── Enums ────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Semantic,
    Episodic,
    Decision,
    Experience,
    UserPreference,
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Episodic => "episodic",
            Self::Decision => "decision",
            Self::Experience => "experience",
            Self::UserPreference => "user_preference",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Personal,
    Agent,
    Task,
    Team,
    Process,
    Factory,
    Enterprise,
}

impl Scope {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Personal => "personal",
            Self::Agent => "agent",
            Self::Task => "task",
            Self::Team => "team",
            Self::Process => "process",
            Self::Factory => "factory",
            Self::Enterprise => "enterprise",
        }
    }

    /// Parse a scope from its `as_str` representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "personal" => Some(Self::Personal),
            "agent" => Some(Self::Agent),
            "task" => Some(Self::Task),
            "team" => Some(Self::Team),
            "process" => Some(Self::Process),
            "factory" => Some(Self::Factory),
            "enterprise" => Some(Self::Enterprise),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Iam,
    Hr,
    UserExplicit,
    AgentInferred,
    ToolResult,
    BusinessEvent,
}

impl SourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Iam => "iam",
            Self::Hr => "hr",
            Self::UserExplicit => "user_explicit",
            Self::AgentInferred => "agent_inferred",
            Self::ToolResult => "tool_result",
            Self::BusinessEvent => "business_event",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityLevel {
    L0SourceOfTruth,
    L1Authoritative,
    L2Verified,
    L3Inferred,
}

impl AuthorityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::L0SourceOfTruth => "L0_source_of_truth",
            Self::L1Authoritative => "L1_authoritative",
            Self::L2Verified => "L2_verified",
            Self::L3Inferred => "L3_inferred",
        }
    }

    /// Higher number = more authoritative. Used for conflict resolution.
    pub fn rank(&self) -> u8 {
        match self {
            Self::L3Inferred => 0,
            Self::L2Verified => 1,
            Self::L1Authoritative => 2,
            Self::L0SourceOfTruth => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyClass {
    Public,
    Internal,
    Confidential,
    Restricted,
}

impl PrivacyClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Internal => "internal",
            Self::Confidential => "confidential",
            Self::Restricted => "restricted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Candidate,
    Active,
    Archived,
    Revoked,
    Quarantined,
}

impl MemoryStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Active => "active",
            Self::Archived => "archived",
            Self::Revoked => "revoked",
            Self::Quarantined => "quarantined",
        }
    }
}

// ──────────────────────────── MemoryItem ────────────────────────

/// A single memory record — the unified storage unit.
///
/// `memory_type` is a field, not a separate database. Semantic, episodic,
/// decision, experience and user preference memories all live in this table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub memory_id: Uuid,
    pub tenant_id: String,
    pub memory_type: MemoryType,
    pub scope: Scope,
    pub subject_type: Option<String>,
    pub subject_id: Option<String>,
    pub entity_refs: Vec<Value>,
    pub content: Option<String>,
    pub structured_payload: Option<Value>,
    pub embedding: Option<Vec<f32>>,
    pub source_type: SourceType,
    pub source_reference: Option<String>,
    pub evidence_refs: Vec<Value>,
    pub confidence: f32,
    pub authority_level: AuthorityLevel,
    pub importance: f32,
    pub observed_at: Option<DateTime<Utc>>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub privacy_class: PrivacyClass,
    pub acl: Value,
    pub retention_policy: Option<String>,
    pub status: MemoryStatus,
    pub version: i32,
    pub derived_from: Vec<Value>,
    pub created_by_user: Option<String>,
    pub created_by_agent: Option<String>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Builder for creating a new memory item candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCandidate {
    pub tenant_id: String,
    pub memory_type: MemoryType,
    pub scope: Scope,
    pub subject_type: Option<String>,
    pub subject_id: Option<String>,
    pub content: Option<String>,
    pub structured_payload: Option<Value>,
    pub source_type: SourceType,
    pub source_reference: Option<String>,
    pub confidence: f32,
    pub authority_level: AuthorityLevel,
    pub privacy_class: PrivacyClass,
    pub created_by_user: Option<String>,
    pub created_by_agent: Option<String>,
}

// ──────────────────────────── CanonicalUserProfile ──────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalUserProfile {
    pub user_id: String,
    pub organization_id: String,
    pub factory_id: Option<String>,
    pub identity_links: Vec<Value>,
    pub role: Option<String>,
    pub position: Option<String>,
    pub language: Option<String>,
    pub timezone: Option<String>,
    pub preferences: Value,
    /// Explicit preferences set by the user or authoritative systems (IAM/HR).
    /// Strong consistency, versioned, auditable. Takes priority over inferred.
    #[serde(default)]
    pub explicit_preferences: Value,
    /// Inferred preferences projected from Mem0 or behavioral analysis.
    /// Eventually consistent, carries confidence, can be corrected/forgotten.
    /// Must NOT override HR/IAM master data or explicit user settings.
    #[serde(default)]
    pub inferred_preferences: Value,
    pub common_entities: Vec<Value>,
    pub active_projects: Vec<Value>,
    pub consent_scope: Value,
    pub privacy_class: PrivacyClass,
    pub source: SourceType,
    pub authority_level: AuthorityLevel,
    pub version: i32,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

// ──────────────────────────── SharedTaskContext ─────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedTaskContext {
    pub task_id: Uuid,
    pub parent_task_id: Option<Uuid>,
    pub initiator_user_id: String,
    pub organization_scope: String,
    pub goal: String,
    pub constraints: Vec<Value>,
    pub success_criteria: Vec<Value>,
    pub entities: Vec<Value>,
    pub business_refs: Vec<Value>,
    pub current_findings: Vec<Value>,
    pub evidence_refs: Vec<Value>,
    pub decisions: Vec<Value>,
    pub assumptions: Vec<Value>,
    pub completed_steps: Vec<Value>,
    pub pending_steps: Vec<Value>,
    pub current_owner_agent: Option<String>,
    pub participant_agents: Vec<Value>,
    pub artifact_refs: Vec<Value>,
    pub source_system_refs: Vec<Value>,
    pub status: String,
    pub version: i32,
    pub expires_at: Option<DateTime<Utc>>,
    pub acl: Value,
    pub privacy_class: PrivacyClass,
    pub audit_ref: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ──────────────────────────── OutboxEvent ───────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    UserContextUpdated,
    PreferenceUpdated,
    TaskContextUpdated,
    MemoryPromoted,
    MemoryRevoked,
    MemoryExpired,
    PermissionChanged,
    EngineProjectionFailed,
    MemoryProposed,
    MemoryQuarantined,
    ExperienceOutcomeRecorded,
}

impl EventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UserContextUpdated => "USER_CONTEXT_UPDATED",
            Self::PreferenceUpdated => "PREFERENCE_UPDATED",
            Self::TaskContextUpdated => "TASK_CONTEXT_UPDATED",
            Self::MemoryPromoted => "MEMORY_PROMOTED",
            Self::MemoryRevoked => "MEMORY_REVOKED",
            Self::MemoryExpired => "MEMORY_EXPIRED",
            Self::PermissionChanged => "PERMISSION_CHANGED",
            Self::EngineProjectionFailed => "ENGINE_PROJECTION_FAILED",
            Self::MemoryProposed => "MEMORY_PROPOSED",
            Self::MemoryQuarantined => "MEMORY_QUARANTINED",
            Self::ExperienceOutcomeRecorded => "EXPERIENCE_OUTCOME_RECORDED",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutboxStatus {
    Pending,
    Processing,
    Done,
    Failed,
}

impl OutboxStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboxEvent {
    pub event_id: Uuid,
    pub event_type: EventType,
    pub aggregate_id: String,
    pub payload: Value,
    pub status: OutboxStatus,
    pub created_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
}

// ──────────────────────────── Search ───────────────────────────

/// Parameters for hybrid (structured + keyword + vector) search.
#[derive(Debug, Clone, Default)]
pub struct SearchParams {
    pub tenant_id: String,
    pub query_text: Option<String>,
    pub memory_types: Vec<MemoryType>,
    pub scopes: Vec<Scope>,
    pub subject_id: Option<String>,
    pub min_confidence: Option<f32>,
    pub limit: i64,
    pub offset: i64,
    pub embedding: Option<Vec<f32>>,
    pub vector_weight: f32,
    pub keyword_weight: f32,
    pub authority_weight: f32,
    pub freshness_weight: f32,
}

impl SearchParams {
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            limit: 20,
            vector_weight: 0.4,
            keyword_weight: 0.3,
            authority_weight: 0.2,
            freshness_weight: 0.1,
            ..Default::default()
        }
    }
}

/// A single search result with relevance score and the underlying memory item.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub item: MemoryItem,
    pub score: f64,
    pub matched_by: MatchType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    Structured,
    Keyword,
    Vector,
    Hybrid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_from_str_round_trip() {
        for scope in [
            Scope::Personal,
            Scope::Agent,
            Scope::Task,
            Scope::Team,
            Scope::Process,
            Scope::Factory,
            Scope::Enterprise,
        ] {
            let s = scope.as_str();
            assert_eq!(Scope::from_str(s), Some(scope));
        }
        assert_eq!(Scope::from_str("unknown"), None);
    }
}
