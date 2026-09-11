//! External system adapters — OpenClaw (§3.1) and DeerFlow (§3.2).
//!
//! These adapters bridge external agent frameworks to the Oris Enterprise
//! Context & Memory Service. Both connect through the Memory API Gateway
//! (#26) and use the Identity Resolver (#25) — they never touch PostgreSQL
//! or Redis directly.
//!
//! # Design Principles
//!
//! - **Gateway-first**: all memory operations go through the Memory API.
//! - **Identity-aware**: every call is resolved through [`IdentityResolver`].
//! - **Degradation-safe**: timeouts and partial failures degrade gracefully.
//! - **Non-invasive**: adapters wrap external systems without modifying them.

pub mod deerflow;
pub mod openclaw;

pub use deerflow::DeerFlowAdapter;
pub use openclaw::OpenClawAdapter;

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use oris_memory_contract::memory_types::{MemoryItem, Scope};

// ──────────────────────────── AdapterError ────────────────────────────

/// Errors emitted by OpenClaw and DeerFlow adapters.
#[derive(Debug, Error)]
pub enum AdapterError {
    /// Memory API Gateway returned an error or was unreachable.
    #[error("memory API error: {0}")]
    MemoryApi(String),

    /// Identity resolution failed (token invalid, user not found, etc.).
    #[error("identity resolution error: {0}")]
    Identity(String),

    /// Serialization/deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Operation exceeded the configured timeout.
    #[error("timeout after {0:?}")]
    Timeout(Duration),

    /// Assembled context exceeded the token budget.
    #[error("context budget exceeded: requested {requested}, limit {limit}")]
    BudgetExceeded {
        /// Tokens requested by the caller.
        requested: usize,
        /// Maximum tokens allowed.
        limit: usize,
    },

    /// Referenced task was not found in the shared task store.
    #[error("task not found: {0}")]
    TaskNotFound(String),

    /// Adapter is operating in degraded mode (partial failure).
    #[error("degraded: {0}")]
    Degraded(String),

    /// Invalid memory scope specified.
    #[error("invalid scope: {0}")]
    InvalidScope(String),

    /// Shared task handoff failed.
    #[error("handoff error: {0}")]
    Handoff(String),
}

// ──────────────────────────── Supporting Types ────────────────────────────

/// Provenance metadata for a candidate memory submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceInfo {
    /// Agent or user that produced this memory.
    pub source: String,
    /// Authority level of the source (e.g. "L2_verified").
    pub authority: String,
    /// Original source system reference (e.g. ticket ID, URL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
}

impl ProvenanceInfo {
    /// Create provenance from an agent source.
    pub fn from_agent(agent_id: &str) -> Self {
        Self {
            source: agent_id.to_string(),
            authority: "L3_inferred".to_string(),
            reference: None,
        }
    }
}

/// Request to submit a candidate memory to the Enterprise Memory Service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMemoryRequest {
    /// Memory content (free text or structured JSON string).
    pub content: String,
    /// Target scope for this memory.
    pub scope: Scope,
    /// Provenance metadata.
    pub provenance: ProvenanceInfo,
    /// Optional confidence score (0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Filters for pluggable memory queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryQueryFilters {
    /// Filter by tenant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    /// Filter by scope(s).
    pub scopes: Vec<Scope>,
    /// Filter by memory type(s).
    pub memory_types: Vec<String>,
    /// Minimum confidence threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,
    /// Maximum number of results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Timeout configuration for adapter operations.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Timeout for Memory API calls.
    pub api_timeout: Duration,
    /// Timeout for identity resolution.
    pub identity_timeout: Duration,
    /// Timeout for context injection.
    pub injection_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            api_timeout: Duration::from_secs(10),
            identity_timeout: Duration::from_secs(5),
            injection_timeout: Duration::from_secs(15),
        }
    }
}

/// Contract describing the memory backend capabilities required by DeerFlow.
#[derive(Debug, Clone)]
pub struct MemoryBackendContract {
    /// Whether the backend supports context injection.
    pub supports_injection: bool,
    /// Whether the backend supports result submission.
    pub supports_submission: bool,
    /// Whether the backend supports memory queries.
    pub supports_query: bool,
    /// Maximum context length in tokens.
    pub max_context_tokens: usize,
}

impl Default for MemoryBackendContract {
    fn default() -> Self {
        Self {
            supports_injection: true,
            supports_submission: true,
            supports_query: true,
            max_context_tokens: 8192,
        }
    }
}

// ──────────────────────────── MemoryApiClient Trait ────────────────────────────

/// Abstraction over the Memory API Gateway HTTP client.
///
/// OpenClaw connects through this trait — never directly to PostgreSQL/Redis.
/// In production, this is backed by `reqwest` calls to the API Gateway (#26).
/// In tests, it is mocked.
#[async_trait]
pub trait MemoryApiClient: Send + Sync {
    /// Read canonical user context as a JSON string via the Memory API.
    async fn read_user_context(
        &self,
        user_id: &str,
        task_id: Option<Uuid>,
    ) -> Result<String, AdapterError>;

    /// Submit a candidate memory; returns the assigned memory ID.
    async fn submit_candidate(
        &self,
        request: &CandidateMemoryRequest,
    ) -> Result<Uuid, AdapterError>;

    /// Promote a memory to a wider scope.
    async fn promote(&self, memory_id: Uuid, target_scope: Scope) -> Result<(), AdapterError>;

    /// Query memories with text query and filters.
    async fn query_memories(
        &self,
        query: &str,
        filters: &MemoryQueryFilters,
    ) -> Result<Vec<MemoryItem>, AdapterError>;
}

// ──────────────────────────── PluggableMemoryBackend Trait ────────────────────────────

/// Pluggable memory backend for agent frameworks (e.g. DeerFlow).
///
/// DeerFlow calls this trait at task start (inject context), task end
/// (submit results + candidates), and for ad-hoc queries. The adapter
/// handles context limiting, protocol conversion, timeout, and degradation.
///
/// # Degradation Strategy
///
/// - `inject_context` returns an empty string on timeout/failure.
/// - `submit_results` returns `Err` (propagates to caller).
/// - `query_memory` returns an empty vector on timeout/failure.
#[async_trait]
pub trait PluggableMemoryBackend: Send + Sync {
    /// Inject necessary context at task start.
    ///
    /// Returns the assembled context string. On timeout or partial failure,
    /// returns a degraded (possibly empty) string rather than an error.
    async fn inject_context(&self, task_id: Uuid, budget: usize) -> String;

    /// Submit results and memory candidates at task end.
    async fn submit_results(
        &self,
        task_id: Uuid,
        results: &str,
        memory_candidates: &[CandidateMemoryRequest],
    ) -> Result<(), AdapterError>;

    /// Pluggable memory query.
    ///
    /// Returns matching memory items. On timeout or partial failure,
    /// returns an empty vector (degraded mode).
    async fn query_memory(&self, query: &str, filters: &MemoryQueryFilters) -> Vec<MemoryItem>;
}

// ──────────────────────────── Test Support ────────────────────────────

#[cfg(test)]
pub mod test_support {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use chrono::Utc;
    use uuid::Uuid;

    use oris_memory_contract::memory_types::{
        AuthorityLevel, MemoryItem, MemoryStatus, MemoryType, PrivacyClass, Scope, SourceType,
    };

    use crate::adapters::{
        AdapterError, CandidateMemoryRequest, MemoryApiClient, MemoryQueryFilters,
    };

    /// Mock Memory API client for unit tests.
    ///
    /// Fields are immutable; construct a new instance per test scenario.
    pub struct MockMemoryApiClient {
        pub context_response: String,
        pub candidate_id: Uuid,
        pub query_results: Vec<MemoryItem>,
        pub delay_ms: u64,
        pub error_on_read: bool,
        pub error_on_promote: bool,
        pub error_on_submit: bool,
        pub error_on_query: bool,
        pub read_count: AtomicUsize,
        pub submit_count: AtomicUsize,
        pub promote_count: AtomicUsize,
    }

    impl MockMemoryApiClient {
        /// Create a mock that returns the given context response and candidate ID.
        pub fn new(
            context_response: &str,
            candidate_id: Uuid,
            query_results: Vec<MemoryItem>,
        ) -> Self {
            Self {
                context_response: context_response.to_string(),
                candidate_id,
                query_results,
                delay_ms: 0,
                error_on_read: false,
                error_on_promote: false,
                error_on_submit: false,
                error_on_query: false,
                read_count: AtomicUsize::new(0),
                submit_count: AtomicUsize::new(0),
                promote_count: AtomicUsize::new(0),
            }
        }

        /// Set a simulated delay (milliseconds) for all API calls.
        pub fn with_delay(mut self, ms: u64) -> Self {
            self.delay_ms = ms;
            self
        }

        /// Make `read_user_context` return an error.
        pub fn with_read_error(mut self) -> Self {
            self.error_on_read = true;
            self
        }

        /// Make `promote` return an error.
        pub fn with_promote_error(mut self) -> Self {
            self.error_on_promote = true;
            self
        }

        /// Make `submit_candidate` return an error.
        pub fn with_submit_error(mut self) -> Self {
            self.error_on_submit = true;
            self
        }

        /// Make `query_memories` return an error.
        pub fn with_query_error(mut self) -> Self {
            self.error_on_query = true;
            self
        }

        /// Number of times `read_user_context` was called.
        pub fn read_call_count(&self) -> usize {
            self.read_count.load(Ordering::Relaxed)
        }

        /// Number of times `submit_candidate` was called.
        pub fn submit_call_count(&self) -> usize {
            self.submit_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl MemoryApiClient for MockMemoryApiClient {
        async fn read_user_context(
            &self,
            _user_id: &str,
            _task_id: Option<Uuid>,
        ) -> Result<String, AdapterError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            self.read_count.fetch_add(1, Ordering::Relaxed);
            if self.error_on_read {
                return Err(AdapterError::MemoryApi("read failed".into()));
            }
            Ok(self.context_response.clone())
        }

        async fn submit_candidate(
            &self,
            _request: &CandidateMemoryRequest,
        ) -> Result<Uuid, AdapterError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            self.submit_count.fetch_add(1, Ordering::Relaxed);
            if self.error_on_submit {
                return Err(AdapterError::MemoryApi("submit failed".into()));
            }
            Ok(self.candidate_id)
        }

        async fn promote(
            &self,
            _memory_id: Uuid,
            _target_scope: Scope,
        ) -> Result<(), AdapterError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            self.promote_count.fetch_add(1, Ordering::Relaxed);
            if self.error_on_promote {
                return Err(AdapterError::MemoryApi("promotion denied".into()));
            }
            Ok(())
        }

        async fn query_memories(
            &self,
            _query: &str,
            _filters: &MemoryQueryFilters,
        ) -> Result<Vec<MemoryItem>, AdapterError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            if self.error_on_query {
                return Err(AdapterError::MemoryApi("query failed".into()));
            }
            Ok(self.query_results.clone())
        }
    }

    /// Construct a minimal `MemoryItem` for testing.
    pub fn make_memory_item(content: &str) -> MemoryItem {
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "acme".to_string(),
            memory_type: MemoryType::Semantic,
            scope: Scope::Enterprise,
            subject_type: None,
            subject_id: None,
            entity_refs: Vec::new(),
            content: Some(content.to_string()),
            structured_payload: None,
            embedding: None,
            source_type: SourceType::AgentInferred,
            source_reference: None,
            evidence_refs: Vec::new(),
            confidence: 0.8,
            authority_level: AuthorityLevel::L2Verified,
            importance: 0.5,
            observed_at: None,
            valid_from: None,
            valid_to: None,
            privacy_class: PrivacyClass::Internal,
            acl: serde_json::Value::Null,
            retention_policy: None,
            status: MemoryStatus::Active,
            version: 1,
            derived_from: Vec::new(),
            created_by_user: None,
            created_by_agent: Some("test-agent".to_string()),
            last_verified_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// A sample JSON context response that the Memory API would return.
    pub fn sample_context_json() -> String {
        serde_json::json!({
            "context_text": "## Canonical User Context\nUser: u1\nRole: operator",
            "token_count": 12,
            "compressed": false,
            "sources_used": ["Identity", "CanonicalUser", "SharedTask"],
            "degraded_sources": [],
            "low_confidence_items": []
        })
        .to_string()
    }
}
