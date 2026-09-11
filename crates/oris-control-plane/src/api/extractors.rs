//! Request body types, response DTOs, error mapping, and request-context
//! extraction for the V1 REST API (§5.1, §10.2).
//!
//! # Layout
//!
//! - [`ApiError`] — unified handler error with HTTP status mapping and
//!   `From` conversions for every domain error type the handlers touch.
//! - [`RequestContext`] — tenant / user / agent / purpose / task / trace
//!   context populated by the §26 API-gateway middleware (via request
//!   extensions) or extracted directly from `x-*` headers as a fallback.
//! - Request body DTOs — typed deserialisers for each `POST`/`PATCH` body.
//! - Response DTOs — serialisable shapes that hide internal fields (e.g.
//!   embeddings) and add API-level metadata.

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use uuid::Uuid;

use chrono::{DateTime, Utc};

use oris_memory_contract::memory_types::{
    AuthorityLevel, CanonicalUserProfile, MemoryItem, MemoryStatus, MemoryType, PrivacyClass,
    Scope, SharedTaskContext, SourceType,
};

use crate::api::gateway::ApiContract;
use crate::canonical_user::CanonicalUserError;
use crate::context_assembler::{ConflictFlag, ContextSource};
use crate::governance::{ForgetError, VersionError};
use crate::identity::{PermissionSet, ResolvedIdentity};
use crate::rerank::RerankedResult;
use crate::shared_task::SharedTaskError;
use crate::write_pipeline::{SafetyVerdictSummary, WritePipelineError, WriteResult};
use oris_memory_store::postgres::{MemoryRepoError, SearchError};

// ═══════════════════════════ ApiError ═══════════════════════════

/// Unified error type for all V1 REST handlers.
///
/// Maps to HTTP status codes via [`IntoResponse`]. Domain errors are
/// converted through `From` impls so handlers can use `?` directly.
#[derive(Debug, Error)]
pub enum ApiError {
    /// Resource does not exist → `404 Not Found`.
    #[error("not found: {0}")]
    NotFound(String),

    /// Caller is not authenticated → `401 Unauthorized`.
    #[error("unauthorized: {0}")]
    Unauthorized(String),

    /// Caller is authenticated but lacks permission → `403 Forbidden`.
    #[error("forbidden: {0}")]
    Forbidden(String),

    /// State conflict (e.g. duplicate, version mismatch) → `409 Conflict`.
    #[error("conflict: {0}")]
    Conflict(String),

    /// Malformed or invalid request → `400 Bad Request`.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// Unexpected internal failure → `500 Internal Server Error`.
    #[error("internal error: {0}")]
    Internal(String),
}

impl ApiError {
    /// Convenience constructor for [`ApiError::Internal`].
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Convenience constructor for [`ApiError::BadRequest`].
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }

    /// Convenience constructor for [`ApiError::NotFound`].
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    /// HTTP status code for this error variant.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::Forbidden(_) => StatusCode::FORBIDDEN,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Stable, UPPER_SNAKE_CASE error code for the JSON response body.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "NOT_FOUND",
            Self::Unauthorized(_) => "UNAUTHORIZED",
            Self::Forbidden(_) => "FORBIDDEN",
            Self::Conflict(_) => "CONFLICT",
            Self::BadRequest(_) => "BAD_REQUEST",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = json!({
            "error": self.to_string(),
            "error_code": self.error_code(),
        });
        Response::builder()
            .status(status)
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }
}

// ────────────────────── Domain error → ApiError conversions ──────────────

impl From<MemoryRepoError> for ApiError {
    fn from(err: MemoryRepoError) -> Self {
        match err {
            MemoryRepoError::NotFound => Self::NotFound("memory item not found".into()),
            MemoryRepoError::Database(e) => Self::internal(format!("database error: {e}")),
        }
    }
}

impl From<WritePipelineError> for ApiError {
    fn from(err: WritePipelineError) -> Self {
        match err {
            WritePipelineError::PoisonBlocked { detail } => {
                Self::BadRequest(format!("content blocked by poison guard: {detail}"))
            }
            WritePipelineError::Storage(msg) => Self::internal(format!("storage error: {msg}")),
            WritePipelineError::Outbox(msg) => Self::internal(format!("outbox error: {msg}")),
            WritePipelineError::Serialization(e) => {
                Self::internal(format!("serialization error: {e}"))
            }
            WritePipelineError::Database(e) => Self::internal(format!("database error: {e}")),
        }
    }
}

impl From<VersionError> for ApiError {
    fn from(err: VersionError) -> Self {
        match err {
            VersionError::NotFound(id) => Self::not_found(format!("memory {id} not found")),
            VersionError::VersionNotFound { memory_id, version } => Self::not_found(format!(
                "version {version} not found for memory {memory_id}"
            )),
            VersionError::Database(e) => Self::internal(format!("database error: {e}")),
            VersionError::Serialization(e) => Self::internal(format!("serialization error: {e}")),
        }
    }
}

impl From<ForgetError> for ApiError {
    fn from(err: ForgetError) -> Self {
        match err {
            ForgetError::NotFound(id) => Self::not_found(format!("memory {id} not found")),
            ForgetError::Database(e) => Self::internal(format!("database error: {e}")),
            ForgetError::Outbox(e) => Self::internal(format!("outbox error: {e}")),
        }
    }
}

impl From<CanonicalUserError> for ApiError {
    fn from(err: CanonicalUserError) -> Self {
        match err {
            CanonicalUserError::Repo(repo_err) => match repo_err {
                oris_memory_store::postgres::UserRepoError::NotFound => {
                    Self::NotFound("canonical user profile not found".into())
                }
                oris_memory_store::postgres::UserRepoError::Database(e) => {
                    Self::internal(format!("database error: {e}"))
                }
            },
            CanonicalUserError::Cache(msg) => Self::internal(format!("cache error: {msg}")),
        }
    }
}

impl From<SharedTaskError> for ApiError {
    fn from(err: SharedTaskError) -> Self {
        match err {
            SharedTaskError::NotFound => Self::NotFound("task not found".into()),
            SharedTaskError::InvalidHandoff(msg) => {
                Self::BadRequest(format!("invalid handoff: {msg}"))
            }
            SharedTaskError::Cache(msg) => Self::internal(format!("cache error: {msg}")),
            SharedTaskError::Lock(msg) => Self::Conflict(format!("lock error: {msg}")),
            SharedTaskError::Repo(repo_err) => {
                Self::internal(format!("database error: {repo_err}"))
            }
        }
    }
}

impl From<SearchError> for ApiError {
    fn from(err: SearchError) -> Self {
        Self::internal(format!("search error: {err}"))
    }
}

impl From<crate::context_assembler::AssemblerError> for ApiError {
    fn from(err: crate::context_assembler::AssemblerError) -> Self {
        match err {
            crate::context_assembler::AssemblerError::Source(msg) => {
                Self::internal(format!("context source error: {msg}"))
            }
            crate::context_assembler::AssemblerError::Serialization(e) => {
                Self::internal(format!("serialization error: {e}"))
            }
        }
    }
}

// ═══════════════════════════ RequestContext ═══════════════════════════

/// Validated request context extracted from gateway headers or middleware
/// extensions.
///
/// Populated by the §26 API-gateway middleware (which injects an
/// [`ApiContract`] into request extensions). When the middleware has not
/// run — e.g. in tests or behind a different router — the extractor falls
/// back to reading `x-*` headers directly.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestContext {
    /// Tenant / organisation identifier (required).
    pub tenant_id: String,
    /// Canonical user identifier (required).
    pub user_id: String,
    /// Agent making the request on behalf of the user (optional).
    pub agent_id: Option<String>,
    /// Stated purpose of the access, e.g. `"memory_read"`.
    pub purpose: String,
    /// Task context UUID (optional; required for cross-agent ops).
    pub task_id: Option<Uuid>,
    /// Trace identifier for the request lifecycle.
    pub trace_id: String,
}

impl RequestContext {
    /// Build a [`RequestContext`] from HTTP headers without requiring an
    /// `Authorization` header (the token is handled separately by the
    /// gateway).
    pub fn from_headers(headers: &HeaderMap) -> Result<Self, ApiError> {
        let require = |name: &'static str| -> Result<String, ApiError> {
            match headers.get(name) {
                None => Err(ApiError::Unauthorized(format!("missing header: {name}"))),
                Some(val) => {
                    let raw = val
                        .to_str()
                        .map_err(|_| ApiError::bad_request(format!("invalid header: {name}")))?;
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        return Err(ApiError::bad_request(format!("empty header: {name}")));
                    }
                    Ok(trimmed.to_string())
                }
            }
        };

        let tenant_id = require("x-tenant-id")?;
        let user_id = require("x-user-id")?;
        let purpose = require("x-purpose")?;

        let agent_id = headers
            .get("x-agent-id")
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string);

        let task_id = headers
            .get("x-task-id")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| Uuid::parse_str(s.trim()).ok());

        let trace_id = headers
            .get("x-trace-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        Ok(Self {
            tenant_id,
            user_id,
            agent_id,
            purpose,
            task_id,
            trace_id,
        })
    }

    /// Build a minimal [`ResolvedIdentity`] from this context.
    ///
    /// This is used by the context-assembly handler when the gateway has
    /// not injected a full [`ResolvedIdentity`] into extensions (e.g. in
    /// tests). In production the gateway's resolved identity is preferred.
    pub fn to_resolved_identity(&self) -> ResolvedIdentity {
        ResolvedIdentity {
            user_id: self.user_id.clone(),
            organization_id: self.tenant_id.clone(),
            factory_id: None,
            roles: vec![],
            delegated_agent_id: self.agent_id.clone(),
            permissions: PermissionSet::new(vec![], vec![]),
            purpose: self.purpose.clone(),
            trace_id: self.trace_id.clone(),
            resolved_at: Utc::now(),
        }
    }

    /// Authorisation check: is the caller allowed to act on the given
    /// tenant's data?
    pub fn require_tenant(&self, tenant_id: &str) -> Result<(), ApiError> {
        if self.tenant_id == tenant_id {
            Ok(())
        } else {
            Err(ApiError::Forbidden(format!(
                "tenant mismatch: caller '{}' vs target '{tenant_id}'",
                self.tenant_id
            )))
        }
    }
}

/// Axum extractor: resolves [`RequestContext`] from request extensions
/// (populated by the gateway middleware) or, as a fallback, from `x-*`
/// headers.
impl<S> FromRequestParts<S> for RequestContext
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // 1. Prefer the ApiContract injected by the gateway middleware.
        if let Some(contract) = parts.extensions.get::<ApiContract>() {
            return Ok(RequestContext {
                tenant_id: contract.tenant_id.clone(),
                user_id: contract.user_id.clone(),
                agent_id: contract.agent_id.clone(),
                purpose: contract.purpose.clone(),
                task_id: contract.task_id,
                trace_id: contract.trace_id.clone(),
            });
        }
        // 2. Fall back to reading headers directly.
        Self::from_headers(&parts.headers)
    }
}

// ═══════════════════════════ Request DTOs ═══════════════════════════

/// `POST /v1/memories/candidates` — submit a candidate memory.
#[derive(Debug, Clone, Deserialize)]
pub struct SubmitCandidateRequest {
    /// Raw content text of the memory.
    pub content: String,
    /// Provenance source type (e.g. `user_explicit`, `agent_inferred`).
    pub source_type: SourceType,
    /// Memory classification (semantic, episodic, …).
    pub memory_type: MemoryType,
    /// Organisational breadth of the memory.
    pub scope: Scope,
    /// Optional subject type (e.g. `"equipment"`, `"person"`).
    #[serde(default)]
    pub subject_type: Option<String>,
    /// Optional subject identifier.
    #[serde(default)]
    pub subject_id: Option<String>,
    /// Entity references (JSON array, default empty).
    #[serde(default)]
    pub entity_refs: Vec<Value>,
    /// Optional structured payload (JSONB).
    #[serde(default)]
    pub structured_payload: Option<Value>,
    /// Optional reference to the originating system / document.
    #[serde(default)]
    pub source_reference: Option<String>,
    /// Caller confidence in the content (0.0–1.0).
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    0.5
}

/// `POST /v1/memories/search` — hybrid retrieval request.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    /// Natural-language or keyword query text.
    #[serde(default)]
    pub query_text: Option<String>,
    /// Filter by memory types (default: all).
    #[serde(default)]
    pub memory_types: Vec<MemoryType>,
    /// Filter by scopes (default: all caller-accessible).
    #[serde(default)]
    pub scopes: Vec<Scope>,
    /// Filter by subject ID.
    #[serde(default)]
    pub subject_id: Option<String>,
    /// Minimum confidence threshold.
    #[serde(default)]
    pub min_confidence: Option<f32>,
    /// Maximum results after reranking.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Pagination limit (raw retrieval candidates).
    #[serde(default = "default_search_limit")]
    pub limit: i64,
    /// Pagination offset.
    #[serde(default)]
    pub offset: i64,
    /// Optional embedding vector for semantic search.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

fn default_top_k() -> usize {
    10
}

fn default_search_limit() -> i64 {
    50
}

/// `POST /v1/context/assemble` — assemble context for an LLM prompt.
#[derive(Debug, Clone, Deserialize)]
pub struct AssembleContextRequest {
    /// The user's query / prompt text.
    pub query: String,
    /// Optional explicit intent override (auto-detected if absent).
    #[serde(default)]
    pub intent: Option<crate::context_router::RequestIntent>,
    /// Task ID for cross-agent context assembly.
    #[serde(default)]
    pub task_id: Option<Uuid>,
    /// Caller cap on latency (ms).
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    /// Caller cap on token budget.
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// `PATCH /v1/tasks/{id}/context` — update shared task context.
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateTaskContextRequest {
    /// Findings to append to the task.
    #[serde(default)]
    pub add_findings: Vec<Value>,
    /// Optional new task status.
    #[serde(default)]
    pub status: Option<String>,
}

/// `POST /v1/memories/{id}/promote` — scope promotion.
#[derive(Debug, Clone, Deserialize)]
pub struct PromoteRequest {
    /// Target scope to promote to.
    pub target_scope: Scope,
    /// Human-readable reason for the promotion.
    #[serde(default)]
    pub reason: Option<String>,
}

/// `POST /v1/memories/{id}/verify` — source verification.
#[derive(Debug, Clone, Deserialize)]
pub struct VerifyRequest {
    /// Who is performing the verification.
    pub verified_by: String,
    /// Optional verification notes.
    #[serde(default)]
    pub notes: Option<String>,
}

/// Forget operation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForgetMethod {
    /// Soft-delete a single memory (mark revoked, keep row).
    Soft,
    /// Hard-delete a single memory (remove row + derived data).
    Hard,
    /// Cascade soft-delete a memory and its derived chain.
    Cascade,
    /// GDPR bulk soft-delete all memories for a user.
    UserData,
}

/// `POST /v1/memories/forget` — forget / delete (governance).
#[derive(Debug, Clone, Deserialize)]
pub struct ForgetRequest {
    /// Memory to forget (required for `soft` / `hard` / `cascade`).
    #[serde(default)]
    pub memory_id: Option<Uuid>,
    /// User whose data to forget (required for `user_data`).
    #[serde(default)]
    pub user_id: Option<String>,
    /// Forget mode.
    pub mode: ForgetMethod,
}

// ═══════════════════════════ Response DTOs ═══════════════════════════

/// Response for `POST /v1/memories/candidates`.
#[derive(Debug, Clone, Serialize)]
pub struct SubmitCandidateResponse {
    pub memory_id: Uuid,
    pub status: MemoryStatus,
    pub importance: f32,
    pub is_duplicate: bool,
    pub safety_verdict: SafetyVerdictSummary,
    pub outbox_event_id: Uuid,
}

impl From<WriteResult> for SubmitCandidateResponse {
    fn from(r: WriteResult) -> Self {
        Self {
            memory_id: r.memory_id,
            status: r.status,
            importance: r.importance,
            is_duplicate: r.is_duplicate,
            safety_verdict: r.safety_verdict,
            outbox_event_id: r.outbox_event_id,
        }
    }
}

/// Response for `GET /v1/memories/{id}` — exposes all API-relevant fields
/// without the raw embedding vector.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryResponse {
    pub memory_id: Uuid,
    pub tenant_id: String,
    pub memory_type: MemoryType,
    pub scope: Scope,
    pub subject_type: Option<String>,
    pub subject_id: Option<String>,
    pub content: Option<String>,
    pub source_type: SourceType,
    pub source_reference: Option<String>,
    pub confidence: f32,
    pub authority_level: AuthorityLevel,
    pub importance: f32,
    pub privacy_class: PrivacyClass,
    pub status: MemoryStatus,
    pub version: i32,
    pub derived_from: Vec<Value>,
    pub created_by_user: Option<String>,
    pub created_by_agent: Option<String>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<MemoryItem> for MemoryResponse {
    fn from(item: MemoryItem) -> Self {
        Self {
            memory_id: item.memory_id,
            tenant_id: item.tenant_id,
            memory_type: item.memory_type,
            scope: item.scope,
            subject_type: item.subject_type,
            subject_id: item.subject_id,
            content: item.content,
            source_type: item.source_type,
            source_reference: item.source_reference,
            confidence: item.confidence,
            authority_level: item.authority_level,
            importance: item.importance,
            privacy_class: item.privacy_class,
            status: item.status,
            version: item.version,
            derived_from: item.derived_from,
            created_by_user: item.created_by_user,
            created_by_agent: item.created_by_agent,
            last_verified_at: item.last_verified_at,
            created_at: item.created_at,
            updated_at: item.updated_at,
        }
    }
}

/// Response for `POST /v1/memories/search`.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub results: Vec<RerankedResult>,
    pub count: usize,
}

/// Serialisable conflict flag for the assemble response.
#[derive(Debug, Clone, Serialize)]
pub struct ConflictFlagDto {
    pub field: String,
    pub description: String,
    pub conflicting_values: Vec<String>,
}

impl From<&ConflictFlag> for ConflictFlagDto {
    fn from(f: &ConflictFlag) -> Self {
        Self {
            field: f.field.clone(),
            description: f.description.clone(),
            conflicting_values: f.conflicting_values.clone(),
        }
    }
}

/// Response for `POST /v1/context/assemble`.
#[derive(Debug, Clone, Serialize)]
pub struct AssembleContextResponse {
    pub context_text: String,
    pub token_count: usize,
    pub compressed: bool,
    pub conflict_flags: Vec<ConflictFlagDto>,
    pub low_confidence_items: Vec<String>,
    pub sources_used: Vec<ContextSource>,
    pub degraded_sources: Vec<ContextSource>,
}

/// Response for `GET /v1/users/{id}/canonical-context`.
#[derive(Debug, Clone, Serialize)]
pub struct CanonicalContextResponse {
    pub profile: Option<CanonicalUserProfile>,
}

/// Response for `GET|PATCH /v1/tasks/{id}/context`.
#[derive(Debug, Clone, Serialize)]
pub struct TaskContextResponse {
    pub context: Option<SharedTaskContext>,
}

/// Response for `POST /v1/memories/{id}/promote`.
#[derive(Debug, Clone, Serialize)]
pub struct PromoteResponse {
    pub memory_id: Uuid,
    pub scope: Scope,
    pub message: String,
}

/// Response for `POST /v1/memories/{id}/verify`.
#[derive(Debug, Clone, Serialize)]
pub struct VerifyResponse {
    pub memory_id: Uuid,
    pub verified_at: DateTime<Utc>,
    pub message: String,
}

/// Response for `POST /v1/memories/forget`.
#[derive(Debug, Clone, Serialize)]
pub struct ForgetResponse {
    pub mode: ForgetMethod,
    pub affected_count: u64,
    pub message: String,
}

/// A single version entry in the lineage chain.
#[derive(Debug, Clone, Serialize)]
pub struct VersionInfo {
    pub version_id: Uuid,
    pub memory_id: Uuid,
    pub version_number: i32,
    pub changed_by: String,
    pub change_reason: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Response for `GET /v1/memories/{id}/lineage`.
#[derive(Debug, Clone, Serialize)]
pub struct LineageResponse {
    pub memory_id: Uuid,
    pub versions: Vec<VersionInfo>,
}

/// Simple health-check response.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

impl HealthResponse {
    pub fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

// ═══════════════════════════ Tests ═══════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    // ── ApiError status code mapping ──

    #[test]
    fn api_error_status_codes() {
        assert_eq!(
            ApiError::NotFound("x".into()).status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiError::Unauthorized("x".into()).status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ApiError::Forbidden("x".into()).status_code(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            ApiError::Conflict("x".into()).status_code(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            ApiError::BadRequest("x".into()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ApiError::Internal("x".into()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn api_error_codes_are_stable() {
        assert_eq!(ApiError::NotFound("".into()).error_code(), "NOT_FOUND");
        assert_eq!(
            ApiError::Unauthorized("".into()).error_code(),
            "UNAUTHORIZED"
        );
        assert_eq!(ApiError::Forbidden("".into()).error_code(), "FORBIDDEN");
        assert_eq!(ApiError::Conflict("".into()).error_code(), "CONFLICT");
        assert_eq!(ApiError::BadRequest("".into()).error_code(), "BAD_REQUEST");
        assert_eq!(ApiError::Internal("".into()).error_code(), "INTERNAL_ERROR");
    }

    #[tokio::test]
    async fn api_error_into_response_produces_json_body() {
        let resp = ApiError::NotFound("memory 123".into()).into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"], "not found: memory 123");
        assert_eq!(json["error_code"], "NOT_FOUND");
    }

    // ── Domain error conversions ──

    #[test]
    fn memory_repo_not_found_maps_to_404() {
        let err: ApiError = MemoryRepoError::NotFound.into();
        assert!(matches!(err, ApiError::NotFound(_)));
        assert_eq!(err.status_code(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn write_pipeline_poison_blocked_maps_to_400() {
        let err: ApiError = WritePipelineError::PoisonBlocked {
            detail: "bad".into(),
        }
        .into();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn version_not_found_maps_to_404() {
        let id = Uuid::new_v4();
        let err: ApiError = VersionError::NotFound(id).into();
        assert!(matches!(err, ApiError::NotFound(_)));
    }

    #[test]
    fn forget_not_found_maps_to_404() {
        let id = Uuid::new_v4();
        let err: ApiError = ForgetError::NotFound(id).into();
        assert!(matches!(err, ApiError::NotFound(_)));
    }

    #[test]
    fn shared_task_not_found_maps_to_404() {
        let err: ApiError = SharedTaskError::NotFound.into();
        assert!(matches!(err, ApiError::NotFound(_)));
    }

    #[test]
    fn shared_task_lock_maps_to_conflict() {
        let err: ApiError = SharedTaskError::Lock("busy".into()).into();
        assert!(matches!(err, ApiError::Conflict(_)));
    }

    // ── RequestContext header extraction ──

    fn valid_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-tenant-id", "acme-corp".parse().unwrap());
        h.insert("x-user-id", "user-42".parse().unwrap());
        h.insert("x-purpose", "memory_read".parse().unwrap());
        h.insert("x-trace-id", "trace-abc".parse().unwrap());
        h.insert("x-agent-id", "agent-007".parse().unwrap());
        h
    }

    #[test]
    fn request_context_from_valid_headers() {
        let ctx = RequestContext::from_headers(&valid_headers()).unwrap();
        assert_eq!(ctx.tenant_id, "acme-corp");
        assert_eq!(ctx.user_id, "user-42");
        assert_eq!(ctx.purpose, "memory_read");
        assert_eq!(ctx.agent_id.as_deref(), Some("agent-007"));
        assert_eq!(ctx.trace_id, "trace-abc");
    }

    #[test]
    fn request_context_missing_tenant_returns_unauthorized() {
        let mut h = valid_headers();
        h.remove("x-tenant-id");
        let err = RequestContext::from_headers(&h).unwrap_err();
        assert!(matches!(err, ApiError::Unauthorized(_)));
    }

    #[test]
    fn request_context_empty_user_returns_bad_request() {
        let mut h = valid_headers();
        h.insert("x-user-id", "  ".parse().unwrap());
        let err = RequestContext::from_headers(&h).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn request_context_parses_task_id() {
        let mut h = valid_headers();
        let tid = Uuid::new_v4();
        h.insert("x-task-id", tid.to_string().parse().unwrap());
        let ctx = RequestContext::from_headers(&h).unwrap();
        assert_eq!(ctx.task_id, Some(tid));
    }

    #[test]
    fn request_context_ignores_malformed_task_id() {
        let mut h = valid_headers();
        h.insert("x-task-id", "not-a-uuid".parse().unwrap());
        let ctx = RequestContext::from_headers(&h).unwrap();
        assert!(ctx.task_id.is_none());
    }

    #[test]
    fn request_context_auto_generates_trace_id() {
        let mut h = valid_headers();
        h.remove("x-trace-id");
        let ctx = RequestContext::from_headers(&h).unwrap();
        assert!(!ctx.trace_id.is_empty());
    }

    #[test]
    fn request_context_empty_agent_becomes_none() {
        let mut h = valid_headers();
        h.insert("x-agent-id", "  ".parse().unwrap());
        let ctx = RequestContext::from_headers(&h).unwrap();
        assert!(ctx.agent_id.is_none());
    }

    #[test]
    fn request_context_require_tenant_matches() {
        let ctx = RequestContext::from_headers(&valid_headers()).unwrap();
        assert!(ctx.require_tenant("acme-corp").is_ok());
        assert!(ctx.require_tenant("other").is_err());
    }

    // ── DTO (de)serialisation ──

    #[test]
    fn submit_candidate_request_round_trips() {
        let json = r#"{
            "content": "The pump X-200 needs oil every 500h",
            "source_type": "user_explicit",
            "memory_type": "semantic",
            "scope": "personal",
            "confidence": 0.9
        }"#;
        let req: SubmitCandidateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.content, "The pump X-200 needs oil every 500h");
        assert_eq!(req.source_type, SourceType::UserExplicit);
        assert_eq!(req.memory_type, MemoryType::Semantic);
        assert_eq!(req.scope, Scope::Personal);
        assert!((req.confidence - 0.9).abs() < 0.001);
        assert!(req.entity_refs.is_empty());
    }

    #[test]
    fn submit_candidate_request_defaults_confidence() {
        let json = r#"{"content":"x","source_type":"agent_inferred","memory_type":"episodic","scope":"agent"}"#;
        let req: SubmitCandidateRequest = serde_json::from_str(json).unwrap();
        assert!((req.confidence - 0.5).abs() < 0.001);
    }

    #[test]
    fn search_request_defaults() {
        let json = r#"{"query_text":"pump maintenance"}"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query_text.as_deref(), Some("pump maintenance"));
        assert_eq!(req.top_k, 10);
        assert_eq!(req.limit, 50);
        assert!(req.memory_types.is_empty());
    }

    #[test]
    fn forget_request_parses_soft_mode() {
        let json = r#"{"memory_id":"00000000-0000-0000-0000-000000000001","mode":"soft"}"#;
        let req: ForgetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.mode, ForgetMethod::Soft);
        assert!(req.memory_id.is_some());
    }

    #[test]
    fn forget_request_parses_user_data_mode() {
        let json = r#"{"user_id":"user-42","mode":"user_data"}"#;
        let req: ForgetRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.mode, ForgetMethod::UserData);
        assert_eq!(req.user_id.as_deref(), Some("user-42"));
    }

    #[test]
    fn promote_request_parses() {
        let json = r#"{"target_scope":"team","reason":"promoted by QA"}"#;
        let req: PromoteRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.target_scope, Scope::Team);
        assert_eq!(req.reason.as_deref(), Some("promoted by QA"));
    }

    #[test]
    fn memory_response_excludes_embedding() {
        let item = make_test_memory_item();
        let resp: MemoryResponse = item.into();
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("embedding"));
    }

    // ── Test helpers ──

    fn make_test_memory_item() -> MemoryItem {
        use chrono::Utc;
        use oris_memory_contract::memory_types::*;
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "acme".into(),
            memory_type: MemoryType::Semantic,
            scope: Scope::Personal,
            subject_type: None,
            subject_id: None,
            entity_refs: vec![],
            content: Some("test".into()),
            structured_payload: None,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            source_type: SourceType::UserExplicit,
            source_reference: None,
            evidence_refs: vec![],
            confidence: 0.8,
            authority_level: AuthorityLevel::L1Authoritative,
            importance: 0.7,
            observed_at: None,
            valid_from: None,
            valid_to: None,
            privacy_class: PrivacyClass::Internal,
            acl: serde_json::json!({}),
            retention_policy: None,
            status: MemoryStatus::Active,
            version: 1,
            derived_from: vec![],
            created_by_user: Some("user-1".into()),
            created_by_agent: None,
            last_verified_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}
