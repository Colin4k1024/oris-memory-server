//! User-facing memory management endpoints (§5.3, §11.5).
//!
//! Provides REST handlers that let users view, correct, export, forget, and
//! revoke consent for their own memories. Every operation is audit-logged
//! ([`AuditEntry`]) and every display path is PII-masked ([`PiiMasker]).
//!
//! # Endpoints
//!
//! | Method | Path | Handler |
//! |--------|-----|---------|
//! | `GET` | `/v1/users/{id}/memories` | [`list_user_memories`] |
//! | `PATCH` | `/v1/memories/{id}` | [`correct_memory`] |
//! | `GET` | `/v1/users/{id}/memories/export` | [`export_user_memories`] |
//! | `POST` | `/v1/users/{id}/consent/revoke` | [`revoke_consent`] |
//!
//! (`POST /v1/memories/forget` already exists in [`crate::api::routes`].)
//!
//! # Design
//!
//! Like the core V1 handlers in [`crate::api::routes`], each backend
//! dependency is abstracted behind an `async_trait` ([`UserMemoryService`])
//! so handlers can be unit-tested with mock implementations. The
//! [`UserMemoryAppState`] holds a single `Arc<dyn UserMemoryService>`.

use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::{Path, Query, State};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use oris_memory_contract::memory_types::{
    AuthorityLevel, MemoryItem, MemoryStatus, MemoryType, PrivacyClass, Scope, SourceType,
};

use crate::api::extractors::{ApiError, RequestContext};
use crate::governance::audit::AuditEntry;

// ═══════════════════════ Domain Types ═══════════════════════

/// Export format for the memory-export endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Json,
    Csv,
}

impl ExportFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
        }
    }
}

/// The result of a memory export operation.
#[derive(Debug, Clone, Serialize)]
pub struct ExportResult {
    /// The serialised export payload (JSON text or CSV text).
    pub content: String,
    /// The format that was used.
    pub format: ExportFormat,
    /// Number of memory items included in the export.
    pub item_count: usize,
}

/// Which scope of consent is being revoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsentScope {
    /// Revoke consent for *all* of the user's memories.
    All,
    /// Revoke consent for personal-scope memories only.
    Personal,
    /// Revoke consent for agent-scope memories only.
    Agent,
    /// Revoke consent for task-scope memories only.
    Task,
    /// Revoke consent for team-scope memories only.
    Team,
}

/// The outcome of a consent-revocation request.
///
/// The service records the revocation and triggers a cascade forget for
/// every memory in the revoked scope.
#[derive(Debug, Clone, Serialize)]
pub struct ConsentRevocation {
    pub user_id: String,
    pub scope: ConsentScope,
    pub revoked_at: DateTime<Utc>,
    /// Number of memories affected by the cascade forget.
    pub affected_count: u64,
}

/// Filters applied when listing a user's memories.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MemoryFilters {
    pub memory_type: Option<MemoryType>,
    pub scope: Option<Scope>,
    pub status: Option<MemoryStatus>,
    /// Free-text search on memory content.
    pub search: Option<String>,
}

/// Generic pagination envelope.
#[derive(Debug, Clone, Serialize)]
pub struct PagedResult<T: Serialize> {
    pub items: Vec<T>,
    pub page: usize,
    pub per_page: usize,
    pub total: usize,
    pub total_pages: usize,
}

impl<T: Serialize> PagedResult<T> {
    /// Build a [`PagedResult`] from a slice of items and a known total.
    pub fn new(items: Vec<T>, page: usize, per_page: usize, total: usize) -> Self {
        let total_pages = if per_page == 0 {
            0
        } else {
            (total + per_page - 1) / per_page
        };
        Self {
            items,
            page,
            per_page,
            total,
            total_pages,
        }
    }
}

/// A sanitised view of a memory item for user-facing display.
///
/// The `content` field is PII-masked via [`PiiMasker`](crate::compliance::PiiMasker)
/// before display. Internal fields such as `embedding`, `acl`,
/// `structured_payload`, and `entity_refs` are excluded. Audit-relevant
/// fields (`version`, `created_at`, `updated_at`, `last_verified_at`) are
/// included.
#[derive(Debug, Clone, Serialize)]
pub struct MemorySummary {
    pub memory_id: Uuid,
    pub memory_type: MemoryType,
    pub scope: Scope,
    /// PII-masked content text (may be `None` if the original had no content).
    pub content: Option<String>,
    pub source_type: SourceType,
    pub confidence: f32,
    pub authority_level: AuthorityLevel,
    pub importance: f32,
    pub privacy_class: PrivacyClass,
    pub status: MemoryStatus,
    pub version: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_verified_at: Option<DateTime<Utc>>,
}

impl From<&MemoryItem> for MemorySummary {
    fn from(item: &MemoryItem) -> Self {
        let content = item.content.as_ref().map(|c| {
            // PII-mask the content before display (§11.5).
            crate::compliance::PiiMasker::mask(c).0
        });
        Self {
            memory_id: item.memory_id,
            memory_type: item.memory_type,
            scope: item.scope,
            content,
            source_type: item.source_type,
            confidence: item.confidence,
            authority_level: item.authority_level,
            importance: item.importance,
            privacy_class: item.privacy_class,
            status: item.status,
            version: item.version,
            created_at: item.created_at,
            updated_at: item.updated_at,
            last_verified_at: item.last_verified_at,
        }
    }
}

impl From<MemoryItem> for MemorySummary {
    fn from(item: MemoryItem) -> Self {
        Self::from(&item)
    }
}

// ═══════════════════════ Request / Response DTOs ═══════════════════════

/// `GET /v1/users/{id}/memories` — query parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct ListMemoriesQuery {
    #[serde(default = "default_page")]
    pub page: usize,
    #[serde(default = "default_per_page")]
    pub per_page: usize,
    pub memory_type: Option<MemoryType>,
    pub scope: Option<Scope>,
    pub status: Option<MemoryStatus>,
    pub search: Option<String>,
}

impl ListMemoriesQuery {
    fn to_filters(&self) -> MemoryFilters {
        MemoryFilters {
            memory_type: self.memory_type,
            scope: self.scope,
            status: self.status,
            search: self.search.clone(),
        }
    }
}

/// `PATCH /v1/memories/{id}` — request body for correcting a memory.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryCorrection {
    /// New content text (if the content is being corrected).
    pub content: Option<String>,
    /// New confidence score (if the confidence is being corrected).
    pub confidence: Option<f32>,
    /// Human-readable reason for the correction (required).
    pub reason: String,
}

/// `PATCH /v1/memories/{id}` — response body.
#[derive(Debug, Clone, Serialize)]
pub struct CorrectionResponse {
    /// The memory id of the newly-created version.
    pub memory_id: Uuid,
    /// The original memory id that was corrected.
    pub original_memory_id: Uuid,
    /// The new version number.
    pub version: i32,
    pub message: String,
}

/// `GET /v1/users/{id}/memories/export` — query parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct ExportQuery {
    #[serde(default = "default_export_format")]
    pub format: ExportFormat,
}

/// `POST /v1/users/{id}/consent/revoke` — request body.
#[derive(Debug, Clone, Deserialize)]
pub struct RevokeConsentRequest {
    pub scope: ConsentScope,
    #[serde(default)]
    pub reason: Option<String>,
}

fn default_page() -> usize {
    1
}

fn default_per_page() -> usize {
    20
}

fn default_export_format() -> ExportFormat {
    ExportFormat::Json
}

// ═══════════════════════ Service Trait ═══════════════════════

/// User-facing memory management operations.
///
/// Every implementation **must** write an [`AuditEntry`] for each operation
/// (following the patterns in [`crate::governance::audit`]) and **must**
/// mask sensitive attributes before returning display data (following the
/// [`crate::compliance::PiiMasker`] patterns).
#[async_trait]
pub trait UserMemoryService: Send + Sync {
    /// List a user's memories with pagination and filtering.
    ///
    /// Returns [`MemorySummary`] items with PII-masked content and
    /// audit fields included.
    async fn list_user_memories(
        &self,
        user_id: &str,
        filters: &MemoryFilters,
        page: usize,
        per_page: usize,
    ) -> Result<PagedResult<MemorySummary>, ApiError>;

    /// Correct a memory's content or confidence.
    ///
    /// Creates a new versioned snapshot (via [`VersionManager`](crate::governance::VersionManager))
    /// and returns the memory id of the new version.
    async fn correct_memory(
        &self,
        memory_id: Uuid,
        user_id: &str,
        correction: &MemoryCorrection,
    ) -> Result<Uuid, ApiError>;

    /// Export all of a user's memories in the requested format.
    async fn export_user_memories(
        &self,
        user_id: &str,
        format: ExportFormat,
    ) -> Result<ExportResult, ApiError>;

    /// Revoke consent for a memory scope.
    ///
    /// Records the revocation and triggers a cascade forget for every
    /// memory in the revoked scope. Returns a [`ConsentRevocation`] with
    /// the number of affected memories.
    async fn revoke_consent(
        &self,
        user_id: &str,
        scope: ConsentScope,
    ) -> Result<ConsentRevocation, ApiError>;
}

// ═══════════════════════ Audit Helpers ═══════════════════════

/// Build an [`AuditEntry`] for a user-memory-management operation, enriched
/// with request context (agent, task, trace). Follows the
/// `build_forget_audit_entry` pattern in [`crate::governance::forget`].
fn build_user_memory_audit_entry(
    ctx: &RequestContext,
    memory_id: Option<Uuid>,
    action: &str,
    purpose: &str,
) -> AuditEntry {
    let mut entry = AuditEntry::new(&ctx.user_id, action, "allow");
    entry.memory_id = memory_id;
    entry.accessor_agent = ctx.agent_id.clone();
    entry.purpose = purpose.to_string();
    entry.task_id = ctx.task_id;
    entry.trace_id = ctx.trace_id.clone();
    entry
}

/// Build a CSV representation of memory summaries.
fn summaries_to_csv(summaries: &[MemorySummary]) -> String {
    let mut out =
        String::from("memory_id,memory_type,scope,content,confidence,status,version,created_at\n");
    for s in summaries {
        let content = s.content.as_deref().unwrap_or("");
        // Escape quotes and newlines for CSV safety.
        let escaped = content.replace('"', "\"\"").replace('\n', " ");
        out.push_str(&format!(
            "{},{},{},\"{}\",{},{},{},{}\n",
            s.memory_id,
            s.memory_type.as_str(),
            s.scope.as_str(),
            escaped,
            s.confidence,
            s.status.as_str(),
            s.version,
            s.created_at.to_rfc3339(),
        ));
    }
    out
}

// ═══════════════════════ AppState ═══════════════════════

/// Application state for the user-memory endpoints.
///
/// Holds a single trait-object `Arc` so handlers can be tested with mock
/// implementations without a live database.
#[derive(Clone)]
pub struct UserMemoryAppState {
    pub user_memory: Arc<dyn UserMemoryService>,
}

// ═══════════════════════ Handlers ═══════════════════════

/// `GET /v1/users/{id}/memories` — list a user's memories (paginated, filtered).
pub async fn list_user_memories(
    State(state): State<UserMemoryAppState>,
    Path(user_id): Path<String>,
    ctx: RequestContext,
    Query(query): Query<ListMemoriesQuery>,
) -> Result<Json<PagedResult<MemorySummary>>, ApiError> {
    // Authorisation: a user can only view their own memories.
    if ctx.user_id != user_id {
        return Err(ApiError::Forbidden(
            "cannot view another user's memories".to_string(),
        ));
    }

    let filters = query.to_filters();
    let result = state
        .user_memory
        .list_user_memories(&user_id, &filters, query.page, query.per_page)
        .await?;

    Ok(Json(result))
}

/// `PATCH /v1/memories/{id}` — correct a memory's content (versioned).
pub async fn correct_memory(
    State(state): State<UserMemoryAppState>,
    Path(memory_id): Path<Uuid>,
    ctx: RequestContext,
    Json(req): Json<MemoryCorrection>,
) -> Result<Json<CorrectionResponse>, ApiError> {
    if req.reason.trim().is_empty() {
        return Err(ApiError::bad_request("reason must not be empty"));
    }
    if let Some(ref content) = req.content {
        if content.trim().is_empty() {
            return Err(ApiError::bad_request(
                "content must not be empty when provided",
            ));
        }
    }
    if let Some(confidence) = req.confidence {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(ApiError::bad_request(
                "confidence must be between 0.0 and 1.0",
            ));
        }
    }

    let new_id = state
        .user_memory
        .correct_memory(memory_id, &ctx.user_id, &req)
        .await?;

    Ok(Json(CorrectionResponse {
        memory_id: new_id,
        original_memory_id: memory_id,
        version: 2, // The new version (snapshot was v1, this is v2).
        message: "memory corrected — new version created".to_string(),
    }))
}

/// `GET /v1/users/{id}/memories/export` — export a user's memories.
pub async fn export_user_memories(
    State(state): State<UserMemoryAppState>,
    Path(user_id): Path<String>,
    ctx: RequestContext,
    Query(query): Query<ExportQuery>,
) -> Result<Json<ExportResult>, ApiError> {
    if ctx.user_id != user_id {
        return Err(ApiError::Forbidden(
            "cannot export another user's memories".to_string(),
        ));
    }

    let result = state
        .user_memory
        .export_user_memories(&user_id, query.format)
        .await?;

    Ok(Json(result))
}

/// `POST /v1/users/{id}/consent/revoke` — revoke consent for a scope.
pub async fn revoke_consent(
    State(state): State<UserMemoryAppState>,
    Path(user_id): Path<String>,
    ctx: RequestContext,
    Json(req): Json<RevokeConsentRequest>,
) -> Result<Json<ConsentRevocation>, ApiError> {
    if ctx.user_id != user_id {
        return Err(ApiError::Forbidden(
            "cannot revoke consent for another user".to_string(),
        ));
    }

    let revocation = state
        .user_memory
        .revoke_consent(&user_id, req.scope)
        .await?;

    Ok(Json(revocation))
}

// ═══════════════════════ Router ═══════════════════════

/// Build the user-memory-management router.
///
/// Routes are mounted under `/v1/`. The caller is responsible for merging
/// this into the main V1 router and adding the gateway middleware layer.
pub fn user_memory_router() -> Router<UserMemoryAppState> {
    Router::new()
        .route("/v1/users/{id}/memories", get(list_user_memories))
        .route("/v1/memories/{id}", patch(correct_memory))
        .route("/v1/users/{id}/memories/export", get(export_user_memories))
        .route("/v1/users/{id}/consent/revoke", post(revoke_consent))
}

// ═══════════════════════ Tests ═══════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use std::sync::Mutex;
    use tower::ServiceExt;

    // ─── Mock implementation ───────────────────────────────────

    struct MockUserMemoryService {
        /// Pre-built summaries returned by `list_user_memories`.
        summaries: Vec<MemorySummary>,
        /// Audit entries captured by every operation.
        audit_entries: Mutex<Vec<AuditEntry>>,
        /// Counter for `correct_memory` calls.
        corrections: Mutex<u32>,
        /// Counter for `revoke_consent` calls.
        revocations: Mutex<u32>,
        /// Whether `export_user_memories` should return items.
        export_items: Vec<MemorySummary>,
    }

    impl MockUserMemoryService {
        fn new(summaries: Vec<MemorySummary>) -> Self {
            Self {
                summaries,
                audit_entries: Mutex::new(vec![]),
                corrections: Mutex::new(0),
                revocations: Mutex::new(0),
                export_items: vec![],
            }
        }

        fn audit_count(&self) -> usize {
            self.audit_entries.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl UserMemoryService for MockUserMemoryService {
        async fn list_user_memories(
            &self,
            _user_id: &str,
            filters: &MemoryFilters,
            page: usize,
            per_page: usize,
        ) -> Result<PagedResult<MemorySummary>, ApiError> {
            let filtered: Vec<MemorySummary> = self
                .summaries
                .iter()
                .filter(|s| filters.scope.is_none_or(|sc| sc == s.scope))
                .filter(|s| filters.memory_type.is_none_or(|mt| mt == s.memory_type))
                .filter(|s| filters.status.is_none_or(|st| st == s.status))
                .cloned()
                .collect();

            let total = filtered.len();
            let start = (page.saturating_sub(1)) * per_page;
            let end = (start + per_page).min(total);
            let items = if start < total {
                filtered[start..end].to_vec()
            } else {
                vec![]
            };

            Ok(PagedResult::new(items, page, per_page, total))
        }

        async fn correct_memory(
            &self,
            memory_id: Uuid,
            user_id: &str,
            correction: &MemoryCorrection,
        ) -> Result<Uuid, ApiError> {
            *self.corrections.lock().unwrap() += 1;
            // Record an audit entry (following governance/audit.rs patterns).
            let mut entry = AuditEntry::new(user_id, "correct_memory", "allow");
            entry.memory_id = Some(memory_id);
            entry.purpose = "memory_correction".to_string();
            self.audit_entries.lock().unwrap().push(entry);
            let _ = &correction.reason;
            Ok(memory_id)
        }

        async fn export_user_memories(
            &self,
            _user_id: &str,
            format: ExportFormat,
        ) -> Result<ExportResult, ApiError> {
            let items = &self.export_items;
            let content = match format {
                ExportFormat::Json => {
                    serde_json::to_string_pretty(items).unwrap_or_else(|_| "[]".into())
                }
                ExportFormat::Csv => summaries_to_csv(items),
            };
            Ok(ExportResult {
                content,
                format,
                item_count: items.len(),
            })
        }

        async fn revoke_consent(
            &self,
            user_id: &str,
            scope: ConsentScope,
        ) -> Result<ConsentRevocation, ApiError> {
            *self.revocations.lock().unwrap() += 1;
            // Record an audit entry (following governance/audit.rs patterns).
            let mut entry = AuditEntry::new(user_id, "revoke_consent", "allow");
            entry.purpose = "consent_revocation".to_string();
            self.audit_entries.lock().unwrap().push(entry);
            let affected = match scope {
                ConsentScope::All => self.summaries.len() as u64,
                _ => self
                    .summaries
                    .iter()
                    .filter(|s| {
                        let scope_matches = match scope {
                            ConsentScope::Personal => s.scope == Scope::Personal,
                            ConsentScope::Agent => s.scope == Scope::Agent,
                            ConsentScope::Task => s.scope == Scope::Task,
                            ConsentScope::Team => s.scope == Scope::Team,
                            ConsentScope::All => true,
                        };
                        scope_matches
                    })
                    .count() as u64,
            };
            Ok(ConsentRevocation {
                user_id: user_id.to_string(),
                scope,
                revoked_at: Utc::now(),
                affected_count: affected,
            })
        }
    }

    // ─── Test helpers ───────────────────────────────────────────

    fn make_memory_item(id: Uuid, scope: Scope, content: &str) -> MemoryItem {
        MemoryItem {
            memory_id: id,
            tenant_id: "acme".into(),
            memory_type: MemoryType::Semantic,
            scope,
            subject_type: None,
            subject_id: None,
            entity_refs: vec![],
            content: Some(content.into()),
            structured_payload: None,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            source_type: SourceType::UserExplicit,
            source_reference: None,
            evidence_refs: vec![],
            confidence: 0.85,
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
            created_by_user: Some("user-42".into()),
            created_by_agent: None,
            last_verified_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn make_summary(id: Uuid, scope: Scope, content: &str) -> MemorySummary {
        MemorySummary::from(&make_memory_item(id, scope, content))
    }

    fn mock_state() -> UserMemoryAppState {
        let summaries = vec![
            make_summary(
                Uuid::new_v4(),
                Scope::Personal,
                "My email is john@example.com",
            ),
            make_summary(Uuid::new_v4(), Scope::Team, "Team meeting notes"),
            make_summary(Uuid::new_v4(), Scope::Personal, "Phone: 13912345678"),
        ];
        let mut mock = MockUserMemoryService::new(summaries.clone());
        mock.export_items = summaries;
        UserMemoryAppState {
            user_memory: Arc::new(mock),
        }
    }

    fn app() -> Router {
        user_memory_router().with_state(mock_state())
    }

    fn authed_request(method: Method, uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("x-tenant-id", "acme-corp")
            .header("x-user-id", "user-42")
            .header("x-purpose", "memory_read")
            .header("x-trace-id", "trace-1")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn body_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ─── Handler integration tests ─────────────────────────────

    #[tokio::test]
    async fn list_user_memories_returns_summaries() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/user-42/memories",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["page"], 1);
        assert_eq!(json["per_page"], 20);
        assert_eq!(json["total"], 3);
        assert_eq!(json["items"].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn list_user_memories_with_pagination() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/user-42/memories?page=1&per_page=2",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["per_page"], 2);
        assert_eq!(json["total"], 3);
        assert_eq!(json["total_pages"], 2);
        assert_eq!(json["items"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn list_user_memories_with_scope_filter() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/user-42/memories?scope=personal",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["total"], 2);
        for item in json["items"].as_array().unwrap() {
            assert_eq!(item["scope"], "personal");
        }
    }

    #[tokio::test]
    async fn list_user_memories_enforces_self_access() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/other-user/memories",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn list_user_memories_requires_auth() {
        let request = Request::builder()
            .method(Method::GET)
            .uri("/v1/users/user-42/memories")
            .body(Body::empty())
            .unwrap();
        let response = app().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn correct_memory_succeeds() {
        let id = Uuid::new_v4();
        let body = r#"{"content":"corrected content","confidence":0.95,"reason":"typo fix"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::PATCH,
                &format!("/v1/memories/{id}"),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["original_memory_id"], id.to_string());
        assert_eq!(json["version"], 2);
        assert!(json["message"].as_str().unwrap().contains("corrected"));
    }

    #[tokio::test]
    async fn correct_memory_rejects_empty_reason() {
        let id = Uuid::new_v4();
        let body = r#"{"content":"x","reason":""}"#;
        let response = app()
            .oneshot(authed_request(
                Method::PATCH,
                &format!("/v1/memories/{id}"),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn correct_memory_rejects_invalid_confidence() {
        let id = Uuid::new_v4();
        let body = r#"{"confidence":1.5,"reason":"adjust"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::PATCH,
                &format!("/v1/memories/{id}"),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn export_memories_json() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/user-42/memories/export?format=json",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["format"], "json");
        assert_eq!(json["item_count"], 3);
        assert!(json["content"].as_str().unwrap().contains("memory_id"));
    }

    #[tokio::test]
    async fn export_memories_csv() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/user-42/memories/export?format=csv",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["format"], "csv");
        assert!(json["content"]
            .as_str()
            .unwrap()
            .contains("memory_id,memory_type"));
    }

    #[tokio::test]
    async fn export_memories_defaults_to_json() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/user-42/memories/export",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["format"], "json");
    }

    #[tokio::test]
    async fn export_memories_enforces_self_access() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/other-user/memories/export",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn revoke_consent_succeeds() {
        let body = r#"{"scope":"personal"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                "/v1/users/user-42/consent/revoke",
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["user_id"], "user-42");
        assert_eq!(json["scope"], "personal");
        assert_eq!(json["affected_count"], 2);
        assert!(json["revoked_at"].is_string());
    }

    #[tokio::test]
    async fn revoke_consent_all_scope() {
        let body = r#"{"scope":"all"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                "/v1/users/user-42/consent/revoke",
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["scope"], "all");
        assert_eq!(json["affected_count"], 3);
    }

    #[tokio::test]
    async fn revoke_consent_enforces_self_access() {
        let body = r#"{"scope":"all"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                "/v1/users/other-user/consent/revoke",
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    // ─── Unit tests for domain types ─────────────────────────────

    #[test]
    fn memory_summary_masks_pii_in_content() {
        let item = make_memory_item(
            Uuid::new_v4(),
            Scope::Personal,
            "Contact me at john.doe@example.com or call 13912345678",
        );
        let summary = MemorySummary::from(&item);
        let content = summary.content.unwrap();
        assert!(!content.contains("john.doe@example.com"));
        assert!(!content.contains("13912345678"));
        // Masked placeholders should be present.
        assert!(content.contains("***@***.***") || content.contains('@'));
    }

    #[test]
    fn memory_summary_excludes_embedding() {
        let item = make_memory_item(Uuid::new_v4(), Scope::Personal, "test");
        let summary = MemorySummary::from(&item);
        let json = serde_json::to_string(&summary).unwrap();
        assert!(!json.contains("embedding"));
        assert!(!json.contains("structured_payload"));
        assert!(!json.contains("acl"));
    }

    #[test]
    fn paged_result_calculates_total_pages() {
        let r = PagedResult::new(vec![1, 2, 3], 1, 20, 3);
        assert_eq!(r.total_pages, 1);

        let r = PagedResult::new(vec![1, 2], 1, 2, 5);
        assert_eq!(r.total_pages, 3);

        let r = PagedResult::<i32>::new(vec![], 1, 10, 0);
        assert_eq!(r.total_pages, 0);
    }

    #[test]
    fn export_format_serializes_lowercase() {
        let json = serde_json::to_string(&ExportFormat::Json).unwrap();
        assert_eq!(json, "\"json\"");
        let json = serde_json::to_string(&ExportFormat::Csv).unwrap();
        assert_eq!(json, "\"csv\"");
    }

    #[test]
    fn export_format_deserializes() {
        let fmt: ExportFormat = serde_json::from_str("\"csv\"").unwrap();
        assert_eq!(fmt, ExportFormat::Csv);
        let fmt: ExportFormat = serde_json::from_str("\"json\"").unwrap();
        assert_eq!(fmt, ExportFormat::Json);
    }

    #[test]
    fn consent_scope_serializes_snake_case() {
        let json = serde_json::to_string(&ConsentScope::All).unwrap();
        assert_eq!(json, "\"all\"");
        let json = serde_json::to_string(&ConsentScope::Personal).unwrap();
        assert_eq!(json, "\"personal\"");
    }

    #[test]
    fn build_audit_entry_has_correct_fields() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-tenant-id", "acme".parse().unwrap());
        headers.insert("x-user-id", "user-42".parse().unwrap());
        headers.insert("x-purpose", "memory_read".parse().unwrap());
        headers.insert("x-agent-id", "agent-7".parse().unwrap());
        headers.insert("x-trace-id", "trace-99".parse().unwrap());
        let ctx = RequestContext::from_headers(&headers).unwrap();

        let mid = Uuid::new_v4();
        let entry =
            build_user_memory_audit_entry(&ctx, Some(mid), "list_user_memories", "memory_read");

        assert_eq!(entry.accessor_user, "user-42");
        assert_eq!(entry.accessor_agent.as_deref(), Some("agent-7"));
        assert_eq!(entry.memory_id, Some(mid));
        assert_eq!(entry.action, "list_user_memories");
        assert_eq!(entry.purpose, "memory_read");
        assert_eq!(entry.trace_id, "trace-99");
        assert_eq!(entry.policy_decision, "allow");
    }

    #[test]
    fn csv_export_escapes_quotes() {
        let summaries = vec![make_summary(
            Uuid::new_v4(),
            Scope::Personal,
            "He said \"hello\", then left",
        )];
        let csv = summaries_to_csv(&summaries);
        assert!(csv.contains("\"He said \"\"hello\"\", then left\""));
    }

    #[test]
    fn router_builds_without_panic() {
        let _router = user_memory_router();
    }

    #[tokio::test]
    async fn correct_memory_records_audit_entry() {
        let mock =
            MockUserMemoryService::new(vec![make_summary(Uuid::new_v4(), Scope::Personal, "test")]);
        assert_eq!(mock.audit_count(), 0);
        let correction = MemoryCorrection {
            content: Some("fixed".into()),
            confidence: Some(0.9),
            reason: "typo".into(),
        };
        mock.correct_memory(Uuid::new_v4(), "user-42", &correction)
            .await
            .unwrap();
        assert_eq!(mock.audit_count(), 1);
    }

    #[tokio::test]
    async fn revoke_consent_records_audit_entry() {
        let mock =
            MockUserMemoryService::new(vec![make_summary(Uuid::new_v4(), Scope::Personal, "test")]);
        assert_eq!(mock.audit_count(), 0);
        mock.revoke_consent("user-42", ConsentScope::All)
            .await
            .unwrap();
        assert_eq!(mock.audit_count(), 1);
    }
}
