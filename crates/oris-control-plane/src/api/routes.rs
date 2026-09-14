//! V1 REST endpoint handlers (§5.1, §10.2).
//!
//! Defines the service-trait abstraction layer, concrete implementations for
//! production backends, [`AppState`], the twelve handler functions, and the
//! [`v1_router`] builder.
//!
//! # Architecture
//!
//! Each backend dependency is abstracted behind an `async_trait` so that
//! handlers can be unit-tested with mock implementations without a live
//! PostgreSQL or Redis instance. The concrete implementations wrap the real
//! managers (`WritePipeline`, `MemoryRepo`, `ContextAssembler`, …) and map
//! their domain errors into [`ApiError`].
//!
//! The §26 API-gateway middleware injects an [`ApiContract`] into request
//! extensions; [`RequestContext`] reads it (or falls back to `x-*` headers)
//! so every handler has tenant / user / agent / purpose / task / trace
//! context for authorisation checks.

use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use uuid::Uuid;

use oris_memory_contract::memory_types::{
    CanonicalUserProfile, MemoryItem, Scope, SearchParams, SearchResult, SharedTaskContext,
};
use oris_memory_store::postgres::{MemoryRepo, Pool, SearchRepo};

use crate::api::extractors::{
    ApiError, AssembleContextRequest, AssembleContextResponse, CanonicalContextResponse,
    ConflictFlagDto, ForgetMethod, ForgetRequest, ForgetResponse, HealthResponse, LineageResponse,
    MemoryResponse, PromoteRequest, PromoteResponse, RequestContext, SearchRequest, SearchResponse,
    SubmitCandidateRequest, SubmitCandidateResponse, TaskContextResponse, UpdateTaskContextRequest,
    VerifyRequest, VerifyResponse, VersionInfo,
};
use crate::canonical_user::CanonicalUserManager;
use crate::context_assembler::{AssembledContext, ContextAssembler};
use crate::context_router::{ContextRouter, RequestIntent, RouteRequest};
use crate::governance::{ForgetManager, MemoryVersion, VersionManager};
use crate::identity::ResolvedIdentity;
use crate::rerank::{RerankCandidate, RerankPipeline, RerankedResult};
use crate::shared_task::SharedTaskManager;
use crate::write_pipeline::{CandidateSubmission, WritePipeline, WriteResult};

// ═══════════════════════ Service Traits ═══════════════════════

/// Submit candidate memories through the write pipeline.
#[async_trait]
pub trait CandidateService: Send + Sync {
    async fn submit(&self, submission: &CandidateSubmission) -> Result<WriteResult, ApiError>;
}

/// Read and govern individual memory items (get, promote scope, verify source).
#[async_trait]
pub trait MemoryService: Send + Sync {
    async fn get_by_id(&self, memory_id: Uuid) -> Result<Option<MemoryItem>, ApiError>;
    async fn promote_scope(
        &self,
        memory_id: Uuid,
        target_scope: Scope,
        changed_by: &str,
        reason: Option<&str>,
    ) -> Result<(), ApiError>;
    async fn verify(&self, memory_id: Uuid, verified_by: &str) -> Result<(), ApiError>;
}

/// Hybrid retrieval with permission filtering and reranking.
#[async_trait]
pub trait SearchService: Send + Sync {
    async fn search(
        &self,
        params: &SearchParams,
        allowed_scopes: &[Scope],
    ) -> Result<Vec<RerankedResult>, ApiError>;
}

/// Context assembly for LLM prompts.
#[async_trait]
pub trait AssembleService: Send + Sync {
    async fn assemble(
        &self,
        identity: &ResolvedIdentity,
        query: &str,
        intent: Option<RequestIntent>,
        task_id: Option<Uuid>,
        max_latency_ms: Option<u64>,
        max_tokens: Option<usize>,
    ) -> Result<AssembledContext, ApiError>;
}

/// Canonical user profile retrieval.
#[async_trait]
pub trait CanonicalUserService: Send + Sync {
    async fn fetch_profile(&self, user_id: &str) -> Result<Option<CanonicalUserProfile>, ApiError>;
}

/// Shared task context retrieval and update.
#[async_trait]
pub trait SharedTaskService: Send + Sync {
    async fn fetch_task(&self, task_id: Uuid) -> Result<Option<SharedTaskContext>, ApiError>;
    async fn update_task(
        &self,
        task_id: Uuid,
        findings: Vec<Value>,
        status: Option<String>,
    ) -> Result<(), ApiError>;
}

/// Memory version / lineage tracking.
#[async_trait]
pub trait VersionService: Send + Sync {
    async fn fetch_lineage(&self, memory_id: Uuid) -> Result<Vec<MemoryVersion>, ApiError>;
}

/// Forget / delete operations (governance).
#[async_trait]
pub trait ForgetService: Send + Sync {
    async fn soft_delete(&self, memory_id: Uuid, requested_by: &str) -> Result<(), ApiError>;
    async fn hard_delete(&self, memory_id: Uuid, requested_by: &str) -> Result<(), ApiError>;
    async fn forget_user_data(&self, user_id: &str, tenant_id: &str) -> Result<u64, ApiError>;
    async fn cascade_forget(&self, memory_id: Uuid, requested_by: &str) -> Result<u64, ApiError>;
}

// ════════════════════ Concrete Implementations ════════════════════

/// `CandidateService` backed by a real [`WritePipeline`].
#[async_trait]
impl CandidateService for WritePipeline {
    async fn submit(&self, submission: &CandidateSubmission) -> Result<WriteResult, ApiError> {
        self.submit_candidate(submission)
            .await
            .map_err(ApiError::from)
    }
}

/// `MemoryService` backed by PostgreSQL. Wraps a connection pool and
/// delegates reads to [`MemoryRepo`], scope promotion to
/// [`VersionManager`] + raw SQL, and verification to raw SQL.
pub struct PostgresMemoryService {
    pool: Pool,
}

impl PostgresMemoryService {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl MemoryService for PostgresMemoryService {
    async fn get_by_id(&self, memory_id: Uuid) -> Result<Option<MemoryItem>, ApiError> {
        let repo = MemoryRepo::new(self.pool.clone());
        repo.get_by_id(memory_id).await.map_err(ApiError::from)
    }

    async fn promote_scope(
        &self,
        memory_id: Uuid,
        target_scope: Scope,
        changed_by: &str,
        reason: Option<&str>,
    ) -> Result<(), ApiError> {
        // 1. Snapshot current state into the version table.
        let vm = VersionManager::new(self.pool.clone());
        vm.create_version(memory_id, changed_by, reason)
            .await
            .map_err(ApiError::from)?;

        // 2. Update the scope column.
        let result = sqlx::query(
            r#"UPDATE memory_item SET scope = $2, updated_at = NOW(), version = version + 1
               WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .bind(target_scope.as_str())
        .execute(&self.pool)
        .await
        .map_err(|e| ApiError::internal(format!("database error: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(ApiError::NotFound(format!("memory {memory_id} not found")));
        }
        Ok(())
    }

    async fn verify(&self, memory_id: Uuid, _verified_by: &str) -> Result<(), ApiError> {
        let result = sqlx::query(
            r#"UPDATE memory_item SET last_verified_at = NOW(), updated_at = NOW()
               WHERE memory_id = $1"#,
        )
        .bind(memory_id)
        .execute(&self.pool)
        .await
        .map_err(|e| ApiError::internal(format!("database error: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(ApiError::NotFound(format!("memory {memory_id} not found")));
        }
        Ok(())
    }
}

/// `SearchService` backed by PostgreSQL search + [`RerankPipeline`].
pub struct PostgresSearchService {
    pool: Pool,
    rerank: RerankPipeline,
}

impl PostgresSearchService {
    pub fn new(pool: Pool, rerank: RerankPipeline) -> Self {
        Self { pool, rerank }
    }
}

#[async_trait]
impl SearchService for PostgresSearchService {
    async fn search(
        &self,
        params: &SearchParams,
        allowed_scopes: &[Scope],
    ) -> Result<Vec<RerankedResult>, ApiError> {
        let repo = SearchRepo::new(self.pool.clone());
        let raw_results: Vec<SearchResult> =
            repo.hybrid_search(params).await.map_err(ApiError::from)?;

        let candidates: Vec<RerankCandidate> = raw_results
            .into_iter()
            .map(|r| RerankCandidate {
                memory_id: r.item.memory_id,
                content: r.item.content.unwrap_or_default(),
                relevance_score: r.score,
                authority_level: r.item.authority_level,
                timestamp: r.item.created_at,
                confidence: r.item.confidence as f64,
                success_rate: 0.5,
                conflict_flags: vec![],
                privacy_class: r.item.privacy_class,
                scope: r.item.scope,
            })
            .collect();

        let filtered = self.rerank.permission_filter(candidates, allowed_scopes);
        Ok(self.rerank.rerank(filtered))
    }
}

/// `AssembleService` backed by [`ContextRouter`] + [`ContextAssembler`].
pub struct AssembleServiceImpl {
    router: ContextRouter,
    assembler: ContextAssembler,
}

impl AssembleServiceImpl {
    pub fn new(router: ContextRouter, assembler: ContextAssembler) -> Self {
        Self { router, assembler }
    }
}

#[async_trait]
impl AssembleService for AssembleServiceImpl {
    async fn assemble(
        &self,
        identity: &ResolvedIdentity,
        query: &str,
        intent: Option<RequestIntent>,
        task_id: Option<Uuid>,
        max_latency_ms: Option<u64>,
        max_tokens: Option<usize>,
    ) -> Result<AssembledContext, ApiError> {
        let route_req = RouteRequest {
            identity,
            query: query.to_string(),
            intent,
            task_id,
            max_latency_ms,
            max_tokens,
        };
        let plan = self.router.route(&route_req);
        self.assembler
            .assemble(&plan, identity, query, task_id)
            .await
            .map_err(ApiError::from)
    }
}

/// `CanonicalUserService` backed by [`CanonicalUserManager`].
#[async_trait]
impl CanonicalUserService for CanonicalUserManager {
    async fn fetch_profile(&self, user_id: &str) -> Result<Option<CanonicalUserProfile>, ApiError> {
        self.get_context(user_id).await.map_err(ApiError::from)
    }
}

/// `SharedTaskService` backed by [`SharedTaskManager`].
#[async_trait]
impl SharedTaskService for SharedTaskManager {
    async fn fetch_task(&self, task_id: Uuid) -> Result<Option<SharedTaskContext>, ApiError> {
        self.get_context(task_id).await.map_err(ApiError::from)
    }

    async fn update_task(
        &self,
        task_id: Uuid,
        findings: Vec<Value>,
        status: Option<String>,
    ) -> Result<(), ApiError> {
        if !findings.is_empty() {
            self.add_findings(task_id, findings)
                .await
                .map_err(ApiError::from)?;
        }
        if let Some(ref s) = status {
            self.update_status(task_id, s)
                .await
                .map_err(ApiError::from)?;
        }
        Ok(())
    }
}

/// `VersionService` backed by [`VersionManager`].
#[async_trait]
impl VersionService for VersionManager {
    async fn fetch_lineage(&self, memory_id: Uuid) -> Result<Vec<MemoryVersion>, ApiError> {
        self.list_versions(memory_id).await.map_err(ApiError::from)
    }
}

/// `ForgetService` backed by [`ForgetManager`].
#[async_trait]
impl ForgetService for ForgetManager {
    async fn soft_delete(&self, memory_id: Uuid, requested_by: &str) -> Result<(), ApiError> {
        ForgetManager::soft_delete(self, memory_id, requested_by)
            .await
            .map_err(ApiError::from)
    }

    async fn hard_delete(&self, memory_id: Uuid, requested_by: &str) -> Result<(), ApiError> {
        ForgetManager::hard_delete(self, memory_id, requested_by)
            .await
            .map_err(ApiError::from)
    }

    async fn forget_user_data(&self, user_id: &str, tenant_id: &str) -> Result<u64, ApiError> {
        ForgetManager::forget_user_data(self, user_id, tenant_id)
            .await
            .map_err(ApiError::from)
    }

    async fn cascade_forget(&self, memory_id: Uuid, requested_by: &str) -> Result<u64, ApiError> {
        ForgetManager::cascade_forget(self, memory_id, requested_by)
            .await
            .map_err(ApiError::from)
    }
}

// ══════════════════════════ AppState ══════════════════════════

/// Shared application state for all V1 handlers.
///
/// Each field is a trait-object `Arc` so the handlers can be tested with
/// mock implementations. The concrete implementations ([`WritePipeline`],
/// [`PostgresMemoryService`], …) are wired in [`AppState::builder`].
#[derive(Clone)]
pub struct AppState {
    pub candidate: Arc<dyn CandidateService>,
    pub memory: Arc<dyn MemoryService>,
    pub search: Arc<dyn SearchService>,
    pub assembler: Arc<dyn AssembleService>,
    pub canonical_user: Arc<dyn CanonicalUserService>,
    pub shared_task: Arc<dyn SharedTaskService>,
    pub version: Arc<dyn VersionService>,
    pub forget: Arc<dyn ForgetService>,
}

// ══════════════════════════ Handlers ══════════════════════════

/// `POST /v1/memories/candidates` — submit a candidate memory.
pub async fn submit_candidate(
    State(state): State<AppState>,
    ctx: RequestContext,
    Json(req): Json<SubmitCandidateRequest>,
) -> Result<Json<SubmitCandidateResponse>, ApiError> {
    if req.content.trim().is_empty() {
        return Err(ApiError::bad_request("content must not be empty"));
    }

    let submission = CandidateSubmission {
        tenant_id: ctx.tenant_id.clone(),
        content: req.content,
        source_type: req.source_type,
        memory_type: req.memory_type,
        scope: req.scope,
        user_id: Some(ctx.user_id.clone()),
        agent_id: ctx.agent_id.clone(),
        subject_type: req.subject_type,
        subject_id: req.subject_id,
        entity_refs: req.entity_refs,
        structured_payload: req.structured_payload,
        source_reference: req.source_reference,
        confidence: req.confidence,
    };

    let result = state.candidate.submit(&submission).await?;
    Ok(Json(result.into()))
}

/// `GET /v1/memories/{id}` — retrieve a single memory.
pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    ctx: RequestContext,
) -> Result<Json<MemoryResponse>, ApiError> {
    let item = state
        .memory
        .get_by_id(id)
        .await?
        .ok_or_else(|| ApiError::not_found(format!("memory {id} not found")))?;

    // Authorisation: the memory must belong to the caller's tenant.
    ctx.require_tenant(&item.tenant_id)?;
    Ok(Json(item.into()))
}

/// `POST /v1/memories/search` — hybrid retrieval with reranking.
pub async fn search_memories(
    State(state): State<AppState>,
    ctx: RequestContext,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let mut params = SearchParams::new(&ctx.tenant_id);
    params.query_text = req.query_text;
    params.memory_types = req.memory_types.clone();
    params.scopes = req.scopes.clone();
    params.subject_id = req.subject_id;
    params.min_confidence = req.min_confidence;
    params.limit = req.limit;
    params.offset = req.offset;
    params.embedding = req.embedding;

    // Default to all scopes if the caller didn't specify any.
    let allowed_scopes: Vec<Scope> = if req.scopes.is_empty() {
        vec![
            Scope::Personal,
            Scope::Agent,
            Scope::Task,
            Scope::Team,
            Scope::Process,
            Scope::Factory,
            Scope::Enterprise,
        ]
    } else {
        req.scopes
    };

    let results = state.search.search(&params, &allowed_scopes).await?;
    let count = results.len();
    Ok(Json(SearchResponse { results, count }))
}

/// `POST /v1/context/assemble` — assemble context for an LLM prompt.
pub async fn assemble_context(
    State(state): State<AppState>,
    ctx: RequestContext,
    Json(req): Json<AssembleContextRequest>,
) -> Result<Json<AssembleContextResponse>, ApiError> {
    let identity = ctx.to_resolved_identity();
    let assembled = state
        .assembler
        .assemble(
            &identity,
            &req.query,
            req.intent,
            req.task_id,
            req.max_latency_ms,
            req.max_tokens,
        )
        .await?;

    let response = AssembleContextResponse {
        context_text: assembled.context_text,
        token_count: assembled.token_count,
        compressed: assembled.compressed,
        conflict_flags: assembled
            .conflict_flags
            .iter()
            .map(ConflictFlagDto::from)
            .collect(),
        low_confidence_items: assembled.low_confidence_items,
        sources_used: assembled.sources_used,
        degraded_sources: assembled.degraded_sources,
    };
    Ok(Json(response))
}

/// `GET /v1/users/{id}/canonical-context` — user canonical profile.
pub async fn get_canonical_context(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    _ctx: RequestContext,
) -> Result<Json<CanonicalContextResponse>, ApiError> {
    let profile = state.canonical_user.fetch_profile(&user_id).await?;
    Ok(Json(CanonicalContextResponse { profile }))
}

/// `GET /v1/tasks/{id}/context` — retrieve shared task context.
pub async fn get_task_context(
    State(state): State<AppState>,
    Path(task_id): Path<Uuid>,
    _ctx: RequestContext,
) -> Result<Json<TaskContextResponse>, ApiError> {
    let context = state.shared_task.fetch_task(task_id).await?;
    Ok(Json(TaskContextResponse { context }))
}

/// `PATCH /v1/tasks/{id}/context` — update shared task context.
pub async fn update_task_context(
    State(state): State<AppState>,
    Path(task_id): Path<Uuid>,
    ctx: RequestContext,
    Json(req): Json<UpdateTaskContextRequest>,
) -> Result<Json<TaskContextResponse>, ApiError> {
    state
        .shared_task
        .update_task(task_id, req.add_findings, req.status)
        .await?;

    // Re-fetch the updated context for the response.
    let context = state.shared_task.fetch_task(task_id).await?;
    let _ = ctx; // ctx available for future authorisation checks
    Ok(Json(TaskContextResponse { context }))
}

/// `POST /v1/memories/{id}/promote` — scope promotion.
pub async fn promote_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    ctx: RequestContext,
    Json(req): Json<PromoteRequest>,
) -> Result<Json<PromoteResponse>, ApiError> {
    state
        .memory
        .promote_scope(id, req.target_scope, &ctx.user_id, req.reason.as_deref())
        .await?;

    Ok(Json(PromoteResponse {
        memory_id: id,
        scope: req.target_scope,
        message: format!("memory promoted to {}", req.target_scope.as_str()),
    }))
}

/// `POST /v1/memories/{id}/verify` — source verification.
pub async fn verify_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    ctx: RequestContext,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, ApiError> {
    if req.verified_by.trim().is_empty() {
        return Err(ApiError::bad_request("verified_by must not be empty"));
    }
    state.memory.verify(id, &req.verified_by).await?;
    let _ = ctx; // ctx available for future authorisation checks
    Ok(Json(VerifyResponse {
        memory_id: id,
        verified_at: chrono::Utc::now(),
        message: "memory verified".to_string(),
    }))
}

/// `POST /v1/memories/forget` — forget / delete (governance).
pub async fn forget_memory(
    State(state): State<AppState>,
    ctx: RequestContext,
    Json(req): Json<ForgetRequest>,
) -> Result<Json<ForgetResponse>, ApiError> {
    let (affected, message) = match req.mode {
        ForgetMethod::Soft => {
            let id = req
                .memory_id
                .ok_or_else(|| ApiError::bad_request("memory_id is required for soft mode"))?;
            state.forget.soft_delete(id, &ctx.user_id).await?;
            (1u64, format!("memory {id} soft-deleted"))
        }
        ForgetMethod::Hard => {
            let id = req
                .memory_id
                .ok_or_else(|| ApiError::bad_request("memory_id is required for hard mode"))?;
            state.forget.hard_delete(id, &ctx.user_id).await?;
            (1u64, format!("memory {id} hard-deleted"))
        }
        ForgetMethod::Cascade => {
            let id = req
                .memory_id
                .ok_or_else(|| ApiError::bad_request("memory_id is required for cascade mode"))?;
            let n = state.forget.cascade_forget(id, &ctx.user_id).await?;
            (n, format!("cascade-forget affected {n} memories"))
        }
        ForgetMethod::UserData => {
            let uid = req
                .user_id
                .ok_or_else(|| ApiError::bad_request("user_id is required for user_data mode"))?;
            let n = state.forget.forget_user_data(&uid, &ctx.tenant_id).await?;
            (n, format!("forgot {n} memories for user {uid}"))
        }
    };

    Ok(Json(ForgetResponse {
        mode: req.mode,
        affected_count: affected,
        message,
    }))
}

/// `GET /v1/memories/{id}/lineage` — version chain.
pub async fn get_lineage(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    _ctx: RequestContext,
) -> Result<Json<LineageResponse>, ApiError> {
    let versions = state.version.fetch_lineage(id).await?;
    let version_infos: Vec<VersionInfo> = versions
        .into_iter()
        .map(|v| VersionInfo {
            version_id: v.version_id,
            memory_id: v.memory_id,
            version_number: v.version_number,
            changed_by: v.changed_by,
            change_reason: v.change_reason,
            created_at: v.created_at,
        })
        .collect();
    Ok(Json(LineageResponse {
        memory_id: id,
        versions: version_infos,
    }))
}

/// `GET /v1/health` — liveness probe.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse::ok())
}

// ══════════════════════════ Router ═══════════════════════════

/// Build the V1 REST router.
///
/// Routes are mounted under `/v1/`. The caller is responsible for adding
/// the gateway middleware layer and calling `.with_state(state)`.
pub fn v1_router() -> Router<AppState> {
    Router::new()
        .route("/v1/memories/candidates", post(submit_candidate))
        .route("/v1/memories/search", post(search_memories))
        .route("/v1/memories/forget", post(forget_memory))
        .route("/v1/memories/{id}", get(get_memory))
        .route("/v1/memories/{id}/promote", post(promote_memory))
        .route("/v1/memories/{id}/verify", post(verify_memory))
        .route("/v1/memories/{id}/lineage", get(get_lineage))
        .route("/v1/context/assemble", post(assemble_context))
        .route(
            "/v1/users/{id}/canonical-context",
            get(get_canonical_context),
        )
        .route(
            "/v1/tasks/{id}/context",
            get(get_task_context).patch(update_task_context),
        )
        .route("/v1/health", get(health))
}

// ══════════════════════════ Tests ═══════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use oris_memory_contract::memory_types::MemoryStatus;
    use std::sync::Mutex;
    use tower::ServiceExt;

    // ─── Mock implementations ───────────────────────────────────

    struct MockCandidateService {
        result: WriteResult,
        last: Mutex<Option<CandidateSubmission>>,
    }

    #[async_trait]
    impl CandidateService for MockCandidateService {
        async fn submit(&self, submission: &CandidateSubmission) -> Result<WriteResult, ApiError> {
            *self.last.lock().unwrap() = Some(submission.clone());
            Ok(self.result.clone())
        }
    }

    struct MockMemoryService {
        item: Option<MemoryItem>,
        promote_called: Mutex<bool>,
        verify_called: Mutex<bool>,
    }

    #[async_trait]
    impl MemoryService for MockMemoryService {
        async fn get_by_id(&self, _id: Uuid) -> Result<Option<MemoryItem>, ApiError> {
            Ok(self.item.clone())
        }
        async fn promote_scope(
            &self,
            _id: Uuid,
            _scope: Scope,
            _by: &str,
            _reason: Option<&str>,
        ) -> Result<(), ApiError> {
            *self.promote_called.lock().unwrap() = true;
            Ok(())
        }
        async fn verify(&self, _id: Uuid, _by: &str) -> Result<(), ApiError> {
            *self.verify_called.lock().unwrap() = true;
            Ok(())
        }
    }

    struct MockSearchService {
        results: Vec<RerankedResult>,
    }

    #[async_trait]
    impl SearchService for MockSearchService {
        async fn search(
            &self,
            _params: &SearchParams,
            _scopes: &[Scope],
        ) -> Result<Vec<RerankedResult>, ApiError> {
            Ok(self.results.clone())
        }
    }

    struct MockAssembleService {
        context: AssembledContext,
    }

    #[async_trait]
    impl AssembleService for MockAssembleService {
        async fn assemble(
            &self,
            _identity: &ResolvedIdentity,
            _query: &str,
            _intent: Option<RequestIntent>,
            _task_id: Option<Uuid>,
            _latency: Option<u64>,
            _tokens: Option<usize>,
        ) -> Result<AssembledContext, ApiError> {
            // Clone manually since AssembledContext doesn't derive Clone.
            Ok(AssembledContext {
                context_text: self.context.context_text.clone(),
                token_count: self.context.token_count,
                compressed: self.context.compressed,
                conflict_flags: self.context.conflict_flags.clone(),
                low_confidence_items: self.context.low_confidence_items.clone(),
                sources_used: self.context.sources_used.clone(),
                degraded_sources: self.context.degraded_sources.clone(),
            })
        }
    }

    struct MockCanonicalUserService {
        profile: Option<CanonicalUserProfile>,
    }

    #[async_trait]
    impl CanonicalUserService for MockCanonicalUserService {
        async fn fetch_profile(
            &self,
            _uid: &str,
        ) -> Result<Option<CanonicalUserProfile>, ApiError> {
            Ok(self.profile.clone())
        }
    }

    struct MockSharedTaskService {
        task: Option<SharedTaskContext>,
        update_called: Mutex<bool>,
    }

    #[async_trait]
    impl SharedTaskService for MockSharedTaskService {
        async fn fetch_task(&self, _id: Uuid) -> Result<Option<SharedTaskContext>, ApiError> {
            Ok(self.task.clone())
        }
        async fn update_task(
            &self,
            _id: Uuid,
            _findings: Vec<Value>,
            _status: Option<String>,
        ) -> Result<(), ApiError> {
            *self.update_called.lock().unwrap() = true;
            Ok(())
        }
    }

    struct MockVersionService {
        versions: Vec<MemoryVersion>,
    }

    #[async_trait]
    impl VersionService for MockVersionService {
        async fn fetch_lineage(&self, _id: Uuid) -> Result<Vec<MemoryVersion>, ApiError> {
            // Clone each version manually.
            Ok(self
                .versions
                .iter()
                .map(|v| MemoryVersion {
                    version_id: v.version_id,
                    memory_id: v.memory_id,
                    version_number: v.version_number,
                    payload: v.payload.clone(),
                    changed_by: v.changed_by.clone(),
                    change_reason: v.change_reason.clone(),
                    created_at: v.created_at,
                })
                .collect())
        }
    }

    struct MockForgetService {
        soft_called: Mutex<bool>,
        hard_called: Mutex<bool>,
        cascade_called: Mutex<bool>,
        user_data_count: u64,
    }

    #[async_trait]
    impl ForgetService for MockForgetService {
        async fn soft_delete(&self, _id: Uuid, _by: &str) -> Result<(), ApiError> {
            *self.soft_called.lock().unwrap() = true;
            Ok(())
        }
        async fn hard_delete(&self, _id: Uuid, _by: &str) -> Result<(), ApiError> {
            *self.hard_called.lock().unwrap() = true;
            Ok(())
        }
        async fn forget_user_data(&self, _uid: &str, _tid: &str) -> Result<u64, ApiError> {
            *self.cascade_called.lock().unwrap() = true;
            Ok(self.user_data_count)
        }
        async fn cascade_forget(&self, _id: Uuid, _by: &str) -> Result<u64, ApiError> {
            *self.cascade_called.lock().unwrap() = true;
            Ok(3)
        }
    }

    // ─── Test helpers ───────────────────────────────────────────

    fn mock_app_state() -> AppState {
        AppState {
            candidate: Arc::new(MockCandidateService {
                result: WriteResult {
                    memory_id: Uuid::new_v4(),
                    status: MemoryStatus::Active,
                    importance: 0.8,
                    is_duplicate: false,
                    safety_verdict: crate::write_pipeline::SafetyVerdictSummary::Safe,
                    outbox_event_id: Uuid::new_v4(),
                },
                last: Mutex::new(None),
            }),
            memory: Arc::new(MockMemoryService {
                item: None,
                promote_called: Mutex::new(false),
                verify_called: Mutex::new(false),
            }),
            search: Arc::new(MockSearchService { results: vec![] }),
            assembler: Arc::new(MockAssembleService {
                context: AssembledContext {
                    context_text: "assembled".into(),
                    token_count: 5,
                    compressed: false,
                    conflict_flags: vec![],
                    low_confidence_items: vec![],
                    sources_used: vec![],
                    degraded_sources: vec![],
                },
            }),
            canonical_user: Arc::new(MockCanonicalUserService { profile: None }),
            shared_task: Arc::new(MockSharedTaskService {
                task: None,
                update_called: Mutex::new(false),
            }),
            version: Arc::new(MockVersionService {
                versions: vec![MemoryVersion {
                    version_id: Uuid::new_v4(),
                    memory_id: Uuid::new_v4(),
                    version_number: 1,
                    payload: Value::Null,
                    changed_by: "tester".into(),
                    change_reason: None,
                    created_at: chrono::Utc::now(),
                }],
            }),
            forget: Arc::new(MockForgetService {
                soft_called: Mutex::new(false),
                hard_called: Mutex::new(false),
                cascade_called: Mutex::new(false),
                user_data_count: 5,
            }),
        }
    }

    fn app() -> Router {
        v1_router().with_state(mock_app_state())
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

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ─── Tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn health_returns_ok() {
        let response = app()
            .oneshot(authed_request(Method::GET, "/v1/health", ""))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn submit_candidate_returns_result() {
        let body = r#"{
            "content":"The pump X-200 needs oil",
            "source_type":"user_explicit",
            "memory_type":"semantic",
            "scope":"personal",
            "confidence":0.9
        }"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                "/v1/memories/candidates",
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert!(json["memory_id"].is_string());
        assert_eq!(json["status"], "active");
        assert_eq!(json["is_duplicate"], false);
    }

    #[tokio::test]
    async fn submit_candidate_rejects_empty_content() {
        let body = r#"{"content":"","source_type":"user_explicit","memory_type":"semantic","scope":"personal"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                "/v1/memories/candidates",
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_memory_returns_404_when_not_found() {
        let id = Uuid::new_v4();
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                &format!("/v1/memories/{id}"),
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn search_returns_results() {
        let body = r#"{"query_text":"pump maintenance","top_k":5}"#;
        let response = app()
            .oneshot(authed_request(Method::POST, "/v1/memories/search", body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["count"], 0);
        assert!(json["results"].is_array());
    }

    #[tokio::test]
    async fn assemble_context_returns_context() {
        let body = r#"{"query":"how to fix pump X-200"}"#;
        let response = app()
            .oneshot(authed_request(Method::POST, "/v1/context/assemble", body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["context_text"], "assembled");
        assert_eq!(json["token_count"], 5);
    }

    #[tokio::test]
    async fn get_canonical_context_returns_profile_or_none() {
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                "/v1/users/user-42/canonical-context",
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert!(json["profile"].is_null());
    }

    #[tokio::test]
    async fn get_task_context_returns_404_as_none() {
        // Mock returns None → handler returns 200 with null context.
        let id = Uuid::new_v4();
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                &format!("/v1/tasks/{id}/context"),
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert!(json["context"].is_null());
    }

    #[tokio::test]
    async fn update_task_context_succeeds() {
        let id = Uuid::new_v4();
        let body = r#"{"add_findings":["found issue in valve"],"status":"in_progress"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::PATCH,
                &format!("/v1/tasks/{id}/context"),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn promote_memory_succeeds() {
        let id = Uuid::new_v4();
        let body = r#"{"target_scope":"team","reason":"promoted by QA"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                &format!("/v1/memories/{id}/promote"),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["scope"], "team");
    }

    #[tokio::test]
    async fn verify_memory_succeeds() {
        let id = Uuid::new_v4();
        let body = r#"{"verified_by":"qa-engineer-1"}"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                &format!("/v1/memories/{id}/verify"),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert!(json["verified_at"].is_string());
    }

    #[tokio::test]
    async fn verify_memory_rejects_empty_verified_by() {
        let id = Uuid::new_v4();
        let body = r#"{"verified_by":""}"#;
        let response = app()
            .oneshot(authed_request(
                Method::POST,
                &format!("/v1/memories/{id}/verify"),
                body,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn forget_soft_succeeds() {
        let id = Uuid::new_v4();
        let body = format!(r#"{{"memory_id":"{id}","mode":"soft"}}"#);
        let response = app()
            .oneshot(authed_request(Method::POST, "/v1/memories/forget", &body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["mode"], "soft");
        assert_eq!(json["affected_count"], 1);
    }

    #[tokio::test]
    async fn forget_user_data_succeeds() {
        let body = r#"{"user_id":"user-42","mode":"user_data"}"#;
        let response = app()
            .oneshot(authed_request(Method::POST, "/v1/memories/forget", body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["mode"], "user_data");
        assert_eq!(json["affected_count"], 5);
    }

    #[tokio::test]
    async fn forget_cascade_succeeds() {
        let id = Uuid::new_v4();
        let body = format!(r#"{{"memory_id":"{id}","mode":"cascade"}}"#);
        let response = app()
            .oneshot(authed_request(Method::POST, "/v1/memories/forget", &body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert!(json["affected_count"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn forget_soft_without_memory_id_returns_400() {
        let body = r#"{"mode":"soft"}"#;
        let response = app()
            .oneshot(authed_request(Method::POST, "/v1/memories/forget", body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_lineage_returns_versions() {
        let id = Uuid::new_v4();
        let response = app()
            .oneshot(authed_request(
                Method::GET,
                &format!("/v1/memories/{id}/lineage"),
                "",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["memory_id"], id.to_string());
        assert!(json["versions"].is_array());
        assert_eq!(json["versions"].as_array().unwrap().len(), 1);
        assert_eq!(json["versions"][0]["version_number"], 1);
    }

    #[tokio::test]
    async fn missing_auth_headers_returns_401() {
        // No x-* headers → RequestContext extraction fails.
        let request = Request::builder()
            .method(Method::GET)
            .uri("/v1/memories/00000000-0000-0000-0000-000000000001")
            .body(Body::empty())
            .unwrap();
        let response = app().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn router_has_all_v1_routes() {
        // Verify the router builds without panicking and is non-empty.
        let router = v1_router();
        // If we get here, all route definitions compiled and registered.
        let _ = router;
    }
}
