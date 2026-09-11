//! MCP Tools V1 — LLM agent tool definitions wrapping REST API service traits (§5.3).
//!
//! Each of the 13 tools maps to one or more service traits from
//! [`crate::api::routes`], so an LLM agent can invoke memory operations
//! through a uniform tool-call interface instead of raw HTTP.
//!
//! # Tools
//!
//! | Tool | Service trait | REST endpoint |
//! |------|--------------|--------------|
//! | `oris_memory_remember` | `CandidateService` | `POST /v1/memories/candidates` |
//! | `oris_memory_recall` | `MemoryService` | `GET /v1/memories/{id}` |
//! | `oris_memory_search` | `SearchService` | `POST /v1/memories/search` |
//! | `oris_memory_get_context` | `AssembleService` | `POST /v1/context/assemble` |
//! | `oris_memory_update` | `VersionService` | `GET /v1/memories/{id}/lineage` |
//! | `oris_memory_forget` | `ForgetService` | `POST /v1/memories/forget` |
//! | `oris_memory_share` | `AssembleService` | (ContextPackage projection) |
//! | `oris_memory_promote` | `MemoryService` | `POST /v1/memories/{id}/promote` |
//! | `oris_memory_reflect` | `CandidateService` | (experience reflection) |
//! | `oris_memory_verify` | `MemoryService` | `POST /v1/memories/{id}/verify` |
//! | `oris_user_get_context` | `CanonicalUserService` | `GET /v1/users/{id}/canonical-context` |
//! | `oris_task_get_context` | `SharedTaskService` | `GET /v1/tasks/{id}/context` |
//! | `oris_task_update_context` | `SharedTaskService` | `PATCH /v1/tasks/{id}/context` |

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use oris_memory_contract::memory_types::{
    CanonicalUserProfile, MemoryItem, MemoryType, Scope, SearchParams, SharedTaskContext,
    SourceType,
};

use crate::api::extractors::{ApiError, ForgetMethod, RequestContext};
use crate::api::routes::{
    AppState, AssembleService, CandidateService, CanonicalUserService, ForgetService,
    MemoryService, SearchService, SharedTaskService, VersionService,
};
use crate::context_assembler::{AssembledContext, ConflictFlag, ContextSource};
use crate::context_router::RequestIntent;
use crate::governance::MemoryVersion;
use crate::identity::ResolvedIdentity;
use crate::rerank::RerankedResult;
use crate::write_pipeline::{CandidateSubmission, WriteResult};

// ═════════════════════════════ Error ═════════════════════════════

/// Errors emitted by the MCP tool layer.
#[derive(Debug, thiserror::Error)]
pub enum McpToolError {
    /// The requested tool name is not registered.
    #[error("unknown tool: {0}")]
    UnknownTool(String),

    /// No `AppState` was attached to the registry — tools cannot execute.
    #[error("no application state attached to registry")]
    NoState,

    /// A required parameter is missing or has the wrong type.
    #[error("invalid parameters: {0}")]
    InvalidParams(String),

    /// A downstream service returned an [`ApiError`].
    #[error("service error: {0}")]
    Service(String),

    /// A serialisation failure when building the JSON result.
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<ApiError> for McpToolError {
    fn from(err: ApiError) -> Self {
        Self::Service(err.to_string())
    }
}

impl From<serde_json::Error> for McpToolError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl McpToolError {
    fn missing(field: &str) -> Self {
        Self::InvalidParams(format!("missing or invalid parameter: {field}"))
    }
}

// ═══════════════════════ Tool Definition ═══════════════════════

/// Static definition of a single MCP tool exposed to LLM agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    /// Tool name, e.g. `"oris_memory_remember"`.
    pub name: &'static str,
    /// Human-readable description shown to the LLM.
    pub description: &'static str,
    /// JSON Schema describing the tool's input parameters.
    pub input_schema: Value,
    /// Names of parameters the caller must supply.
    pub required_params: Vec<String>,
}

// ═══════════════════════ Request / Response ═══════════════════════

/// A single tool invocation request from an LLM agent.
#[derive(Debug, Clone, Serialize)]
pub struct McpToolRequest {
    /// Name of the tool to invoke.
    pub tool_name: String,
    /// Request context (tenant, user, agent, purpose, …).
    pub context: RequestContext,
    /// Tool-specific input parameters as a JSON object.
    pub params: Value,
}

/// The result of a tool invocation, ready to be sent back to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResponse {
    /// Name of the tool that produced this response.
    pub tool_name: String,
    /// Serialised tool result (or error details when `is_error` is true).
    pub result: Value,
    /// Whether the result represents an error condition.
    pub is_error: bool,
}

impl McpToolResponse {
    fn ok(tool_name: &str, result: Value) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            result,
            is_error: false,
        }
    }

    fn error(tool_name: &str, message: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            result: json!({ "error": message }),
            is_error: true,
        }
    }
}

// ═════════════════════════════ Registry ═════════════════════════════

/// Holds all 13 tool definitions and an optional [`AppState`] for live
/// execution against real backends.
pub struct McpToolRegistry {
    tools: Vec<McpToolDefinition>,
    state: Option<AppState>,
}

impl Default for McpToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl McpToolRegistry {
    /// Create a registry with all 13 tool definitions but no backend state.
    ///
    /// Call [`with_state`](Self::with_state) to attach an `AppState` before
    /// invoking [`execute_tool`](Self::execute_tool).
    pub fn new() -> Self {
        Self {
            tools: tool_definitions(),
            state: None,
        }
    }

    /// Attach an [`AppState`] so tools can execute against real backends.
    pub fn with_state(mut self, state: AppState) -> Self {
        self.state = Some(state);
        self
    }

    /// Look up a tool definition by name.
    pub fn get_tool(&self, name: &str) -> Option<&McpToolDefinition> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Return all registered tool definitions.
    pub fn list_tools(&self) -> &[McpToolDefinition] {
        &self.tools
    }

    /// Execute a tool by dispatching to the appropriate service trait method.
    ///
    /// Returns [`McpToolError::UnknownTool`] when the name is not registered,
    /// [`McpToolError::NoState`] when no `AppState` has been attached.
    pub async fn execute_tool(
        &self,
        request: &McpToolRequest,
    ) -> Result<McpToolResponse, McpToolError> {
        // Validate tool name first — works even without state.
        if self.get_tool(&request.tool_name).is_none() {
            return Err(McpToolError::UnknownTool(request.tool_name.clone()));
        }

        let state = self.state.as_ref().ok_or(McpToolError::NoState)?;

        let params = &request.params;
        let ctx = &request.context;
        let name = request.tool_name.as_str();

        let result = match name {
            "oris_memory_remember" => exec_remember(state, ctx, params).await,
            "oris_memory_recall" => exec_recall(state, ctx, params).await,
            "oris_memory_search" => exec_search(state, ctx, params).await,
            "oris_memory_get_context" => exec_get_context(state, ctx, params).await,
            "oris_memory_update" => exec_update(state, ctx, params).await,
            "oris_memory_forget" => exec_forget(state, ctx, params).await,
            "oris_memory_share" => exec_share(state, ctx, params).await,
            "oris_memory_promote" => exec_promote(state, ctx, params).await,
            "oris_memory_reflect" => exec_reflect(state, ctx, params).await,
            "oris_memory_verify" => exec_verify(state, ctx, params).await,
            "oris_user_get_context" => exec_user_get_context(state, ctx, params).await,
            "oris_task_get_context" => exec_task_get_context(state, ctx, params).await,
            "oris_task_update_context" => exec_task_update_context(state, ctx, params).await,
            _ => return Err(McpToolError::UnknownTool(name.to_string())),
        };

        match result {
            Ok(value) => Ok(McpToolResponse::ok(name, value)),
            Err(e) => Ok(McpToolResponse::error(name, &e.to_string())),
        }
    }
}

// ═══════════════════════ Tool Definitions ═══════════════════════

/// Build the 13 MCP tool definitions with JSON Schema input schemas.
fn tool_definitions() -> Vec<McpToolDefinition> {
    vec![
        McpToolDefinition {
            name: "oris_memory_remember",
            description: "Submit a candidate memory through the write pipeline (ingestion, scoring, dedup, safety scan).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Raw content text of the memory."},
                    "source_type": {"type": "string", "enum": ["user_explicit","agent_inferred","system_observed","external_import"], "description": "Provenance source type."},
                    "memory_type": {"type": "string", "enum": ["semantic","episodic","procedure","preference","decision"], "description": "Memory classification."},
                    "scope": {"type": "string", "enum": ["personal","agent","task","team","process","factory","enterprise"], "description": "Organisational breadth."},
                    "subject_type": {"type": "string", "description": "Optional subject type (e.g. equipment, person)."},
                    "subject_id": {"type": "string", "description": "Optional subject identifier."},
                    "entity_refs": {"type": "array", "items": {"type": "string"}, "description": "Entity references."},
                    "structured_payload": {"description": "Optional structured JSONB payload."},
                    "source_reference": {"type": "string", "description": "Optional reference to originating system/document."},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Caller confidence (0.0–1.0). Defaults to 0.5."}
                }
            }),
            required_params: vec!["content".into(), "source_type".into(), "memory_type".into(), "scope".into()],
        },
        McpToolDefinition {
            name: "oris_memory_recall",
            description: "Recall a single memory by ID, subject to tenant permission checks.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid", "description": "Canonical memory UUID."}
                }
            }),
            required_params: vec!["memory_id".into()],
        },
        McpToolDefinition {
            name: "oris_memory_search",
            description: "Hybrid retrieval (keyword + vector + structured) with permission filtering and reranking.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "Natural-language or keyword query."},
                    "memory_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by memory types."},
                    "scopes": {"type": "array", "items": {"type": "string"}, "description": "Filter by scopes (default: all caller-accessible)."},
                    "subject_id": {"type": "string", "description": "Filter by subject ID."},
                    "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Minimum confidence threshold."},
                    "limit": {"type": "integer", "minimum": 1, "description": "Max results after reranking (default 50)."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Pagination offset."},
                    "embedding": {"type": "array", "items": {"type": "number"}, "description": "Optional embedding vector for semantic search."}
                }
            }),
            required_params: vec![],
        },
        McpToolDefinition {
            name: "oris_memory_get_context",
            description: "Assemble minimal high-value context for an LLM prompt from multiple sources.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's query / prompt text."},
                    "intent": {"type": "string", "enum": ["chat","personal_task","business_query","cross_agent_task","decision_support","deep_research"], "description": "Optional explicit intent override."},
                    "task_id": {"type": "string", "format": "uuid", "description": "Task ID for cross-agent context assembly."},
                    "max_latency_ms": {"type": "integer", "minimum": 1, "description": "Caller cap on latency (ms)."},
                    "max_tokens": {"type": "integer", "minimum": 1, "description": "Caller cap on token budget."}
                }
            }),
            required_params: vec!["query".into()],
        },
        McpToolDefinition {
            name: "oris_memory_update",
            description: "Retrieve the version lineage (change history) for a memory, enabling versioned updates and rollback.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid", "description": "Canonical memory UUID."}
                }
            }),
            required_params: vec!["memory_id".into()],
        },
        McpToolDefinition {
            name: "oris_memory_forget",
            description: "Forget / delete memories (soft, hard, cascade, or GDPR user-data bulk).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["soft","hard","cascade","user_data"], "description": "Forget operation mode."},
                    "memory_id": {"type": "string", "format": "uuid", "description": "Required for soft/hard/cascade modes."},
                    "user_id": {"type": "string", "description": "Required for user_data mode."}
                }
            }),
            required_params: vec!["mode".into()],
        },
        McpToolDefinition {
            name: "oris_memory_share",
            description: "Assemble context and package it as a cross-Agent ContextPackage for controlled sharing.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to assemble context for."},
                    "target_agent": {"type": "string", "description": "Agent receiving the shared package."},
                    "purpose": {"type": "string", "description": "Stated purpose of the share (for audit + policy)."},
                    "task_id": {"type": "string", "format": "uuid", "description": "Optional shared-task ID."},
                    "memory_refs": {"type": "array", "items": {"type": "string", "format": "uuid"}, "description": "Specific memory IDs to include."},
                    "max_tokens": {"type": "integer", "minimum": 1, "description": "Token budget for the package."}
                }
            }),
            required_params: vec!["query".into(), "target_agent".into()],
        },
        McpToolDefinition {
            name: "oris_memory_promote",
            description: "Promote a memory's scope to a wider organisational breadth (e.g. personal → team).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid", "description": "Memory to promote."},
                    "target_scope": {"type": "string", "enum": ["personal","agent","task","team","process","factory","enterprise"], "description": "Target scope."},
                    "reason": {"type": "string", "description": "Human-readable reason for the promotion."}
                }
            }),
            required_params: vec!["memory_id".into(), "target_scope".into()],
        },
        McpToolDefinition {
            name: "oris_memory_reflect",
            description: "Reflect on an experience and submit it as a candidate memory (experience → knowledge).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The reflected experience as structured text."},
                    "memory_type": {"type": "string", "enum": ["semantic","episodic","procedure","preference","decision"], "description": "Memory classification (default: procedure)."},
                    "scope": {"type": "string", "enum": ["personal","agent","task","team","process","factory","enterprise"], "description": "Organisational breadth (default: agent)."},
                    "source_type": {"type": "string", "enum": ["user_explicit","agent_inferred","system_observed","external_import"], "description": "Provenance source type (default: agent_inferred)."},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Caller confidence (default 0.5)."},
                    "subject_type": {"type": "string", "description": "Optional subject type."},
                    "subject_id": {"type": "string", "description": "Optional subject identifier."}
                }
            }),
            required_params: vec!["content".into()],
        },
        McpToolDefinition {
            name: "oris_memory_verify",
            description: "Verify the source of a memory, marking it as verified for downstream consumers.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid", "description": "Memory to verify."},
                    "verified_by": {"type": "string", "description": "Who is performing the verification."}
                }
            }),
            required_params: vec!["memory_id".into(), "verified_by".into()],
        },
        McpToolDefinition {
            name: "oris_user_get_context",
            description: "Retrieve the canonical user profile (identity, role, preferences, entities).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Canonical user identifier."}
                }
            }),
            required_params: vec!["user_id".into()],
        },
        McpToolDefinition {
            name: "oris_task_get_context",
            description: "Retrieve the shared task context (goal, constraints, findings, decisions).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "format": "uuid", "description": "Task UUID."}
                }
            }),
            required_params: vec!["task_id".into()],
        },
        McpToolDefinition {
            name: "oris_task_update_context",
            description: "Update the shared task context by appending findings and/or changing status.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "format": "uuid", "description": "Task UUID."},
                    "add_findings": {"type": "array", "items": {}, "description": "Findings to append."},
                    "status": {"type": "string", "description": "Optional new task status."}
                }
            }),
            required_params: vec!["task_id".into()],
        },
    ]
}

// ═══════════════════════ Dispatch — each tool ═══════════════════════

async fn exec_remember(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let content = p
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| McpToolError::missing("content"))?;
    if content.trim().is_empty() {
        return Err(McpToolError::InvalidParams(
            "content must not be empty".into(),
        ));
    }

    let submission = CandidateSubmission {
        tenant_id: ctx.tenant_id.clone(),
        content: content.to_string(),
        source_type: parse_enum(p, "source_type", default_source_type())?,
        memory_type: parse_enum(p, "memory_type", default_memory_type())?,
        scope: parse_enum(p, "scope", Scope::Personal)?,
        user_id: Some(ctx.user_id.clone()),
        agent_id: ctx.agent_id.clone(),
        subject_type: p
            .get("subject_type")
            .and_then(Value::as_str)
            .map(Into::into),
        subject_id: p.get("subject_id").and_then(Value::as_str).map(Into::into),
        entity_refs: p
            .get("entity_refs")
            .and_then(Value::as_array)
            .map(|a| a.iter().cloned().collect())
            .unwrap_or_default(),
        structured_payload: p.get("structured_payload").cloned(),
        source_reference: p
            .get("source_reference")
            .and_then(Value::as_str)
            .map(Into::into),
        confidence: p
            .get("confidence")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(0.5),
    };

    let result = state.candidate.submit(&submission).await?;
    serde_json::to_value(&result).map_err(Into::into)
}

async fn exec_recall(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let memory_id = parse_uuid(p, "memory_id")?;
    let item = state.memory.get_by_id(memory_id).await?;

    match item {
        Some(item) => {
            // Authorisation: the memory must belong to the caller's tenant.
            if item.tenant_id != ctx.tenant_id {
                return Err(ApiError::Forbidden(format!(
                    "tenant mismatch: caller '{}' vs memory tenant '{}'",
                    ctx.tenant_id, item.tenant_id
                ))
                .into());
            }
            serde_json::to_value(&item).map_err(Into::into)
        }
        None => Ok(json!({ "memory_id": memory_id, "found": false })),
    }
}

async fn exec_search(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let mut params = SearchParams::new(&ctx.tenant_id);
    params.query_text = p.get("query_text").and_then(Value::as_str).map(Into::into);
    params.memory_types = parse_enum_vec(p, "memory_types");
    params.scopes = parse_enum_vec(p, "scopes");
    params.subject_id = p.get("subject_id").and_then(Value::as_str).map(Into::into);
    params.min_confidence = p
        .get("min_confidence")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    params.limit = p.get("limit").and_then(Value::as_i64).unwrap_or(50);
    params.offset = p.get("offset").and_then(Value::as_i64).unwrap_or(0);
    params.embedding = p.get("embedding").and_then(Value::as_array).map(|a| {
        a.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect()
    });

    let allowed_scopes: Vec<Scope> = if params.scopes.is_empty() {
        all_scopes()
    } else {
        params.scopes.clone()
    };

    let results = state.search.search(&params, &allowed_scopes).await?;
    let count = results.len();
    Ok(json!({ "results": serde_json::to_value(&results)?, "count": count }))
}

async fn exec_get_context(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let query = p
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| McpToolError::missing("query"))?;
    if query.trim().is_empty() {
        return Err(McpToolError::InvalidParams(
            "query must not be empty".into(),
        ));
    }

    let intent = parse_optional_enum(p, "intent");
    let task_id = parse_optional_uuid(p, "task_id");
    let max_latency_ms = p.get("max_latency_ms").and_then(Value::as_u64);
    let max_tokens = p
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as usize);

    let identity = ctx.to_resolved_identity();
    let assembled = state
        .assembler
        .assemble(
            &identity,
            query,
            intent,
            task_id,
            max_latency_ms,
            max_tokens,
        )
        .await?;

    Ok(assembled_context_to_json(&assembled))
}

async fn exec_update(
    state: &AppState,
    _ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let memory_id = parse_uuid(p, "memory_id")?;
    let versions = state.version.fetch_lineage(memory_id).await?;
    let version_json: Vec<Value> = versions.iter().map(memory_version_to_json).collect();
    Ok(json!({ "memory_id": memory_id, "versions": version_json }))
}

async fn exec_forget(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let mode = parse_enum(p, "mode", ForgetMethod::Soft)?;

    let (affected, message) = match mode {
        ForgetMethod::Soft => {
            let id = parse_uuid(p, "memory_id")?;
            state.forget.soft_delete(id, &ctx.user_id).await?;
            (1u64, format!("memory {id} soft-deleted"))
        }
        ForgetMethod::Hard => {
            let id = parse_uuid(p, "memory_id")?;
            state.forget.hard_delete(id, &ctx.user_id).await?;
            (1u64, format!("memory {id} hard-deleted"))
        }
        ForgetMethod::Cascade => {
            let id = parse_uuid(p, "memory_id")?;
            let n = state.forget.cascade_forget(id, &ctx.user_id).await?;
            (n, format!("cascade-forget affected {n} memories"))
        }
        ForgetMethod::UserData => {
            let uid = p
                .get("user_id")
                .and_then(Value::as_str)
                .ok_or_else(|| McpToolError::missing("user_id"))?;
            let n = state.forget.forget_user_data(uid, &ctx.tenant_id).await?;
            (n, format!("forgot {n} memories for user {uid}"))
        }
    };

    Ok(
        json!({ "mode": serde_json::to_value(mode)?, "affected_count": affected, "message": message }),
    )
}

async fn exec_share(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let query = p
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| McpToolError::missing("query"))?;
    let target_agent = p
        .get("target_agent")
        .and_then(Value::as_str)
        .ok_or_else(|| McpToolError::missing("target_agent"))?;
    let purpose = p
        .get("purpose")
        .and_then(Value::as_str)
        .unwrap_or("cross_agent_share");
    let task_id = parse_optional_uuid(p, "task_id");
    let max_tokens = p
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as usize);

    // Assemble the context that will be packaged.
    let identity = ctx.to_resolved_identity();
    let assembled = state
        .assembler
        .assemble(&identity, query, None, task_id, None, max_tokens)
        .await?;

    // Build a ContextPackage projection (§7.2) as JSON.
    let context_id = Uuid::new_v4();
    let valid_until = chrono::Utc::now() + chrono::Duration::minutes(15);

    let sources_json: Vec<Value> = assembled
        .sources_used
        .iter()
        .map(|s| serde_json::to_value(s).unwrap_or(json!("unknown")))
        .collect();

    Ok(json!({
        "context_id": context_id,
        "requester_user": ctx.user_id,
        "requester_agent": ctx.agent_id,
        "target_agent": target_agent,
        "purpose": purpose,
        "task_id": task_id,
        "compressed_context": assembled.context_text,
        "token_count": assembled.token_count,
        "sources_used": sources_json,
        "conflict_flags": assembled.conflict_flags.iter().map(conflict_flag_to_json).collect::<Vec<_>>(),
        "low_confidence_items": assembled.low_confidence_items,
        "valid_until": valid_until.to_rfc3339()
    }))
}

async fn exec_promote(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let memory_id = parse_uuid(p, "memory_id")?;
    let target_scope = parse_enum(p, "target_scope", Scope::Personal)?;
    let reason: Option<String> = p.get("reason").and_then(Value::as_str).map(Into::into);

    state
        .memory
        .promote_scope(memory_id, target_scope, &ctx.user_id, reason.as_deref())
        .await?;

    Ok(json!({
        "memory_id": memory_id,
        "scope": serde_json::to_value(target_scope)?,
        "message": format!("memory promoted to {}", target_scope.as_str())
    }))
}

async fn exec_reflect(
    state: &AppState,
    ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let content = p
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| McpToolError::missing("content"))?;
    if content.trim().is_empty() {
        return Err(McpToolError::InvalidParams(
            "content must not be empty".into(),
        ));
    }

    // Reflection defaults: agent-inferred procedure at agent scope.
    let submission = CandidateSubmission {
        tenant_id: ctx.tenant_id.clone(),
        content: content.to_string(),
        source_type: parse_enum(p, "source_type", SourceType::AgentInferred)?,
        memory_type: parse_enum(p, "memory_type", default_memory_type())?,
        scope: parse_enum(p, "scope", Scope::Agent)?,
        user_id: Some(ctx.user_id.clone()),
        agent_id: ctx.agent_id.clone(),
        subject_type: p
            .get("subject_type")
            .and_then(Value::as_str)
            .map(Into::into),
        subject_id: p.get("subject_id").and_then(Value::as_str).map(Into::into),
        entity_refs: p
            .get("entity_refs")
            .and_then(Value::as_array)
            .map(|a| a.iter().cloned().collect())
            .unwrap_or_default(),
        structured_payload: p.get("structured_payload").cloned(),
        source_reference: p
            .get("source_reference")
            .and_then(Value::as_str)
            .map(Into::into),
        confidence: p
            .get("confidence")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(0.5),
    };

    let result = state.candidate.submit(&submission).await?;
    Ok(
        json!({ "result": serde_json::to_value(&result)?, "reflection": "experience submitted as candidate memory" }),
    )
}

async fn exec_verify(
    state: &AppState,
    _ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let memory_id = parse_uuid(p, "memory_id")?;
    let verified_by = p
        .get("verified_by")
        .and_then(Value::as_str)
        .ok_or_else(|| McpToolError::missing("verified_by"))?;
    if verified_by.trim().is_empty() {
        return Err(McpToolError::InvalidParams(
            "verified_by must not be empty".into(),
        ));
    }

    state.memory.verify(memory_id, verified_by).await?;

    Ok(json!({
        "memory_id": memory_id,
        "verified_at": chrono::Utc::now().to_rfc3339(),
        "message": "memory verified"
    }))
}

async fn exec_user_get_context(
    state: &AppState,
    _ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let user_id = p
        .get("user_id")
        .and_then(Value::as_str)
        .ok_or_else(|| McpToolError::missing("user_id"))?;

    let profile = state.canonical_user.fetch_profile(user_id).await?;
    Ok(json!({ "user_id": user_id, "profile": serde_json::to_value(&profile)? }))
}

async fn exec_task_get_context(
    state: &AppState,
    _ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let task_id = parse_uuid(p, "task_id")?;
    let context = state.shared_task.fetch_task(task_id).await?;
    Ok(json!({ "task_id": task_id, "context": serde_json::to_value(&context)? }))
}

async fn exec_task_update_context(
    state: &AppState,
    _ctx: &RequestContext,
    p: &Value,
) -> Result<Value, McpToolError> {
    let task_id = parse_uuid(p, "task_id")?;
    let findings: Vec<Value> = p
        .get("add_findings")
        .and_then(Value::as_array)
        .map(|a| a.iter().cloned().collect())
        .unwrap_or_default();
    let status = p.get("status").and_then(Value::as_str).map(Into::into);

    state
        .shared_task
        .update_task(task_id, findings, status)
        .await?;

    // Re-fetch the updated context for the response.
    let context = state.shared_task.fetch_task(task_id).await?;
    Ok(json!({ "task_id": task_id, "context": serde_json::to_value(&context)? }))
}

// ═══════════════════════ Serialization Helpers ═══════════════════════

/// Manually serialize [`AssembledContext`] (does not derive `Serialize`).
fn assembled_context_to_json(ctx: &AssembledContext) -> Value {
    json!({
        "context_text": ctx.context_text,
        "token_count": ctx.token_count,
        "compressed": ctx.compressed,
        "conflict_flags": ctx.conflict_flags.iter().map(conflict_flag_to_json).collect::<Vec<_>>(),
        "low_confidence_items": ctx.low_confidence_items,
        "sources_used": ctx.sources_used.iter().map(|s| serde_json::to_value(s).unwrap_or(json!("unknown"))).collect::<Vec<_>>(),
        "degraded_sources": ctx.degraded_sources.iter().map(|s| serde_json::to_value(s).unwrap_or(json!("unknown"))).collect::<Vec<_>>()
    })
}

/// Manually serialize [`ConflictFlag`] (does not derive `Serialize`).
fn conflict_flag_to_json(f: &ConflictFlag) -> Value {
    json!({
        "field": f.field,
        "description": f.description,
        "conflicting_values": f.conflicting_values
    })
}

/// Manually serialize [`MemoryVersion`] (does not derive `Serialize`).
fn memory_version_to_json(v: &MemoryVersion) -> Value {
    json!({
        "version_id": v.version_id,
        "memory_id": v.memory_id,
        "version_number": v.version_number,
        "payload": v.payload,
        "changed_by": v.changed_by,
        "change_reason": v.change_reason,
        "created_at": v.created_at.to_rfc3339()
    })
}

// ═══════════════════════ Param Helpers ═══════════════════════

fn parse_uuid(p: &Value, key: &str) -> Result<Uuid, McpToolError> {
    p.get(key)
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or_else(|| McpToolError::missing(key))
}

fn parse_optional_uuid(p: &Value, key: &str) -> Option<Uuid> {
    p.get(key)
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
}

fn parse_enum<T: serde::de::DeserializeOwned>(
    p: &Value,
    key: &str,
    default: T,
) -> Result<T, McpToolError> {
    match p.get(key) {
        None => Ok(default),
        Some(val) => serde_json::from_value(val.clone())
            .map_err(|e| McpToolError::InvalidParams(format!("{key}: {e}"))),
    }
}

fn parse_optional_enum<T: serde::de::DeserializeOwned>(p: &Value, key: &str) -> Option<T> {
    p.get(key)
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn parse_enum_vec<T: serde::de::DeserializeOwned>(p: &Value, key: &str) -> Vec<T> {
    p.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn default_source_type() -> SourceType {
    SourceType::AgentInferred
}

fn default_memory_type() -> MemoryType {
    MemoryType::Semantic
}

fn all_scopes() -> Vec<Scope> {
    vec![
        Scope::Personal,
        Scope::Agent,
        Scope::Task,
        Scope::Team,
        Scope::Process,
        Scope::Factory,
        Scope::Enterprise,
    ]
}

// ═════════════════════════════ Tests ═════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::extractors::RequestContext;
    use oris_memory_contract::memory_types::{
        AuthorityLevel, MemoryStatus, MemoryType, PrivacyClass, SourceType,
    };
    use std::sync::Mutex;

    // ─── Mock service implementations ─────────────────────────────

    struct MockCandidateService {
        result: WriteResult,
        last: Mutex<Option<CandidateSubmission>>,
    }

    #[async_trait::async_trait]
    impl CandidateService for MockCandidateService {
        async fn submit(&self, sub: &CandidateSubmission) -> Result<WriteResult, ApiError> {
            *self.last.lock().unwrap() = Some(CandidateSubmission {
                tenant_id: sub.tenant_id.clone(),
                content: sub.content.clone(),
                source_type: sub.source_type,
                memory_type: sub.memory_type,
                scope: sub.scope,
                user_id: sub.user_id.clone(),
                agent_id: sub.agent_id.clone(),
                subject_type: sub.subject_type.clone(),
                subject_id: sub.subject_id.clone(),
                entity_refs: sub.entity_refs.clone(),
                structured_payload: sub.structured_payload.clone(),
                source_reference: sub.source_reference.clone(),
                confidence: sub.confidence,
            });
            Ok(WriteResult {
                memory_id: self.result.memory_id,
                status: self.result.status,
                importance: self.result.importance,
                is_duplicate: self.result.is_duplicate,
                safety_verdict: self.result.safety_verdict.clone(),
                outbox_event_id: self.result.outbox_event_id,
            })
        }
    }

    struct MockMemoryService {
        item: Option<MemoryItem>,
        promote_called: Mutex<bool>,
        verify_called: Mutex<bool>,
    }

    #[async_trait::async_trait]
    impl MemoryService for MockMemoryService {
        async fn get_by_id(&self, _id: Uuid) -> Result<Option<MemoryItem>, ApiError> {
            Ok(self.item.as_ref().map(|i| clone_memory_item(i)))
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

    #[async_trait::async_trait]
    impl SearchService for MockSearchService {
        async fn search(
            &self,
            _params: &SearchParams,
            _scopes: &[Scope],
        ) -> Result<Vec<RerankedResult>, ApiError> {
            Ok(self
                .results
                .iter()
                .map(|r| RerankedResult {
                    memory_id: r.memory_id,
                    content: r.content.clone(),
                    final_score: r.final_score,
                    relevance_score: r.relevance_score,
                    authority_score: r.authority_score,
                    freshness_score: r.freshness_score,
                    outcome_score: r.outcome_score,
                    conflict_penalty: r.conflict_penalty,
                    rank: r.rank,
                })
                .collect())
        }
    }

    struct MockAssembleService {
        context: AssembledContext,
    }

    #[async_trait::async_trait]
    impl AssembleService for MockAssembleService {
        async fn assemble(
            &self,
            _identity: &ResolvedIdentity,
            _query: &str,
            _intent: Option<RequestIntent>,
            _task_id: Option<Uuid>,
            _max_latency_ms: Option<u64>,
            _max_tokens: Option<usize>,
        ) -> Result<AssembledContext, ApiError> {
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

    #[async_trait::async_trait]
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

    #[async_trait::async_trait]
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

    #[async_trait::async_trait]
    impl VersionService for MockVersionService {
        async fn fetch_lineage(&self, _id: Uuid) -> Result<Vec<MemoryVersion>, ApiError> {
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

    #[async_trait::async_trait]
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

    // ─── Test fixtures ────────────────────────────────────────────

    fn test_context() -> RequestContext {
        RequestContext {
            tenant_id: "acme-corp".into(),
            user_id: "user-1".into(),
            agent_id: Some("agent-7".into()),
            purpose: "memory_read".into(),
            task_id: None,
            trace_id: "trace-1".into(),
        }
    }

    fn mock_state() -> AppState {
        let candidate = MockCandidateService {
            result: WriteResult {
                memory_id: Uuid::new_v4(),
                status: MemoryStatus::Active,
                importance: 0.8,
                is_duplicate: false,
                safety_verdict: crate::write_pipeline::SafetyVerdictSummary::Safe,
                outbox_event_id: Uuid::new_v4(),
            },
            last: Mutex::new(None),
        };

        let memory = MockMemoryService {
            item: Some(make_test_memory_item()),
            promote_called: Mutex::new(false),
            verify_called: Mutex::new(false),
        };

        let search = MockSearchService {
            results: vec![RerankedResult {
                memory_id: Uuid::new_v4(),
                content: "pump maintenance".into(),
                final_score: 0.9,
                relevance_score: 0.85,
                authority_score: 0.8,
                freshness_score: 0.7,
                outcome_score: 0.6,
                conflict_penalty: 0.0,
                rank: 1,
            }],
        };

        let assembler = MockAssembleService {
            context: AssembledContext {
                context_text: "assembled context".into(),
                token_count: 10,
                compressed: false,
                conflict_flags: vec![],
                low_confidence_items: vec![],
                sources_used: vec![ContextSource::EnterpriseMemory],
                degraded_sources: vec![],
            },
        };

        let canonical_user = MockCanonicalUserService {
            profile: Some(make_test_profile()),
        };

        let shared_task = MockSharedTaskService {
            task: Some(make_test_task()),
            update_called: Mutex::new(false),
        };

        let version = MockVersionService {
            versions: vec![MemoryVersion {
                version_id: Uuid::new_v4(),
                memory_id: Uuid::new_v4(),
                version_number: 1,
                payload: json!({}),
                changed_by: "user-1".into(),
                change_reason: None,
                created_at: chrono::Utc::now(),
            }],
        };

        let forget = MockForgetService {
            soft_called: Mutex::new(false),
            hard_called: Mutex::new(false),
            cascade_called: Mutex::new(false),
            user_data_count: 5,
        };

        AppState {
            candidate: Arc::new(candidate),
            memory: Arc::new(memory),
            search: Arc::new(search),
            assembler: Arc::new(assembler),
            canonical_user: Arc::new(canonical_user),
            shared_task: Arc::new(shared_task),
            version: Arc::new(version),
            forget: Arc::new(forget),
        }
    }

    fn make_test_memory_item() -> MemoryItem {
        use chrono::Utc;
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "acme-corp".into(),
            memory_type: MemoryType::Semantic,
            scope: Scope::Personal,
            subject_type: None,
            subject_id: None,
            entity_refs: vec![],
            content: Some("test memory".into()),
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
            acl: json!({}),
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

    fn clone_memory_item(i: &MemoryItem) -> MemoryItem {
        MemoryItem {
            memory_id: i.memory_id,
            tenant_id: i.tenant_id.clone(),
            memory_type: i.memory_type,
            scope: i.scope,
            subject_type: i.subject_type.clone(),
            subject_id: i.subject_id.clone(),
            entity_refs: i.entity_refs.clone(),
            content: i.content.clone(),
            structured_payload: i.structured_payload.clone(),
            embedding: i.embedding.clone(),
            source_type: i.source_type,
            source_reference: i.source_reference.clone(),
            evidence_refs: i.evidence_refs.clone(),
            confidence: i.confidence,
            authority_level: i.authority_level,
            importance: i.importance,
            observed_at: i.observed_at,
            valid_from: i.valid_from,
            valid_to: i.valid_to,
            privacy_class: i.privacy_class,
            acl: i.acl.clone(),
            retention_policy: i.retention_policy.clone(),
            status: i.status,
            version: i.version,
            derived_from: i.derived_from.clone(),
            created_by_user: i.created_by_user.clone(),
            created_by_agent: i.created_by_agent.clone(),
            last_verified_at: i.last_verified_at,
            created_at: i.created_at,
            updated_at: i.updated_at,
        }
    }

    fn make_test_profile() -> CanonicalUserProfile {
        CanonicalUserProfile {
            user_id: "user-1".into(),
            organization_id: "acme-corp".into(),
            factory_id: None,
            identity_links: vec![],
            role: Some("engineer".into()),
            position: None,
            language: Some("en".into()),
            timezone: Some("UTC".into()),
            preferences: json!({}),
            common_entities: vec![],
            active_projects: vec![],
            consent_scope: json!({}),
            privacy_class: PrivacyClass::Internal,
            source: SourceType::UserExplicit,
            authority_level: AuthorityLevel::L1Authoritative,
            version: 1,
            valid_from: None,
            valid_to: None,
            last_verified_at: None,
            updated_at: chrono::Utc::now(),
        }
    }

    fn make_test_task() -> SharedTaskContext {
        SharedTaskContext {
            task_id: Uuid::new_v4(),
            parent_task_id: None,
            initiator_user_id: "user-1".into(),
            organization_scope: "acme-corp".into(),
            goal: "fix the pump".into(),
            constraints: vec![],
            success_criteria: vec![],
            entities: vec![],
            business_refs: vec![],
            current_findings: vec![],
            evidence_refs: vec![],
            decisions: vec![],
            assumptions: vec![],
            completed_steps: vec![],
            pending_steps: vec![],
            current_owner_agent: None,
            participant_agents: vec![],
            artifact_refs: vec![],
            source_system_refs: vec![],
            status: "in_progress".into(),
            version: 1,
            expires_at: None,
            acl: json!({}),
            privacy_class: PrivacyClass::Internal,
            audit_ref: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    fn registry() -> McpToolRegistry {
        McpToolRegistry::new().with_state(mock_state())
    }

    async fn run_tool(name: &str, params: Value) -> McpToolResponse {
        registry()
            .execute_tool(&McpToolRequest {
                tool_name: name.into(),
                context: test_context(),
                params,
            })
            .await
            .unwrap()
    }

    // ─── Registry structure tests ─────────────────────────────────

    #[test]
    fn registry_has_13_tools() {
        let reg = McpToolRegistry::new();
        assert_eq!(reg.list_tools().len(), 13);
    }

    #[test]
    fn get_tool_finds_known() {
        let reg = McpToolRegistry::new();
        assert!(reg.get_tool("oris_memory_remember").is_some());
        assert!(reg.get_tool("oris_memory_share").is_some());
        assert!(reg.get_tool("oris_task_update_context").is_some());
    }

    #[test]
    fn get_tool_returns_none_for_unknown() {
        let reg = McpToolRegistry::new();
        assert!(reg.get_tool("nonexistent_tool").is_none());
    }

    #[test]
    fn all_tool_names_are_distinct() {
        let reg = McpToolRegistry::new();
        let names: Vec<&str> = reg.list_tools().iter().map(|t| t.name).collect();
        let unique: std::collections::HashSet<&str> = names.iter().cloned().collect();
        assert_eq!(names.len(), unique.len(), "duplicate tool names detected");
    }

    #[test]
    fn every_tool_has_input_schema() {
        let reg = McpToolRegistry::new();
        for tool in reg.list_tools() {
            assert!(
                tool.input_schema.get("type").is_some(),
                "{} missing schema type",
                tool.name
            );
            assert!(
                tool.input_schema.get("properties").is_some(),
                "{} missing schema properties",
                tool.name
            );
        }
    }

    // ─── execute_tool dispatch tests ──────────────────────────────

    #[tokio::test]
    async fn execute_unknown_tool_returns_error() {
        let reg = registry();
        let err = reg
            .execute_tool(&McpToolRequest {
                tool_name: "bogus".into(),
                context: test_context(),
                params: json!({}),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, McpToolError::UnknownTool(_)));
    }

    #[tokio::test]
    async fn execute_without_state_returns_no_state() {
        let reg = McpToolRegistry::new();
        let err = reg
            .execute_tool(&McpToolRequest {
                tool_name: "oris_memory_remember".into(),
                context: test_context(),
                params: json!({"content":"x","source_type":"agent_inferred","memory_type":"semantic","scope":"personal"}),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, McpToolError::NoState));
    }

    // ─── remember / reflect ───────────────────────────────────────

    #[tokio::test]
    async fn remember_returns_write_result() {
        let resp = run_tool(
            "oris_memory_remember",
            json!({
                "content": "Pump X-200 needs oil every 500h",
                "source_type": "user_explicit",
                "memory_type": "semantic",
                "scope": "personal",
                "confidence": 0.9
            }),
        )
        .await;
        assert!(!resp.is_error);
        assert!(resp.result["memory_id"].is_string());
        assert_eq!(resp.result["status"], "active");
        assert!((resp.result["importance"].as_f64().unwrap() - 0.8).abs() < 0.01);
    }

    #[tokio::test]
    async fn remember_rejects_empty_content() {
        let resp = run_tool(
            "oris_memory_remember",
            json!({"content":"  ","source_type":"agent_inferred","memory_type":"semantic","scope":"personal"}),
        )
        .await;
        assert!(resp.is_error);
    }

    #[tokio::test]
    async fn reflect_returns_reflection_and_result() {
        let resp = run_tool(
            "oris_memory_reflect",
            json!({"content": "Reflected on pump maintenance procedure"}),
        )
        .await;
        assert!(!resp.is_error);
        assert!(resp.result["result"]["memory_id"].is_string());
        assert!(resp.result["reflection"].is_string());
    }

    // ─── recall ───────────────────────────────────────────────────

    #[tokio::test]
    async fn recall_returns_memory_item() {
        let id = Uuid::new_v4();
        let resp = run_tool("oris_memory_recall", json!({"memory_id": id})).await;
        assert!(!resp.is_error);
        assert!(resp.result["memory_id"].is_string());
        assert_eq!(resp.result["tenant_id"], "acme-corp");
    }

    #[tokio::test]
    async fn recall_rejects_missing_memory_id() {
        let resp = run_tool("oris_memory_recall", json!({})).await;
        assert!(resp.is_error);
    }

    // ─── search ────────────────────────────────────────────────────

    #[tokio::test]
    async fn search_returns_results_and_count() {
        let resp = run_tool(
            "oris_memory_search",
            json!({"query_text": "pump maintenance"}),
        )
        .await;
        assert!(!resp.is_error);
        assert_eq!(resp.result["count"], 1);
        assert!(resp.result["results"].is_array());
    }

    // ─── get_context / share ──────────────────────────────────────

    #[tokio::test]
    async fn get_context_returns_assembled_context() {
        let resp = run_tool(
            "oris_memory_get_context",
            json!({"query": "how to fix the pump"}),
        )
        .await;
        assert!(!resp.is_error);
        assert_eq!(resp.result["context_text"], "assembled context");
        assert_eq!(resp.result["token_count"], 10);
    }

    #[tokio::test]
    async fn get_context_rejects_empty_query() {
        let resp = run_tool("oris_memory_get_context", json!({"query": ""})).await;
        assert!(resp.is_error);
    }

    #[tokio::test]
    async fn share_returns_context_package() {
        let resp = run_tool(
            "oris_memory_share",
            json!({"query": "pump maintenance", "target_agent": "agent-42"}),
        )
        .await;
        assert!(!resp.is_error);
        assert!(resp.result["context_id"].is_string());
        assert_eq!(resp.result["target_agent"], "agent-42");
        assert!(resp.result["valid_until"].is_string());
    }

    // ─── update (lineage) ─────────────────────────────────────────

    #[tokio::test]
    async fn update_returns_version_lineage() {
        let id = Uuid::new_v4();
        let resp = run_tool("oris_memory_update", json!({"memory_id": id})).await;
        assert!(!resp.is_error);
        assert_eq!(resp.result["memory_id"], id.to_string());
        assert!(resp.result["versions"].is_array());
        assert_eq!(resp.result["versions"].as_array().unwrap().len(), 1);
        assert_eq!(resp.result["versions"][0]["version_number"], 1);
    }

    // ─── forget ────────────────────────────────────────────────────

    #[tokio::test]
    async fn forget_soft_succeeds() {
        let id = Uuid::new_v4();
        let resp = run_tool(
            "oris_memory_forget",
            json!({"memory_id": id, "mode": "soft"}),
        )
        .await;
        assert!(!resp.is_error);
        assert_eq!(resp.result["mode"], "soft");
        assert_eq!(resp.result["affected_count"], 1);
    }

    #[tokio::test]
    async fn forget_user_data_succeeds() {
        let resp = run_tool(
            "oris_memory_forget",
            json!({"user_id": "user-42", "mode": "user_data"}),
        )
        .await;
        assert!(!resp.is_error);
        assert_eq!(resp.result["mode"], "user_data");
        assert_eq!(resp.result["affected_count"], 5);
    }

    #[tokio::test]
    async fn forget_cascade_succeeds() {
        let id = Uuid::new_v4();
        let resp = run_tool(
            "oris_memory_forget",
            json!({"memory_id": id, "mode": "cascade"}),
        )
        .await;
        assert!(!resp.is_error);
        assert!(resp.result["affected_count"].as_u64().unwrap() >= 1);
    }

    // ─── promote / verify ──────────────────────────────────────────

    #[tokio::test]
    async fn promote_returns_scope_and_message() {
        let id = Uuid::new_v4();
        let resp = run_tool(
            "oris_memory_promote",
            json!({"memory_id": id, "target_scope": "team", "reason": "promoted by QA"}),
        )
        .await;
        assert!(!resp.is_error);
        assert_eq!(resp.result["scope"], "team");
        assert!(resp.result["message"]
            .as_str()
            .unwrap()
            .contains("promoted"));
    }

    #[tokio::test]
    async fn verify_returns_verified_at() {
        let id = Uuid::new_v4();
        let resp = run_tool(
            "oris_memory_verify",
            json!({"memory_id": id, "verified_by": "qa-engineer-1"}),
        )
        .await;
        assert!(!resp.is_error);
        assert!(resp.result["verified_at"].is_string());
    }

    #[tokio::test]
    async fn verify_rejects_empty_verified_by() {
        let id = Uuid::new_v4();
        let resp = run_tool(
            "oris_memory_verify",
            json!({"memory_id": id, "verified_by": ""}),
        )
        .await;
        assert!(resp.is_error);
    }

    // ─── user / task context ───────────────────────────────────────

    #[tokio::test]
    async fn user_get_context_returns_profile() {
        let resp = run_tool("oris_user_get_context", json!({"user_id": "user-1"})).await;
        assert!(!resp.is_error);
        assert!(resp.result["profile"].is_object());
        assert_eq!(resp.result["profile"]["role"], "engineer");
    }

    #[tokio::test]
    async fn task_get_context_returns_context() {
        let resp = run_tool("oris_task_get_context", json!({"task_id": Uuid::new_v4()})).await;
        assert!(!resp.is_error);
        assert!(resp.result["context"].is_object());
        assert_eq!(resp.result["context"]["goal"], "fix the pump");
    }

    #[tokio::test]
    async fn task_update_context_appends_findings() {
        let resp = run_tool(
            "oris_task_update_context",
            json!({"task_id": Uuid::new_v4(), "add_findings": ["finding-1"], "status": "done"}),
        )
        .await;
        assert!(!resp.is_error);
        assert!(resp.result["context"].is_object());
    }
}
