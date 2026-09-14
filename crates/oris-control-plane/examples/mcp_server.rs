//! Oris MCP JSON-RPC Server — PostgreSQL-backed.
//!
//! Serves the 13 MCP tools (§5.3) over JSON-RPC 2.0 on stdin/stdout.
//! Each tool call is routed to the V1 API service layer, backed by
//! PostgreSQL + pgvector.
//!
//! ## Protocol
//!
//! The server reads one JSON-RPC request per line from stdin and writes
//! one response per line to stdout. Supported methods:
//!
//! - `initialize` — MCP handshake
//! - `tools/list` — return all 13 tool definitions
//! - `tools/call` — execute a tool
//!
//! ## Usage
//!
//! ```bash
//! DATABASE_URL=postgres://localhost/oris_memory cargo run --example mcp_server
//! ```
//!
//! Then send JSON-RPC messages via stdin:
//!
//! ```json
//! {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}
//! {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
//! {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"oris_memory_search","arguments":{"query":"bearing","tenant_id":"tenant-1","user_id":"user-1","purpose":"general"}}}
//! ```

use std::sync::Arc;

use oris_control_plane::api::routes::{v1_router, AppState, AssembleServiceImpl, PostgresMemoryService, PostgresSearchService};
use oris_control_plane::canonical_user::CanonicalUserManager;
use oris_control_plane::context_assembler::{
    CanonicalUserAdapter, ContextAssembler, MemoryRepoAdapter, SearchRepoAdapter, SharedTaskAdapter,
};
use oris_control_plane::context_router::ContextRouter;
use oris_control_plane::governance::forget::ForgetManager;
use oris_control_plane::governance::version::VersionManager;
use oris_control_plane::mcp_tools::McpToolRegistry;
use oris_control_plane::poison_guard::PoisonGuard;
use oris_control_plane::rerank::{RerankConfig, RerankPipeline};
use oris_control_plane::shared_task::SharedTaskManager;
use oris_control_plane::write_pipeline::WritePipeline;
use oris_memory_store::postgres::{MemoryRepo, Pool, SearchRepo};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost/oris_memory".into());

    println!("Connecting to PostgreSQL: {}", db_url);
    let pool: Pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(&db_url)
        .await?;

    // Build AppState (same as memory_server.rs).
    let poison_guard = PoisonGuard::default();
    let write_pipeline = WritePipeline::from_pool(Arc::new(pool.clone()), poison_guard);
    let memory_service = PostgresMemoryService::new(pool.clone());
    let search_service = PostgresSearchService::new(
        pool.clone(),
        RerankPipeline::new(RerankConfig::default()),
    );

    let memory_repo = MemoryRepo::new(pool.clone());
    let search_repo = SearchRepo::new(pool.clone());
    let assembler = AssembleServiceImpl::new(
        ContextRouter::new(),
        ContextAssembler::new()
            .with_memory(Arc::new(MemoryRepoAdapter::from(memory_repo)))
            .with_search(Arc::new(SearchRepoAdapter::from(search_repo)))
            .with_canonical_user(Arc::new(CanonicalUserAdapter::from(
                CanonicalUserManager::new(pool.clone()),
            )))
            .with_shared_task(Arc::new(SharedTaskAdapter::from(SharedTaskManager::new(
                pool.clone(),
            )))),
    );
    let canonical_user = CanonicalUserManager::new(pool.clone());
    let shared_task = SharedTaskManager::new(pool.clone());
    let version = VersionManager::new(pool.clone());
    let forget = ForgetManager::new(pool.clone());

    let state = AppState {
        candidate: Arc::new(write_pipeline),
        memory: Arc::new(memory_service),
        search: Arc::new(search_service),
        assembler: Arc::new(assembler),
        canonical_user: Arc::new(canonical_user),
        shared_task: Arc::new(shared_task),
        version: Arc::new(version),
        forget: Arc::new(forget),
    };

    let registry = McpToolRegistry::new().with_state(state);

    eprintln!("MCP server ready — {} tools registered", registry.list_tools().len());

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(req) => handle_request(&registry, req).await,
            Err(e) => Some(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: format!("parse error: {e}"),
                    data: None,
                }),
            }),
        };

        if let Some(resp) = response {
            let json = serde_json::to_string(&resp)?;
            stdout.write_all(json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}

async fn handle_request(registry: &McpToolRegistry, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
    if req.jsonrpc != "2.0" {
        return Some(JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "invalid JSON-RPC version".into(),
                data: None,
            }),
        });
    }

    if req.id.is_none() {
        return None; // notification — no response
    }

    let id = req.id.clone();
    let result = dispatch(registry, &req.method, &req.params).await;

    Some(match result {
        Ok(value) => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(value),
            error: None,
        },
        Err(err) => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(err),
        },
    })
}

async fn dispatch(
    registry: &McpToolRegistry,
    method: &str,
    params: &Value,
) -> Result<Value, JsonRpcError> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "oris-memory-server",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "capabilities": {
                "tools": { "listChanged": false }
            }
        })),

        "tools/list" => {
            let tools: Vec<Value> = registry
                .list_tools()
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    })
                })
                .collect();
            Ok(json!({ "tools": tools }))
        }

        "tools/call" => {
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| JsonRpcError {
                    code: -32602,
                    message: "missing 'name' in tools/call params".into(),
                    data: None,
                })?;

            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

            let request = oris_control_plane::mcp_tools::McpToolRequest {
                tool_name: tool_name.to_string(),
                params: arguments,
                context: oris_control_plane::api::extractors::RequestContext {
                    tenant_id: std::env::var("ORIS_TENANT_ID").unwrap_or_else(|_| "default".into()),
                    user_id: std::env::var("ORIS_USER_ID").unwrap_or_else(|_| "default".into()),
                    agent_id: std::env::var("ORIS_AGENT_ID").ok(),
                    purpose: std::env::var("ORIS_PURPOSE").unwrap_or_else(|_| "general".into()),
                    task_id: None,
                    trace_id: uuid::Uuid::new_v4().to_string(),
                },
            };

            match registry.execute_tool(&request).await {
                Ok(resp) => {
                    if !resp.is_error {
                        Ok(json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&resp.result).unwrap_or_default()
                            }]
                        }))
                    } else {
                        Ok(json!({
                            "isError": true,
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&resp.result).unwrap_or_default()
                            }]
                        }))
                    }
                }
                Err(e) => Err(JsonRpcError {
                    code: -32603,
                    message: format!("tool execution error: {e}"),
                    data: None,
                }),
            }
        }

        _ => Err(JsonRpcError {
            code: -32601,
            message: format!("method not found: {method}"),
            data: None,
        }),
    }
}
