//! Oris Memory Server — PostgreSQL-backed V1 API server.
//!
//! Serves the full V1 memory REST API (§10.2) backed by PostgreSQL + pgvector.
//! Start PostgreSQL, create the `oris_memory` database, apply the schema,
//! then run:
//!
//! ```bash
//! DATABASE_URL=postgres://localhost/oris_memory cargo run --example memory_server
//! ```

use std::sync::Arc;

use axum::{routing::get, Router};
use oris_control_plane::api::routes::{v1_router, AppState};
use oris_control_plane::api::routes::{
    AssembleServiceImpl, PostgresMemoryService, PostgresSearchService,
};
use oris_control_plane::canonical_user::CanonicalUserManager;
use oris_control_plane::context_assembler::ContextAssembler;
use oris_control_plane::context_router::ContextRouter;
use oris_control_plane::governance::forget::ForgetManager;
use oris_control_plane::governance::version::VersionManager;
use oris_control_plane::poison_guard::PoisonGuard;
use oris_control_plane::rerank::{RerankConfig, RerankPipeline};
use oris_control_plane::shared_task::SharedTaskManager;
use oris_control_plane::write_pipeline::WritePipeline;
use oris_memory_store::postgres::Pool;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

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

    // Apply schema (idempotent).
    // Schema already applied via psql; uncomment for fresh DB:
    // oris_memory_store::postgres::schema::PostgresSchema::apply(&pool).await?;
    println!("Schema applied.");

    // Build all PostgreSQL-backed services.
    let poison_guard = PoisonGuard::default();
    let write_pipeline = WritePipeline::from_pool(Arc::new(pool.clone()), poison_guard);

    let memory_service = PostgresMemoryService::new(pool.clone());
    let search_service = PostgresSearchService::new(
        pool.clone(),
        RerankPipeline::new(RerankConfig::default()),
    );
    let assembler = AssembleServiceImpl::new(
        ContextRouter::new(),
        ContextAssembler::new(),
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

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = v1_router()
        .with_state(state)
        .route("/health", get(health))
        .route("/", get(root))
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    let addr = std::env::var("ORIS_BIND_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("\n🚀 Oris Memory Server running at http://{}", addr);
    println!("\nV1 API endpoints:");
    println!("  POST   /v1/memories/candidates      — submit candidate memory");
    println!("  GET    /v1/memories/{{id}}             — get memory by ID");
    println!("  POST   /v1/memories/search          — hybrid search");
    println!("  POST   /v1/context/assemble          — assemble context");
    println!("  GET    /v1/users/{{id}}/canonical-context — user canonical context");
    println!("  GET    /v1/tasks/{{id}}/context         — task context");
    println!("  PATCH  /v1/tasks/{{id}}/context         — update task context");
    println!("  POST   /v1/memories/{{id}}/promote      — promote scope");
    println!("  POST   /v1/memories/{{id}}/verify       — verify memory");
    println!("  POST   /v1/memories/forget           — forget memory");
    println!("  GET    /v1/memories/{{id}}/lineage       — lineage trace");
    println!("  GET    /health                      — health check");
    println!("\nRequired headers: x-tenant-id, x-user-id, x-purpose");
    println!("Optional headers: x-agent-id, x-task-id, x-trace-id\n");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn root() -> &'static str {
    "Oris Enterprise Context & Memory Service — see /health and /v1/* endpoints"
}
