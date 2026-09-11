//! PostgreSQL repository implementations.
//!
//! Uses `sqlx` with connection pooling and `pgvector` for vector similarity
//! search. The schema is defined in [`schema`] and the core CRUD operations
//! are in [`memory_repo`], [`user_repo`], and [`task_repo`].

pub mod memory_repo;
pub mod outbox;
pub mod rls;
pub mod schema;
pub mod search;
pub mod task_repo;
pub mod user_repo;

pub use memory_repo::{MemoryRepo, MemoryRepoError};
pub use outbox::{OutboxError, OutboxRepo};
pub use rls::RlsManager;
pub use schema::PostgresSchema;
pub use search::{SearchError, SearchRepo};
pub use task_repo::{TaskRepo, TaskRepoError};
pub use user_repo::{UserRepo, UserRepoError};

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

/// Connection pool for PostgreSQL.
pub type Pool = PgPool;

/// Create a connection pool from a database URL.
pub async fn connect(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
}

/// Create a pool with custom max connections.
pub async fn connect_with_max(
    database_url: &str,
    max_connections: u32,
) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .connect(database_url)
        .await
}
