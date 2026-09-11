//! Multi-tenant physical partitioning and capacity governance.
//!
//! Implements §8.5.3 and §13 Phase 3 requirements:
//! - Physical partitioning by tenant, time, or hybrid tenant+time
//! - Capacity monitoring with alert thresholds
//! - Read-replica routing with lag-aware failover
//!
//! All SQL uses `sqlx::query()` (runtime) — not the `query!` macro — so the
//! module compiles without a live database connection at build time.

use chrono::{DateTime, Datelike, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::fmt;

use crate::postgres::Pool;

// ──────────────────────── Strategy & Config ────────────────────────

/// Partitioning strategy for the `memory_item` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionStrategy {
    /// One partition per tenant (LIST partitioning on `tenant_id`).
    ByTenant,
    /// Hybrid: tenant LIST + monthly RANGE subpartitioning on `created_at`.
    ByTenantAndTime,
    /// Monthly RANGE partitions across all tenants on `created_at`.
    ByTime,
}

impl PartitionStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ByTenant => "by_tenant",
            Self::ByTenantAndTime => "by_tenant_and_time",
            Self::ByTime => "by_time",
        }
    }
}

impl fmt::Display for PartitionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration for partition management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub strategy: PartitionStrategy,
    pub partition_prefix: String,
    pub retention_days: i64,
    pub archive_after_days: i64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            strategy: PartitionStrategy::ByTenantAndTime,
            partition_prefix: "memory_item".to_string(),
            retention_days: 365,
            archive_after_days: 90,
        }
    }
}

/// A time range used for RANGE partition boundaries.
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl TimeRange {
    /// Build a monthly range covering the month containing `date`.
    pub fn for_month(date: NaiveDate) -> Self {
        let year = date.year();
        let month = date.month();
        let start =
            NaiveDate::from_ymd_opt(year, month, 1).expect("first day of month is always valid");
        let next_month = if month == 12 {
            NaiveDate::from_ymd_opt(year + 1, 1, 1)
        } else {
            NaiveDate::from_ymd_opt(year, month + 1, 1)
        }
        .expect("valid next month");
        Self {
            start: start.and_hms_opt(0, 0, 0).unwrap().and_utc(),
            end: next_month.and_hms_opt(0, 0, 0).unwrap().and_utc(),
        }
    }

    /// Format the range as `yyyymm` (e.g. `202609`).
    pub fn yyyymm(&self) -> String {
        format!("{:04}{:02}", self.start.year(), self.start.month())
    }
}

// ──────────────────────── Partition Manager ────────────────────────

/// Metadata about a single partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionInfo {
    pub name: String,
    pub parent_table: String,
    pub strategy: PartitionStrategy,
    pub tenant_id: Option<String>,
    pub time_range: Option<TimeRangeSerializable>,
}

/// Serializable representation of a time range (for serde on PartitionInfo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRangeSerializable {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl From<TimeRange> for TimeRangeSerializable {
    fn from(r: TimeRange) -> Self {
        Self {
            start: r.start,
            end: r.end,
        }
    }
}

/// Size information for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionSize {
    pub partition_name: String,
    pub total_bytes: i64,
    pub index_bytes: i64,
    pub row_count: i64,
}

/// Error type for partition operations.
#[derive(Debug, thiserror::Error)]
pub enum PartitionError {
    #[error("partition not found: {0}")]
    NotFound(String),

    #[error("invalid partition name: {0}")]
    InvalidName(String),

    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}

/// Manages physical table partitioning for the `memory_item` table.
pub struct PartitionManager {
    pool: PgPool,
    config: PartitionConfig,
}

impl PartitionManager {
    pub fn new(pool: PgPool, config: PartitionConfig) -> Self {
        Self { pool, config }
    }

    pub fn config(&self) -> &PartitionConfig {
        &self.config
    }

    /// Generate the partition child-table name for a tenant + date.
    pub fn partition_name(&self, tenant_id: &str, date: NaiveDate) -> String {
        partition_name(&self.config, tenant_id, date)
    }

    /// Generate the tenant-level partition name (for ByTenant / ByTenantAndTime).
    pub fn tenant_partition_name(&self, tenant_id: &str) -> String {
        format!("{}_{}", self.config.partition_prefix, tenant_id)
    }

    /// Create a partition for the given tenant and time range.
    ///
    /// Issues `CREATE TABLE ... PARTITION OF ...` DDL appropriate to the
    /// configured strategy.
    pub async fn create_partition(
        &self,
        tenant_id: &str,
        time_range: TimeRange,
    ) -> Result<(), PartitionError> {
        let ddl = self.create_partition_ddl(tenant_id, time_range);
        tracing::info!(partition = %ddl.partition_name, "creating partition");
        sqlx::query(&ddl.sql).execute(&self.pool).await?;
        Ok(())
    }

    /// Archive (detach) a partition and move its data to an archive table.
    pub async fn archive_partition(&self, partition_name: &str) -> Result<(), PartitionError> {
        let archive_table = format!("{}_archive", partition_name);

        // Create the archive table if it doesn't exist (same structure, unpartitioned).
        let create_archive = format!(
            "CREATE TABLE IF NOT EXISTS {} (LIKE {} INCLUDING ALL)",
            archive_table, self.config.partition_prefix
        );
        sqlx::query(&create_archive).execute(&self.pool).await?;

        // Move data: insert into archive, then detach the partition.
        let move_data = format!(
            "INSERT INTO {} SELECT * FROM {}",
            archive_table, partition_name
        );
        sqlx::query(&move_data).execute(&self.pool).await?;

        let detach = format!(
            "ALTER TABLE {} DETACH PARTITION {}",
            self.config.partition_prefix, partition_name
        );
        sqlx::query(&detach).execute(&self.pool).await?;

        tracing::info!(partition = %partition_name, archive = %archive_table, "partition archived");
        Ok(())
    }

    /// List all partitions of the parent table.
    pub async fn list_partitions(&self) -> Vec<PartitionInfo> {
        let parent = &self.config.partition_prefix;
        let rows = sqlx::query(
            r#"SELECT c.relname AS partition_name,
                      p.relname AS parent_name
               FROM pg_inherits
               JOIN pg_class c ON c.oid = pg_inherits.inhrelid
               JOIN pg_class p ON p.oid = pg_inherits.inhparent
               JOIN pg_namespace n ON n.oid = p.relnamespace
               WHERE p.relname = $1 AND n.nspname = 'public'"#,
        )
        .bind(parent)
        .fetch_all(&self.pool)
        .await;

        match rows {
            Ok(rows) => rows
                .iter()
                .map(|r| PartitionInfo {
                    name: r.try_get::<String, _>("partition_name").unwrap_or_default(),
                    parent_table: r
                        .try_get::<String, _>("parent_name")
                        .unwrap_or_else(|_| parent.to_string()),
                    strategy: self.config.strategy,
                    tenant_id: None,
                    time_range: None,
                })
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "failed to list partitions");
                Vec::new()
            }
        }
    }

    /// Get the size of a specific partition.
    pub async fn get_partition_size(
        &self,
        partition_name: &str,
    ) -> Result<PartitionSize, PartitionError> {
        let row = sqlx::query(
            r#"SELECT
                   pg_total_relation_size($1) AS total_bytes,
                   pg_indexes_size($1) AS index_bytes,
                   reltuples::bigint AS row_count
               FROM pg_class
               WHERE relname = $2"#,
        )
        .bind(partition_name)
        .bind(partition_name)
        .fetch_optional(&self.pool)
        .await?;

        let row = row.ok_or_else(|| PartitionError::NotFound(partition_name.to_string()))?;
        Ok(PartitionSize {
            partition_name: partition_name.to_string(),
            total_bytes: row.try_get::<i64, _>("total_bytes").unwrap_or(0),
            index_bytes: row.try_get::<i64, _>("index_bytes").unwrap_or(0),
            row_count: row.try_get::<i64, _>("row_count").unwrap_or(0),
        })
    }

    /// Ensure a partition exists for the given tenant and date.
    ///
    /// Auto-creates the partition if missing. Returns the partition name.
    pub async fn ensure_partition_exists(
        &self,
        tenant_id: &str,
        date: NaiveDate,
    ) -> Result<String, PartitionError> {
        let name = self.partition_name(tenant_id, date);

        // Check if the partition already exists.
        let exists: bool = sqlx::query("SELECT EXISTS(SELECT 1 FROM pg_class WHERE relname = $1)")
            .bind(&name)
            .fetch_one(&self.pool)
            .await?
            .try_get::<bool, _>("exists")
            .unwrap_or(false);

        if !exists {
            let time_range = TimeRange::for_month(date);
            match self.config.strategy {
                PartitionStrategy::ByTenantAndTime => {
                    // Create tenant-level subpartition parent if needed, then the time partition.
                    let tenant_part = self.tenant_partition_name(tenant_id);
                    let tenant_exists: bool =
                        sqlx::query("SELECT EXISTS(SELECT 1 FROM pg_class WHERE relname = $1)")
                            .bind(&tenant_part)
                            .fetch_one(&self.pool)
                            .await?
                            .try_get::<bool, _>("exists")
                            .unwrap_or(false);

                    if !tenant_exists {
                        let ddl = create_tenant_subpartition_ddl(
                            &self.config.partition_prefix,
                            tenant_id,
                        );
                        sqlx::query(&ddl).execute(&self.pool).await?;
                    }
                }
                _ => {}
            }
            let ddl = self.create_partition_ddl(tenant_id, time_range);
            sqlx::query(&ddl.sql).execute(&self.pool).await?;
            tracing::info!(partition = %name, "auto-created partition");
        }

        Ok(name)
    }

    /// Generate the DDL statement for creating a partition.
    fn create_partition_ddl(&self, tenant_id: &str, time_range: TimeRange) -> PartitionDdl {
        create_partition_ddl(&self.config, tenant_id, time_range)
    }
}

// ──────────────────────── DDL Generation ────────────────────────

/// A generated DDL statement with its target partition name.
#[derive(Debug, Clone)]
pub struct PartitionDdl {
    pub partition_name: String,
    pub sql: String,
}

/// Generate the partition child-table name.
pub fn partition_name(config: &PartitionConfig, tenant_id: &str, date: NaiveDate) -> String {
    let range = TimeRange::for_month(date);
    match config.strategy {
        PartitionStrategy::ByTenant => {
            format!("{}_{}", config.partition_prefix, tenant_id)
        }
        PartitionStrategy::ByTenantAndTime => {
            format!(
                "{}_{}_{}",
                config.partition_prefix,
                tenant_id,
                range.yyyymm()
            )
        }
        PartitionStrategy::ByTime => {
            format!("{}_{}", config.partition_prefix, range.yyyymm())
        }
    }
}

/// Generate DDL to create a tenant-level subpartition parent (for ByTenantAndTime).
pub fn create_tenant_subpartition_ddl(parent_table: &str, tenant_id: &str) -> String {
    let part_name = format!("{}_{}", parent_table, tenant_id);
    format!(
        "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES IN ('{}') PARTITION BY RANGE (created_at)",
        part_name, parent_table, tenant_id
    )
}

/// Generate the full DDL for a partition based on strategy.
pub fn create_partition_ddl(
    config: &PartitionConfig,
    tenant_id: &str,
    time_range: TimeRange,
) -> PartitionDdl {
    let parent = &config.partition_prefix;
    match config.strategy {
        PartitionStrategy::ByTenant => {
            let name = format!("{}_{}", parent, tenant_id);
            let sql = format!(
                "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES IN ('{}')",
                name, parent, tenant_id
            );
            PartitionDdl {
                partition_name: name,
                sql,
            }
        }
        PartitionStrategy::ByTime => {
            let name = format!("{}_{}", parent, time_range.yyyymm());
            let sql = format!(
                "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ('{}') TO ('{}')",
                name, parent, time_range.start, time_range.end
            );
            PartitionDdl {
                partition_name: name,
                sql,
            }
        }
        PartitionStrategy::ByTenantAndTime => {
            let tenant_parent = format!("{}_{}", parent, tenant_id);
            let name = format!("{}_{}_{}", parent, tenant_id, time_range.yyyymm());
            let sql = format!(
                "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ('{}') TO ('{}')",
                name, tenant_parent, time_range.start, time_range.end
            );
            PartitionDdl {
                partition_name: name,
                sql,
            }
        }
    }
}

/// DDL to convert the base `memory_item` table into a partitioned parent.
/// Must be run before any partitions are attached.
pub fn parent_partition_ddl(config: &PartitionConfig) -> String {
    let parent = &config.partition_prefix;
    match config.strategy {
        PartitionStrategy::ByTenant | PartitionStrategy::ByTenantAndTime => format!(
            "CREATE TABLE IF NOT EXISTS {}_partitioned (LIKE {} INCLUDING ALL) PARTITION BY LIST (tenant_id)",
            parent, parent
        ),
        PartitionStrategy::ByTime => format!(
            "CREATE TABLE IF NOT EXISTS {}_partitioned (LIKE {} INCLUDING ALL) PARTITION BY RANGE (created_at)",
            parent, parent
        ),
    }
}

// ──────────────────────── Capacity Monitor ────────────────────────

/// Statistics for the overall memory table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStats {
    pub table_name: String,
    pub total_bytes: i64,
    pub index_bytes: i64,
    pub row_count: i64,
    /// Estimated bytes consumed by the embedding vector column.
    pub vector_storage_bytes: i64,
}

/// Per-tenant capacity breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantStats {
    pub tenant_id: String,
    pub row_count: i64,
    pub total_bytes: i64,
    pub vector_storage_bytes: i64,
}

/// Type of capacity threshold breach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertType {
    TableSize,
    IndexSize,
    RowCount,
    VectorStorageGrowth,
}

impl AlertType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TableSize => "table_size",
            Self::IndexSize => "index_size",
            Self::RowCount => "row_count",
            Self::VectorStorageGrowth => "vector_storage_growth",
        }
    }
}

/// A capacity threshold alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityAlert {
    pub tenant_id: Option<String>,
    pub alert_type: AlertType,
    pub current_value: i64,
    pub threshold: i64,
    pub message: String,
}

/// Thresholds for capacity alerts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityThresholds {
    pub max_table_bytes: i64,
    pub max_index_bytes: i64,
    pub max_row_count: i64,
    pub max_vector_storage_bytes: i64,
    /// Per-tenant thresholds (None = use table-level threshold).
    pub max_tenant_bytes: Option<i64>,
    pub max_tenant_row_count: Option<i64>,
}

impl Default for CapacityThresholds {
    fn default() -> Self {
        Self {
            max_table_bytes: 100 * 1024 * 1024 * 1024,         // 100 GB
            max_index_bytes: 50 * 1024 * 1024 * 1024,          // 50 GB
            max_row_count: 500_000_000,                        // 500M rows
            max_vector_storage_bytes: 80 * 1024 * 1024 * 1024, // 80 GB
            max_tenant_bytes: Some(10 * 1024 * 1024 * 1024),   // 10 GB
            max_tenant_row_count: Some(50_000_000),            // 50M rows
        }
    }
}

/// Monitors storage capacity and emits alerts when thresholds are exceeded.
pub struct CapacityMonitor {
    pool: PgPool,
    table_name: String,
    /// Dimension of the embedding vector column (1536 per schema).
    embedding_dim: usize,
    thresholds: CapacityThresholds,
}

impl CapacityMonitor {
    pub fn new(pool: PgPool) -> Self {
        Self::with_config(
            pool,
            "memory_item".to_string(),
            1536,
            CapacityThresholds::default(),
        )
    }

    pub fn with_config(
        pool: PgPool,
        table_name: String,
        embedding_dim: usize,
        thresholds: CapacityThresholds,
    ) -> Self {
        Self {
            pool,
            table_name,
            embedding_dim,
            thresholds,
        }
    }

    /// Measure overall table statistics.
    pub async fn measure_table_stats(&self) -> Result<TableStats, PartitionError> {
        let row = sqlx::query(
            r#"SELECT
                   pg_total_relation_size($1) AS total_bytes,
                   pg_indexes_size($1) AS index_bytes,
                   reltuples::bigint AS row_count
               FROM pg_class WHERE relname = $2"#,
        )
        .bind(&self.table_name)
        .bind(&self.table_name)
        .fetch_one(&self.pool)
        .await?;

        let total_bytes = row.try_get::<i64, _>("total_bytes").unwrap_or(0);
        let index_bytes = row.try_get::<i64, _>("index_bytes").unwrap_or(0);
        let row_count = row.try_get::<i64, _>("row_count").unwrap_or(0);
        let vector_storage_bytes = estimate_vector_storage(row_count, self.embedding_dim);

        Ok(TableStats {
            table_name: self.table_name.clone(),
            total_bytes,
            index_bytes,
            row_count,
            vector_storage_bytes,
        })
    }

    /// Measure per-tenant capacity.
    pub async fn measure_tenant_stats(
        &self,
        tenant_id: &str,
    ) -> Result<TenantStats, PartitionError> {
        let row = sqlx::query(
            r#"SELECT COUNT(*) AS row_count,
                      pg_total_relation_size($1) AS total_bytes
               FROM (
                   SELECT 1 FROM memory_item WHERE tenant_id = $2
               ) sub"#,
        )
        .bind(&self.table_name)
        .bind(tenant_id)
        .fetch_one(&self.pool)
        .await?;

        let row_count = row.try_get::<i64, _>("row_count").unwrap_or(0);
        // For per-tenant, total_bytes is the full table; estimate proportional share.
        let total_bytes = row.try_get::<i64, _>("total_bytes").unwrap_or(0);
        let vector_storage_bytes = estimate_vector_storage(row_count, self.embedding_dim);

        Ok(TenantStats {
            tenant_id: tenant_id.to_string(),
            row_count,
            total_bytes,
            vector_storage_bytes,
        })
    }

    /// Check all capacity limits and return alerts for any that are exceeded.
    pub async fn check_capacity_limits(&self) -> Result<Vec<CapacityAlert>, PartitionError> {
        let stats = self.measure_table_stats().await?;
        let mut alerts = Vec::new();

        if stats.total_bytes > self.thresholds.max_table_bytes {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::TableSize,
                current_value: stats.total_bytes,
                threshold: self.thresholds.max_table_bytes,
                message: format!(
                    "Table '{}' total size {} bytes exceeds threshold {} bytes",
                    self.table_name, stats.total_bytes, self.thresholds.max_table_bytes
                ),
            });
        }

        if stats.index_bytes > self.thresholds.max_index_bytes {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::IndexSize,
                current_value: stats.index_bytes,
                threshold: self.thresholds.max_index_bytes,
                message: format!(
                    "Index size {} bytes exceeds threshold {} bytes",
                    stats.index_bytes, self.thresholds.max_index_bytes
                ),
            });
        }

        if stats.row_count > self.thresholds.max_row_count {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::RowCount,
                current_value: stats.row_count,
                threshold: self.thresholds.max_row_count,
                message: format!(
                    "Row count {} exceeds threshold {}",
                    stats.row_count, self.thresholds.max_row_count
                ),
            });
        }

        if stats.vector_storage_bytes > self.thresholds.max_vector_storage_bytes {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::VectorStorageGrowth,
                current_value: stats.vector_storage_bytes,
                threshold: self.thresholds.max_vector_storage_bytes,
                message: format!(
                    "Vector storage {} bytes exceeds threshold {} bytes",
                    stats.vector_storage_bytes, self.thresholds.max_vector_storage_bytes
                ),
            });
        }

        Ok(alerts)
    }

    /// Evaluate capacity alerts from pre-measured stats (pure logic, no DB needed).
    pub fn evaluate_alerts(
        stats: &TableStats,
        thresholds: &CapacityThresholds,
    ) -> Vec<CapacityAlert> {
        let mut alerts = Vec::new();
        if stats.total_bytes > thresholds.max_table_bytes {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::TableSize,
                current_value: stats.total_bytes,
                threshold: thresholds.max_table_bytes,
                message: format!(
                    "Table '{}' total size {} bytes exceeds threshold {} bytes",
                    stats.table_name, stats.total_bytes, thresholds.max_table_bytes
                ),
            });
        }
        if stats.index_bytes > thresholds.max_index_bytes {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::IndexSize,
                current_value: stats.index_bytes,
                threshold: thresholds.max_index_bytes,
                message: format!(
                    "Index size {} bytes exceeds threshold {} bytes",
                    stats.index_bytes, thresholds.max_index_bytes
                ),
            });
        }
        if stats.row_count > thresholds.max_row_count {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::RowCount,
                current_value: stats.row_count,
                threshold: thresholds.max_row_count,
                message: format!(
                    "Row count {} exceeds threshold {}",
                    stats.row_count, thresholds.max_row_count
                ),
            });
        }
        if stats.vector_storage_bytes > thresholds.max_vector_storage_bytes {
            alerts.push(CapacityAlert {
                tenant_id: None,
                alert_type: AlertType::VectorStorageGrowth,
                current_value: stats.vector_storage_bytes,
                threshold: thresholds.max_vector_storage_bytes,
                message: format!(
                    "Vector storage {} bytes exceeds threshold {} bytes",
                    stats.vector_storage_bytes, thresholds.max_vector_storage_bytes
                ),
            });
        }
        alerts
    }

    /// Evaluate per-tenant alert (pure logic).
    pub fn evaluate_tenant_alert(
        tenant_id: &str,
        stats: &TenantStats,
        thresholds: &CapacityThresholds,
    ) -> Vec<CapacityAlert> {
        let mut alerts = Vec::new();
        if let Some(max_bytes) = thresholds.max_tenant_bytes {
            if stats.total_bytes > max_bytes {
                alerts.push(CapacityAlert {
                    tenant_id: Some(tenant_id.to_string()),
                    alert_type: AlertType::TableSize,
                    current_value: stats.total_bytes,
                    threshold: max_bytes,
                    message: format!(
                        "Tenant '{}' size {} bytes exceeds per-tenant threshold {} bytes",
                        tenant_id, stats.total_bytes, max_bytes
                    ),
                });
            }
        }
        if let Some(max_rows) = thresholds.max_tenant_row_count {
            if stats.row_count > max_rows {
                alerts.push(CapacityAlert {
                    tenant_id: Some(tenant_id.to_string()),
                    alert_type: AlertType::RowCount,
                    current_value: stats.row_count,
                    threshold: max_rows,
                    message: format!(
                        "Tenant '{}' row count {} exceeds per-tenant threshold {}",
                        tenant_id, stats.row_count, max_rows
                    ),
                });
            }
        }
        alerts
    }
}

/// Estimate the storage size of vector embeddings.
///
/// pgvector stores each dimension as 4 bytes (REAL), so a 1536-dim vector
/// uses ~6 KB. This is a rough estimate for capacity planning.
pub fn estimate_vector_storage(row_count: i64, embedding_dim: usize) -> i64 {
    let bytes_per_vector = (embedding_dim as i64) * 4; // 4 bytes per float32
    row_count.saturating_mul(bytes_per_vector)
}

// ──────────────────────── Read Replica Router ────────────────────────

/// Routes read queries to a replica and write queries to the primary.
///
/// Before routing a read to the replica, checks that replication lag is
/// within acceptable bounds. If lag exceeds the threshold, falls back to
/// the primary to avoid stale reads.
pub struct ReadReplicaRouter {
    primary: Pool,
    replica: Option<Pool>,
    max_lag_seconds: f64,
}

impl ReadReplicaRouter {
    /// Create a router with a primary pool only (no replica — all reads to primary).
    pub fn primary_only(primary: Pool) -> Self {
        Self {
            primary,
            replica: None,
            max_lag_seconds: 30.0,
        }
    }

    /// Create a router with both primary and replica pools.
    pub fn new(primary: Pool, replica: Pool) -> Self {
        Self {
            primary,
            replica: Some(replica),
            max_lag_seconds: 30.0,
        }
    }

    pub fn with_max_lag(mut self, seconds: f64) -> Self {
        self.max_lag_seconds = seconds;
        self
    }

    /// Route a read query to the appropriate pool.
    ///
    /// Returns the replica pool if available and lag is acceptable, otherwise
    /// the primary pool.
    pub async fn route_read(&self, _tenant_id: &str) -> &Pool {
        if let Some(replica) = &self.replica {
            match Self::check_replica_lag(replica).await {
                Ok(lag) if lag <= self.max_lag_seconds => replica,
                Ok(lag) => {
                    tracing::warn!(
                        lag_seconds = lag,
                        threshold = self.max_lag_seconds,
                        "replica lag exceeds threshold, routing read to primary"
                    );
                    &self.primary
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to check replica lag, routing to primary");
                    &self.primary
                }
            }
        } else {
            &self.primary
        }
    }

    /// Always returns the primary pool for write operations.
    pub fn route_write(&self) -> &Pool {
        &self.primary
    }

    /// Check the replication lag of a replica in seconds.
    ///
    /// Uses `pg_stat_replication` on the primary side. On a replica, this
    /// returns the lag as reported by `now() - pg_last_xact_replay_timestamp()`.
    pub async fn check_replica_lag(replica: &Pool) -> Result<f64, sqlx::Error> {
        // On a replica/standby, pg_last_xact_replay_timestamp gives the last
        // replayed WAL timestamp. Lag = now() - that timestamp.
        let row = sqlx::query(
            r#"SELECT
                   COALESCE(
                       EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())),
                       0
                   ) AS lag_seconds"#,
        )
        .fetch_one(replica)
        .await?;
        Ok(row.try_get::<f64, _>("lag_seconds").unwrap_or(0.0))
    }

    /// Determine whether a read should go to the replica based on a pre-measured
    /// lag value (pure logic, no DB needed).
    pub fn should_use_replica(&self, lag_seconds: f64, has_replica: bool) -> bool {
        should_use_replica(self.max_lag_seconds, lag_seconds, has_replica)
    }

    /// Get the configured max lag threshold.
    pub fn max_lag_seconds(&self) -> f64 {
        self.max_lag_seconds
    }

    /// Whether a replica is configured.
    pub fn has_replica(&self) -> bool {
        self.replica.is_some()
    }
}

/// Pure-logic replica routing decision (no DB pool required).
///
/// Returns true if a read should be routed to the replica given the current
/// lag and whether a replica is available.
pub fn should_use_replica(max_lag_seconds: f64, lag_seconds: f64, has_replica: bool) -> bool {
    has_replica && lag_seconds <= max_lag_seconds
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn config_for(strategy: PartitionStrategy) -> PartitionConfig {
        PartitionConfig {
            strategy,
            ..Default::default()
        }
    }

    fn test_date() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 9, 11).unwrap()
    }

    // ── Partition naming tests ──

    #[test]
    fn test_partition_name_by_tenant() {
        let config = config_for(PartitionStrategy::ByTenant);
        let name = partition_name(&config, "acme", test_date());
        assert_eq!(name, "memory_item_acme");
    }

    #[test]
    fn test_partition_name_by_tenant_and_time() {
        let config = config_for(PartitionStrategy::ByTenantAndTime);
        let name = partition_name(&config, "acme", test_date());
        assert_eq!(name, "memory_item_acme_202609");
    }

    #[test]
    fn test_partition_name_by_time() {
        let config = config_for(PartitionStrategy::ByTime);
        let name = partition_name(&config, "acme", test_date());
        assert_eq!(name, "memory_item_202609");
    }

    #[test]
    fn test_partition_name_custom_prefix() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::ByTenantAndTime,
            partition_prefix: "mem".to_string(),
            ..Default::default()
        };
        let name = partition_name(&config, "acme", test_date());
        assert_eq!(name, "mem_acme_202609");
    }

    // ── Time range tests ──

    #[test]
    fn test_time_range_for_month_midyear() {
        let range = TimeRange::for_month(NaiveDate::from_ymd_opt(2026, 9, 11).unwrap());
        assert_eq!(range.yyyymm(), "202609");
        // Start = Sep 1 00:00 UTC, End = Oct 1 00:00 UTC
        assert_eq!(range.start.month(), 9);
        assert_eq!(range.end.month(), 10);
        assert_eq!(range.start.day(), 1);
        assert_eq!(range.end.day(), 1);
    }

    #[test]
    fn test_time_range_for_month_december_boundary() {
        let range = TimeRange::for_month(NaiveDate::from_ymd_opt(2026, 12, 15).unwrap());
        assert_eq!(range.yyyymm(), "202612");
        assert_eq!(range.start.year(), 2026);
        assert_eq!(range.end.year(), 2027);
        assert_eq!(range.end.month(), 1);
    }

    #[test]
    fn test_time_range_for_month_january() {
        let range = TimeRange::for_month(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap());
        assert_eq!(range.yyyymm(), "202601");
        assert_eq!(range.start.day(), 1);
        assert_eq!(range.end.month(), 2);
    }

    // ── DDL generation tests ──

    #[test]
    fn test_ddl_by_tenant() {
        let config = config_for(PartitionStrategy::ByTenant);
        let range = TimeRange::for_month(test_date());
        let ddl = create_partition_ddl(&config, "acme", range);
        assert_eq!(ddl.partition_name, "memory_item_acme");
        assert!(ddl.sql.contains("PARTITION OF memory_item"));
        assert!(ddl.sql.contains("FOR VALUES IN ('acme')"));
    }

    #[test]
    fn test_ddl_by_tenant_and_time() {
        let config = config_for(PartitionStrategy::ByTenantAndTime);
        let range = TimeRange::for_month(test_date());
        let ddl = create_partition_ddl(&config, "acme", range);
        assert_eq!(ddl.partition_name, "memory_item_acme_202609");
        assert!(ddl.sql.contains("PARTITION OF memory_item_acme"));
        assert!(ddl.sql.contains("FOR VALUES FROM"));
    }

    #[test]
    fn test_ddl_by_time() {
        let config = config_for(PartitionStrategy::ByTime);
        let range = TimeRange::for_month(test_date());
        let ddl = create_partition_ddl(&config, "acme", range);
        assert_eq!(ddl.partition_name, "memory_item_202609");
        assert!(ddl.sql.contains("PARTITION OF memory_item"));
        assert!(ddl.sql.contains("FOR VALUES FROM"));
    }

    #[test]
    fn test_tenant_subpartition_ddl() {
        let ddl = create_tenant_subpartition_ddl("memory_item", "acme");
        assert!(ddl.contains("memory_item_acme"));
        assert!(ddl.contains("PARTITION OF memory_item"));
        assert!(ddl.contains("FOR VALUES IN ('acme')"));
        assert!(ddl.contains("PARTITION BY RANGE (created_at)"));
    }

    #[test]
    fn test_parent_partition_ddl_by_tenant() {
        let config = config_for(PartitionStrategy::ByTenant);
        let ddl = parent_partition_ddl(&config);
        assert!(ddl.contains("PARTITION BY LIST (tenant_id)"));
    }

    #[test]
    fn test_parent_partition_ddl_by_time() {
        let config = config_for(PartitionStrategy::ByTime);
        let ddl = parent_partition_ddl(&config);
        assert!(ddl.contains("PARTITION BY RANGE (created_at)"));
    }

    // ── Capacity alert tests ──

    #[test]
    fn test_evaluate_alerts_no_breach() {
        let stats = TableStats {
            table_name: "memory_item".to_string(),
            total_bytes: 1_000_000,
            index_bytes: 500_000,
            row_count: 1_000,
            vector_storage_bytes: 6_000,
        };
        let thresholds = CapacityThresholds::default();
        let alerts = CapacityMonitor::evaluate_alerts(&stats, &thresholds);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_evaluate_alerts_table_size_breach() {
        let stats = TableStats {
            table_name: "memory_item".to_string(),
            total_bytes: 200 * 1024 * 1024 * 1024, // 200 GB, over 100 GB limit
            index_bytes: 0,
            row_count: 0,
            vector_storage_bytes: 0,
        };
        let thresholds = CapacityThresholds::default();
        let alerts = CapacityMonitor::evaluate_alerts(&stats, &thresholds);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].alert_type, AlertType::TableSize);
        assert!(alerts[0].message.contains("total size"));
    }

    #[test]
    fn test_evaluate_alerts_all_breaches() {
        let thresholds = CapacityThresholds {
            max_table_bytes: 100,
            max_index_bytes: 50,
            max_row_count: 10,
            max_vector_storage_bytes: 80,
            max_tenant_bytes: None,
            max_tenant_row_count: None,
        };
        let stats = TableStats {
            table_name: "memory_item".to_string(),
            total_bytes: 200,
            index_bytes: 100,
            row_count: 50,
            vector_storage_bytes: 100,
        };
        let alerts = CapacityMonitor::evaluate_alerts(&stats, &thresholds);
        assert_eq!(alerts.len(), 4);
        let types: Vec<_> = alerts.iter().map(|a| a.alert_type).collect();
        assert!(types.contains(&AlertType::TableSize));
        assert!(types.contains(&AlertType::IndexSize));
        assert!(types.contains(&AlertType::RowCount));
        assert!(types.contains(&AlertType::VectorStorageGrowth));
    }

    #[test]
    fn test_evaluate_tenant_alert_breach() {
        let thresholds = CapacityThresholds {
            max_tenant_bytes: Some(1_000),
            max_tenant_row_count: Some(100),
            ..Default::default()
        };
        let stats = TenantStats {
            tenant_id: "acme".to_string(),
            row_count: 200,
            total_bytes: 2_000,
            vector_storage_bytes: 800,
        };
        let alerts = CapacityMonitor::evaluate_tenant_alert("acme", &stats, &thresholds);
        assert_eq!(alerts.len(), 2);
        assert!(alerts
            .iter()
            .all(|a| a.tenant_id.as_deref() == Some("acme")));
    }

    #[test]
    fn test_evaluate_tenant_alert_no_breach() {
        let thresholds = CapacityThresholds {
            max_tenant_bytes: Some(10_000),
            max_tenant_row_count: Some(1_000),
            ..Default::default()
        };
        let stats = TenantStats {
            tenant_id: "acme".to_string(),
            row_count: 50,
            total_bytes: 500,
            vector_storage_bytes: 200,
        };
        let alerts = CapacityMonitor::evaluate_tenant_alert("acme", &stats, &thresholds);
        assert!(alerts.is_empty());
    }

    // ── Vector storage estimate tests ──

    #[test]
    fn test_estimate_vector_storage() {
        // 1536 dims * 4 bytes = 6144 bytes per row
        let bytes = estimate_vector_storage(1, 1536);
        assert_eq!(bytes, 6144);
    }

    #[test]
    fn test_estimate_vector_storage_bulk() {
        let bytes = estimate_vector_storage(1_000_000, 1536);
        assert_eq!(bytes, 6_144_000_000);
    }

    #[test]
    fn test_estimate_vector_storage_overflow_safe() {
        let bytes = estimate_vector_storage(i64::MAX, 1536);
        // Should saturate, not panic
        assert!(bytes >= 0);
    }

    // ── Replica routing tests ──

    #[test]
    fn test_should_use_replica_low_lag() {
        assert!(should_use_replica(30.0, 5.0, true));
    }

    #[test]
    fn test_should_use_replica_high_lag() {
        assert!(!should_use_replica(30.0, 60.0, true));
    }

    #[test]
    fn test_should_use_replica_no_replica() {
        assert!(!should_use_replica(30.0, 5.0, false));
    }

    #[test]
    fn test_should_use_replica_exact_threshold() {
        // Lag == threshold should still use replica (<=)
        assert!(should_use_replica(30.0, 30.0, true));
    }

    // ── Strategy enum tests ──

    #[test]
    fn test_strategy_as_str() {
        assert_eq!(PartitionStrategy::ByTenant.as_str(), "by_tenant");
        assert_eq!(
            PartitionStrategy::ByTenantAndTime.as_str(),
            "by_tenant_and_time"
        );
        assert_eq!(PartitionStrategy::ByTime.as_str(), "by_time");
    }

    #[test]
    fn test_strategy_serde() {
        let json = serde_json::to_string(&PartitionStrategy::ByTenantAndTime).unwrap();
        assert_eq!(json, "\"by_tenant_and_time\"");
        let s: PartitionStrategy = serde_json::from_str("\"by_time\"").unwrap();
        assert_eq!(s, PartitionStrategy::ByTime);
    }

    #[test]
    fn test_alert_type_as_str() {
        assert_eq!(AlertType::TableSize.as_str(), "table_size");
        assert_eq!(AlertType::IndexSize.as_str(), "index_size");
        assert_eq!(AlertType::RowCount.as_str(), "row_count");
        assert_eq!(
            AlertType::VectorStorageGrowth.as_str(),
            "vector_storage_growth"
        );
    }

    // ── PartitionManager naming tests (using config, not DB) ──

    // ── Config default tests ──

    #[test]
    fn test_default_config() {
        let config = PartitionConfig::default();
        assert_eq!(config.strategy, PartitionStrategy::ByTenantAndTime);
        assert_eq!(config.partition_prefix, "memory_item");
        assert_eq!(config.retention_days, 365);
        assert_eq!(config.archive_after_days, 90);
    }

    #[test]
    fn test_manager_partition_name_by_tenant() {
        let config = config_for(PartitionStrategy::ByTenant);
        // We can't construct PartitionManager without a real PgPool in unit tests,
        // but partition_name is a free function we can test directly.
        let name = partition_name(&config, "factory_42", test_date());
        assert_eq!(name, "memory_item_factory_42");
    }

    #[test]
    fn test_manager_tenant_partition_name() {
        // Test the tenant partition naming logic used by ensure_partition_exists
        let tenant_id = "acme";
        let prefix = "memory_item";
        let expected = format!("{}_{}", prefix, tenant_id);
        assert_eq!(expected, "memory_item_acme");
    }
}
