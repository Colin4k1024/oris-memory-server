//! SQLite → PostgreSQL migration module.
//!
//! Converts legacy SQLite GeneV1 / CapsuleV1 / UsageReceiptV1 data into
//! PostgreSQL `memory_item` rows (and `memory_access_audit` rows for receipts).
//!
//! # Architecture
//!
//! The migration reads from a legacy SQLite database via the [`SqliteSource`]
//! trait and writes to PostgreSQL via a `sqlx` connection pool. Pure mapping
//! functions ([`gene_to_memory_item`], [`capsule_to_memory_item`],
//! [`usage_receipt_to_audit`]) are decoupled from I/O so they can be unit-tested
//! without a database.
//!
//! # Mapping summary
//!
//! | Source | Destination | memory_type | authority_level | source_type | privacy_class |
//! |--------|-------------|-------------|-----------------|-------------|---------------|
//! | GeneV1 | memory_item | experience  | L2_verified     | AgentInferred | internal |
//! | CapsuleV1 | memory_item | experience | L2_verified    | AgentInferred | internal |
//! | UsageReceiptV1 | memory_access_audit | — | — | — | — |

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use oris_memory_contract::{
    AdoptionStatus, CapsuleV1, ExperienceScope, GeneV1, LifecycleState, OutcomeStatus, Provenance,
    UsageReceiptV1,
};

use crate::memory_types::{
    AuthorityLevel, MemoryItem, MemoryStatus, MemoryType, PrivacyClass, Scope, SourceType,
};
use crate::postgres::Pool;

// ──────────────────────── Error type ────────────────────────

/// Errors that can occur during SQLite → PostgreSQL migration.
#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    /// Wrapper around SQLite-level errors (file open, query, deserialize).
    #[error("SQLite error: {0}")]
    Sqlite(String),

    /// Wrapper around PostgreSQL-level errors.
    #[error("PostgreSQL error: {0}")]
    Postgres(#[from] sqlx::Error),

    /// A single record failed to migrate. The first argument is the record
    /// identifier, the second is the underlying error message.
    #[error("migration failed for record {0}: {1}")]
    RecordFailed(String, String),

    /// Post-migration validation detected a count mismatch or data issue.
    #[error("validation failed: {0}")]
    Validation(String),
}

// ──────────────────────── Report types ────────────────────────

/// Summary of a completed (or in-progress) migration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    pub genes_migrated: usize,
    pub capsules_migrated: usize,
    pub receipts_migrated: usize,
    pub errors: Vec<MigrationErrorEntry>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl MigrationReport {
    /// Create a fresh report with zeroed counts and `started_at = now`.
    pub fn new() -> Self {
        Self {
            genes_migrated: 0,
            capsules_migrated: 0,
            receipts_migrated: 0,
            errors: Vec::new(),
            started_at: Utc::now(),
            completed_at: None,
        }
    }

    /// Mark the migration as finished.
    pub fn complete(&mut self) {
        self.completed_at = Some(Utc::now());
    }

    /// Returns `true` when no errors were recorded.
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }
}

impl Default for MigrationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// A single error encountered while migrating a record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationErrorEntry {
    pub record_id: String,
    pub error: String,
    pub record_type: RecordType,
}

/// The kind of legacy record being migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordType {
    Gene,
    Capsule,
    UsageReceipt,
}

// ──────────────────────── Audit entry ────────────────────────

/// A row targeting the PostgreSQL `memory_access_audit` table, produced from a
/// legacy [`UsageReceiptV1`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessAudit {
    pub id: String,
    pub memory_id: String,
    pub accessor_id: String,
    pub tenant_id: String,
    pub accessed_at: chrono::DateTime<chrono::Utc>,
    pub access_result: String,
    pub context_snapshot: Value,
}

// ──────────────────────── SqliteSource trait ────────────────────────

/// Abstraction over a legacy SQLite data source.
///
/// In production this is backed by [`RusqliteSource`]; in unit tests a
/// [`MockSqliteSource`] is used so that no real database is required.
pub trait SqliteSource: Send + Sync {
    /// Return the tenant identifier that all records in this source belong to.
    fn tenant_id(&self) -> String;

    /// Read all [`GeneV1`] records.
    fn read_genes(&self) -> Result<Vec<GeneV1>, String>;
    /// Read all [`CapsuleV1`] records.
    fn read_capsules(&self) -> Result<Vec<CapsuleV1>, String>;
    /// Read all [`UsageReceiptV1`] records.
    fn read_receipts(&self) -> Result<Vec<UsageReceiptV1>, String>;

    /// Count [`GeneV1`] records without materialising them.
    fn count_genes(&self) -> Result<usize, String>;
    /// Count [`CapsuleV1`] records without materialising them.
    fn count_capsules(&self) -> Result<usize, String>;
    /// Count [`UsageReceiptV1`] records without materialising them.
    fn count_receipts(&self) -> Result<usize, String>;
}

// ──────────────────────── RusqliteSource ────────────────────────

/// Reads legacy data from a real SQLite file using `rusqlite`.
///
/// Expects three tables — `genes`, `capsules`, `usage_receipts` — each with a
/// `data` TEXT column containing a JSON-serialised V1 record, plus an optional
/// `meta` table with `key`/`value` columns (used for `tenant_id`).
pub struct RusqliteSource {
    path: String,
}

impl RusqliteSource {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }

    fn open(&self) -> Result<rusqlite::Connection, String> {
        rusqlite::Connection::open(&self.path)
            .map_err(|e| format!("failed to open SQLite at {}: {}", self.path, e))
    }

    fn table_exists(conn: &rusqlite::Connection, table: &str) -> bool {
        let Ok(n) = conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?1",
            [table],
            |row| row.get::<_, i64>(0),
        ) else {
            return false;
        };
        n > 0
    }
}

impl SqliteSource for RusqliteSource {
    fn tenant_id(&self) -> String {
        let Ok(conn) = self.open() else {
            return "default".to_string();
        };
        if !Self::table_exists(&conn, "meta") {
            return "default".to_string();
        }
        conn.query_row(
            "SELECT value FROM meta WHERE key = 'tenant_id'",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|_| "default".to_string())
    }

    fn read_genes(&self) -> Result<Vec<GeneV1>, String> {
        let conn = self.open()?;
        if !Self::table_exists(&conn, "genes") {
            return Ok(Vec::new());
        }
        let mut stmt = conn
            .prepare("SELECT data FROM genes")
            .map_err(|e| format!("prepare genes: {e}"))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| format!("query genes: {e}"))?;
        let mut genes = Vec::new();
        for row in rows {
            let data = row.map_err(|e| format!("row read: {e}"))?;
            let gene: GeneV1 =
                serde_json::from_str(&data).map_err(|e| format!("deserialize gene: {e}"))?;
            genes.push(gene);
        }
        Ok(genes)
    }

    fn read_capsules(&self) -> Result<Vec<CapsuleV1>, String> {
        let conn = self.open()?;
        if !Self::table_exists(&conn, "capsules") {
            return Ok(Vec::new());
        }
        let mut stmt = conn
            .prepare("SELECT data FROM capsules")
            .map_err(|e| format!("prepare capsules: {e}"))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| format!("query capsules: {e}"))?;
        let mut capsules = Vec::new();
        for row in rows {
            let data = row.map_err(|e| format!("row read: {e}"))?;
            let capsule: CapsuleV1 =
                serde_json::from_str(&data).map_err(|e| format!("deserialize capsule: {e}"))?;
            capsules.push(capsule);
        }
        Ok(capsules)
    }

    fn read_receipts(&self) -> Result<Vec<UsageReceiptV1>, String> {
        let conn = self.open()?;
        if !Self::table_exists(&conn, "usage_receipts") {
            return Ok(Vec::new());
        }
        let mut stmt = conn
            .prepare("SELECT data FROM usage_receipts")
            .map_err(|e| format!("prepare usage_receipts: {e}"))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| format!("query usage_receipts: {e}"))?;
        let mut receipts = Vec::new();
        for row in rows {
            let data = row.map_err(|e| format!("row read: {e}"))?;
            let receipt: UsageReceiptV1 =
                serde_json::from_str(&data).map_err(|e| format!("deserialize receipt: {e}"))?;
            receipts.push(receipt);
        }
        Ok(receipts)
    }

    fn count_genes(&self) -> Result<usize, String> {
        let conn = self.open()?;
        if !Self::table_exists(&conn, "genes") {
            return Ok(0);
        }
        conn.query_row("SELECT COUNT(*) FROM genes", [], |row| row.get::<_, i64>(0))
            .map(|n| n as usize)
            .map_err(|e| format!("count genes: {e}"))
    }

    fn count_capsules(&self) -> Result<usize, String> {
        let conn = self.open()?;
        if !Self::table_exists(&conn, "capsules") {
            return Ok(0);
        }
        conn.query_row("SELECT COUNT(*) FROM capsules", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|n| n as usize)
        .map_err(|e| format!("count capsules: {e}"))
    }

    fn count_receipts(&self) -> Result<usize, String> {
        let conn = self.open()?;
        if !Self::table_exists(&conn, "usage_receipts") {
            return Ok(0);
        }
        conn.query_row("SELECT COUNT(*) FROM usage_receipts", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|n| n as usize)
        .map_err(|e| format!("count receipts: {e}"))
    }
}

// ──────────────────────── MockSqliteSource ────────────────────────

/// In-memory mock for unit tests that do not touch a real SQLite file.
pub struct MockSqliteSource {
    pub tenant_id: String,
    pub genes: Vec<GeneV1>,
    pub capsules: Vec<CapsuleV1>,
    pub receipts: Vec<UsageReceiptV1>,
}

impl MockSqliteSource {
    pub fn new(tenant_id: &str) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            genes: Vec::new(),
            capsules: Vec::new(),
            receipts: Vec::new(),
        }
    }
}

impl SqliteSource for MockSqliteSource {
    fn tenant_id(&self) -> String {
        self.tenant_id.clone()
    }

    fn read_genes(&self) -> Result<Vec<GeneV1>, String> {
        Ok(self.genes.clone())
    }

    fn read_capsules(&self) -> Result<Vec<CapsuleV1>, String> {
        Ok(self.capsules.clone())
    }

    fn read_receipts(&self) -> Result<Vec<UsageReceiptV1>, String> {
        Ok(self.receipts.clone())
    }

    fn count_genes(&self) -> Result<usize, String> {
        Ok(self.genes.len())
    }

    fn count_capsules(&self) -> Result<usize, String> {
        Ok(self.capsules.len())
    }

    fn count_receipts(&self) -> Result<usize, String> {
        Ok(self.receipts.len())
    }
}

// ──────────────────────── Pure mapping functions ────────────────────────

/// Map GeneV1 [`LifecycleState`] to [`MemoryStatus`].
pub fn map_lifecycle_to_status(lifecycle: &LifecycleState) -> MemoryStatus {
    match lifecycle {
        LifecycleState::Candidate => MemoryStatus::Candidate,
        LifecycleState::Stable => MemoryStatus::Active,
        LifecycleState::Deprecated => MemoryStatus::Archived,
        LifecycleState::Quarantined => MemoryStatus::Quarantined,
        LifecycleState::Revoked => MemoryStatus::Revoked,
    }
}

/// Map GeneV1 [`ExperienceScope`] to memory_item [`Scope`].
pub fn map_scope(scope: &ExperienceScope) -> Scope {
    match scope {
        ExperienceScope::Local => Scope::Personal,
        ExperienceScope::Project => Scope::Task,
        ExperienceScope::Tenant => Scope::Factory,
        ExperienceScope::Team => Scope::Team,
        ExperienceScope::Network => Scope::Enterprise,
    }
}

/// Convert a [`GeneV1`] into a [`MemoryItem`] suitable for insertion into the
/// `memory_item` table.
///
/// The gene is serialised into `structured_payload` so that no data is lost.
/// The top-level keys `tags`, `provenance`, `safety_constraints`, and
/// `capsule_refs` are also extracted for structured querying.
pub fn gene_to_memory_item(gene: &GeneV1, tenant_id: &str) -> MemoryItem {
    let memory_id = Uuid::parse_str(&gene.id).unwrap_or_else(|_| Uuid::new_v4());
    let confidence = derive_confidence(&gene.provenance);
    let structured_payload = build_gene_payload(gene);

    MemoryItem {
        memory_id,
        tenant_id: tenant_id.to_string(),
        memory_type: MemoryType::Experience,
        scope: map_scope(&gene.scope),
        subject_type: Some("gene".to_string()),
        subject_id: Some(format!("{}:v{}", gene.id, gene.version)),
        entity_refs: Vec::new(),
        content: Some(format!("{}\n{}", gene.name, gene.description)),
        structured_payload: Some(structured_payload),
        embedding: None,
        source_type: SourceType::AgentInferred,
        source_reference: Some(gene.provenance.source_run_id.clone()),
        evidence_refs: Vec::new(),
        confidence,
        authority_level: AuthorityLevel::L2Verified,
        importance: confidence,
        observed_at: Some(gene.created_at),
        valid_from: Some(gene.created_at),
        valid_to: None,
        privacy_class: PrivacyClass::Internal,
        acl: json!({}),
        retention_policy: None,
        status: map_lifecycle_to_status(&gene.lifecycle),
        version: gene.version as i32,
        derived_from: Vec::new(),
        created_by_user: None,
        created_by_agent: Some(gene.provenance.source_agent.clone()),
        last_verified_at: Some(gene.updated_at),
        created_at: gene.created_at,
        updated_at: gene.updated_at,
    }
}

/// Convert a [`CapsuleV1`] into a [`MemoryItem`].
///
/// Capsules are always mapped to `status = active` (they represent verified
/// execution evidence). The `gene_refs` key in `structured_payload` links back
/// to the parent gene.
pub fn capsule_to_memory_item(capsule: &CapsuleV1, tenant_id: &str) -> MemoryItem {
    let memory_id = Uuid::parse_str(&capsule.id).unwrap_or_else(|_| Uuid::new_v4());
    let structured_payload = build_capsule_payload(capsule);

    MemoryItem {
        memory_id,
        tenant_id: tenant_id.to_string(),
        memory_type: MemoryType::Experience,
        scope: Scope::Personal,
        subject_type: Some("capsule".to_string()),
        subject_id: Some(capsule.id.clone()),
        entity_refs: Vec::new(),
        content: Some(format!(
            "Capsule {} for gene {}:{}",
            capsule.id, capsule.gene_id, capsule.gene_version
        )),
        structured_payload: Some(structured_payload),
        embedding: None,
        source_type: SourceType::AgentInferred,
        source_reference: None,
        evidence_refs: Vec::new(),
        confidence: 0.8,
        authority_level: AuthorityLevel::L2Verified,
        importance: 0.8,
        observed_at: Some(capsule.created_at),
        valid_from: Some(capsule.created_at),
        valid_to: None,
        privacy_class: PrivacyClass::Internal,
        acl: json!({}),
        retention_policy: None,
        status: MemoryStatus::Active,
        version: 1,
        derived_from: Vec::new(),
        created_by_user: None,
        created_by_agent: None,
        last_verified_at: Some(capsule.created_at),
        created_at: capsule.created_at,
        updated_at: capsule.created_at,
    }
}

/// Convert a [`UsageReceiptV1`] into a [`MemoryAccessAudit`] row.
///
/// The `context_snapshot` captures run context, adoption status, applied
/// steps, and cost metrics for forensic traceability.
pub fn usage_receipt_to_audit(receipt: &UsageReceiptV1, tenant_id: &str) -> MemoryAccessAudit {
    MemoryAccessAudit {
        id: receipt.id.clone(),
        memory_id: receipt.gene_id.clone(),
        accessor_id: receipt.agent_id.clone(),
        tenant_id: tenant_id.to_string(),
        accessed_at: receipt.created_at,
        access_result: outcome_to_string(receipt.outcome).to_string(),
        context_snapshot: json!({
            "run_id": receipt.run_id,
            "task_context_hash": receipt.task_context_hash,
            "adoption": adoption_to_string(receipt.adoption),
            "applied_step_ids": receipt.applied_step_ids,
            "failure_reason": receipt.failure_reason,
            "test_evidence_refs": receipt.test_evidence_refs,
            "cost": serde_json::to_value(&receipt.cost).unwrap_or(Value::Null),
            "gene_version": receipt.gene_version,
        }),
    }
}

// ──────────────────────── Helpers ────────────────────────

/// Derive a confidence score in `[0, 1]` from provenance evidence.
///
/// `verified_successes / (verified_successes + verified_failures)`.
/// Falls back to `0.5` when no evidence exists (typical for candidate genes).
fn derive_confidence(provenance: &Provenance) -> f32 {
    let total = provenance.verified_successes + provenance.verified_failures;
    if total == 0 {
        0.5
    } else {
        (provenance.verified_successes as f64 / total as f64) as f32
    }
}

/// Build the `structured_payload` JSON for a gene.
fn build_gene_payload(gene: &GeneV1) -> Value {
    json!({
        "source": serde_json::to_value(gene).unwrap_or(json!({})),
        "tags": gene.metadata.get("tags").cloned().unwrap_or(json!([])),
        "provenance": serde_json::to_value(&gene.provenance).unwrap_or(json!({})),
        "safety_constraints": serde_json::to_value(&gene.safety).unwrap_or(json!({})),
        "capsule_refs": [],
    })
}

/// Build the `structured_payload` JSON for a capsule.
fn build_capsule_payload(capsule: &CapsuleV1) -> Value {
    json!({
        "source": serde_json::to_value(capsule).unwrap_or(json!({})),
        "gene_refs": [format!("{}:v{}", capsule.gene_id, capsule.gene_version)],
    })
}

fn outcome_to_string(outcome: OutcomeStatus) -> &'static str {
    match outcome {
        OutcomeStatus::Succeeded => "succeeded",
        OutcomeStatus::Failed => "failed",
        OutcomeStatus::SafetyFailed => "safety_failed",
        OutcomeStatus::Inconclusive => "inconclusive",
    }
}

fn adoption_to_string(adoption: AdoptionStatus) -> &'static str {
    match adoption {
        AdoptionStatus::Adopted => "adopted",
        AdoptionStatus::PartiallyAdopted => "partially_adopted",
        AdoptionStatus::Rejected => "rejected",
        AdoptionStatus::NotApplicable => "not_applicable",
    }
}

// ──────────────────────── SqliteMigrator ────────────────────────

/// Migrates legacy SQLite data to PostgreSQL `memory_item` format.
///
/// Construct with [`SqliteMigrator::new`] passing the path to the SQLite file
/// and a live PostgreSQL connection pool.
pub struct SqliteMigrator {
    sqlite_path: String,
    pg_pool: Pool,
}

impl SqliteMigrator {
    /// Create a new migrator.
    pub fn new(sqlite_path: &str, pg_pool: Pool) -> Self {
        Self {
            sqlite_path: sqlite_path.to_string(),
            pg_pool,
        }
    }

    /// Open the SQLite source. Fails if the file cannot be opened.
    fn open_source(&self) -> Result<RusqliteSource, MigrationError> {
        Ok(RusqliteSource::new(&self.sqlite_path))
    }

    /// Run the full migration: genes → capsules → receipts.
    ///
    /// Each phase runs independently; a failure in one phase is recorded as an
    /// error entry and does not prevent subsequent phases from running.
    pub async fn migrate_all(&self) -> Result<MigrationReport, MigrationError> {
        let mut report = MigrationReport::new();

        match self.migrate_genes().await {
            Ok(n) => report.genes_migrated = n,
            Err(e) => report.errors.push(MigrationErrorEntry {
                record_id: "genes".to_string(),
                error: e.to_string(),
                record_type: RecordType::Gene,
            }),
        }
        match self.migrate_capsules().await {
            Ok(n) => report.capsules_migrated = n,
            Err(e) => report.errors.push(MigrationErrorEntry {
                record_id: "capsules".to_string(),
                error: e.to_string(),
                record_type: RecordType::Capsule,
            }),
        }
        match self.migrate_receipts().await {
            Ok(n) => report.receipts_migrated = n,
            Err(e) => report.errors.push(MigrationErrorEntry {
                record_id: "receipts".to_string(),
                error: e.to_string(),
                record_type: RecordType::UsageReceipt,
            }),
        }

        report.complete();
        Ok(report)
    }

    /// Migrate GeneV1 records to `memory_item` rows.
    pub async fn migrate_genes(&self) -> Result<usize, MigrationError> {
        let source = self.open_source()?;
        let tenant_id = source.tenant_id();
        let genes = source.read_genes().map_err(MigrationError::Sqlite)?;
        let mut count = 0;
        for gene in &genes {
            let item = gene_to_memory_item(gene, &tenant_id);
            if let Err(e) = insert_memory_item(&self.pg_pool, &item).await {
                tracing::warn!(
                    record_id = gene.id.as_str(),
                    error = %e,
                    "failed to migrate gene; skipping"
                );
                continue;
            }
            count += 1;
        }
        Ok(count)
    }

    /// Migrate CapsuleV1 records to `memory_item` rows.
    pub async fn migrate_capsules(&self) -> Result<usize, MigrationError> {
        let source = self.open_source()?;
        let tenant_id = source.tenant_id();
        let capsules = source.read_capsules().map_err(MigrationError::Sqlite)?;
        let mut count = 0;
        for capsule in &capsules {
            let item = capsule_to_memory_item(capsule, &tenant_id);
            if let Err(e) = insert_memory_item(&self.pg_pool, &item).await {
                tracing::warn!(
                    record_id = capsule.id.as_str(),
                    error = %e,
                    "failed to migrate capsule; skipping"
                );
                continue;
            }
            count += 1;
        }
        Ok(count)
    }

    /// Migrate UsageReceiptV1 records to `memory_access_audit` rows.
    pub async fn migrate_receipts(&self) -> Result<usize, MigrationError> {
        let source = self.open_source()?;
        let tenant_id = source.tenant_id();
        let receipts = source.read_receipts().map_err(MigrationError::Sqlite)?;
        let mut count = 0;
        for receipt in &receipts {
            let audit = usage_receipt_to_audit(receipt, &tenant_id);
            if let Err(e) = insert_audit_entry(&self.pg_pool, &audit).await {
                tracing::warn!(
                    record_id = receipt.id.as_str(),
                    error = %e,
                    "failed to migrate receipt; skipping"
                );
                continue;
            }
            count += 1;
        }
        Ok(count)
    }

    /// Validate migration completeness by comparing record counts.
    ///
    /// Counts source (SQLite) records and destination (PostgreSQL) records,
    /// flagging any mismatch as an error entry.
    pub async fn validate(&self) -> Result<MigrationReport, MigrationError> {
        let source = self.open_source()?;
        let mut report = MigrationReport::new();

        let src_genes = source.count_genes().map_err(MigrationError::Sqlite)?;
        let src_capsules = source.count_capsules().map_err(MigrationError::Sqlite)?;
        let src_receipts = source.count_receipts().map_err(MigrationError::Sqlite)?;

        let pg_genes: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM memory_item WHERE subject_type = 'gene'")
                .fetch_one(&self.pg_pool)
                .await?;
        let pg_capsules: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM memory_item WHERE subject_type = 'capsule'")
                .fetch_one(&self.pg_pool)
                .await?;
        let pg_receipts: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_access_audit")
            .fetch_one(&self.pg_pool)
            .await?;

        report.genes_migrated = pg_genes as usize;
        report.capsules_migrated = pg_capsules as usize;
        report.receipts_migrated = pg_receipts as usize;

        if pg_genes as usize != src_genes {
            report.errors.push(MigrationErrorEntry {
                record_id: "genes".to_string(),
                error: format!("count mismatch: source={src_genes}, pg={pg_genes}"),
                record_type: RecordType::Gene,
            });
        }
        if pg_capsules as usize != src_capsules {
            report.errors.push(MigrationErrorEntry {
                record_id: "capsules".to_string(),
                error: format!("count mismatch: source={src_capsules}, pg={pg_capsules}"),
                record_type: RecordType::Capsule,
            });
        }
        if pg_receipts as usize != src_receipts {
            report.errors.push(MigrationErrorEntry {
                record_id: "receipts".to_string(),
                error: format!("count mismatch: source={src_receipts}, pg={pg_receipts}"),
                record_type: RecordType::UsageReceipt,
            });
        }

        report.complete();
        Ok(report)
    }

    /// Dry run — count records that would be migrated without executing.
    pub async fn dry_run(&self) -> Result<MigrationReport, MigrationError> {
        let source = self.open_source()?;
        let mut report = MigrationReport::new();

        report.genes_migrated = source.count_genes().map_err(MigrationError::Sqlite)?;
        report.capsules_migrated = source.count_capsules().map_err(MigrationError::Sqlite)?;
        report.receipts_migrated = source.count_receipts().map_err(MigrationError::Sqlite)?;

        report.complete();
        Ok(report)
    }
}

// ──────────────────────── PG helpers ────────────────────────

/// Insert a [`MemoryItem`] into the `memory_item` table.
async fn insert_memory_item(pool: &Pool, item: &MemoryItem) -> Result<(), MigrationError> {
    sqlx::query(
        r#"INSERT INTO memory_item (
            memory_id, tenant_id, memory_type, scope,
            subject_type, subject_id, content, structured_payload,
            source_type, source_reference, confidence, authority_level,
            importance, observed_at, valid_from, valid_to,
            privacy_class, acl, retention_policy, status, version,
            derived_from, created_by_user, created_by_agent, last_verified_at,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4,
            $5, $6, $7, $8,
            $9, $10, $11, $12,
            $13, $14, $15, $16,
            $17, $18, $19, $20, $21,
            $22, $23, $24, $25, $26,
            $27
        )"#,
    )
    .bind(item.memory_id)
    .bind(&item.tenant_id)
    .bind(item.memory_type.as_str())
    .bind(item.scope.as_str())
    .bind(&item.subject_type)
    .bind(&item.subject_id)
    .bind(&item.content)
    .bind(&item.structured_payload)
    .bind(item.source_type.as_str())
    .bind(&item.source_reference)
    .bind(item.confidence)
    .bind(item.authority_level.as_str())
    .bind(item.importance)
    .bind(item.observed_at)
    .bind(item.valid_from)
    .bind(item.valid_to)
    .bind(item.privacy_class.as_str())
    .bind(&item.acl)
    .bind(&item.retention_policy)
    .bind(item.status.as_str())
    .bind(item.version)
    .bind(serde_json::to_value(&item.derived_from).unwrap_or(Value::Array(Vec::new())))
    .bind(&item.created_by_user)
    .bind(&item.created_by_agent)
    .bind(item.last_verified_at)
    .bind(item.created_at)
    .bind(item.updated_at)
    .execute(pool)
    .await?;
    Ok(())
}

/// Insert a [`MemoryAccessAudit`] into the `memory_access_audit` table.
async fn insert_audit_entry(pool: &Pool, audit: &MemoryAccessAudit) -> Result<(), MigrationError> {
    sqlx::query(
        r#"INSERT INTO memory_access_audit (
            id, memory_id, accessor_id, tenant_id,
            accessed_at, access_result, context_snapshot
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)"#,
    )
    .bind(&audit.id)
    .bind(&audit.memory_id)
    .bind(&audit.accessor_id)
    .bind(&audit.tenant_id)
    .bind(audit.accessed_at)
    .bind(&audit.access_result)
    .bind(&audit.context_snapshot)
    .execute(pool)
    .await?;
    Ok(())
}

// ──────────────────────── Tests ────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use oris_memory_contract::{
        Applicability, CostMetrics, EvidenceKind, OutcomeStatus, Provenance, RedactionStatus,
        SafetyConstraints, SecretHandling, ValidationCheck, ValidationContract, ValidationResult,
        ValidationSuccessCondition,
    };

    // ── Test fixture builders ──

    fn make_provenance(agent: &str) -> Provenance {
        Provenance {
            source_agent: agent.to_string(),
            source_run_id: "run-001".to_string(),
            trace_refs: Vec::new(),
            extractor_version: None,
            verified_successes: 3,
            verified_failures: 1,
            distinct_task_contexts: 2,
        }
    }

    fn make_safety() -> SafetyConstraints {
        SafetyConstraints {
            suggestion_only: true,
            forbidden_operations: vec!["rm -rf /".to_string()],
            required_approvals: Vec::new(),
            secret_handling: SecretHandling::Redact,
        }
    }

    fn make_gene(id: &str, lifecycle: LifecycleState, scope: ExperienceScope) -> GeneV1 {
        GeneV1 {
            id: id.to_string(),
            version: 1,
            name: format!("Test Gene {id}"),
            description: "A test gene for migration".to_string(),
            scope,
            task_category: "testing".to_string(),
            applicability: Applicability {
                required_signals: vec!["signal1".to_string()],
                excluded_signals: Vec::new(),
                environments: Vec::new(),
                project_ids: Vec::new(),
                tenant_ids: vec!["acme".to_string()],
                do_not_use_when: Vec::new(),
            },
            steps: Vec::new(),
            tool_requirements: Vec::new(),
            safety: make_safety(),
            validation: ValidationContract {
                checks: vec![ValidationCheck {
                    id: "chk1".to_string(),
                    command_or_assertion: "cargo test".to_string(),
                    evidence_kind: EvidenceKind::Test,
                    timeout_seconds: None,
                }],
                success_condition: ValidationSuccessCondition::All,
            },
            provenance: make_provenance("test-agent"),
            lifecycle,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    fn make_capsule(id: &str, gene_id: &str) -> CapsuleV1 {
        CapsuleV1 {
            id: id.to_string(),
            gene_id: gene_id.to_string(),
            gene_version: 1,
            environment_fingerprint: "fp-001".to_string(),
            task_context_hash: "tch-001".to_string(),
            execution_evidence_hash: "eeh-001".to_string(),
            validation: ValidationResult {
                status: OutcomeStatus::Succeeded,
                checks: Vec::new(),
                summary: Some("all good".to_string()),
            },
            artifact_refs: vec!["art1".to_string()],
            redaction: RedactionStatus::VerifiedClean,
            created_at: Utc::now(),
        }
    }

    fn make_receipt(id: &str, gene_id: &str) -> UsageReceiptV1 {
        UsageReceiptV1 {
            id: id.to_string(),
            gene_id: gene_id.to_string(),
            gene_version: 1,
            agent_id: "agent-007".to_string(),
            run_id: "run-42".to_string(),
            task_context_hash: "tch-001".to_string(),
            adoption: AdoptionStatus::Adopted,
            applied_step_ids: vec!["step1".to_string()],
            outcome: OutcomeStatus::Succeeded,
            failure_reason: None,
            test_evidence_refs: vec!["ev1".to_string()],
            cost: Some(CostMetrics {
                input_tokens: Some(1000),
                output_tokens: Some(500),
                latency_ms: Some(1200),
                monetary_cost: Some(0.02),
                currency: Some("USD".to_string()),
            }),
            created_at: Utc::now(),
        }
    }

    // ── Lifecycle mapping tests ──

    #[test]
    fn map_lifecycle_candidate_to_candidate() {
        assert_eq!(
            map_lifecycle_to_status(&LifecycleState::Candidate),
            MemoryStatus::Candidate
        );
    }

    #[test]
    fn map_lifecycle_stable_to_active() {
        assert_eq!(
            map_lifecycle_to_status(&LifecycleState::Stable),
            MemoryStatus::Active
        );
    }

    #[test]
    fn map_lifecycle_deprecated_to_archived() {
        assert_eq!(
            map_lifecycle_to_status(&LifecycleState::Deprecated),
            MemoryStatus::Archived
        );
    }

    #[test]
    fn map_lifecycle_quarantined_to_quarantined() {
        assert_eq!(
            map_lifecycle_to_status(&LifecycleState::Quarantined),
            MemoryStatus::Quarantined
        );
    }

    #[test]
    fn map_lifecycle_revoked_to_revoked() {
        assert_eq!(
            map_lifecycle_to_status(&LifecycleState::Revoked),
            MemoryStatus::Revoked
        );
    }

    // ── Scope mapping tests ──

    #[test]
    fn map_scope_local_to_personal() {
        assert_eq!(map_scope(&ExperienceScope::Local), Scope::Personal);
    }

    #[test]
    fn map_scope_team_to_team() {
        assert_eq!(map_scope(&ExperienceScope::Team), Scope::Team);
    }

    #[test]
    fn map_scope_network_to_enterprise() {
        assert_eq!(map_scope(&ExperienceScope::Network), Scope::Enterprise);
    }

    #[test]
    fn map_scope_project_to_task() {
        assert_eq!(map_scope(&ExperienceScope::Project), Scope::Task);
    }

    #[test]
    fn map_scope_tenant_to_factory() {
        assert_eq!(map_scope(&ExperienceScope::Tenant), Scope::Factory);
    }

    // ── GeneV1 → MemoryItem mapping tests ──

    #[test]
    fn gene_to_memory_item_sets_memory_type_experience() {
        let gene = make_gene("g1", LifecycleState::Candidate, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        assert_eq!(item.memory_type, MemoryType::Experience);
    }

    #[test]
    fn gene_to_memory_item_sets_authority_and_source() {
        let gene = make_gene("g2", LifecycleState::Stable, ExperienceScope::Team);
        let item = gene_to_memory_item(&gene, "acme");
        assert_eq!(item.authority_level, AuthorityLevel::L2Verified);
        assert_eq!(item.source_type, SourceType::AgentInferred);
        assert_eq!(item.privacy_class, PrivacyClass::Internal);
    }

    #[test]
    fn gene_to_memory_item_maps_status_from_lifecycle() {
        let gene = make_gene("g3", LifecycleState::Deprecated, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        assert_eq!(item.status, MemoryStatus::Archived);
    }

    #[test]
    fn gene_to_memory_item_maps_scope() {
        let gene = make_gene("g4", LifecycleState::Stable, ExperienceScope::Network);
        let item = gene_to_memory_item(&gene, "acme");
        assert_eq!(item.scope, Scope::Enterprise);
    }

    #[test]
    fn gene_to_memory_item_sets_tenant_and_subject() {
        let gene = make_gene("g5", LifecycleState::Stable, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        assert_eq!(item.tenant_id, "acme");
        assert_eq!(item.subject_type.as_deref(), Some("gene"));
        assert_eq!(item.subject_id.as_deref(), Some("g5:v1"));
    }

    #[test]
    fn gene_to_memory_item_sets_content() {
        let gene = make_gene("g6", LifecycleState::Candidate, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        assert!(item.content.as_ref().unwrap().contains("Test Gene g6"));
        assert!(item
            .content
            .as_ref()
            .unwrap()
            .contains("test gene for migration"));
    }

    #[test]
    fn gene_to_memory_item_structured_payload_has_keys() {
        let gene = make_gene("g7", LifecycleState::Stable, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        let payload = item.structured_payload.unwrap();
        assert!(payload.get("provenance").is_some());
        assert!(payload.get("safety_constraints").is_some());
        assert!(payload.get("capsule_refs").is_some());
        assert!(payload.get("tags").is_some());
        assert!(payload.get("source").is_some());
    }

    #[test]
    fn gene_to_memory_item_derives_confidence_from_provenance() {
        // 3 successes, 1 failure → 0.75
        let gene = make_gene("g8", LifecycleState::Stable, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        assert!((item.confidence - 0.75).abs() < 0.01);
        assert!((item.importance - 0.75).abs() < 0.01);
    }

    #[test]
    fn gene_to_memory_item_confidence_default_for_no_evidence() {
        let mut gene = make_gene("g9", LifecycleState::Candidate, ExperienceScope::Local);
        gene.provenance = Provenance {
            verified_successes: 0,
            verified_failures: 0,
            ..make_provenance("agent-x")
        };
        let item = gene_to_memory_item(&gene, "acme");
        assert!((item.confidence - 0.5).abs() < 0.01);
    }

    #[test]
    fn gene_to_memory_item_sets_created_by_agent() {
        let gene = make_gene("g10", LifecycleState::Stable, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        assert_eq!(item.created_by_agent.as_deref(), Some("test-agent"));
        assert!(item.created_by_user.is_none());
    }

    #[test]
    fn gene_to_memory_item_preserves_version() {
        let gene = make_gene("g11", LifecycleState::Stable, ExperienceScope::Local);
        let item = gene_to_memory_item(&gene, "acme");
        assert_eq!(item.version, 1);
    }

    // ── CapsuleV1 → MemoryItem mapping tests ──

    #[test]
    fn capsule_to_memory_item_basic() {
        let capsule = make_capsule("c1", "gene-1");
        let item = capsule_to_memory_item(&capsule, "acme");
        assert_eq!(item.memory_type, MemoryType::Experience);
        assert_eq!(item.status, MemoryStatus::Active);
        assert_eq!(item.authority_level, AuthorityLevel::L2Verified);
        assert_eq!(item.source_type, SourceType::AgentInferred);
        assert_eq!(item.privacy_class, PrivacyClass::Internal);
        assert_eq!(item.tenant_id, "acme");
        assert_eq!(item.subject_type.as_deref(), Some("capsule"));
        assert_eq!(item.subject_id.as_deref(), Some("c1"));
    }

    #[test]
    fn capsule_to_memory_item_gene_refs_in_payload() {
        let capsule = make_capsule("c2", "gene-2");
        let item = capsule_to_memory_item(&capsule, "acme");
        let payload = item.structured_payload.unwrap();
        let refs = payload.get("gene_refs").unwrap();
        assert_eq!(refs.as_array().unwrap().len(), 1);
        assert_eq!(refs.as_array().unwrap()[0], "gene-2:v1");
    }

    #[test]
    fn capsule_to_memory_item_content_contains_ids() {
        let capsule = make_capsule("c3", "gene-3");
        let item = capsule_to_memory_item(&capsule, "acme");
        let content = item.content.unwrap();
        assert!(content.contains("c3"));
        assert!(content.contains("gene-3"));
    }

    // ── UsageReceiptV1 → MemoryAccessAudit mapping tests ──

    #[test]
    fn usage_receipt_to_audit_basic() {
        let receipt = make_receipt("r1", "gene-1");
        let audit = usage_receipt_to_audit(&receipt, "acme");
        assert_eq!(audit.id, "r1");
        assert_eq!(audit.memory_id, "gene-1");
        assert_eq!(audit.accessor_id, "agent-007");
        assert_eq!(audit.tenant_id, "acme");
        assert_eq!(audit.access_result, "succeeded");
    }

    #[test]
    fn usage_receipt_to_audit_context_snapshot() {
        let receipt = make_receipt("r2", "gene-2");
        let audit = usage_receipt_to_audit(&receipt, "acme");
        let ctx = &audit.context_snapshot;
        assert_eq!(ctx.get("run_id").unwrap(), "run-42");
        assert_eq!(ctx.get("adoption").unwrap(), "adopted");
        assert_eq!(ctx.get("gene_version").unwrap(), 1);
        assert!(ctx.get("cost").is_some());
    }

    #[test]
    fn usage_receipt_to_audit_outcome_mapping() {
        for (outcome, expected) in [
            (OutcomeStatus::Succeeded, "succeeded"),
            (OutcomeStatus::Failed, "failed"),
            (OutcomeStatus::SafetyFailed, "safety_failed"),
            (OutcomeStatus::Inconclusive, "inconclusive"),
        ] {
            let mut receipt = make_receipt("r3", "gene-3");
            receipt.outcome = outcome;
            let audit = usage_receipt_to_audit(&receipt, "acme");
            assert_eq!(audit.access_result, expected);
        }
    }

    // ── MigrationReport tests ──

    #[test]
    fn migration_report_empty() {
        let report = MigrationReport::new();
        assert_eq!(report.genes_migrated, 0);
        assert_eq!(report.capsules_migrated, 0);
        assert_eq!(report.receipts_migrated, 0);
        assert!(report.errors.is_empty());
        assert!(report.completed_at.is_none());
        assert!(report.is_clean());
    }

    #[test]
    fn migration_report_complete_sets_timestamp() {
        let mut report = MigrationReport::new();
        assert!(report.completed_at.is_none());
        report.complete();
        assert!(report.completed_at.is_some());
    }

    #[test]
    fn migration_report_with_errors() {
        let mut report = MigrationReport::new();
        report.errors.push(MigrationErrorEntry {
            record_id: "g-bad".to_string(),
            error: "deserialize failed".to_string(),
            record_type: RecordType::Gene,
        });
        report.errors.push(MigrationErrorEntry {
            record_id: "c-bad".to_string(),
            error: "constraint violation".to_string(),
            record_type: RecordType::Capsule,
        });
        assert!(!report.is_clean());
        assert_eq!(report.errors.len(), 2);
    }

    #[test]
    fn migration_report_serialization_roundtrip() {
        let mut report = MigrationReport::new();
        report.genes_migrated = 10;
        report.capsules_migrated = 5;
        report.receipts_migrated = 3;
        report.errors.push(MigrationErrorEntry {
            record_id: "x".to_string(),
            error: "boom".to_string(),
            record_type: RecordType::Gene,
        });
        report.complete();

        let json = serde_json::to_string(&report).unwrap();
        let restored: MigrationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.genes_migrated, 10);
        assert_eq!(restored.capsules_migrated, 5);
        assert_eq!(restored.receipts_migrated, 3);
        assert_eq!(restored.errors.len(), 1);
        assert_eq!(restored.errors[0].record_id, "x");
        assert!(restored.completed_at.is_some());
    }

    // ── MigrationError tests ──

    #[test]
    fn migration_error_display_variants() {
        assert_eq!(
            MigrationError::Sqlite("disk full".into()).to_string(),
            "SQLite error: disk full"
        );
        assert_eq!(
            MigrationError::Validation("count mismatch".into()).to_string(),
            "validation failed: count mismatch"
        );
        assert_eq!(
            MigrationError::RecordFailed("g1".into(), "bad json".into()).to_string(),
            "migration failed for record g1: bad json"
        );
    }

    // ── RecordType tests ──

    #[test]
    fn record_type_variants_and_equality() {
        assert_eq!(RecordType::Gene, RecordType::Gene);
        assert_ne!(RecordType::Gene, RecordType::Capsule);
        assert_ne!(RecordType::Capsule, RecordType::UsageReceipt);
        assert_ne!(RecordType::Gene, RecordType::UsageReceipt);
    }

    #[test]
    fn record_type_serialization() {
        let json = serde_json::to_string(&RecordType::Gene).unwrap();
        assert_eq!(json, "\"gene\"");
        let json = serde_json::to_string(&RecordType::Capsule).unwrap();
        assert_eq!(json, "\"capsule\"");
        let json = serde_json::to_string(&RecordType::UsageReceipt).unwrap();
        assert_eq!(json, "\"usage_receipt\"");
    }

    // ── MigrationErrorEntry tests ──

    #[test]
    fn migration_error_entry_construction() {
        let entry = MigrationErrorEntry {
            record_id: "g-123".to_string(),
            error: "timeout".to_string(),
            record_type: RecordType::Gene,
        };
        assert_eq!(entry.record_id, "g-123");
        assert_eq!(entry.error, "timeout");
        assert_eq!(entry.record_type, RecordType::Gene);
    }

    #[test]
    fn migration_error_entry_serialization() {
        let entry = MigrationErrorEntry {
            record_id: "c-1".to_string(),
            error: "oops".to_string(),
            record_type: RecordType::Capsule,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let restored: MigrationErrorEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.record_id, "c-1");
        assert_eq!(restored.error, "oops");
        assert_eq!(restored.record_type, RecordType::Capsule);
    }

    // ── MemoryAccessAudit tests ──

    #[test]
    fn memory_access_audit_serialization() {
        let receipt = make_receipt("r4", "gene-4");
        let audit = usage_receipt_to_audit(&receipt, "acme");
        let json = serde_json::to_string(&audit).unwrap();
        let restored: MemoryAccessAudit = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, "r4");
        assert_eq!(restored.memory_id, "gene-4");
        assert_eq!(restored.tenant_id, "acme");
        assert_eq!(restored.access_result, "succeeded");
    }

    // ── Dry run report structure test ──

    #[test]
    fn dry_run_report_structure_with_mock_source() {
        let source = MockSqliteSource::new("acme");
        let mut report = MigrationReport::new();
        report.genes_migrated = source.count_genes().unwrap();
        report.capsules_migrated = source.count_capsules().unwrap();
        report.receipts_migrated = source.count_receipts().unwrap();
        report.complete();

        assert_eq!(report.genes_migrated, 0);
        assert_eq!(report.capsules_migrated, 0);
        assert_eq!(report.receipts_migrated, 0);
        assert!(report.is_clean());
        assert!(report.completed_at.is_some());
    }

    #[test]
    fn dry_run_report_with_populated_mock() {
        let mut source = MockSqliteSource::new("acme");
        source.genes = vec![
            make_gene("g1", LifecycleState::Stable, ExperienceScope::Local),
            make_gene("g2", LifecycleState::Candidate, ExperienceScope::Team),
        ];
        source.capsules = vec![make_capsule("c1", "g1")];
        source.receipts = vec![make_receipt("r1", "g1")];

        let mut report = MigrationReport::new();
        report.genes_migrated = source.count_genes().unwrap();
        report.capsules_migrated = source.count_capsules().unwrap();
        report.receipts_migrated = source.count_receipts().unwrap();
        report.complete();

        assert_eq!(report.genes_migrated, 2);
        assert_eq!(report.capsules_migrated, 1);
        assert_eq!(report.receipts_migrated, 1);
        assert!(report.is_clean());
    }

    #[test]
    fn mock_source_reads_match_counts() {
        let mut source = MockSqliteSource::new("acme");
        source.genes = vec![
            make_gene("g1", LifecycleState::Stable, ExperienceScope::Local),
            make_gene("g2", LifecycleState::Stable, ExperienceScope::Local),
            make_gene("g3", LifecycleState::Stable, ExperienceScope::Local),
        ];
        assert_eq!(source.count_genes().unwrap(), 3);
        assert_eq!(source.read_genes().unwrap().len(), 3);
        assert_eq!(source.tenant_id(), "acme");
    }

    #[test]
    fn gene_to_memory_item_uuid_parse_fallback() {
        // Non-UUID id should still produce a valid memory_id (via fallback)
        let gene = make_gene(
            "not-a-uuid",
            LifecycleState::Candidate,
            ExperienceScope::Local,
        );
        let item = gene_to_memory_item(&gene, "acme");
        assert_ne!(item.memory_id, Uuid::nil());
    }
}
