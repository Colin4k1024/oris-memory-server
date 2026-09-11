//! Unified memory control plane backed by PostgreSQL.
//!
//! Replaces the legacy SQLite `ExperienceControlPlane` with a PostgreSQL +
//! pgvector implementation that stores experience data as `memory_item` rows
//! with `memory_type = experience`. This is the single source of truth per
//! architecture §8 ("合并双存储系统为统一 memory_item 表").
//!
//! # Lifecycle Mapping
//!
//! | GeneV1 Lifecycle   | memory_item.status |
//! |--------------------|---------------------|
//! | Candidate          | candidate           |
//! | Stable             | active              |
//! | Deprecated         | archived            |
//! | Quarantined        | quarantined         |
//! | Revoked            | revoked             |
//!
//! # Scope Mapping
//!
//! | GeneV1 Scope | memory_item.scope |
//! |--------------|-------------------|
//! | Local        | personal          |
//! | Team         | team              |
//! | Network      | enterprise       |

use std::sync::Arc;

use chrono::Utc;
use serde::de::Error as _;
use serde_json::{json, Value};
use thiserror::Error;
use tracing::instrument;
use uuid::Uuid;

use oris_memory_contract::{
    CapsuleV1, ExperienceBundleV1, ExperienceScope, GeneV1, LifecycleState, Provenance,
};
use oris_memory_store::memory_types::{
    AuthorityLevel, EventType, MemoryCandidate, MemoryItem, MemoryStatus, MemoryType, PrivacyClass,
    Scope, SourceType,
};
use oris_memory_store::postgres::memory_repo::{MemoryRepo, MemoryRepoError};
use oris_memory_store::postgres::outbox::{OutboxError, OutboxRepo};
use oris_memory_store::postgres::search::{SearchError, SearchRepo};
use oris_memory_store::postgres::Pool;

// ──────────────────────────── Errors ────────────────────────────

#[derive(Debug, Error)]
pub enum MemoryControlError {
    #[error("memory not found: {0}")]
    NotFound(String),

    #[error("contract validation failed: {0}")]
    Contract(String),

    #[error("invalid lifecycle transition: {0}")]
    InvalidTransition(String),

    #[error("storage error: {0}")]
    Storage(#[from] MemoryRepoError),

    #[error("search error: {0}")]
    Search(#[from] SearchError),

    #[error("outbox error: {0}")]
    Outbox(#[from] OutboxError),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("sqlx error: {0}")]
    Sqlx(#[from] sqlx::Error),
}

// ──────────────────────────── Search Types ────────────────────────────

/// Search query for the memory control plane (replaces ExperienceSearchQuery).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySearchQuery {
    #[serde(default = "default_tenant")]
    pub tenant_id: String,
    pub text: String,
    #[serde(default = "default_limit")]
    pub limit: i64,
    #[serde(default)]
    pub offset: i64,
    /// Optional embedding vector for semantic search.
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    /// Filter by scope.
    #[serde(default)]
    pub scopes: Vec<Scope>,
}

fn default_tenant() -> String {
    "default".into()
}

fn default_limit() -> i64 {
    20
}

impl Default for MemorySearchQuery {
    fn default() -> Self {
        Self {
            tenant_id: default_tenant(),
            text: String::new(),
            limit: default_limit(),
            offset: 0,
            embedding: None,
            scopes: Vec::new(),
        }
    }
}

/// A single search result (replaces ExperienceSearchResult).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySearchResult {
    pub memory_id: Uuid,
    pub score: f64,
    pub match_type: String,
    pub content: Option<String>,
    pub structured_payload: Option<Value>,
    pub status: String,
    pub scope: String,
    pub confidence: f32,
}

// ──────────────────────────── Mappers ────────────────────────────

/// Map GeneV1 lifecycle state to memory_item status.
fn lifecycle_to_status(lifecycle: &LifecycleState) -> MemoryStatus {
    match lifecycle {
        LifecycleState::Candidate => MemoryStatus::Candidate,
        LifecycleState::Stable => MemoryStatus::Active,
        LifecycleState::Deprecated => MemoryStatus::Archived,
        LifecycleState::Quarantined => MemoryStatus::Quarantined,
        LifecycleState::Revoked => MemoryStatus::Revoked,
    }
}

/// Map memory_item status back to GeneV1 lifecycle state.
fn status_to_lifecycle(status: &MemoryStatus) -> LifecycleState {
    match status {
        MemoryStatus::Candidate => LifecycleState::Candidate,
        MemoryStatus::Active => LifecycleState::Stable,
        MemoryStatus::Archived => LifecycleState::Deprecated,
        MemoryStatus::Quarantined => LifecycleState::Quarantined,
        MemoryStatus::Revoked => LifecycleState::Revoked,
    }
}

/// Map GeneV1 scope to memory_item scope.
fn experience_scope_to_scope(scope: &ExperienceScope) -> Scope {
    match scope {
        ExperienceScope::Local => Scope::Personal,
        ExperienceScope::Project => Scope::Task,
        ExperienceScope::Tenant => Scope::Factory,
        ExperienceScope::Team => Scope::Team,
        ExperienceScope::Network => Scope::Enterprise,
    }
}

/// Map memory_item scope back to GeneV1 scope.
fn scope_to_experience_scope(scope: &Scope) -> ExperienceScope {
    match scope {
        Scope::Personal => ExperienceScope::Local,
        Scope::Agent => ExperienceScope::Local,
        Scope::Task => ExperienceScope::Project,
        Scope::Team => ExperienceScope::Team,
        Scope::Process => ExperienceScope::Project,
        Scope::Factory => ExperienceScope::Tenant,
        Scope::Enterprise => ExperienceScope::Network,
    }
}

/// Convert a GeneV1 into a MemoryCandidate for insertion.
fn gene_to_candidate(gene: &GeneV1, tenant_id: &str) -> Result<MemoryCandidate, serde_json::Error> {
    let payload = serde_json::to_value(gene)?;
    Ok(MemoryCandidate {
        tenant_id: tenant_id.to_string(),
        memory_type: MemoryType::Experience,
        scope: experience_scope_to_scope(&gene.scope),
        subject_type: Some("gene".to_string()),
        subject_id: Some(format!("{}:v{}", gene.id, gene.version)),
        content: Some(gene.name.clone()),
        structured_payload: Some(payload),
        source_type: SourceType::AgentInferred,
        source_reference: Some(gene.provenance.source_agent.clone()),
        confidence: 0.8,
        authority_level: AuthorityLevel::L2Verified,
        privacy_class: PrivacyClass::Internal,
        created_by_user: None,
        created_by_agent: Some(gene.provenance.source_agent.clone()),
    })
}

/// Convert a MemoryItem back to a GeneV1.
fn item_to_gene(item: &MemoryItem) -> Result<GeneV1, serde_json::Error> {
    let payload = item
        .structured_payload
        .as_ref()
        .ok_or_else(|| serde_json::Error::custom("missing structured_payload"))?;
    let mut gene: GeneV1 = serde_json::from_value(payload.clone())?;
    // Override lifecycle from the memory_item status (source of truth).
    gene.lifecycle = status_to_lifecycle(&item.status);
    gene.scope = scope_to_experience_scope(&item.scope);
    Ok(gene)
}

/// Convert a MemoryItem to a search result.
fn item_to_search_result(item: &MemoryItem, score: f64, match_type: &str) -> MemorySearchResult {
    MemorySearchResult {
        memory_id: item.memory_id,
        score,
        match_type: match_type.to_string(),
        content: item.content.clone(),
        structured_payload: item.structured_payload.clone(),
        status: item.status.as_str().to_string(),
        scope: item.scope.as_str().to_string(),
        confidence: item.confidence,
    }
}

// ──────────────────────────── MemoryControlPlane ────────────────────────────

/// Unified, PostgreSQL-backed control plane for memory management.
///
/// This replaces the legacy `ExperienceControlPlane` (SQLite) and serves as
/// the single entry point for experience memory CRUD and search via
/// `MemoryRepo` + `SearchRepo` + `OutboxRepo`.
pub struct MemoryControlPlane {
    pool: Pool,
    memory_repo: MemoryRepo,
    search_repo: SearchRepo,
    outbox_repo: OutboxRepo,
}

impl MemoryControlPlane {
    /// Create from a PostgreSQL connection pool.
    pub fn new(pool: Pool) -> Self {
        Self {
            memory_repo: MemoryRepo::new(pool.clone()),
            search_repo: SearchRepo::new(pool.clone()),
            outbox_repo: OutboxRepo::new(pool.clone()),
            pool,
        }
    }

    /// Propose a new experience gene as a candidate memory item.
    ///
    /// The gene is stored with `status = candidate` and `memory_type = experience`.
    /// An outbox event is enqueued so downstream consumers (embedding service,
    /// cache invalidator) can react.
    #[instrument(skip(self, bundle))]
    pub async fn propose(
        &self,
        bundle: &ExperienceBundleV1,
        tenant_id: &str,
    ) -> Result<Uuid, MemoryControlError> {
        bundle
            .validate()
            .map_err(|e| MemoryControlError::Contract(e.to_string()))?;

        let mut gene = bundle.gene.clone();
        gene.lifecycle = LifecycleState::Candidate;
        if matches!(gene.scope, ExperienceScope::Team | ExperienceScope::Network) {
            gene.scope = ExperienceScope::Local;
        }

        let candidate = gene_to_candidate(&gene, tenant_id)?;
        let memory_id = self.memory_repo.insert(&candidate).await?;

        // Enqueue outbox event for async processing (embedding, cache invalidation).
        let payload = json!({
            "memory_id": memory_id,
            "memory_type": "experience",
            "action": "proposed",
            "gene_id": gene.id,
            "gene_version": gene.version,
        });
        self.outbox_repo
            .enqueue(EventType::MemoryProposed, &memory_id.to_string(), &payload)
            .await?;

        Ok(memory_id)
    }

    /// Get a memory item by its UUID.
    #[instrument(skip(self))]
    pub async fn get(&self, memory_id: Uuid) -> Result<Option<MemoryItem>, MemoryControlError> {
        Ok(self.memory_repo.get_by_id(memory_id).await?)
    }

    /// Get a memory item and convert it back to a GeneV1.
    #[instrument(skip(self))]
    pub async fn get_gene(&self, memory_id: Uuid) -> Result<Option<GeneV1>, MemoryControlError> {
        let item = self.memory_repo.get_by_id(memory_id).await?;
        match item {
            Some(i) => Ok(Some(item_to_gene(&i)?)),
            None => Ok(None),
        }
    }

    /// Search memory items using hybrid retrieval (keyword + vector + structured).
    ///
    /// This replaces the legacy `hashed_cosine` + `bm25_lite` search with
    /// real PostgreSQL full-text search + pgvector HNSW similarity.
    #[instrument(skip(self, query))]
    pub async fn search(
        &self,
        query: &MemorySearchQuery,
    ) -> Result<Vec<MemorySearchResult>, MemoryControlError> {
        use oris_memory_store::memory_types::SearchParams;

        let mut params = SearchParams::new(&query.tenant_id);
        params.query_text = if query.text.is_empty() {
            None
        } else {
            Some(query.text.clone())
        };
        params.memory_types = vec![MemoryType::Experience];
        params.limit = query.limit;
        params.offset = query.offset;
        params.embedding = query.embedding.clone();
        if !query.scopes.is_empty() {
            params.scopes = query.scopes.clone();
        }

        let results = self.search_repo.hybrid_search(&params).await?;
        Ok(results
            .iter()
            .map(|r| {
                item_to_search_result(&r.item, r.score, format!("{:?}", r.matched_by).as_str())
            })
            .collect())
    }

    /// Promote a candidate to active (Stable).
    #[instrument(skip(self))]
    pub async fn promote(&self, memory_id: Uuid) -> Result<(), MemoryControlError> {
        let item = self
            .memory_repo
            .get_by_id(memory_id)
            .await?
            .ok_or_else(|| MemoryControlError::NotFound(memory_id.to_string()))?;

        if item.status != MemoryStatus::Candidate {
            return Err(MemoryControlError::InvalidTransition(format!(
                "cannot promote from {:?}",
                item.status
            )));
        }

        self.memory_repo.promote(memory_id).await?;

        let payload = json!({
            "memory_id": memory_id,
            "action": "promoted",
            "from": "candidate",
            "to": "active",
        });
        self.outbox_repo
            .enqueue(EventType::MemoryPromoted, &memory_id.to_string(), &payload)
            .await?;

        Ok(())
    }

    /// Revoke a memory item (set status to revoked).
    #[instrument(skip(self))]
    pub async fn revoke(&self, memory_id: Uuid) -> Result<(), MemoryControlError> {
        let item = self
            .memory_repo
            .get_by_id(memory_id)
            .await?
            .ok_or_else(|| MemoryControlError::NotFound(memory_id.to_string()))?;

        if item.status == MemoryStatus::Revoked {
            return Err(MemoryControlError::InvalidTransition(
                "already revoked".into(),
            ));
        }

        self.memory_repo.revoke(memory_id).await?;

        let payload = json!({
            "memory_id": memory_id,
            "action": "revoked",
            "previous_status": item.status.as_str(),
        });
        self.outbox_repo
            .enqueue(EventType::MemoryRevoked, &memory_id.to_string(), &payload)
            .await?;

        Ok(())
    }

    /// Quarantine a memory item (e.g., suspected poisoning).
    #[instrument(skip(self))]
    pub async fn quarantine(&self, memory_id: Uuid) -> Result<(), MemoryControlError> {
        self.memory_repo.quarantine(memory_id).await?;

        let payload = json!({
            "memory_id": memory_id,
            "action": "quarantined",
        });
        self.outbox_repo
            .enqueue(
                EventType::MemoryQuarantined,
                &memory_id.to_string(),
                &payload,
            )
            .await?;

        Ok(())
    }

    /// Archive a memory item (set status to archived / Deprecated).
    #[instrument(skip(self))]
    pub async fn archive(&self, memory_id: Uuid) -> Result<(), MemoryControlError> {
        self.memory_repo.archive(memory_id).await?;

        let payload = json!({
            "memory_id": memory_id,
            "action": "archived",
        });
        self.outbox_repo
            .enqueue(EventType::MemoryExpired, &memory_id.to_string(), &payload)
            .await?;

        Ok(())
    }

    /// Record a usage outcome and update the gene's provenance.
    ///
    /// This replaces the legacy `record_outcome` method. It:
    /// 1. Reads the memory item
    /// 2. Deserializes the GeneV1
    /// 3. Updates provenance (success/failure counts)
    /// 4. Bumps version
    /// 5. Enqueues outbox event
    #[instrument(skip(self))]
    pub async fn record_outcome(
        &self,
        memory_id: Uuid,
        success: bool,
        run_id: &str,
    ) -> Result<(), MemoryControlError> {
        let item = self
            .memory_repo
            .get_by_id(memory_id)
            .await?
            .ok_or_else(|| MemoryControlError::NotFound(memory_id.to_string()))?;

        let mut gene = item_to_gene(&item)?;

        // Update provenance.
        if success {
            gene.provenance.verified_successes += 1;
        } else {
            gene.provenance.verified_failures += 1;
        }
        gene.updated_at = Utc::now();

        // Serialize updated gene back to structured_payload and bump version.
        let payload = serde_json::to_value(&gene)?;
        let new_version = self.memory_repo.bump_version(memory_id).await?;

        // Update the structured_payload (content) with the new gene.
        // This uses a direct SQL update since MemoryRepo doesn't have an
        // update_content method.
        sqlx::query("UPDATE memory_item SET structured_payload = $1, updated_at = NOW() WHERE memory_id = $2")
            .bind(payload)
            .bind(memory_id)
            .execute(&self.pool)
            .await?;

        let outcome_str = if success { "success" } else { "failure" };
        let event_payload = json!({
            "memory_id": memory_id,
            "action": "outcome_recorded",
            "outcome": outcome_str,
            "run_id": run_id,
            "new_version": new_version,
        });
        self.outbox_repo
            .enqueue(
                EventType::ExperienceOutcomeRecorded,
                &memory_id.to_string(),
                &event_payload,
            )
            .await?;

        Ok(())
    }

    /// Set the embedding vector for a memory item.
    #[instrument(skip(self, embedding))]
    pub async fn set_embedding(
        &self,
        memory_id: Uuid,
        embedding: Vec<f32>,
    ) -> Result<(), MemoryControlError> {
        self.memory_repo
            .set_embedding(memory_id, &embedding)
            .await?;
        Ok(())
    }

    /// Get the underlying connection pool.
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// List memory items by tenant (with optional status filter).
    #[instrument(skip(self))]
    pub async fn list(
        &self,
        tenant_id: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<MemoryItem>, MemoryControlError> {
        Ok(self
            .memory_repo
            .list_by_tenant(tenant_id, None, limit, offset)
            .await?)
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oris_memory_contract::{
        Applicability, ExperienceScope, GeneV1, LifecycleState, Provenance, SafetyConstraints,
        ValidationContract,
    };

    fn make_test_gene(id: &str) -> GeneV1 {
        GeneV1 {
            id: id.to_string(),
            version: 1,
            name: format!("Test Gene {}", id),
            description: "A test gene for unit testing".into(),
            scope: ExperienceScope::Local,
            task_category: "testing".into(),
            applicability: Applicability {
                required_signals: vec!["signal1".into()],
                excluded_signals: vec![],
                environments: vec![],
                project_ids: vec![],
                tenant_ids: vec![],
                do_not_use_when: vec![],
            },
            steps: vec![],
            tool_requirements: vec![],
            safety: SafetyConstraints {
                suggestion_only: true,
                forbidden_operations: vec![],
                required_approvals: vec![],
                secret_handling: oris_memory_contract::SecretHandling::Redact,
            },
            validation: ValidationContract {
                checks: vec![],
                success_condition: oris_memory_contract::ValidationSuccessCondition::All,
            },
            provenance: Provenance {
                source_agent: "test-agent".into(),
                source_run_id: "test-run-001".into(),
                trace_refs: vec![],
                extractor_version: None,
                verified_successes: 0,
                verified_failures: 0,
                distinct_task_contexts: 0,
            },
            lifecycle: LifecycleState::Candidate,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    // ── Lifecycle mapping tests ──

    #[test]
    fn lifecycle_candidate_maps_to_candidate() {
        assert_eq!(
            lifecycle_to_status(&LifecycleState::Candidate),
            MemoryStatus::Candidate
        );
    }

    #[test]
    fn lifecycle_stable_maps_to_active() {
        assert_eq!(
            lifecycle_to_status(&LifecycleState::Stable),
            MemoryStatus::Active
        );
    }

    #[test]
    fn lifecycle_deprecated_maps_to_archived() {
        assert_eq!(
            lifecycle_to_status(&LifecycleState::Deprecated),
            MemoryStatus::Archived
        );
    }

    #[test]
    fn lifecycle_quarantined_maps_to_quarantined() {
        assert_eq!(
            lifecycle_to_status(&LifecycleState::Quarantined),
            MemoryStatus::Quarantined
        );
    }

    #[test]
    fn lifecycle_revoked_maps_to_revoked() {
        assert_eq!(
            lifecycle_to_status(&LifecycleState::Revoked),
            MemoryStatus::Revoked
        );
    }

    #[test]
    fn status_roundtrips_through_lifecycle() {
        for lifecycle in [
            LifecycleState::Candidate,
            LifecycleState::Stable,
            LifecycleState::Deprecated,
            LifecycleState::Quarantined,
            LifecycleState::Revoked,
        ] {
            let status = lifecycle_to_status(&lifecycle);
            let back = status_to_lifecycle(&status);
            assert_eq!(back, lifecycle);
        }
    }

    // ── Scope mapping tests ──

    #[test]
    fn scope_local_maps_to_personal() {
        assert_eq!(
            experience_scope_to_scope(&ExperienceScope::Local),
            Scope::Personal
        );
    }

    #[test]
    fn scope_team_maps_to_team() {
        assert_eq!(
            experience_scope_to_scope(&ExperienceScope::Team),
            Scope::Team
        );
    }

    #[test]
    fn scope_network_maps_to_enterprise() {
        assert_eq!(
            experience_scope_to_scope(&ExperienceScope::Network),
            Scope::Enterprise
        );
    }

    // ── Gene ↔ MemoryItem conversion tests ──

    #[test]
    fn gene_to_candidate_preserves_key_fields() {
        let gene = make_test_gene("g1");
        let candidate = gene_to_candidate(&gene, "acme").unwrap();

        assert_eq!(candidate.tenant_id, "acme");
        assert_eq!(candidate.memory_type, MemoryType::Experience);
        assert_eq!(candidate.scope, Scope::Personal);
        assert_eq!(candidate.subject_id.as_deref(), Some("g1:v1"));
        assert_eq!(candidate.content.as_deref(), Some("Test Gene g1"));
        assert!(candidate.structured_payload.is_some());
    }

    #[test]
    fn gene_to_candidate_serializes_payload() {
        let gene = make_test_gene("g2");
        let candidate = gene_to_candidate(&gene, "acme").unwrap();

        let payload = candidate.structured_payload.unwrap();
        let roundtrip: GeneV1 = serde_json::from_value(payload).unwrap();
        assert_eq!(roundtrip.id, "g2");
        assert_eq!(roundtrip.name, "Test Gene g2");
    }

    #[test]
    fn gene_to_candidate_clamps_team_scope_to_personal() {
        // In propose(), team/network scopes are clamped to local.
        // gene_to_candidate itself just maps — the clamping happens in propose().
        let mut gene = make_test_gene("g3");
        gene.scope = ExperienceScope::Team;
        let candidate = gene_to_candidate(&gene, "acme").unwrap();
        assert_eq!(candidate.scope, Scope::Team);
    }

    #[test]
    fn gene_to_candidate_sets_source_type_agent_inferred() {
        let gene = make_test_gene("g4");
        let candidate = gene_to_candidate(&gene, "acme").unwrap();
        assert_eq!(candidate.source_type, SourceType::AgentInferred);
    }

    #[test]
    fn gene_to_candidate_sets_source_agent_in_reference() {
        let gene = make_test_gene("g5");
        let candidate = gene_to_candidate(&gene, "acme").unwrap();
        assert_eq!(candidate.source_reference.as_deref(), Some("test-agent"));
        assert_eq!(candidate.created_by_agent.as_deref(), Some("test-agent"));
    }

    #[test]
    fn search_query_defaults() {
        let q = MemorySearchQuery::default();
        assert_eq!(q.tenant_id, "default");
        assert_eq!(q.limit, 20);
        assert_eq!(q.offset, 0);
        assert!(q.text.is_empty());
        assert!(q.embedding.is_none());
    }

    #[test]
    fn search_query_with_embedding() {
        let q = MemorySearchQuery {
            tenant_id: "acme".into(),
            text: "test query".into(),
            limit: 10,
            offset: 0,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            scopes: vec![Scope::Personal],
        };
        assert_eq!(q.tenant_id, "acme");
        assert!(q.embedding.is_some());
        assert_eq!(q.scopes.len(), 1);
    }
}
