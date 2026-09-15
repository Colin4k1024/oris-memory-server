//! Context Assembler — assembles minimal high-value context for an LLM.
//!
//! Given a [`RoutingPlan`] produced by the context router, this module
//! collects context from multiple sources (identity, canonical user, shared
//! task, agent-private memory, enterprise memory, hot context, business
//! state), resolves conflicts by authority level, marks low-confidence items,
//! and compresses the result to fit within the plan's token budget.
//!
//! # Architecture
//!
//! - §7.4 Global + Local shared model — assembly order.
//! - §8.5 Token budget mechanism (2K–8K tokens).
//!
//! # Assembly Order (§7.4)
//!
//! 1. Identity context (always included)
//! 2. Canonical User profile (if `plan.canonical_user`)
//! 3. Shared Task context (if `plan.shared_task`)
//! 4. Agent Private memory (if applicable)
//! 5. Enterprise Memory — from vector/structured/graph search
//! 6. Current Business State (from structured search)
//!
//! Sources are fetched in parallel with per-source timeouts. A source that
//! times out or errors is marked as *degraded* and the assembly continues
//! with whatever data was collected.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, instrument};
use uuid::Uuid;

use oris_memory_store::memory_types::{
    AuthorityLevel, CanonicalUserProfile, MatchType, MemoryItem, MemoryStatus, SearchParams,
    SearchResult, SharedTaskContext,
};

use crate::canonical_user::{CanonicalUserError, CanonicalUserManager};
use crate::context_router::{RequestIntent, RoutingPlan};
use crate::identity::ResolvedIdentity;
use crate::shared_task::{SharedTaskError, SharedTaskManager};
use oris_memory_store::postgres::memory_repo::MemoryRepo;
use oris_memory_store::postgres::search::SearchRepo;
use oris_memory_store::redis::hot_context::HotContextRepo;

// ──────────────────────────── Errors ────────────────────────────

/// Errors emitted by the context assembler.
#[derive(Debug, Error)]
pub enum AssemblerError {
    /// A downstream source returned an error.
    #[error("source error: {0}")]
    Source(String),

    /// Serialization failure when building payloads.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ──────────────────────────── Public Types ────────────────────────────

/// Which context source a piece of information came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextSource {
    Identity,
    CanonicalUser,
    SharedTask,
    AgentPrivate,
    EnterpriseMemory,
    HotContext,
    BusinessState,
    StructuredSearch,
    VectorSearch,
    GraphSearch,
}

/// The fully assembled context, ready for injection into an LLM prompt.
#[derive(Debug, Clone)]
pub struct AssembledContext {
    /// The assembled context text (ready for LLM prompt).
    pub context_text: String,
    /// Token count of the assembled context (estimated as `len / 4`).
    pub token_count: usize,
    /// Whether compression was applied to fit the token budget.
    pub compressed: bool,
    /// Conflict flags detected during assembly.
    pub conflict_flags: Vec<ConflictFlag>,
    /// Items marked as low-confidence (speculative/unverified).
    pub low_confidence_items: Vec<String>,
    /// Which context sources were actually used.
    pub sources_used: Vec<ContextSource>,
    /// Which sources were degraded/skipped due to timeout or error.
    pub degraded_sources: Vec<ContextSource>,
}

/// A single field-level conflict detected across sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictFlag {
    /// Field that had conflicting values (e.g. `"role"`, `"timezone"`).
    pub field: String,
    /// Explanation of which value won and why.
    pub description: String,
    /// The lower-authority values that were overridden.
    pub conflicting_values: Vec<String>,
}

// ──────────────────────────── Source Traits ────────────────────────────

/// Abstraction over the Redis hot-context cache.
#[async_trait]
pub trait HotContextSource: Send + Sync {
    /// Fetch materialized hot context as a string blob (may be JSON).
    async fn get_hot_context(
        &self,
        tenant_id: &str,
        user_id: &str,
    ) -> Result<Option<String>, AssemblerError>;
}

/// Abstraction over the PostgreSQL baseline memory store.
#[async_trait]
pub trait BaselineMemorySource: Send + Sync {
    async fn list_by_tenant(
        &self,
        tenant_id: &str,
        status: Option<MemoryStatus>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<MemoryItem>, AssemblerError>;

    async fn get_by_id(&self, id: Uuid) -> Result<Option<MemoryItem>, AssemblerError>;
}

/// Abstraction over the search repository (vector + keyword + structured).
#[async_trait]
pub trait SearchSource: Send + Sync {
    async fn search(&self, params: &SearchParams) -> Result<Vec<SearchResult>, AssemblerError>;
}

/// Abstraction over the canonical user manager.
#[async_trait]
pub trait CanonicalUserSource: Send + Sync {
    async fn get_context(
        &self,
        user_id: &str,
    ) -> Result<Option<CanonicalUserProfile>, AssemblerError>;
}

/// Abstraction over the shared task manager.
#[async_trait]
pub trait SharedTaskSource: Send + Sync {
    async fn get_context(&self, task_id: Uuid)
        -> Result<Option<SharedTaskContext>, AssemblerError>;
}

// ──────────────────────────── Internal Types ────────────────────────────

/// A single piece of context with metadata for ordering, conflict detection,
/// and low-confidence marking.
#[derive(Debug, Clone)]
struct ContextSegment {
    /// Human-readable text (may include `[待验证]` prefix if low confidence).
    text: String,
    source: ContextSource,
    /// Higher = more important (kept during compression).
    priority: u8,
    authority: AuthorityLevel,
    confidence: f32,
    status: Option<MemoryStatus>,
    /// Field key for conflict detection (e.g. `"role"`, subject_id).
    field: Option<String>,
    /// Section header label for the assembled text.
    section: &'static str,
}

/// Outcome of a parallel source fetch.
struct FetchOutcome<T> {
    data: Option<T>,
    degraded: bool,
}

impl<T> FetchOutcome<T> {
    fn ok(data: T) -> Self {
        Self {
            data: Some(data),
            degraded: false,
        }
    }

    fn empty() -> Self {
        Self {
            data: None,
            degraded: false,
        }
    }

    fn degraded() -> Self {
        Self {
            data: None,
            degraded: true,
        }
    }
}

// ──────────────────────────── Constants ────────────────────────────

/// Confidence at or below this value is considered "low" and gets the
/// `[待验证]` prefix.
const LOW_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Default token budget when the plan budget is zero or unset.
const DEFAULT_TOKEN_BUDGET: usize = 4096;

/// Chars-per-token approximation (4 chars ≈ 1 token).
const CHARS_PER_TOKEN: usize = 4;

// ──────────────────────────── ContextAssembler ────────────────────────────

/// Assembles minimal high-value context for an LLM based on a [`RoutingPlan`].
///
/// All source dependencies are behind trait objects so unit tests can inject
/// in-memory stubs without a live PostgreSQL or Redis instance. Use the
/// `with_*` builder methods to attach sources; unattached sources are treated
/// as degraded at assembly time.
pub struct ContextAssembler {
    hot_context: Option<Arc<dyn HotContextSource>>,
    memory: Option<Arc<dyn BaselineMemorySource>>,
    search: Option<Arc<dyn SearchSource>>,
    canonical_user: Option<Arc<dyn CanonicalUserSource>>,
    shared_task: Option<Arc<dyn SharedTaskSource>>,
    default_token_budget: usize,
}

impl Default for ContextAssembler {
    fn default() -> Self {
        Self {
            hot_context: None,
            memory: None,
            search: None,
            canonical_user: None,
            shared_task: None,
            default_token_budget: DEFAULT_TOKEN_BUDGET,
        }
    }
}

impl ContextAssembler {
    /// Create a new assembler with no sources attached.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the default token budget (used when `plan.token_budget` is 0).
    pub fn with_default_token_budget(mut self, budget: usize) -> Self {
        self.default_token_budget = budget;
        self
    }

    /// Attach a hot-context source.
    pub fn with_hot_context(mut self, src: Arc<dyn HotContextSource>) -> Self {
        self.hot_context = Some(src);
        self
    }

    /// Attach a baseline memory source.
    pub fn with_memory(mut self, src: Arc<dyn BaselineMemorySource>) -> Self {
        self.memory = Some(src);
        self
    }

    /// Attach a search source.
    pub fn with_search(mut self, src: Arc<dyn SearchSource>) -> Self {
        self.search = Some(src);
        self
    }

    /// Attach a canonical-user source.
    pub fn with_canonical_user(mut self, src: Arc<dyn CanonicalUserSource>) -> Self {
        self.canonical_user = Some(src);
        self
    }

    /// Attach a shared-task source.
    pub fn with_shared_task(mut self, src: Arc<dyn SharedTaskSource>) -> Self {
        self.shared_task = Some(src);
        self
    }

    /// Assemble context for an LLM based on the routing plan.
    ///
    /// The `task_id` parameter is required when `plan.shared_task` is `true`;
    /// pass `None` otherwise. If `shared_task` is active but `task_id` is
    /// `None`, the shared-task source is marked as degraded.
    ///
    /// This method never returns an error for individual source failures —
    /// those are recorded in `degraded_sources`. It only errors on
    /// unrecoverable internal failures.
    #[instrument(skip(self, plan, identity, query), fields(intent = ?plan.intent))]
    pub async fn assemble(
        &self,
        plan: &RoutingPlan,
        identity: &ResolvedIdentity,
        query: &str,
        task_id: Option<Uuid>,
    ) -> Result<AssembledContext, AssemblerError> {
        let budget = resolve_budget(plan.token_budget, self.default_token_budget);
        let latency = Duration::from_millis(plan.latency_budget_ms);
        let tenant = identity.organization_id.as_str();
        let user = identity.user_id.as_str();

        debug!(
            intent = ?plan.intent,
            budget,
            latency_ms = plan.latency_budget_ms,
            "assembling context"
        );

        // ── Parallel source collection ──

        let hot_fut = fetch_hot(
            self.hot_context.as_deref(),
            plan.hot_context,
            tenant,
            user,
            latency,
        );
        let canon_fut = fetch_canonical(
            self.canonical_user.as_deref(),
            plan.canonical_user,
            user,
            latency,
        );
        let task_fut = fetch_shared_task(
            self.shared_task.as_deref(),
            plan.shared_task,
            task_id,
            latency,
        );
        let agent_fut = fetch_agent_private(
            self.memory.as_deref(),
            plan.intent,
            identity.delegated_agent_id.as_deref(),
            tenant,
            latency,
        );
        let enterprise_fut = fetch_enterprise(
            self.search.as_deref(),
            plan.vector_search,
            plan.graph_search,
            tenant,
            user,
            query,
            latency,
        );
        let business_fut = fetch_business_state(
            self.search.as_deref(),
            plan.structured_search,
            tenant,
            query,
            latency,
        );

        let (hot, canon, task, agent, enterprise, business) = tokio::join!(
            hot_fut,
            canon_fut,
            task_fut,
            agent_fut,
            enterprise_fut,
            business_fut
        );

        // ── Build segments in assembly order ──

        let mut segments: Vec<ContextSegment> = Vec::new();
        let mut sources_used: Vec<ContextSource> = Vec::new();
        let mut degraded_sources: Vec<ContextSource> = Vec::new();

        // 1. Identity — always included.
        segments.extend(build_identity_segments(identity));
        sources_used.push(ContextSource::Identity);

        // 2. Hot context.
        if let Some(data) = &hot.data {
            if !data.is_empty() {
                segments.push(ContextSegment {
                    text: data.clone(),
                    source: ContextSource::HotContext,
                    priority: 80,
                    authority: AuthorityLevel::L2Verified,
                    confidence: 0.85,
                    status: None,
                    field: None,
                    section: "Hot Context",
                });
                sources_used.push(ContextSource::HotContext);
            }
        }
        if hot.degraded {
            degraded_sources.push(ContextSource::HotContext);
        }

        // 3. Canonical user.
        if let Some(profile) = &canon.data {
            let segs = build_canonical_user_segments(profile);
            if !segs.is_empty() {
                sources_used.push(ContextSource::CanonicalUser);
            }
            segments.extend(segs);
        }
        if canon.degraded {
            degraded_sources.push(ContextSource::CanonicalUser);
        }

        // 4. Shared task.
        if let Some(task_ctx) = &task.data {
            let segs = build_shared_task_segments(task_ctx);
            if !segs.is_empty() {
                sources_used.push(ContextSource::SharedTask);
            }
            segments.extend(segs);
        }
        if task.degraded {
            degraded_sources.push(ContextSource::SharedTask);
        }

        // 5. Agent private memory.
        if let Some(items) = &agent.data {
            if !items.is_empty() {
                sources_used.push(ContextSource::AgentPrivate);
            }
            segments.extend(build_memory_segments(
                items,
                ContextSource::AgentPrivate,
                "Agent Private Memory",
            ));
        }
        if agent.degraded {
            degraded_sources.push(ContextSource::AgentPrivate);
        }

        // 6. Enterprise memory (vector + graph search).
        if let Some(results) = &enterprise.data {
            if !results.is_empty() {
                let segs = build_search_result_segments(
                    results,
                    ContextSource::EnterpriseMemory,
                    "Enterprise Memory",
                );
                if !segs.is_empty() {
                    sources_used.push(ContextSource::EnterpriseMemory);
                }
                segments.extend(segs);
            }
        }
        if enterprise.degraded {
            degraded_sources.push(ContextSource::EnterpriseMemory);
        }

        // 7. Business state (structured search).
        if let Some(results) = &business.data {
            if !results.is_empty() {
                let segs = build_search_result_segments(
                    results,
                    ContextSource::BusinessState,
                    "Business State",
                );
                if !segs.is_empty() {
                    sources_used.push(ContextSource::BusinessState);
                }
                segments.extend(segs);
            }
        }
        if business.degraded {
            degraded_sources.push(ContextSource::BusinessState);
        }

        // ── Conflict detection (before low-confidence prefix) ──
        let conflict_flags = detect_conflicts(&segments);

        // ── Low-confidence marking ──
        let mut low_confidence_items = Vec::new();
        for seg in &mut segments {
            let is_low = seg.confidence < LOW_CONFIDENCE_THRESHOLD
                || seg.status == Some(MemoryStatus::Candidate);
            if is_low {
                seg.text = format!("[待验证] {}", seg.text);
                low_confidence_items.push(seg.text.clone());
            }
        }

        // ── Token budget & compression ──
        let (context_text, token_count, compressed) = assemble_within_budget(&segments, budget);

        debug!(
            segments = segments.len(),
            token_count,
            compressed,
            conflicts = conflict_flags.len(),
            low_confidence = low_confidence_items.len(),
            degraded = degraded_sources.len(),
            "context assembled"
        );

        Ok(AssembledContext {
            context_text,
            token_count,
            compressed,
            conflict_flags,
            low_confidence_items,
            sources_used,
            degraded_sources,
        })
    }
}

// ──────────────────────────── Parallel Fetch Helpers ────────────────────────────

async fn fetch_hot(
    src: Option<&dyn HotContextSource>,
    enabled: bool,
    tenant: &str,
    user: &str,
    latency: Duration,
) -> FetchOutcome<String> {
    if !enabled {
        return FetchOutcome::empty();
    }
    let Some(src) = src else {
        return FetchOutcome::degraded();
    };
    match tokio::time::timeout(latency, src.get_hot_context(tenant, user)).await {
        Ok(Ok(Some(data))) => FetchOutcome::ok(data),
        Ok(Ok(None)) => FetchOutcome::empty(),
        Ok(Err(e)) => {
            debug!(error = %e, "hot context fetch failed");
            FetchOutcome::degraded()
        }
        Err(_) => {
            debug!("hot context fetch timed out");
            FetchOutcome::degraded()
        }
    }
}

async fn fetch_canonical(
    src: Option<&dyn CanonicalUserSource>,
    enabled: bool,
    user: &str,
    latency: Duration,
) -> FetchOutcome<CanonicalUserProfile> {
    if !enabled {
        return FetchOutcome::empty();
    }
    let Some(src) = src else {
        return FetchOutcome::degraded();
    };
    match tokio::time::timeout(latency, src.get_context(user)).await {
        Ok(Ok(Some(p))) => FetchOutcome::ok(p),
        Ok(Ok(None)) => FetchOutcome::empty(),
        Ok(Err(e)) => {
            debug!(error = %e, "canonical user fetch failed");
            FetchOutcome::degraded()
        }
        Err(_) => {
            debug!("canonical user fetch timed out");
            FetchOutcome::degraded()
        }
    }
}

async fn fetch_shared_task(
    src: Option<&dyn SharedTaskSource>,
    enabled: bool,
    task_id: Option<Uuid>,
    latency: Duration,
) -> FetchOutcome<SharedTaskContext> {
    if !enabled {
        return FetchOutcome::empty();
    }
    let Some(task_id) = task_id else {
        return FetchOutcome::degraded();
    };
    let Some(src) = src else {
        return FetchOutcome::degraded();
    };
    match tokio::time::timeout(latency, src.get_context(task_id)).await {
        Ok(Ok(Some(t))) => FetchOutcome::ok(t),
        Ok(Ok(None)) => FetchOutcome::empty(),
        Ok(Err(e)) => {
            debug!(error = %e, "shared task fetch failed");
            FetchOutcome::degraded()
        }
        Err(_) => {
            debug!("shared task fetch timed out");
            FetchOutcome::degraded()
        }
    }
}

async fn fetch_agent_private(
    src: Option<&dyn BaselineMemorySource>,
    intent: RequestIntent,
    agent_id: Option<&str>,
    tenant: &str,
    latency: Duration,
) -> FetchOutcome<Vec<MemoryItem>> {
    // Agent private memory is applicable when an agent is delegated and the
    // intent is above casual chat.
    let agent_id = match agent_id {
        Some(a) if !a.is_empty() => a,
        _ => return FetchOutcome::empty(),
    };
    if intent == RequestIntent::Chat {
        return FetchOutcome::empty();
    }
    let Some(src) = src else {
        return FetchOutcome::degraded();
    };
    let params = (tenant, agent_id);
    match tokio::time::timeout(latency, async {
        let items = src
            .list_by_tenant(params.0, Some(MemoryStatus::Active), 50, 0)
            .await?;
        let filtered: Vec<MemoryItem> = items
            .into_iter()
            .filter(|i| i.subject_id.as_deref() == Some(params.1))
            .collect();
        Ok::<_, AssemblerError>(filtered)
    })
    .await
    {
        Ok(Ok(items)) => FetchOutcome::ok(items),
        Ok(Err(e)) => {
            debug!(error = %e, "agent private fetch failed");
            FetchOutcome::degraded()
        }
        Err(_) => {
            debug!("agent private fetch timed out");
            FetchOutcome::degraded()
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn fetch_enterprise(
    src: Option<&dyn SearchSource>,
    vector_enabled: bool,
    graph_enabled: bool,
    tenant: &str,
    user: &str,
    query: &str,
    latency: Duration,
) -> FetchOutcome<Vec<SearchResult>> {
    if !vector_enabled && !graph_enabled {
        return FetchOutcome::empty();
    }
    let Some(src) = src else {
        return FetchOutcome::degraded();
    };
    let mut params = SearchParams::new(tenant);
    params.query_text = Some(query.to_string());
    params.subject_id = Some(user.to_string());
    params.limit = 20;
    match tokio::time::timeout(latency, src.search(&params)).await {
        Ok(Ok(results)) => FetchOutcome::ok(results),
        Ok(Err(e)) => {
            debug!(error = %e, "enterprise search failed");
            FetchOutcome::degraded()
        }
        Err(_) => {
            debug!("enterprise search timed out");
            FetchOutcome::degraded()
        }
    }
}

async fn fetch_business_state(
    src: Option<&dyn SearchSource>,
    enabled: bool,
    tenant: &str,
    query: &str,
    latency: Duration,
) -> FetchOutcome<Vec<SearchResult>> {
    if !enabled {
        return FetchOutcome::empty();
    }
    let Some(src) = src else {
        return FetchOutcome::degraded();
    };
    let mut params = SearchParams::new(tenant);
    params.query_text = Some(query.to_string());
    params.limit = 10;
    match tokio::time::timeout(latency, src.search(&params)).await {
        Ok(Ok(results)) => {
            // Only keep structured matches for business state.
            let filtered: Vec<SearchResult> = results
                .into_iter()
                .filter(|r| {
                    r.matched_by == MatchType::Structured || r.matched_by == MatchType::Hybrid
                })
                .collect();
            FetchOutcome::ok(filtered)
        }
        Ok(Err(e)) => {
            debug!(error = %e, "business state search failed");
            FetchOutcome::degraded()
        }
        Err(_) => {
            debug!("business state search timed out");
            FetchOutcome::degraded()
        }
    }
}

// ──────────────────────────── Segment Builders ────────────────────────────

fn build_identity_segments(identity: &ResolvedIdentity) -> Vec<ContextSegment> {
    let mut segments = vec![ContextSegment {
        text: format!(
            "User: {}, Tenant: {}, Roles: {}",
            identity.user_id,
            identity.organization_id,
            identity.roles.join(", ")
        ),
        source: ContextSource::Identity,
        priority: 100,
        authority: AuthorityLevel::L1Authoritative,
        confidence: 1.0,
        status: None,
        field: Some("identity".to_string()),
        section: "Identity",
    }];

    if let Some(ref factory) = identity.factory_id {
        segments.push(ContextSegment {
            text: format!("Factory: {}", factory),
            source: ContextSource::Identity,
            priority: 100,
            authority: AuthorityLevel::L1Authoritative,
            confidence: 1.0,
            status: None,
            field: Some("factory_id".to_string()),
            section: "Identity",
        });
    }

    if let Some(ref agent) = identity.delegated_agent_id {
        segments.push(ContextSegment {
            text: format!("Delegated Agent: {}", agent),
            source: ContextSource::Identity,
            priority: 100,
            authority: AuthorityLevel::L1Authoritative,
            confidence: 1.0,
            status: None,
            field: Some("delegated_agent".to_string()),
            section: "Identity",
        });
    }

    segments
}

fn build_canonical_user_segments(profile: &CanonicalUserProfile) -> Vec<ContextSegment> {
    let auth = profile.authority_level;
    let mut segments = Vec::new();

    if let Some(ref role) = profile.role {
        segments.push(ContextSegment {
            text: format!("Role: {}", role),
            source: ContextSource::CanonicalUser,
            priority: 90,
            authority: auth,
            confidence: 1.0,
            status: None,
            field: Some("role".to_string()),
            section: "Canonical User",
        });
    }

    if let Some(ref position) = profile.position {
        segments.push(ContextSegment {
            text: format!("Position: {}", position),
            source: ContextSource::CanonicalUser,
            priority: 88,
            authority: auth,
            confidence: 1.0,
            status: None,
            field: Some("position".to_string()),
            section: "Canonical User",
        });
    }

    if let Some(ref language) = profile.language {
        segments.push(ContextSegment {
            text: format!("Language: {}", language),
            source: ContextSource::CanonicalUser,
            priority: 82,
            authority: auth,
            confidence: 1.0,
            status: None,
            field: Some("language".to_string()),
            section: "Canonical User",
        });
    }

    if let Some(ref timezone) = profile.timezone {
        segments.push(ContextSegment {
            text: format!("Timezone: {}", timezone),
            source: ContextSource::CanonicalUser,
            priority: 82,
            authority: auth,
            confidence: 1.0,
            status: None,
            field: Some("timezone".to_string()),
            section: "Canonical User",
        });
    }

    if !profile.preferences.is_null() {
        segments.push(ContextSegment {
            text: format!("Preferences: {}", profile.preferences),
            source: ContextSource::CanonicalUser,
            priority: 70,
            authority: auth,
            confidence: 1.0,
            status: None,
            field: Some("preferences".to_string()),
            section: "Canonical User",
        });
    }

    segments
}

fn build_shared_task_segments(task: &SharedTaskContext) -> Vec<ContextSegment> {
    let mut segments = vec![ContextSegment {
        text: format!("Task Goal: {}", task.goal),
        source: ContextSource::SharedTask,
        priority: 88,
        authority: AuthorityLevel::L1Authoritative,
        confidence: 1.0,
        status: None,
        field: Some("task_goal".to_string()),
        section: "Shared Task",
    }];

    if let Some(ref owner) = task.current_owner_agent {
        segments.push(ContextSegment {
            text: format!("Current Owner: {}", owner),
            source: ContextSource::SharedTask,
            priority: 86,
            authority: AuthorityLevel::L1Authoritative,
            confidence: 1.0,
            status: None,
            field: Some("task_owner".to_string()),
            section: "Shared Task",
        });
    }

    if !task.current_findings.is_empty() {
        let findings: Vec<String> = task
            .current_findings
            .iter()
            .map(|f| f.to_string())
            .collect();
        segments.push(ContextSegment {
            text: format!("Findings: {}", findings.join("; ")),
            source: ContextSource::SharedTask,
            priority: 78,
            authority: AuthorityLevel::L2Verified,
            confidence: 0.9,
            status: None,
            field: Some("task_findings".to_string()),
            section: "Shared Task",
        });
    }

    if !task.decisions.is_empty() {
        let decisions: Vec<String> = task.decisions.iter().map(|d| d.to_string()).collect();
        segments.push(ContextSegment {
            text: format!("Decisions: {}", decisions.join("; ")),
            source: ContextSource::SharedTask,
            priority: 76,
            authority: AuthorityLevel::L2Verified,
            confidence: 0.9,
            status: None,
            field: Some("task_decisions".to_string()),
            section: "Shared Task",
        });
    }

    segments
}

fn build_memory_segments(
    items: &[MemoryItem],
    source: ContextSource,
    section: &'static str,
) -> Vec<ContextSegment> {
    items
        .iter()
        .map(|item| memory_item_to_segment(item, source, section))
        .collect()
}

fn build_search_result_segments(
    results: &[SearchResult],
    source: ContextSource,
    section: &'static str,
) -> Vec<ContextSegment> {
    results
        .iter()
        .map(|r| memory_item_to_segment(&r.item, source, section))
        .collect()
}

fn memory_item_to_segment(
    item: &MemoryItem,
    source: ContextSource,
    section: &'static str,
) -> ContextSegment {
    let text = item.content.clone().unwrap_or_else(|| {
        item.structured_payload
            .as_ref()
            .map(|p| p.to_string())
            .unwrap_or_default()
    });

    ContextSegment {
        text,
        source,
        priority: match source {
            ContextSource::AgentPrivate => 75,
            ContextSource::EnterpriseMemory => 60,
            ContextSource::BusinessState => 55,
            _ => 60,
        },
        authority: item.authority_level,
        confidence: item.confidence,
        status: Some(item.status),
        field: item.subject_id.clone(),
        section,
    }
}

// ──────────────────────────── Conflict Detection ────────────────────────────

/// Detects conflicts between items from different sources that refer to the
/// same field. Higher authority wins; lower-authority values are flagged.
fn detect_conflicts(segments: &[ContextSegment]) -> Vec<ConflictFlag> {
    let mut groups: HashMap<String, Vec<&ContextSegment>> = HashMap::new();
    for seg in segments {
        if let Some(ref field) = seg.field {
            groups.entry(field.clone()).or_default().push(seg);
        }
    }

    let mut flags = Vec::new();
    for (field, group) in &groups {
        if group.len() < 2 {
            continue;
        }

        // Conflicts only between *different* sources.
        let unique_sources: std::collections::HashSet<ContextSource> =
            group.iter().map(|s| s.source).collect();
        if unique_sources.len() < 2 {
            continue;
        }

        // Sort by authority desc (highest first).
        let mut sorted = group.clone();
        sorted.sort_by_key(|s| std::cmp::Reverse(s.authority.rank()));

        let winner = &sorted[0];

        // Lower-authority items with *different* text are conflicting.
        let conflicting: Vec<String> = sorted[1..]
            .iter()
            .filter(|s| s.text != winner.text)
            .map(|s| s.text.clone())
            .collect();

        if !conflicting.is_empty() {
            flags.push(ConflictFlag {
                field: field.clone(),
                description: format!(
                    "Authority {} overrides {} lower-authority value(s)",
                    winner.authority.as_str(),
                    conflicting.len()
                ),
                conflicting_values: conflicting,
            });
        }
    }

    flags
}

// ──────────────────────────── Token Budget & Compression ────────────────────────────

/// Estimate token count: `text.len() / 4`.
fn estimate_tokens(text: &str) -> usize {
    text.len() / CHARS_PER_TOKEN
}

/// Resolve the effective token budget.
fn resolve_budget(plan_budget: usize, default_budget: usize) -> usize {
    if plan_budget == 0 {
        default_budget
    } else {
        plan_budget
    }
}

/// Assemble segments into text within the token budget.
///
/// Segments are sorted by priority (desc) and authority (desc), then greedily
/// added until the budget is reached. Any segments that don't fit are dropped
/// and `compressed = true` is returned.
fn assemble_within_budget(segments: &[ContextSegment], budget: usize) -> (String, usize, bool) {
    if segments.is_empty() {
        return (String::new(), 0, false);
    }

    // Sort indices by priority desc, then authority desc, then confidence desc.
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by(|&a, &b| {
        segments[b]
            .priority
            .cmp(&segments[a].priority)
            .then_with(|| {
                segments[b]
                    .authority
                    .rank()
                    .cmp(&segments[a].authority.rank())
            })
            .then_with(|| {
                segments[b]
                    .confidence
                    .partial_cmp(&segments[a].confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Convert token budget to a char budget (4 chars per token) so we can
    // account for section headers and newlines that are not part of any
    // individual segment's text.
    let char_budget = budget.saturating_mul(CHARS_PER_TOKEN);

    // Greedily select segments, accounting for section headers + newlines.
    let mut selected: Vec<usize> = Vec::new();
    let mut total_chars: usize = 0;
    let mut compressed = false;
    let mut seen_sections: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for &idx in &order {
        let seg = &segments[idx];

        // Header chars if this is the first selected segment in this section.
        let header_chars = if !seen_sections.contains(seg.section) {
            // "## Section\n" plus a possible inter-section separator newline.
            seg.section.len() + 6
        } else {
            0
        };
        // Segment text + trailing newline.
        let seg_chars = seg.text.len() + 1;

        if total_chars + header_chars + seg_chars > char_budget {
            compressed = true;
            continue;
        }

        total_chars += header_chars + seg_chars;
        seen_sections.insert(seg.section);
        selected.push(idx);
    }

    // Re-sort selected by original order (assembly order -> section order).
    selected.sort();

    // Build text grouped by section.
    let mut text = String::new();
    let mut current_section = "";
    for &idx in &selected {
        let seg = &segments[idx];
        if seg.section != current_section {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&format!("## {}\n", seg.section));
            current_section = seg.section;
        }
        text.push_str(&seg.text);
        text.push('\n');
    }

    let token_count = estimate_tokens(&text);
    (text, token_count, compressed)
}

// ──────────────────────────── Production Adapters ────────────────────────────

/// Production adapter wrapping [`HotContextRepo`].
pub struct HotContextAdapter {
    repo: HotContextRepo,
}

impl From<HotContextRepo> for HotContextAdapter {
    fn from(repo: HotContextRepo) -> Self {
        Self { repo }
    }
}

#[async_trait]
impl HotContextSource for HotContextAdapter {
    async fn get_hot_context(
        &self,
        tenant_id: &str,
        user_id: &str,
    ) -> Result<Option<String>, AssemblerError> {
        let data = self
            .repo
            .get_user_context(user_id, tenant_id)
            .await
            .map_err(|e| AssemblerError::Source(e.to_string()))?;
        Ok(data.map(|bytes| String::from_utf8_lossy(&bytes).to_string()))
    }
}

/// Production adapter wrapping [`MemoryRepo`].
pub struct MemoryRepoAdapter {
    repo: MemoryRepo,
}

impl From<MemoryRepo> for MemoryRepoAdapter {
    fn from(repo: MemoryRepo) -> Self {
        Self { repo }
    }
}

#[async_trait]
impl BaselineMemorySource for MemoryRepoAdapter {
    async fn list_by_tenant(
        &self,
        tenant_id: &str,
        status: Option<MemoryStatus>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<MemoryItem>, AssemblerError> {
        self.repo
            .list_by_tenant(tenant_id, status, limit, offset)
            .await
            .map_err(|e| AssemblerError::Source(e.to_string()))
    }

    async fn get_by_id(&self, id: Uuid) -> Result<Option<MemoryItem>, AssemblerError> {
        self.repo
            .get_by_id(id)
            .await
            .map_err(|e| AssemblerError::Source(e.to_string()))
    }
}

/// Production adapter wrapping [`SearchRepo`].
pub struct SearchRepoAdapter {
    repo: SearchRepo,
}

impl From<SearchRepo> for SearchRepoAdapter {
    fn from(repo: SearchRepo) -> Self {
        Self { repo }
    }
}

#[async_trait]
impl SearchSource for SearchRepoAdapter {
    async fn search(&self, params: &SearchParams) -> Result<Vec<SearchResult>, AssemblerError> {
        self.repo
            .hybrid_search(params)
            .await
            .map_err(|e| AssemblerError::Source(e.to_string()))
    }
}

/// Production adapter wrapping [`CanonicalUserManager`].
pub struct CanonicalUserAdapter {
    mgr: CanonicalUserManager,
}

impl From<CanonicalUserManager> for CanonicalUserAdapter {
    fn from(mgr: CanonicalUserManager) -> Self {
        Self { mgr }
    }
}

#[async_trait]
impl CanonicalUserSource for CanonicalUserAdapter {
    async fn get_context(
        &self,
        user_id: &str,
    ) -> Result<Option<CanonicalUserProfile>, AssemblerError> {
        self.mgr
            .get_context(user_id)
            .await
            .map_err(|e: CanonicalUserError| AssemblerError::Source(e.to_string()))
    }
}

/// Production adapter wrapping [`SharedTaskManager`].
pub struct SharedTaskAdapter {
    mgr: SharedTaskManager,
}

impl From<SharedTaskManager> for SharedTaskAdapter {
    fn from(mgr: SharedTaskManager) -> Self {
        Self { mgr }
    }
}

#[async_trait]
impl SharedTaskSource for SharedTaskAdapter {
    async fn get_context(
        &self,
        task_id: Uuid,
    ) -> Result<Option<SharedTaskContext>, AssemblerError> {
        self.mgr
            .get_context(task_id)
            .await
            .map_err(|e: SharedTaskError| AssemblerError::Source(e.to_string()))
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use oris_memory_store::memory_types::{
        AuthorityLevel, CanonicalUserProfile, MatchType, MemoryItem, MemoryStatus, MemoryType,
        PrivacyClass, Scope, SearchParams, SearchResult, SharedTaskContext, SourceType,
    };
    use serde_json::json;
    use uuid::Uuid;

    // ── Mock Implementations ──

    struct MockHotContext {
        data: Option<String>,
        delay_ms: u64,
    }

    #[async_trait]
    impl HotContextSource for MockHotContext {
        async fn get_hot_context(
            &self,
            _tenant: &str,
            _user: &str,
        ) -> Result<Option<String>, AssemblerError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            Ok(self.data.clone())
        }
    }

    struct MockMemorySource {
        items: Vec<MemoryItem>,
        delay_ms: u64,
    }

    #[async_trait]
    impl BaselineMemorySource for MockMemorySource {
        async fn list_by_tenant(
            &self,
            tenant_id: &str,
            _status: Option<MemoryStatus>,
            _limit: i64,
            _offset: i64,
        ) -> Result<Vec<MemoryItem>, AssemblerError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            Ok(self
                .items
                .iter()
                .filter(|i| i.tenant_id == tenant_id)
                .cloned()
                .collect())
        }

        async fn get_by_id(&self, id: Uuid) -> Result<Option<MemoryItem>, AssemblerError> {
            Ok(self.items.iter().find(|i| i.memory_id == id).cloned())
        }
    }

    struct MockSearchSource {
        results: Vec<SearchResult>,
        delay_ms: u64,
    }

    #[async_trait]
    impl SearchSource for MockSearchSource {
        async fn search(
            &self,
            _params: &SearchParams,
        ) -> Result<Vec<SearchResult>, AssemblerError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            Ok(self.results.clone())
        }
    }

    struct MockCanonicalUser {
        profile: Option<CanonicalUserProfile>,
        delay_ms: u64,
    }

    #[async_trait]
    impl CanonicalUserSource for MockCanonicalUser {
        async fn get_context(
            &self,
            _user_id: &str,
        ) -> Result<Option<CanonicalUserProfile>, AssemblerError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            Ok(self.profile.clone())
        }
    }

    struct MockSharedTask {
        task: Option<SharedTaskContext>,
        delay_ms: u64,
    }

    #[async_trait]
    impl SharedTaskSource for MockSharedTask {
        async fn get_context(
            &self,
            _task_id: Uuid,
        ) -> Result<Option<SharedTaskContext>, AssemblerError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            Ok(self.task.clone())
        }
    }

    // ── Test Helpers ──

    fn test_identity() -> ResolvedIdentity {
        ResolvedIdentity {
            user_id: "user-1".into(),
            organization_id: "acme".into(),
            factory_id: None,
            roles: vec!["operator".into()],
            delegated_agent_id: None,
            permissions: crate::identity::PermissionSet::new(vec![], vec![]),
            purpose: "test".into(),
            trace_id: "trace-1".into(),
            resolved_at: Utc::now(),
        }
    }

    fn identity_with_agent() -> ResolvedIdentity {
        let mut id = test_identity();
        id.delegated_agent_id = Some("agent-007".into());
        id
    }

    fn make_profile(authority: AuthorityLevel, source: SourceType) -> CanonicalUserProfile {
        CanonicalUserProfile {
            user_id: "user-1".into(),
            organization_id: "acme".into(),
            factory_id: None,
            identity_links: vec![],
            role: Some("engineer".into()),
            position: Some("Senior Dev".into()),
            language: Some("en".into()),
            timezone: Some("UTC".into()),
            preferences: json!({"theme": "dark"}),
            explicit_preferences: json!({}),
            inferred_preferences: json!({}),
            common_entities: vec![],
            active_projects: vec![],
            consent_scope: json!(null),
            privacy_class: PrivacyClass::Internal,
            source,
            authority_level: authority,
            version: 1,
            valid_from: None,
            valid_to: None,
            last_verified_at: None,
            updated_at: Utc::now(),
        }
    }

    fn make_memory_item(
        content: &str,
        authority: AuthorityLevel,
        confidence: f32,
        status: MemoryStatus,
        subject_id: Option<&str>,
    ) -> MemoryItem {
        MemoryItem {
            memory_id: Uuid::new_v4(),
            tenant_id: "acme".into(),
            memory_type: MemoryType::Semantic,
            scope: Scope::Agent,
            subject_type: Some("agent".into()),
            subject_id: subject_id.map(String::from),
            entity_refs: vec![],
            content: Some(content.into()),
            structured_payload: None,
            embedding: None,
            source_type: SourceType::AgentInferred,
            source_reference: None,
            evidence_refs: vec![],
            confidence,
            authority_level: authority,
            importance: 0.5,
            observed_at: None,
            valid_from: None,
            valid_to: None,
            privacy_class: PrivacyClass::Internal,
            acl: json!({}),
            retention_policy: None,
            status,
            version: 1,
            derived_from: vec![],
            created_by_user: None,
            created_by_agent: Some("agent-007".into()),
            last_verified_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn make_search_result(item: MemoryItem, matched_by: MatchType) -> SearchResult {
        SearchResult {
            item,
            score: 0.9,
            matched_by,
        }
    }

    fn make_task_context() -> SharedTaskContext {
        SharedTaskContext {
            task_id: Uuid::new_v4(),
            parent_task_id: None,
            initiator_user_id: "user-1".into(),
            organization_scope: "acme".into(),
            goal: "Complete order fulfillment".into(),
            constraints: vec![],
            success_criteria: vec![],
            entities: vec![],
            business_refs: vec![],
            current_findings: vec![json!("finding A"), json!("finding B")],
            evidence_refs: vec![],
            decisions: vec![json!("go with plan B")],
            assumptions: vec![],
            completed_steps: vec![],
            pending_steps: vec![],
            current_owner_agent: Some("agent-007".into()),
            participant_agents: vec![],
            artifact_refs: vec![],
            source_system_refs: vec![],
            status: "active".into(),
            version: 1,
            expires_at: None,
            acl: json!({}),
            privacy_class: PrivacyClass::Internal,
            audit_ref: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn chat_plan() -> RoutingPlan {
        RoutingPlan {
            intent: RequestIntent::Chat,
            hot_context: true,
            canonical_user: false,
            shared_task: false,
            structured_search: false,
            vector_search: false,
            graph_search: false,
            sot_verification: false,
            latency_budget_ms: 5000,
            token_budget: 2048,
        }
    }

    fn deep_research_plan() -> RoutingPlan {
        RoutingPlan {
            intent: RequestIntent::DeepResearch,
            hot_context: true,
            canonical_user: true,
            shared_task: true,
            structured_search: true,
            vector_search: true,
            graph_search: true,
            sot_verification: true,
            latency_budget_ms: 10000,
            token_budget: 8192,
        }
    }

    fn plan_with(
        intent: RequestIntent,
        hot: bool,
        canon: bool,
        task: bool,
        structured: bool,
        vector: bool,
        graph: bool,
    ) -> RoutingPlan {
        RoutingPlan {
            intent,
            hot_context: hot,
            canonical_user: canon,
            shared_task: task,
            structured_search: structured,
            vector_search: vector,
            graph_search: graph,
            sot_verification: false,
            latency_budget_ms: 10000,
            token_budget: 4096,
        }
    }

    // ── Tests ──

    #[tokio::test]
    async fn basic_assembly_identity_only() {
        let assembler = ContextAssembler::new();
        let plan = chat_plan();
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "hello", None)
            .await
            .unwrap();

        assert!(!ctx.context_text.is_empty());
        assert!(ctx.sources_used.contains(&ContextSource::Identity));
        assert!(!ctx.compressed);
        assert!(ctx.conflict_flags.is_empty());
    }

    #[tokio::test]
    async fn basic_assembly_with_canonical_user() {
        let assembler = ContextAssembler::new().with_canonical_user(Arc::new(MockCanonicalUser {
            profile: Some(make_profile(
                AuthorityLevel::L1Authoritative,
                SourceType::Iam,
            )),
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::PersonalTask,
            true,
            true,
            false,
            false,
            false,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "my task", None)
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::CanonicalUser));
        assert!(ctx.context_text.contains("Role: engineer"));
        assert!(ctx.context_text.contains("## Canonical User"));
    }

    #[tokio::test]
    async fn basic_assembly_with_shared_task() {
        let assembler = ContextAssembler::new().with_shared_task(Arc::new(MockSharedTask {
            task: Some(make_task_context()),
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::CrossAgentTask,
            true,
            true,
            true,
            false,
            false,
            false,
        );
        let identity = test_identity();
        let task_id = Uuid::new_v4();

        let ctx = assembler
            .assemble(&plan, &identity, "task query", Some(task_id))
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::SharedTask));
        assert!(ctx
            .context_text
            .contains("Task Goal: Complete order fulfillment"));
        assert!(ctx.context_text.contains("## Shared Task"));
    }

    #[tokio::test]
    async fn token_budget_enforcement() {
        let big_content = "x".repeat(10_000); // ~2500 tokens
        let assembler = ContextAssembler::new().with_hot_context(Arc::new(MockHotContext {
            data: Some(big_content),
            delay_ms: 0,
        }));
        let mut plan = chat_plan();
        plan.token_budget = 100; // very small
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "hi", None)
            .await
            .unwrap();

        assert!(
            ctx.token_count <= 100,
            "token_count {} should be <= 100",
            ctx.token_count
        );
    }

    #[tokio::test]
    async fn compression_triggered() {
        // Many memory items exceeding the token budget.
        let items: Vec<MemoryItem> = (0..50)
            .map(|i| {
                make_memory_item(
                    &format!("Item {} content here with enough text to fill budget", i),
                    AuthorityLevel::L3Inferred,
                    0.9,
                    MemoryStatus::Active,
                    None,
                )
            })
            .collect();

        let results: Vec<SearchResult> = items
            .iter()
            .cloned()
            .map(|i| make_search_result(i, MatchType::Vector))
            .collect();

        let assembler = ContextAssembler::new().with_search(Arc::new(MockSearchSource {
            results,
            delay_ms: 0,
        }));
        let mut plan = deep_research_plan();
        plan.token_budget = 200;
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "research", None)
            .await
            .unwrap();

        assert!(ctx.compressed, "compression should have been triggered");
        assert!(ctx.token_count <= 200);
    }

    #[tokio::test]
    async fn no_compression_when_within_budget() {
        let assembler = ContextAssembler::new().with_canonical_user(Arc::new(MockCanonicalUser {
            profile: Some(make_profile(
                AuthorityLevel::L1Authoritative,
                SourceType::Iam,
            )),
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::PersonalTask,
            true,
            true,
            false,
            false,
            false,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "my task", None)
            .await
            .unwrap();

        assert!(!ctx.compressed);
    }

    #[tokio::test]
    async fn conflict_detection_different_sources() {
        // Canonical user says role = "engineer" at L1.
        let canon_profile = make_profile(AuthorityLevel::L1Authoritative, SourceType::Iam);

        // Search result says role = "manager" at L3.
        let conflicting_item = make_memory_item(
            "manager",
            AuthorityLevel::L3Inferred,
            0.6,
            MemoryStatus::Active,
            Some("role"),
        );
        let results = vec![make_search_result(conflicting_item, MatchType::Vector)];

        let assembler = ContextAssembler::new()
            .with_canonical_user(Arc::new(MockCanonicalUser {
                profile: Some(canon_profile),
                delay_ms: 0,
            }))
            .with_search(Arc::new(MockSearchSource {
                results,
                delay_ms: 0,
            }));

        let plan = plan_with(
            RequestIntent::DecisionSupport,
            true,
            true,
            false,
            false,
            true,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "decision", None)
            .await
            .unwrap();

        let role_conflict = ctx
            .conflict_flags
            .iter()
            .find(|c| c.field == "role")
            .expect("should detect role conflict");

        assert_eq!(
            role_conflict.conflicting_values,
            vec!["manager".to_string()]
        );
    }

    #[tokio::test]
    async fn conflict_higher_authority_wins() {
        // Canonical user (L0) says timezone = "Asia/Shanghai".
        let mut profile = make_profile(AuthorityLevel::L0SourceOfTruth, SourceType::UserExplicit);
        profile.timezone = Some("Asia/Shanghai".into());

        // Search result (L3) says timezone = "UTC".
        let conflicting_item = make_memory_item(
            "UTC",
            AuthorityLevel::L3Inferred,
            0.5,
            MemoryStatus::Active,
            Some("timezone"),
        );
        let results = vec![make_search_result(conflicting_item, MatchType::Vector)];

        let assembler = ContextAssembler::new()
            .with_canonical_user(Arc::new(MockCanonicalUser {
                profile: Some(profile),
                delay_ms: 0,
            }))
            .with_search(Arc::new(MockSearchSource {
                results,
                delay_ms: 0,
            }));

        let plan = plan_with(
            RequestIntent::DecisionSupport,
            true,
            true,
            false,
            false,
            true,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "decision", None)
            .await
            .unwrap();

        // The L0 value should appear in the text; the L3 value is flagged.
        assert!(ctx.context_text.contains("Asia/Shanghai"));
        let tz_conflict = ctx
            .conflict_flags
            .iter()
            .find(|c| c.field == "timezone")
            .expect("timezone conflict");
        assert_eq!(tz_conflict.conflicting_values, vec!["UTC".to_string()]);
    }

    #[tokio::test]
    async fn low_confidence_marking_by_score() {
        let item = make_memory_item(
            "unverified claim",
            AuthorityLevel::L3Inferred,
            0.5, // < 0.7
            MemoryStatus::Active,
            None,
        );
        let results = vec![make_search_result(item, MatchType::Vector)];

        let assembler = ContextAssembler::new().with_search(Arc::new(MockSearchSource {
            results,
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::DecisionSupport,
            true,
            false,
            false,
            false,
            true,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "query", None)
            .await
            .unwrap();

        assert!(!ctx.low_confidence_items.is_empty());
        assert!(ctx.context_text.contains("[待验证]"));
    }

    #[tokio::test]
    async fn low_confidence_marking_by_candidate_status() {
        let item = make_memory_item(
            "candidate memory",
            AuthorityLevel::L2Verified,
            0.95, // high confidence but candidate status
            MemoryStatus::Candidate,
            None,
        );
        let results = vec![make_search_result(item, MatchType::Vector)];

        let assembler = ContextAssembler::new().with_search(Arc::new(MockSearchSource {
            results,
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::DecisionSupport,
            true,
            false,
            false,
            false,
            true,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "query", None)
            .await
            .unwrap();

        assert!(!ctx.low_confidence_items.is_empty());
        assert!(ctx.context_text.contains("[待验证]"));
    }

    #[tokio::test]
    async fn degraded_source_on_timeout() {
        let assembler = ContextAssembler::new().with_hot_context(Arc::new(MockHotContext {
            data: Some("hot data".into()),
            delay_ms: 500, // will timeout
        }));
        let mut plan = chat_plan();
        plan.latency_budget_ms = 50; // 50ms timeout
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "hi", None)
            .await
            .unwrap();

        assert!(ctx.degraded_sources.contains(&ContextSource::HotContext));
        assert!(!ctx.sources_used.contains(&ContextSource::HotContext));
    }

    #[tokio::test]
    async fn degraded_source_not_configured() {
        // No hot context source attached but plan requests it.
        let assembler = ContextAssembler::new();
        let plan = chat_plan();
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "hi", None)
            .await
            .unwrap();

        assert!(ctx.degraded_sources.contains(&ContextSource::HotContext));
    }

    #[tokio::test]
    async fn degraded_shared_task_without_task_id() {
        let assembler = ContextAssembler::new().with_shared_task(Arc::new(MockSharedTask {
            task: Some(make_task_context()),
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::CrossAgentTask,
            true,
            true,
            true,
            false,
            false,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "task", None) // no task_id
            .await
            .unwrap();

        assert!(ctx.degraded_sources.contains(&ContextSource::SharedTask));
    }

    #[tokio::test]
    async fn chat_intent_minimal_context() {
        let assembler = ContextAssembler::new()
            .with_canonical_user(Arc::new(MockCanonicalUser {
                profile: Some(make_profile(
                    AuthorityLevel::L1Authoritative,
                    SourceType::Iam,
                )),
                delay_ms: 0,
            }))
            .with_search(Arc::new(MockSearchSource {
                results: vec![make_search_result(
                    make_memory_item(
                        "data",
                        AuthorityLevel::L2Verified,
                        0.9,
                        MemoryStatus::Active,
                        None,
                    ),
                    MatchType::Vector,
                )],
                delay_ms: 0,
            }));

        let plan = chat_plan();
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "hi", None)
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::Identity));
        assert!(!ctx.sources_used.contains(&ContextSource::CanonicalUser));
        assert!(!ctx.sources_used.contains(&ContextSource::EnterpriseMemory));
    }

    #[tokio::test]
    async fn personal_task_adds_canonical_user() {
        let assembler = ContextAssembler::new().with_canonical_user(Arc::new(MockCanonicalUser {
            profile: Some(make_profile(
                AuthorityLevel::L1Authoritative,
                SourceType::Iam,
            )),
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::PersonalTask,
            true,
            true,
            false,
            false,
            false,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "my preferences", None)
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::CanonicalUser));
    }

    #[tokio::test]
    async fn business_query_adds_structured_search() {
        let item = make_memory_item(
            "Order #123 shipped",
            AuthorityLevel::L1Authoritative,
            0.9,
            MemoryStatus::Active,
            None,
        );
        let results = vec![make_search_result(item, MatchType::Structured)];

        let assembler = ContextAssembler::new().with_search(Arc::new(MockSearchSource {
            results,
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::BusinessQuery,
            true,
            true,
            false,
            true,
            false,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "track order", None)
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::BusinessState));
    }

    #[tokio::test]
    async fn cross_agent_task_adds_shared_task() {
        let assembler = ContextAssembler::new().with_shared_task(Arc::new(MockSharedTask {
            task: Some(make_task_context()),
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::CrossAgentTask,
            true,
            true,
            true,
            true,
            false,
            false,
        );
        let identity = test_identity();
        let task_id = Uuid::new_v4();

        let ctx = assembler
            .assemble(&plan, &identity, "task", Some(task_id))
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::SharedTask));
    }

    #[tokio::test]
    async fn decision_support_adds_vector_search() {
        let item = make_memory_item(
            "analysis result",
            AuthorityLevel::L2Verified,
            0.8,
            MemoryStatus::Active,
            None,
        );
        let results = vec![make_search_result(item, MatchType::Vector)];

        let assembler = ContextAssembler::new().with_search(Arc::new(MockSearchSource {
            results,
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::DecisionSupport,
            true,
            true,
            true,
            true,
            true,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "should we proceed", None)
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::EnterpriseMemory));
    }

    #[tokio::test]
    async fn deep_research_adds_graph_search() {
        let item = make_memory_item(
            "deep research data",
            AuthorityLevel::L2Verified,
            0.8,
            MemoryStatus::Active,
            None,
        );
        let results = vec![make_search_result(item, MatchType::Hybrid)];

        let assembler = ContextAssembler::new().with_search(Arc::new(MockSearchSource {
            results,
            delay_ms: 0,
        }));
        let plan = deep_research_plan();
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "research topic", None)
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::EnterpriseMemory));
    }

    #[tokio::test]
    async fn parallel_fetch_all_sources() {
        let assembler = ContextAssembler::new()
            .with_hot_context(Arc::new(MockHotContext {
                data: Some("recent activity".into()),
                delay_ms: 0,
            }))
            .with_canonical_user(Arc::new(MockCanonicalUser {
                profile: Some(make_profile(
                    AuthorityLevel::L1Authoritative,
                    SourceType::Iam,
                )),
                delay_ms: 0,
            }))
            .with_shared_task(Arc::new(MockSharedTask {
                task: Some(make_task_context()),
                delay_ms: 0,
            }))
            .with_memory(Arc::new(MockMemorySource {
                items: vec![make_memory_item(
                    "agent note",
                    AuthorityLevel::L2Verified,
                    0.8,
                    MemoryStatus::Active,
                    Some("agent-007"),
                )],
                delay_ms: 0,
            }))
            .with_search(Arc::new(MockSearchSource {
                results: vec![make_search_result(
                    make_memory_item(
                        "enterprise fact",
                        AuthorityLevel::L1Authoritative,
                        0.9,
                        MemoryStatus::Active,
                        None,
                    ),
                    MatchType::Hybrid,
                )],
                delay_ms: 0,
            }));

        let plan = deep_research_plan();
        let identity = identity_with_agent();
        let task_id = Uuid::new_v4();

        let ctx = assembler
            .assemble(&plan, &identity, "deep research", Some(task_id))
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::Identity));
        assert!(ctx.sources_used.contains(&ContextSource::HotContext));
        assert!(ctx.sources_used.contains(&ContextSource::CanonicalUser));
        assert!(ctx.sources_used.contains(&ContextSource::SharedTask));
        assert!(ctx.sources_used.contains(&ContextSource::AgentPrivate));
        assert!(ctx.sources_used.contains(&ContextSource::EnterpriseMemory));
        assert!(ctx.sources_used.contains(&ContextSource::BusinessState));
        assert!(ctx.degraded_sources.is_empty());
    }

    #[tokio::test]
    async fn agent_private_memory_skipped_for_chat() {
        let assembler = ContextAssembler::new().with_memory(Arc::new(MockMemorySource {
            items: vec![make_memory_item(
                "agent note",
                AuthorityLevel::L2Verified,
                0.8,
                MemoryStatus::Active,
                Some("agent-007"),
            )],
            delay_ms: 0,
        }));
        let plan = chat_plan();
        let identity = identity_with_agent();

        let ctx = assembler
            .assemble(&plan, &identity, "hi", None)
            .await
            .unwrap();

        assert!(!ctx.sources_used.contains(&ContextSource::AgentPrivate));
    }

    #[tokio::test]
    async fn agent_private_memory_included_for_personal_task() {
        let assembler = ContextAssembler::new().with_memory(Arc::new(MockMemorySource {
            items: vec![make_memory_item(
                "agent note",
                AuthorityLevel::L2Verified,
                0.8,
                MemoryStatus::Active,
                Some("agent-007"),
            )],
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::PersonalTask,
            true,
            true,
            false,
            false,
            false,
            false,
        );
        let identity = identity_with_agent();

        let ctx = assembler
            .assemble(&plan, &identity, "my task", None)
            .await
            .unwrap();

        assert!(ctx.sources_used.contains(&ContextSource::AgentPrivate));
        assert!(ctx.context_text.contains("## Agent Private Memory"));
    }

    #[tokio::test]
    async fn no_conflict_same_source() {
        // Two items from the same source with same field but different values.
        let item1 = make_memory_item(
            "value A",
            AuthorityLevel::L2Verified,
            0.9,
            MemoryStatus::Active,
            Some("field1"),
        );
        let item2 = make_memory_item(
            "value B",
            AuthorityLevel::L3Inferred,
            0.5,
            MemoryStatus::Active,
            Some("field1"),
        );
        let results = vec![
            make_search_result(item1, MatchType::Vector),
            make_search_result(item2, MatchType::Vector),
        ];

        let assembler = ContextAssembler::new().with_search(Arc::new(MockSearchSource {
            results,
            delay_ms: 0,
        }));
        let plan = plan_with(
            RequestIntent::DecisionSupport,
            true,
            false,
            false,
            false,
            true,
            false,
        );
        let identity = test_identity();

        let ctx = assembler
            .assemble(&plan, &identity, "query", None)
            .await
            .unwrap();

        // Both items are from EnterpriseMemory (same source) → no cross-source conflict.
        let field1_conflicts: Vec<_> = ctx
            .conflict_flags
            .iter()
            .filter(|c| c.field == "field1")
            .collect();
        assert!(
            field1_conflicts.is_empty(),
            "no cross-source conflict expected"
        );
    }
}
