//! Eval Dataset & Baseline Benchmarking (§13 Phase 0, §12.2 Memory Quality).
//!
//! Provides synthetic evaluation datasets, quality-metric computation, and
//! benchmark runners for measuring retrieval quality against ground truth.
//!
//! ## Architecture references
//!
//! - §13 Phase 0 — requires baseline benchmarks and evaluation datasets.
//! - §12.2 — defines memory quality metrics (recall, precision, groundedness,
//!   token utilization, conflict rate, stale rate, evidence coverage).
//! - [`crate::slo_monitoring`] — runtime SLO collection that consumes the
//!   same metric definitions defined here as pure functions.
//! - [`crate::engine::MemoryEngine`] — the pluggable engine trait that
//!   `BenchmarkRunner` exercises.

use crate::engine::{EngineError, EngineHealth, EngineQuery, EngineResult, MemoryEngine};
use chrono::{DateTime, Utc};
use oris_memory_store::memory_types::{AuthorityLevel, Scope};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

// ──────────────────────────── EvalMemory ───────────────────────

/// A memory in an evaluation dataset — the unit that gets stored, retrieved,
/// and scored against ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMemory {
    pub id: Uuid,
    pub content: String,
    pub scope: Scope,
    pub entity_refs: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
    pub authority_level: AuthorityLevel,
}

// ──────────────────────────── GroundTruth ───────────────────────

/// The known-correct answer key for a scenario.
///
/// `relevant_memory_ids` are the memories that *should* be retrieved;
/// `irrelevant_memory_ids` are distractors that should *not* surface;
/// `expected_conflict_fields` are field names where contradictory memories
/// exist and the conflict resolver is expected to flag them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroundTruth {
    pub relevant_memory_ids: Vec<Uuid>,
    pub irrelevant_memory_ids: Vec<Uuid>,
    pub expected_conflict_fields: Vec<String>,
}

// ──────────────────────── AssembledContext ──────────────────────

/// A segment in an assembled context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSegment {
    pub memory_id: Uuid,
    pub content: String,
    pub has_source: bool,
    pub is_expired: bool,
    pub conflict_flags: Vec<String>,
    pub evidence_refs: Vec<String>,
}

/// The assembled context produced by the context assembler, annotated with
/// quality signals so metrics can be computed without re-reading the store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssembledContext {
    pub segments: Vec<ContextSegment>,
    pub used_tokens: usize,
    pub token_budget: usize,
}

// ──────────────────────── Scenario Structs ──────────────────────

/// Equipment-fault scenario — device fault memories with known correct
/// recalls (anonymised device IDs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquipmentFailureScenario {
    /// Unique scenario identifier (e.g. `"equip-fault-003"`).
    pub id: String,
    /// Natural-language query / description used to drive the search.
    pub description: String,
    /// The memories loaded into the engine for this scenario.
    pub memories: Vec<EvalMemory>,
    /// Memory IDs that a correct retrieval must surface (ordered by importance).
    pub expected_recalls: Vec<Uuid>,
    /// The answer key.
    pub ground_truth: GroundTruth,
    pub device_id: String,
    pub fault_type: String,
}

/// Quality-defect scenario — defect memories with entity relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDefectScenario {
    pub id: String,
    pub description: String,
    pub memories: Vec<EvalMemory>,
    pub expected_recalls: Vec<Uuid>,
    pub ground_truth: GroundTruth,
    pub defect_category: String,
    /// (entity, relationship) pairs expected to be resolved.
    pub entity_relationships: Vec<(String, String)>,
}

/// User-preference scenario — preferences with temporal decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferenceScenario {
    pub id: String,
    pub description: String,
    pub memories: Vec<EvalMemory>,
    pub expected_recalls: Vec<Uuid>,
    pub ground_truth: GroundTruth,
    pub user_id: String,
    /// Half-life in days for the freshness decay function.
    pub temporal_decay_days: f64,
}

/// A single eval scenario — one of three anonymised types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "scenario_type", rename_all = "snake_case")]
pub enum EvalScenario {
    EquipmentFailure(EquipmentFailureScenario),
    QualityDefect(QualityDefectScenario),
    UserPreference(UserPreferenceScenario),
}

impl EvalScenario {
    pub fn id(&self) -> &str {
        match self {
            Self::EquipmentFailure(s) => &s.id,
            Self::QualityDefect(s) => &s.id,
            Self::UserPreference(s) => &s.id,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::EquipmentFailure(s) => &s.description,
            Self::QualityDefect(s) => &s.description,
            Self::UserPreference(s) => &s.description,
        }
    }

    pub fn memories(&self) -> &[EvalMemory] {
        match self {
            Self::EquipmentFailure(s) => &s.memories,
            Self::QualityDefect(s) => &s.memories,
            Self::UserPreference(s) => &s.memories,
        }
    }

    pub fn expected_recalls(&self) -> &[Uuid] {
        match self {
            Self::EquipmentFailure(s) => &s.expected_recalls,
            Self::QualityDefect(s) => &s.expected_recalls,
            Self::UserPreference(s) => &s.expected_recalls,
        }
    }

    pub fn ground_truth(&self) -> &GroundTruth {
        match self {
            Self::EquipmentFailure(s) => &s.ground_truth,
            Self::QualityDefect(s) => &s.ground_truth,
            Self::UserPreference(s) => &s.ground_truth,
        }
    }
}

/// A collection of eval scenarios forming a benchmark dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalDataset {
    pub id: String,
    pub name: String,
    pub scenarios: Vec<EvalScenario>,
}

// ──────────────────────────── QualityMetrics ───────────────────

/// Snapshot of all computed quality metrics for a single run or baseline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub context_relevance_rate: f64,
    pub token_utilization: f64,
    pub conflict_rate: f64,
    pub stale_memory_rate: f64,
    pub unsourceed_memory_rate: f64,
    pub groundedness: f64,
    pub evidence_coverage: f64,
}

impl QualityMetrics {
    // ── Recall@K: |top-k ∩ relevant| / |relevant| ──

    pub fn recall_at_k(retrieved: &[Uuid], ground_truth: &GroundTruth, k: usize) -> f64 {
        if ground_truth.relevant_memory_ids.is_empty() {
            return 0.0;
        }
        let relevant: std::collections::HashSet<Uuid> =
            ground_truth.relevant_memory_ids.iter().copied().collect();
        let hits = retrieved
            .iter()
            .take(k)
            .filter(|id| relevant.contains(id))
            .count();
        hits as f64 / ground_truth.relevant_memory_ids.len() as f64
    }

    // ── Precision@K: |top-k ∩ relevant| / k ──

    pub fn precision_at_k(retrieved: &[Uuid], ground_truth: &GroundTruth, k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }
        let relevant: std::collections::HashSet<Uuid> =
            ground_truth.relevant_memory_ids.iter().copied().collect();
        let hits = retrieved
            .iter()
            .take(k)
            .filter(|id| relevant.contains(id))
            .count();
        hits as f64 / k as f64
    }

    // ── Context relevance: relevant segments / total segments ──

    pub fn context_relevance_rate(
        assembled_context: &AssembledContext,
        ground_truth: &GroundTruth,
    ) -> f64 {
        if assembled_context.segments.is_empty() {
            return 0.0;
        }
        let relevant: std::collections::HashSet<Uuid> =
            ground_truth.relevant_memory_ids.iter().copied().collect();
        let relevant_count = assembled_context
            .segments
            .iter()
            .filter(|s| relevant.contains(&s.memory_id))
            .count();
        relevant_count as f64 / assembled_context.segments.len() as f64
    }

    // ── Token utilization: used / budget (clamped to [0,1]) ──

    pub fn token_utilization(used_tokens: usize, budget: usize) -> f64 {
        if budget == 0 {
            return 0.0;
        }
        (used_tokens as f64 / budget as f64).min(1.0)
    }

    // ── Conflict rate: conflicting segments / total segments ──

    pub fn conflict_rate(conflict_flags: usize, total_segments: usize) -> f64 {
        if total_segments == 0 {
            return 0.0;
        }
        conflict_flags as f64 / total_segments as f64
    }

    // ── Stale memory rate: expired / total ──

    pub fn stale_memory_rate(expired_count: usize, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        expired_count as f64 / total as f64
    }

    // ── Unsourceed memory rate: memories without a source / total ──

    pub fn unsourceed_memory_rate(unsourceed_count: usize, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        unsourceed_count as f64 / total as f64
    }

    // ── Groundedness: fraction of context claims backed by evidence ──

    pub fn groundedness(context: &str, evidence_refs: &[String]) -> f64 {
        let sentences = context.split('.').filter(|s| !s.trim().is_empty()).count();
        if sentences == 0 {
            return 1.0; // vacuously grounded — no claims to refute
        }
        if evidence_refs.is_empty() {
            return 0.0; // claims exist but nothing backs them
        }
        (evidence_refs.len() as f64 / sentences as f64).min(1.0)
    }

    // ── Evidence coverage: |retrieved ∩ relevant| / |retrieved ∩ annotated| ──

    pub fn evidence_coverage(retrieved: &[Uuid], ground_truth: &GroundTruth) -> f64 {
        let relevant: std::collections::HashSet<Uuid> =
            ground_truth.relevant_memory_ids.iter().copied().collect();
        let irrelevant: std::collections::HashSet<Uuid> =
            ground_truth.irrelevant_memory_ids.iter().copied().collect();
        let annotated: Vec<&Uuid> = retrieved
            .iter()
            .filter(|id| relevant.contains(id) || irrelevant.contains(id))
            .collect();
        if annotated.is_empty() {
            return 0.0;
        }
        let covered = annotated.iter().filter(|id| relevant.contains(id)).count();
        covered as f64 / annotated.len() as f64
    }

    /// Compute all nine metrics from a single retrieval + assembled context.
    pub fn compute(
        retrieved: &[Uuid],
        ground_truth: &GroundTruth,
        ctx: &AssembledContext,
        k: usize,
    ) -> Self {
        let conflict_flags = ctx
            .segments
            .iter()
            .filter(|s| !s.conflict_flags.is_empty())
            .count();
        let expired = ctx.segments.iter().filter(|s| s.is_expired).count();
        let unsourceed = ctx.segments.iter().filter(|s| !s.has_source).count();
        let all_evidence: Vec<String> = ctx
            .segments
            .iter()
            .flat_map(|s| s.evidence_refs.iter().cloned())
            .collect();
        let context_text: String = ctx
            .segments
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join(". ");
        Self {
            recall_at_k: Self::recall_at_k(retrieved, ground_truth, k),
            precision_at_k: Self::precision_at_k(retrieved, ground_truth, k),
            context_relevance_rate: Self::context_relevance_rate(ctx, ground_truth),
            token_utilization: Self::token_utilization(ctx.used_tokens, ctx.token_budget),
            conflict_rate: Self::conflict_rate(conflict_flags, ctx.segments.len()),
            stale_memory_rate: Self::stale_memory_rate(expired, ctx.segments.len()),
            unsourceed_memory_rate: Self::unsourceed_memory_rate(unsourceed, ctx.segments.len()),
            groundedness: Self::groundedness(&context_text, &all_evidence),
            evidence_coverage: Self::evidence_coverage(retrieved, ground_truth),
        }
    }
}

// ──────────────────────── Benchmark Results ────────────────────

/// Per-dataset baseline benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub scenario_id: String,
    pub latency_p50_ms: u64,
    pub latency_p95_ms: u64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub precision_at_5: f64,
    pub groundedness: f64,
    pub token_utilization: f64,
}

/// Concurrent-load benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentBenchmarkResult {
    pub total_requests: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub throughput_rps: f64,
}

// ──────────────────────── BenchmarkRunner ──────────────────────

/// Runs baseline and concurrent benchmarks against a [`MemoryEngine`].
#[derive(Debug, Clone)]
pub struct BenchmarkRunner {
    /// Token budget used for token-utilization estimation.
    pub token_budget: usize,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self { token_budget: 4096 }
    }
}

impl BenchmarkRunner {
    pub fn new(token_budget: usize) -> Self {
        Self { token_budget }
    }

    /// Run a baseline benchmark over every scenario in `dataset`.
    ///
    /// Each scenario is searched once via `engine`; latencies are aggregated
    /// into P50/P95, and quality metrics are averaged across scenarios.
    pub async fn run_baseline(
        &self,
        dataset: &EvalDataset,
        engine: &dyn MemoryEngine,
    ) -> Result<BenchmarkResult, EngineError> {
        let mut latencies: Vec<u64> = Vec::new();
        let mut recall5_vals: Vec<f64> = Vec::new();
        let mut recall10_vals: Vec<f64> = Vec::new();
        let mut precision5_vals: Vec<f64> = Vec::new();
        let mut groundedness_vals: Vec<f64> = Vec::new();
        let mut token_util_vals: Vec<f64> = Vec::new();

        for scenario in &dataset.scenarios {
            let query = EngineQuery {
                text: scenario.description().to_string(),
                tenant_id: "eval".to_string(),
                user_id: None,
                task_id: None,
                top_k: 10,
                filters: HashMap::new(),
            };

            let start = Instant::now();
            let results = engine.search(&query).await?;
            let elapsed_ms = start.elapsed().as_millis() as u64;
            latencies.push(elapsed_ms);

            let retrieved: Vec<Uuid> = results
                .iter()
                .filter_map(|r| Uuid::parse_str(&r.memory_id).ok())
                .collect();

            let used_tokens: usize = results.iter().map(|r| r.content.len() / 4).sum();
            let ctx = AssembledContext {
                segments: results
                    .iter()
                    .map(|r| {
                        let mid = Uuid::parse_str(&r.memory_id).unwrap_or_default();
                        ContextSegment {
                            memory_id: mid,
                            content: r.content.clone(),
                            has_source: !r.evidence_refs.is_empty(),
                            is_expired: false,
                            conflict_flags: vec![],
                            evidence_refs: r.evidence_refs.clone(),
                        }
                    })
                    .collect(),
                used_tokens,
                token_budget: self.token_budget,
            };

            let gt = scenario.ground_truth();
            recall5_vals.push(QualityMetrics::recall_at_k(&retrieved, gt, 5));
            recall10_vals.push(QualityMetrics::recall_at_k(&retrieved, gt, 10));
            precision5_vals.push(QualityMetrics::precision_at_k(&retrieved, gt, 5));
            groundedness_vals.push(QualityMetrics::groundedness(
                &ctx.segments
                    .iter()
                    .map(|s| s.content.as_str())
                    .collect::<Vec<_>>()
                    .join(". "),
                &ctx.segments
                    .iter()
                    .flat_map(|s| s.evidence_refs.iter().cloned())
                    .collect::<Vec<_>>(),
            ));
            token_util_vals.push(QualityMetrics::token_utilization(
                used_tokens,
                self.token_budget,
            ));
        }

        Ok(BenchmarkResult {
            scenario_id: "aggregate".to_string(),
            latency_p50_ms: percentile(&latencies, 50.0),
            latency_p95_ms: percentile(&latencies, 95.0),
            recall_at_5: avg(&recall5_vals),
            recall_at_10: avg(&recall10_vals),
            precision_at_5: avg(&precision5_vals),
            groundedness: avg(&groundedness_vals),
            token_utilization: avg(&token_util_vals),
        })
        .map(|mut r| {
            // Guard against empty dataset producing nonsensical zeros.
            if dataset.scenarios.is_empty() {
                r.scenario_id = "empty".to_string();
            }
            r
        })
    }

    /// Run a concurrent-load benchmark cycling through `dataset` scenarios.
    ///
    /// Spawns up to `concurrency` searches in parallel, running `iterations`
    /// total requests.  Measures throughput and tail latency.
    pub async fn run_concurrent(
        &self,
        dataset: &EvalDataset,
        engine: Arc<dyn MemoryEngine>,
        concurrency: usize,
        iterations: usize,
    ) -> ConcurrentBenchmarkResult {
        if dataset.scenarios.is_empty() || iterations == 0 {
            return ConcurrentBenchmarkResult {
                total_requests: 0,
                success_count: 0,
                error_count: 0,
                avg_latency_ms: 0.0,
                p95_latency_ms: 0,
                p99_latency_ms: 0,
                throughput_rps: 0.0,
            };
        }

        let queries: Vec<EngineQuery> = dataset
            .scenarios
            .iter()
            .map(|s| EngineQuery {
                text: s.description().to_string(),
                tenant_id: "eval".to_string(),
                user_id: None,
                task_id: None,
                top_k: 10,
                filters: HashMap::new(),
            })
            .collect();

        let mut join_set = tokio::task::JoinSet::new();
        let mut latencies: Vec<u64> = Vec::with_capacity(iterations);
        let mut success_count = 0usize;
        let mut error_count = 0usize;
        let mut spawned = 0usize;
        let concurrency = concurrency.max(1);

        let wall_start = Instant::now();

        while spawned < iterations {
            while join_set.len() < concurrency && spawned < iterations {
                let engine = Arc::clone(&engine);
                let query = queries[spawned % queries.len()].clone();
                join_set.spawn(async move {
                    let start = Instant::now();
                    let result = engine.search(&query).await;
                    let elapsed_ms = start.elapsed().as_millis() as u64;
                    (result.is_ok(), elapsed_ms)
                });
                spawned += 1;
            }
            if let Some(res) = join_set.join_next().await {
                let (ok, ms) = res.unwrap_or((false, 0));
                if ok {
                    success_count += 1;
                } else {
                    error_count += 1;
                }
                latencies.push(ms);
            }
        }

        while let Some(res) = join_set.join_next().await {
            let (ok, ms) = res.unwrap_or((false, 0));
            if ok {
                success_count += 1;
            } else {
                error_count += 1;
            }
            latencies.push(ms);
        }

        let elapsed_secs = wall_start.elapsed().as_secs_f64();
        let total_requests = success_count + error_count;
        ConcurrentBenchmarkResult {
            total_requests,
            success_count,
            error_count,
            avg_latency_ms: avg_u64(&latencies),
            p95_latency_ms: percentile(&latencies, 95.0),
            p99_latency_ms: percentile(&latencies, 99.0),
            throughput_rps: if elapsed_secs > 0.0 {
                total_requests as f64 / elapsed_secs
            } else {
                0.0
            },
        }
    }
}

// ──────────────────────── ComparisonBaseline ───────────────────

/// The system being benchmarked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemType {
    PostgresPgvector,
    Mem0,
    Cognee,
}

impl SystemType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PostgresPgvector => "postgres_pgvector",
            Self::Mem0 => "mem0",
            Self::Cognee => "cognee",
        }
    }
}

/// Latency statistics for a baseline entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// One system's baseline metrics and latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub system: SystemType,
    pub metrics: QualityMetrics,
    pub latency: LatencyStats,
}

/// Reference baselines for PostgreSQL/pgvector vs Mem0 vs Cognee.
///
/// Values are derived from §12.2 target metrics and typical published
/// performance characteristics.  They serve as the comparison anchor for
/// regression detection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComparisonBaseline {
    pub entries: Vec<BaselineEntry>,
}

impl ComparisonBaseline {
    /// Return the default comparison baselines for all three systems.
    pub fn default_baselines() -> Self {
        Self {
            entries: vec![
                BaselineEntry {
                    system: SystemType::PostgresPgvector,
                    metrics: QualityMetrics {
                        recall_at_k: 0.92,
                        precision_at_k: 0.88,
                        context_relevance_rate: 0.90,
                        token_utilization: 0.75,
                        conflict_rate: 0.02,
                        stale_memory_rate: 0.01,
                        unsourceed_memory_rate: 0.00,
                        groundedness: 0.95,
                        evidence_coverage: 0.93,
                    },
                    latency: LatencyStats {
                        p50_ms: 12.0,
                        p95_ms: 45.0,
                        p99_ms: 80.0,
                    },
                },
                BaselineEntry {
                    system: SystemType::Mem0,
                    metrics: QualityMetrics {
                        recall_at_k: 0.82,
                        precision_at_k: 0.75,
                        context_relevance_rate: 0.80,
                        token_utilization: 0.68,
                        conflict_rate: 0.05,
                        stale_memory_rate: 0.03,
                        unsourceed_memory_rate: 0.04,
                        groundedness: 0.78,
                        evidence_coverage: 0.80,
                    },
                    latency: LatencyStats {
                        p50_ms: 22.0,
                        p95_ms: 85.0,
                        p99_ms: 150.0,
                    },
                },
                BaselineEntry {
                    system: SystemType::Cognee,
                    metrics: QualityMetrics {
                        recall_at_k: 0.85,
                        precision_at_k: 0.79,
                        context_relevance_rate: 0.83,
                        token_utilization: 0.70,
                        conflict_rate: 0.04,
                        stale_memory_rate: 0.02,
                        unsourceed_memory_rate: 0.02,
                        groundedness: 0.85,
                        evidence_coverage: 0.82,
                    },
                    latency: LatencyStats {
                        p50_ms: 28.0,
                        p95_ms: 100.0,
                        p99_ms: 200.0,
                    },
                },
            ],
        }
    }

    /// Find a baseline entry by system type.
    pub fn get(&self, system: SystemType) -> Option<&BaselineEntry> {
        self.entries.iter().find(|e| e.system == system)
    }
}

// ──────────────────────── DatasetBuilder ───────────────────────

/// Generates synthetic eval data for the three scenario types.
///
/// UUIDs are deterministic (`Uuid::from_u128` with a monotonic counter) so
/// that benchmarks are reproducible across runs.
pub struct DatasetBuilder;

impl DatasetBuilder {
    /// Build `count` equipment-failure scenarios.
    pub fn build_equipment_failure(count: usize) -> Vec<EquipmentFailureScenario> {
        let mut counter: u128 = 1;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let memories = Self::equip_memories(&mut counter, i);
            let relevant: Vec<Uuid> = memories.iter().take(3).map(|m| m.id).collect();
            let irrelevant: Vec<Uuid> = memories.iter().skip(3).take(2).map(|m| m.id).collect();
            out.push(EquipmentFailureScenario {
                id: format!("equip-fault-{i:03}"),
                description: format!("Device DEV-{i:04} reported motor overheating fault"),
                expected_recalls: relevant.clone(),
                memories,
                ground_truth: GroundTruth {
                    relevant_memory_ids: relevant,
                    irrelevant_memory_ids: irrelevant,
                    expected_conflict_fields: vec!["temperature".into(), "status".into()],
                },
                device_id: format!("DEV-{i:04}"),
                fault_type: "motor_overheating".into(),
            });
        }
        out
    }

    /// Build `count` quality-defect scenarios.
    pub fn build_quality_defect(count: usize) -> Vec<QualityDefectScenario> {
        let mut counter: u128 = 100_000;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let memories = Self::defect_memories(&mut counter, i);
            let relevant: Vec<Uuid> = memories.iter().take(3).map(|m| m.id).collect();
            let irrelevant: Vec<Uuid> = memories.iter().skip(3).take(2).map(|m| m.id).collect();
            out.push(QualityDefectScenario {
                id: format!("quality-defect-{i:03}"),
                description: format!("Surface scratch detected on product P-{i:04} from line L-3"),
                expected_recalls: relevant.clone(),
                memories,
                ground_truth: GroundTruth {
                    relevant_memory_ids: relevant,
                    irrelevant_memory_ids: irrelevant,
                    expected_conflict_fields: vec!["severity".into()],
                },
                defect_category: "surface_scratch".into(),
                entity_relationships: vec![
                    (format!("defect-{i}"), format!("product P-{i:04}")),
                    (format!("defect-{i}"), "line L-3".into()),
                ],
            });
        }
        out
    }

    /// Build `count` user-preference scenarios.
    pub fn build_user_preference(count: usize) -> Vec<UserPreferenceScenario> {
        let mut counter: u128 = 200_000;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let memories = Self::pref_memories(&mut counter, i);
            let relevant: Vec<Uuid> = memories.iter().take(2).map(|m| m.id).collect();
            let irrelevant: Vec<Uuid> = memories.iter().skip(2).take(3).map(|m| m.id).collect();
            out.push(UserPreferenceScenario {
                id: format!("user-pref-{i:03}"),
                description: format!("What UI theme does user U-{i:04} prefer?"),
                expected_recalls: relevant.clone(),
                memories,
                ground_truth: GroundTruth {
                    relevant_memory_ids: relevant,
                    irrelevant_memory_ids: irrelevant,
                    expected_conflict_fields: vec!["theme".into()],
                },
                user_id: format!("U-{i:04}"),
                temporal_decay_days: 30.0,
            });
        }
        out
    }

    /// Build a full dataset with `count_per_type` scenarios of each type.
    pub fn build_dataset(count_per_type: usize) -> EvalDataset {
        let equip = Self::build_equipment_failure(count_per_type);
        let defect = Self::build_quality_defect(count_per_type);
        let pref = Self::build_user_preference(count_per_type);
        let total = equip.len() + defect.len() + pref.len();
        let mut scenarios: Vec<EvalScenario> = Vec::with_capacity(total);
        scenarios.extend(equip.into_iter().map(EvalScenario::EquipmentFailure));
        scenarios.extend(defect.into_iter().map(EvalScenario::QualityDefect));
        scenarios.extend(pref.into_iter().map(EvalScenario::UserPreference));
        EvalDataset {
            id: "oris-baseline-v1".into(),
            name: "Oris Baseline Eval Dataset".into(),
            scenarios,
        }
    }

    // ── private memory generators ──

    fn next_uuid(counter: &mut u128) -> Uuid {
        let id = Uuid::from_u128(*counter);
        *counter += 1;
        id
    }

    fn equip_memories(counter: &mut u128, i: usize) -> Vec<EvalMemory> {
        let now = Utc::now();
        (0..5)
            .map(|j| {
                let is_relevant = j < 3;
                EvalMemory {
                    id: Self::next_uuid(counter),
                    content: if is_relevant {
                        format!("Device DEV-{i:04} motor temperature exceeded 95C threshold")
                    } else {
                        format!("Device DEV-{i:04} routine maintenance completed last week")
                    },
                    scope: if is_relevant {
                        Scope::Factory
                    } else {
                        Scope::Process
                    },
                    entity_refs: vec![format!("DEV-{i:04}")],
                    timestamp: now,
                    confidence: if is_relevant { 0.95 } else { 0.50 },
                    authority_level: if is_relevant {
                        AuthorityLevel::L1Authoritative
                    } else {
                        AuthorityLevel::L3Inferred
                    },
                }
            })
            .collect()
    }

    fn defect_memories(counter: &mut u128, i: usize) -> Vec<EvalMemory> {
        let now = Utc::now();
        (0..5)
            .map(|j| {
                let is_relevant = j < 3;
                EvalMemory {
                    id: Self::next_uuid(counter),
                    content: if is_relevant {
                        format!("Surface scratch severity 3 on product P-{i:04} line L-3")
                    } else {
                        format!("Product P-{i:04} passed QC inspection")
                    },
                    scope: Scope::Factory,
                    entity_refs: vec![format!("P-{i:04}"), "L-3".into()],
                    timestamp: now,
                    confidence: if is_relevant { 0.90 } else { 0.40 },
                    authority_level: if is_relevant {
                        AuthorityLevel::L2Verified
                    } else {
                        AuthorityLevel::L3Inferred
                    },
                }
            })
            .collect()
    }

    fn pref_memories(counter: &mut u128, i: usize) -> Vec<EvalMemory> {
        let now = Utc::now();
        (0..5)
            .map(|j| {
                let is_relevant = j < 2;
                let ts = if is_relevant {
                    now
                } else {
                    now - chrono::Duration::days(60)
                };
                EvalMemory {
                    id: Self::next_uuid(counter),
                    content: if is_relevant {
                        format!("User U-{i:04} prefers dark theme")
                    } else {
                        format!("User U-{i:04} previously used light theme")
                    },
                    scope: if is_relevant {
                        Scope::Personal
                    } else {
                        Scope::Agent
                    },
                    entity_refs: vec![format!("U-{i:04}")],
                    timestamp: ts,
                    confidence: if is_relevant { 0.85 } else { 0.30 },
                    authority_level: if is_relevant {
                        AuthorityLevel::L1Authoritative
                    } else {
                        AuthorityLevel::L3Inferred
                    },
                }
            })
            .collect()
    }
}

// ──────────────────────── Helpers ───────────────────────────────

/// Percentile of a latency slice (linear-interpolation, matches
/// [`crate::slo_monitoring::SloCollector::compute_p95`] convention).
fn percentile(latencies: &[u64], p: f64) -> u64 {
    if latencies.is_empty() {
        return 0;
    }
    let mut sorted = latencies.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() as f64) * p / 100.0).ceil() as usize;
    let idx = idx.min(sorted.len()) - 1;
    sorted[idx]
}

fn avg(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn avg_u64(vals: &[u64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<u64>() as f64 / vals.len() as f64
}

// ──────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. Recall@K — perfect recall ──

    #[test]
    fn recall_at_k_perfect() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1, id2],
            irrelevant_memory_ids: vec![],
            expected_conflict_fields: vec![],
        };
        let retrieved = vec![id1, id2];
        assert!((QualityMetrics::recall_at_k(&retrieved, &gt, 5) - 1.0).abs() < 1e-9);
    }

    // ── 2. Recall@K — partial recall ──

    #[test]
    fn recall_at_k_partial() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1, id2, id3],
            irrelevant_memory_ids: vec![],
            expected_conflict_fields: vec![],
        };
        let retrieved = vec![id1, id2];
        let r = QualityMetrics::recall_at_k(&retrieved, &gt, 5);
        assert!((r - 2.0 / 3.0).abs() < 1e-9);
    }

    // ── 3. Recall@K — no hits ──

    #[test]
    fn recall_at_k_none() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1],
            irrelevant_memory_ids: vec![],
            expected_conflict_fields: vec![],
        };
        let retrieved = vec![id2];
        assert!((QualityMetrics::recall_at_k(&retrieved, &gt, 5) - 0.0).abs() < 1e-9);
    }

    // ── 4. Recall@K — k limits the top-k window ──

    #[test]
    fn recall_at_k_window_truncation() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1, id2],
            irrelevant_memory_ids: vec![],
            expected_conflict_fields: vec![],
        };
        // id2 is at position 3 (> k=1), so only 1 of 2 relevant retrieved.
        let retrieved = vec![Uuid::new_v4(), Uuid::new_v4(), id2];
        let r = QualityMetrics::recall_at_k(&retrieved, &gt, 1);
        assert!((r - 0.0).abs() < 1e-9); // id2 beyond k=1 window
    }

    // ── 5. Precision@K — perfect precision ──

    #[test]
    fn precision_at_k_perfect() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1, id2],
            irrelevant_memory_ids: vec![],
            expected_conflict_fields: vec![],
        };
        let retrieved = vec![id1, id2];
        assert!((QualityMetrics::precision_at_k(&retrieved, &gt, 5) - 2.0 / 5.0).abs() < 1e-9);
    }

    // ── 6. Precision@K — partial ──

    #[test]
    fn precision_at_k_partial() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1],
            irrelevant_memory_ids: vec![id2],
            expected_conflict_fields: vec![],
        };
        let retrieved = vec![id1, id2];
        let p = QualityMetrics::precision_at_k(&retrieved, &gt, 2);
        assert!((p - 0.5).abs() < 1e-9);
    }

    // ── 7. Precision@K — k=0 returns 0 ──

    #[test]
    fn precision_at_k_zero_k() {
        let gt = GroundTruth::default();
        assert_eq!(QualityMetrics::precision_at_k(&[], &gt, 0), 0.0);
    }

    // ── 8. Context relevance rate — all relevant ──

    #[test]
    fn context_relevance_all_relevant() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1, id2],
            irrelevant_memory_ids: vec![],
            expected_conflict_fields: vec![],
        };
        let ctx = AssembledContext {
            segments: vec![seg(id1, true), seg(id2, true)],
            used_tokens: 100,
            token_budget: 4096,
        };
        assert!((QualityMetrics::context_relevance_rate(&ctx, &gt) - 1.0).abs() < 1e-9);
    }

    // ── 9. Context relevance rate — partial ──

    #[test]
    fn context_relevance_partial() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1],
            irrelevant_memory_ids: vec![id2],
            expected_conflict_fields: vec![],
        };
        let ctx = AssembledContext {
            segments: vec![seg(id1, true), seg(id2, false)],
            used_tokens: 100,
            token_budget: 4096,
        };
        assert!((QualityMetrics::context_relevance_rate(&ctx, &gt) - 0.5).abs() < 1e-9);
    }

    // ── 10. Token utilization — within budget ──

    #[test]
    fn token_utilization_within_budget() {
        let u = QualityMetrics::token_utilization(2048, 4096);
        assert!((u - 0.5).abs() < 1e-9);
    }

    // ── 11. Token utilization — over budget clamped ──

    #[test]
    fn token_utilization_over_budget_clamped() {
        let u = QualityMetrics::token_utilization(8192, 4096);
        assert!((u - 1.0).abs() < 1e-9);
    }

    // ── 12. Conflict rate ──

    #[test]
    fn conflict_rate_computation() {
        let r = QualityMetrics::conflict_rate(2, 10);
        assert!((r - 0.2).abs() < 1e-9);
    }

    // ── 13. Stale memory rate ──

    #[test]
    fn stale_memory_rate_computation() {
        let r = QualityMetrics::stale_memory_rate(3, 10);
        assert!((r - 0.3).abs() < 1e-9);
    }

    // ── 14. Unsourceed memory rate ──

    #[test]
    fn unsourceed_memory_rate_computation() {
        let r = QualityMetrics::unsourceed_memory_rate(4, 10);
        assert!((r - 0.4).abs() < 1e-9);
    }

    // ── 15. Groundedness — with evidence ──

    #[test]
    fn groundedness_with_evidence() {
        let context = "Claim one. Claim two.";
        let refs = vec!["ref1".into(), "ref2".into()];
        let g = QualityMetrics::groundedness(context, &refs);
        assert!((g - 1.0).abs() < 1e-9);
    }

    // ── 16. Groundedness — no evidence ──

    #[test]
    fn groundedness_no_evidence() {
        let context = "Claim one. Claim two.";
        let g = QualityMetrics::groundedness(context, &[]);
        assert!((g - 0.0).abs() < 1e-9);
    }

    // ── 17. Groundedness — empty context ──

    #[test]
    fn groundedness_empty_context() {
        let g = QualityMetrics::groundedness("", &["ref1".into()]);
        assert!((g - 1.0).abs() < 1e-9);
    }

    // ── 18. Evidence coverage ──

    #[test]
    fn evidence_coverage_computation() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1, id2],
            irrelevant_memory_ids: vec![id3],
            expected_conflict_fields: vec![],
        };
        // retrieved: id1 (relevant), id3 (irrelevant) → annotated=2, covered=1
        let retrieved = vec![id1, id3];
        let ec = QualityMetrics::evidence_coverage(&retrieved, &gt);
        assert!((ec - 0.5).abs() < 1e-9);
    }

    // ── 19. Evidence coverage — no annotated hits ──

    #[test]
    fn evidence_coverage_no_annotated() {
        let id1 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1],
            irrelevant_memory_ids: vec![],
            expected_conflict_fields: vec![],
        };
        let retrieved = vec![Uuid::new_v4()]; // not in ground truth at all
        assert_eq!(QualityMetrics::evidence_coverage(&retrieved, &gt), 0.0);
    }

    // ── 20. DatasetBuilder — equipment failure ──

    #[test]
    fn dataset_builder_equipment_failure() {
        let scenarios = DatasetBuilder::build_equipment_failure(3);
        assert_eq!(scenarios.len(), 3);
        assert_eq!(scenarios[0].memories.len(), 5);
        assert_eq!(scenarios[0].expected_recalls.len(), 3);
        assert!(!scenarios[0].device_id.is_empty());
        assert_eq!(scenarios[0].fault_type, "motor_overheating");
        // deterministic UUIDs — run twice, get same IDs
        let again = DatasetBuilder::build_equipment_failure(3);
        assert_eq!(scenarios[0].memories[0].id, again[0].memories[0].id);
    }

    // ── 21. DatasetBuilder — quality defect ──

    #[test]
    fn dataset_builder_quality_defect() {
        let scenarios = DatasetBuilder::build_quality_defect(2);
        assert_eq!(scenarios.len(), 2);
        assert_eq!(scenarios[0].memories.len(), 5);
        assert!(!scenarios[0].entity_relationships.is_empty());
        assert_eq!(scenarios[0].defect_category, "surface_scratch");
    }

    // ── 22. DatasetBuilder — user preference ──

    #[test]
    fn dataset_builder_user_preference() {
        let scenarios = DatasetBuilder::build_user_preference(2);
        assert_eq!(scenarios.len(), 2);
        assert_eq!(scenarios[0].memories.len(), 5);
        assert_eq!(scenarios[0].expected_recalls.len(), 2);
        assert!((scenarios[0].temporal_decay_days - 30.0).abs() < 1e-9);
    }

    // ── 23. DatasetBuilder — full dataset ──

    #[test]
    fn dataset_builder_full_dataset() {
        let ds = DatasetBuilder::build_dataset(3);
        assert_eq!(ds.scenarios.len(), 9);
        assert!(ds
            .scenarios
            .iter()
            .any(|s| matches!(s, EvalScenario::EquipmentFailure(_))));
        assert!(ds
            .scenarios
            .iter()
            .any(|s| matches!(s, EvalScenario::QualityDefect(_))));
        assert!(ds
            .scenarios
            .iter()
            .any(|s| matches!(s, EvalScenario::UserPreference(_))));
    }

    // ── 24. ComparisonBaseline — default baselines ──

    #[test]
    fn comparison_baseline_defaults() {
        let cb = ComparisonBaseline::default_baselines();
        assert_eq!(cb.entries.len(), 3);
        assert!(cb.get(SystemType::PostgresPgvector).is_some());
        assert!(cb.get(SystemType::Mem0).is_some());
        assert!(cb.get(SystemType::Cognee).is_some());
        let pg = cb.get(SystemType::PostgresPgvector).unwrap();
        assert!(pg.metrics.recall_at_k > 0.9);
        assert!(pg.latency.p95_ms < 100.0);
    }

    // ── 25. BenchmarkResult serialization ──

    #[test]
    fn benchmark_result_serialization() {
        let result = BenchmarkResult {
            scenario_id: "test".into(),
            latency_p50_ms: 10,
            latency_p95_ms: 50,
            recall_at_5: 0.8,
            recall_at_10: 0.9,
            precision_at_5: 0.7,
            groundedness: 0.95,
            token_utilization: 0.6,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.scenario_id, "test");
        assert_eq!(deserialized.latency_p95_ms, 50);
        assert!((deserialized.recall_at_5 - 0.8).abs() < 1e-9);
    }

    // ── 26. ConcurrentBenchmarkResult serialization ──

    #[test]
    fn concurrent_benchmark_result_serialization() {
        let result = ConcurrentBenchmarkResult {
            total_requests: 100,
            success_count: 95,
            error_count: 5,
            avg_latency_ms: 12.5,
            p95_latency_ms: 30,
            p99_latency_ms: 50,
            throughput_rps: 800.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ConcurrentBenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_requests, 100);
        assert_eq!(deserialized.success_count, 95);
        assert!((deserialized.throughput_rps - 800.0).abs() < 1e-9);
    }

    // ── 27. Percentile helper ──

    #[test]
    fn percentile_computation() {
        let latencies: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&latencies, 50.0), 50);
        assert_eq!(percentile(&latencies, 95.0), 95);
        assert_eq!(percentile(&latencies, 99.0), 99);
    }

    // ── 28. Percentile — empty ──

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 95.0), 0);
    }

    // ── 29. QualityMetrics::compute ──

    #[test]
    fn quality_metrics_compute() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let gt = GroundTruth {
            relevant_memory_ids: vec![id1],
            irrelevant_memory_ids: vec![id2],
            expected_conflict_fields: vec![],
        };
        let ctx = AssembledContext {
            segments: vec![
                ContextSegment {
                    memory_id: id1,
                    content: "Relevant claim".into(),
                    has_source: true,
                    is_expired: false,
                    conflict_flags: vec![],
                    evidence_refs: vec!["ref1".into()],
                },
                ContextSegment {
                    memory_id: id2,
                    content: "Irrelevant claim".into(),
                    has_source: false,
                    is_expired: true,
                    conflict_flags: vec!["field1".into()],
                    evidence_refs: vec![],
                },
            ],
            used_tokens: 50,
            token_budget: 4096,
        };
        let m = QualityMetrics::compute(&[id1, id2], &gt, &ctx, 5);
        assert!((m.recall_at_k - 1.0).abs() < 1e-9);
        assert!((m.precision_at_k - 0.2).abs() < 1e-9);
        assert!((m.context_relevance_rate - 0.5).abs() < 1e-9);
        assert!((m.conflict_rate - 0.5).abs() < 1e-9);
        assert!((m.stale_memory_rate - 0.5).abs() < 1e-9);
        assert!((m.unsourceed_memory_rate - 0.5).abs() < 1e-9);
    }

    // ── 30. Run baseline with NoopEngine ──

    #[tokio::test]
    async fn run_baseline_with_noop_engine() {
        use crate::engine::NoopEngine;
        let ds = DatasetBuilder::build_dataset(3);
        let runner = BenchmarkRunner::default();
        let engine = NoopEngine::new("noop");
        let result = runner.run_baseline(&ds, &engine).await.unwrap();
        assert_eq!(result.scenario_id, "aggregate");
        // NoopEngine returns empty results → all quality metrics 0
        assert!((result.recall_at_5 - 0.0).abs() < 1e-9);
        assert!((result.precision_at_5 - 0.0).abs() < 1e-9);
        assert!((result.groundedness - 1.0).abs() < 1e-9); // empty context = vacuously grounded
    }

    // ── 31. Run concurrent with NoopEngine ──

    #[tokio::test]
    async fn run_concurrent_with_noop_engine() {
        use crate::engine::NoopEngine;
        let ds = DatasetBuilder::build_dataset(2);
        let runner = BenchmarkRunner::default();
        let engine: Arc<dyn MemoryEngine> = Arc::new(NoopEngine::new("noop"));
        let result = runner.run_concurrent(&ds, engine, 4, 20).await;
        assert_eq!(result.total_requests, 20);
        assert_eq!(result.success_count, 20);
        assert_eq!(result.error_count, 0);
        assert!(result.throughput_rps > 0.0);
    }

    // ── 32. Run concurrent — empty dataset ──

    #[tokio::test]
    async fn run_concurrent_empty_dataset() {
        use crate::engine::NoopEngine;
        let ds = EvalDataset::default();
        let runner = BenchmarkRunner::default();
        let engine: Arc<dyn MemoryEngine> = Arc::new(NoopEngine::new("noop"));
        let result = runner.run_concurrent(&ds, engine, 4, 10).await;
        assert_eq!(result.total_requests, 0);
    }

    // ── 33. Run baseline — empty dataset ──

    #[tokio::test]
    async fn run_baseline_empty_dataset() {
        use crate::engine::NoopEngine;
        let ds = EvalDataset::default();
        let runner = BenchmarkRunner::default();
        let engine = NoopEngine::new("noop");
        let result = runner.run_baseline(&ds, &engine).await.unwrap();
        assert_eq!(result.scenario_id, "empty");
        assert_eq!(result.latency_p50_ms, 0);
    }

    // ── 34. EvalScenario accessor methods ──

    #[test]
    fn eval_scenario_accessors() {
        let scenarios = DatasetBuilder::build_equipment_failure(1);
        let s = EvalScenario::EquipmentFailure(scenarios.into_iter().next().unwrap());
        assert!(s.id().starts_with("equip-fault"));
        assert!(!s.description().is_empty());
        assert_eq!(s.memories().len(), 5);
        assert_eq!(s.expected_recalls().len(), 3);
        assert_eq!(s.ground_truth().expected_conflict_fields.len(), 2);
    }

    // ── 35. SystemType as_str ──

    #[test]
    fn system_type_as_str() {
        assert_eq!(SystemType::PostgresPgvector.as_str(), "postgres_pgvector");
        assert_eq!(SystemType::Mem0.as_str(), "mem0");
        assert_eq!(SystemType::Cognee.as_str(), "cognee");
    }

    // ── 36. EvalDataset serde round-trip ──

    #[test]
    fn eval_dataset_serde_roundtrip() {
        let ds = DatasetBuilder::build_dataset(2);
        let json = serde_json::to_string(&ds).unwrap();
        let restored: EvalDataset = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.scenarios.len(), 6);
        assert!(matches!(
            restored.scenarios[0],
            EvalScenario::EquipmentFailure(_)
        ));
    }

    // ── Helper ──

    fn seg(id: Uuid, has_source: bool) -> ContextSegment {
        ContextSegment {
            memory_id: id,
            content: "test content".into(),
            has_source,
            is_expired: false,
            conflict_flags: vec![],
            evidence_refs: if has_source {
                vec!["ref".into()]
            } else {
                vec![]
            },
        }
    }
}
