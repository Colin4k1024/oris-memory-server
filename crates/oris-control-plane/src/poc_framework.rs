//! Mem0/Cognee PoC Framework (§13.5, §3.6, §3.7).
//!
//! Provides structured proof-of-concept orchestration for evaluating Mem0 and
//! Cognee memory systems against the PostgreSQL/pgvector baseline. Each trial
//! carries a frozen hypothesis with explicit success thresholds, stop
//! conditions, and data boundaries — satisfying the enterprise governance
//! requirement that no system is adopted without empirical evidence.
//!
//! ## Architecture references
//!
//! - §13.5 — PoC must define hypotheses, control groups, success thresholds,
//!   stop conditions, and data boundaries before execution.
//! - §3.6 — Mem0 PoC focus: conflict merge, delete propagation, idempotency,
//!   Chinese recall quality, explainability, self-hosted operations.
//! - §3.7 — Cognee PoC focus: multi-hop accuracy, entity duplication,
//!   relationship correctness, source tracing, ACL mapping, incremental
//!   update, delete propagation, latency and total cost.
//! - [`crate::eval`] — `SystemType`, `QualityMetrics`, and `BenchmarkResult`
//!   patterns reused here for metric collection.
//! - [`crate::slo_monitoring`] — runtime SLO patterns that this framework
//!   mirrors for threshold evaluation.

use crate::eval::SystemType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// ──────────────────────────── PocError ────────────────────────────

/// Errors emitted by the PoC framework during trial execution.
#[derive(Debug, Error)]
pub enum PocError {
    /// The hypothesis control group must be the PostgreSQL/pgvector baseline.
    #[error("control group must be postgres_pgvector, got {0:?}")]
    InvalidControlGroup(String),

    /// The experimental group must differ from the control group.
    #[error("experimental group must differ from control group: both are {0:?}")]
    SameGroup(String),

    /// Success threshold values must be within valid ranges (0.0–1.0 for
    /// ratios, > 0 for latency).
    #[error("invalid success threshold: {0}")]
    InvalidThreshold(String),

    /// Data boundary limits must be positive.
    #[error("invalid data boundary: {0}")]
    InvalidBoundary(String),

    /// At least one stop condition must be configured.
    #[error("at least one stop condition is required")]
    NoStopConditions,
}

// ──────────────────────────── FocusArea ────────────────────────────

/// A single evaluation focus area within a PoC spec.
///
/// Each focus area maps to a specific capability described in §3.6 (Mem0)
/// or §3.7 (Cognee). Disabled focus areas are tracked but excluded from
/// the final recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusArea {
    /// Short identifier (e.g. `"conflict_merge"`).
    pub name: String,
    /// Human-readable description of what this area evaluates.
    pub description: String,
    /// Whether this focus area is active in the current trial.
    pub enabled: bool,
    /// Optional target metric name this area maps to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_metric: Option<String>,
}

impl FocusArea {
    /// Create a new enabled focus area.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            enabled: true,
            target_metric: None,
        }
    }

    /// Create a new enabled focus area with a target metric.
    pub fn with_metric(name: &str, description: &str, metric: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            enabled: true,
            target_metric: Some(metric.to_string()),
        }
    }
}

// ──────────────────────────── SuccessThreshold ────────────────────────────

/// Frozen success criteria for a PoC hypothesis.
///
/// All four conditions must be satisfied simultaneously for a trial to
/// be eligible for adoption (§13.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessThreshold {
    /// Minimum recall@k the experimental system must achieve.
    pub min_recall_at_k: f64,
    /// Minimum precision@k the experimental system must achieve.
    pub min_precision_at_k: f64,
    /// Maximum P95 latency in milliseconds.
    pub max_latency_p95_ms: u64,
    /// Minimum groundedness score (0.0–1.0).
    pub min_groundedness: f64,
}

impl Default for SuccessThreshold {
    fn default() -> Self {
        Self {
            min_recall_at_k: 0.80,
            min_precision_at_k: 0.75,
            max_latency_p95_ms: 200,
            min_groundedness: 0.70,
        }
    }
}

impl SuccessThreshold {
    /// Returns `true` when all four thresholds are met by `metrics`.
    pub fn all_met(&self, m: &PocMetrics) -> bool {
        m.recall_at_k >= self.min_recall_at_k
            && m.precision_at_k >= self.min_precision_at_k
            && m.latency_p95_ms <= self.max_latency_p95_ms
            && m.groundedness >= self.min_groundedness
    }

    /// Returns `true` when any single metric is below 50 % of its target —
    /// indicating catastrophic failure that warrants immediate rejection.
    pub fn is_catastrophic(&self, m: &PocMetrics) -> bool {
        m.recall_at_k < self.min_recall_at_k * 0.5
            || m.precision_at_k < self.min_precision_at_k * 0.5
            || m.groundedness < self.min_groundedness * 0.5
    }

    /// Validate that threshold values are within sensible ranges.
    pub fn validate(&self) -> Result<(), PocError> {
        if !(0.0..=1.0).contains(&self.min_recall_at_k) {
            return Err(PocError::InvalidThreshold(
                "min_recall_at_k must be in [0.0, 1.0]".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_precision_at_k) {
            return Err(PocError::InvalidThreshold(
                "min_precision_at_k must be in [0.0, 1.0]".into(),
            ));
        }
        if self.max_latency_p95_ms == 0 {
            return Err(PocError::InvalidThreshold(
                "max_latency_p95_ms must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_groundedness) {
            return Err(PocError::InvalidThreshold(
                "min_groundedness must be in [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

// ──────────────────────────── StopCondition ────────────────────────────

/// Conditions under which a PoC trial terminates.
///
/// Each variant maps to a specific §13.5 stop condition. The runner
/// evaluates these in priority order: budget → data boundary →
/// threshold met → threshold failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopCondition {
    /// Success criteria achieved — trial can stop with a positive verdict.
    ThresholdMet,
    /// Cannot meet success criteria after the configured iteration budget.
    ThresholdFailed,
    /// Compute, time, or cost budget exhausted before a conclusion was reached.
    BudgetExhausted,
    /// Test data set exhausted before a conclusion was reached.
    DataBoundaryReached,
}

impl StopCondition {
    /// Human-readable label for reports.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ThresholdMet => "threshold_met",
            Self::ThresholdFailed => "threshold_failed",
            Self::BudgetExhausted => "budget_exhausted",
            Self::DataBoundaryReached => "data_boundary_reached",
        }
    }

    /// Returns the default set of all four stop conditions to monitor.
    pub fn all() -> Vec<Self> {
        vec![
            Self::ThresholdMet,
            Self::ThresholdFailed,
            Self::BudgetExhausted,
            Self::DataBoundaryReached,
        ]
    }
}

// ──────────────────────────── DataBoundary ────────────────────────────

/// Hard limits on test data, iterations, time, and cost (§13.5).
///
/// These boundaries are frozen at hypothesis creation and cannot be
/// relaxed mid-trial — any breach triggers a stop condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBoundary {
    /// Maximum number of test items (queries, scenarios) to evaluate.
    pub max_test_items: usize,
    /// Maximum number of iterations (full passes over the dataset).
    pub max_iterations: usize,
    /// Wall-clock time limit in hours.
    pub time_limit_hours: f64,
    /// Maximum cumulative cost in USD.
    pub cost_limit_usd: f64,
}

impl Default for DataBoundary {
    fn default() -> Self {
        Self {
            max_test_items: 1000,
            max_iterations: 10,
            time_limit_hours: 24.0,
            cost_limit_usd: 500.0,
        }
    }
}

impl DataBoundary {
    /// Returns `true` when cost or wall-clock time budget is exhausted.
    ///
    /// Iteration exhaustion is **not** included here — it maps to
    /// [`StopCondition::ThresholdFailed`] ("can't meet criteria after N
    /// iterations") rather than a budget stop.
    pub fn is_budget_exhausted(&self, m: &PocMetrics) -> bool {
        m.cost_usd >= self.cost_limit_usd || m.time_elapsed_hours >= self.time_limit_hours
    }

    /// Returns `true` when the iteration budget is exhausted.
    pub fn is_iterations_exhausted(&self, m: &PocMetrics) -> bool {
        m.iterations_run >= self.max_iterations
    }

    /// Returns `true` when the test-item boundary is reached.
    pub fn is_data_exhausted(&self, m: &PocMetrics) -> bool {
        m.test_items_evaluated >= self.max_test_items
    }

    /// Validate that all boundary values are positive.
    pub fn validate(&self) -> Result<(), PocError> {
        if self.max_test_items == 0 {
            return Err(PocError::InvalidBoundary(
                "max_test_items must be > 0".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(PocError::InvalidBoundary(
                "max_iterations must be > 0".into(),
            ));
        }
        if self.time_limit_hours <= 0.0 {
            return Err(PocError::InvalidBoundary(
                "time_limit_hours must be > 0".into(),
            ));
        }
        if self.cost_limit_usd <= 0.0 {
            return Err(PocError::InvalidBoundary(
                "cost_limit_usd must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ──────────────────────────── PocHypothesis ────────────────────────────

/// A frozen hypothesis that governs a PoC trial.
///
/// Once created, none of these fields should change during the trial —
/// this is the scientific contract required by §13.5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocHypothesis {
    /// The statement being tested (e.g. "Mem0 improves Chinese recall by ≥10 %").
    pub hypothesis_statement: String,
    /// Variables the experimental system changes (e.g. "merge_strategy", "language_model").
    pub independent_variables: Vec<String>,
    /// The baseline system (always PostgreSQL/pgvector).
    pub control_group: SystemType,
    /// The system under test (Mem0 or Cognee).
    pub experimental_group: SystemType,
    /// Frozen success criteria.
    pub success_threshold: SuccessThreshold,
    /// Stop conditions to monitor during the trial.
    pub stop_conditions: Vec<StopCondition>,
    /// Hard limits on data, time, and cost.
    pub data_boundary: DataBoundary,
}

impl Default for PocHypothesis {
    fn default() -> Self {
        Self {
            hypothesis_statement: String::new(),
            independent_variables: Vec::new(),
            control_group: SystemType::PostgresPgvector,
            experimental_group: SystemType::Mem0,
            success_threshold: SuccessThreshold::default(),
            stop_conditions: StopCondition::all(),
            data_boundary: DataBoundary::default(),
        }
    }
}

impl PocHypothesis {
    /// Validate the hypothesis against §13.5 constraints.
    pub fn validate(&self) -> Result<(), PocError> {
        if self.control_group != SystemType::PostgresPgvector {
            return Err(PocError::InvalidControlGroup(
                self.control_group.as_str().into(),
            ));
        }
        if self.control_group == self.experimental_group {
            return Err(PocError::SameGroup(self.control_group.as_str().into()));
        }
        self.success_threshold.validate()?;
        self.data_boundary.validate()?;
        if self.stop_conditions.is_empty() {
            return Err(PocError::NoStopConditions);
        }
        if self.hypothesis_statement.is_empty() {
            return Err(PocError::InvalidThreshold(
                "hypothesis_statement must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Create a builder-style hypothesis with a statement and experimental group.
    pub fn new(statement: &str, experimental: SystemType) -> Self {
        Self {
            hypothesis_statement: statement.to_string(),
            experimental_group: experimental,
            ..Default::default()
        }
    }
}

// ──────────────────────────── PocMetrics ────────────────────────────

/// Quality and usage metrics collected for one system (baseline or experimental)
/// during a PoC trial.
///
/// This struct bundles the four success-threshold metrics with usage tracking
/// so the runner can evaluate stop conditions in a single pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PocMetrics {
    // ── Quality metrics ──
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub latency_p95_ms: u64,
    pub groundedness: f64,

    // ── Usage / cost tracking ──
    pub cost_usd: f64,
    pub test_items_evaluated: usize,
    pub iterations_run: usize,
    pub time_elapsed_hours: f64,
}

impl PocMetrics {
    /// Create a metrics snapshot from the four core quality values.
    pub fn from_quality(recall: f64, precision: f64, latency_p95: u64, groundedness: f64) -> Self {
        Self {
            recall_at_k: recall,
            precision_at_k: precision,
            latency_p95_ms: latency_p95,
            groundedness,
            ..Default::default()
        }
    }
}

// ──────────────────────────── PocComparison ────────────────────────────

/// Delta between experimental and baseline metrics.
///
/// Positive deltas for recall/precision/groundedness mean the experimental
/// system is better. Positive `latency_delta_ms` means the experimental
/// system is slower. Positive `cost_delta_usd` means more expensive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PocComparison {
    pub recall_delta: f64,
    pub precision_delta: f64,
    pub latency_delta_ms: i64,
    pub cost_delta_usd: f64,
    pub groundedness_delta: f64,
}

impl PocComparison {
    /// Compute deltas as `experimental - baseline`.
    pub fn compute(baseline: &PocMetrics, experimental: &PocMetrics) -> Self {
        Self {
            recall_delta: experimental.recall_at_k - baseline.recall_at_k,
            precision_delta: experimental.precision_at_k - baseline.precision_at_k,
            latency_delta_ms: experimental.latency_p95_ms as i64 - baseline.latency_p95_ms as i64,
            cost_delta_usd: experimental.cost_usd - baseline.cost_usd,
            groundedness_delta: experimental.groundedness - baseline.groundedness,
        }
    }

    /// Returns `true` when the experimental system is strictly better than
    /// baseline on recall (the primary adoption signal).
    pub fn experimental_better_recall(&self) -> bool {
        self.recall_delta > 0.0
    }
}

// ──────────────────────────── PocVerdict ────────────────────────────

/// Final recommendation for the experimental system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PocVerdict {
    /// Success thresholds met and experimental outperforms baseline — adopt.
    Adopt,
    /// Success thresholds clearly not met — do not adopt.
    Reject,
    /// Results are mixed or inconclusive — further trials needed.
    Inconclusive,
}

impl PocVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Adopt => "adopt",
            Self::Reject => "reject",
            Self::Inconclusive => "inconclusive",
        }
    }
}

// ──────────────────────────── PocResult ────────────────────────────

/// The complete outcome of a single PoC trial evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocResult {
    /// The frozen hypothesis that was tested.
    pub hypothesis: PocHypothesis,
    /// Metrics from the control group (PostgreSQL/pgvector).
    pub baseline_metrics: PocMetrics,
    /// Metrics from the experimental system (Mem0 or Cognee).
    pub experimental_metrics: PocMetrics,
    /// Delta between the two metric sets.
    pub comparison: PocComparison,
    /// Final recommendation.
    pub verdict: PocVerdict,
    /// Which stop condition terminated the trial.
    pub stop_reason: StopCondition,
}

// ──────────────────────────── Mem0PocSpec ────────────────────────────

/// Mem0-specific PoC specification (§3.6).
///
/// Encodes the six focus areas required by §3.6 for Mem0 evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mem0PocSpec {
    pub hypothesis: PocHypothesis,
    /// §3.6: conflict merge — contradictory memories resolved correctly.
    pub conflict_merge: FocusArea,
    /// §3.6: delete propagation — deletes cascade to derived memories.
    pub delete_propagation: FocusArea,
    /// §3.6: idempotency — repeated writes don't duplicate.
    pub idempotency: FocusArea,
    /// §3.6: Chinese recall quality — CJK tokenisation and recall.
    pub chinese_recall_quality: FocusArea,
    /// §3.6: explainability — memory provenance traceable to source.
    pub explainability: FocusArea,
    /// §3.6: self-hosted operations — operational feasibility on-prem.
    pub self_hosted_operations: FocusArea,
}

impl Mem0PocSpec {
    /// Create a spec with the standard §3.6 focus areas and a hypothesis.
    pub fn new(hypothesis: PocHypothesis) -> Self {
        Self {
            hypothesis,
            conflict_merge: FocusArea::with_metric(
                "conflict_merge",
                "Contradictory memories resolved without data loss",
                "conflict_rate",
            ),
            delete_propagation: FocusArea::with_metric(
                "delete_propagation",
                "Deletes cascade to derived and linked memories",
                "stale_memory_rate",
            ),
            idempotency: FocusArea::with_metric(
                "idempotency",
                "Repeated writes produce no duplicates",
                "duplicate_rate",
            ),
            chinese_recall_quality: FocusArea::with_metric(
                "chinese_recall_quality",
                "CJK tokenisation preserves recall on Chinese queries",
                "recall_at_k",
            ),
            explainability: FocusArea::with_metric(
                "explainability",
                "Every surfaced memory traces to a source document",
                "groundedness",
            ),
            self_hosted_operations: FocusArea::new(
                "self_hosted_operations",
                "Operational feasibility when self-hosted on-premises",
            ),
        }
    }

    /// Collect all focus areas.
    pub fn focus_areas(&self) -> Vec<&FocusArea> {
        vec![
            &self.conflict_merge,
            &self.delete_propagation,
            &self.idempotency,
            &self.chinese_recall_quality,
            &self.explainability,
            &self.self_hosted_operations,
        ]
    }
}

// ──────────────────────────── CogneePocSpec ────────────────────────────

/// Cognee-specific PoC specification (§3.7).
///
/// Encodes the eight focus areas required by §3.7 for Cognee evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CogneePocSpec {
    pub hypothesis: PocHypothesis,
    /// §3.7: multi-hop question accuracy.
    pub multi_hop_accuracy: FocusArea,
    /// §3.7: entity duplication rate — low is better.
    pub entity_duplication_rate: FocusArea,
    /// §3.7: relationship correctness rate.
    pub relationship_correctness: FocusArea,
    /// §3.7: source tracing — answers traceable to source nodes.
    pub source_tracing: FocusArea,
    /// §3.7: ACL mapping — access control preserved in graph.
    pub acl_mapping: FocusArea,
    /// §3.7: incremental update — graph updates without full rebuild.
    pub incremental_update: FocusArea,
    /// §3.7: delete propagation — deletes remove affected edges/nodes.
    pub delete_propagation: FocusArea,
    /// §3.7: latency and total cost trade-off.
    pub latency_and_cost: FocusArea,
}

impl CogneePocSpec {
    /// Create a spec with the standard §3.7 focus areas and a hypothesis.
    pub fn new(hypothesis: PocHypothesis) -> Self {
        Self {
            hypothesis,
            multi_hop_accuracy: FocusArea::with_metric(
                "multi_hop_accuracy",
                "Multi-hop questions answered correctly via graph traversal",
                "recall_at_k",
            ),
            entity_duplication_rate: FocusArea::with_metric(
                "entity_duplication_rate",
                "Duplicate entity nodes minimised",
                "entity_duplication_rate",
            ),
            relationship_correctness: FocusArea::with_metric(
                "relationship_correctness",
                "Extracted relationships match ground truth",
                "relationship_correctness_rate",
            ),
            source_tracing: FocusArea::with_metric(
                "source_tracing",
                "Answer paths traceable to source documents",
                "groundedness",
            ),
            acl_mapping: FocusArea::with_metric(
                "acl_mapping",
                "Access control levels preserved on graph edges",
                "acl_violation_rate",
            ),
            incremental_update: FocusArea::new(
                "incremental_update",
                "Graph updates incrementally without full rebuild",
            ),
            delete_propagation: FocusArea::new(
                "delete_propagation",
                "Deletes remove affected nodes and edges correctly",
            ),
            latency_and_cost: FocusArea::with_metric(
                "latency_and_cost",
                "Latency and total cost within acceptable bounds",
                "latency_p95_ms",
            ),
        }
    }

    /// Collect all focus areas.
    pub fn focus_areas(&self) -> Vec<&FocusArea> {
        vec![
            &self.multi_hop_accuracy,
            &self.entity_duplication_rate,
            &self.relationship_correctness,
            &self.source_tracing,
            &self.acl_mapping,
            &self.incremental_update,
            &self.delete_propagation,
            &self.latency_and_cost,
        ]
    }
}

// ──────────────────────────── PocSpec ────────────────────────────

/// Tagged union of Mem0 and Cognee PoC specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "system", rename_all = "snake_case")]
pub enum PocSpec {
    Mem0(Mem0PocSpec),
    Cognee(CogneePocSpec),
}

impl PocSpec {
    /// Borrow the frozen hypothesis from whichever variant is active.
    pub fn hypothesis(&self) -> &PocHypothesis {
        match self {
            Self::Mem0(s) => &s.hypothesis,
            Self::Cognee(s) => &s.hypothesis,
        }
    }

    /// The experimental system under test.
    pub fn experimental_group(&self) -> SystemType {
        match self {
            Self::Mem0(_) => SystemType::Mem0,
            Self::Cognee(_) => SystemType::Cognee,
        }
    }

    /// The baseline control group (from the hypothesis).
    pub fn control_group(&self) -> SystemType {
        self.hypothesis().control_group
    }

    /// Collect all focus areas from the active variant.
    pub fn focus_areas(&self) -> Vec<&FocusArea> {
        match self {
            Self::Mem0(s) => s.focus_areas(),
            Self::Cognee(s) => s.focus_areas(),
        }
    }
}

// ──────────────────────────── PocRunner ────────────────────────────

/// Evaluates PoC trial metrics against a frozen hypothesis.
///
/// The runner is stateless — it takes pre-collected metrics snapshots and
/// produces a verdict. The caller is responsible for actually running the
/// baseline and experimental systems and collecting metrics.
#[derive(Debug, Clone)]
pub struct PocRunner {
    /// The `k` value used for recall@k / precision@k (informational).
    pub k: usize,
}

impl Default for PocRunner {
    fn default() -> Self {
        Self { k: 10 }
    }
}

impl PocRunner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_k(k: usize) -> Self {
        Self { k }
    }

    /// Run a PoC evaluation against pre-collected baseline and experimental
    /// metrics.
    ///
    /// The method computes the comparison, evaluates stop conditions in
    /// priority order (budget → data → threshold-met → threshold-failed),
    /// and returns the full result.
    pub async fn run_poc(
        &self,
        spec: &PocSpec,
        baseline_metrics: &PocMetrics,
        experimental_metrics: &PocMetrics,
    ) -> Result<PocResult, PocError> {
        let hypothesis = spec.hypothesis().clone();
        hypothesis.validate()?;

        let comparison = PocComparison::compute(baseline_metrics, experimental_metrics);
        let (stop_reason, verdict) =
            self.evaluate(&hypothesis, baseline_metrics, experimental_metrics);

        Ok(PocResult {
            hypothesis,
            baseline_metrics: baseline_metrics.clone(),
            experimental_metrics: experimental_metrics.clone(),
            comparison,
            verdict,
            stop_reason,
        })
    }

    /// Determine which stop condition triggered and what verdict to render.
    ///
    /// Evaluation priority:
    /// 1. **Budget exhausted** — if configured and budget is spent, trial stops.
    /// 2. **Data boundary reached** — if configured and data is exhausted.
    /// 3. **Threshold met** — if configured and all success thresholds pass.
    /// 4. **Threshold failed** — fallback when thresholds aren't met.
    fn evaluate(
        &self,
        hypothesis: &PocHypothesis,
        baseline: &PocMetrics,
        experimental: &PocMetrics,
    ) -> (StopCondition, PocVerdict) {
        let configured = &hypothesis.stop_conditions;
        let threshold = &hypothesis.success_threshold;
        let boundary = &hypothesis.data_boundary;

        let all_met = threshold.all_met(experimental);

        // 1. Budget exhaustion (highest-priority safety stop).
        if configured.contains(&StopCondition::BudgetExhausted)
            && boundary.is_budget_exhausted(experimental)
        {
            let verdict = if all_met && experimental.recall_at_k >= baseline.recall_at_k {
                PocVerdict::Adopt
            } else if all_met {
                PocVerdict::Inconclusive
            } else {
                PocVerdict::Inconclusive
            };
            return (StopCondition::BudgetExhausted, verdict);
        }

        // 2. Data boundary reached.
        if configured.contains(&StopCondition::DataBoundaryReached)
            && boundary.is_data_exhausted(experimental)
        {
            let verdict = if all_met && experimental.recall_at_k >= baseline.recall_at_k {
                PocVerdict::Adopt
            } else {
                PocVerdict::Inconclusive
            };
            return (StopCondition::DataBoundaryReached, verdict);
        }

        // 3. Threshold met — success.
        if configured.contains(&StopCondition::ThresholdMet) && all_met {
            if experimental.recall_at_k >= baseline.recall_at_k {
                return (StopCondition::ThresholdMet, PocVerdict::Adopt);
            }
            // Met absolute thresholds but worse than baseline — inconclusive.
            return (StopCondition::ThresholdMet, PocVerdict::Inconclusive);
        }

        // 4. Threshold failed — either catastrophic or exhausted iterations.
        if configured.contains(&StopCondition::ThresholdFailed) {
            if threshold.is_catastrophic(experimental) {
                return (StopCondition::ThresholdFailed, PocVerdict::Reject);
            }
            let iterations_exhausted = experimental.iterations_run >= boundary.max_iterations;
            if iterations_exhausted {
                return (StopCondition::ThresholdFailed, PocVerdict::Reject);
            }
            // Not met, not catastrophic, iterations remaining → inconclusive.
            return (StopCondition::ThresholdFailed, PocVerdict::Inconclusive);
        }

        // No configured stop condition triggered — default to inconclusive.
        (StopCondition::ThresholdFailed, PocVerdict::Inconclusive)
    }
}

// ──────────────────────────── PocFramework ────────────────────────────

/// Top-level orchestrator for PoC trials against the pgvector baseline.
///
/// Wraps [`PocRunner`] and provides convenience methods for running
/// trials and generating serializable reports.
#[derive(Debug, Clone)]
pub struct PocFramework {
    pub runner: PocRunner,
}

impl Default for PocFramework {
    fn default() -> Self {
        Self {
            runner: PocRunner::new(),
        }
    }
}

impl PocFramework {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_runner(runner: PocRunner) -> Self {
        Self { runner }
    }

    /// Run a single PoC trial — delegates to the inner runner.
    pub async fn run_trial(
        &self,
        spec: &PocSpec,
        baseline_metrics: &PocMetrics,
        experimental_metrics: &PocMetrics,
    ) -> Result<PocResult, PocError> {
        self.runner
            .run_poc(spec, baseline_metrics, experimental_metrics)
            .await
    }

    /// Generate a serializable report from a completed trial result.
    pub fn generate_report(&self, result: PocResult, spec: PocSpec) -> PocReport {
        PocReport::from_result(result, spec)
    }
}

// ──────────────────────────── PocReport ────────────────────────────

/// Serializable final report containing all trial metrics and a recommendation.
///
/// This is the artifact that gets persisted to `docs/artifacts/` per the
/// artifact-persistence runbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocReport {
    /// Unique trial identifier.
    pub trial_id: Uuid,
    /// When the report was generated.
    pub generated_at: DateTime<Utc>,
    /// The PoC specification (Mem0 or Cognee focus areas + hypothesis).
    pub spec: PocSpec,
    /// The full trial result.
    pub result: PocResult,
    /// Human-readable recommendation string.
    pub recommendation: String,
}

impl PocReport {
    /// Build a report from a result and spec, generating a recommendation
    /// based on the verdict.
    pub fn from_result(result: PocResult, spec: PocSpec) -> Self {
        let system = spec.experimental_group();
        let recommendation = match result.verdict {
            PocVerdict::Adopt => format!(
                "Adopt {} — all success thresholds met and recall improved by {:.1} % over the pgvector baseline.",
                system.as_str(),
                result.comparison.recall_delta * 100.0,
            ),
            PocVerdict::Reject => format!(
                "Reject {} — success thresholds not met (stop: {}).",
                system.as_str(),
                result.stop_reason.as_str(),
            ),
            PocVerdict::Inconclusive => format!(
                "Inconclusive for {} — further trials needed (stop: {}).",
                system.as_str(),
                result.stop_reason.as_str(),
            ),
        };
        Self {
            trial_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            spec,
            result,
            recommendation,
        }
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. SuccessThreshold default values ──

    #[test]
    fn success_threshold_defaults() {
        let t = SuccessThreshold::default();
        assert!((t.min_recall_at_k - 0.80).abs() < 1e-9);
        assert!((t.min_precision_at_k - 0.75).abs() < 1e-9);
        assert_eq!(t.max_latency_p95_ms, 200);
        assert!((t.min_groundedness - 0.70).abs() < 1e-9);
    }

    // ── 2. SuccessThreshold::all_met — all pass ──

    #[test]
    fn threshold_all_met() {
        let t = SuccessThreshold::default();
        let m = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        assert!(t.all_met(&m));
    }

    // ── 3. SuccessThreshold::all_met — recall fails ──

    #[test]
    fn threshold_recall_not_met() {
        let t = SuccessThreshold::default();
        let m = PocMetrics::from_quality(0.50, 0.80, 150, 0.85);
        assert!(!t.all_met(&m));
    }

    // ── 4. SuccessThreshold::all_met — latency fails ──

    #[test]
    fn threshold_latency_not_met() {
        let t = SuccessThreshold::default();
        let m = PocMetrics::from_quality(0.90, 0.80, 300, 0.85);
        assert!(!t.all_met(&m));
    }

    // ── 5. SuccessThreshold::is_catastrophic ──

    #[test]
    fn threshold_catastrophic() {
        let t = SuccessThreshold::default();
        // recall < 50% of 0.80 = 0.40 → catastrophic
        let m = PocMetrics::from_quality(0.30, 0.80, 150, 0.85);
        assert!(t.is_catastrophic(&m));
    }

    // ── 6. SuccessThreshold::is_catastrophic — not catastrophic ──

    #[test]
    fn threshold_not_catastrophic() {
        let t = SuccessThreshold::default();
        let m = PocMetrics::from_quality(0.60, 0.80, 150, 0.85);
        assert!(!t.is_catastrophic(&m));
    }

    // ── 7. SuccessThreshold::validate — valid ──

    #[test]
    fn threshold_validate_ok() {
        assert!(SuccessThreshold::default().validate().is_ok());
    }

    // ── 8. SuccessThreshold::validate — invalid recall ──

    #[test]
    fn threshold_validate_bad_recall() {
        let t = SuccessThreshold {
            min_recall_at_k: 1.5,
            ..Default::default()
        };
        assert!(t.validate().is_err());
    }

    // ── 9. StopCondition::all returns four variants ──

    #[test]
    fn stop_condition_all_four() {
        let all = StopCondition::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&StopCondition::ThresholdMet));
        assert!(all.contains(&StopCondition::ThresholdFailed));
        assert!(all.contains(&StopCondition::BudgetExhausted));
        assert!(all.contains(&StopCondition::DataBoundaryReached));
    }

    // ── 10. StopCondition serde round-trip ──

    #[test]
    fn stop_condition_serde_roundtrip() {
        let sc = StopCondition::ThresholdMet;
        let json = serde_json::to_string(&sc).unwrap();
        assert!(json.contains("threshold_met"));
        let restored: StopCondition = serde_json::from_str(&json).unwrap();
        assert_eq!(sc, restored);
    }

    // ── 11. PocVerdict serde round-trip ──

    #[test]
    fn verdict_serde_roundtrip() {
        let v = PocVerdict::Adopt;
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("adopt"));
        let restored: PocVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, restored);
    }

    // ── 12. DataBoundary defaults ──

    #[test]
    fn data_boundary_defaults() {
        let b = DataBoundary::default();
        assert_eq!(b.max_test_items, 1000);
        assert_eq!(b.max_iterations, 10);
        assert!((b.time_limit_hours - 24.0).abs() < 1e-9);
        assert!((b.cost_limit_usd - 500.0).abs() < 1e-9);
    }

    // ── 13. DataBoundary::is_budget_exhausted ──

    #[test]
    fn data_boundary_budget_exhausted_by_cost() {
        let b = DataBoundary::default();
        let m = PocMetrics {
            cost_usd: 600.0,
            ..Default::default()
        };
        assert!(b.is_budget_exhausted(&m));
    }

    // ── 14. DataBoundary::is_data_exhausted ──

    #[test]
    fn data_boundary_data_exhausted() {
        let b = DataBoundary::default();
        let m = PocMetrics {
            test_items_evaluated: 1000,
            ..Default::default()
        };
        assert!(b.is_data_exhausted(&m));
    }

    // ── 15. PocMetrics::from_quality ──

    #[test]
    fn poc_metrics_from_quality() {
        let m = PocMetrics::from_quality(0.9, 0.8, 120, 0.7);
        assert!((m.recall_at_k - 0.9).abs() < 1e-9);
        assert!((m.precision_at_k - 0.8).abs() < 1e-9);
        assert_eq!(m.latency_p95_ms, 120);
        assert!((m.groundedness - 0.7).abs() < 1e-9);
        assert_eq!(m.cost_usd, 0.0);
        assert_eq!(m.test_items_evaluated, 0);
    }

    // ── 16. PocComparison::compute ──

    #[test]
    fn comparison_compute_deltas() {
        let baseline = PocMetrics::from_quality(0.70, 0.60, 150, 0.65);
        let experimental = PocMetrics::from_quality(0.85, 0.75, 120, 0.80);
        let c = PocComparison::compute(&baseline, &experimental);
        assert!((c.recall_delta - 0.15).abs() < 1e-9);
        assert!((c.precision_delta - 0.15).abs() < 1e-9);
        assert_eq!(c.latency_delta_ms, -30);
        assert!((c.groundedness_delta - 0.15).abs() < 1e-9);
        assert!(c.experimental_better_recall());
    }

    // ── 17. Mem0PocSpec::new has six focus areas ──

    #[test]
    fn mem0_spec_six_focus_areas() {
        let spec = Mem0PocSpec::new(PocHypothesis::new("test", SystemType::Mem0));
        let areas = spec.focus_areas();
        assert_eq!(areas.len(), 6);
        assert!(areas.iter().all(|a| a.enabled));
        let names: Vec<&str> = areas.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"conflict_merge"));
        assert!(names.contains(&"chinese_recall_quality"));
        assert!(names.contains(&"self_hosted_operations"));
    }

    // ── 18. CogneePocSpec::new has eight focus areas ──

    #[test]
    fn cognee_spec_eight_focus_areas() {
        let spec = CogneePocSpec::new(PocHypothesis::new("test", SystemType::Cognee));
        let areas = spec.focus_areas();
        assert_eq!(areas.len(), 8);
        let names: Vec<&str> = areas.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"multi_hop_accuracy"));
        assert!(names.contains(&"acl_mapping"));
        assert!(names.contains(&"latency_and_cost"));
    }

    // ── 19. PocSpec accessors — Mem0 ──

    #[test]
    fn poc_spec_mem0_accessors() {
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new(
            "mem0 test",
            SystemType::Mem0,
        )));
        assert_eq!(spec.experimental_group(), SystemType::Mem0);
        assert_eq!(spec.control_group(), SystemType::PostgresPgvector);
        assert_eq!(spec.hypothesis().experimental_group, SystemType::Mem0);
        assert_eq!(spec.focus_areas().len(), 6);
    }

    // ── 20. PocSpec accessors — Cognee ──

    #[test]
    fn poc_spec_cognee_accessors() {
        let spec = PocSpec::Cognee(CogneePocSpec::new(PocHypothesis::new(
            "cognee test",
            SystemType::Cognee,
        )));
        assert_eq!(spec.experimental_group(), SystemType::Cognee);
        assert_eq!(spec.control_group(), SystemType::PostgresPgvector);
        assert_eq!(spec.focus_areas().len(), 8);
    }

    // ── 21. run_poc — adopt verdict ──

    #[tokio::test]
    async fn run_poc_adopt() {
        let runner = PocRunner::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert_eq!(result.verdict, PocVerdict::Adopt);
        assert_eq!(result.stop_reason, StopCondition::ThresholdMet);
    }

    // ── 22. run_poc — reject verdict (catastrophic) ──

    #[tokio::test]
    async fn run_poc_reject_catastrophic() {
        let runner = PocRunner::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.20, 0.80, 150, 0.85);
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert_eq!(result.verdict, PocVerdict::Reject);
        assert_eq!(result.stop_reason, StopCondition::ThresholdFailed);
    }

    // ── 23. run_poc — reject verdict (iterations exhausted) ──

    #[tokio::test]
    async fn run_poc_reject_iterations_exhausted() {
        let runner = PocRunner::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        // Not catastrophic but not meeting thresholds; iterations exhausted.
        let experimental = PocMetrics {
            recall_at_k: 0.75,
            precision_at_k: 0.70,
            latency_p95_ms: 250,
            groundedness: 0.65,
            iterations_run: 10,
            ..Default::default()
        };
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert_eq!(result.verdict, PocVerdict::Reject);
        assert_eq!(result.stop_reason, StopCondition::ThresholdFailed);
    }

    // ── 24. run_poc — inconclusive (thresholds not met, not exhausted) ──

    #[tokio::test]
    async fn run_poc_inconclusive() {
        let runner = PocRunner::new();
        let spec = PocSpec::Cognee(CogneePocSpec::new(PocHypothesis::new(
            "h",
            SystemType::Cognee,
        )));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        // Below thresholds but not catastrophic, iterations remaining.
        let experimental = PocMetrics::from_quality(0.75, 0.70, 180, 0.65);
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert_eq!(result.verdict, PocVerdict::Inconclusive);
        assert_eq!(result.stop_reason, StopCondition::ThresholdFailed);
    }

    // ── 25. run_poc — budget exhausted ──

    #[tokio::test]
    async fn run_poc_budget_exhausted() {
        let runner = PocRunner::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics {
            recall_at_k: 0.75,
            precision_at_k: 0.70,
            latency_p95_ms: 180,
            groundedness: 0.65,
            cost_usd: 600.0, // exceeds default 500
            ..Default::default()
        };
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopCondition::BudgetExhausted);
        assert_eq!(result.verdict, PocVerdict::Inconclusive);
    }

    // ── 26. run_poc — data boundary reached ──

    #[tokio::test]
    async fn run_poc_data_boundary_reached() {
        let runner = PocRunner::new();
        let spec = PocSpec::Cognee(CogneePocSpec::new(PocHypothesis::new(
            "h",
            SystemType::Cognee,
        )));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics {
            recall_at_k: 0.75,
            precision_at_k: 0.70,
            latency_p95_ms: 180,
            groundedness: 0.65,
            test_items_evaluated: 1000, // equals default max_test_items
            ..Default::default()
        };
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopCondition::DataBoundaryReached);
    }

    // ── 27. run_poc — threshold met but worse than baseline → Inconclusive ──

    #[tokio::test]
    async fn run_poc_threshold_met_worse_than_baseline() {
        let runner = PocRunner::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        // Baseline has higher recall but experimental still meets absolute thresholds.
        let baseline = PocMetrics::from_quality(0.95, 0.90, 100, 0.90);
        let experimental = PocMetrics::from_quality(0.85, 0.80, 150, 0.75);
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert_eq!(result.stop_reason, StopCondition::ThresholdMet);
        assert_eq!(result.verdict, PocVerdict::Inconclusive);
    }

    // ── 28. run_poc — validation error (empty hypothesis) ──

    #[tokio::test]
    async fn run_poc_validation_error_empty_statement() {
        let runner = PocRunner::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::default()));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        let result = runner.run_poc(&spec, &baseline, &experimental).await;
        assert!(result.is_err());
    }

    // ── 29. run_poc — validation error (same group) ──

    #[tokio::test]
    async fn run_poc_validation_error_same_group() {
        let runner = PocRunner::new();
        let mut h = PocHypothesis::new("h", SystemType::Mem0);
        h.control_group = SystemType::Mem0; // same as experimental
        let spec = PocSpec::Mem0(Mem0PocSpec::new(h));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        let result = runner.run_poc(&spec, &baseline, &experimental).await;
        assert!(result.is_err());
    }

    // ── 30. PocFramework::run_trial delegates to runner ──

    #[tokio::test]
    async fn framework_run_trial() {
        let fw = PocFramework::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        let result = fw.run_trial(&spec, &baseline, &experimental).await.unwrap();
        assert_eq!(result.verdict, PocVerdict::Adopt);
    }

    // ── 31. PocFramework::generate_report ──

    #[tokio::test]
    async fn framework_generate_report() {
        let fw = PocFramework::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        let result = fw.run_trial(&spec, &baseline, &experimental).await.unwrap();
        let report = fw.generate_report(result, spec);
        assert!(!report.recommendation.is_empty());
        assert!(report.recommendation.contains("Adopt"));
        assert!(!report.trial_id.to_string().is_empty());
    }

    // ── 32. PocReport serde round-trip ──

    #[tokio::test]
    async fn poc_report_serde_roundtrip() {
        let fw = PocFramework::new();
        let spec = PocSpec::Cognee(CogneePocSpec::new(PocHypothesis::new(
            "h",
            SystemType::Cognee,
        )));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        let result = fw.run_trial(&spec, &baseline, &experimental).await.unwrap();
        let report = fw.generate_report(result, spec);
        let json = serde_json::to_string(&report).unwrap();
        let restored: PocReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.result.verdict, PocVerdict::Adopt);
        assert!(restored.recommendation.contains("Adopt"));
    }

    // ── 33. PocHypothesis serde round-trip ──

    #[test]
    fn hypothesis_serde_roundtrip() {
        let h = PocHypothesis::new("Mem0 improves recall", SystemType::Mem0);
        let json = serde_json::to_string(&h).unwrap();
        let restored: PocHypothesis = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.hypothesis_statement, "Mem0 improves recall");
        assert_eq!(restored.control_group, SystemType::PostgresPgvector);
        assert_eq!(restored.experimental_group, SystemType::Mem0);
        assert_eq!(restored.stop_conditions.len(), 4);
    }

    // ── 34. PocHypothesis::validate — valid ──

    #[test]
    fn hypothesis_validate_ok() {
        let h = PocHypothesis::new("valid hypothesis", SystemType::Cognee);
        assert!(h.validate().is_ok());
    }

    // ── 35. PocHypothesis::validate — no stop conditions ──

    #[test]
    fn hypothesis_validate_no_stop_conditions() {
        let mut h = PocHypothesis::new("h", SystemType::Mem0);
        h.stop_conditions = Vec::new();
        assert!(h.validate().is_err());
    }

    // ── 36. PocResult contains correct comparison ──

    #[tokio::test]
    async fn result_contains_correct_comparison() {
        let runner = PocRunner::new();
        let spec = PocSpec::Mem0(Mem0PocSpec::new(PocHypothesis::new("h", SystemType::Mem0)));
        let baseline = PocMetrics::from_quality(0.70, 0.60, 180, 0.65);
        let experimental = PocMetrics::from_quality(0.90, 0.80, 150, 0.85);
        let result = runner
            .run_poc(&spec, &baseline, &experimental)
            .await
            .unwrap();
        assert!((result.comparison.recall_delta - 0.20).abs() < 1e-9);
        assert_eq!(result.comparison.latency_delta_ms, -30);
    }
}
