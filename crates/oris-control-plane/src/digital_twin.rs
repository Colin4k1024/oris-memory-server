//! Digital Twin Verification Loop (§13 Phase 4).
//!
//! Provides the bridge between the Memory Service and a Digital Twin
//! simulation, implementing the feedback loop:
//!
//! ```text
//! Memory ──inject──▶ Simulation ──run──▶ Result ──capture──▶ Memory
//!                                         │
//!                                         └──validate──▶ Historical
//! ```
//!
//! ## Architecture references
//!
//! - §13 Phase 4 — requires Memory + Digital Twin verification loop and a
//!   unified operations dashboard.
//! - §12.2 — defines memory quality metrics (recall, precision, groundedness,
//!   conflict rate, stale rate, evidence coverage) consumed by
//!   [`OperationsDashboard`].
//! - [`crate::eval`] — `BenchmarkRunner` and `QualityMetrics` patterns for
//!   quality-metric computation.
//! - [`crate::slo_monitoring`] — `SloMetrics` patterns for metric aggregation.
//! - `oris_memory_store::memory_types` — shared domain types (`Scope`,
//!   `AuthorityLevel`).

use chrono::Utc;
use oris_memory_store::memory_types::{AuthorityLevel, Scope};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use thiserror::Error;
use uuid::Uuid;

/// Minimum confidence for a simulation result to be considered verified.
const VERIFICATION_THRESHOLD: f64 = 0.70;

// ──────────────────────────── InjectedMemory ────────────────────────────

/// A memory injected into the digital twin for simulation.
///
/// Carries the data the twin needs (content, scope, authority, confidence)
/// so the simulation can weight inputs by source trustworthiness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectedMemory {
    pub id: Uuid,
    pub content: String,
    pub scope: Scope,
    pub entity_refs: Vec<String>,
    pub authority_level: AuthorityLevel,
    pub confidence: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// ──────────────────────────── MemoryInterface ────────────────────────────

/// Specifies which memories to inject into the twin.
///
/// Filters by scope, entity reference, and authority level.  An empty filter
/// vector means "no restriction on that dimension".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryInterface {
    pub scopes: Vec<Scope>,
    pub entity_refs: Vec<String>,
    pub authority_levels: Vec<AuthorityLevel>,
    /// Maximum number of memories to inject (`None` = unlimited).
    pub max_memories: Option<usize>,
}

// ──────────────────────────── SimulationConfig ────────────────────────────

/// Fidelity of the digital twin simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationFidelity {
    Low,
    Medium,
    High,
}

impl Default for SimulationFidelity {
    fn default() -> Self {
        Self::Medium
    }
}

/// Parameters controlling how the simulation executes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Number of simulation iterations / steps.
    pub iterations: u32,
    /// Simulated time horizon in hours (`None` = open-ended).
    pub time_horizon_hours: Option<u64>,
    /// Desired fidelity / granularity of the twin.
    pub fidelity: SimulationFidelity,
    /// Deterministic seed for reproducible runs (`None` = non-deterministic).
    pub random_seed: Option<u64>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            time_horizon_hours: None,
            fidelity: SimulationFidelity::default(),
            random_seed: None,
        }
    }
}

// ──────────────────────────── ResultCaptureConfig ────────────────────────────

/// Controls what gets captured from a simulation result back into memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResultCaptureConfig {
    /// Capture the simulation's textual outputs.
    pub capture_outputs: bool,
    /// Capture a metrics summary as a memory.
    pub capture_metrics: bool,
    /// Capture the simulation log line.
    pub capture_logs: bool,
    /// If set, only outputs containing at least one of these substrings are
    /// captured.
    pub output_filter: Option<Vec<String>>,
}

// ──────────────────────────── TwinDataInterface ────────────────────────────

/// The full data interface for the twin — what to inject, how to simulate,
/// and what to capture.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TwinDataInterface {
    pub memory_interface: MemoryInterface,
    pub simulation_config: SimulationConfig,
    pub result_capture_config: ResultCaptureConfig,
}

// ──────────────────────────── SimulationRequest ────────────────────────────

/// A request to run a simulation against the injected memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationRequest {
    pub scenario_id: String,
    pub memory_refs: Vec<Uuid>,
    pub config: SimulationConfig,
}

// ──────────────────────────── SimulationMetrics ────────────────────────────

/// Metrics computed from a single simulation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationMetrics {
    /// Weighted accuracy of the injected memories (confidence × authority).
    pub accuracy: f64,
    /// How consistent this run is with historical runs (0–1, 1 = identical).
    pub consistency_with_history: f64,
    /// How far this run deviates from the historical norm (1 − consistency).
    pub deviation_score: f64,
    /// Fraction of requested memory refs that were resolved (0–1).
    pub coverage: f64,
}

// ──────────────────────────── SimulationResult ────────────────────────────

/// The outcome of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub scenario_id: String,
    pub success: bool,
    pub metrics: SimulationMetrics,
    pub outputs: Vec<String>,
    pub duration_ms: u64,
}

// ──────────────────────────── HistoricalComparison ────────────────────────────

/// Side-by-side comparison of a result against the historical baseline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HistoricalComparison {
    pub historical_sample_count: usize,
    pub avg_historical_accuracy: f64,
    pub avg_historical_coverage: f64,
    pub current_accuracy: f64,
    pub current_coverage: f64,
    pub delta_accuracy: f64,
    pub delta_coverage: f64,
}

// ──────────────────────────── ValidationResult ────────────────────────────

/// The result of validating a simulation against historical data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub simulation_result: SimulationResult,
    pub historical_comparison: HistoricalComparison,
    pub verified: bool,
    pub confidence: f64,
}

// ──────────────────────────── CapturedMemory ────────────────────────────

/// A memory derived from a simulation result, ready to be written back to
/// the Memory Service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedMemory {
    pub source_scenario_id: String,
    pub content: String,
    pub scope: Scope,
    pub authority_level: AuthorityLevel,
    pub confidence: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// ──────────────────────────── TrendDirection ────────────────────────────

/// Direction of a cost or metric trend over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
}

// ──────────────────────────── CostSummary ────────────────────────────

/// Cost summary for the operations dashboard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostSummary {
    pub daily_cost: f64,
    pub monthly_projection: f64,
    pub trend: TrendDirection,
}

impl Default for TrendDirection {
    fn default() -> Self {
        Self::Stable
    }
}

// ──────────────────────────── DashboardData ────────────────────────────

/// Aggregated snapshot for the unified operations dashboard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DashboardData {
    pub quality_score: f64,
    pub value_score: f64,
    pub cost: CostSummary,
    pub active_memories: u64,
    pub conflict_rate: f64,
    pub stale_rate: f64,
    pub simulation_count: u64,
    pub verification_pass_rate: f64,
}

// ──────────────────────────── DashboardInput ────────────────────────────

/// Raw metrics fed into [`OperationsDashboard`] to compute scores.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DashboardInput {
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub groundedness: f64,
    pub conflict_rate: f64,
    pub stale_rate: f64,
    pub active_memories: u64,
    pub simulation_count: u64,
    pub verification_passed: u64,
    pub verification_total: u64,
    pub daily_cost: f64,
    /// Chronological cost samples used to compute [`CostSummary::trend`].
    pub cost_history: Vec<f64>,
}

// ──────────────────────────── TwinError ────────────────────────────

/// Errors produced by the digital twin verification loop.
#[derive(Debug, Error)]
pub enum TwinError {
    #[error("memory {0} not found in injected context")]
    MemoryNotFound(Uuid),

    #[error("no memories injected for scenario {0}")]
    NoMemoriesInjected(String),

    #[error("simulation config invalid: {0}")]
    InvalidConfig(String),

    #[error("historical data unavailable for scenario {0}")]
    NoHistoricalData(String),

    #[error("capture failed: {0}")]
    CaptureFailed(String),
}

// ──────────────────────────── DigitalTwinBridge ────────────────────────────

/// Bridges the Memory Service with a Digital Twin simulation.
///
/// Holds the set of injected experience memories and the accumulated history
/// of past simulation results.  The typical feedback loop is:
///
/// 1. [`inject_experience_to_simulation`](Self::inject_experience_to_simulation)
///    — feed experience memories into the twin.
/// 2. [`run_simulation`](Self::run_simulation) — execute the simulation.
/// 3. [`validate_against_historical`](Self::validate_against_historical) —
///    compare the result to past runs.
/// 4. [`record_simulation_result`](Self::record_simulation_result) — persist
///    the result into history for future validations.
/// 5. [`capture_simulation_result`](Self::capture_simulation_result) — convert
///    the result back into memories for the Memory Service.
pub struct DigitalTwinBridge {
    /// Memories available for injection, keyed by ID.
    injected: HashMap<Uuid, InjectedMemory>,
    /// Chronological history of completed simulation results.
    history: Vec<SimulationResult>,
}

impl Default for DigitalTwinBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl DigitalTwinBridge {
    /// Create an empty bridge with no injected memories and no history.
    pub fn new() -> Self {
        Self {
            injected: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Number of memories currently injected into the twin.
    pub fn injected_count(&self) -> usize {
        self.injected.len()
    }

    /// Number of simulation results recorded in history.
    pub fn history_count(&self) -> usize {
        self.history.len()
    }

    // ── Memory → Simulation ──

    /// Feed experience memories into the twin for later simulation.
    ///
    /// Existing memories with the same ID are overwritten.  Returns the
    /// number of memories stored.
    pub fn inject_experience_to_simulation(
        &mut self,
        memories: Vec<InjectedMemory>,
    ) -> Result<usize, TwinError> {
        let count = memories.len();
        for m in memories {
            self.injected.insert(m.id, m);
        }
        Ok(count)
    }

    // ── Simulation execution ──

    /// Execute a simulation using the injected memories referenced by
    /// `request.memory_refs`.
    ///
    /// Metrics are derived deterministically from the injected data:
    ///
    /// - **coverage** — fraction of requested refs that were resolved.
    /// - **accuracy** — confidence weighted by authority rank.
    /// - **consistency_with_history** — similarity to past runs (1.0 if no
    ///   history).
    /// - **deviation_score** — `1 − consistency_with_history`.
    pub fn run_simulation(
        &self,
        request: &SimulationRequest,
    ) -> Result<SimulationResult, TwinError> {
        let start = Instant::now();

        if request.memory_refs.is_empty() {
            return Err(TwinError::InvalidConfig("memory_refs is empty".into()));
        }

        // Resolve the requested memories from the injected set.
        let mut found: Vec<&InjectedMemory> = Vec::new();
        let mut missing: Vec<Uuid> = Vec::new();
        for id in &request.memory_refs {
            match self.injected.get(id) {
                Some(m) => found.push(m),
                None => missing.push(*id),
            }
        }

        if found.is_empty() {
            return Err(TwinError::NoMemoriesInjected(request.scenario_id.clone()));
        }

        let coverage = found.len() as f64 / request.memory_refs.len() as f64;

        // Accuracy: confidence weighted by authority rank (higher = more trusted).
        let total_weight: f64 = found
            .iter()
            .map(|m| (m.authority_level.rank() + 1) as f64)
            .sum();
        let weighted_conf: f64 = found
            .iter()
            .map(|m| m.confidence * (m.authority_level.rank() + 1) as f64)
            .sum();
        let accuracy = if total_weight > 0.0 {
            weighted_conf / total_weight
        } else {
            0.0
        };

        // Consistency with historical runs.
        let consistency = self.compute_consistency(accuracy, coverage);
        let deviation_score = 1.0 - consistency;

        let metrics = SimulationMetrics {
            accuracy,
            consistency_with_history: consistency,
            deviation_score,
            coverage,
        };

        let success = coverage >= 1.0 && accuracy > 0.0;

        // Generate textual outputs from the injected memories.
        let outputs: Vec<String> = found
            .iter()
            .map(|m| format!("[{}] {}", m.scope.as_str(), m.content))
            .collect();

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(SimulationResult {
            scenario_id: request.scenario_id.clone(),
            success,
            metrics,
            outputs,
            duration_ms,
        })
    }

    // ── Result → Memory ──

    /// Capture a simulation result back into memory-ready entries.
    ///
    /// Only captures from a *successful* simulation.  Respects the flags and
    /// filter in `config` to determine which outputs, metrics, and logs are
    /// materialised as [`CapturedMemory`] items.
    pub fn capture_simulation_result(
        &self,
        result: &SimulationResult,
        config: &ResultCaptureConfig,
    ) -> Result<Vec<CapturedMemory>, TwinError> {
        if !result.success {
            return Err(TwinError::CaptureFailed(format!(
                "scenario {} did not succeed",
                result.scenario_id
            )));
        }

        let mut captured = Vec::new();

        if config.capture_outputs {
            for output in &result.outputs {
                let passes_filter = match &config.output_filter {
                    Some(filters) => filters.iter().any(|f| output.contains(f.as_str())),
                    None => true,
                };
                if passes_filter {
                    captured.push(CapturedMemory {
                        source_scenario_id: result.scenario_id.clone(),
                        content: output.clone(),
                        scope: Scope::Process,
                        authority_level: AuthorityLevel::L2Verified,
                        confidence: result.metrics.accuracy,
                        timestamp: Utc::now(),
                    });
                }
            }
        }

        if config.capture_metrics {
            let summary = format!(
                "metrics: accuracy={:.3}, consistency={:.3}, deviation={:.3}, coverage={:.3}",
                result.metrics.accuracy,
                result.metrics.consistency_with_history,
                result.metrics.deviation_score,
                result.metrics.coverage
            );
            captured.push(CapturedMemory {
                source_scenario_id: result.scenario_id.clone(),
                content: summary,
                scope: Scope::Process,
                authority_level: AuthorityLevel::L3Inferred,
                confidence: result.metrics.accuracy,
                timestamp: Utc::now(),
            });
        }

        if config.capture_logs {
            let log_entry = format!(
                "simulation log: scenario={}, success={}, duration_ms={}",
                result.scenario_id, result.success, result.duration_ms
            );
            captured.push(CapturedMemory {
                source_scenario_id: result.scenario_id.clone(),
                content: log_entry,
                scope: Scope::Agent,
                authority_level: AuthorityLevel::L3Inferred,
                confidence: 1.0,
                timestamp: Utc::now(),
            });
        }

        Ok(captured)
    }

    // ── Validation ──

    /// Validate a simulation result against the accumulated history.
    ///
    /// `confidence` blends consistency-with-history (60 %) and accuracy
    /// (40 %).  A result is `verified` when confidence ≥
    /// [`VERIFICATION_THRESHOLD`].
    pub fn validate_against_historical(
        &self,
        result: &SimulationResult,
    ) -> Result<ValidationResult, TwinError> {
        let comparison = self.build_historical_comparison(result);
        let confidence =
            result.metrics.consistency_with_history * 0.6 + result.metrics.accuracy * 0.4;
        let verified = confidence >= VERIFICATION_THRESHOLD;
        Ok(ValidationResult {
            simulation_result: result.clone(),
            historical_comparison: comparison,
            verified,
            confidence,
        })
    }

    /// Record a simulation result into history for future validations.
    pub fn record_simulation_result(&mut self, result: SimulationResult) {
        self.history.push(result);
    }

    // ── Private helpers ──

    /// Compute how consistent `(accuracy, coverage)` is with the historical
    /// average.  Returns 1.0 when there is no history.
    fn compute_consistency(&self, current_accuracy: f64, current_coverage: f64) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        let n = self.history.len() as f64;
        let avg_acc: f64 = self.history.iter().map(|r| r.metrics.accuracy).sum::<f64>() / n;
        let avg_cov: f64 = self.history.iter().map(|r| r.metrics.coverage).sum::<f64>() / n;
        let acc_diff = (current_accuracy - avg_acc).abs();
        let cov_diff = (current_coverage - avg_cov).abs();
        let consistency = 1.0 - (acc_diff + cov_diff) / 2.0;
        consistency.clamp(0.0, 1.0)
    }

    /// Build a side-by-side comparison of the current result vs. history.
    fn build_historical_comparison(&self, result: &SimulationResult) -> HistoricalComparison {
        if self.history.is_empty() {
            return HistoricalComparison {
                historical_sample_count: 0,
                avg_historical_accuracy: 0.0,
                avg_historical_coverage: 0.0,
                current_accuracy: result.metrics.accuracy,
                current_coverage: result.metrics.coverage,
                delta_accuracy: 0.0,
                delta_coverage: 0.0,
            };
        }
        let n = self.history.len() as f64;
        let avg_acc: f64 = self.history.iter().map(|r| r.metrics.accuracy).sum::<f64>() / n;
        let avg_cov: f64 = self.history.iter().map(|r| r.metrics.coverage).sum::<f64>() / n;
        HistoricalComparison {
            historical_sample_count: self.history.len(),
            avg_historical_accuracy: avg_acc,
            avg_historical_coverage: avg_cov,
            current_accuracy: result.metrics.accuracy,
            current_coverage: result.metrics.coverage,
            delta_accuracy: result.metrics.accuracy - avg_acc,
            delta_coverage: result.metrics.coverage - avg_cov,
        }
    }
}

// ──────────────────────────── OperationsDashboard ────────────────────────────

/// Unified operations dashboard aggregating memory quality, business value,
/// and cost into a single snapshot.
pub struct OperationsDashboard {
    input: DashboardInput,
}

impl OperationsDashboard {
    /// Create a dashboard from the given raw metric input.
    pub fn new(input: DashboardInput) -> Self {
        Self { input }
    }

    /// Overall memory quality score in [0, 1].
    ///
    /// Weighted blend: recall (25 %), precision (25 %), groundedness (20 %),
    /// (1 − conflict_rate) (15 %), (1 − stale_rate) (15 %).
    pub fn memory_quality_score(&self) -> f64 {
        let i = &self.input;
        let score = i.recall_at_k * 0.25
            + i.precision_at_k * 0.25
            + i.groundedness * 0.20
            + (1.0 - i.conflict_rate) * 0.15
            + (1.0 - i.stale_rate) * 0.15;
        score.clamp(0.0, 1.0)
    }

    /// Business value score in [0, 1].
    ///
    /// Weighted blend: verification pass rate (50 %), memory quality (30 %),
    /// simulation adoption (20 %, saturated at 100 simulations).
    pub fn business_value_score(&self) -> f64 {
        let i = &self.input;
        let pass_rate = self.verification_pass_rate();
        let quality = self.memory_quality_score();
        let sim_factor = (i.simulation_count as f64 / 100.0).min(1.0);
        let score = pass_rate * 0.5 + quality * 0.3 + sim_factor * 0.2;
        score.clamp(0.0, 1.0)
    }

    /// Cost summary with daily cost, 30-day projection, and trend.
    pub fn cost_summary(&self) -> CostSummary {
        let daily = self.input.daily_cost;
        let monthly = daily * 30.0;
        let trend = Self::compute_trend(&self.input.cost_history);
        CostSummary {
            daily_cost: daily,
            monthly_projection: monthly,
            trend,
        }
    }

    /// Generate the full dashboard snapshot.
    pub fn generate_dashboard(&self) -> DashboardData {
        let i = &self.input;
        DashboardData {
            quality_score: self.memory_quality_score(),
            value_score: self.business_value_score(),
            cost: self.cost_summary(),
            active_memories: i.active_memories,
            conflict_rate: i.conflict_rate,
            stale_rate: i.stale_rate,
            simulation_count: i.simulation_count,
            verification_pass_rate: self.verification_pass_rate(),
        }
    }

    // ── Private helpers ──

    /// Verification pass rate; 0.0 when no verifications have been run.
    fn verification_pass_rate(&self) -> f64 {
        if self.input.verification_total == 0 {
            return 0.0;
        }
        self.input.verification_passed as f64 / self.input.verification_total as f64
    }

    /// Determine trend from a chronological cost series.
    ///
    /// Compares the average of the second half to the first half.  A change
    /// of more than ±5 % is classified as Increasing / Decreasing; otherwise
    /// Stable.  Fewer than two samples → Stable.
    fn compute_trend(history: &[f64]) -> TrendDirection {
        if history.len() < 2 {
            return TrendDirection::Stable;
        }
        let mid = history.len() / 2;
        let first_avg: f64 = history[..mid].iter().sum::<f64>() / mid as f64;
        let second_half_len = history.len() - mid;
        let second_avg: f64 = history[mid..].iter().sum::<f64>() / second_half_len as f64;
        let change_ratio = if first_avg > 0.0 {
            (second_avg - first_avg) / first_avg
        } else if second_avg > 0.0 {
            1.0 // went from zero to positive → increasing
        } else {
            0.0
        };
        if change_ratio > 0.05 {
            TrendDirection::Increasing
        } else if change_ratio < -0.05 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──

    fn make_memory(
        id: Uuid,
        content: &str,
        confidence: f64,
        authority: AuthorityLevel,
    ) -> InjectedMemory {
        InjectedMemory {
            id,
            content: content.to_string(),
            scope: Scope::Process,
            entity_refs: vec!["device-1".into()],
            authority_level: authority,
            confidence,
            timestamp: Utc::now(),
        }
    }

    fn make_request(scenario: &str, refs: &[Uuid]) -> SimulationRequest {
        SimulationRequest {
            scenario_id: scenario.to_string(),
            memory_refs: refs.to_vec(),
            config: SimulationConfig::default(),
        }
    }

    fn full_capture_config() -> ResultCaptureConfig {
        ResultCaptureConfig {
            capture_outputs: true,
            capture_metrics: true,
            capture_logs: true,
            output_filter: None,
        }
    }

    fn sample_dashboard_input() -> DashboardInput {
        DashboardInput {
            recall_at_k: 0.9,
            precision_at_k: 0.85,
            groundedness: 0.92,
            conflict_rate: 0.05,
            stale_rate: 0.03,
            active_memories: 5000,
            simulation_count: 50,
            verification_passed: 45,
            verification_total: 50,
            daily_cost: 12.5,
            cost_history: vec![10.0, 11.0, 12.0, 13.0],
        }
    }

    // ── 1. Inject stores memories and returns count ──

    #[test]
    fn inject_stores_memories_and_returns_count() {
        let mut bridge = DigitalTwinBridge::new();
        let m1 = make_memory(
            Uuid::new_v4(),
            "fault A",
            0.9,
            AuthorityLevel::L1Authoritative,
        );
        let m2 = make_memory(Uuid::new_v4(), "fault B", 0.8, AuthorityLevel::L2Verified);
        let count = bridge
            .inject_experience_to_simulation(vec![m1, m2])
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(bridge.injected_count(), 2);
    }

    // ── 2. Inject empty returns zero ──

    #[test]
    fn inject_empty_returns_zero() {
        let mut bridge = DigitalTwinBridge::new();
        let count = bridge.inject_experience_to_simulation(vec![]).unwrap();
        assert_eq!(count, 0);
        assert_eq!(bridge.injected_count(), 0);
    }

    // ── 3. Inject overwrites duplicate IDs ──

    #[test]
    fn inject_overwrites_duplicate_ids() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        let m1 = make_memory(id, "old", 0.5, AuthorityLevel::L3Inferred);
        let m2 = make_memory(id, "new", 0.9, AuthorityLevel::L1Authoritative);
        bridge.inject_experience_to_simulation(vec![m1]).unwrap();
        bridge.inject_experience_to_simulation(vec![m2]).unwrap();
        assert_eq!(bridge.injected_count(), 1);
    }

    // ── 4. Run simulation with empty refs → config error ──

    #[test]
    fn run_simulation_empty_refs_config_error() {
        let bridge = DigitalTwinBridge::new();
        let req = make_request("s1", &[]);
        let err = bridge.run_simulation(&req).unwrap_err();
        assert!(matches!(err, TwinError::InvalidConfig(_)));
    }

    // ── 5. Run simulation with no injected → error ──

    #[test]
    fn run_simulation_no_injected_error() {
        let bridge = DigitalTwinBridge::new();
        let req = make_request("s1", &[Uuid::new_v4()]);
        let err = bridge.run_simulation(&req).unwrap_err();
        assert!(matches!(err, TwinError::NoMemoriesInjected(_)));
    }

    // ── 6. Run simulation computes coverage ──

    #[test]
    fn run_simulation_computes_coverage() {
        let mut bridge = DigitalTwinBridge::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let missing = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![
                make_memory(id1, "m1", 0.9, AuthorityLevel::L1Authoritative),
                make_memory(id2, "m2", 0.8, AuthorityLevel::L2Verified),
            ])
            .unwrap();
        let req = make_request("s1", &[id1, id2, missing]);
        let result = bridge.run_simulation(&req).unwrap();
        // 2 of 3 refs found → coverage = 2/3
        assert!((result.metrics.coverage - (2.0 / 3.0)).abs() < 1e-9);
        // Partial coverage → not success
        assert!(!result.success);
    }

    // ── 7. Run simulation full coverage → success ──

    #[test]
    fn run_simulation_full_coverage_success() {
        let mut bridge = DigitalTwinBridge::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![
                make_memory(id1, "m1", 0.9, AuthorityLevel::L1Authoritative),
                make_memory(id2, "m2", 0.8, AuthorityLevel::L2Verified),
            ])
            .unwrap();
        let req = make_request("s1", &[id1, id2]);
        let result = bridge.run_simulation(&req).unwrap();
        assert!((result.metrics.coverage - 1.0).abs() < 1e-9);
        assert!(result.success);
    }

    // ── 8. Accuracy weighted by authority rank ──

    #[test]
    fn accuracy_weighted_by_authority() {
        let mut bridge = DigitalTwinBridge::new();
        let id_lo = Uuid::new_v4(); // L3 rank 0, weight 1, conf 1.0
        let id_hi = Uuid::new_v4(); // L0 rank 3, weight 4, conf 0.5
        bridge
            .inject_experience_to_simulation(vec![
                make_memory(id_lo, "lo", 1.0, AuthorityLevel::L3Inferred),
                make_memory(id_hi, "hi", 0.5, AuthorityLevel::L0SourceOfTruth),
            ])
            .unwrap();
        let req = make_request("s1", &[id_lo, id_hi]);
        let result = bridge.run_simulation(&req).unwrap();
        // weighted = (1.0*1 + 0.5*4) / (1+4) = 3.0/5 = 0.6
        assert!((result.metrics.accuracy - 0.6).abs() < 1e-9);
    }

    // ── 9. First run consistency is 1.0 ──

    #[test]
    fn first_run_consistency_is_one() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id]);
        let result = bridge.run_simulation(&req).unwrap();
        assert!((result.metrics.consistency_with_history - 1.0).abs() < 1e-9);
        assert!((result.metrics.deviation_score - 0.0).abs() < 1e-9);
    }

    // ── 10. Run simulation generates outputs ──

    #[test]
    fn run_simulation_generates_outputs() {
        let mut bridge = DigitalTwinBridge::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![
                make_memory(id1, "pump fault", 0.9, AuthorityLevel::L1Authoritative),
                make_memory(id2, "valve leak", 0.8, AuthorityLevel::L2Verified),
            ])
            .unwrap();
        let req = make_request("s1", &[id1, id2]);
        let result = bridge.run_simulation(&req).unwrap();
        assert_eq!(result.outputs.len(), 2);
        assert!(result.outputs[0].contains("pump fault"));
        assert!(result.outputs[1].contains("valve leak"));
    }

    // ── 11. Duration is recorded ──

    #[test]
    fn duration_ms_recorded() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id]);
        let result = bridge.run_simulation(&req).unwrap();
        // Trivial computation; duration is a valid u64 (likely 0).
        let _ = result.duration_ms;
    }

    // ── 12. Validate with no history → verified ──

    #[test]
    fn validate_no_history_verified() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.95,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id]);
        let result = bridge.run_simulation(&req).unwrap();
        let validation = bridge.validate_against_historical(&result).unwrap();
        // confidence = 1.0*0.6 + 0.95*0.4 = 0.98 ≥ 0.70
        assert!((validation.confidence - 0.98).abs() < 1e-9);
        assert!(validation.verified);
        assert_eq!(validation.historical_comparison.historical_sample_count, 0);
    }

    // ── 13. Validate with history computes confidence ──

    #[test]
    fn validate_with_history_computes_confidence() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();

        // First run → record.
        let req = make_request("s1", &[id]);
        let r1 = bridge.run_simulation(&req).unwrap();
        bridge.record_simulation_result(r1.clone());

        // Second run with same data → high consistency.
        let r2 = bridge.run_simulation(&req).unwrap();
        let validation = bridge.validate_against_historical(&r2).unwrap();
        assert_eq!(validation.historical_comparison.historical_sample_count, 1);
        // Same accuracy & coverage → consistency 1.0
        assert!((validation.historical_comparison.delta_accuracy).abs() < 1e-9);
        assert!(validation.verified);
    }

    // ── 14. Validate detects deviation from history ──

    #[test]
    fn validate_detects_deviation() {
        let mut bridge = DigitalTwinBridge::new();
        let id_hi = Uuid::new_v4();
        let id_lo = Uuid::new_v4();

        // Seed history with a high-accuracy run.
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id_hi,
                "hi",
                0.95,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req_hi = make_request("baseline", &[id_hi]);
        let r_hi = bridge.run_simulation(&req_hi).unwrap();
        bridge.record_simulation_result(r_hi);

        // Now inject a lower-confidence memory and run.
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id_lo,
                "lo",
                0.3,
                AuthorityLevel::L3Inferred,
            )])
            .unwrap();
        let req_lo = make_request("degraded", &[id_lo]);
        let r_lo = bridge.run_simulation(&req_lo).unwrap();
        let validation = bridge.validate_against_historical(&r_lo).unwrap();
        // Accuracy dropped → consistency < 1.0 → deviation > 0
        assert!(r_lo.metrics.deviation_score > 0.0);
        assert!(validation.historical_comparison.delta_accuracy < 0.0);
    }

    // ── 15. Capture outputs ──

    #[test]
    fn capture_outputs() {
        let mut bridge = DigitalTwinBridge::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![
                make_memory(id1, "pump fault", 0.9, AuthorityLevel::L1Authoritative),
                make_memory(id2, "valve leak", 0.8, AuthorityLevel::L2Verified),
            ])
            .unwrap();
        let req = make_request("s1", &[id1, id2]);
        let result = bridge.run_simulation(&req).unwrap();
        let config = ResultCaptureConfig {
            capture_outputs: true,
            capture_metrics: false,
            capture_logs: false,
            output_filter: None,
        };
        let captured = bridge.capture_simulation_result(&result, &config).unwrap();
        assert_eq!(captured.len(), 2);
        assert!(captured.iter().all(|c| c.scope == Scope::Process));
        assert!(captured
            .iter()
            .all(|c| c.authority_level == AuthorityLevel::L2Verified));
    }

    // ── 16. Capture respects capture_outputs = false ──

    #[test]
    fn capture_outputs_false_yields_none() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id]);
        let result = bridge.run_simulation(&req).unwrap();
        let config = ResultCaptureConfig {
            capture_outputs: false,
            capture_metrics: false,
            capture_logs: false,
            output_filter: None,
        };
        let captured = bridge.capture_simulation_result(&result, &config).unwrap();
        assert!(captured.is_empty());
    }

    // ── 17. Capture applies output filter ──

    #[test]
    fn capture_applies_output_filter() {
        let mut bridge = DigitalTwinBridge::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![
                make_memory(
                    id1,
                    "pump fault detected",
                    0.9,
                    AuthorityLevel::L1Authoritative,
                ),
                make_memory(id2, "valve leak found", 0.8, AuthorityLevel::L2Verified),
            ])
            .unwrap();
        let req = make_request("s1", &[id1, id2]);
        let result = bridge.run_simulation(&req).unwrap();
        let config = ResultCaptureConfig {
            capture_outputs: true,
            capture_metrics: false,
            capture_logs: false,
            output_filter: Some(vec!["pump".into()]),
        };
        let captured = bridge.capture_simulation_result(&result, &config).unwrap();
        assert_eq!(captured.len(), 1);
        assert!(captured[0].content.contains("pump"));
    }

    // ── 18. Capture metrics summary ──

    #[test]
    fn capture_metrics_summary() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id]);
        let result = bridge.run_simulation(&req).unwrap();
        let config = ResultCaptureConfig {
            capture_outputs: false,
            capture_metrics: true,
            capture_logs: false,
            output_filter: None,
        };
        let captured = bridge.capture_simulation_result(&result, &config).unwrap();
        assert_eq!(captured.len(), 1);
        assert!(captured[0].content.contains("accuracy="));
        assert!(captured[0].content.contains("coverage="));
        assert_eq!(captured[0].authority_level, AuthorityLevel::L3Inferred);
    }

    // ── 19. Capture logs ──

    #[test]
    fn capture_logs() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id]);
        let result = bridge.run_simulation(&req).unwrap();
        let config = ResultCaptureConfig {
            capture_outputs: false,
            capture_metrics: false,
            capture_logs: true,
            output_filter: None,
        };
        let captured = bridge.capture_simulation_result(&result, &config).unwrap();
        assert_eq!(captured.len(), 1);
        assert!(captured[0].content.contains("simulation log"));
        assert_eq!(captured[0].scope, Scope::Agent);
    }

    // ── 20. Capture failed simulation → error ──

    #[test]
    fn capture_failed_simulation_error() {
        let mut bridge = DigitalTwinBridge::new();
        let id1 = Uuid::new_v4();
        let missing = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id1,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id1, missing]);
        let result = bridge.run_simulation(&req).unwrap();
        // Partial coverage → not success
        assert!(!result.success);
        let err = bridge
            .capture_simulation_result(&result, &full_capture_config())
            .unwrap_err();
        assert!(matches!(err, TwinError::CaptureFailed(_)));
    }

    // ── 21. Record updates history count ──

    #[test]
    fn record_updates_history() {
        let mut bridge = DigitalTwinBridge::new();
        let id = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![make_memory(
                id,
                "m1",
                0.9,
                AuthorityLevel::L1Authoritative,
            )])
            .unwrap();
        let req = make_request("s1", &[id]);
        let result = bridge.run_simulation(&req).unwrap();
        assert_eq!(bridge.history_count(), 0);
        bridge.record_simulation_result(result);
        assert_eq!(bridge.history_count(), 1);
    }

    // ── 22. Full feedback loop end-to-end ──

    #[test]
    fn full_feedback_loop() {
        let mut bridge = DigitalTwinBridge::new();

        // 1. Inject
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        bridge
            .inject_experience_to_simulation(vec![
                make_memory(id1, "pump fault", 0.9, AuthorityLevel::L1Authoritative),
                make_memory(id2, "valve leak", 0.8, AuthorityLevel::L2Verified),
            ])
            .unwrap();

        // 2. Run
        let req = make_request("scenario-1", &[id1, id2]);
        let result = bridge.run_simulation(&req).unwrap();
        assert!(result.success);

        // 3. Validate (no history yet)
        let validation = bridge.validate_against_historical(&result).unwrap();
        assert!(validation.verified);

        // 4. Record into history
        bridge.record_simulation_result(result.clone());

        // 5. Capture back to memories
        let captured = bridge
            .capture_simulation_result(&result, &full_capture_config())
            .unwrap();
        // 2 outputs + 1 metrics + 1 log = 4
        assert_eq!(captured.len(), 4);
        assert!(captured
            .iter()
            .all(|c| c.source_scenario_id == "scenario-1"));

        // 6. Second iteration — now has history
        let req2 = make_request("scenario-2", &[id1, id2]);
        let result2 = bridge.run_simulation(&req2).unwrap();
        let validation2 = bridge.validate_against_historical(&result2).unwrap();
        assert_eq!(validation2.historical_comparison.historical_sample_count, 1);
        // Same data → high consistency
        assert!((result2.metrics.consistency_with_history - 1.0).abs() < 1e-9);
    }

    // ── 23. OperationsDashboard memory_quality_score ──

    #[test]
    fn dashboard_memory_quality_score() {
        let dash = OperationsDashboard::new(sample_dashboard_input());
        let score = dash.memory_quality_score();
        // 0.9*0.25 + 0.85*0.25 + 0.92*0.20 + 0.95*0.15 + 0.97*0.15
        // = 0.225 + 0.2125 + 0.184 + 0.1425 + 0.1455 = 0.9095
        assert!((score - 0.9095).abs() < 1e-4);
        assert!(score <= 1.0 && score >= 0.0);
    }

    // ── 24. OperationsDashboard business_value_score ──

    #[test]
    fn dashboard_business_value_score() {
        let dash = OperationsDashboard::new(sample_dashboard_input());
        let value = dash.business_value_score();
        // pass_rate = 45/50 = 0.9; quality = 0.9095; sim_factor = 50/100 = 0.5
        // 0.9*0.5 + 0.9095*0.3 + 0.5*0.2 = 0.45 + 0.27285 + 0.1 = 0.82285
        assert!((value - 0.82285).abs() < 1e-4);
        assert!(value <= 1.0 && value >= 0.0);
    }

    // ── 25. OperationsDashboard cost_summary trend increasing ──

    #[test]
    fn dashboard_cost_trend_increasing() {
        let mut input = sample_dashboard_input();
        input.cost_history = vec![10.0, 11.0, 12.0, 13.0];
        let dash = OperationsDashboard::new(input);
        let cost = dash.cost_summary();
        assert_eq!(cost.trend, TrendDirection::Increasing);
        assert!((cost.monthly_projection - 12.5 * 30.0).abs() < 1e-9);
    }

    // ── 26. OperationsDashboard cost_summary trend decreasing ──

    #[test]
    fn dashboard_cost_trend_decreasing() {
        let mut input = sample_dashboard_input();
        input.cost_history = vec![13.0, 12.0, 11.0, 10.0];
        let dash = OperationsDashboard::new(input);
        let cost = dash.cost_summary();
        assert_eq!(cost.trend, TrendDirection::Decreasing);
    }

    // ── 27. OperationsDashboard cost_summary trend stable ──

    #[test]
    fn dashboard_cost_trend_stable() {
        let mut input = sample_dashboard_input();
        input.cost_history = vec![12.0, 12.0, 12.0, 12.0];
        let dash = OperationsDashboard::new(input);
        let cost = dash.cost_summary();
        assert_eq!(cost.trend, TrendDirection::Stable);
    }

    // ── 28. Trend single element → Stable ──

    #[test]
    fn trend_single_element_stable() {
        let input = DashboardInput {
            cost_history: vec![5.0],
            ..Default::default()
        };
        let dash = OperationsDashboard::new(input);
        assert_eq!(dash.cost_summary().trend, TrendDirection::Stable);
    }

    // ── 29. Trend from zero to positive → Increasing ──

    #[test]
    fn trend_zero_to_positive_increasing() {
        let input = DashboardInput {
            cost_history: vec![0.0, 5.0],
            ..Default::default()
        };
        let dash = OperationsDashboard::new(input);
        assert_eq!(dash.cost_summary().trend, TrendDirection::Increasing);
    }

    // ── 30. OperationsDashboard generate_dashboard ──

    #[test]
    fn dashboard_generate_aggregates() {
        let input = sample_dashboard_input();
        let dash = OperationsDashboard::new(input.clone());
        let data = dash.generate_dashboard();
        assert!((data.quality_score - dash.memory_quality_score()).abs() < 1e-9);
        assert!((data.value_score - dash.business_value_score()).abs() < 1e-9);
        assert_eq!(data.active_memories, input.active_memories);
        assert_eq!(data.conflict_rate, input.conflict_rate);
        assert_eq!(data.stale_rate, input.stale_rate);
        assert_eq!(data.simulation_count, input.simulation_count);
        assert!((data.verification_pass_rate - 0.9).abs() < 1e-9);
    }

    // ── 31. Dashboard with zero verifications → pass rate 0 ──

    #[test]
    fn dashboard_zero_verifications_pass_rate_zero() {
        let input = DashboardInput {
            verification_passed: 0,
            verification_total: 0,
            ..Default::default()
        };
        let dash = OperationsDashboard::new(input);
        let data = dash.generate_dashboard();
        assert!((data.verification_pass_rate - 0.0).abs() < 1e-9);
    }

    // ── 32. SimulationMetrics serde round-trip ──

    #[test]
    fn simulation_metrics_serde_roundtrip() {
        let metrics = SimulationMetrics {
            accuracy: 0.85,
            consistency_with_history: 0.92,
            deviation_score: 0.08,
            coverage: 1.0,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: SimulationMetrics = serde_json::from_str(&json).unwrap();
        assert!((restored.accuracy - metrics.accuracy).abs() < 1e-9);
        assert!((restored.coverage - metrics.coverage).abs() < 1e-9);
    }

    // ── 33. SimulationResult serde round-trip ──

    #[test]
    fn simulation_result_serde_roundtrip() {
        let result = SimulationResult {
            scenario_id: "s-99".into(),
            success: true,
            metrics: SimulationMetrics {
                accuracy: 0.7,
                consistency_with_history: 0.8,
                deviation_score: 0.2,
                coverage: 0.9,
            },
            outputs: vec!["out1".into()],
            duration_ms: 42,
        };
        let json = serde_json::to_string(&result).unwrap();
        let restored: SimulationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.scenario_id, "s-99");
        assert!(restored.success);
        assert_eq!(restored.duration_ms, 42);
        assert_eq!(restored.outputs, vec!["out1"]);
    }

    // ── 34. TrendDirection serde round-trip ──

    #[test]
    fn trend_direction_serde_roundtrip() {
        let json = serde_json::to_string(&TrendDirection::Increasing).unwrap();
        assert_eq!(json, "\"increasing\"");
        let restored: TrendDirection = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, TrendDirection::Increasing);
    }

    // ── 35. TwinDataInterface default ──

    #[test]
    fn twin_data_interface_default() {
        let tdi = TwinDataInterface::default();
        assert!(tdi.memory_interface.scopes.is_empty());
        assert!(tdi.memory_interface.entity_refs.is_empty());
        assert_eq!(tdi.simulation_config.iterations, 1);
        assert_eq!(tdi.simulation_config.fidelity, SimulationFidelity::Medium);
        assert!(!tdi.result_capture_config.capture_outputs);
    }

    // ── 36. MemoryNotFound error variant display ──

    #[test]
    fn twin_error_display() {
        let id = Uuid::nil();
        let err = TwinError::MemoryNotFound(id);
        let msg = format!("{}", err);
        assert!(msg.contains("not found"));
    }
}
