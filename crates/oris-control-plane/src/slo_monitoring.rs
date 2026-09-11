//! SLO Monitoring & Memory Quality Metrics (§12 Observability).
//!
//! Collects and aggregates technical SLO metrics (latency, permission
//! fail-open, timeouts, circuit opens) alongside memory-quality metrics
//! (recall@k, precision@k, groundedness, etc.) so operators can detect
//! SLO violations and quality regressions in real time.
//!
//! ## Architecture references
//!
//! - §12 — Observability: technical SLOs and memory quality metrics.
//! - [`crate::resilience`] — `SloEvent`/`SloEventType` feed into this
//!   collector at the monitoring boundary.
//! - [`crate::red_team_tests::PerfBenchmark`] — existing SLO constants
//!   (hot context 60ms, standard recall 200ms).

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// ──────────────────────────── SloMetrics ────────────────────────────

/// Aggregated SLO and memory-quality metrics snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SloMetrics {
    // ── Technical SLOs ──
    pub hot_context_p95_ms: Option<u64>,
    pub standard_recall_p95_ms: Option<u64>,
    pub permission_fail_open_count: u64,

    // ── Memory quality metrics ──
    pub recall_at_k: Option<f64>,
    pub precision_at_k: Option<f64>,
    pub context_relevance_rate: Option<f64>,
    pub token_utilization_rate: Option<f64>,
    pub conflict_rate: Option<f64>,
    pub stale_memory_rate: Option<f64>,
    pub no_source_memory_rate: Option<f64>,
    pub groundedness_score: Option<f64>,
    pub evidence_coverage_rate: Option<f64>,
    pub user_correction_rate: Option<f64>,
    pub forget_request_avg_ms: Option<u64>,
    pub low_confidence_mispromotion_rate: Option<f64>,
    pub cross_agent_handover_success_rate: Option<f64>,

    // ── Counters ──
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_timeouts: u64,
    pub total_circuit_opens: u64,
}

// ──────────────────────────── LatencyMeasurement ────────────────────────────

/// A single latency observation tagged with operation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMeasurement {
    pub operation: String,
    pub latency_ms: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tenant_id: Option<String>,
}

// ──────────────────────────── QualityMetricsInput ────────────────────────────

/// Memory-quality metrics submitted after a search/assembly operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetricsInput {
    pub recall_at_k: Option<f64>,
    pub precision_at_k: Option<f64>,
    pub context_relevance_rate: Option<f64>,
    pub token_utilization_rate: Option<f64>,
    pub conflict_rate: Option<f64>,
    pub stale_memory_rate: Option<f64>,
    pub groundedness_score: Option<f64>,
    pub evidence_coverage_rate: Option<f64>,
}

// ──────────────────────────── SloTargets ────────────────────────────

/// §12 SLO targets and evaluation predicates.
pub struct SloTargets;

impl SloTargets {
    /// Hot-context P95 latency must be ≤ 60ms.
    pub const HOT_CONTEXT_P95_MS: u64 = 60;
    /// Standard recall P95 latency must be ≤ 200ms.
    pub const STANDARD_RECALL_P95_MS: u64 = 200;

    /// True when the hot-context P95 meets the SLO.
    pub fn hot_context_slo_met(p95_ms: u64) -> bool {
        p95_ms <= Self::HOT_CONTEXT_P95_MS
    }

    /// True when the standard-recall P95 meets the SLO.
    pub fn standard_recall_slo_met(p95_ms: u64) -> bool {
        p95_ms <= Self::STANDARD_RECALL_P95_MS
    }

    /// Permission subsystem must **never** fail open.
    pub fn permission_never_fail_open(count: u64) -> bool {
        count == 0
    }
}

// ──────────────────────────── ViolationSeverity ────────────────────────────

/// Severity of an SLO violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Approaching the SLO limit (≥ 80% of budget).
    Warning,
    /// SLO violated.
    Critical,
}

// ──────────────────────────── SloViolation ────────────────────────────

/// A detected SLO violation with context for alerting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloViolation {
    pub metric: String,
    pub target: String,
    pub actual: String,
    pub severity: ViolationSeverity,
}

// ──────────────────────────── Internal state ────────────────────────────

/// Internal mutable state kept behind the collector's `RwLock`.
#[derive(Debug, Clone, Default)]
struct CollectorState {
    metrics: SloMetrics,
    hot_context_latencies: Vec<u64>,
    standard_recall_latencies: Vec<u64>,
    /// Latencies for forget operations (for `forget_request_avg_ms`).
    forget_latencies: Vec<u64>,
}

// ──────────────────────────── SloCollector ────────────────────────────

/// Collects and aggregates SLO metrics from system events.
pub struct SloCollector {
    state: RwLock<CollectorState>,
}

impl SloCollector {
    /// Create a new empty collector.
    pub fn new() -> Self {
        Self {
            state: RwLock::new(CollectorState::default()),
        }
    }

    /// Record a latency measurement for an operation.
    ///
    /// The operation name is matched case-insensitively against known
    /// SLO-tagged operations (`hot_context`, `standard_recall`,
    /// `forget`).  Each recorded latency also bumps `total_requests`.
    pub async fn record_latency(&self, measurement: LatencyMeasurement) {
        let mut state = self.state.write().await;
        let op = measurement.operation.to_ascii_lowercase();

        match op.as_str() {
            "hot_context" | "hot-context" | "hotcontext" => {
                state.hot_context_latencies.push(measurement.latency_ms);
            }
            "standard_recall" | "standard-recall" | "standardrecall" | "recall" => {
                state.standard_recall_latencies.push(measurement.latency_ms);
            }
            "forget" | "forget_request" | "forget-request" => {
                state.forget_latencies.push(measurement.latency_ms);
            }
            _ => {}
        }

        state.metrics.total_requests += 1;
        self.recompute_p95s(&mut state);
    }

    /// Record a permission fail-open event (should never happen).
    pub async fn record_permission_fail_open(&self) {
        let mut state = self.state.write().await;
        state.metrics.permission_fail_open_count += 1;
        state.metrics.total_errors += 1;
    }

    /// Record a timeout event.
    pub async fn record_timeout(&self) {
        let mut state = self.state.write().await;
        state.metrics.total_timeouts += 1;
        state.metrics.total_errors += 1;
    }

    /// Record a circuit breaker open event.
    pub async fn record_circuit_open(&self) {
        let mut state = self.state.write().await;
        state.metrics.total_circuit_opens += 1;
    }

    /// Record memory-quality metrics from a search/assembly operation.
    ///
    /// Only fields set to `Some` in the input overwrite the stored value;
    /// `None` fields leave existing values untouched.
    pub async fn record_quality_metrics(&self, metrics: QualityMetricsInput) {
        let mut state = self.state.write().await;
        let m = &mut state.metrics;

        if let Some(v) = metrics.recall_at_k {
            m.recall_at_k = Some(v);
        }
        if let Some(v) = metrics.precision_at_k {
            m.precision_at_k = Some(v);
        }
        if let Some(v) = metrics.context_relevance_rate {
            m.context_relevance_rate = Some(v);
        }
        if let Some(v) = metrics.token_utilization_rate {
            m.token_utilization_rate = Some(v);
        }
        if let Some(v) = metrics.conflict_rate {
            m.conflict_rate = Some(v);
        }
        if let Some(v) = metrics.stale_memory_rate {
            m.stale_memory_rate = Some(v);
        }
        if let Some(v) = metrics.groundedness_score {
            m.groundedness_score = Some(v);
        }
        if let Some(v) = metrics.evidence_coverage_rate {
            m.evidence_coverage_rate = Some(v);
        }
    }

    /// Get a snapshot of current metrics.
    pub async fn snapshot(&self) -> SloMetrics {
        let state = self.state.read().await;
        state.metrics.clone()
    }

    /// Compute P95 latency from a list of measurements.
    ///
    /// Sorts the list, takes the value at the 95th percentile index
    /// (0-based).  Returns `None` for an empty slice.
    pub fn compute_p95(latencies: &[u64]) -> Option<u64> {
        if latencies.is_empty() {
            return None;
        }
        let mut sorted = latencies.to_vec();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
        let idx = idx.min(sorted.len()) - 1; // clamp + convert to 0-based
        Some(sorted[idx])
    }

    /// Check all SLO targets and return violations.
    ///
    /// A metric at or above 80% of its budget is flagged `Warning`;
    /// exceeding the budget is flagged `Critical`.
    pub async fn check_slo_violations(&self) -> Vec<SloViolation> {
        let state = self.state.read().await;
        let m = &state.metrics;
        let mut violations = Vec::new();

        // Hot-context P95.
        if let Some(p95) = m.hot_context_p95_ms {
            let target = SloTargets::HOT_CONTEXT_P95_MS;
            if p95 > target {
                violations.push(SloViolation {
                    metric: "hot_context_p95_ms".into(),
                    target: format!("≤{target}ms"),
                    actual: format!("{p95}ms"),
                    severity: ViolationSeverity::Critical,
                });
            } else if p95 >= (target as f64 * 0.8) as u64 {
                violations.push(SloViolation {
                    metric: "hot_context_p95_ms".into(),
                    target: format!("≤{target}ms"),
                    actual: format!("{p95}ms"),
                    severity: ViolationSeverity::Warning,
                });
            }
        }

        // Standard-recall P95.
        if let Some(p95) = m.standard_recall_p95_ms {
            let target = SloTargets::STANDARD_RECALL_P95_MS;
            if p95 > target {
                violations.push(SloViolation {
                    metric: "standard_recall_p95_ms".into(),
                    target: format!("≤{target}ms"),
                    actual: format!("{p95}ms"),
                    severity: ViolationSeverity::Critical,
                });
            } else if p95 >= (target as f64 * 0.8) as u64 {
                violations.push(SloViolation {
                    metric: "standard_recall_p95_ms".into(),
                    target: format!("≤{target}ms"),
                    actual: format!("{p95}ms"),
                    severity: ViolationSeverity::Warning,
                });
            }
        }

        // Permission fail-open — any count is Critical.
        if m.permission_fail_open_count > 0 {
            violations.push(SloViolation {
                metric: "permission_fail_open_count".into(),
                target: "0".into(),
                actual: m.permission_fail_open_count.to_string(),
                severity: ViolationSeverity::Critical,
            });
        }

        violations
    }

    // ── private helpers ──

    /// Recompute P95 values and forget average from stored latency vectors.
    fn recompute_p95s(&self, state: &mut CollectorState) {
        state.metrics.hot_context_p95_ms = Self::compute_p95(&state.hot_context_latencies);
        state.metrics.standard_recall_p95_ms = Self::compute_p95(&state.standard_recall_latencies);
        if state.forget_latencies.is_empty() {
            state.metrics.forget_request_avg_ms = None;
        } else {
            let sum: u64 = state.forget_latencies.iter().sum();
            let count = state.forget_latencies.len() as u64;
            state.metrics.forget_request_avg_ms = Some(sum / count);
        }
    }
}

impl Default for SloCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SloCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SloCollector").finish_non_exhaustive()
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_latency(op: &str, ms: u64) -> LatencyMeasurement {
        LatencyMeasurement {
            operation: op.to_string(),
            latency_ms: ms,
            timestamp: Utc::now(),
            tenant_id: None,
        }
    }

    // ── 1. Empty metrics → all None ──

    #[tokio::test]
    async fn empty_metrics_all_none() {
        let collector = SloCollector::new();
        let snap = collector.snapshot().await;
        assert!(snap.hot_context_p95_ms.is_none());
        assert!(snap.standard_recall_p95_ms.is_none());
        assert_eq!(snap.permission_fail_open_count, 0);
        assert!(snap.recall_at_k.is_none());
        assert!(snap.precision_at_k.is_none());
        assert_eq!(snap.total_requests, 0);
        assert_eq!(snap.total_errors, 0);
    }

    // ── 2. P95 computation with 100 values ──

    #[test]
    fn p95_with_100_values() {
        // Values 1..=100; P95 index = ceil(100*0.95)=95, 0-based → sorted[94]=95.
        let latencies: Vec<u64> = (1..=100u64).collect();
        let p95 = SloCollector::compute_p95(&latencies);
        assert_eq!(p95, Some(95));
    }

    // ── 3. P95 with single value ──

    #[test]
    fn p95_with_single_value() {
        let p95 = SloCollector::compute_p95(&[42]);
        assert_eq!(p95, Some(42));
    }

    // ── 4. P95 with empty list → None ──

    #[test]
    fn p95_with_empty_list() {
        assert_eq!(SloCollector::compute_p95(&[]), None);
    }

    // ── 5. Hot context SLO met (60ms) ──

    #[test]
    fn hot_context_slo_met() {
        assert!(SloTargets::hot_context_slo_met(60));
        assert!(SloTargets::hot_context_slo_met(30));
    }

    // ── 6. Hot context SLO violated (61ms) ──

    #[test]
    fn hot_context_slo_violated() {
        assert!(!SloTargets::hot_context_slo_met(61));
    }

    // ── 7. Standard recall SLO met (200ms) ──

    #[test]
    fn standard_recall_slo_met() {
        assert!(SloTargets::standard_recall_slo_met(200));
        assert!(SloTargets::standard_recall_slo_met(150));
    }

    // ── 8. Standard recall SLO violated (201ms) ──

    #[test]
    fn standard_recall_slo_violated() {
        assert!(!SloTargets::standard_recall_slo_met(201));
    }

    // ── 9. Permission never fail open ──

    #[test]
    fn permission_never_fail_open() {
        assert!(SloTargets::permission_never_fail_open(0));
        assert!(!SloTargets::permission_never_fail_open(1));
    }

    // ── 10. Record latency updates P95 ──

    #[tokio::test]
    async fn record_latency_updates_p95() {
        let collector = SloCollector::new();
        for ms in [30u64, 40, 50, 60, 70] {
            collector
                .record_latency(make_latency("hot_context", ms))
                .await;
        }
        // 5 values sorted: [30,40,50,60,70]; idx=ceil(5*0.95)=5→0-based 4 → 70.
        let snap = collector.snapshot().await;
        assert_eq!(snap.hot_context_p95_ms, Some(70));
        assert_eq!(snap.total_requests, 5);
    }

    // ── 11. Record quality metrics updates fields ──

    #[tokio::test]
    async fn record_quality_metrics_updates_fields() {
        let collector = SloCollector::new();
        collector
            .record_quality_metrics(QualityMetricsInput {
                recall_at_k: Some(0.92),
                precision_at_k: Some(0.88),
                context_relevance_rate: None,
                token_utilization_rate: Some(0.75),
                conflict_rate: Some(0.01),
                stale_memory_rate: None,
                groundedness_score: Some(0.95),
                evidence_coverage_rate: Some(0.90),
            })
            .await;
        let snap = collector.snapshot().await;
        assert_eq!(snap.recall_at_k, Some(0.92));
        assert_eq!(snap.precision_at_k, Some(0.88));
        assert_eq!(snap.token_utilization_rate, Some(0.75));
        assert_eq!(snap.conflict_rate, Some(0.01));
        assert_eq!(snap.groundedness_score, Some(0.95));
        assert_eq!(snap.evidence_coverage_rate, Some(0.90));
        // None fields stay None.
        assert!(snap.context_relevance_rate.is_none());
        assert!(snap.stale_memory_rate.is_none());
    }

    // ── 12. SLO violations returned for exceeded targets ──

    #[tokio::test]
    async fn slo_violations_for_exceeded_targets() {
        let collector = SloCollector::new();
        // Push hot-context latencies so P95 > 60ms.
        for ms in [50u64, 55, 61, 62, 63] {
            collector
                .record_latency(make_latency("hot_context", ms))
                .await;
        }
        // Push standard-recall latencies so P95 > 200ms.
        for ms in [150u64, 180, 201, 210, 220] {
            collector
                .record_latency(make_latency("standard_recall", ms))
                .await;
        }
        collector.record_permission_fail_open().await;

        let violations = collector.check_slo_violations().await;
        assert_eq!(violations.len(), 3);

        let metrics: Vec<&str> = violations.iter().map(|v| v.metric.as_str()).collect();
        assert!(metrics.contains(&"hot_context_p95_ms"));
        assert!(metrics.contains(&"standard_recall_p95_ms"));
        assert!(metrics.contains(&"permission_fail_open_count"));

        // All three are Critical.
        assert!(violations
            .iter()
            .all(|v| v.severity == ViolationSeverity::Critical));
    }

    // ── 13. ViolationSeverity Warning vs Critical ──

    #[tokio::test]
    async fn violation_severity_warning_vs_critical() {
        let collector = SloCollector::new();
        // 50ms is ≥80% of 60ms budget (48) but ≤ 60ms → Warning.
        collector
            .record_latency(make_latency("hot_context", 50))
            .await;
        let violations = collector.check_slo_violations().await;
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].severity, ViolationSeverity::Warning);

        // Now add 70ms → P95 exceeds 60ms → Critical.
        let collector2 = SloCollector::new();
        for ms in [50u64, 55, 70] {
            collector2
                .record_latency(make_latency("hot_context", ms))
                .await;
        }
        let violations2 = collector2.check_slo_violations().await;
        assert_eq!(violations2.len(), 1);
        assert_eq!(violations2[0].severity, ViolationSeverity::Critical);
    }

    // ── 14. Snapshot returns current state ──

    #[tokio::test]
    async fn snapshot_returns_current_state() {
        let collector = SloCollector::new();
        collector
            .record_latency(make_latency("standard_recall", 100))
            .await;
        collector.record_timeout().await;

        let snap = collector.snapshot().await;
        assert_eq!(snap.standard_recall_p95_ms, Some(100));
        assert_eq!(snap.total_requests, 1);
        assert_eq!(snap.total_timeouts, 1);
        assert_eq!(snap.total_errors, 1);

        // A second snapshot reflects incremental changes.
        collector
            .record_latency(make_latency("standard_recall", 200))
            .await;
        let snap2 = collector.snapshot().await;
        assert_eq!(snap2.total_requests, 2);
        // 2 values [100,200]; idx=ceil(2*0.95)=2→0-based 1 → 200.
        assert_eq!(snap2.standard_recall_p95_ms, Some(200));
    }

    // ── 15. Record timeout/error increments counters ──

    #[tokio::test]
    async fn record_timeout_error_increments_counters() {
        let collector = SloCollector::new();
        collector.record_timeout().await;
        collector.record_timeout().await;
        collector.record_circuit_open().await;
        collector.record_permission_fail_open().await;

        let snap = collector.snapshot().await;
        assert_eq!(snap.total_timeouts, 2);
        assert_eq!(snap.total_circuit_opens, 1);
        assert_eq!(snap.permission_fail_open_count, 1);
        // timeout + fail-open each bump total_errors → 3.
        assert_eq!(snap.total_errors, 3);
    }

    // ── 16. Forget-request average is tracked ──

    #[tokio::test]
    async fn forget_request_avg_tracked() {
        let collector = SloCollector::new();
        for ms in [100u64, 200, 300] {
            collector.record_latency(make_latency("forget", ms)).await;
        }
        let snap = collector.snapshot().await;
        assert_eq!(snap.forget_request_avg_ms, Some(200));
    }

    // ── 17. No violations when within budget ──

    #[tokio::test]
    async fn no_violations_when_within_budget() {
        let collector = SloCollector::new();
        collector
            .record_latency(make_latency("hot_context", 20))
            .await;
        collector
            .record_latency(make_latency("standard_recall", 50))
            .await;
        let violations = collector.check_slo_violations().await;
        assert!(violations.is_empty());
    }

    // ── 18. P95 handles unsorted input correctly ──

    #[test]
    fn p95_unsorted_input() {
        let p95 = SloCollector::compute_p95(&[100, 10, 50, 30, 80]);
        // sorted: [10,30,50,80,100]; idx=ceil(5*0.95)=5→0-based 4 → 100.
        assert_eq!(p95, Some(100));
    }
}
