//! Cost Tracking & LLM Resource Management (§12, §15).
//!
//! Tracks LLM call costs, embedding costs, and storage costs by tenant and
//! agent dimensions.  Provides budget alerting, engine cost comparison, and
//! aggregated cost reporting for operational cost assessment.
//!
//! ## Architecture references
//!
//! - §12 — Observability: cost as an operational metric.
//! - §15 — Cost optimization: storage cost optimization and engine comparison.
//! - [`crate::slo_monitoring`] — shared `RwLock` collector pattern.
//! - [`crate::engine`] — `MemoryEngine` trait for engine comparison.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// ──────────────────────────── CostType ────────────────────────────

/// The category of a cost entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostType {
    Embedding,
    LlmCall,
    Storage,
    Total,
}

// ──────────────────────────── TimePeriod ────────────────────────────

/// Reporting period for cost aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimePeriod {
    Daily,
    Weekly,
    Monthly,
}

impl TimePeriod {
    /// Duration of the period in seconds.
    pub fn duration_secs(self) -> i64 {
        match self {
            Self::Daily => 86_400,
            Self::Weekly => 86_400 * 7,
            Self::Monthly => 86_400 * 30,
        }
    }
}

// ──────────────────────────── CostEntry ────────────────────────────

/// A single recorded cost event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub id: uuid::Uuid,
    pub tenant_id: String,
    pub agent_id: Option<String>,
    pub cost_type: CostType,
    pub amount: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: serde_json::Value,
}

// ──────────────────────────── CostBreakdown ────────────────────────────

/// Cost breakdown by category.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub embedding_cost: f64,
    pub llm_cost: f64,
    pub storage_cost: f64,
}

// ──────────────────────────── CostReport ────────────────────────────

/// Aggregated cost report for a tenant over a period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostReport {
    pub tenant_id: String,
    pub period: TimePeriod,
    pub entries: Vec<CostEntry>,
    pub total_cost: f64,
    pub breakdown: CostBreakdown,
}

// ──────────────────────────── AlertType ────────────────────────────

/// The type of cost alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertType {
    DailyBudgetExceeded,
    MonthlyBudgetExceeded,
    UnusualSpike,
    StorageGrowthRate,
}

// ──────────────────────────── CostAlert ────────────────────────────

/// A cost alert triggered when a threshold is exceeded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlert {
    pub tenant_id: String,
    pub alert_type: AlertType,
    pub current_amount: f64,
    pub threshold: f64,
    pub message: String,
}

// ──────────────────────────── CostThresholds ────────────────────────────

/// Configurable cost limits for a tenant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostThresholds {
    /// Maximum total cost allowed per day.
    pub daily_budget: Option<f64>,
    /// Maximum total cost allowed per month.
    pub monthly_budget: Option<f64>,
    /// Multiplier over the rolling hourly average that constitutes a spike (e.g. 3.0).
    pub spike_multiplier: Option<f64>,
    /// Maximum acceptable storage cost growth rate (fraction, e.g. 0.5 = 50%).
    pub storage_growth_rate: Option<f64>,
}

// ──────────────────────────── EngineCostComparison ────────────────────────────

/// Per-engine cost and performance comparison data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineCostComparison {
    pub engine_name: String,
    pub avg_cost_per_request: f64,
    pub avg_latency_ms: f64,
    pub quality_score: f64,
}

// ──────────────────────────── EngineMetricsInput ────────────────────────────

/// Engine performance data recorded for cost-vs-quality comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineMetricsInput {
    pub engine_name: String,
    pub cost: f64,
    pub latency_ms: f64,
    pub quality_score: f64,
}

// ──────────────────────────── CostError ────────────────────────────

/// Errors returned by the cost tracking subsystem.
#[derive(Debug, thiserror::Error)]
pub enum CostError {
    #[error("tenant not found: {0}")]
    TenantNotFound(String),
    #[error("invalid cost amount: {0}")]
    InvalidAmount(f64),
    #[error("invalid token count: {0}")]
    InvalidTokenCount(u64),
    #[error("report generation failed: {0}")]
    ReportFailed(String),
}

// ──────────────────────────── Internal state ────────────────────────────

/// Internal mutable state kept behind the tracker's `RwLock`.
#[derive(Debug, Clone, Default)]
struct TrackerState {
    entries: Vec<CostEntry>,
    thresholds: HashMap<String, CostThresholds>,
    /// Per-engine accumulators for `compare_engines`.
    engine_accumulators: HashMap<String, EngineAccumulator>,
}

/// Running totals for a single engine.
#[derive(Debug, Clone, Default)]
struct EngineAccumulator {
    total_cost: f64,
    total_latency_ms: f64,
    total_quality_score: f64,
    count: u64,
}

// ──────────────────────────── CostTracker ────────────────────────────

/// Collects cost entries and thresholds, providing the raw data that
/// [`CostDashboard`] turns into reports and alerts.
///
/// Designed to be wrapped in `Arc` and shared across the control plane.
/// All mutations go through the `record_*` methods; readers obtain
/// snapshots via the dashboard.
pub struct CostTracker {
    state: RwLock<TrackerState>,
}

impl CostTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            state: RwLock::new(TrackerState::default()),
        }
    }

    /// Record an embedding cost entry.
    ///
    /// `cost_per_token` is in USD (or any consistent currency unit).
    pub async fn record_embedding_cost(
        &self,
        tenant_id: &str,
        agent_id: &str,
        tokens: u64,
        cost_per_token: f64,
    ) -> Result<uuid::Uuid, CostError> {
        if tokens == 0 {
            return Err(CostError::InvalidTokenCount(0));
        }
        if cost_per_token < 0.0 {
            return Err(CostError::InvalidAmount(cost_per_token));
        }
        let amount = tokens as f64 * cost_per_token;
        let id = uuid::Uuid::new_v4();
        let entry = CostEntry {
            id,
            tenant_id: tenant_id.to_string(),
            agent_id: Some(agent_id.to_string()),
            cost_type: CostType::Embedding,
            amount,
            timestamp: chrono::Utc::now(),
            metadata: serde_json::json!({
                "tokens": tokens,
                "cost_per_token": cost_per_token,
            }),
        };
        self.state.write().await.entries.push(entry);
        Ok(id)
    }

    /// Record an LLM call cost entry.
    ///
    /// Pricing is determined by the model name via [`model_pricing`].
    pub async fn record_llm_call(
        &self,
        tenant_id: &str,
        agent_id: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<uuid::Uuid, CostError> {
        if input_tokens == 0 && output_tokens == 0 {
            return Err(CostError::InvalidTokenCount(0));
        }
        let (input_price, output_price) = model_pricing(model);
        let amount = (input_tokens as f64 / 1000.0) * input_price
            + (output_tokens as f64 / 1000.0) * output_price;
        let id = uuid::Uuid::new_v4();
        let entry = CostEntry {
            id,
            tenant_id: tenant_id.to_string(),
            agent_id: Some(agent_id.to_string()),
            cost_type: CostType::LlmCall,
            amount,
            timestamp: chrono::Utc::now(),
            metadata: serde_json::json!({
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_price_per_1k": input_price,
                "output_price_per_1k": output_price,
            }),
        };
        self.state.write().await.entries.push(entry);
        Ok(id)
    }

    /// Record a storage cost entry.
    ///
    /// `cost_per_gb_month` is the monthly cost per gigabyte.  The recorded
    /// `amount` is prorated for a single day (1/30 of the monthly rate).
    pub async fn record_storage_cost(
        &self,
        tenant_id: &str,
        bytes: u64,
        cost_per_gb_month: f64,
    ) -> Result<uuid::Uuid, CostError> {
        if cost_per_gb_month < 0.0 {
            return Err(CostError::InvalidAmount(cost_per_gb_month));
        }
        let gb = bytes as f64 / 1_073_741_824.0; // 1 GiB
        let amount = gb * cost_per_gb_month / 30.0; // daily proration
        let id = uuid::Uuid::new_v4();
        let entry = CostEntry {
            id,
            tenant_id: tenant_id.to_string(),
            agent_id: None,
            cost_type: CostType::Storage,
            amount,
            timestamp: chrono::Utc::now(),
            metadata: serde_json::json!({
                "bytes": bytes,
                "cost_per_gb_month": cost_per_gb_month,
                "gb": gb,
            }),
        };
        self.state.write().await.entries.push(entry);
        Ok(id)
    }

    /// Set cost thresholds for a tenant.
    pub async fn set_thresholds(&self, tenant_id: &str, thresholds: CostThresholds) {
        self.state
            .write()
            .await
            .thresholds
            .insert(tenant_id.to_string(), thresholds);
    }

    /// Record engine performance metrics for cost-vs-quality comparison.
    pub async fn record_engine_metrics(&self, metrics: EngineMetricsInput) {
        let mut state = self.state.write().await;
        let acc = state
            .engine_accumulators
            .entry(metrics.engine_name.clone())
            .or_default();
        acc.total_cost += metrics.cost;
        acc.total_latency_ms += metrics.latency_ms;
        acc.total_quality_score += metrics.quality_score;
        acc.count += 1;
    }

    // ── read helpers (used by CostDashboard) ──

    /// Return all entries for a tenant within the given period.
    async fn entries_for_period(&self, tenant_id: &str, period: TimePeriod) -> Vec<CostEntry> {
        let state = self.state.read().await;
        let now = chrono::Utc::now();
        let cutoff = now - chrono::Duration::seconds(period.duration_secs());
        state
            .entries
            .iter()
            .filter(|e| e.tenant_id == tenant_id && e.timestamp >= cutoff)
            .cloned()
            .collect()
    }

    /// Return a snapshot of all thresholds.
    async fn thresholds_snapshot(&self) -> HashMap<String, CostThresholds> {
        self.state.read().await.thresholds.clone()
    }

    /// Return engine comparison data.
    async fn engine_comparisons(&self) -> Vec<EngineCostComparison> {
        let state = self.state.read().await;
        state
            .engine_accumulators
            .iter()
            .map(|(name, acc)| {
                let count = acc.count.max(1) as f64;
                EngineCostComparison {
                    engine_name: name.clone(),
                    avg_cost_per_request: acc.total_cost / count,
                    avg_latency_ms: acc.total_latency_ms / count,
                    quality_score: acc.total_quality_score / count,
                }
            })
            .collect()
    }

    /// Return all entries for a tenant (any period) — used for alert checks.
    async fn all_entries_for_tenant(&self, tenant_id: &str) -> Vec<CostEntry> {
        let state = self.state.read().await;
        state
            .entries
            .iter()
            .filter(|e| e.tenant_id == tenant_id)
            .cloned()
            .collect()
    }

    /// Insert a raw entry with a specific timestamp (test/helper use).
    #[cfg(test)]
    pub(crate) async fn push_entry_raw(&self, entry: CostEntry) {
        self.state.write().await.entries.push(entry);
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CostTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostTracker").finish_non_exhaustive()
    }
}

// ──────────────────────────── CostDashboard ────────────────────────────

/// High-level cost dashboard that wraps a [`CostTracker`] and produces
/// reports, alerts, and engine comparisons.
pub struct CostDashboard {
    tracker: Arc<CostTracker>,
}

impl CostDashboard {
    /// Create a new dashboard backed by the given tracker.
    pub fn new(tracker: Arc<CostTracker>) -> Self {
        Self { tracker }
    }

    /// Generate a cost report for a tenant over a period.
    pub async fn generate_report(
        &self,
        tenant_id: &str,
        period: TimePeriod,
    ) -> Result<CostReport, CostError> {
        let entries = self.tracker.entries_for_period(tenant_id, period).await;
        let mut breakdown = CostBreakdown::default();
        for e in &entries {
            match e.cost_type {
                CostType::Embedding => breakdown.embedding_cost += e.amount,
                CostType::LlmCall => breakdown.llm_cost += e.amount,
                CostType::Storage => breakdown.storage_cost += e.amount,
                CostType::Total => {}
            }
        }
        let total_cost = breakdown.embedding_cost + breakdown.llm_cost + breakdown.storage_cost;
        Ok(CostReport {
            tenant_id: tenant_id.to_string(),
            period,
            entries,
            total_cost,
            breakdown,
        })
    }

    /// Check all tenants' thresholds and return triggered alerts.
    pub async fn check_alerts(&self) -> Vec<CostAlert> {
        let thresholds = self.tracker.thresholds_snapshot().await;
        let mut alerts = Vec::new();

        for (tenant_id, t) in &thresholds {
            let entries = self.tracker.all_entries_for_tenant(tenant_id).await;
            let now = chrono::Utc::now();

            // Daily budget.
            if let Some(daily_budget) = t.daily_budget {
                let daily_total: f64 = entries
                    .iter()
                    .filter(|e| e.timestamp >= now - chrono::Duration::days(1))
                    .map(|e| e.amount)
                    .sum();
                if daily_total > daily_budget {
                    alerts.push(CostAlert {
                        tenant_id: tenant_id.clone(),
                        alert_type: AlertType::DailyBudgetExceeded,
                        current_amount: daily_total,
                        threshold: daily_budget,
                        message: format!(
                            "Daily budget exceeded: ${daily_total:.2} > ${daily_budget:.2}"
                        ),
                    });
                }
            }

            // Monthly budget.
            if let Some(monthly_budget) = t.monthly_budget {
                let monthly_total: f64 = entries
                    .iter()
                    .filter(|e| e.timestamp >= now - chrono::Duration::days(30))
                    .map(|e| e.amount)
                    .sum();
                if monthly_total > monthly_budget {
                    alerts.push(CostAlert {
                        tenant_id: tenant_id.clone(),
                        alert_type: AlertType::MonthlyBudgetExceeded,
                        current_amount: monthly_total,
                        threshold: monthly_budget,
                        message: format!(
                            "Monthly budget exceeded: ${monthly_total:.2} > ${monthly_budget:.2}"
                        ),
                    });
                }
            }

            // Unusual spike — compare last hour to rolling daily average.
            if let Some(spike_mult) = t.spike_multiplier {
                let one_hour_ago = now - chrono::Duration::hours(1);
                let one_day_ago = now - chrono::Duration::days(1);
                let recent: f64 = entries
                    .iter()
                    .filter(|e| e.timestamp >= one_hour_ago)
                    .map(|e| e.amount)
                    .sum();
                let baseline: f64 = entries
                    .iter()
                    .filter(|e| e.timestamp >= one_day_ago && e.timestamp < one_hour_ago)
                    .map(|e| e.amount)
                    .sum();
                if baseline > 0.0 {
                    let hourly_avg = baseline / 23.0; // 23 hours in the baseline window
                    if recent > hourly_avg * spike_mult {
                        alerts.push(CostAlert {
                            tenant_id: tenant_id.clone(),
                            alert_type: AlertType::UnusualSpike,
                            current_amount: recent,
                            threshold: hourly_avg * spike_mult,
                            message: format!(
                                "Unusual cost spike: ${recent:.2} > {spike_mult}x normal ${hourly_avg:.2}/hr"
                            ),
                        });
                    }
                }
            }

            // Storage growth rate.
            if let Some(growth_threshold) = t.storage_growth_rate {
                let storage_now: f64 = entries
                    .iter()
                    .filter(|e| {
                        e.cost_type == CostType::Storage
                            && e.timestamp >= now - chrono::Duration::days(1)
                    })
                    .map(|e| e.amount)
                    .sum();
                let storage_prev: f64 = entries
                    .iter()
                    .filter(|e| {
                        e.cost_type == CostType::Storage
                            && e.timestamp < now - chrono::Duration::days(1)
                            && e.timestamp >= now - chrono::Duration::days(2)
                    })
                    .map(|e| e.amount)
                    .sum();
                if storage_prev > 0.0 {
                    let growth = (storage_now - storage_prev) / storage_prev;
                    if growth > growth_threshold {
                        alerts.push(CostAlert {
                            tenant_id: tenant_id.clone(),
                            alert_type: AlertType::StorageGrowthRate,
                            current_amount: growth,
                            threshold: growth_threshold,
                            message: format!(
                                "Storage growth rate {:.1}% exceeds threshold {:.1}%",
                                growth * 100.0,
                                growth_threshold * 100.0
                            ),
                        });
                    }
                }
            }
        }

        alerts
    }

    /// Compare cost and performance across recorded engines.
    pub async fn compare_engines(&self) -> Vec<EngineCostComparison> {
        self.tracker.engine_comparisons().await
    }
}

impl std::fmt::Debug for CostDashboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostDashboard").finish_non_exhaustive()
    }
}

// ──────────────────────────── Model pricing ────────────────────────────

/// Return `(input_price_per_1k_tokens, output_price_per_1k_tokens)` for
/// common LLM models.  Unknown models fall back to a conservative default.
pub fn model_pricing(model: &str) -> (f64, f64) {
    match model.to_ascii_lowercase().as_str() {
        "gpt-4" | "gpt-4-turbo" => (0.03, 0.06),
        "gpt-4o" => (0.005, 0.015),
        "gpt-4o-mini" => (0.00015, 0.0006),
        "gpt-3.5-turbo" => (0.0005, 0.0015),
        "claude-3-opus" => (0.015, 0.075),
        "claude-3-sonnet" | "claude-3-5-sonnet" => (0.003, 0.015),
        "claude-3-haiku" => (0.00025, 0.00125),
        "text-embedding-3-small" => (0.00002, 0.0),
        "text-embedding-3-large" => (0.00013, 0.0),
        _ => (0.01, 0.03),
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        tenant: &str,
        cost_type: CostType,
        amount: f64,
        ts: chrono::DateTime<chrono::Utc>,
    ) -> CostEntry {
        CostEntry {
            id: uuid::Uuid::new_v4(),
            tenant_id: tenant.to_string(),
            agent_id: Some("agent-1".to_string()),
            cost_type,
            amount,
            timestamp: ts,
            metadata: serde_json::Value::Null,
        }
    }

    // ── 1. record_embedding_cost basic ──

    #[tokio::test]
    async fn record_embedding_cost_basic() {
        let tracker = CostTracker::new();
        let id = tracker
            .record_embedding_cost("t1", "a1", 1000, 0.00002)
            .await
            .unwrap();
        assert!(!id.to_string().is_empty());
        let report = CostDashboard::new(Arc::new(CostTracker::new()))
            .generate_report("t1", TimePeriod::Daily)
            .await
            .unwrap();
        // Fresh tracker → empty report.
        assert_eq!(report.total_cost, 0.0);
    }

    // ── 2. record_embedding_cost computes amount ──

    #[tokio::test]
    async fn record_embedding_cost_computes_amount() {
        let tracker = CostTracker::new();
        tracker
            .record_embedding_cost("t1", "a1", 1000, 0.00002)
            .await
            .unwrap();
        let dash = CostDashboard::new(Arc::new(tracker));
        let report = dash.generate_report("t1", TimePeriod::Daily).await.unwrap();
        assert_eq!(report.entries.len(), 1);
        assert!((report.entries[0].amount - 0.02).abs() < 1e-9);
        assert_eq!(report.breakdown.embedding_cost, 0.02);
        assert_eq!(report.total_cost, 0.02);
    }

    // ── 3. record_embedding_cost zero tokens errors ──

    #[tokio::test]
    async fn record_embedding_cost_zero_tokens() {
        let tracker = CostTracker::new();
        let err = tracker
            .record_embedding_cost("t1", "a1", 0, 0.00002)
            .await
            .unwrap_err();
        assert!(matches!(err, CostError::InvalidTokenCount(0)));
    }

    // ── 4. record_embedding_cost negative cost errors ──

    #[tokio::test]
    async fn record_embedding_cost_negative_cost() {
        let tracker = CostTracker::new();
        let err = tracker
            .record_embedding_cost("t1", "a1", 100, -0.01)
            .await
            .unwrap_err();
        assert!(matches!(err, CostError::InvalidAmount(_)));
    }

    // ── 5. record_llm_call basic ──

    #[tokio::test]
    async fn record_llm_call_basic() {
        let tracker = CostTracker::new();
        tracker
            .record_llm_call("t1", "a1", "gpt-4o", 1000, 500)
            .await
            .unwrap();
        let dash = CostDashboard::new(Arc::new(tracker));
        let report = dash.generate_report("t1", TimePeriod::Daily).await.unwrap();
        // gpt-4o: 0.005/1k in, 0.015/1k out → 1000*0.005/1000 + 500*0.015/1000
        let expected = (1000.0 / 1000.0) * 0.005 + (500.0 / 1000.0) * 0.015;
        assert!((report.total_cost - expected).abs() < 1e-9);
        assert_eq!(report.breakdown.llm_cost, expected);
    }

    // ── 6. record_llm_call zero tokens errors ──

    #[tokio::test]
    async fn record_llm_call_zero_tokens() {
        let tracker = CostTracker::new();
        let err = tracker
            .record_llm_call("t1", "a1", "gpt-4o", 0, 0)
            .await
            .unwrap_err();
        assert!(matches!(err, CostError::InvalidTokenCount(0)));
    }

    // ── 7. record_llm_call unknown model uses default ──

    #[tokio::test]
    async fn record_llm_call_unknown_model() {
        let tracker = CostTracker::new();
        tracker
            .record_llm_call("t1", "a1", "unknown-model", 1000, 0)
            .await
            .unwrap();
        let dash = CostDashboard::new(Arc::new(tracker));
        let report = dash.generate_report("t1", TimePeriod::Daily).await.unwrap();
        // default pricing: (0.01, 0.03) → 1000*0.01/1000 = 0.01
        assert!((report.total_cost - 0.01).abs() < 1e-9);
    }

    // ── 8. record_storage_cost basic ──

    #[tokio::test]
    async fn record_storage_cost_basic() {
        let tracker = CostTracker::new();
        // 1 GiB = 1073741824 bytes, $0.10/GB-month → daily = 0.10/30
        tracker
            .record_storage_cost("t1", 1_073_741_824, 0.10)
            .await
            .unwrap();
        let dash = CostDashboard::new(Arc::new(tracker));
        let report = dash.generate_report("t1", TimePeriod::Daily).await.unwrap();
        let expected = 1.0 * 0.10 / 30.0;
        assert!((report.total_cost - expected).abs() < 1e-9);
        assert_eq!(report.breakdown.storage_cost, expected);
    }

    // ── 9. record_storage_cost negative cost errors ──

    #[tokio::test]
    async fn record_storage_cost_negative_cost() {
        let tracker = CostTracker::new();
        let err = tracker
            .record_storage_cost("t1", 1000, -0.10)
            .await
            .unwrap_err();
        assert!(matches!(err, CostError::InvalidAmount(_)));
    }

    // ── 10. generate_report empty ──

    #[tokio::test]
    async fn generate_report_empty() {
        let tracker = CostTracker::new();
        let dash = CostDashboard::new(Arc::new(tracker));
        let report = dash
            .generate_report("nonexistent", TimePeriod::Monthly)
            .await
            .unwrap();
        assert!(report.entries.is_empty());
        assert_eq!(report.total_cost, 0.0);
        assert_eq!(report.breakdown.embedding_cost, 0.0);
        assert_eq!(report.breakdown.llm_cost, 0.0);
        assert_eq!(report.breakdown.storage_cost, 0.0);
    }

    // ── 11. generate_report breakdown by type ──

    #[tokio::test]
    async fn generate_report_breakdown_by_type() {
        let tracker = CostTracker::new();
        tracker
            .record_embedding_cost("t1", "a1", 1000, 0.00002)
            .await
            .unwrap();
        tracker
            .record_llm_call("t1", "a1", "gpt-4o", 1000, 500)
            .await
            .unwrap();
        tracker
            .record_storage_cost("t1", 1_073_741_824, 0.10)
            .await
            .unwrap();
        let dash = CostDashboard::new(Arc::new(tracker));
        let report = dash.generate_report("t1", TimePeriod::Daily).await.unwrap();
        assert_eq!(report.entries.len(), 3);
        assert!((report.breakdown.embedding_cost - 0.02).abs() < 1e-9);
        assert!(report.breakdown.llm_cost > 0.0);
        assert!(report.breakdown.storage_cost > 0.0);
        assert!(
            (report.total_cost
                - (report.breakdown.embedding_cost
                    + report.breakdown.llm_cost
                    + report.breakdown.storage_cost))
                .abs()
                < 1e-9
        );
    }

    // ── 12. check_alerts daily budget exceeded ──

    #[tokio::test]
    async fn check_alerts_daily_budget_exceeded() {
        let tracker = CostTracker::new();
        tracker
            .record_llm_call("t1", "a1", "gpt-4o", 100_000, 50_000)
            .await
            .unwrap();
        tracker
            .set_thresholds(
                "t1",
                CostThresholds {
                    daily_budget: Some(0.01),
                    ..Default::default()
                },
            )
            .await;
        let dash = CostDashboard::new(Arc::new(tracker));
        let alerts = dash.check_alerts().await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].alert_type, AlertType::DailyBudgetExceeded);
        assert!(alerts[0].current_amount > alerts[0].threshold);
    }

    // ── 13. check_alerts no alerts within budget ──

    #[tokio::test]
    async fn check_alerts_no_alerts_within_budget() {
        let tracker = CostTracker::new();
        tracker
            .record_llm_call("t1", "a1", "gpt-4o-mini", 100, 50)
            .await
            .unwrap();
        tracker
            .set_thresholds(
                "t1",
                CostThresholds {
                    daily_budget: Some(100.0),
                    monthly_budget: Some(1000.0),
                    ..Default::default()
                },
            )
            .await;
        let dash = CostDashboard::new(Arc::new(tracker));
        let alerts = dash.check_alerts().await;
        assert!(alerts.is_empty());
    }

    // ── 14. check_alerts monthly budget exceeded ──

    #[tokio::test]
    async fn check_alerts_monthly_budget_exceeded() {
        let tracker = CostTracker::new();
        tracker
            .record_llm_call("t1", "a1", "gpt-4", 100_000, 50_000)
            .await
            .unwrap();
        tracker
            .set_thresholds(
                "t1",
                CostThresholds {
                    monthly_budget: Some(0.01),
                    ..Default::default()
                },
            )
            .await;
        let dash = CostDashboard::new(Arc::new(tracker));
        let alerts = dash.check_alerts().await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].alert_type, AlertType::MonthlyBudgetExceeded);
    }

    // ── 15. check_alerts unusual spike ──

    #[tokio::test]
    async fn check_alerts_unusual_spike() {
        let tracker = CostTracker::new();
        let now = chrono::Utc::now();
        // 23 baseline entries at $1 each, timestamps ~2 hours ago.
        for i in 0..23u32 {
            let ts = now - chrono::Duration::hours(2) - chrono::Duration::seconds(i as i64);
            tracker
                .push_entry_raw(make_entry("t1", CostType::LlmCall, 1.0, ts))
                .await;
        }
        // 5 recent entries at $10 each, timestamps now.
        for i in 0..5u32 {
            let ts = now - chrono::Duration::seconds(i as i64);
            tracker
                .push_entry_raw(make_entry("t1", CostType::LlmCall, 10.0, ts))
                .await;
        }
        tracker
            .set_thresholds(
                "t1",
                CostThresholds {
                    spike_multiplier: Some(2.0),
                    ..Default::default()
                },
            )
            .await;
        let dash = CostDashboard::new(Arc::new(tracker));
        let alerts = dash.check_alerts().await;
        let spike_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.alert_type == AlertType::UnusualSpike)
            .collect();
        assert_eq!(spike_alerts.len(), 1);
        // recent = 50, hourly_avg = 23/23 = 1, threshold = 2.0
        assert!((spike_alerts[0].current_amount - 50.0).abs() < 1e-9);
        assert!((spike_alerts[0].threshold - 2.0).abs() < 1e-9);
    }

    // ── 16. check_alerts storage growth rate ──

    #[tokio::test]
    async fn check_alerts_storage_growth_rate() {
        let tracker = CostTracker::new();
        let now = chrono::Utc::now();
        // Yesterday: $1 storage.
        tracker
            .push_entry_raw(make_entry(
                "t1",
                CostType::Storage,
                1.0,
                now - chrono::Duration::hours(30),
            ))
            .await;
        // Today: $3 storage (200% growth).
        tracker
            .push_entry_raw(make_entry(
                "t1",
                CostType::Storage,
                3.0,
                now - chrono::Duration::minutes(10),
            ))
            .await;
        tracker
            .set_thresholds(
                "t1",
                CostThresholds {
                    storage_growth_rate: Some(0.5),
                    ..Default::default()
                },
            )
            .await;
        let dash = CostDashboard::new(Arc::new(tracker));
        let alerts = dash.check_alerts().await;
        let growth_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.alert_type == AlertType::StorageGrowthRate)
            .collect();
        assert_eq!(growth_alerts.len(), 1);
        assert!((growth_alerts[0].current_amount - 2.0).abs() < 1e-9);
        assert!((growth_alerts[0].threshold - 0.5).abs() < 1e-9);
    }

    // ── 17. compare_engines basic ──

    #[tokio::test]
    async fn compare_engines_basic() {
        let tracker = CostTracker::new();
        tracker
            .record_engine_metrics(EngineMetricsInput {
                engine_name: "mem0".into(),
                cost: 0.10,
                latency_ms: 50.0,
                quality_score: 0.85,
            })
            .await;
        tracker
            .record_engine_metrics(EngineMetricsInput {
                engine_name: "mem0".into(),
                cost: 0.30,
                latency_ms: 70.0,
                quality_score: 0.95,
            })
            .await;
        tracker
            .record_engine_metrics(EngineMetricsInput {
                engine_name: "cognee".into(),
                cost: 0.20,
                latency_ms: 100.0,
                quality_score: 0.90,
            })
            .await;
        let dash = CostDashboard::new(Arc::new(tracker));
        let comparisons = dash.compare_engines().await;
        assert_eq!(comparisons.len(), 2);
        let mem0 = comparisons
            .iter()
            .find(|c| c.engine_name == "mem0")
            .unwrap();
        assert!((mem0.avg_cost_per_request - 0.20).abs() < 1e-9);
        assert!((mem0.avg_latency_ms - 60.0).abs() < 1e-9);
        assert!((mem0.quality_score - 0.90).abs() < 1e-9);
        let cognee = comparisons
            .iter()
            .find(|c| c.engine_name == "cognee")
            .unwrap();
        assert!((cognee.avg_cost_per_request - 0.20).abs() < 1e-9);
        assert!((cognee.avg_latency_ms - 100.0).abs() < 1e-9);
    }

    // ── 18. compare_engines empty ──

    #[tokio::test]
    async fn compare_engines_empty() {
        let tracker = CostTracker::new();
        let dash = CostDashboard::new(Arc::new(tracker));
        let comparisons = dash.compare_engines().await;
        assert!(comparisons.is_empty());
    }

    // ── 19. model_pricing known models ──

    #[test]
    fn model_pricing_known_models() {
        let (in_p, out_p) = model_pricing("gpt-4o");
        assert_eq!(in_p, 0.005);
        assert_eq!(out_p, 0.015);

        let (in_p, _) = model_pricing("GPT-4"); // case-insensitive
        assert_eq!(in_p, 0.03);

        let (in_p, out_p) = model_pricing("claude-3-opus");
        assert_eq!(in_p, 0.015);
        assert_eq!(out_p, 0.075);
    }

    // ── 20. model_pricing unknown model ──

    #[test]
    fn model_pricing_unknown_model() {
        let (in_p, out_p) = model_pricing("custom-finetune-v2");
        assert_eq!(in_p, 0.01);
        assert_eq!(out_p, 0.03);
    }

    // ── 21. time_period durations ──

    #[test]
    fn time_period_durations() {
        assert_eq!(TimePeriod::Daily.duration_secs(), 86_400);
        assert_eq!(TimePeriod::Weekly.duration_secs(), 86_400 * 7);
        assert_eq!(TimePeriod::Monthly.duration_secs(), 86_400 * 30);
    }

    // ── 22. cost_type serde round-trip ──

    #[test]
    fn cost_type_serde_roundtrip() {
        for variant in [
            CostType::Embedding,
            CostType::LlmCall,
            CostType::Storage,
            CostType::Total,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: CostType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
        assert_eq!(
            serde_json::to_string(&CostType::LlmCall).unwrap(),
            "\"llm_call\""
        );
    }

    // ── 23. set_thresholds persists across calls ──

    #[tokio::test]
    async fn set_thresholds_persists() {
        let tracker = CostTracker::new();
        tracker
            .set_thresholds(
                "t1",
                CostThresholds {
                    daily_budget: Some(5.0),
                    monthly_budget: Some(100.0),
                    spike_multiplier: Some(3.0),
                    storage_growth_rate: Some(0.5),
                },
            )
            .await;
        // Record a small cost that's within budget.
        tracker
            .record_embedding_cost("t1", "a1", 100, 0.00002)
            .await
            .unwrap();
        let dash = CostDashboard::new(Arc::new(tracker));
        let alerts = dash.check_alerts().await;
        assert!(alerts.is_empty());
    }

    // ── 24. report filters by tenant ──

    #[tokio::test]
    async fn report_filters_by_tenant() {
        let tracker = CostTracker::new();
        tracker
            .record_embedding_cost("t1", "a1", 1000, 0.00002)
            .await
            .unwrap();
        tracker
            .record_embedding_cost("t2", "a1", 2000, 0.00002)
            .await
            .unwrap();
        let dash = CostDashboard::new(Arc::new(tracker));
        let r1 = dash.generate_report("t1", TimePeriod::Daily).await.unwrap();
        let r2 = dash.generate_report("t2", TimePeriod::Daily).await.unwrap();
        assert_eq!(r1.entries.len(), 1);
        assert_eq!(r2.entries.len(), 1);
        assert!((r1.total_cost - 0.02).abs() < 1e-9);
        assert!((r2.total_cost - 0.04).abs() < 1e-9);
    }
}
