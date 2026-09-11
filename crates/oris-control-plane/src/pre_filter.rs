//! Bounded retrieval pre-filter for permission, time, entity, and status filtering.
//!
//! This module constructs optimised SQL WHERE-clause fragments that are applied
//! **before** vector similarity search, reducing the candidate set to meet the
//! P95 ≤ 200 ms SLO defined in §8.5.2 (Bounded Retrieval) and §8.5.3 (Index
//! Design for Filtered Vector Search) of the control-plane architecture.
//!
//! ## Filter Pipeline
//!
//! | #  | Filter       | Condition                                              | Default  |
//! |----|-------------|--------------------------------------------------------|----------|
//! | 1  | Tenant      | `tenant_id = $N`                                       | always   |
//! | 2  | Scope       | `scope = ANY($N)`                                      | always   |
//! | 3  | Status      | `status = $N`                                          | `active` |
//! | 4  | Time window | `valid_from <= $N AND (valid_to IS NULL OR valid_to > $N)` | optional |
//! | 5  | Entity      | `entity_refs @> $N`                                    | optional |
//! | 6  | Top-k       | `LIMIT $N`                                             | 10       |

use oris_memory_store::memory_types::{MemoryStatus, Scope};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ──────────────────────────── TimeWindow ───────────────────────

/// A temporal validity window for filtering memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// The point-in-time at which memories must be valid.
    pub valid_at: chrono::DateTime<chrono::Utc>,
    /// When `false` (default) only non-expired memories are included.
    /// When `true` the `valid_to` check is omitted, allowing expired
    /// memories to appear in results.
    pub include_expired: bool,
}

// ──────────────────────────── FilterParam ──────────────────────

/// A bind parameter value for the generated SQL fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterParam {
    Text(String),
    Uuid(Uuid),
    I64(i64),
    Array(Vec<String>),
}

// ──────────────────────────── PreFilterResult ──────────────────

/// The result of building a pre-filter — SQL fragment + bind parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFilterResult {
    /// SQL WHERE clause fragment (without the `WHERE` keyword).
    pub where_clause: String,
    /// Bind parameters in positional order (for `sqlx::query()`).
    pub params: Vec<FilterParam>,
    /// Estimated selectivity — fraction of the table expected to match.
    pub estimated_selectivity: f64,
    /// The bounded top-k limit.
    pub top_k: usize,
}

// ──────────────────────────── PreFilterError ───────────────────

#[derive(Debug, Clone, thiserror::Error)]
pub enum PreFilterError {
    #[error("tenant_id is required")]
    MissingTenant,
    #[error("at least one scope must be allowed")]
    NoScopes,
    #[error("top_k must be between 1 and 100, got {0}")]
    InvalidTopK(usize),
}

// ──────────────────────────── PreFilterBuilder ─────────────────

/// Builds optimised pre-filter conditions for bounded retrieval.
///
/// The filter is applied **before** vector similarity search to reduce
/// the candidate set and meet the P95 ≤ 200 ms SLO.
#[derive(Debug, Clone)]
pub struct PreFilterBuilder {
    tenant_id: String,
    allowed_scopes: Vec<Scope>,
    status_filter: Option<MemoryStatus>,
    time_window: Option<TimeWindow>,
    entity_filter: Option<Vec<String>>,
    top_k: usize,
}

// Selectivity factors (see §8.5.3).
const SEL_TENANT: f64 = 0.1; // ~10 tenants
const SEL_SCOPE: f64 = 0.5; // 2-3 scopes per tenant
const SEL_STATUS: f64 = 0.3; // ~30% active
const SEL_TIME: f64 = 0.8; // most memories are valid
const SEL_ENTITY: f64 = 0.05; // entity-specific

/// Default and bounded top-k values.
const DEFAULT_TOP_K: usize = 10;
const MAX_TOP_K: usize = 100;
const MIN_TOP_K: usize = 1;

impl PreFilterBuilder {
    /// Create a new builder with the given tenant and allowed scopes.
    pub fn new(tenant_id: &str, allowed_scopes: Vec<Scope>) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            allowed_scopes,
            status_filter: None,
            time_window: None,
            entity_filter: None,
            top_k: DEFAULT_TOP_K,
        }
    }

    /// Set the status filter (defaults to `active` if not called).
    pub fn with_status(mut self, status: MemoryStatus) -> Self {
        self.status_filter = Some(status);
        self
    }

    /// Set the temporal validity window.
    pub fn with_time_window(mut self, window: TimeWindow) -> Self {
        self.time_window = Some(window);
        self
    }

    /// Set the entity filter (JSONB containment on `entity_refs`).
    pub fn with_entities(mut self, entities: Vec<String>) -> Self {
        self.entity_filter = Some(entities);
        self
    }

    /// Set the top-k limit (clamped to `[1, 100]` during `build`).
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Validate that the builder has the minimum required filters.
    pub fn validate(&self) -> Result<(), PreFilterError> {
        if self.tenant_id.trim().is_empty() {
            return Err(PreFilterError::MissingTenant);
        }
        if self.allowed_scopes.is_empty() {
            return Err(PreFilterError::NoScopes);
        }
        if self.top_k < MIN_TOP_K || self.top_k > MAX_TOP_K {
            return Err(PreFilterError::InvalidTopK(self.top_k));
        }
        Ok(())
    }

    /// Estimate selectivity based on active filter conditions.
    /// More filters = lower selectivity = faster query.
    fn estimate_selectivity(&self) -> f64 {
        let mut sel = 1.0_f64;

        // Always-applied filters.
        sel *= SEL_TENANT;
        sel *= SEL_SCOPE;
        sel *= SEL_STATUS;

        // Optional filters.
        if self.time_window.is_some() {
            sel *= SEL_TIME;
        }
        if let Some(ref entities) = self.entity_filter {
            if !entities.is_empty() {
                sel *= SEL_ENTITY;
            }
        }

        sel
    }

    /// Build the pre-filter SQL fragment and bind parameters.
    pub fn build(&self) -> PreFilterResult {
        let mut conditions: Vec<String> = Vec::with_capacity(6);
        let mut params: Vec<FilterParam> = Vec::with_capacity(6);
        let mut idx = 1usize;

        // 1 — Tenant filter (always applied, multi-tenant isolation).
        conditions.push(format!("tenant_id = ${idx}"));
        params.push(FilterParam::Text(self.tenant_id.clone()));
        idx += 1;

        // 2 — Scope filter (always applied, only allowed scopes).
        let scopes: Vec<String> = self
            .allowed_scopes
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        conditions.push(format!("scope = ANY(${idx})"));
        params.push(FilterParam::Array(scopes));
        idx += 1;

        // 3 — Status filter (always applied, defaults to active).
        let status = self.status_filter.unwrap_or(MemoryStatus::Active);
        conditions.push(format!("status = ${idx}"));
        params.push(FilterParam::Text(status.as_str().to_string()));
        idx += 1;

        // 4 — Time window (optional).
        if let Some(ref window) = self.time_window {
            if window.include_expired {
                conditions.push(format!("valid_from <= ${idx}"));
            } else {
                conditions.push(format!(
                    "(valid_from <= ${idx} AND (valid_to IS NULL OR valid_to > ${idx}))"
                ));
            }
            params.push(FilterParam::Text(window.valid_at.to_rfc3339()));
            idx += 1;
        }

        // 5 — Entity filter (optional, GIN-indexed JSONB containment).
        if let Some(ref entities) = self.entity_filter {
            if !entities.is_empty() {
                conditions.push(format!("entity_refs @> ${idx}"));
                params.push(FilterParam::Array(entities.clone()));
            }
        }

        let where_clause = conditions.join(" AND ");

        // Clamp top-k to [1, 100].
        let top_k = self.top_k.clamp(MIN_TOP_K, MAX_TOP_K);
        if top_k != self.top_k {
            tracing::warn!(
                requested = self.top_k,
                clamped = top_k,
                "top_k was clamped to valid range [1, 100]"
            );
        }

        PreFilterResult {
            where_clause,
            params,
            estimated_selectivity: self.estimate_selectivity(),
            top_k,
        }
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use oris_memory_store::memory_types::{MemoryStatus, Scope};

    fn base_builder() -> PreFilterBuilder {
        PreFilterBuilder::new("tenant-1", vec![Scope::Personal, Scope::Agent])
    }

    // 1 — Basic builder with tenant + scope → correct WHERE clause
    #[test]
    fn basic_tenant_scope_where_clause() {
        let result = base_builder().build();
        assert!(result.where_clause.contains("tenant_id = $1"));
        assert!(result.where_clause.contains("scope = ANY($2)"));
        assert!(result.where_clause.contains("status = $3"));
        // Default status is active.
        match &result.params[2] {
            FilterParam::Text(t) => assert_eq!(t, "active"),
            other => panic!("expected Text(\"active\"), got {other:?}"),
        }
    }

    // 2 — Status filter added → status condition in clause
    #[test]
    fn status_filter_in_clause() {
        let result = base_builder().with_status(MemoryStatus::Archived).build();
        match &result.params[2] {
            FilterParam::Text(t) => assert_eq!(t, "archived"),
            other => panic!("expected Text(\"archived\"), got {other:?}"),
        }
        assert!(result.where_clause.contains("status = $3"));
    }

    // 3 — Time window added → valid_from/valid_to condition
    #[test]
    fn time_window_in_clause() {
        let window = TimeWindow {
            valid_at: Utc::now(),
            include_expired: false,
        };
        let result = base_builder().with_time_window(window).build();
        assert!(result.where_clause.contains("valid_from <= $4"));
        assert!(result
            .where_clause
            .contains("valid_to IS NULL OR valid_to > $4"));
    }

    // 4 — Entity filter added → entity_refs @> condition
    #[test]
    fn entity_filter_in_clause() {
        let result = base_builder()
            .with_entities(vec!["entity-1".into(), "entity-2".into()])
            .build();
        assert!(result.where_clause.contains("entity_refs @> $4"));
    }

    // 5 — Top-k default is 10
    #[test]
    fn default_top_k_is_10() {
        let result = base_builder().build();
        assert_eq!(result.top_k, 10);
    }

    // 6 — Top-k clamped to 100 when exceeded
    #[test]
    fn top_k_clamped_to_100() {
        let result = base_builder().with_top_k(500).build();
        assert_eq!(result.top_k, 100);
    }

    // 7 — Top-k minimum is 1
    #[test]
    fn top_k_minimum_is_1() {
        let result = base_builder().with_top_k(0).build();
        assert_eq!(result.top_k, 1);
    }

    // 8 — Validate: missing tenant → error
    #[test]
    fn validate_missing_tenant() {
        let builder = PreFilterBuilder::new("", vec![Scope::Personal]);
        assert!(matches!(
            builder.validate(),
            Err(PreFilterError::MissingTenant)
        ));
    }

    // 9 — Validate: no scopes → error
    #[test]
    fn validate_no_scopes() {
        let builder = PreFilterBuilder::new("tenant-1", vec![]);
        assert!(matches!(builder.validate(), Err(PreFilterError::NoScopes)));
    }

    // 10 — Validate: valid builder → Ok
    #[test]
    fn validate_valid_builder() {
        assert!(base_builder().validate().is_ok());
    }

    // 11 — Selectivity: defaults (tenant + scope + status) → 0.015
    #[test]
    fn selectivity_defaults() {
        let sel = base_builder().estimate_selectivity();
        // 0.1 * 0.5 * 0.3 = 0.015
        assert!((sel - 0.015).abs() < 1e-9);
    }

    // 12 — Selectivity: all filters → low value
    #[test]
    fn selectivity_all_filters() {
        let builder = base_builder()
            .with_time_window(TimeWindow {
                valid_at: Utc::now(),
                include_expired: false,
            })
            .with_entities(vec!["e1".into()]);
        let sel = builder.estimate_selectivity();
        // 0.1 * 0.5 * 0.3 * 0.8 * 0.05 = 0.0006
        assert!((sel - 0.0006).abs() < 1e-9);
        assert!(sel < base_builder().estimate_selectivity());
    }

    // 13 — Params in correct order
    #[test]
    fn params_in_correct_order() {
        let window = TimeWindow {
            valid_at: "2025-01-01T00:00:00Z".parse().unwrap(),
            include_expired: false,
        };
        let result = base_builder()
            .with_status(MemoryStatus::Archived)
            .with_time_window(window)
            .with_entities(vec!["entity-1".into()])
            .build();

        assert_eq!(result.params.len(), 5);
        match &result.params[0] {
            FilterParam::Text(t) => assert_eq!(t, "tenant-1"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &result.params[1] {
            FilterParam::Array(a) => {
                assert_eq!(a.len(), 2);
                assert_eq!(a[0], "personal");
                assert_eq!(a[1], "agent");
            }
            other => panic!("expected Array, got {other:?}"),
        }
        match &result.params[2] {
            FilterParam::Text(t) => assert_eq!(t, "archived"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &result.params[3] {
            FilterParam::Text(_) => {}
            other => panic!("expected Text (timestamp), got {other:?}"),
        }
        match &result.params[4] {
            FilterParam::Array(a) => {
                assert_eq!(a.len(), 1);
                assert_eq!(a[0], "entity-1");
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    // 14 — Empty entity filter → no entity condition
    #[test]
    fn empty_entity_filter_no_condition() {
        let result = base_builder().with_entities(vec![]).build();
        assert!(!result.where_clause.contains("entity_refs"));
    }

    // 15 — Include expired flag changes time window clause
    #[test]
    fn include_expired_changes_clause() {
        let window = TimeWindow {
            valid_at: Utc::now(),
            include_expired: true,
        };
        let result = base_builder().with_time_window(window).build();
        assert!(result.where_clause.contains("valid_from <= $4"));
        assert!(!result.where_clause.contains("valid_to IS NULL"));
    }

    // 16 — include_expired=false includes the valid_to check
    #[test]
    fn include_expired_false_has_valid_to_check() {
        let window = TimeWindow {
            valid_at: Utc::now(),
            include_expired: false,
        };
        let result = base_builder().with_time_window(window).build();
        assert!(result
            .where_clause
            .contains("valid_to IS NULL OR valid_to > $4"));
    }

    // 17 — Conditions joined with AND
    #[test]
    fn conditions_joined_with_and() {
        let result = base_builder().build();
        let parts: Vec<&str> = result.where_clause.split(" AND ").collect();
        assert_eq!(parts.len(), 3); // tenant + scope + status
    }

    // 18 — where_clause does not include the WHERE keyword
    #[test]
    fn build_does_not_include_where_keyword() {
        let result = base_builder().build();
        assert!(!result.where_clause.to_lowercase().starts_with("where"));
    }

    // 19 — Validate: invalid top_k (0) → error
    #[test]
    fn validate_invalid_top_k_zero() {
        let builder = base_builder().with_top_k(0);
        assert!(matches!(
            builder.validate(),
            Err(PreFilterError::InvalidTopK(0))
        ));
    }

    // 20 — Validate: invalid top_k (> 100) → error
    #[test]
    fn validate_invalid_top_k_above_max() {
        let builder = base_builder().with_top_k(200);
        assert!(matches!(
            builder.validate(),
            Err(PreFilterError::InvalidTopK(200))
        ));
    }

    // 21 — Entity filter without time window uses $4
    #[test]
    fn entity_without_time_window_uses_correct_index() {
        let result = base_builder().with_entities(vec!["e1".into()]).build();
        assert!(result.where_clause.contains("entity_refs @> $4"));
        assert_eq!(result.params.len(), 4);
    }

    // 22 — Entity filter with time window uses $5
    #[test]
    fn entity_with_time_window_uses_correct_index() {
        let window = TimeWindow {
            valid_at: Utc::now(),
            include_expired: false,
        };
        let result = base_builder()
            .with_time_window(window)
            .with_entities(vec!["e1".into()])
            .build();
        assert!(result.where_clause.contains("entity_refs @> $5"));
        assert_eq!(result.params.len(), 5);
    }
}
