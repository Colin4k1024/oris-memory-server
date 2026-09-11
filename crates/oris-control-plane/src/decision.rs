//! Decision Record — lifecycle management for human-in-the-loop decisions.
//!
//! Implements the decision lifecycle defined in architecture doc §3.4:
//! situation → options → recommendation → human_decision → action → outcome.
//!
//! Each [`DecisionRecord`] captures the full chain from problem framing
//! through final outcome, with evidence references for auditability. The
//! [`DecisionManager`] enforces a state machine that prevents skipping
//! lifecycle stages (e.g., recording an outcome before a human decision).
//!
//! # State machine
//!
//! ```text
//! Proposed → Recommended → Decided → Executed → Resolved
//!    │           │             │          │          │
//!    └───────────┴─────────────┴──────────┴──────────┴──→ Withdrawn
//! ```

use chrono::{DateTime, Utc};
use oris_memory_store::postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::Row;
use thiserror::Error;
use tracing::instrument;
use uuid::Uuid;

// ──────────────────────────── Constants ────────────────────────────

/// Marker value stored in `human_decision` to indicate a withdrawn decision.
const WITHDRAWN_MARKER: &str = "withdrawn";

// ──────────────────────────── Errors ────────────────────────────

/// Errors that can occur during decision record operations.
#[derive(Debug, Error)]
pub enum DecisionError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("decision not found: {0}")]
    NotFound(Uuid),

    #[error("invalid state transition: {0}")]
    InvalidTransition(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ──────────────────────────── Types ────────────────────────────

/// A decision record captures the full decision lifecycle:
/// situation → options → recommendation → human_decision → action → outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub decision_id: Uuid,
    pub task_id: Option<Uuid>,
    pub situation: String,
    pub options: Vec<DecisionOption>,
    pub recommendation: Option<String>,
    pub human_decision: Option<String>,
    pub reason: Option<String>,
    pub action: Option<String>,
    pub outcome: Option<String>,
    pub evidence_refs: Vec<Value>,
    pub decided_at: Option<DateTime<Utc>>,
    pub outcome_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// A single option considered during decision-making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    pub id: String,
    pub description: String,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
    pub risk_level: RiskLevel,
    pub estimated_impact: Option<String>,
}

/// Risk classification for a decision option.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Decision lifecycle states.
///
/// The state is *derived* from the record's populated fields (see
/// [`DecisionManager::get_state`]) rather than stored in a dedicated column.
/// [`DecisionState::Withdrawn`] is encoded by setting `human_decision` to
/// [`WITHDRAWN_MARKER`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionState {
    /// Options gathered, no recommendation yet.
    Proposed,
    /// AI/system recommendation made, awaiting human decision.
    Recommended,
    /// Human has decided.
    Decided,
    /// Action has been taken based on decision.
    Executed,
    /// Outcome has been recorded.
    Resolved,
    /// Decision was superseded or withdrawn.
    Withdrawn,
}

/// Request payload for creating a new decision record.
#[derive(Debug, Clone)]
pub struct CreateDecisionRequest {
    pub task_id: Option<Uuid>,
    pub situation: String,
    pub options: Vec<DecisionOption>,
    pub evidence_refs: Vec<Value>,
}

// ──────────────────────────── State machine ────────────────────────────

/// Check whether a transition from one [`DecisionState`] to another is valid.
///
/// Rules:
/// - `Proposed → Recommended`, `Recommended → Decided`,
///   `Decided → Executed`, `Executed → Resolved` are valid forward steps.
/// - Any non-`Withdrawn` state may transition to `Withdrawn`.
/// - All other transitions (skips, reversals, re-withdrawal) are invalid.
fn is_valid_transition(from: DecisionState, to: DecisionState) -> bool {
    use DecisionState::*;
    match (from, to) {
        (Proposed, Recommended)
        | (Recommended, Decided)
        | (Decided, Executed)
        | (Executed, Resolved) => true,
        (_, Withdrawn) => from != Withdrawn,
        _ => false,
    }
}

// ──────────────────────────── Manager ────────────────────────────

/// Manages the lifecycle of decision records in the `decision_record` table.
pub struct DecisionManager {
    pool: Pool,
}

impl DecisionManager {
    /// Create a new manager backed by the given PostgreSQL pool.
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    /// Create a new decision record and return its generated `decision_id`.
    #[instrument(skip(self, req))]
    pub async fn create(&self, req: &CreateDecisionRequest) -> Result<Uuid, DecisionError> {
        let options_json = serde_json::to_value(&req.options)?;
        let evidence_json = serde_json::to_value(&req.evidence_refs)?;

        let row = sqlx::query(
            r#"INSERT INTO decision_record (task_id, situation, options, evidence_refs)
               VALUES ($1, $2, $3, $4)
               RETURNING decision_id"#,
        )
        .bind(req.task_id)
        .bind(&req.situation)
        .bind(options_json)
        .bind(evidence_json)
        .fetch_one(&self.pool)
        .await?;

        let decision_id: Uuid = row.try_get("decision_id")?;
        Ok(decision_id)
    }

    /// Get a decision record by ID.
    #[instrument(skip(self))]
    pub async fn get(&self, decision_id: Uuid) -> Result<Option<DecisionRecord>, DecisionError> {
        let row = sqlx::query(
            r#"SELECT decision_id, task_id, situation, options, recommendation,
                      human_decision, reason, action, outcome, evidence_refs,
                      decided_at, outcome_at, created_at
               FROM decision_record
               WHERE decision_id = $1"#,
        )
        .bind(decision_id)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|r| map_row_to_record(&r)).transpose()
    }

    /// List all decisions for a given task, newest first.
    #[instrument(skip(self))]
    pub async fn list_by_task(&self, task_id: Uuid) -> Result<Vec<DecisionRecord>, DecisionError> {
        let rows = sqlx::query(
            r#"SELECT decision_id, task_id, situation, options, recommendation,
                      human_decision, reason, action, outcome, evidence_refs,
                      decided_at, outcome_at, created_at
               FROM decision_record
               WHERE task_id = $1
               ORDER BY created_at DESC"#,
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(map_row_to_record).collect()
    }

    /// Add a recommendation to a decision in `Proposed` state.
    #[instrument(skip(self))]
    pub async fn set_recommendation(
        &self,
        decision_id: Uuid,
        recommendation: &str,
        reason: &str,
    ) -> Result<(), DecisionError> {
        let record = self
            .get(decision_id)
            .await?
            .ok_or(DecisionError::NotFound(decision_id))?;

        let current = Self::get_state(&record);
        if !is_valid_transition(current, DecisionState::Recommended) {
            return Err(DecisionError::InvalidTransition(format!(
                "cannot set recommendation from {current:?} state"
            )));
        }

        sqlx::query(
            r#"UPDATE decision_record
               SET recommendation = $2, reason = $3
               WHERE decision_id = $1"#,
        )
        .bind(decision_id)
        .bind(recommendation)
        .bind(reason)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Record a human decision on a decision in `Recommended` state.
    #[instrument(skip(self))]
    pub async fn record_human_decision(
        &self,
        decision_id: Uuid,
        decision: &str,
        reason: &str,
    ) -> Result<(), DecisionError> {
        let record = self
            .get(decision_id)
            .await?
            .ok_or(DecisionError::NotFound(decision_id))?;

        let current = Self::get_state(&record);
        if !is_valid_transition(current, DecisionState::Decided) {
            return Err(DecisionError::InvalidTransition(format!(
                "cannot record human decision from {current:?} state"
            )));
        }

        sqlx::query(
            r#"UPDATE decision_record
               SET human_decision = $2, reason = $3, decided_at = NOW()
               WHERE decision_id = $1"#,
        )
        .bind(decision_id)
        .bind(decision)
        .bind(reason)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Record an action taken on a decision in `Decided` state.
    #[instrument(skip(self))]
    pub async fn record_action(
        &self,
        decision_id: Uuid,
        action: &str,
    ) -> Result<(), DecisionError> {
        let record = self
            .get(decision_id)
            .await?
            .ok_or(DecisionError::NotFound(decision_id))?;

        let current = Self::get_state(&record);
        if !is_valid_transition(current, DecisionState::Executed) {
            return Err(DecisionError::InvalidTransition(format!(
                "cannot record action from {current:?} state"
            )));
        }

        sqlx::query(
            r#"UPDATE decision_record
               SET action = $2
               WHERE decision_id = $1"#,
        )
        .bind(decision_id)
        .bind(action)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Record the outcome of a decision in `Executed` state.
    #[instrument(skip(self, evidence))]
    pub async fn record_outcome(
        &self,
        decision_id: Uuid,
        outcome: &str,
        evidence: Vec<Value>,
    ) -> Result<(), DecisionError> {
        let record = self
            .get(decision_id)
            .await?
            .ok_or(DecisionError::NotFound(decision_id))?;

        let current = Self::get_state(&record);
        if !is_valid_transition(current, DecisionState::Resolved) {
            return Err(DecisionError::InvalidTransition(format!(
                "cannot record outcome from {current:?} state"
            )));
        }

        let evidence_json = serde_json::to_value(&evidence)?;

        sqlx::query(
            r#"UPDATE decision_record
               SET outcome = $2, evidence_refs = $3, outcome_at = NOW()
               WHERE decision_id = $1"#,
        )
        .bind(decision_id)
        .bind(outcome)
        .bind(evidence_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Withdraw or supersede a decision from any non-`Withdrawn` state.
    #[instrument(skip(self))]
    pub async fn withdraw(&self, decision_id: Uuid, reason: &str) -> Result<(), DecisionError> {
        let record = self
            .get(decision_id)
            .await?
            .ok_or(DecisionError::NotFound(decision_id))?;

        let current = Self::get_state(&record);
        if !is_valid_transition(current, DecisionState::Withdrawn) {
            return Err(DecisionError::InvalidTransition(format!(
                "cannot withdraw from {current:?} state"
            )));
        }

        sqlx::query(
            r#"UPDATE decision_record
               SET human_decision = 'withdrawn', reason = $2, decided_at = NOW()
               WHERE decision_id = $1"#,
        )
        .bind(decision_id)
        .bind(reason)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Derive the current [`DecisionState`] from a record's populated fields.
    ///
    /// `Withdrawn` is checked first (it overrides the normal lifecycle) and is
    /// encoded by `human_decision == "withdrawn"`. Otherwise the state is
    /// determined by the most advanced lifecycle field that has been set.
    pub fn get_state(record: &DecisionRecord) -> DecisionState {
        if record.human_decision.as_deref() == Some(WITHDRAWN_MARKER) {
            return DecisionState::Withdrawn;
        }
        if record.outcome.is_some() {
            return DecisionState::Resolved;
        }
        if record.action.is_some() {
            return DecisionState::Executed;
        }
        if record.human_decision.is_some() {
            return DecisionState::Decided;
        }
        if record.recommendation.is_some() {
            return DecisionState::Recommended;
        }
        DecisionState::Proposed
    }
}

// ──────────────────────────── Row mapping ────────────────────────────

/// Map a database row to a [`DecisionRecord`].
fn map_row_to_record(row: &sqlx::postgres::PgRow) -> Result<DecisionRecord, DecisionError> {
    let options_value: Value = row.try_get("options").unwrap_or_default();
    let options: Vec<DecisionOption> = serde_json::from_value(options_value).unwrap_or_default();

    let evidence_value: Value = row.try_get("evidence_refs").unwrap_or_default();
    let evidence_refs: Vec<Value> = evidence_value.as_array().cloned().unwrap_or_default();

    Ok(DecisionRecord {
        decision_id: row.try_get("decision_id")?,
        task_id: row.try_get("task_id")?,
        situation: row.try_get("situation")?,
        options,
        recommendation: row.try_get("recommendation")?,
        human_decision: row.try_get("human_decision")?,
        reason: row.try_get("reason")?,
        action: row.try_get("action")?,
        outcome: row.try_get("outcome")?,
        evidence_refs,
        decided_at: row.try_get("decided_at")?,
        outcome_at: row.try_get("outcome_at")?,
        created_at: row.try_get("created_at").unwrap_or_else(|_| Utc::now()),
    })
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a fresh decision record with no lifecycle fields populated.
    fn make_record() -> DecisionRecord {
        DecisionRecord {
            decision_id: Uuid::new_v4(),
            task_id: None,
            situation: "test situation".to_string(),
            options: vec![],
            recommendation: None,
            human_decision: None,
            reason: None,
            action: None,
            outcome: None,
            evidence_refs: vec![],
            decided_at: None,
            outcome_at: None,
            created_at: Utc::now(),
        }
    }

    // ── get_state tests ──

    #[test]
    fn get_state_proposed_for_empty_record() {
        let record = make_record();
        assert_eq!(DecisionManager::get_state(&record), DecisionState::Proposed);
    }

    #[test]
    fn get_state_recommended_when_recommendation_set() {
        let mut record = make_record();
        record.recommendation = Some("option A".into());
        assert_eq!(
            DecisionManager::get_state(&record),
            DecisionState::Recommended
        );
    }

    #[test]
    fn get_state_decided_when_human_decision_set() {
        let mut record = make_record();
        record.recommendation = Some("option A".into());
        record.human_decision = Some("agree".into());
        assert_eq!(DecisionManager::get_state(&record), DecisionState::Decided);
    }

    #[test]
    fn get_state_executed_when_action_set() {
        let mut record = make_record();
        record.recommendation = Some("option A".into());
        record.human_decision = Some("agree".into());
        record.action = Some("deployed".into());
        assert_eq!(DecisionManager::get_state(&record), DecisionState::Executed);
    }

    #[test]
    fn get_state_resolved_when_outcome_set() {
        let mut record = make_record();
        record.recommendation = Some("option A".into());
        record.human_decision = Some("agree".into());
        record.action = Some("deployed".into());
        record.outcome = Some("success".into());
        assert_eq!(DecisionManager::get_state(&record), DecisionState::Resolved);
    }

    #[test]
    fn get_state_withdrawn_when_marker_present() {
        let mut record = make_record();
        record.human_decision = Some(WITHDRAWN_MARKER.into());
        assert_eq!(
            DecisionManager::get_state(&record),
            DecisionState::Withdrawn
        );
    }

    #[test]
    fn get_state_withdrawn_overrides_resolved() {
        // Even when outcome is set, a withdrawn marker takes priority.
        let mut record = make_record();
        record.recommendation = Some("option A".into());
        record.human_decision = Some(WITHDRAWN_MARKER.into());
        record.reason = Some("superseded".into());
        record.action = Some("deployed".into());
        record.outcome = Some("success".into());
        assert_eq!(
            DecisionManager::get_state(&record),
            DecisionState::Withdrawn
        );
    }

    #[test]
    fn get_state_lifecycle_progression() {
        let mut record = make_record();

        assert_eq!(DecisionManager::get_state(&record), DecisionState::Proposed);

        record.recommendation = Some("opt-1".into());
        assert_eq!(
            DecisionManager::get_state(&record),
            DecisionState::Recommended
        );

        record.human_decision = Some("approve".into());
        assert_eq!(DecisionManager::get_state(&record), DecisionState::Decided);

        record.action = Some("executed".into());
        assert_eq!(DecisionManager::get_state(&record), DecisionState::Executed);

        record.outcome = Some("done".into());
        assert_eq!(DecisionManager::get_state(&record), DecisionState::Resolved);
    }

    // ── is_valid_transition tests ──

    #[test]
    fn transition_proposed_to_recommended_is_valid() {
        assert!(is_valid_transition(
            DecisionState::Proposed,
            DecisionState::Recommended
        ));
    }

    #[test]
    fn transition_recommended_to_decided_is_valid() {
        assert!(is_valid_transition(
            DecisionState::Recommended,
            DecisionState::Decided
        ));
    }

    #[test]
    fn transition_decided_to_executed_is_valid() {
        assert!(is_valid_transition(
            DecisionState::Decided,
            DecisionState::Executed
        ));
    }

    #[test]
    fn transition_executed_to_resolved_is_valid() {
        assert!(is_valid_transition(
            DecisionState::Executed,
            DecisionState::Resolved
        ));
    }

    #[test]
    fn transition_skipping_states_is_invalid() {
        assert!(!is_valid_transition(
            DecisionState::Proposed,
            DecisionState::Decided
        ));
        assert!(!is_valid_transition(
            DecisionState::Recommended,
            DecisionState::Executed
        ));
        assert!(!is_valid_transition(
            DecisionState::Decided,
            DecisionState::Resolved
        ));
        assert!(!is_valid_transition(
            DecisionState::Proposed,
            DecisionState::Resolved
        ));
    }

    #[test]
    fn transition_any_state_to_withdrawn_is_valid() {
        assert!(is_valid_transition(
            DecisionState::Proposed,
            DecisionState::Withdrawn
        ));
        assert!(is_valid_transition(
            DecisionState::Recommended,
            DecisionState::Withdrawn
        ));
        assert!(is_valid_transition(
            DecisionState::Decided,
            DecisionState::Withdrawn
        ));
        assert!(is_valid_transition(
            DecisionState::Executed,
            DecisionState::Withdrawn
        ));
        assert!(is_valid_transition(
            DecisionState::Resolved,
            DecisionState::Withdrawn
        ));
    }

    #[test]
    fn transition_withdrawn_to_withdrawn_is_invalid() {
        assert!(!is_valid_transition(
            DecisionState::Withdrawn,
            DecisionState::Withdrawn
        ));
    }

    #[test]
    fn transition_reverse_is_invalid() {
        assert!(!is_valid_transition(
            DecisionState::Resolved,
            DecisionState::Executed
        ));
        assert!(!is_valid_transition(
            DecisionState::Executed,
            DecisionState::Decided
        ));
        assert!(!is_valid_transition(
            DecisionState::Decided,
            DecisionState::Recommended
        ));
        assert!(!is_valid_transition(
            DecisionState::Recommended,
            DecisionState::Proposed
        ));
    }

    // ── Serialization tests ──

    #[test]
    fn risk_level_serializes_to_snake_case() {
        assert_eq!(serde_json::to_string(&RiskLevel::Low).unwrap(), "\"low\"");
        assert_eq!(
            serde_json::to_string(&RiskLevel::Medium).unwrap(),
            "\"medium\""
        );
        assert_eq!(serde_json::to_string(&RiskLevel::High).unwrap(), "\"high\"");
        assert_eq!(
            serde_json::to_string(&RiskLevel::Critical).unwrap(),
            "\"critical\""
        );
    }

    #[test]
    fn risk_level_deserializes_from_snake_case() {
        let low: RiskLevel = serde_json::from_str("\"low\"").unwrap();
        assert_eq!(low, RiskLevel::Low);

        let critical: RiskLevel = serde_json::from_str("\"critical\"").unwrap();
        assert_eq!(critical, RiskLevel::Critical);
    }

    #[test]
    fn decision_option_round_trips_through_json() {
        let option = DecisionOption {
            id: "opt-1".into(),
            description: "Do nothing".into(),
            pros: vec!["safe".into()],
            cons: vec!["no progress".into()],
            risk_level: RiskLevel::Medium,
            estimated_impact: Some("minimal".into()),
        };

        let json = serde_json::to_string(&option).unwrap();
        let deserialized: DecisionOption = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, option.id);
        assert_eq!(deserialized.description, option.description);
        assert_eq!(deserialized.pros, option.pros);
        assert_eq!(deserialized.cons, option.cons);
        assert_eq!(deserialized.risk_level, option.risk_level);
        assert_eq!(deserialized.estimated_impact, option.estimated_impact);
    }

    #[test]
    fn decision_option_with_null_estimated_impact_round_trips() {
        let option = DecisionOption {
            id: "opt-2".into(),
            description: "High risk bet".into(),
            pros: vec![],
            cons: vec!["risky".into()],
            risk_level: RiskLevel::Critical,
            estimated_impact: None,
        };

        let json = serde_json::to_string(&option).unwrap();
        let deserialized: DecisionOption = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, option.id);
        assert_eq!(deserialized.risk_level, RiskLevel::Critical);
        assert!(deserialized.estimated_impact.is_none());
    }
}
