//! Conflict resolution — detects and resolves conflicts when multiple sources
//! provide different values for the same fact.
//!
//! Implements the merge & conflict principles from architecture doc §5.3 (6
//! rules) and §5.4 (preference conflict priority), using the source trust
//! levels from §11.1 (L0–L3).
//!
//! This module is **stateless** — pure logic, no database or cache required.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use oris_memory_store::memory_types::{AuthorityLevel, Scope};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ──────────────────────── Core Types ──────────────────────────

/// A single fact value with its provenance metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactValue {
    /// Field name this value belongs to (e.g. `"role"`, `"timezone"`).
    pub field: String,
    /// The actual value.
    pub value: Value,
    /// Source trust level — L0 > L1 > L2 > L3.
    pub authority: AuthorityLevel,
    /// Provenance identifier (e.g. `"iam"`, `"hr"`, `"agent_inferred"`).
    pub source: String,
    /// Organisational scope of the fact.
    pub scope: Scope,
    /// When the value was observed / recorded.
    pub timestamp: DateTime<Utc>,
    /// Has this fact been verified by an external check?
    pub verified: bool,
    /// Explicitly confirmed by the user?
    pub user_confirmed: bool,
}

impl FactValue {
    /// Convenience builder helper for tests.
    #[cfg(test)]
    pub fn new(
        field: &str,
        value: Value,
        authority: AuthorityLevel,
        source: &str,
        scope: Scope,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            field: field.to_string(),
            value,
            authority,
            source: source.to_string(),
            scope,
            timestamp,
            verified: false,
            user_confirmed: false,
        }
    }

    /// Builder: mark as verified.
    #[cfg(test)]
    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Builder: mark as user-confirmed.
    #[cfg(test)]
    pub fn user_confirmed(mut self) -> Self {
        self.user_confirmed = true;
        self
    }
}

/// The result of resolving conflicts for a set of values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedFact {
    /// Field that was resolved.
    pub field: String,
    /// The winning value after resolution.
    pub winning_value: Value,
    /// Source of the winning value.
    pub winning_source: String,
    /// Authority level of the winner.
    pub winning_authority: AuthorityLevel,
    /// Whether a conflict was detected (multiple disagreeing values).
    pub conflict_detected: bool,
    /// Human-readable explanation of the resolution.
    pub conflict_reason: Option<String>,
    /// All contributing values (never silently dropped — §5.3 rule 1).
    pub all_values: Vec<FactValue>,
    /// Flags for downstream rerank (#18).
    pub conflict_flags: Vec<String>,
    /// Severity of the conflict.
    pub severity: ConflictSeverity,
}

/// Severity of a conflict for prioritization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ConflictSeverity {
    /// No conflict — single source or all agree.
    #[default]
    None,
    /// Different sources, same authority — resolved by recency or user
    /// confirmation.
    Low,
    /// Different authority levels — higher authority wins.
    Medium,
    /// Contradictory facts at same high authority — needs human review.
    High,
}

// ────────────────────── ConflictResolver ───────────────────────

/// Stateless conflict resolver.
///
/// Given a set of competing [`FactValue`]s for a single field, applies the
/// six resolution rules from §5.3 / §5.4 and returns a [`ResolvedFact`]
/// that records both the winner and every conflict detected.
#[derive(Debug, Default)]
pub struct ConflictResolver;

impl ConflictResolver {
    /// Create a new (stateless) resolver.
    pub fn new() -> Self {
        Self
    }

    /// Resolve conflicts for a single field with multiple competing values.
    pub fn resolve(&self, field: &str, values: Vec<FactValue>) -> ResolvedFact {
        // Rule 6 (§5.3 rule 1): all values are retained — never silently
        // dropped.  The losing values stay in `all_values` and are surfaced
        // via `conflict_flags`.
        let all_values = values.clone();

        // ── Empty input ───────────────────────────────────────────
        if values.is_empty() {
            return ResolvedFact {
                field: field.to_string(),
                winning_value: Value::Null,
                winning_source: String::new(),
                winning_authority: AuthorityLevel::L3Inferred,
                conflict_detected: false,
                conflict_reason: None,
                all_values: Vec::new(),
                conflict_flags: Vec::new(),
                severity: ConflictSeverity::None,
            };
        }

        // ── Check agreement ──────────────────────────────────────
        // If every value is equal, there is no conflict regardless of how
        // many sources contributed.
        let all_agree = values.windows(2).all(|w| w[0].value == w[1].value);

        if values.len() == 1 || all_agree {
            let winner = &values[0];
            return ResolvedFact {
                field: field.to_string(),
                winning_value: winner.value.clone(),
                winning_source: winner.source.clone(),
                winning_authority: winner.authority,
                conflict_detected: false,
                conflict_reason: None,
                all_values,
                conflict_flags: Vec::new(),
                severity: ConflictSeverity::None,
            };
        }

        // ── Conflict detected — resolve by priority ───────────────
        let has_authority_mismatch = !values.iter().all(|v| v.authority == values[0].authority);

        let has_scope_mismatch = !values.iter().all(|v| v.scope == values[0].scope);

        // Sort by resolution priority (stable, deterministic):
        //   1. authority.rank()      — Rule 1 (§5.3 rule 1, §11.1)
        //   2. user_confirmed        — Rule 2 (§5.4)
        //   3. verified              — Rule 5 (§5.3 rule 5)
        //   4. scope_rank (business) — Rule 4 (§5.3)
        //   5. timestamp (recency)   — Rule 3 (§5.3 rule 3)
        //   6. source_rank           — §5.4 tiebreaker
        let mut sorted = values;
        sorted.sort_by(|a, b| {
            // 1. Authority — higher rank wins.
            b.authority
                .rank()
                .cmp(&a.authority.rank())
                // 2. User confirmation — confirmed beats non-confirmed.
                .then(b.user_confirmed.cmp(&a.user_confirmed))
                // 3. Verified beats unverified.
                .then(b.verified.cmp(&a.verified))
                // 4. Scope priority (business facts: Enterprise > … > Personal).
                .then(scope_rank_business(b.scope).cmp(&scope_rank_business(a.scope)))
                // 5. Recency — most recent wins (tiebreaker only when
                //    same authority & scope, but safe as a final sort key).
                .then(b.timestamp.cmp(&a.timestamp))
                // 6. Source tiebreaker (§5.4 preference priority).
                .then(source_rank(&b.source).cmp(&source_rank(&a.source)))
        });

        let winner = sorted[0].clone();

        // ── Severity computation ──────────────────────────────────
        let severity = if has_authority_mismatch {
            ConflictSeverity::Medium
        } else {
            // Same authority across all values.
            let auth = winner.authority;
            match auth {
                AuthorityLevel::L0SourceOfTruth | AuthorityLevel::L1Authoritative => {
                    ConflictSeverity::High
                }
                _ => ConflictSeverity::Low,
            }
        };

        // ── Conflict flags ────────────────────────────────────────
        let mut flags = Vec::new();
        if has_authority_mismatch {
            flags.push(format!("conflict:{field}:authority_mismatch"));
        }
        if has_scope_mismatch {
            flags.push(format!("conflict:{field}:scope_mismatch"));
        }
        // Recency tiebreaker flag: the winner was chosen by recency when
        // authority and scope were all equal.
        if !has_authority_mismatch && !has_scope_mismatch {
            // If all values share authority and scope but disagree, the
            // recency sort key is what decided the winner.
            let any_different_ts = sorted.windows(2).any(|w| w[0].timestamp != w[1].timestamp);
            if any_different_ts {
                flags.push(format!("conflict:{field}:recency_tiebreaker"));
            }
        }
        if severity == ConflictSeverity::High {
            flags.push(format!("conflict:{field}:needs_review"));
        }

        // ── Human-readable reason ─────────────────────────────────
        let reason = build_reason(
            field,
            &winner,
            has_authority_mismatch,
            has_scope_mismatch,
            severity,
        );

        debug_resolved(field, &winner, &flags, severity);

        ResolvedFact {
            field: field.to_string(),
            winning_value: winner.value.clone(),
            winning_source: winner.source.clone(),
            winning_authority: winner.authority,
            conflict_detected: true,
            conflict_reason: Some(reason),
            all_values,
            conflict_flags: flags,
            severity,
        }
    }

    /// Batch resolve multiple fields.
    pub fn resolve_batch(&self, facts: HashMap<String, Vec<FactValue>>) -> Vec<ResolvedFact> {
        // Sort keys for deterministic output order.
        let mut keys: Vec<String> = facts.keys().cloned().collect();
        keys.sort();
        keys.into_iter()
            .map(|k| {
                let vals = facts.get(&k).cloned().unwrap_or_default();
                self.resolve(&k, vals)
            })
            .collect()
    }
}

// ────────────────────── Helper Functions ───────────────────────

/// Scope priority for business facts (§5.3 rule 4):
/// `Enterprise > Factory > Process > Team > Task > Agent > Personal`.
///
/// For personal preferences the order is inverted (Personal wins), but that
/// requires caller-side knowledge of the field type; this module applies the
/// business-fact ranking by default.
fn scope_rank_business(scope: Scope) -> u8 {
    match scope {
        Scope::Enterprise => 6,
        Scope::Factory => 5,
        Scope::Process => 4,
        Scope::Team => 3,
        Scope::Task => 2,
        Scope::Agent => 1,
        Scope::Personal => 0,
    }
}

/// Source preference priority from §5.4:
/// `UserExplicit > ExplicitAgent > ExplicitDomain > ExplicitGlobal >
/// RepeatedInference > SingleInference`.
///
/// Used as a final tiebreaker when authority, confirmation, verification,
/// scope, and recency are all equal.
fn source_rank(source: &str) -> u8 {
    match source {
        "user_explicit" | "explicit_user" => 5,
        "agent_explicit" | "explicit_agent" => 4,
        "domain_explicit" | "explicit_domain" => 3,
        "global_explicit" | "explicit_global" => 2,
        "repeated_inference" => 1,
        _ => 0, // single_inference and unknown
    }
}

/// Build a human-readable conflict-resolution reason string.
fn build_reason(
    field: &str,
    winner: &FactValue,
    has_authority_mismatch: bool,
    has_scope_mismatch: bool,
    severity: ConflictSeverity,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    if has_authority_mismatch {
        parts.push(format!(
            "authority '{}' ({}) beats lower-authority sources",
            winner.authority.as_str(),
            winner.source
        ));
    }

    if winner.user_confirmed {
        parts.push("winner is user-confirmed".to_string());
    }

    if winner.verified {
        parts.push("winner is verified".to_string());
    }

    if !has_authority_mismatch && has_scope_mismatch {
        parts.push(format!(
            "scope '{}' has higher priority at equal authority",
            winner.scope.as_str()
        ));
    }

    if !has_authority_mismatch && !has_scope_mismatch {
        parts.push("resolved by recency (most recent wins)".to_string());
    }

    match severity {
        ConflictSeverity::High => {
            parts.push("NEEDS HUMAN REVIEW — contradictory facts at high authority".into());
        }
        ConflictSeverity::Medium => {}
        ConflictSeverity::Low => {}
        ConflictSeverity::None => {}
    }

    format!("field '{field}': {}", parts.join("; "))
}

#[inline]
fn debug_resolved(field: &str, winner: &FactValue, flags: &[String], severity: ConflictSeverity) {
    tracing::debug!(
        field = %field,
        winner_source = %winner.source,
        winner_authority = %winner.authority.as_str(),
        severity = ?severity,
        flags_count = flags.len(),
        "conflict resolved"
    );
}

// ─────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(seconds: i64) -> DateTime<Utc> {
        let base = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        base + chrono::Duration::seconds(seconds)
    }

    // ── Test 1: Single value — no conflict ─────────────────────
    #[test]
    fn single_value_no_conflict() {
        let resolver = ConflictResolver::new();
        let val = FactValue::new(
            "role",
            Value::String("engineer".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(100),
        );

        let result = resolver.resolve("role", vec![val]);

        assert!(!result.conflict_detected);
        assert_eq!(result.severity, ConflictSeverity::None);
        assert_eq!(result.winning_source, "iam");
        assert_eq!(result.winning_value, Value::String("engineer".into()));
        assert!(result.conflict_flags.is_empty());
        assert!(result.conflict_reason.is_none());
    }

    // ── Test 2: Multiple agreeing values — no conflict ─────────
    #[test]
    fn multiple_agreeing_values_no_conflict() {
        let resolver = ConflictResolver::new();
        let v1 = FactValue::new(
            "timezone",
            Value::String("UTC".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(100),
        );
        let v2 = FactValue::new(
            "timezone",
            Value::String("UTC".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(200),
        );

        let result = resolver.resolve("timezone", vec![v1, v2]);

        assert!(!result.conflict_detected);
        assert_eq!(result.severity, ConflictSeverity::None);
        assert!(result.conflict_flags.is_empty());
    }

    // ── Test 3: L0 vs L3 — L0 wins, Medium severity ────────────
    #[test]
    fn l0_vs_l3_l0_wins() {
        let resolver = ConflictResolver::new();
        let l3 = FactValue::new(
            "role",
            Value::String("engineer".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Personal,
            ts(200),
        );
        let l0 = FactValue::new(
            "role",
            Value::String("director".into()),
            AuthorityLevel::L0SourceOfTruth,
            "iam",
            Scope::Enterprise,
            ts(100),
        );

        let result = resolver.resolve("role", vec![l3, l0]);

        assert!(result.conflict_detected);
        assert_eq!(result.severity, ConflictSeverity::Medium);
        assert_eq!(result.winning_value, Value::String("director".into()));
        assert_eq!(result.winning_source, "iam");
        assert_eq!(result.winning_authority, AuthorityLevel::L0SourceOfTruth);
        assert!(result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:role:authority_mismatch"));
    }

    // ── Test 4: L1 vs L2 — L1 wins, Medium severity ────────────
    #[test]
    fn l1_vs_l2_l1_wins() {
        let resolver = ConflictResolver::new();
        let l2 = FactValue::new(
            "department",
            Value::String("sales".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(200),
        );
        let l1 = FactValue::new(
            "department",
            Value::String("engineering".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(100),
        );

        let result = resolver.resolve("department", vec![l2, l1]);

        assert!(result.conflict_detected);
        assert_eq!(result.severity, ConflictSeverity::Medium);
        assert_eq!(result.winning_value, Value::String("engineering".into()));
        assert_eq!(result.winning_authority, AuthorityLevel::L1Authoritative);
    }

    // ── Test 5: Same authority, different recency — recent wins ─
    #[test]
    fn same_authority_recency_wins() {
        let resolver = ConflictResolver::new();
        let old = FactValue::new(
            "timezone",
            Value::String("UTC".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        );
        let recent = FactValue::new(
            "timezone",
            Value::String("Asia/Shanghai".into()),
            AuthorityLevel::L2Verified,
            "iam",
            Scope::Team,
            ts(500),
        );

        let result = resolver.resolve("timezone", vec![old, recent]);

        assert!(result.conflict_detected);
        assert_eq!(result.severity, ConflictSeverity::Low);
        assert_eq!(result.winning_value, Value::String("Asia/Shanghai".into()));
        assert_eq!(result.winning_source, "iam");
        assert!(result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:timezone:recency_tiebreaker"));
    }

    // ── Test 6: Same authority, same recency, different scope ───
    #[test]
    fn same_authority_scope_rule() {
        let resolver = ConflictResolver::new();
        let team_scope = FactValue::new(
            "region",
            Value::String("east".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        );
        let enterprise_scope = FactValue::new(
            "region",
            Value::String("global".into()),
            AuthorityLevel::L2Verified,
            "iam",
            Scope::Enterprise,
            ts(100),
        );

        let result = resolver.resolve("region", vec![team_scope, enterprise_scope]);

        assert!(result.conflict_detected);
        assert_eq!(result.winning_value, Value::String("global".into()));
        assert!(result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:region:scope_mismatch"));
    }

    // ── Test 7: User-confirmed beats non-confirmed ─────────────
    #[test]
    fn user_confirmed_beats_non_confirmed() {
        let resolver = ConflictResolver::new();
        let unconfirmed = FactValue::new(
            "language",
            Value::String("en".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        );
        let confirmed = FactValue::new(
            "language",
            Value::String("zh".into()),
            AuthorityLevel::L2Verified,
            "user_explicit",
            Scope::Personal,
            ts(50),
        )
        .user_confirmed();

        let result = resolver.resolve("language", vec![unconfirmed, confirmed]);

        assert!(result.conflict_detected);
        assert_eq!(result.winning_value, Value::String("zh".into()));
        assert_eq!(result.winning_source, "user_explicit");
    }

    // ── Test 8: Verified beats unverified ───────────────────────
    #[test]
    fn verified_beats_unverified() {
        let resolver = ConflictResolver::new();
        let unverified = FactValue::new(
            "role",
            Value::String("intern".into()),
            AuthorityLevel::L2Verified,
            "agent_inferred",
            Scope::Team,
            ts(200),
        );
        let verified = FactValue::new(
            "role",
            Value::String("lead".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        )
        .verified();

        let result = resolver.resolve("role", vec![unverified, verified]);

        assert!(result.conflict_detected);
        assert_eq!(result.winning_value, Value::String("lead".into()));
        assert_eq!(result.winning_source, "hr");
    }

    // ── Test 9: Two L0 contradicting — High severity, review ──
    #[test]
    fn two_l0_contradicting_high_severity() {
        let resolver = ConflictResolver::new();
        let a = FactValue::new(
            "role",
            Value::String("director".into()),
            AuthorityLevel::L0SourceOfTruth,
            "iam",
            Scope::Enterprise,
            ts(100),
        );
        let b = FactValue::new(
            "role",
            Value::String("vp".into()),
            AuthorityLevel::L0SourceOfTruth,
            "hr",
            Scope::Enterprise,
            ts(200),
        );

        let result = resolver.resolve("role", vec![a, b]);

        assert!(result.conflict_detected);
        assert_eq!(result.severity, ConflictSeverity::High);
        assert!(result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:role:needs_review"));
        assert!(result
            .conflict_reason
            .as_ref()
            .unwrap()
            .contains("NEEDS HUMAN REVIEW"));
    }

    // ── Test 10: Conflict flags generated correctly ─────────────
    #[test]
    fn conflict_flags_generated_correctly() {
        let resolver = ConflictResolver::new();
        let v1 = FactValue::new(
            "role",
            Value::String("a".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Personal,
            ts(100),
        );
        let v2 = FactValue::new(
            "role",
            Value::String("b".into()),
            AuthorityLevel::L0SourceOfTruth,
            "iam",
            Scope::Enterprise,
            ts(200),
        );

        let result = resolver.resolve("role", vec![v1, v2]);

        // Should have authority_mismatch and scope_mismatch flags.
        assert!(result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:role:authority_mismatch"));
        assert!(result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:role:scope_mismatch"));
        // No needs_review because different authority = Medium, not High.
        assert!(!result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:role:needs_review"));
    }

    // ── Test 11: Batch resolution with multiple fields ─────────
    #[test]
    fn batch_resolution_multiple_fields() {
        let resolver = ConflictResolver::new();
        let mut facts = HashMap::new();
        facts.insert(
            "role".to_string(),
            vec![
                FactValue::new(
                    "role",
                    Value::String("engineer".into()),
                    AuthorityLevel::L3Inferred,
                    "agent_inferred",
                    Scope::Personal,
                    ts(100),
                ),
                FactValue::new(
                    "role",
                    Value::String("director".into()),
                    AuthorityLevel::L0SourceOfTruth,
                    "iam",
                    Scope::Enterprise,
                    ts(200),
                ),
            ],
        );
        facts.insert(
            "timezone".to_string(),
            vec![FactValue::new(
                "timezone",
                Value::String("UTC".into()),
                AuthorityLevel::L1Authoritative,
                "iam",
                Scope::Enterprise,
                ts(100),
            )],
        );

        let results = resolver.resolve_batch(facts);

        assert_eq!(results.len(), 2);
        let role_result = results.iter().find(|r| r.field == "role").unwrap();
        let tz_result = results.iter().find(|r| r.field == "timezone").unwrap();
        assert!(role_result.conflict_detected);
        assert!(!tz_result.conflict_detected);
    }

    // ── Test 12: Empty values list — returns empty ResolvedFact ─
    #[test]
    fn empty_values_returns_empty_resolved_fact() {
        let resolver = ConflictResolver::new();
        let result = resolver.resolve("role", vec![]);

        assert!(!result.conflict_detected);
        assert_eq!(result.severity, ConflictSeverity::None);
        assert_eq!(result.winning_value, Value::Null);
        assert!(result.winning_source.is_empty());
        assert_eq!(result.all_values.len(), 0);
        assert!(result.conflict_flags.is_empty());
    }

    // ── Test 13: Rule 1 — Authority always wins over recency ───
    #[test]
    fn rule1_authority_beats_recency() {
        let resolver = ConflictResolver::new();
        let recent_low = FactValue::new(
            "role",
            Value::String("engineer".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Personal,
            ts(999),
        );
        let old_high = FactValue::new(
            "role",
            Value::String("director".into()),
            AuthorityLevel::L0SourceOfTruth,
            "iam",
            Scope::Enterprise,
            ts(1),
        );

        let result = resolver.resolve("role", vec![recent_low, old_high]);

        assert_eq!(result.winning_value, Value::String("director".into()));
        assert_eq!(result.winning_authority, AuthorityLevel::L0SourceOfTruth);
    }

    // ── Test 13b: Rule 2 — User confirmation priority ──────────
    #[test]
    fn rule2_user_confirmation_priority() {
        let resolver = ConflictResolver::new();
        let not_confirmed = FactValue::new(
            "theme",
            Value::String("light".into()),
            AuthorityLevel::L2Verified,
            "agent_inferred",
            Scope::Personal,
            ts(200),
        );
        let confirmed = FactValue::new(
            "theme",
            Value::String("dark".into()),
            AuthorityLevel::L2Verified,
            "user_explicit",
            Scope::Personal,
            ts(100),
        )
        .user_confirmed();

        let result = resolver.resolve("theme", vec![not_confirmed, confirmed]);

        assert_eq!(result.winning_value, Value::String("dark".into()));
    }

    // ── Test 13c: Rule 3 — Recency only at same auth & scope ────
    #[test]
    fn rule3_recency_tiebreaker_conditions() {
        let resolver = ConflictResolver::new();
        // Same authority, same scope, different timestamps → recency decides.
        let old = FactValue::new(
            "status",
            Value::String("active".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Task,
            ts(10),
        );
        let recent = FactValue::new(
            "status",
            Value::String("done".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Task,
            ts(50),
        );

        let result = resolver.resolve("status", vec![old, recent]);

        assert_eq!(result.winning_value, Value::String("done".into()));
        assert_eq!(result.severity, ConflictSeverity::Low);
    }

    // ── Test 13d: Rule 4 — Scope priority for business facts ───
    #[test]
    fn rule4_scope_priority_business_facts() {
        let resolver = ConflictResolver::new();
        let personal = FactValue::new(
            "cost_center",
            Value::String("P-001".into()),
            AuthorityLevel::L1Authoritative,
            "hr",
            Scope::Personal,
            ts(200),
        );
        let enterprise = FactValue::new(
            "cost_center",
            Value::String("E-001".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(100),
        );
        let team = FactValue::new(
            "cost_center",
            Value::String("T-001".into()),
            AuthorityLevel::L1Authoritative,
            "tool_result",
            Scope::Team,
            ts(150),
        );

        let result = resolver.resolve("cost_center", vec![personal, enterprise, team]);

        assert_eq!(result.winning_value, Value::String("E-001".into()));
        assert_eq!(result.winning_authority, AuthorityLevel::L1Authoritative);
    }

    // ── Test 13e: Rule 5 — Verified beats unverified ───────────
    #[test]
    fn rule5_verified_beats_unverified() {
        let resolver = ConflictResolver::new();
        let unverified = FactValue::new(
            "manager",
            Value::String("alice".into()),
            AuthorityLevel::L2Verified,
            "agent_inferred",
            Scope::Team,
            ts(200),
        );
        let verified = FactValue::new(
            "manager",
            Value::String("bob".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        )
        .verified();

        let result = resolver.resolve("manager", vec![unverified, verified]);

        assert_eq!(result.winning_value, Value::String("bob".into()));
    }

    // ── Test 13f: Rule 6 — Never silently overwrite ────────────
    #[test]
    fn rule6_never_silently_overwrite() {
        let resolver = ConflictResolver::new();
        let loser = FactValue::new(
            "role",
            Value::String("engineer".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Personal,
            ts(100),
        );
        let winner = FactValue::new(
            "role",
            Value::String("director".into()),
            AuthorityLevel::L0SourceOfTruth,
            "iam",
            Scope::Enterprise,
            ts(200),
        );

        let result = resolver.resolve("role", vec![loser.clone(), winner.clone()]);

        // The losing value is retained in all_values.
        assert_eq!(result.all_values.len(), 2);
        assert!(result
            .all_values
            .iter()
            .any(|v| v.value == Value::String("engineer".into())));
        assert!(result
            .all_values
            .iter()
            .any(|v| v.value == Value::String("director".into())));
        assert!(!result.conflict_flags.is_empty());
    }

    // ── Test 14: Scope priority for business facts (full ladder) ─
    #[test]
    fn scope_priority_enterprise_over_factory_over_team_over_personal() {
        let resolver = ConflictResolver::new();
        let personal = FactValue::new(
            "policy",
            Value::String("personal".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Personal,
            ts(100),
        );
        let team = FactValue::new(
            "policy",
            Value::String("team".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        );
        let factory = FactValue::new(
            "policy",
            Value::String("factory".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Factory,
            ts(100),
        );
        let enterprise = FactValue::new(
            "policy",
            Value::String("enterprise".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Enterprise,
            ts(100),
        );

        let result = resolver.resolve("policy", vec![personal, team, factory, enterprise]);

        assert_eq!(result.winning_value, Value::String("enterprise".into()));
    }

    // ── Test 15: ConflictSeverity computation for all levels ───
    #[test]
    fn severity_none_for_single_value() {
        let resolver = ConflictResolver::new();
        let val = FactValue::new(
            "x",
            Value::String("v".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(100),
        );
        let result = resolver.resolve("x", vec![val]);
        assert_eq!(result.severity, ConflictSeverity::None);
    }

    #[test]
    fn severity_low_for_same_authority_l2() {
        let resolver = ConflictResolver::new();
        let v1 = FactValue::new(
            "x",
            Value::String("a".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        );
        let v2 = FactValue::new(
            "x",
            Value::String("b".into()),
            AuthorityLevel::L2Verified,
            "iam",
            Scope::Team,
            ts(200),
        );
        let result = resolver.resolve("x", vec![v1, v2]);
        assert_eq!(result.severity, ConflictSeverity::Low);
    }

    #[test]
    fn severity_medium_for_different_authority() {
        let resolver = ConflictResolver::new();
        let v1 = FactValue::new(
            "x",
            Value::String("a".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Personal,
            ts(100),
        );
        let v2 = FactValue::new(
            "x",
            Value::String("b".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(200),
        );
        let result = resolver.resolve("x", vec![v1, v2]);
        assert_eq!(result.severity, ConflictSeverity::Medium);
    }

    #[test]
    fn severity_high_for_two_l1_contradicting() {
        let resolver = ConflictResolver::new();
        let v1 = FactValue::new(
            "x",
            Value::String("a".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(100),
        );
        let v2 = FactValue::new(
            "x",
            Value::String("b".into()),
            AuthorityLevel::L1Authoritative,
            "hr",
            Scope::Enterprise,
            ts(200),
        );
        let result = resolver.resolve("x", vec![v1, v2]);
        assert_eq!(result.severity, ConflictSeverity::High);
    }

    // ── Test 16: all_values retains every input ────────────────
    #[test]
    fn all_values_retains_every_input() {
        let resolver = ConflictResolver::new();
        let vals: Vec<FactValue> = (0..5)
            .map(|i| {
                FactValue::new(
                    "f",
                    Value::String(format!("v{i}")),
                    AuthorityLevel::L3Inferred,
                    "agent_inferred",
                    Scope::Personal,
                    ts(i),
                )
            })
            .collect();

        let result = resolver.resolve("f", vals.clone());

        assert_eq!(result.all_values.len(), 5);
    }

    // ── Test 17: conflict_reason is populated on conflict ──────
    #[test]
    fn conflict_reason_populated_on_conflict() {
        let resolver = ConflictResolver::new();
        let v1 = FactValue::new(
            "role",
            Value::String("a".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Personal,
            ts(100),
        );
        let v2 = FactValue::new(
            "role",
            Value::String("b".into()),
            AuthorityLevel::L0SourceOfTruth,
            "iam",
            Scope::Enterprise,
            ts(200),
        );

        let result = resolver.resolve("role", vec![v1, v2]);

        let reason = result.conflict_reason.expect("reason should be set");
        assert!(reason.contains("field 'role'"));
        assert!(reason.contains("authority"));
    }

    // ── Test 18: user_confirmed overrides verified at same auth ─
    #[test]
    fn user_confirmed_overrides_verified() {
        let resolver = ConflictResolver::new();
        // Both L2, same scope, same timestamp. One is verified but not
        // confirmed; the other is confirmed but not verified. Confirmation
        // has higher priority than verification.
        let verified_only = FactValue::new(
            "lang",
            Value::String("en".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(100),
        )
        .verified();
        let confirmed_only = FactValue::new(
            "lang",
            Value::String("zh".into()),
            AuthorityLevel::L2Verified,
            "user_explicit",
            Scope::Team,
            ts(100),
        )
        .user_confirmed();

        let result = resolver.resolve("lang", vec![verified_only, confirmed_only]);

        assert_eq!(result.winning_value, Value::String("zh".into()));
    }

    // ── Test 19: resolve_batch with empty map ──────────────────
    #[test]
    fn resolve_batch_empty_map() {
        let resolver = ConflictResolver::new();
        let results = resolver.resolve_batch(HashMap::new());
        assert!(results.is_empty());
    }

    // ── Test 20: Default trait works ────────────────────────────
    #[test]
    fn default_creates_resolver() {
        let resolver = ConflictResolver::default();
        let val = FactValue::new(
            "x",
            Value::String("v".into()),
            AuthorityLevel::L1Authoritative,
            "iam",
            Scope::Enterprise,
            ts(100),
        );
        let result = resolver.resolve("x", vec![val]);
        assert!(!result.conflict_detected);
    }

    // ── Test 21: Three+ values with mixed authority ────────────
    #[test]
    fn three_values_mixed_authority() {
        let resolver = ConflictResolver::new();
        let l3 = FactValue::new(
            "role",
            Value::String("intern".into()),
            AuthorityLevel::L3Inferred,
            "agent_inferred",
            Scope::Personal,
            ts(300),
        );
        let l2 = FactValue::new(
            "role",
            Value::String("engineer".into()),
            AuthorityLevel::L2Verified,
            "hr",
            Scope::Team,
            ts(200),
        );
        let l0 = FactValue::new(
            "role",
            Value::String("director".into()),
            AuthorityLevel::L0SourceOfTruth,
            "iam",
            Scope::Enterprise,
            ts(100),
        );

        let result = resolver.resolve("role", vec![l3, l2, l0]);

        assert_eq!(result.winning_value, Value::String("director".into()));
        assert_eq!(result.severity, ConflictSeverity::Medium);
        assert_eq!(result.all_values.len(), 3);
    }

    // ── Test 22: recency tiebreaker only flag when same auth+scope
    #[test]
    fn recency_flag_only_when_same_auth_and_scope() {
        let resolver = ConflictResolver::new();
        // Different authority and scope → no recency flag.
        let v1 = FactValue::new(
            "x",
            Value::String("a".into()),
            AuthorityLevel::L3Inferred,
            "s1",
            Scope::Personal,
            ts(100),
        );
        let v2 = FactValue::new(
            "x",
            Value::String("b".into()),
            AuthorityLevel::L0SourceOfTruth,
            "s2",
            Scope::Enterprise,
            ts(200),
        );
        let result = resolver.resolve("x", vec![v1, v2]);
        assert!(!result
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:x:recency_tiebreaker"));

        // Same authority and scope, different timestamps → recency flag.
        let v3 = FactValue::new(
            "x",
            Value::String("a".into()),
            AuthorityLevel::L2Verified,
            "s1",
            Scope::Team,
            ts(100),
        );
        let v4 = FactValue::new(
            "x",
            Value::String("b".into()),
            AuthorityLevel::L2Verified,
            "s2",
            Scope::Team,
            ts(200),
        );
        let result2 = resolver.resolve("x", vec![v3, v4]);
        assert!(result2
            .conflict_flags
            .iter()
            .any(|f| f == "conflict:x:recency_tiebreaker"));
    }
}
