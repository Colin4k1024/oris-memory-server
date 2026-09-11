//! Hybrid retrieval rerank pipeline.
//!
//! Implements §9.2 of the architecture document: after the retrieval stage
//! (Metadata + Keyword + Vector + Time + Graph) produces a set of candidate
//! memories, the [`RerankPipeline`] re-scores and ranks them using a
//! multi-factor formula:
//!
//! ```text
//! final_score = (relevance × w_rel
//!              + authority × w_auth
//!              + freshness × w_fresh
//!              + outcome  × w_out)
//!              × (1.0 − conflict_penalty)
//! ```
//!
//! The pipeline is **stateless** — pure computation, no database or cache
//! required.  Permission filtering runs first (§9.2 step 1), then scoring.
//! The permission filter **never fails open**: when in doubt, deny.

use chrono::{DateTime, Duration, Utc};
use oris_memory_store::memory_types::{AuthorityLevel, PrivacyClass, Scope};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ──────────────────────────── Config ──────────────────────────

/// Configuration for the rerank pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Weight for relevance score (0.0–1.0).
    pub relevance_weight: f64,
    /// Weight for authority level (0.0–1.0).
    pub authority_weight: f64,
    /// Weight for freshness/recency (0.0–1.0).
    pub freshness_weight: f64,
    /// Weight for outcome value — success rate / confidence (0.0–1.0).
    pub outcome_weight: f64,
    /// Maximum results to return after reranking.
    pub top_k: usize,
    /// Whether to downweight conflicting memories.
    pub downweight_conflicts: bool,
    /// Half-life for the freshness decay function, in days.
    pub freshness_half_life_days: f64,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            relevance_weight: 0.4,
            authority_weight: 0.3,
            freshness_weight: 0.15,
            outcome_weight: 0.15,
            top_k: 10,
            downweight_conflicts: true,
            freshness_half_life_days: 30.0,
        }
    }
}

// ──────────────────────── Core Data Types ─────────────────────

/// A candidate memory item with all scoring signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankCandidate {
    pub memory_id: Uuid,
    pub content: String,
    /// Relevance from vector / keyword search (0.0–1.0).
    pub relevance_score: f64,
    pub authority_level: AuthorityLevel,
    pub timestamp: DateTime<Utc>,
    /// Confidence from gene provenance (0.0–1.0).
    pub confidence: f64,
    /// verified_successes / (verified_successes + verified_failures).
    pub success_rate: f64,
    /// Conflict flags from [`ConflictResolver`](crate::conflict_resolver) (#28).
    pub conflict_flags: Vec<String>,
    pub privacy_class: PrivacyClass,
    pub scope: Scope,
}

/// The final reranked result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankedResult {
    pub memory_id: Uuid,
    pub content: String,
    pub final_score: f64,
    pub relevance_score: f64,
    pub authority_score: f64,
    pub freshness_score: f64,
    pub outcome_score: f64,
    pub conflict_penalty: f64,
    pub rank: usize,
}

// ─────────────────────────── Pipeline ─────────────────────────

/// The hybrid retrieval rerank pipeline.
///
/// Implements §9.2: Relevance × Authority × Freshness × Outcome Value,
/// with conflict penalty applied.
#[derive(Debug, Clone)]
pub struct RerankPipeline {
    config: RerankConfig,
}

impl RerankPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: RerankConfig) -> Self {
        Self { config }
    }

    /// Rerank candidates using the formula:
    /// `Relevance × Authority × Freshness × Outcome` with conflict penalty.
    ///
    /// Steps (§9.2, post-permission):
    /// 1. Score each candidate on four dimensions.
    /// 2. Apply conflict penalty.
    /// 3. Sort by descending final score (stable — ties keep original order).
    /// 4. Truncate to `top_k` and assign ranks.
    pub fn rerank(&self, candidates: Vec<RerankCandidate>) -> Vec<RerankedResult> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // ── Score + penalty ───────────────────────────────────────
        let mut scored: Vec<RerankedResult> = candidates
            .into_iter()
            .map(|c| {
                let authority_score = self.authority_score(c.authority_level);
                let freshness_score = self.freshness_score(
                    c.timestamp,
                    Utc::now(),
                    self.config.freshness_half_life_days,
                );
                let outcome_score = self.outcome_score(c.success_rate, c.confidence);
                let conflict_penalty = if self.config.downweight_conflicts {
                    self.conflict_penalty(&c.conflict_flags)
                } else {
                    0.0
                };

                let weighted = c.relevance_score * self.config.relevance_weight
                    + authority_score * self.config.authority_weight
                    + freshness_score * self.config.freshness_weight
                    + outcome_score * self.config.outcome_weight;

                let final_score = weighted * (1.0 - conflict_penalty);

                RerankedResult {
                    memory_id: c.memory_id,
                    content: c.content,
                    final_score,
                    relevance_score: c.relevance_score,
                    authority_score,
                    freshness_score,
                    outcome_score,
                    conflict_penalty,
                    rank: 0,
                }
            })
            .collect();

        // ── Sort descending by final_score (stable) ───────────────
        scored.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── Truncate + assign ranks (1-based) ────────────────────
        let top_k = self.config.top_k.min(scored.len());
        scored.truncate(top_k);
        for (i, result) in scored.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        scored
    }

    /// Filter out candidates the user doesn't have permission to see.
    ///
    /// Permission filter runs FIRST (§9.2 step 1).  **Never fails open**:
    /// if unsure, deny.
    pub fn permission_filter(
        &self,
        candidates: Vec<RerankCandidate>,
        allowed_scopes: &[Scope],
    ) -> Vec<RerankCandidate> {
        candidates
            .into_iter()
            .filter(|c| {
                // 1. Scope check — candidate scope must be in the allowed set.
                if !allowed_scopes.contains(&c.scope) {
                    return false;
                }
                // 2. Restricted privacy requires explicit permission.
                //    We treat the `allowed_scopes` list as the *only* grant
                //    surface.  There is no separate "has_restricted" flag, so
                //    a Restricted candidate is denied unless the caller
                //    explicitly includes `Scope::Personal` (owner self-read)
                //    or a broader enterprise scope that implies explicit
                //    grant.  To keep the contract simple and fail-safe we
                //    only allow Restricted when the candidate's own scope is
                //    present AND it is a self-owned personal scope — the
                //    narrowest safe allow.
                if c.privacy_class == PrivacyClass::Restricted {
                    // Never fail open: if we cannot prove explicit permission,
                    // deny.  Without an explicit grant we deny Restricted
                    // content.
                    return false;
                }
                true
            })
            .collect()
    }

    // ──────────────────── Scoring helpers ──────────────────────

    /// Authority score per §11.1:
    /// `L0 → 1.0`, `L1 → 0.8`, `L2 → 0.6`, `L3 → 0.4`.
    fn authority_score(&self, level: AuthorityLevel) -> f64 {
        match level {
            AuthorityLevel::L0SourceOfTruth => 1.0,
            AuthorityLevel::L1Authoritative => 0.8,
            AuthorityLevel::L2Verified => 0.6,
            AuthorityLevel::L3Inferred => 0.4,
        }
    }

    /// Freshness score — exponential decay: `2^(-age_days / half_life_days)`.
    ///
    /// Newer memories score higher.  Capped at 1.0, floored at 0.0.
    fn freshness_score(
        &self,
        timestamp: DateTime<Utc>,
        now: DateTime<Utc>,
        half_life_days: f64,
    ) -> f64 {
        if now <= timestamp {
            // Future-dated or exactly now → max freshness.
            return 1.0;
        }
        let age = now - timestamp;
        let age_days = age.num_seconds() as f64 / 86_400.0;
        let score = 2.0_f64.powf(-age_days / half_life_days);
        score.clamp(0.0, 1.0)
    }

    /// Outcome score — uses `success_rate` when outcomes are recorded,
    /// otherwise falls back to `confidence` and finally defaults to 0.5.
    fn outcome_score(&self, success_rate: f64, confidence: f64) -> f64 {
        if success_rate > 0.0 || success_rate.is_finite() && success_rate != 0.0 {
            // A non-zero, finite success_rate means outcomes were recorded.
            return success_rate.clamp(0.0, 1.0);
        }
        // No outcomes recorded — fall back to confidence, then 0.5.
        if confidence > 0.0 {
            return confidence.clamp(0.0, 1.0);
        }
        0.5
    }

    /// Conflict penalty — each flag reduces score by 10%, capped at 50%.
    fn conflict_penalty(&self, flags: &[String]) -> f64 {
        let n = flags.len() as f64;
        (0.1 * n).min(0.5)
    }
}

impl Default for RerankPipeline {
    fn default() -> Self {
        Self::new(RerankConfig::default())
    }
}

// ─────────────────────────── Tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a candidate with sensible defaults.
    fn make_candidate(
        memory_id: Uuid,
        relevance: f64,
        authority: AuthorityLevel,
        timestamp: DateTime<Utc>,
    ) -> RerankCandidate {
        RerankCandidate {
            memory_id,
            content: format!("memory-{memory_id}"),
            relevance_score: relevance,
            authority_level: authority,
            timestamp,
            confidence: 0.5,
            success_rate: 0.5,
            conflict_flags: Vec::new(),
            privacy_class: PrivacyClass::Internal,
            scope: Scope::Team,
        }
    }

    /// Helper: a timestamp N days ago.
    fn days_ago(days: i64) -> DateTime<Utc> {
        Utc::now() - Duration::days(days)
    }

    // ── Test 1: Default config weights sum to 1.0 ─────────────
    #[test]
    fn default_config_weights_sum_to_one() {
        let cfg = RerankConfig::default();
        let sum =
            cfg.relevance_weight + cfg.authority_weight + cfg.freshness_weight + cfg.outcome_weight;
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "weights must sum to 1.0, got {sum}"
        );
    }

    // ── Test 2: Single candidate → rank 1 ────────────────────
    #[test]
    fn single_candidate_gets_rank_1() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let candidates = vec![make_candidate(
            id,
            0.9,
            AuthorityLevel::L1Authoritative,
            days_ago(1),
        )];
        let results = pipeline.rerank(candidates);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[0].memory_id, id);
    }

    // ── Test 3: Higher relevance → higher rank ───────────────
    #[test]
    fn higher_relevance_ranks_higher() {
        let pipeline = RerankPipeline::default();
        let id_low = Uuid::new_v4();
        let id_high = Uuid::new_v4();
        let candidates = vec![
            make_candidate(id_low, 0.3, AuthorityLevel::L2Verified, days_ago(1)),
            make_candidate(id_high, 0.9, AuthorityLevel::L2Verified, days_ago(1)),
        ];
        let results = pipeline.rerank(candidates);
        assert_eq!(results[0].memory_id, id_high);
        assert_eq!(results[1].memory_id, id_low);
        assert!(results[0].final_score > results[1].final_score);
    }

    // ── Test 4: L0 authority beats L3 at same relevance ──────
    #[test]
    fn l0_authority_beats_l3() {
        let pipeline = RerankPipeline::default();
        let id_l3 = Uuid::new_v4();
        let id_l0 = Uuid::new_v4();
        let candidates = vec![
            make_candidate(id_l3, 0.5, AuthorityLevel::L3Inferred, days_ago(1)),
            make_candidate(id_l0, 0.5, AuthorityLevel::L0SourceOfTruth, days_ago(1)),
        ];
        let results = pipeline.rerank(candidates);
        assert_eq!(results[0].memory_id, id_l0);
        assert_eq!(results[1].memory_id, id_l3);
    }

    // ── Test 5: Fresh memory beats old at same relevance+auth ─
    #[test]
    fn fresh_beats_old() {
        let pipeline = RerankPipeline::default();
        let id_old = Uuid::new_v4();
        let id_fresh = Uuid::new_v4();
        let candidates = vec![
            make_candidate(id_old, 0.7, AuthorityLevel::L1Authoritative, days_ago(365)),
            make_candidate(id_fresh, 0.7, AuthorityLevel::L1Authoritative, days_ago(1)),
        ];
        let results = pipeline.rerank(candidates);
        assert_eq!(results[0].memory_id, id_fresh, "fresh should rank first");
    }

    // ── Test 6: Higher success_rate → higher rank ──────────────
    #[test]
    fn higher_success_rate_ranks_higher() {
        let pipeline = RerankPipeline::default();
        let id_low = Uuid::new_v4();
        let id_high = Uuid::new_v4();
        let mut c_low = make_candidate(id_low, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c_low.success_rate = 0.1;
        let mut c_high = make_candidate(id_high, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c_high.success_rate = 0.9;
        let results = pipeline.rerank(vec![c_low, c_high]);
        assert_eq!(results[0].memory_id, id_high);
    }

    // ── Test 7: Conflict penalty reduces score ────────────────
    #[test]
    fn conflict_penalty_reduces_score() {
        let pipeline = RerankPipeline::default();
        let id_clean = Uuid::new_v4();
        let id_conflict = Uuid::new_v4();
        let mut c_clean = make_candidate(id_clean, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c_clean.conflict_flags = vec![];
        let mut c_conflict =
            make_candidate(id_conflict, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c_conflict.conflict_flags = vec!["conflict:role:auth_mismatch".into()];
        let results = pipeline.rerank(vec![c_clean.clone(), c_conflict.clone()]);
        // The clean candidate must rank higher.
        assert_eq!(results[0].memory_id, id_clean);
        assert!(results[0].final_score > results[1].final_score);
        assert!((results[1].conflict_penalty - 0.1).abs() < 1e-9);
        assert!(results[0].conflict_penalty < 1e-9);
    }

    // ── Test 8: Multiple conflict flags → higher penalty (cap 50%)
    #[test]
    fn multiple_conflict_flags_capped() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c.conflict_flags = vec!["f1".into(), "f2".into(), "f3".into()];
        let results = pipeline.rerank(vec![c]);
        // 3 flags → 0.3 penalty
        assert!((results[0].conflict_penalty - 0.3).abs() < 1e-9);
    }

    // ── Test 9: Permission filter removes disallowed scopes ────
    #[test]
    fn permission_filter_removes_disallowed_scopes() {
        let pipeline = RerankPipeline::default();
        let id_allowed = Uuid::new_v4();
        let id_denied = Uuid::new_v4();
        let c_allowed = make_candidate(id_allowed, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        let mut c_denied = make_candidate(id_denied, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c_denied.scope = Scope::Enterprise;
        let results = pipeline.permission_filter(vec![c_allowed, c_denied], &[Scope::Team]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, id_allowed);
    }

    // ── Test 10: Permission filter keeps allowed scopes ─────────
    #[test]
    fn permission_filter_keeps_allowed_scopes() {
        let pipeline = RerankPipeline::default();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let mut c1 = make_candidate(id1, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c1.scope = Scope::Personal;
        let mut c2 = make_candidate(id2, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c2.scope = Scope::Team;
        let results = pipeline.permission_filter(vec![c1, c2], &[Scope::Personal, Scope::Team]);
        assert_eq!(results.len(), 2);
    }

    // ── Test 11: Restricted privacy filtered without explicit perm
    #[test]
    fn restricted_privacy_filtered() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c.privacy_class = PrivacyClass::Restricted;
        c.scope = Scope::Team;
        let results = pipeline.permission_filter(vec![c], &[Scope::Team]);
        assert!(
            results.is_empty(),
            "Restricted should be denied without explicit grant"
        );
    }

    // ── Test 12: Rerank returns top_k results ──────────────────
    #[test]
    fn rerank_returns_top_k() {
        let cfg = RerankConfig {
            top_k: 3,
            ..Default::default()
        };
        let pipeline = RerankPipeline::new(cfg);
        let candidates: Vec<RerankCandidate> = (0..10)
            .map(|i| {
                make_candidate(
                    Uuid::new_v4(),
                    i as f64 * 0.1,
                    AuthorityLevel::L2Verified,
                    days_ago(1),
                )
            })
            .collect();
        let results = pipeline.rerank(candidates);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[2].rank, 3);
    }

    // ── Test 13: Empty candidates → empty results ─────────────
    #[test]
    fn empty_candidates_empty_results() {
        let pipeline = RerankPipeline::default();
        let results = pipeline.rerank(Vec::new());
        assert!(results.is_empty());
    }

    // ── Test 14: All scores computed correctly ─────────────────
    #[test]
    fn all_scores_computed_correctly() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.7, AuthorityLevel::L1Authoritative, days_ago(0));
        c.success_rate = 0.8;
        let results = pipeline.rerank(vec![c]);
        let r = &results[0];
        assert!((r.relevance_score - 0.7).abs() < 1e-9);
        assert!((r.authority_score - 0.8).abs() < 1e-9);
        assert!((r.outcome_score - 0.8).abs() < 1e-9);
        assert!((r.conflict_penalty - 0.0).abs() < 1e-9);
        // Manual check of final_score:
        let expected = (0.7 * 0.4 + 0.8 * 0.3 + r.freshness_score * 0.15 + 0.8 * 0.15) * 1.0;
        assert!((r.final_score - expected).abs() < 1e-9);
    }

    // ── Test 15: Freshness exponential decay ───────────────────
    #[test]
    fn freshness_exponential_decay() {
        let pipeline = RerankPipeline::default();
        let now = Utc::now();
        // 30-day half-life → at 30 days the score should be 0.5
        let ts = now - Duration::days(30);
        let score = pipeline.freshness_score(ts, now, 30.0);
        assert!(
            (score - 0.5).abs() < 0.01,
            "30-day half-life → ~0.5, got {score}"
        );
        // At 0 days → 1.0
        let score0 = pipeline.freshness_score(now, now, 30.0);
        assert!((score0 - 1.0).abs() < 1e-9);
        // At 60 days → 0.25
        let ts60 = now - Duration::days(60);
        let score60 = pipeline.freshness_score(ts60, now, 30.0);
        assert!(
            (score60 - 0.25).abs() < 0.01,
            "60-day → ~0.25, got {score60}"
        );
    }

    // ── Test 16: Outcome score with no data defaults to 0.5 ───
    #[test]
    fn outcome_defaults_to_half() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c.success_rate = 0.0;
        c.confidence = 0.0;
        let results = pipeline.rerank(vec![c]);
        assert!((results[0].outcome_score - 0.5).abs() < 1e-9);
    }

    // ── Test 17: Conflict penalty with 0 flags = 0 ───────────
    #[test]
    fn conflict_penalty_zero_flags() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        let results = pipeline.rerank(vec![c]);
        assert!(results[0].conflict_penalty.abs() < 1e-9);
    }

    // ── Test 18: Conflict penalty with 5+ flags = 0.5 (capped) ─
    #[test]
    fn conflict_penalty_capped_at_half() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c.conflict_flags = vec![
            "f1".into(),
            "f2".into(),
            "f3".into(),
            "f4".into(),
            "f5".into(),
            "f6".into(),
            "f7".into(),
        ];
        let results = pipeline.rerank(vec![c]);
        assert!((results[0].conflict_penalty - 0.5).abs() < 1e-9);
    }

    // ── Test 19: Ties broken by original order ────────────────
    #[test]
    fn ties_broken_by_original_order() {
        let pipeline = RerankPipeline::default();
        let id_first = Uuid::new_v4();
        let id_second = Uuid::new_v4();
        // Identical candidates except id → same final_score → stable sort
        let candidates = vec![
            make_candidate(id_first, 0.5, AuthorityLevel::L2Verified, days_ago(1)),
            make_candidate(id_second, 0.5, AuthorityLevel::L2Verified, days_ago(1)),
        ];
        let results = pipeline.rerank(candidates);
        assert_eq!(results[0].memory_id, id_first);
        assert_eq!(results[1].memory_id, id_second);
    }

    // ── Test 20: Permission filter never fails open ────────────
    #[test]
    fn permission_filter_never_fails_open() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        // Restricted with empty allowed_scopes → must deny.
        c.privacy_class = PrivacyClass::Restricted;
        c.scope = Scope::Enterprise;
        let results = pipeline.permission_filter(vec![c.clone()], &[]);
        assert!(results.is_empty(), "must deny when no explicit grant");
        // Even with matching scope, Restricted is denied.
        let results2 = pipeline.permission_filter(vec![c], &[Scope::Enterprise]);
        assert!(
            results2.is_empty(),
            "Restricted denied even with scope match"
        );
    }

    // ── Test 21: downweight_conflicts=false disables penalty ──
    #[test]
    fn downweight_conflicts_disabled() {
        let cfg = RerankConfig {
            downweight_conflicts: false,
            ..Default::default()
        };
        let pipeline = RerankPipeline::new(cfg);
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c.conflict_flags = vec!["f1".into(), "f2".into()];
        let results = pipeline.rerank(vec![c]);
        assert!(
            results[0].conflict_penalty.abs() < 1e-9,
            "penalty should be 0 when disabled"
        );
    }

    // ── Test 22: Confidence fallback when success_rate is 0 ────
    #[test]
    fn confidence_fallback_when_no_success_rate() {
        let pipeline = RerankPipeline::default();
        let id = Uuid::new_v4();
        let mut c = make_candidate(id, 0.5, AuthorityLevel::L2Verified, days_ago(1));
        c.success_rate = 0.0;
        c.confidence = 0.7;
        let results = pipeline.rerank(vec![c]);
        assert!((results[0].outcome_score - 0.7).abs() < 1e-9);
    }
}
