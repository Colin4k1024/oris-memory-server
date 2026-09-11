//! Reflection & Consolidation batch processing — Issue #21 (P3).
//!
//! Implements the §9.3 Reflection pipeline:
//!
//! ```text
//! Episode → (cluster → deduplicate → evaluate) → Experience
//! Experience → (validate → extract pattern → verify) → DecisionPattern / BestPractice
//! BestPractice → (version → render) → SOP / Rule / Skill
//! ```
//!
//! Three batch jobs ([`ReflectionJob`], [`ConsolidationJob`], [`PublishJob`])
//! implement the pipeline stages. [`BatchScheduler`] drives periodic
//! execution. All jobs are in-memory and deterministic, making them trivially
//! testable without a live database. The rendering logic in [`PublishJob`]
//! follows the patterns established in [`crate::skill_projection`].

use std::collections::{BTreeMap, HashMap, HashSet};

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};
use uuid::Uuid;

use oris_memory_contract::{AuthorityLevel, Scope};

// ──────────────────────────── Constants ────────────────────────

/// Content similarity above this value is treated as a near-duplicate.
const DEDUP_SIMILARITY_THRESHOLD: f64 = 0.9;

/// Minimum cluster size for an experience to be considered reusable.
const MIN_CLUSTER_SIZE: usize = 3;

/// Minimum confidence (success-rate proxy) for reusability.
const REUSABILITY_THRESHOLD: f64 = 0.7;

/// Minimum confidence for an experience to pass validation.
const VALIDATION_CONFIDENCE_THRESHOLD: f64 = 0.5;

/// Minimum success rate for a pattern to be verified.
const PATTERN_VERIFY_THRESHOLD: f64 = 0.6;

/// Seconds in a day — used for the recency decay factor.
const SECS_PER_DAY: f64 = 86_400.0;

/// Convenience alias — the memory scope under which a best practice is published.
pub type MemoryScope = Scope;

// ──────────────────────────── Errors ────────────────────────────

/// Errors emitted by the reflection and consolidation pipeline.
///
/// The in-memory batch jobs currently produce zero-valued reports on empty
/// input rather than returning errors, but this type is defined for future
/// storage-backed implementations that may fail on I/O.
#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum ReflectionError {
    /// No episodes were available to start the reflection cycle.
    #[error("no episodes available for reflection")]
    NoEpisodes,

    /// An experience failed validation.
    #[error("experience {id} failed validation: {reason}")]
    ValidationFailed { id: Uuid, reason: String },

    /// A decision pattern could not be verified against historical data.
    #[error("pattern {id} could not be verified")]
    PatternNotVerified { id: Uuid },

    /// Publishing a best practice failed.
    #[error("publish failed: {0}")]
    PublishFailed(String),
}

// ──────────────────────────── BatchJob Trait ────────────────────────

/// Common interface for periodic batch-processing jobs.
#[async_trait]
pub trait BatchJob: Send + Sync {
    /// The serializable report produced by a single run.
    type Report: Send;

    /// Execute one full batch cycle and return the summary report.
    async fn run(&self) -> Self::Report;
}

// ──────────────────────────── Episode ────────────────────────────

/// Outcome of a single observed episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeOutcome {
    Success,
    Failure,
    Partial,
    Abandoned,
}

impl EpisodeOutcome {
    /// Stable lowercase label for tracing and metrics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Partial => "partial",
            Self::Abandoned => "abandoned",
        }
    }

    /// Numerical success contribution in the range 0.0–1.0.
    pub fn success_rate(self) -> f64 {
        match self {
            Self::Success => 1.0,
            Self::Partial => 0.5,
            Self::Failure | Self::Abandoned => 0.0,
        }
    }
}

/// A single observed task episode — the raw input to the reflection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: Uuid,
    pub tenant_id: String,
    pub user_id: Option<String>,
    pub task_id: Option<Uuid>,
    pub content: String,
    pub outcome: EpisodeOutcome,
    pub timestamp: DateTime<Utc>,
    pub entities: Vec<String>,
}

impl Episode {
    /// Create a new episode with a generated UUID and current timestamp.
    pub fn new(
        tenant_id: &str,
        content: &str,
        outcome: EpisodeOutcome,
        entities: Vec<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            tenant_id: tenant_id.into(),
            user_id: None,
            task_id: None,
            content: content.into(),
            outcome,
            timestamp: Utc::now(),
            entities,
        }
    }
}

// ──────────────────────────── Experience ────────────────────────

/// A reusable experience distilled from one or more episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Experience {
    pub id: Uuid,
    pub source_episodes: Vec<Uuid>,
    pub summary: String,
    pub applicability_conditions: Vec<String>,
    pub confidence: f64,
    pub authority_level: AuthorityLevel,
}

// ──────────────────────────── DecisionPattern ────────────────────

/// A generalized decision pattern extracted from multiple experiences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionPattern {
    pub id: Uuid,
    pub source_experiences: Vec<Uuid>,
    pub pattern_text: String,
    pub conditions: Vec<String>,
    pub success_rate: f64,
    pub verified: bool,
}

// ──────────────────────────── BestPractice ────────────────────────

/// A versioned best practice ready for publication as an SOP, rule, or skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    pub id: Uuid,
    pub source_pattern: Uuid,
    pub practice_text: String,
    pub version: u32,
    pub published: bool,
    pub scope: MemoryScope,
}

// ──────────────────────────── EpisodeCluster ────────────────────────

/// A group of episodes sharing entity and outcome similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeCluster {
    pub cluster_key: String,
    pub episodes: Vec<Episode>,
}

// ──────────────────────────── ValidationResult ────────────────────────

/// Result of validating a single experience before consolidation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub experience_id: Uuid,
    pub valid: bool,
    pub issues: Vec<String>,
}

// ──────────────────────────── Reports ────────────────────────────

/// Summary of a completed reflection run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectionReport {
    pub episodes_processed: usize,
    pub clusters_formed: usize,
    pub experiences_created: usize,
    pub reusable_experiences: usize,
    pub experiences: Vec<Experience>,
}

/// Summary of a completed consolidation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidationReport {
    pub experiences_validated: usize,
    pub patterns_extracted: usize,
    pub patterns_verified: usize,
    pub patterns: Vec<DecisionPattern>,
}

/// The kind of published artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    Sop,
    Rule,
    Skill,
}

/// Result of publishing a best practice as a versioned artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublishResult {
    pub best_practice_id: Uuid,
    pub version: u32,
    pub artifact_type: ArtifactType,
    pub files: BTreeMap<String, String>,
    pub published: bool,
}

// ──────────────────────────── BatchScheduler ────────────────────────

/// Schedules periodic execution of batch jobs.
#[derive(Debug, Clone)]
pub struct BatchScheduler {
    pub interval: Duration,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: DateTime<Utc>,
}

impl BatchScheduler {
    /// Create a scheduler with the given interval; `next_run` is set to
    /// `now + interval`.
    pub fn new(interval: Duration) -> Self {
        let now = Utc::now();
        Self {
            interval,
            last_run: None,
            next_run: now + interval,
        }
    }

    /// Returns `true` when the current time has reached or passed `next_run`.
    pub fn should_run(&self) -> bool {
        Utc::now() >= self.next_run
    }

    /// Record that a run just completed and reschedule the next run.
    pub fn mark_completed(&mut self) {
        let now = Utc::now();
        self.last_run = Some(now);
        self.next_run = now + self.interval;
    }

    /// Wall-clock duration remaining until the next scheduled run (zero if
    /// already due).
    pub fn time_until_next_run(&self) -> Duration {
        let now = Utc::now();
        if now >= self.next_run {
            Duration::zero()
        } else {
            self.next_run - now
        }
    }
}

// ──────────────────────────── Helpers ────────────────────────────

/// Jaccard similarity on lowercased word-token sets.
fn text_similarity(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let sa: HashSet<&str> = a_lower.split_whitespace().collect();
    let sb: HashSet<&str> = b_lower.split_whitespace().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let intersection = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Map a memory scope to the most appropriate published artifact type.
fn artifact_type_for_scope(scope: MemoryScope) -> ArtifactType {
    match scope {
        Scope::Enterprise | Scope::Factory => ArtifactType::Sop,
        Scope::Process => ArtifactType::Rule,
        _ => ArtifactType::Skill,
    }
}

/// Slugify a string for use as a skill or SOP name.
fn slug(value: &str) -> String {
    let value = value.to_lowercase();
    let mut out = String::new();
    let mut dash = false;
    for c in value.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    out.trim_matches('-').chars().take(64).collect()
}

/// Escape a string for use inside a YAML double-quoted scalar.
fn yaml_double(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace(['\n', '\r'], " ")
}

/// Escape a string for use inside a YAML single-quoted scalar.
fn yaml_single_line(value: &str) -> String {
    value.replace(['\n', '\r'], " ").replace('\'', "''")
}

/// Truncate to at most `max` chars.
fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

// ──────────────────────────── ReflectionJob ────────────────────────

/// Stage 1 batch processor: Episode → Experience.
///
/// Clusters similar episodes, removes near-duplicates, and evaluates each
/// cluster's reusability into a serializable [`Experience`].
pub struct ReflectionJob {
    episodes: Vec<Episode>,
}

impl ReflectionJob {
    /// Create a reflection job with the given episodes.
    pub fn new(episodes: Vec<Episode>) -> Self {
        Self { episodes }
    }

    /// Group episodes by sorted-entity signature + outcome.
    pub fn cluster_episodes(episodes: &[Episode]) -> Vec<EpisodeCluster> {
        let mut map: HashMap<String, Vec<Episode>> = HashMap::new();
        for ep in episodes {
            let mut ents = ep.entities.clone();
            ents.sort();
            ents.dedup();
            let key = format!("{}|{}", ents.join(","), ep.outcome.as_str());
            map.entry(key).or_default().push(ep.clone());
        }
        let mut clusters: Vec<EpisodeCluster> = map
            .into_iter()
            .map(|(cluster_key, episodes)| EpisodeCluster {
                cluster_key,
                episodes,
            })
            .collect();
        clusters.sort_by(|a, b| a.cluster_key.cmp(&b.cluster_key));
        clusters
    }

    /// Remove near-duplicate episodes (content similarity > 0.9), keeping
    /// the first occurrence in each group.
    pub fn deduplicate(cluster: &EpisodeCluster) -> Vec<Episode> {
        let mut kept: Vec<Episode> = Vec::new();
        for ep in &cluster.episodes {
            let is_dup = kept
                .iter()
                .any(|k| text_similarity(&k.content, &ep.content) > DEDUP_SIMILARITY_THRESHOLD);
            if !is_dup {
                kept.push(ep.clone());
            }
        }
        kept
    }

    /// Evaluate a cluster into a single [`Experience`], blending frequency,
    /// success rate, and recency into a confidence score.
    pub fn evaluate(cluster: &EpisodeCluster) -> Experience {
        let episodes = &cluster.episodes;
        let frequency = episodes.len();
        if frequency == 0 {
            return Experience {
                id: Uuid::new_v4(),
                source_episodes: vec![],
                summary: String::new(),
                applicability_conditions: vec![],
                confidence: 0.0,
                authority_level: AuthorityLevel::L3Inferred,
            };
        }
        let success_rate: f64 = episodes
            .iter()
            .map(|e| e.outcome.success_rate())
            .sum::<f64>()
            / frequency as f64;
        let now = Utc::now();
        let recency: f64 = episodes
            .iter()
            .map(|e| {
                let age = (now - e.timestamp).num_seconds().max(0) as f64;
                1.0 / (1.0 + age / SECS_PER_DAY)
            })
            .sum::<f64>()
            / frequency as f64;
        let frequency_factor = (frequency as f64 / 10.0).min(1.0);
        let confidence = (success_rate * 0.7 + recency * 0.15 + frequency_factor * 0.15).min(1.0);
        let authority_level = if success_rate >= 0.8 {
            AuthorityLevel::L2Verified
        } else if success_rate >= 0.5 {
            AuthorityLevel::L1Authoritative
        } else {
            AuthorityLevel::L3Inferred
        };
        let source_episodes: Vec<Uuid> = episodes.iter().map(|e| e.id).collect();
        let applicability: Vec<String> = {
            let set: HashSet<String> = episodes
                .iter()
                .flat_map(|e| e.entities.iter().cloned())
                .collect();
            let mut v: Vec<String> = set.into_iter().collect();
            v.sort();
            v
        };
        let summary = episodes
            .first()
            .map(|e| e.content.clone())
            .unwrap_or_default();
        Experience {
            id: Uuid::new_v4(),
            source_episodes,
            summary,
            applicability_conditions: applicability,
            confidence,
            authority_level,
        }
    }

    /// Determine whether an experience is reusable: appeared ≥ 3 times and
    /// confidence (success-rate proxy) ≥ 0.7.
    pub fn is_reusable(experience: &Experience) -> bool {
        experience.source_episodes.len() >= MIN_CLUSTER_SIZE
            && experience.confidence >= REUSABILITY_THRESHOLD
    }
}

#[async_trait]
impl BatchJob for ReflectionJob {
    type Report = ReflectionReport;

    async fn run(&self) -> Self::Report {
        let clusters = Self::cluster_episodes(&self.episodes);
        let mut experiences = Vec::new();
        let mut reusable = 0;

        for cluster in &clusters {
            let deduped = Self::deduplicate(cluster);
            if deduped.is_empty() {
                continue;
            }
            let deduped_cluster = EpisodeCluster {
                cluster_key: cluster.cluster_key.clone(),
                episodes: deduped,
            };
            let exp = Self::evaluate(&deduped_cluster);
            if Self::is_reusable(&exp) {
                reusable += 1;
            }
            experiences.push(exp);
        }

        info!(
            episodes = self.episodes.len(),
            clusters = clusters.len(),
            experiences = experiences.len(),
            reusable,
            "reflection batch completed"
        );
        debug!(
            "reflection clusters: {:?}",
            clusters.iter().map(|c| &c.cluster_key).collect::<Vec<_>>()
        );

        ReflectionReport {
            episodes_processed: self.episodes.len(),
            clusters_formed: clusters.len(),
            experiences_created: experiences.len(),
            reusable_experiences: reusable,
            experiences,
        }
    }
}

// ──────────────────────────── ConsolidationJob ────────────────────

/// Stage 2 batch processor: Experience → DecisionPattern.
///
/// Validates experiences, extracts generalized patterns, and verifies them
/// against historical data.
pub struct ConsolidationJob {
    experiences: Vec<Experience>,
    historical_patterns: Vec<DecisionPattern>,
}

impl ConsolidationJob {
    /// Create a consolidation job with the given experiences.
    pub fn new(experiences: Vec<Experience>) -> Self {
        Self {
            experiences,
            historical_patterns: Vec::new(),
        }
    }

    /// Attach historical patterns for cross-checking during verification.
    pub fn with_historical(mut self, patterns: Vec<DecisionPattern>) -> Self {
        self.historical_patterns = patterns;
        self
    }

    /// Validate a single experience: must have source episodes, sufficient
    /// confidence, and at least one applicability condition.
    pub fn validate_experience(experience: &Experience) -> ValidationResult {
        let mut issues = Vec::new();
        if experience.source_episodes.is_empty() {
            issues.push("no source episodes".into());
        }
        if experience.confidence < VALIDATION_CONFIDENCE_THRESHOLD {
            issues.push("confidence below threshold".into());
        }
        if experience.applicability_conditions.is_empty() {
            issues.push("no applicability conditions".into());
        }
        ValidationResult {
            experience_id: experience.id,
            valid: issues.is_empty(),
            issues,
        }
    }

    /// Generalize a set of experiences into a single [`DecisionPattern`].
    pub fn extract_pattern(experiences: &[Experience]) -> DecisionPattern {
        let source_experiences: Vec<Uuid> = experiences.iter().map(|e| e.id).collect();
        let pattern_text = experiences
            .iter()
            .map(|e| e.summary.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        let conditions: Vec<String> = {
            let set: HashSet<String> = experiences
                .iter()
                .flat_map(|e| e.applicability_conditions.iter().cloned())
                .collect();
            let mut v: Vec<String> = set.into_iter().collect();
            v.sort();
            v
        };
        let success_rate = if experiences.is_empty() {
            0.0
        } else {
            experiences.iter().map(|e| e.confidence).sum::<f64>() / experiences.len() as f64
        };
        DecisionPattern {
            id: Uuid::new_v4(),
            source_experiences,
            pattern_text,
            conditions,
            success_rate,
            verified: false,
        }
    }

    /// Verify a pattern: success rate ≥ threshold, non-empty conditions, and
    /// consistency with historical patterns (if any are stored).
    pub fn verify_pattern(&self, pattern: &DecisionPattern) -> bool {
        if pattern.success_rate < PATTERN_VERIFY_THRESHOLD || pattern.conditions.is_empty() {
            return false;
        }
        if self.historical_patterns.is_empty() {
            return true;
        }
        self.historical_patterns
            .iter()
            .any(|hp| hp.conditions.iter().any(|c| pattern.conditions.contains(c)))
    }
}

#[async_trait]
impl BatchJob for ConsolidationJob {
    type Report = ConsolidationReport;

    async fn run(&self) -> Self::Report {
        let valid: Vec<Experience> = self
            .experiences
            .iter()
            .filter(|e| Self::validate_experience(e).valid)
            .cloned()
            .collect();

        let mut patterns = Vec::new();
        let mut verified_count = 0;

        if !valid.is_empty() {
            let pattern = Self::extract_pattern(&valid);
            let verified = self.verify_pattern(&pattern);
            if verified {
                verified_count = 1;
            }
            patterns.push(pattern);
        }

        info!(
            validated = valid.len(),
            total = self.experiences.len(),
            patterns = patterns.len(),
            verified = verified_count,
            "consolidation batch completed"
        );

        ConsolidationReport {
            experiences_validated: valid.len(),
            patterns_extracted: patterns.len(),
            patterns_verified: verified_count,
            patterns,
        }
    }
}

// ──────────────────────────── PublishJob ────────────────────────

/// Stage 3: publish a best practice as a versioned SOP, rule, or skill.
///
/// Follows the rendering patterns from [`crate::skill_projection`]: produces
/// a front-matter Markdown body plus an `agents/openai.yaml` descriptor.
pub struct PublishJob;

impl PublishJob {
    /// Create a new publish job.
    pub fn new() -> Self {
        Self
    }

    /// Render `best_practice` into a versioned artifact (SOP / Rule / Skill).
    pub fn publish(&self, best_practice: &BestPractice) -> PublishResult {
        let id_short = &best_practice.id.to_string()[..8];
        let skill_name = slug(&format!("oris-bp-{id_short}"));
        let artifact_type = artifact_type_for_scope(best_practice.scope);
        let scope_label = best_practice.scope.as_str();

        let mut body = format!(
            "---\nname: {skill_name}\nversion: {}\nscope: {scope_label}\npublished: {}\ndescription: '{}'\n---\n\n# Best Practice v{}\n\n{}\n\n",
            best_practice.version,
            best_practice.published,
            yaml_single_line(&truncate(&best_practice.practice_text, 120)),
            best_practice.version,
            best_practice.practice_text,
        );
        body.push_str("## Applicability\n\n");
        body.push_str(&format!(
            "This practice applies at the `{scope_label}` scope. Use when its conditions match the current environment.\n\n"
        ));
        body.push_str("## Safety and validation\n\n");
        body.push_str(
            "Treat this practice as a suggestion. Preserve the Agent's permissions, sandbox, approvals, and repository rules.\n",
        );

        let filename = match artifact_type {
            ArtifactType::Sop => "SOP.md",
            ArtifactType::Rule => "RULE.md",
            ArtifactType::Skill => "SKILL.md",
        };

        let mut files = BTreeMap::new();
        files.insert(filename.to_string(), body);
        files.insert(
            "agents/openai.yaml".into(),
            format!(
                "interface:\n  display_name: \"Best Practice v{}\"\n  short_description: \"{}\"\n  default_prompt: \"Apply this Oris best practice only when applicable, validate it, and record the outcome.\"\n",
                best_practice.version,
                yaml_double(&truncate(&best_practice.practice_text, 80)),
            ),
        );
        files.insert(
            "references/oris-best-practice.json".into(),
            serde_json::to_string_pretty(best_practice).unwrap_or_else(|_| "{}".into()),
        );

        info!(
            best_practice_id = %best_practice.id,
            version = best_practice.version,
            artifact = ?artifact_type,
            "best practice published"
        );

        PublishResult {
            best_practice_id: best_practice.id,
            version: best_practice.version,
            artifact_type,
            files,
            published: true,
        }
    }
}

impl Default for PublishJob {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────── Tests ────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Episode / EpisodeOutcome ──

    #[test]
    fn test_episode_outcome_success_rate() {
        assert!((EpisodeOutcome::Success.success_rate() - 1.0).abs() < f64::EPSILON);
        assert!((EpisodeOutcome::Partial.success_rate() - 0.5).abs() < f64::EPSILON);
        assert!((EpisodeOutcome::Failure.success_rate() - 0.0).abs() < f64::EPSILON);
        assert!((EpisodeOutcome::Abandoned.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_episode_outcome_as_str() {
        assert_eq!(EpisodeOutcome::Success.as_str(), "success");
        assert_eq!(EpisodeOutcome::Failure.as_str(), "failure");
        assert_eq!(EpisodeOutcome::Partial.as_str(), "partial");
        assert_eq!(EpisodeOutcome::Abandoned.as_str(), "abandoned");
    }

    #[test]
    fn test_episode_new_generates_id() {
        let ep = Episode::new(
            "acme",
            "did the thing",
            EpisodeOutcome::Success,
            vec!["entity-a".into()],
        );
        assert!(!ep.id.to_string().is_empty());
        assert_eq!(ep.tenant_id, "acme");
        assert!(ep.user_id.is_none());
        assert!(ep.task_id.is_none());
        assert_eq!(ep.outcome, EpisodeOutcome::Success);
        assert_eq!(ep.entities, vec!["entity-a".to_string()]);
    }

    // ── Clustering ──

    #[test]
    fn test_cluster_episodes_groups_by_entity_and_outcome() {
        let eps = vec![
            make_episode(
                "acme",
                "did task A",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "did task B",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "did task C",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
        ];
        let clusters = ReflectionJob::cluster_episodes(&eps);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].episodes.len(), 3);
    }

    #[test]
    fn test_cluster_episodes_separates_different_outcomes() {
        let eps = vec![
            make_episode(
                "acme",
                "success run",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "failure run",
                EpisodeOutcome::Failure,
                vec!["pump".into()],
            ),
        ];
        let clusters = ReflectionJob::cluster_episodes(&eps);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_cluster_episodes_separates_different_entities() {
        let eps = vec![
            make_episode(
                "acme",
                "pump task",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "valve task",
                EpisodeOutcome::Success,
                vec!["valve".into()],
            ),
        ];
        let clusters = ReflectionJob::cluster_episodes(&eps);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_cluster_episodes_empty_input() {
        let clusters = ReflectionJob::cluster_episodes(&[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_cluster_episodes_unordered_entities_group_together() {
        let eps = vec![
            make_episode(
                "acme",
                "task one",
                EpisodeOutcome::Success,
                vec!["b".into(), "a".into()],
            ),
            make_episode(
                "acme",
                "task two",
                EpisodeOutcome::Success,
                vec!["a".into(), "b".into()],
            ),
        ];
        let clusters = ReflectionJob::cluster_episodes(&eps);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].episodes.len(), 2);
    }

    // ── Deduplication ──

    #[test]
    fn test_deduplicate_removes_near_duplicates() {
        let cluster = EpisodeCluster {
            cluster_key: "k".into(),
            episodes: vec![
                make_episode(
                    "acme",
                    "The quick brown fox jumps",
                    EpisodeOutcome::Success,
                    vec!["x".into()],
                ),
                make_episode(
                    "acme",
                    "The quick brown fox jumps",
                    EpisodeOutcome::Success,
                    vec!["x".into()],
                ),
                make_episode(
                    "acme",
                    "The quick brown fox jumps over",
                    EpisodeOutcome::Success,
                    vec!["x".into()],
                ),
            ],
        };
        let deduped = ReflectionJob::deduplicate(&cluster);
        // First two are identical (sim=1.0>0.9) → only first kept.
        // Third shares 5/6 tokens = 0.833 < 0.9 → kept.
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_deduplicate_keeps_distinct_episodes() {
        let cluster = EpisodeCluster {
            cluster_key: "k".into(),
            episodes: vec![
                make_episode(
                    "acme",
                    "completely different content alpha",
                    EpisodeOutcome::Success,
                    vec!["x".into()],
                ),
                make_episode(
                    "acme",
                    "totally unrelated text beta",
                    EpisodeOutcome::Success,
                    vec!["x".into()],
                ),
            ],
        };
        let deduped = ReflectionJob::deduplicate(&cluster);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_deduplicate_empty_cluster() {
        let cluster = EpisodeCluster {
            cluster_key: "k".into(),
            episodes: vec![],
        };
        let deduped = ReflectionJob::deduplicate(&cluster);
        assert!(deduped.is_empty());
    }

    // ── Evaluation ──

    #[test]
    fn test_evaluate_computes_high_confidence_for_all_success() {
        let cluster = make_cluster(
            "pump|success",
            vec![
                make_episode(
                    "acme",
                    "ran pump",
                    EpisodeOutcome::Success,
                    vec!["pump".into()],
                ),
                make_episode(
                    "acme",
                    "ran pump",
                    EpisodeOutcome::Success,
                    vec!["pump".into()],
                ),
                make_episode(
                    "acme",
                    "ran pump",
                    EpisodeOutcome::Success,
                    vec!["pump".into()],
                ),
            ],
        );
        let exp = ReflectionJob::evaluate(&cluster);
        assert!(exp.confidence > 0.7);
        assert_eq!(exp.source_episodes.len(), 3);
    }

    #[test]
    fn test_evaluate_assigns_authority_level() {
        let success_cluster = make_cluster(
            "pump|success",
            vec![
                make_episode("acme", "ok", EpisodeOutcome::Success, vec!["pump".into()]),
                make_episode("acme", "ok", EpisodeOutcome::Success, vec!["pump".into()]),
                make_episode("acme", "ok", EpisodeOutcome::Success, vec!["pump".into()]),
            ],
        );
        let exp = ReflectionJob::evaluate(&success_cluster);
        assert_eq!(exp.authority_level, AuthorityLevel::L2Verified);

        let fail_cluster = make_cluster(
            "pump|failure",
            vec![
                make_episode("acme", "bad", EpisodeOutcome::Failure, vec!["pump".into()]),
                make_episode("acme", "bad", EpisodeOutcome::Failure, vec!["pump".into()]),
                make_episode("acme", "bad", EpisodeOutcome::Failure, vec!["pump".into()]),
            ],
        );
        let exp_fail = ReflectionJob::evaluate(&fail_cluster);
        assert_eq!(exp_fail.authority_level, AuthorityLevel::L3Inferred);
    }

    #[test]
    fn test_evaluate_collects_applicability_conditions() {
        let cluster = make_cluster(
            "multi|success",
            vec![
                make_episode(
                    "acme",
                    "task",
                    EpisodeOutcome::Success,
                    vec!["a".into(), "b".into()],
                ),
                make_episode(
                    "acme",
                    "task",
                    EpisodeOutcome::Success,
                    vec!["b".into(), "c".into()],
                ),
            ],
        );
        let exp = ReflectionJob::evaluate(&cluster);
        assert_eq!(
            exp.applicability_conditions,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_evaluate_empty_cluster_returns_zero_confidence() {
        let cluster = EpisodeCluster {
            cluster_key: "empty".into(),
            episodes: vec![],
        };
        let exp = ReflectionJob::evaluate(&cluster);
        assert!((exp.confidence - 0.0).abs() < f64::EPSILON);
        assert!(exp.source_episodes.is_empty());
        assert_eq!(exp.authority_level, AuthorityLevel::L3Inferred);
    }

    // ── Reusability ──

    #[test]
    fn test_is_reusable_passes_threshold() {
        let exp = Experience {
            id: Uuid::new_v4(),
            source_episodes: vec![Uuid::new_v4(); 5],
            summary: "reusable".into(),
            applicability_conditions: vec!["c".into()],
            confidence: 0.9,
            authority_level: AuthorityLevel::L2Verified,
        };
        assert!(ReflectionJob::is_reusable(&exp));
    }

    #[test]
    fn test_is_reusable_fails_low_frequency() {
        let exp = Experience {
            id: Uuid::new_v4(),
            source_episodes: vec![Uuid::new_v4(); 2],
            summary: "too few".into(),
            applicability_conditions: vec!["c".into()],
            confidence: 0.9,
            authority_level: AuthorityLevel::L2Verified,
        };
        assert!(!ReflectionJob::is_reusable(&exp));
    }

    #[test]
    fn test_is_reusable_fails_low_confidence() {
        let exp = Experience {
            id: Uuid::new_v4(),
            source_episodes: vec![Uuid::new_v4(); 5],
            summary: "low success".into(),
            applicability_conditions: vec!["c".into()],
            confidence: 0.5,
            authority_level: AuthorityLevel::L3Inferred,
        };
        assert!(!ReflectionJob::is_reusable(&exp));
    }

    // ── Reflection run ──

    #[tokio::test]
    async fn test_reflection_run_produces_report() {
        let eps = vec![
            make_episode(
                "acme",
                "ran pump A",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "ran pump B",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "ran pump C",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "valve failed",
                EpisodeOutcome::Failure,
                vec!["valve".into()],
            ),
        ];
        let job = ReflectionJob::new(eps);
        let report = job.run().await;
        assert_eq!(report.episodes_processed, 4);
        assert_eq!(report.clusters_formed, 2);
        assert_eq!(report.experiences_created, 2);
        assert_eq!(report.reusable_experiences, 1);
    }

    #[tokio::test]
    async fn test_reflection_run_empty_episodes() {
        let job = ReflectionJob::new(vec![]);
        let report = job.run().await;
        assert_eq!(report.episodes_processed, 0);
        assert_eq!(report.clusters_formed, 0);
        assert_eq!(report.experiences_created, 0);
        assert_eq!(report.reusable_experiences, 0);
        assert!(report.experiences.is_empty());
    }

    #[tokio::test]
    async fn test_reflection_dedup_reduces_cluster_size() {
        let eps = vec![
            make_episode(
                "acme",
                "identical pump task",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "identical pump task",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
            make_episode(
                "acme",
                "identical pump task",
                EpisodeOutcome::Success,
                vec!["pump".into()],
            ),
        ];
        let job = ReflectionJob::new(eps);
        let report = job.run().await;
        // All three are near-duplicates → dedup keeps 1 → not reusable (< 3)
        assert_eq!(report.experiences_created, 1);
        assert_eq!(report.reusable_experiences, 0);
    }

    // ── Validation ──

    #[test]
    fn test_validate_experience_valid() {
        let exp = make_experience(0.8, AuthorityLevel::L2Verified, 3);
        let vr = ConsolidationJob::validate_experience(&exp);
        assert!(vr.valid);
        assert!(vr.issues.is_empty());
    }

    #[test]
    fn test_validate_experience_invalid_no_episodes() {
        let exp = Experience {
            id: Uuid::new_v4(),
            source_episodes: vec![],
            summary: "empty".into(),
            applicability_conditions: vec!["c".into()],
            confidence: 0.8,
            authority_level: AuthorityLevel::L2Verified,
        };
        let vr = ConsolidationJob::validate_experience(&exp);
        assert!(!vr.valid);
        assert!(vr.issues.iter().any(|i| i.contains("no source episodes")));
    }

    #[test]
    fn test_validate_experience_invalid_low_confidence() {
        let exp = Experience {
            id: Uuid::new_v4(),
            source_episodes: vec![Uuid::new_v4(); 3],
            summary: "low".into(),
            applicability_conditions: vec!["c".into()],
            confidence: 0.3,
            authority_level: AuthorityLevel::L3Inferred,
        };
        let vr = ConsolidationJob::validate_experience(&exp);
        assert!(!vr.valid);
        assert!(vr.issues.iter().any(|i| i.contains("confidence")));
    }

    #[test]
    fn test_validate_experience_invalid_no_applicability() {
        let exp = Experience {
            id: Uuid::new_v4(),
            source_episodes: vec![Uuid::new_v4(); 3],
            summary: "no conditions".into(),
            applicability_conditions: vec![],
            confidence: 0.8,
            authority_level: AuthorityLevel::L2Verified,
        };
        let vr = ConsolidationJob::validate_experience(&exp);
        assert!(!vr.valid);
        assert!(vr.issues.iter().any(|i| i.contains("applicability")));
    }

    // ── Pattern extraction ──

    #[test]
    fn test_extract_pattern_generalizes_from_experiences() {
        let exps = vec![
            Experience {
                id: Uuid::new_v4(),
                source_episodes: vec![Uuid::new_v4()],
                summary: "use approach A".into(),
                applicability_conditions: vec!["cond1".into()],
                confidence: 0.8,
                authority_level: AuthorityLevel::L2Verified,
            },
            Experience {
                id: Uuid::new_v4(),
                source_episodes: vec![Uuid::new_v4()],
                summary: "use approach B".into(),
                applicability_conditions: vec!["cond2".into()],
                confidence: 0.9,
                authority_level: AuthorityLevel::L2Verified,
            },
        ];
        let pattern = ConsolidationJob::extract_pattern(&exps);
        assert_eq!(pattern.source_experiences.len(), 2);
        assert!(pattern.pattern_text.contains("approach A"));
        assert!(pattern.pattern_text.contains("approach B"));
        assert!(pattern.conditions.contains(&"cond1".to_string()));
        assert!(pattern.conditions.contains(&"cond2".to_string()));
        assert!(!pattern.verified);
    }

    #[test]
    fn test_extract_pattern_computes_average_success_rate() {
        let exps = vec![
            make_experience(0.6, AuthorityLevel::L1Authoritative, 1),
            make_experience(0.8, AuthorityLevel::L2Verified, 1),
        ];
        let pattern = ConsolidationJob::extract_pattern(&exps);
        assert!((pattern.success_rate - 0.7).abs() < 0.001);
    }

    // ── Pattern verification ──

    #[test]
    fn test_verify_pattern_passes_with_high_success_rate() {
        let job = ConsolidationJob::new(vec![]);
        let pattern = DecisionPattern {
            id: Uuid::new_v4(),
            source_experiences: vec![],
            pattern_text: "good".into(),
            conditions: vec!["c".into()],
            success_rate: 0.8,
            verified: false,
        };
        assert!(job.verify_pattern(&pattern));
    }

    #[test]
    fn test_verify_pattern_fails_low_success_rate() {
        let job = ConsolidationJob::new(vec![]);
        let pattern = DecisionPattern {
            id: Uuid::new_v4(),
            source_experiences: vec![],
            pattern_text: "weak".into(),
            conditions: vec!["c".into()],
            success_rate: 0.4,
            verified: false,
        };
        assert!(!job.verify_pattern(&pattern));
    }

    #[test]
    fn test_verify_pattern_fails_empty_conditions() {
        let job = ConsolidationJob::new(vec![]);
        let pattern = DecisionPattern {
            id: Uuid::new_v4(),
            source_experiences: vec![],
            pattern_text: "no conditions".into(),
            conditions: vec![],
            success_rate: 0.9,
            verified: false,
        };
        assert!(!job.verify_pattern(&pattern));
    }

    #[test]
    fn test_verify_pattern_cross_checks_historical() {
        let historical = vec![DecisionPattern {
            id: Uuid::new_v4(),
            source_experiences: vec![],
            pattern_text: "old".into(),
            conditions: vec!["shared-cond".into()],
            success_rate: 0.9,
            verified: true,
        }];

        let job = ConsolidationJob::new(vec![]).with_historical(historical);

        // Pattern with overlapping condition → passes
        let pattern_ok = DecisionPattern {
            id: Uuid::new_v4(),
            source_experiences: vec![],
            pattern_text: "new with shared".into(),
            conditions: vec!["shared-cond".into()],
            success_rate: 0.8,
            verified: false,
        };
        assert!(job.verify_pattern(&pattern_ok));

        // Pattern with no overlap → fails
        let pattern_no_overlap = DecisionPattern {
            id: Uuid::new_v4(),
            source_experiences: vec![],
            pattern_text: "new unrelated".into(),
            conditions: vec!["different-cond".into()],
            success_rate: 0.8,
            verified: false,
        };
        assert!(!job.verify_pattern(&pattern_no_overlap));
    }

    // ── Consolidation run ──

    #[tokio::test]
    async fn test_consolidation_run_produces_report() {
        let exps = vec![
            make_experience(0.8, AuthorityLevel::L2Verified, 3),
            make_experience(0.7, AuthorityLevel::L2Verified, 3),
        ];
        let job = ConsolidationJob::new(exps);
        let report = job.run().await;
        assert_eq!(report.experiences_validated, 2);
        assert_eq!(report.patterns_extracted, 1);
        assert_eq!(report.patterns_verified, 1);
    }

    #[tokio::test]
    async fn test_consolidation_run_filters_invalid_experiences() {
        let exps = vec![
            make_experience(0.8, AuthorityLevel::L2Verified, 3),
            make_experience(0.3, AuthorityLevel::L3Inferred, 3),
        ];
        let job = ConsolidationJob::new(exps);
        let report = job.run().await;
        assert_eq!(report.experiences_validated, 1);
        assert_eq!(report.patterns_extracted, 1);
    }

    #[tokio::test]
    async fn test_consolidation_run_empty_experiences() {
        let job = ConsolidationJob::new(vec![]);
        let report = job.run().await;
        assert_eq!(report.experiences_validated, 0);
        assert_eq!(report.patterns_extracted, 0);
        assert_eq!(report.patterns_verified, 0);
    }

    // ── Publishing ──

    #[test]
    fn test_publish_creates_artifact_files() {
        let bp = BestPractice {
            id: Uuid::new_v4(),
            source_pattern: Uuid::new_v4(),
            practice_text: "Always validate before proceeding.".into(),
            version: 1,
            published: false,
            scope: MemoryScope::Team,
        };
        let result = PublishJob::new().publish(&bp);
        assert!(result.published);
        assert_eq!(result.version, 1);
        assert_eq!(result.artifact_type, ArtifactType::Skill);
        assert!(result.files.contains_key("SKILL.md"));
        assert!(result.files.contains_key("agents/openai.yaml"));
        assert!(result
            .files
            .contains_key("references/oris-best-practice.json"));
        assert!(result.files["SKILL.md"].contains("Best Practice v1"));
    }

    #[test]
    fn test_publish_artifact_type_by_scope() {
        let enterprise_bp = make_best_practice(MemoryScope::Enterprise);
        let result = PublishJob::new().publish(&enterprise_bp);
        assert_eq!(result.artifact_type, ArtifactType::Sop);
        assert!(result.files.contains_key("SOP.md"));

        let process_bp = make_best_practice(MemoryScope::Process);
        let result = PublishJob::new().publish(&process_bp);
        assert_eq!(result.artifact_type, ArtifactType::Rule);
        assert!(result.files.contains_key("RULE.md"));

        let team_bp = make_best_practice(MemoryScope::Team);
        let result = PublishJob::new().publish(&team_bp);
        assert_eq!(result.artifact_type, ArtifactType::Skill);
    }

    #[test]
    fn test_publish_includes_version_and_scope() {
        let bp = BestPractice {
            id: Uuid::new_v4(),
            source_pattern: Uuid::new_v4(),
            practice_text: "v3 practice".into(),
            version: 3,
            published: true,
            scope: MemoryScope::Enterprise,
        };
        let result = PublishJob::new().publish(&bp);
        let sop = &result.files["SOP.md"];
        assert!(sop.contains("version: 3"));
        assert!(sop.contains("scope: enterprise"));
        assert!(sop.contains("Best Practice v3"));
    }

    // ── BatchScheduler ──

    #[test]
    fn test_batch_scheduler_new_sets_next_run() {
        let before = Utc::now();
        let scheduler = BatchScheduler::new(Duration::seconds(3600));
        let after = Utc::now();
        assert!(scheduler.next_run >= before + Duration::seconds(3600));
        assert!(scheduler.next_run <= after + Duration::seconds(3600));
        assert!(scheduler.last_run.is_none());
    }

    #[test]
    fn test_batch_scheduler_should_run_false_before_next_run() {
        let now = Utc::now();
        let scheduler = BatchScheduler {
            interval: Duration::seconds(3600),
            last_run: None,
            next_run: now + Duration::seconds(3600),
        };
        assert!(!scheduler.should_run());
    }

    #[test]
    fn test_batch_scheduler_should_run_true_after_next_run() {
        let now = Utc::now();
        let scheduler = BatchScheduler {
            interval: Duration::seconds(3600),
            last_run: None,
            next_run: now - Duration::seconds(3600),
        };
        assert!(scheduler.should_run());
    }

    #[test]
    fn test_batch_scheduler_mark_completed_reschedules() {
        let now = Utc::now();
        let mut scheduler = BatchScheduler {
            interval: Duration::seconds(7200),
            last_run: None,
            next_run: now + Duration::seconds(7200),
        };
        assert!(scheduler.last_run.is_none());
        scheduler.mark_completed();
        assert!(scheduler.last_run.is_some());
        assert!(scheduler.next_run > now);
    }

    #[test]
    fn test_batch_scheduler_time_until_next_run() {
        let now = Utc::now();
        let scheduler = BatchScheduler {
            interval: Duration::seconds(3600),
            last_run: None,
            next_run: now + Duration::seconds(3600),
        };
        let remaining = scheduler.time_until_next_run();
        assert!(remaining > Duration::zero());
        assert!(remaining <= Duration::seconds(3600));

        let past_scheduler = BatchScheduler {
            interval: Duration::seconds(3600),
            last_run: None,
            next_run: now - Duration::seconds(3600),
        };
        assert_eq!(past_scheduler.time_until_next_run(), Duration::zero());
    }

    // ── Text similarity ──

    #[test]
    fn test_text_similarity_identical_content() {
        let s = "The quick brown fox";
        assert!((text_similarity(s, s) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_text_similarity_disjoint_content() {
        let a = "alpha beta";
        let b = "gamma delta";
        assert!((text_similarity(a, b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_text_similarity_partial_overlap() {
        let a = "the quick brown fox";
        let b = "the slow brown cat";
        // shared: the, brown → 2/6 ≈ 0.333
        let sim = text_similarity(a, b);
        assert!((sim - 0.3333).abs() < 0.01);
    }

    #[test]
    fn test_text_similarity_both_empty() {
        assert!((text_similarity("", "") - 1.0).abs() < f64::EPSILON);
    }

    // ── Helpers ──

    fn make_episode(
        tenant: &str,
        content: &str,
        outcome: EpisodeOutcome,
        entities: Vec<String>,
    ) -> Episode {
        Episode {
            id: Uuid::new_v4(),
            tenant_id: tenant.into(),
            user_id: None,
            task_id: None,
            content: content.into(),
            outcome,
            timestamp: Utc::now(),
            entities,
        }
    }

    fn make_cluster(key: &str, episodes: Vec<Episode>) -> EpisodeCluster {
        EpisodeCluster {
            cluster_key: key.into(),
            episodes,
        }
    }

    fn make_experience(
        confidence: f64,
        authority: AuthorityLevel,
        n_episodes: usize,
    ) -> Experience {
        Experience {
            id: Uuid::new_v4(),
            source_episodes: (0..n_episodes).map(|_| Uuid::new_v4()).collect(),
            summary: "test experience".into(),
            applicability_conditions: vec!["cond".into()],
            confidence,
            authority_level: authority,
        }
    }

    fn make_best_practice(scope: MemoryScope) -> BestPractice {
        BestPractice {
            id: Uuid::new_v4(),
            source_pattern: Uuid::new_v4(),
            practice_text: "A best practice for testing.".into(),
            version: 1,
            published: false,
            scope,
        }
    }
}
