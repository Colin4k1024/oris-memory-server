//! Canonical User Memory — unified, deduplicated user profile management.
//!
//! Implements the canonical user model from architecture doc §5.  The
//! canonical profile merges data from IAM, HR, explicit user input, and
//! agent inferences.  Conflicts between sources are resolved by authority
//! level (`L0 > L1 > L2 > L3`) and **recorded**, never silently dropped.
//!
//! Hot-context (Redis) is a non-authoritative read-through cache.  Any write
//! invalidates the cached entry so stale data is never served.

use std::collections::HashSet;

use oris_memory_store::memory_types::{
    AuthorityLevel, CanonicalUserProfile, PrivacyClass, SourceType,
};
use oris_memory_store::postgres::user_repo::{UserRepo, UserRepoError};
use oris_memory_store::postgres::Pool;
use oris_memory_store::redis::hot_context::{HotContextRepo, DEFAULT_TTL_SECS};

/// Manages canonical user context — the unified, deduplicated user profile
/// that merges data from IAM, HR, explicit user input, and agent inferences.
pub struct CanonicalUserManager {
    user_repo: UserRepo,
    hot_context: Option<HotContextRepo>,
}

/// Result of merging multiple identity sources into a single canonical profile.
#[derive(Debug, Clone)]
pub struct MergedUserProfile {
    /// The resolved canonical profile.
    pub profile: CanonicalUserProfile,
    /// Every conflict that was detected during the merge.
    pub conflicts: Vec<ConflictEntry>,
    /// Human-readable names of the sources that contributed.
    pub sources: Vec<String>,
}

/// A single field-level conflict detected during profile merging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictEntry {
    /// Field that had conflicting values (e.g. `"role"`, `"timezone"`).
    pub field: String,
    /// All contributing values paired with their source authority level.
    pub values: Vec<(String, AuthorityLevel)>,
    /// Explanation of which value won and why.
    pub resolution: String,
}

// ── Error ───────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum CanonicalUserError {
    #[error("{0}")]
    Repo(#[from] UserRepoError),
    #[error("cache error: {0}")]
    Cache(String),
}

// ── Manager ─────────────────────────────────────────────────────

impl CanonicalUserManager {
    /// Create a new manager backed by the given PostgreSQL pool.
    pub fn new(pool: Pool) -> Self {
        Self {
            user_repo: UserRepo::new(pool),
            hot_context: None,
        }
    }

    /// Attach a hot-context (Redis) repository.  Builder-style.
    pub fn with_hot_context(mut self, repo: HotContextRepo) -> Self {
        self.hot_context = Some(repo);
        self
    }

    /// Get the canonical user profile.
    ///
    /// Reads from PostgreSQL (authoritative) and populates the hot-context
    /// cache so callers that know the tenant_id benefit from a cache hit on
    /// subsequent reads.  A true cache-first path is not possible here because
    /// the Redis key is scoped by `tenant_id`, which is unavailable from
    /// `user_id` alone.
    pub async fn get_context(
        &self,
        user_id: &str,
    ) -> Result<Option<CanonicalUserProfile>, CanonicalUserError> {
        let profile = self.user_repo.get_by_id(user_id).await?;

        if let Some(ref profile) = profile {
            self.populate_cache(profile).await;
        }

        Ok(profile)
    }

    /// Upsert a user profile and invalidate the hot-context cache.
    pub async fn upsert(&self, profile: &CanonicalUserProfile) -> Result<(), CanonicalUserError> {
        self.user_repo.upsert(profile).await?;

        self.invalidate_cache_best_effort(&profile.user_id, &profile.organization_id)
            .await;
        Ok(())
    }

    /// Merge profiles from different sources (IAM, HR, explicit, inferred).
    ///
    /// Authority level determines the winner for conflicting scalar fields:
    /// `L0_source_of_truth > L1_authoritative > L2_verified > L3_inferred`.
    /// Array fields are unioned (deduplicated).  JSON object fields are merged
    /// key-by-key with higher-authority keys overriding lower ones.  Every
    /// conflict is recorded in the returned [`MergedUserProfile`].
    pub fn merge_profiles(&self, profiles: Vec<CanonicalUserProfile>) -> MergedUserProfile {
        merge_profiles_impl(profiles)
    }

    /// Update user consent scope.
    ///
    /// Hot-context invalidation requires `tenant_id`, which this method does
    /// not receive; the cached entry will expire via TTL (default 5 min).
    /// Callers that know the tenant_id can call [`invalidate_cache`](Self::invalidate_cache)
    /// directly.
    pub async fn update_consent(
        &self,
        user_id: &str,
        consent_scope: serde_json::Value,
    ) -> Result<(), CanonicalUserError> {
        self.user_repo
            .update_consent(user_id, &consent_scope)
            .await?;
        Ok(())
    }

    /// Find a user by an identity link (e.g. SSO ID, employee number).
    pub async fn find_by_identity(
        &self,
        system: &str,
        external_id: &str,
    ) -> Result<Option<CanonicalUserProfile>, CanonicalUserError> {
        Ok(self.user_repo.find_by_identity(system, external_id).await?)
    }

    /// Invalidate the hot-context cache entry for a user.
    pub async fn invalidate_cache(
        &self,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<(), CanonicalUserError> {
        if let Some(ref hc) = self.hot_context {
            hc.invalidate_user_context(user_id, tenant_id)
                .await
                .map_err(|e| CanonicalUserError::Cache(e.to_string()))?;
        }
        Ok(())
    }

    // ── private helpers ──────────────────────────────────────────

    /// Best-effort cache population — logs and swallows Redis errors.
    async fn populate_cache(&self, profile: &CanonicalUserProfile) {
        if let Some(ref hc) = self.hot_context {
            if let Ok(data) = serde_json::to_vec(profile) {
                if let Err(e) = hc
                    .set_user_context(
                        &profile.user_id,
                        &profile.organization_id,
                        &data,
                        DEFAULT_TTL_SECS,
                    )
                    .await
                {
                    tracing::warn!(
                        error = %e,
                        user_id = %profile.user_id,
                        "failed to populate user hot context"
                    );
                }
            }
        }
    }

    /// Best-effort cache invalidation — logs and swallows Redis errors.
    async fn invalidate_cache_best_effort(&self, user_id: &str, tenant_id: &str) {
        if let Some(ref hc) = self.hot_context {
            if let Err(e) = hc.invalidate_user_context(user_id, tenant_id).await {
                tracing::warn!(
                    error = %e,
                    user_id = %user_id,
                    "failed to invalidate user hot context"
                );
            }
        }
    }
}

// ── Merge logic (free function for testability) ─────────────────

/// Merge profiles from different sources into a single canonical profile.
fn merge_profiles_impl(profiles: Vec<CanonicalUserProfile>) -> MergedUserProfile {
    let sources: Vec<String> = profiles
        .iter()
        .map(|p| p.source.as_str().to_string())
        .collect();

    if profiles.is_empty() {
        return MergedUserProfile {
            profile: empty_profile(),
            conflicts: Vec::new(),
            sources,
        };
    }

    if profiles.len() == 1 {
        let profile = profiles.into_iter().next().unwrap();
        return MergedUserProfile {
            profile,
            conflicts: Vec::new(),
            sources,
        };
    }

    // Sort by authority rank descending (most authoritative first).
    let mut sorted: Vec<&CanonicalUserProfile> = profiles.iter().collect();
    sorted.sort_by_key(|p| std::cmp::Reverse(p.authority_level.rank()));

    let mut conflicts = Vec::new();
    let base = sorted[0];

    // Scalar Option<String> fields — highest authority non-None value wins.
    let role = resolve_opt_string("role", &sorted, |p| &p.role, &mut conflicts);
    let position = resolve_opt_string("position", &sorted, |p| &p.position, &mut conflicts);
    let language = resolve_opt_string("language", &sorted, |p| &p.language, &mut conflicts);
    let timezone = resolve_opt_string("timezone", &sorted, |p| &p.timezone, &mut conflicts);

    // JSON fields.
    let preferences = merge_json_object("preferences", &sorted, |p| &p.preferences, &mut conflicts);
    let consent_scope = resolve_json(
        "consent_scope",
        &sorted,
        |p| &p.consent_scope,
        &mut conflicts,
    );

    // Array fields — union with deduplication.
    let identity_links = union_arrays(&sorted, |p| &p.identity_links);
    let common_entities = union_arrays(&sorted, |p| &p.common_entities);
    let active_projects = union_arrays(&sorted, |p| &p.active_projects);

    // Enum / scalar fields taken from the highest-authority profile.
    let privacy_class = base.privacy_class;
    let source = base.source;
    let authority_level = base.authority_level;

    // Version: max across all profiles.
    let version = profiles.iter().map(|p| p.version).max().unwrap_or(0);

    // Timestamps: valid_from / valid_to / last_verified_at from base;
    // updated_at: latest across all profiles.
    let updated_at = profiles
        .iter()
        .map(|p| p.updated_at)
        .max()
        .unwrap_or(base.updated_at);

    let merged = CanonicalUserProfile {
        user_id: base.user_id.clone(),
        organization_id: base.organization_id.clone(),
        factory_id: base.factory_id.clone(),
        identity_links,
        role,
        position,
        language,
        timezone,
        preferences,
        common_entities,
        active_projects,
        consent_scope,
        privacy_class,
        source,
        authority_level,
        version,
        valid_from: base.valid_from,
        valid_to: base.valid_to,
        last_verified_at: base.last_verified_at,
        updated_at,
    };

    MergedUserProfile {
        profile: merged,
        conflicts,
        sources,
    }
}

/// Create a minimal placeholder profile for the empty-merge case.
fn empty_profile() -> CanonicalUserProfile {
    CanonicalUserProfile {
        user_id: String::new(),
        organization_id: String::new(),
        factory_id: None,
        identity_links: Vec::new(),
        role: None,
        position: None,
        language: None,
        timezone: None,
        preferences: serde_json::Value::Null,
        common_entities: Vec::new(),
        active_projects: Vec::new(),
        consent_scope: serde_json::Value::Null,
        privacy_class: PrivacyClass::Internal,
        source: SourceType::UserExplicit,
        authority_level: AuthorityLevel::L3Inferred,
        version: 0,
        valid_from: None,
        valid_to: None,
        last_verified_at: None,
        updated_at: chrono::Utc::now(),
    }
}

/// Resolve an `Option<String>` field across sorted profiles.
///
/// `profiles` must be sorted by authority rank descending.  Returns the
/// highest-authority non-`None` value.  If two or more profiles disagree on
/// the value, a [`ConflictEntry`] is pushed onto `conflicts`.
fn resolve_opt_string(
    field: &str,
    profiles: &[&CanonicalUserProfile],
    get: impl Fn(&CanonicalUserProfile) -> &Option<String>,
    conflicts: &mut Vec<ConflictEntry>,
) -> Option<String> {
    let entries: Vec<(&str, AuthorityLevel)> = profiles
        .iter()
        .filter_map(|p| get(p).as_deref().map(|v| (v, p.authority_level)))
        .collect();

    if entries.is_empty() {
        return None;
    }

    let unique: HashSet<&str> = entries.iter().map(|(v, _)| *v).collect();
    if unique.len() > 1 {
        let (winner_val, winner_auth) = entries[0];
        conflicts.push(ConflictEntry {
            field: field.to_string(),
            values: entries.iter().map(|(v, a)| (v.to_string(), *a)).collect(),
            resolution: format!(
                "Resolved to '{winner_val}' from {auth} — highest authority wins",
                auth = winner_auth.as_str()
            ),
        });
    }

    Some(entries[0].0.to_string())
}

/// Resolve a JSON field — highest authority value wins.
fn resolve_json(
    field: &str,
    profiles: &[&CanonicalUserProfile],
    get: impl Fn(&CanonicalUserProfile) -> &serde_json::Value,
    conflicts: &mut Vec<ConflictEntry>,
) -> serde_json::Value {
    let serialized: Vec<String> = profiles.iter().map(|p| get(p).to_string()).collect();
    let unique: HashSet<&str> = serialized.iter().map(|s| s.as_str()).collect();

    if unique.len() > 1 {
        let winner_auth = profiles[0].authority_level;
        conflicts.push(ConflictEntry {
            field: field.to_string(),
            values: serialized
                .iter()
                .zip(profiles.iter())
                .map(|(v, p)| (v.clone(), p.authority_level))
                .collect(),
            resolution: format!(
                "Resolved to value from {auth} — highest authority wins",
                auth = winner_auth.as_str()
            ),
        });
    }

    get(profiles[0]).clone()
}

/// Merge JSON object fields — higher-authority keys override lower ones.
fn merge_json_object(
    field: &str,
    profiles: &[&CanonicalUserProfile],
    get: impl Fn(&CanonicalUserProfile) -> &serde_json::Value,
    conflicts: &mut Vec<ConflictEntry>,
) -> serde_json::Value {
    // Iterate from lowest to highest authority so higher authority overrides.
    let mut merged = serde_json::Map::new();
    for p in profiles.iter().rev() {
        let val = get(p);
        if let Some(obj) = val.as_object() {
            for (k, v) in obj {
                merged.insert(k.clone(), v.clone());
            }
        } else if !val.is_null() {
            // Non-object value: replace entirely with this source's value.
            return val.clone();
        }
    }

    // Detect conflicts.
    let serialized: Vec<String> = profiles.iter().map(|p| get(p).to_string()).collect();
    let unique: HashSet<&str> = serialized.iter().map(|s| s.as_str()).collect();
    if unique.len() > 1 {
        conflicts.push(ConflictEntry {
            field: field.to_string(),
            values: serialized
                .iter()
                .zip(profiles.iter())
                .map(|(v, p)| (v.clone(), p.authority_level))
                .collect(),
            resolution: "Merged key-by-key — higher-authority keys override".to_string(),
        });
    }

    serde_json::Value::Object(merged)
}

/// Union arrays across all profiles, deduplicating by serialized value.
fn union_arrays(
    profiles: &[&CanonicalUserProfile],
    get: impl Fn(&CanonicalUserProfile) -> &Vec<serde_json::Value>,
) -> Vec<serde_json::Value> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut result = Vec::new();
    for p in profiles {
        for v in get(p) {
            let key = v.to_string();
            if seen.insert(key) {
                result.push(v.clone());
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(
        user_id: &str,
        authority: AuthorityLevel,
        source: SourceType,
    ) -> CanonicalUserProfile {
        CanonicalUserProfile {
            user_id: user_id.to_string(),
            organization_id: "acme".to_string(),
            factory_id: None,
            identity_links: Vec::new(),
            role: None,
            position: None,
            language: None,
            timezone: None,
            preferences: serde_json::Value::Null,
            common_entities: Vec::new(),
            active_projects: Vec::new(),
            consent_scope: serde_json::Value::Null,
            privacy_class: PrivacyClass::Internal,
            source,
            authority_level: authority,
            version: 1,
            valid_from: None,
            valid_to: None,
            last_verified_at: None,
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn merge_single_profile_has_no_conflicts() {
        let p = make_profile("u1", AuthorityLevel::L1Authoritative, SourceType::Iam);
        let result = merge_profiles_impl(vec![p]);

        assert!(result.conflicts.is_empty());
        assert_eq!(result.profile.user_id, "u1");
        assert_eq!(result.sources, vec!["iam".to_string()]);
    }

    #[test]
    fn merge_higher_authority_wins_for_role() {
        let mut inferred =
            make_profile("u1", AuthorityLevel::L3Inferred, SourceType::AgentInferred);
        inferred.role = Some("engineer".into());

        let mut explicit = make_profile(
            "u1",
            AuthorityLevel::L0SourceOfTruth,
            SourceType::UserExplicit,
        );
        explicit.role = Some("director".into());

        let result = merge_profiles_impl(vec![inferred, explicit]);

        assert_eq!(result.profile.role.as_deref(), Some("director"));
        assert!(result.conflicts.iter().any(|c| c.field == "role"));
    }

    #[test]
    fn merge_same_authority_different_values_records_conflict() {
        let mut a = make_profile("u1", AuthorityLevel::L2Verified, SourceType::Hr);
        a.timezone = Some("UTC".into());

        let mut b = make_profile("u1", AuthorityLevel::L2Verified, SourceType::Iam);
        b.timezone = Some("Asia/Shanghai".into());

        let result = merge_profiles_impl(vec![a, b]);

        let tz_conflict = result
            .conflicts
            .iter()
            .find(|c| c.field == "timezone")
            .expect("timezone conflict should be recorded");
        assert_eq!(tz_conflict.values.len(), 2);
        assert!(tz_conflict.resolution.contains("highest authority wins"));
    }

    #[test]
    fn merge_arrays_are_unioned_and_deduplicated() {
        let mut a = make_profile("u1", AuthorityLevel::L1Authoritative, SourceType::Iam);
        a.identity_links = vec![
            serde_json::json!({"system": "sso", "id": "123"}),
            serde_json::json!({"system": "email", "id": "a@b.com"}),
        ];

        let mut b = make_profile("u1", AuthorityLevel::L2Verified, SourceType::Hr);
        b.identity_links = vec![
            serde_json::json!({"system": "sso", "id": "123"}), // duplicate
            serde_json::json!({"system": "ldap", "id": "u1"}),
        ];

        let result = merge_profiles_impl(vec![a, b]);

        assert_eq!(result.profile.identity_links.len(), 3);
    }

    #[test]
    fn merge_preferences_object_merges_keys() {
        let mut a = make_profile("u1", AuthorityLevel::L3Inferred, SourceType::AgentInferred);
        a.preferences = serde_json::json!({"theme": "dark", "lang": "en"});

        let mut b = make_profile(
            "u1",
            AuthorityLevel::L0SourceOfTruth,
            SourceType::UserExplicit,
        );
        b.preferences = serde_json::json!({"lang": "zh", "tz": "UTC"});

        let result = merge_profiles_impl(vec![a, b]);

        let prefs = result.profile.preferences.as_object().unwrap();
        assert_eq!(prefs.get("theme").and_then(|v| v.as_str()), Some("dark"));
        assert_eq!(prefs.get("lang").and_then(|v| v.as_str()), Some("zh")); // L0 overrides
        assert_eq!(prefs.get("tz").and_then(|v| v.as_str()), Some("UTC"));
    }

    #[test]
    fn merge_empty_profiles_returns_default() {
        let result = merge_profiles_impl(vec![]);

        assert!(result.conflicts.is_empty());
        assert!(result.sources.is_empty());
        assert!(result.profile.user_id.is_empty());
        assert_eq!(result.profile.authority_level, AuthorityLevel::L3Inferred);
    }
}
