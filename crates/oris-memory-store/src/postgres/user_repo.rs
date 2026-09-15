//! CRUD for the `canonical_user_profile` table.

use crate::memory_types::{AuthorityLevel, CanonicalUserProfile, PrivacyClass, SourceType};
use sqlx::{PgPool, Row};

pub struct UserRepo {
    pool: PgPool,
}

impl UserRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Upsert a user profile (insert or update with version bump).
    pub async fn upsert(&self, profile: &CanonicalUserProfile) -> Result<(), UserRepoError> {
        sqlx::query(
            r#"INSERT INTO canonical_user_profile (
                user_id, organization_id, factory_id, identity_links, role, position,
                language, timezone, preferences, explicit_preferences, inferred_preferences, common_entities, active_projects,
                consent_scope, privacy_class, source, authority_level, version,
                valid_from, valid_to, last_verified_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
                $18, $19, $20, $21, NOW()
            )
            ON CONFLICT (user_id) DO UPDATE SET
                organization_id = EXCLUDED.organization_id,
                factory_id = EXCLUDED.factory_id,
                identity_links = EXCLUDED.identity_links,
                role = EXCLUDED.role,
                position = EXCLUDED.position,
                language = EXCLUDED.language,
                timezone = EXCLUDED.timezone,
                preferences = EXCLUDED.preferences,
                explicit_preferences = EXCLUDED.explicit_preferences,
                inferred_preferences = EXCLUDED.inferred_preferences,
                common_entities = EXCLUDED.common_entities,
                active_projects = EXCLUDED.active_projects,
                consent_scope = EXCLUDED.consent_scope,
                privacy_class = EXCLUDED.privacy_class,
                source = EXCLUDED.source,
                authority_level = EXCLUDED.authority_level,
                version = canonical_user_profile.version + 1,
                valid_from = EXCLUDED.valid_from,
                valid_to = EXCLUDED.valid_to,
                last_verified_at = EXCLUDED.last_verified_at,
                updated_at = NOW()
            "#,
        )
        .bind(&profile.user_id)
        .bind(&profile.organization_id)
        .bind(&profile.factory_id)
        .bind(serde_json::to_value(&profile.identity_links).unwrap_or_default())
        .bind(&profile.role)
        .bind(&profile.position)
        .bind(&profile.language)
        .bind(&profile.timezone)
        .bind(&profile.preferences)
        .bind(&profile.explicit_preferences)
        .bind(&profile.inferred_preferences)
        .bind(serde_json::to_value(&profile.common_entities).unwrap_or_default())
        .bind(serde_json::to_value(&profile.active_projects).unwrap_or_default())
        .bind(&profile.consent_scope)
        .bind(profile.privacy_class.as_str())
        .bind(profile.source.as_str())
        .bind(profile.authority_level.as_str())
        .bind(profile.version)
        .bind(profile.valid_from)
        .bind(profile.valid_to)
        .bind(profile.last_verified_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Fetch a user profile by user_id.
    pub async fn get_by_id(
        &self,
        user_id: &str,
    ) -> Result<Option<CanonicalUserProfile>, UserRepoError> {
        let row = sqlx::query(r#"SELECT * FROM canonical_user_profile WHERE user_id = $1"#)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await?;

        row.map(|r| map_row_to_profile(&r)).transpose()
    }

    /// Find a user by an identity link (external system + external_id).
    /// Uses JSONB containment query on identity_links.
    pub async fn find_by_identity(
        &self,
        system: &str,
        external_id: &str,
    ) -> Result<Option<CanonicalUserProfile>, UserRepoError> {
        let link = serde_json::json!({"system": system, "external_id": external_id});
        let row = sqlx::query(
            r#"SELECT * FROM canonical_user_profile
               WHERE identity_links @> $1::jsonb"#,
        )
        .bind(link)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|r| map_row_to_profile(&r)).transpose()
    }

    /// Update consent scope (for privacy management).
    pub async fn update_consent(
        &self,
        user_id: &str,
        consent_scope: &serde_json::Value,
    ) -> Result<(), UserRepoError> {
        let result = sqlx::query(
            r#"UPDATE canonical_user_profile
               SET consent_scope = $2, version = version + 1, updated_at = NOW()
               WHERE user_id = $1"#,
        )
        .bind(user_id)
        .bind(consent_scope)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(UserRepoError::NotFound);
        }
        Ok(())
    }
}

fn map_row_to_profile(row: &sqlx::postgres::PgRow) -> Result<CanonicalUserProfile, UserRepoError> {
    Ok(CanonicalUserProfile {
        user_id: row.try_get("user_id")?,
        organization_id: row.try_get("organization_id")?,
        factory_id: row.try_get("factory_id")?,
        identity_links: row
            .try_get::<serde_json::Value, _>("identity_links")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        role: row.try_get("role")?,
        position: row.try_get("position")?,
        language: row.try_get("language")?,
        timezone: row.try_get("timezone")?,
        preferences: row.try_get("preferences").unwrap_or_default(),
        explicit_preferences: row.try_get("explicit_preferences").unwrap_or_default(),
        inferred_preferences: row.try_get("inferred_preferences").unwrap_or_default(),
        common_entities: row
            .try_get::<serde_json::Value, _>("common_entities")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        active_projects: row
            .try_get::<serde_json::Value, _>("active_projects")
            .unwrap_or_default()
            .as_array()
            .cloned()
            .unwrap_or_default(),
        consent_scope: row.try_get("consent_scope").unwrap_or_default(),
        privacy_class: match row
            .try_get::<String, _>("privacy_class")
            .unwrap_or_default()
            .as_str()
        {
            "public" => PrivacyClass::Public,
            "confidential" => PrivacyClass::Confidential,
            "restricted" => PrivacyClass::Restricted,
            _ => PrivacyClass::Internal,
        },
        source: match row
            .try_get::<String, _>("source")
            .unwrap_or_default()
            .as_str()
        {
            "iam" => SourceType::Iam,
            "hr" => SourceType::Hr,
            "agent_inferred" => SourceType::AgentInferred,
            "tool_result" => SourceType::ToolResult,
            "business_event" => SourceType::BusinessEvent,
            _ => SourceType::UserExplicit,
        },
        authority_level: match row
            .try_get::<String, _>("authority_level")
            .unwrap_or_default()
            .as_str()
        {
            "L0_source_of_truth" => AuthorityLevel::L0SourceOfTruth,
            "L2_verified" => AuthorityLevel::L2Verified,
            "L3_inferred" => AuthorityLevel::L3Inferred,
            _ => AuthorityLevel::L1Authoritative,
        },
        version: row.try_get("version")?,
        valid_from: row.try_get("valid_from")?,
        valid_to: row.try_get("valid_to")?,
        last_verified_at: row.try_get("last_verified_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum UserRepoError {
    #[error("user profile not found")]
    NotFound,
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}
