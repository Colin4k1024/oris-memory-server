//! Hybrid search: structured filter + keyword (ILIKE) + vector (pgvector cosine).

use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::memory_types::{MatchType, MemoryItem, SearchParams, SearchResult};

/// Encode a float slice as a PostgreSQL vector string literal.
fn encode_vector(v: &[f32]) -> String {
    let inner: Vec<String> = v.iter().map(|f| f.to_string()).collect();
    format!("[{}]", inner.join(","))
}

/// Repository for hybrid search operations.
pub struct SearchRepo {
    pool: PgPool,
}

impl SearchRepo {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Perform hybrid search combining structured filters, keyword matching,
    /// and vector similarity. Results are reranked by a weighted score.
    pub async fn hybrid_search(
        &self,
        params: &SearchParams,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Build the base WHERE clause for structured + tenant filtering
        let mut conditions = vec![
            "tenant_id = $1".to_string(),
            "status = 'active'".to_string(),
        ];
        let mut bind_idx = 2u32;

        // Memory type filter
        if !params.memory_types.is_empty() {
            let types: Vec<String> = params
                .memory_types
                .iter()
                .map(|t| format!("'{}'", t.as_str()))
                .collect();
            conditions.push(format!("memory_type IN ({})", types.join(",")));
        }

        // Scope filter
        if !params.scopes.is_empty() {
            let scopes: Vec<String> = params
                .scopes
                .iter()
                .map(|s| format!("'{}'", s.as_str()))
                .collect();
            conditions.push(format!("scope IN ({})", scopes.join(",")));
        }

        // Subject filter
        if let Some(ref subject_id) = params.subject_id {
            conditions.push(format!("subject_id = ${bind_idx}"));
            bind_idx += 1;
        }

        // Confidence filter
        if let Some(min_conf) = params.min_confidence {
            conditions.push(format!("confidence >= ${bind_idx}"));
            bind_idx += 1;
        }

        // Keyword filter (ILIKE on content)
        let has_keyword = params
            .query_text
            .as_deref()
            .is_some_and(|t| !t.trim().is_empty());
        if has_keyword {
            conditions.push(format!("content ILIKE ${bind_idx}"));
            bind_idx += 1;
        }

        // Validity window filter (exclude expired)
        conditions.push("(valid_to IS NULL OR valid_to > NOW())".to_string());

        let where_clause = conditions.join(" AND ");

        // Vector search path (if embedding provided)
        if let Some(ref embedding) = params.embedding {
            let p_kw = bind_idx;
            let p_kw_w = bind_idx + 1;
            let p_vec = bind_idx + 2;
            let p_vec_w = bind_idx + 3;
            let p_auth_w = bind_idx + 4;
            let p_fresh_w = bind_idx + 5;
            let p_limit = bind_idx + 6;
            let p_offset = bind_idx + 7;

            // Combined keyword + vector search with weighted scoring
            let query_str = format!(
                r#"SELECT *,
                    (CASE WHEN content ILIKE ${p_kw} THEN ${p_kw_w} ELSE 0 END
                     + (1.0 - (embedding <=> ${p_vec}::vector)) * ${p_vec_w}
                     + authority_rank * ${p_auth_w}
                     + (1.0 / (EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 + 1.0)) * ${p_fresh_w}
                    ) AS score,
                    (CASE WHEN content ILIKE ${p_kw} THEN TRUE ELSE FALSE END) AS matched_keyword,
                    TRUE AS matched_vector
                   FROM memory_item
                   CROSS JOIN LATERAL (
                       SELECT CASE authority_level
                           WHEN 'L0_source_of_truth' THEN 1.0
                           WHEN 'L1_authoritative' THEN 0.8
                           WHEN 'L2_verified' THEN 0.6
                           WHEN 'L3_inferred' THEN 0.4
                           ELSE 0.4
                       END AS authority_rank
                   ) AS ar
                   WHERE {where_clause}
                   ORDER BY score DESC
                   LIMIT ${p_limit} OFFSET ${p_offset}"#,
            );

            let mut q = sqlx::query(&query_str).bind(&params.tenant_id);

            if let Some(ref subject_id) = params.subject_id {
                q = q.bind(subject_id);
            }
            if let Some(min_conf) = params.min_confidence {
                q = q.bind(min_conf);
            }
            if has_keyword {
                q = q.bind(format!("%{}%", params.query_text.as_deref().unwrap_or("")));
            }

            q = q
                .bind(format!("%{}%", params.query_text.as_deref().unwrap_or("")))
                .bind(params.keyword_weight)
                .bind(encode_vector(embedding))
                .bind(params.vector_weight)
                .bind(params.authority_weight)
                .bind(params.freshness_weight)
                .bind(params.limit)
                .bind(params.offset);

            let rows = q.fetch_all(&self.pool).await?;
            let results: Vec<SearchResult> = rows
                .iter()
                .map(|row| {
                    let matched_keyword: bool = row.try_get("matched_keyword").unwrap_or(false);
                    let score: f64 = row.try_get("score").unwrap_or(0.0);
                    let item = map_row_brief(row);
                    SearchResult {
                        item,
                        score,
                        matched_by: if matched_keyword {
                            MatchType::Hybrid
                        } else {
                            MatchType::Vector
                        },
                    }
                })
                .collect();
            Ok(results)
        } else if has_keyword {
            let p_kw = bind_idx;
            let p_kw_w = bind_idx + 1;
            let p_auth_w = bind_idx + 2;
            let p_fresh_w = bind_idx + 3;
            let p_limit = bind_idx + 4;
            let p_offset = bind_idx + 5;

            // Keyword-only search
            let query_str = format!(
                r#"SELECT *,
                    (CASE WHEN content ILIKE ${p_kw} THEN ${p_kw_w} ELSE 0 END
                     + authority_rank * ${p_auth_w}
                     + (1.0 / (EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 + 1.0)) * ${p_fresh_w}
                    ) AS score
                   FROM memory_item
                   CROSS JOIN LATERAL (
                       SELECT CASE authority_level
                           WHEN 'L0_source_of_truth' THEN 1.0
                           WHEN 'L1_authoritative' THEN 0.8
                           WHEN 'L2_verified' THEN 0.6
                           WHEN 'L3_inferred' THEN 0.4
                           ELSE 0.4
                       END AS authority_rank
                   ) AS ar
                   WHERE {where_clause}
                   ORDER BY score DESC
                   LIMIT ${p_limit} OFFSET ${p_offset}"#,
            );

            let mut q = sqlx::query(&query_str).bind(&params.tenant_id);

            if let Some(ref subject_id) = params.subject_id {
                q = q.bind(subject_id);
            }
            if let Some(min_conf) = params.min_confidence {
                q = q.bind(min_conf);
            }

            q = q
                .bind(format!("%{}%", params.query_text.as_deref().unwrap_or("")))
                .bind(params.keyword_weight)
                .bind(params.authority_weight)
                .bind(params.freshness_weight)
                .bind(params.limit)
                .bind(params.offset);

            let rows = q.fetch_all(&self.pool).await?;
            Ok(rows
                .iter()
                .map(|row| SearchResult {
                    item: map_row_brief(row),
                    score: row.try_get("score").unwrap_or(0.0),
                    matched_by: MatchType::Keyword,
                })
                .collect())
        } else {
            // Structured-only search
            let p_limit = bind_idx;
            let p_offset = bind_idx + 1;
            let query_str = format!(
                r#"SELECT *,
                    confidence * 0.5 + importance * 0.5 AS score
                   FROM memory_item
                   WHERE {where_clause}
                   ORDER BY score DESC
                   LIMIT ${p_limit} OFFSET ${p_offset}"#,
            );

            let mut q = sqlx::query(&query_str).bind(&params.tenant_id);

            if let Some(ref subject_id) = params.subject_id {
                q = q.bind(subject_id);
            }
            if let Some(min_conf) = params.min_confidence {
                q = q.bind(min_conf);
            }

            q = q.bind(params.limit).bind(params.offset);

            let rows = q.fetch_all(&self.pool).await?;
            Ok(rows
                .iter()
                .map(|row| SearchResult {
                    item: map_row_brief(row),
                    score: row.try_get("score").unwrap_or(0.0),
                    matched_by: MatchType::Structured,
                })
                .collect())
        }
    }
}

/// Lightweight row mapping for search results (avoids full MemoryItem parse).
fn map_row_brief(row: &sqlx::postgres::PgRow) -> MemoryItem {
    use crate::memory_types::*;
    MemoryItem {
        memory_id: row.try_get("memory_id").unwrap_or_default(),
        tenant_id: row.try_get("tenant_id").unwrap_or_default(),
        memory_type: match row
            .try_get::<String, _>("memory_type")
            .unwrap_or_default()
            .as_str()
        {
            "semantic" => MemoryType::Semantic,
            "episodic" => MemoryType::Episodic,
            "decision" => MemoryType::Decision,
            "experience" => MemoryType::Experience,
            _ => MemoryType::UserPreference,
        },
        scope: match row
            .try_get::<String, _>("scope")
            .unwrap_or_default()
            .as_str()
        {
            "personal" => Scope::Personal,
            "agent" => Scope::Agent,
            "task" => Scope::Task,
            "team" => Scope::Team,
            "process" => Scope::Process,
            "factory" => Scope::Factory,
            _ => Scope::Enterprise,
        },
        subject_type: row.try_get("subject_type").unwrap_or(None),
        subject_id: row.try_get("subject_id").unwrap_or(None),
        entity_refs: vec![],
        content: row.try_get("content").unwrap_or(None),
        structured_payload: row.try_get("structured_payload").unwrap_or(None),
        embedding: None,
        source_type: SourceType::AgentInferred,
        source_reference: row.try_get("source_reference").unwrap_or(None),
        evidence_refs: vec![],
        confidence: row.try_get("confidence").unwrap_or(0.5),
        authority_level: match row
            .try_get::<String, _>("authority_level")
            .unwrap_or_default()
            .as_str()
        {
            "L0_source_of_truth" => AuthorityLevel::L0SourceOfTruth,
            "L1_authoritative" => AuthorityLevel::L1Authoritative,
            "L2_verified" => AuthorityLevel::L2Verified,
            _ => AuthorityLevel::L3Inferred,
        },
        importance: row.try_get("importance").unwrap_or(0.5),
        observed_at: row.try_get("observed_at").unwrap_or(None),
        valid_from: row.try_get("valid_from").unwrap_or(None),
        valid_to: row.try_get("valid_to").unwrap_or(None),
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
        acl: row
            .try_get::<serde_json::Value, _>("acl")
            .unwrap_or_default(),
        retention_policy: row.try_get("retention_policy").unwrap_or(None),
        status: match row
            .try_get::<String, _>("status")
            .unwrap_or_default()
            .as_str()
        {
            "candidate" => MemoryStatus::Candidate,
            "archived" => MemoryStatus::Archived,
            "revoked" => MemoryStatus::Revoked,
            "quarantined" => MemoryStatus::Quarantined,
            _ => MemoryStatus::Active,
        },
        version: row.try_get("version").unwrap_or(1),
        derived_from: vec![],
        created_by_user: row.try_get("created_by_user").unwrap_or(None),
        created_by_agent: row.try_get("created_by_agent").unwrap_or(None),
        last_verified_at: row.try_get("last_verified_at").unwrap_or(None),
        created_at: row.try_get("created_at").unwrap_or_default(),
        updated_at: row.try_get("updated_at").unwrap_or_default(),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_params_defaults() {
        let p = SearchParams::new("tenant-1");
        assert_eq!(p.tenant_id, "tenant-1");
        assert_eq!(p.limit, 20);
        assert_eq!(p.vector_weight, 0.4);
    }
}
