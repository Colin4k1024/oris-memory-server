//! pgvector HNSW vector search engine.
//!
//! Replaces the legacy `hashed_cosine` fake semantic search with real
//! pgvector cosine similarity search using HNSW indexes.  The engine
//! combines a pre-filter (tenant/scope/status/time/entity) with vector
//! similarity to produce ranked results.
//!
//! ## Two search paths
//!
//! | Path | Function | Use case |
//! |------|----------|----------|
//! | DB-side | [`VectorSearchEngine::search`] | Production — uses pgvector HNSW index |
//! | In-memory | [`rank_by_similarity`] | Testing — pure Rust cosine ranking |
//!
//! The in-memory path exists so that ranking logic can be unit-tested
//! without a live PostgreSQL + pgvector instance.

use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::sync::OnceLock;
use uuid::Uuid;

use crate::embedding::{cosine_similarity, encode_vector, EmbeddingProvider};
use crate::pre_filter::{FilterParam, PreFilterBuilder, PreFilterResult};

// ──────────────────────────── Error ────────────────────────────────

/// Errors from vector search operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorSearchError {
    /// Embedding generation failed.
    #[error("embedding error: {0}")]
    Embedding(#[from] crate::embedding::EmbeddingError),
    /// Database query failed.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    /// Pre-filter validation failed.
    #[error("pre-filter error: {0}")]
    PreFilter(String),
}

// ──────────────────────────── HnswConfig ───────────────────────────

/// Configuration for the pgvector HNSW index.
///
/// These values match the `WITH (m = 16, ef_construction = 64)` defaults
/// in the schema DDL and the recommended `ef_search` for recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Max number of connections per node (index build-time).
    pub m: usize,
    /// Search width during index construction.
    pub ef_construction: usize,
    /// Search width at query time (higher = better recall, slower).
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 64,
            ef_search: 40,
        }
    }
}

impl HnswConfig {
    /// Generate the `CREATE INDEX` SQL for this HNSW config.
    pub fn index_sql(&self, table: &str, column: &str) -> String {
        format!(
            "CREATE INDEX IF NOT EXISTS idx_{table}_embedding \
             ON {table} USING hnsw ({column} vector_cosine_ops) \
             WITH (m = {m}, ef_construction = {ef_construction})",
            table = table,
            column = column,
            m = self.m,
            ef_construction = self.ef_construction,
        )
    }
}

// ──────────────────────────── Types ────────────────────────────────

/// A candidate memory with its stored embedding, for in-memory ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCandidate {
    pub memory_id: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
}

/// A ranked search result with cosine similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub memory_id: Uuid,
    pub content: String,
    /// Cosine similarity (0.0–1.0 for normalised vectors).
    pub similarity: f64,
    /// 1-based rank (1 = most similar).
    pub rank: usize,
}

// ──────────────────────────── Engine ───────────────────────────────

/// pgvector HNSW vector search engine.
///
/// Combines a pre-filter (built by [`PreFilterBuilder`]) with cosine
/// similarity search against a `memory_item` table that has an HNSW
/// index on its `embedding` column.
pub struct VectorSearchEngine<P: EmbeddingProvider> {
    pool: PgPool,
    embedding_provider: P,
    hnsw_config: HnswConfig,
}

impl<P: EmbeddingProvider> VectorSearchEngine<P> {
    /// Create a new engine with default HNSW config.
    pub fn new(pool: PgPool, embedding_provider: P) -> Self {
        Self {
            pool,
            embedding_provider,
            hnsw_config: HnswConfig::default(),
        }
    }

    /// Create a new engine with custom HNSW config.
    pub fn with_hnsw_config(pool: PgPool, embedding_provider: P, config: HnswConfig) -> Self {
        Self {
            pool,
            embedding_provider,
            hnsw_config: config,
        }
    }

    /// Execute a pgvector HNSW cosine search.
    ///
    /// Logical flow (§8.5 Bounded Retrieval):
    /// 1. Build pre-filter SQL (tenant/scope/status/time/entity).
    /// 2. Embed the query text via the embedding provider.
    /// 3. Execute a single SQL query combining pre-filter + HNSW vector
    ///    search — the WHERE clause pre-filters candidate IDs, and
    ///    `ORDER BY embedding <=> $1::vector` ranks by cosine distance.
    /// 4. Return ranked results with similarity scores.
    ///
    /// The SQL pattern (cosine distance via `<=>`):
    /// ```sql
    /// SELECT memory_id, content,
    ///        1 - (embedding <=> $1::vector) AS similarity
    /// FROM memory_item
    /// WHERE <pre_filter>
    /// ORDER BY embedding <=> $1::vector
    /// LIMIT $k
    /// ```
    pub async fn search(
        &self,
        query_text: &str,
        pre_filter: PreFilterBuilder,
    ) -> Result<Vec<VectorSearchResult>, VectorSearchError> {
        // 1 — Validate pre-filter
        pre_filter
            .validate()
            .map_err(|e| VectorSearchError::PreFilter(e.to_string()))?;

        // 2 — Build pre-filter SQL + params
        let filter = pre_filter.build();

        // 3 — Embed query
        let query_embedding = self.embedding_provider.embed(query_text).await?;

        // 4 — Execute pgvector search
        let rows = self
            .execute_vector_search(&query_embedding, &filter)
            .await?;

        // 5 — Map to results with 1-based ranks
        Ok(rows
            .into_iter()
            .enumerate()
            .map(|(i, row)| VectorSearchResult {
                memory_id: row.memory_id,
                content: row.content,
                similarity: row.similarity,
                rank: i + 1,
            })
            .collect())
    }

    /// Low-level pgvector search execution within a transaction.
    ///
    /// Sets `hnsw.ef_search` for the transaction, then executes the
    /// combined pre-filter + vector search query.
    async fn execute_vector_search(
        &self,
        query_embedding: &[f32],
        filter: &PreFilterResult,
    ) -> Result<Vec<SearchRow>, VectorSearchError> {
        let vec_literal = encode_vector(query_embedding);

        // Shift pre-filter param indices by 1 (since $1 is the vector).
        let shifted_where = reindex_params(&filter.where_clause, 1);
        let limit_idx = filter.params.len() + 2; // +1 for vector, +1 for limit

        let sql = format!(
            r#"SELECT memory_id,
                      content,
                      1 - (embedding <=> $1::vector) AS similarity
               FROM memory_item
               WHERE {shifted_where}
               ORDER BY embedding <=> $1::vector
               LIMIT ${limit_idx}"#
        );

        // Begin transaction for SET LOCAL hnsw.ef_search
        let mut tx = self.pool.begin().await?;

        // Set HNSW ef_search for this transaction (usize → no injection risk)
        let ef = self.hnsw_config.ef_search;
        sqlx::query(&format!("SET LOCAL hnsw.ef_search = {ef}"))
            .execute(&mut *tx)
            .await?;

        // Build query: $1 = vector, $2.. = pre-filter params, $N = limit
        let mut q = sqlx::query(&sql).bind(vec_literal);

        for param in &filter.params {
            q = match param {
                FilterParam::Text(t) => q.bind(t.clone()),
                FilterParam::Uuid(u) => q.bind(*u),
                FilterParam::I64(i) => q.bind(*i),
                FilterParam::Array(a) => q.bind(a.clone()),
            };
        }

        q = q.bind(filter.top_k as i64); // LIMIT

        let rows = q.fetch_all(&mut *tx).await?;
        tx.commit().await?;

        // Map rows
        Ok(rows
            .iter()
            .map(|row| SearchRow {
                memory_id: row.try_get::<Uuid, _>("memory_id").unwrap_or_default(),
                content: row
                    .try_get::<Option<String>, _>("content")
                    .unwrap_or(None)
                    .unwrap_or_default(),
                similarity: row.try_get::<f64, _>("similarity").unwrap_or(0.0),
            })
            .collect())
    }

    /// In-memory search: embeds the query and ranks candidates by cosine
    /// similarity.  Useful for testing without a live database.
    pub async fn search_in_memory(
        &self,
        query_text: &str,
        candidates: Vec<VectorCandidate>,
        top_k: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorSearchError> {
        let query_embedding = self.embedding_provider.embed(query_text).await?;
        Ok(rank_by_similarity(&query_embedding, candidates, top_k))
    }
}

// ──────────────────────────── Helpers ──────────────────────────────

/// Internal row representation after SQL execution.
struct SearchRow {
    memory_id: Uuid,
    content: String,
    similarity: f64,
}

/// Re-index positional SQL parameters (`$N` → `$(N + shift)`).
///
/// The [`PreFilterBuilder`] generates params starting at `$1`, but in the
/// vector search SQL `$1` is reserved for the query embedding.  This shifts
/// all pre-filter params up by `shift`.
fn reindex_params(where_clause: &str, shift: usize) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\$(\d+)").unwrap());
    re.replace_all(where_clause, |caps: &regex::Captures| {
        let n: usize = caps[1].parse().unwrap_or(0);
        format!("${}", n + shift)
    })
    .to_string()
}

/// Rank candidates by cosine similarity to the query embedding.
///
/// Pure function — no database required.  Sorts by descending similarity
/// (stable — ties preserve original order), truncates to `top_k`, and
/// assigns 1-based ranks.
pub fn rank_by_similarity(
    query_embedding: &[f32],
    candidates: Vec<VectorCandidate>,
    top_k: usize,
) -> Vec<VectorSearchResult> {
    if top_k == 0 || candidates.is_empty() {
        return Vec::new();
    }

    // Score each candidate, keeping original index for stable tie-breaking.
    let mut scored: Vec<(usize, f64, VectorCandidate)> = candidates
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let sim = cosine_similarity(query_embedding, &c.embedding);
            (i, sim, c)
        })
        .collect();

    // Stable sort by descending similarity (ties keep original order).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(rank, (_, sim, c))| VectorSearchResult {
            memory_id: c.memory_id,
            content: c.content,
            similarity: sim,
            rank: rank + 1,
        })
        .collect()
}

// ──────────────────────────── Tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{embed_text, MockEmbeddingProvider, EMBEDDING_DIM_3_SMALL};

    fn make_candidate(text: &str, id: Uuid) -> VectorCandidate {
        VectorCandidate {
            memory_id: id,
            content: text.to_string(),
            embedding: embed_text(text, EMBEDDING_DIM_3_SMALL),
        }
    }

    // 1 — Correct ranking order (most similar first)
    #[tokio::test]
    async fn rank_correct_order() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("deploy kubernetes service").await.unwrap();
        let relevant_id = Uuid::new_v4();
        let irrelevant_id = Uuid::new_v4();
        let c1 = make_candidate("deploy service to kubernetes cluster", relevant_id);
        let c2 = make_candidate("cook pasta with tomato sauce", irrelevant_id);
        let results = rank_by_similarity(&query, vec![c2, c1], 10);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].memory_id, relevant_id,
            "most relevant should rank first"
        );
        assert!(results[0].similarity > results[1].similarity);
    }

    // 2 — Top-k limit truncates results
    #[tokio::test]
    async fn rank_respects_top_k() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("test query").await.unwrap();
        let candidates: Vec<VectorCandidate> = (0..10)
            .map(|i| make_candidate(&format!("doc number {i}"), Uuid::new_v4()))
            .collect();
        let results = rank_by_similarity(&query, candidates, 3);
        assert_eq!(results.len(), 3);
    }

    // 3 — Empty candidates → empty results
    #[tokio::test]
    async fn rank_empty_candidates() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("test").await.unwrap();
        let results = rank_by_similarity(&query, vec![], 10);
        assert!(results.is_empty());
    }

    // 4 — top_k=0 → empty results
    #[tokio::test]
    async fn rank_top_k_zero() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("test").await.unwrap();
        let c = make_candidate("test doc", Uuid::new_v4());
        let results = rank_by_similarity(&query, vec![c], 0);
        assert!(results.is_empty());
    }

    // 5 — Correct 1-based ranks
    #[tokio::test]
    async fn rank_correct_ranks() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("alpha beta").await.unwrap();
        let candidates: Vec<VectorCandidate> = (0..5)
            .map(|i| make_candidate(&format!("alpha beta gamma {i}"), Uuid::new_v4()))
            .collect();
        let results = rank_by_similarity(&query, candidates, 5);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1, "rank should be 1-based");
        }
    }

    // 6 — Tie-breaking preserves original order
    #[test]
    fn rank_tie_breaking() {
        // Candidates with identical embeddings (same text → same vector)
        let emb = embed_text("identical text", 256);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let candidates = vec![
            VectorCandidate {
                memory_id: id1,
                content: "a".into(),
                embedding: emb.clone(),
            },
            VectorCandidate {
                memory_id: id2,
                content: "b".into(),
                embedding: emb.clone(),
            },
            VectorCandidate {
                memory_id: id3,
                content: "c".into(),
                embedding: emb,
            },
        ];
        let query = embed_text("query", 256);
        let results = rank_by_similarity(&query, candidates, 10);
        assert_eq!(results[0].memory_id, id1);
        assert_eq!(results[1].memory_id, id2);
        assert_eq!(results[2].memory_id, id3);
    }

    // 7 — top_k exceeds candidate count → return all
    #[tokio::test]
    async fn rank_top_k_exceeds_candidates() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("query").await.unwrap();
        let candidates = vec![
            make_candidate("doc one", Uuid::new_v4()),
            make_candidate("doc two", Uuid::new_v4()),
        ];
        let results = rank_by_similarity(&query, candidates, 100);
        assert_eq!(results.len(), 2);
    }

    // 8 — Zero query embedding → all candidates get 0.0 similarity
    #[test]
    fn rank_zero_query_embedding() {
        let query = vec![0.0f32; 256];
        let candidates = vec![
            make_candidate("some text", Uuid::new_v4()),
            make_candidate("other text", Uuid::new_v4()),
        ];
        let results = rank_by_similarity(&query, candidates, 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.similarity == 0.0));
    }

    // 9 — HnswConfig defaults match schema DDL
    #[test]
    fn hnsw_config_defaults() {
        let cfg = HnswConfig::default();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construction, 64);
        assert_eq!(cfg.ef_search, 40);
    }

    // 10 — reindex_params shifts $N by the given offset
    #[test]
    fn reindex_shifts_indices() {
        let input = "tenant_id = $1 AND scope = ANY($2) AND status = $3";
        let output = reindex_params(input, 1);
        assert_eq!(output, "tenant_id = $2 AND scope = ANY($3) AND status = $4");
    }

    // 11 — reindex_params with shift=0 is identity
    #[test]
    fn reindex_zero_shift_identity() {
        let input = "a = $1 AND b = $2";
        assert_eq!(reindex_params(input, 0), input);
    }

    // 12 — HnswConfig::index_sql generates valid CREATE INDEX
    #[test]
    fn hnsw_index_sql() {
        let cfg = HnswConfig::default();
        let sql = cfg.index_sql("memory_item", "embedding");
        assert!(sql.contains("USING hnsw"));
        assert!(sql.contains("vector_cosine_ops"));
        assert!(sql.contains("m = 16"));
        assert!(sql.contains("ef_construction = 64"));
    }

    // 13 — VectorSearchResult serializes and deserializes
    #[test]
    fn result_serialization() {
        let result = VectorSearchResult {
            memory_id: Uuid::new_v4(),
            content: "test content".into(),
            similarity: 0.95,
            rank: 1,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: VectorSearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "test content");
        assert!((back.similarity - 0.95).abs() < 1e-9);
        assert_eq!(back.rank, 1);
    }

    // 14 — search_in_memory integration (embed + rank)
    #[tokio::test]
    async fn search_in_memory_integration() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("deploy kubernetes service").await.unwrap();
        let relevant_id = Uuid::new_v4();
        let irrelevant_id = Uuid::new_v4();
        let candidates = vec![
            make_candidate("cook pasta with tomato sauce", irrelevant_id),
            make_candidate("deploy service to kubernetes cluster", relevant_id),
        ];
        let results = rank_by_similarity(&query, candidates, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].memory_id, relevant_id);
        assert!(results[0].similarity > results[1].similarity);
    }

    // 15 — Results are sorted by descending similarity
    #[tokio::test]
    async fn results_sorted_descending() {
        let provider = MockEmbeddingProvider::default();
        let query = provider.embed("alpha beta gamma").await.unwrap();
        let candidates: Vec<VectorCandidate> = (0..8)
            .map(|i| make_candidate(&format!("alpha beta gamma delta {i}"), Uuid::new_v4()))
            .collect();
        let results = rank_by_similarity(&query, candidates, 8);
        for w in results.windows(2) {
            assert!(
                w[0].similarity >= w[1].similarity,
                "similarity must be non-increasing: {} >= {}",
                w[0].similarity,
                w[1].similarity
            );
        }
    }
}
