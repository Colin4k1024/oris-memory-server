//! Embedding service — trait + mock implementation.
//!
//! Replaces the legacy `hashed_cosine` fake semantic search in
//! `control_plane.rs` with a pluggable embedding provider abstraction.
//! The [`EmbeddingProvider`] trait decouples vector generation from the
//! search engine, allowing production deployments to swap in an OpenAI
//! (or other) embedding API while tests use the deterministic
//! [`MockEmbeddingProvider`].
//!
//! ## MockEmbeddingProvider
//!
//! Uses consistent feature hashing: each token activates a fixed set of
//! dimensions via FNV-1a hashing with sign randomisation.  Same text
//! always produces the same unit-length vector, and texts sharing tokens
//! have measurably higher cosine similarity.  This is **not** a
//! production-grade embedding — it exists for testing and local dev.
//!
//! ## OpenAIEmbeddingConfig
//!
//! Configuration-only struct for the OpenAI embeddings API.  The actual
//! HTTP call is intentionally omitted; a future `OpenAIEmbeddingProvider`
//! can consume this config.

use serde::{Deserialize, Serialize};

// ──────────────────────────── Constants ────────────────────────────

/// Default embedding dimensions for `text-embedding-3-small`.
pub const EMBEDDING_DIM_3_SMALL: usize = 1536;

/// Number of hash bands — each token activates this many dimensions.
const HASH_BANDS: usize = 8;

/// FNV-1a offset basis.
const FNV_OFFSET: u64 = 1469598103934665603;

/// FNV-1a prime.
const FNV_PRIME: u64 = 1099511628211;

// ──────────────────────────── Error ────────────────────────────────

/// Errors from embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// The input text was empty or whitespace-only.
    #[error("empty input text")]
    EmptyInput,
    /// The embedding service returned an error.
    #[error("embedding service error: {0}")]
    Service(String),
}

// ──────────────────────────── Trait ────────────────────────────────

/// Pluggable embedding provider.
///
/// Implementations convert text into a fixed-dimension `f32` vector
/// suitable for cosine similarity comparison or pgvector storage.
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a text string into a dense vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// The dimensionality of vectors produced by this provider.
    fn dimensions(&self) -> usize;
}

// ──────────────────────────── MockProvider ─────────────────────────

/// Deterministic hash-based embedding provider for testing.
///
/// **Not for production.**  Uses consistent feature hashing so that
/// identical text always yields the same unit-length vector, and texts
/// sharing tokens have higher cosine similarity.  This is strictly
/// better than the legacy `hashed_cosine` (64-dim, unsigned,
/// unnormalised) because it:
///
/// - Uses 1536 dimensions (matches pgvector schema, fewer collisions).
/// - Applies sign randomisation (+1/−1) to reduce degeneracy.
/// - L2-normalises the output for direct cosine comparison.
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    dimensions: usize,
}

impl Default for MockEmbeddingProvider {
    fn default() -> Self {
        Self {
            dimensions: EMBEDDING_DIM_3_SMALL,
        }
    }
}

impl MockEmbeddingProvider {
    /// Create a mock provider with the given embedding dimensionality.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(embed_text(text, self.dimensions))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// ──────────────────────────── OpenAIConfig ────────────────────────

/// Configuration for the OpenAI embeddings API (config-only, no HTTP call).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingConfig {
    /// Model identifier, e.g. `text-embedding-3-small`.
    pub model: String,
    /// Output vector dimensionality (1536 for 3-small).
    pub dimensions: usize,
    /// Base URL of the embeddings API endpoint.
    pub api_base: String,
}

impl Default for OpenAIEmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-3-small".to_string(),
            dimensions: EMBEDDING_DIM_3_SMALL,
            api_base: "https://api.openai.com/v1".to_string(),
        }
    }
}

// ──────────────────────────── Helpers ─────────────────────────────

/// FNV-1a hash with a seed.
fn fnv1a(seed: u64, data: &[u8]) -> u64 {
    let mut h = seed;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Tokenise text into lowercase alphanumeric tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Produce a deterministic unit-length embedding for `text`.
///
/// Each token activates `HASH_BANDS` dimensions via FNV-1a hashing with
/// per-band seeds.  Sign randomisation (+1/−1) prevents the all-positive
/// degeneracy of the old `hashed_cosine`.  The result is L2-normalised.
pub fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let tokens = tokenize(text);
    for token in &tokens {
        for band in 0..HASH_BANDS {
            let seed = FNV_OFFSET.wrapping_add((band as u64).wrapping_mul(FNV_PRIME));
            let h = fnv1a(seed, token.as_bytes());
            let idx = (h as usize) % dim;
            let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
    }
    // L2 normalise
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Cosine similarity between two vectors, returns `0.0` for zero-length
/// or mismatched vectors.  Computation is done in `f64` for precision.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Encode a float vector as a pgvector text literal: `[0.1,0.2,...]`.
pub fn encode_vector(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

// ──────────────────────────── Tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1 — Default dimensions are 1536
    #[test]
    fn default_dimensions() {
        let mock = MockEmbeddingProvider::default();
        assert_eq!(mock.dimensions(), EMBEDDING_DIM_3_SMALL);
        assert_eq!(mock.dimensions(), 1536);
    }

    // 2 — Custom dimensions
    #[tokio::test]
    async fn custom_dimensions() {
        let mock = MockEmbeddingProvider::new(384);
        let v = mock.embed("hello world").await.unwrap();
        assert_eq!(v.len(), 384);
    }

    // 3 — Same text → identical embedding (consistency)
    #[tokio::test]
    async fn consistency_same_text_same_vector() {
        let mock = MockEmbeddingProvider::default();
        let a = mock.embed("the quick brown fox").await.unwrap();
        let b = mock.embed("the quick brown fox").await.unwrap();
        assert_eq!(a, b, "identical text must produce identical vectors");
    }

    // 4 — Different text → different embedding
    #[tokio::test]
    async fn different_text_different_vector() {
        let mock = MockEmbeddingProvider::default();
        let a = mock.embed("machine learning model").await.unwrap();
        let b = mock.embed("factory production line").await.unwrap();
        assert_ne!(a, b, "different text should produce different vectors");
    }

    // 5 — Empty text → zero vector (all zeros)
    #[tokio::test]
    async fn empty_text_zero_vector() {
        let mock = MockEmbeddingProvider::default();
        let v = mock.embed("").await.unwrap();
        assert!(v.iter().all(|&x| x == 0.0), "empty text → zero vector");
        assert_eq!(v.len(), 1536);
    }

    // 6 — Output is unit-length (L2 normalised) for non-empty text
    #[tokio::test]
    async fn output_is_unit_length() {
        let mock = MockEmbeddingProvider::default();
        let v = mock.embed("normalisation test").await.unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2 norm should be 1.0, got {norm}"
        );
    }

    // 7 — Similar texts have higher similarity than dissimilar texts
    #[tokio::test]
    async fn similar_texts_higher_similarity() {
        let mock = MockEmbeddingProvider::default();
        let q = mock
            .embed("deploy microservice to kubernetes")
            .await
            .unwrap();
        let similar = mock
            .embed("deploy service to kubernetes cluster")
            .await
            .unwrap();
        let dissimilar = mock.embed("cook pasta with tomato sauce").await.unwrap();
        let sim_score = cosine_similarity(&q, &similar);
        let dis_score = cosine_similarity(&q, &dissimilar);
        assert!(
            sim_score > dis_score,
            "similar ({sim_score}) should beat dissimilar ({dis_score})"
        );
    }

    // 8 — Identical vectors → similarity 1.0
    #[test]
    fn cosine_similarity_identical_is_one() {
        let v = embed_text("test vector", 256);
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "identical → 1.0, got {sim}");
    }

    // 9 — Zero vectors → similarity 0.0
    #[test]
    fn cosine_similarity_zero_vectors() {
        let a = vec![0.0f32; 128];
        let b = vec![0.0f32; 128];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // 10 — Mismatched lengths → 0.0
    #[test]
    fn cosine_similarity_mismatched_length() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // 11 — Cosine similarity is symmetric
    #[test]
    fn cosine_similarity_symmetric() {
        let a = embed_text("alpha beta gamma", 512);
        let b = embed_text("beta gamma delta", 512);
        let ab = cosine_similarity(&a, &b);
        let ba = cosine_similarity(&b, &a);
        assert!((ab - ba).abs() < 1e-6, "similarity must be symmetric");
    }

    // 12 — encode_vector produces pgvector text format
    #[test]
    fn encode_vector_format() {
        let v = vec![0.1f32, 0.2, 0.3];
        let s = encode_vector(&v);
        assert!(s.starts_with('['));
        assert!(s.ends_with(']'));
        assert!(s.contains("0.1"));
        assert!(s.contains("0.2"));
        assert!(s.contains("0.3"));
    }

    // 13 — encode_vector handles empty vector
    #[test]
    fn encode_vector_empty() {
        let s = encode_vector(&[]);
        assert_eq!(s, "[]");
    }

    // 14 — OpenAIEmbeddingConfig defaults
    #[test]
    fn openai_config_defaults() {
        let cfg = OpenAIEmbeddingConfig::default();
        assert_eq!(cfg.model, "text-embedding-3-small");
        assert_eq!(cfg.dimensions, 1536);
        assert_eq!(cfg.api_base, "https://api.openai.com/v1");
    }

    // 15 — Case-insensitive tokenisation
    #[tokio::test]
    async fn case_insensitive_embedding() {
        let mock = MockEmbeddingProvider::default();
        let a = mock.embed("Hello World").await.unwrap();
        let b = mock.embed("hello world").await.unwrap();
        assert_eq!(a, b, "tokenisation should be case-insensitive");
    }

    // 16 — Partial overlap yields higher similarity than no overlap
    #[tokio::test]
    async fn partial_overlap_intermediate() {
        let mock = MockEmbeddingProvider::default();
        let base = mock.embed("alpha beta gamma delta").await.unwrap();
        let half = mock.embed("alpha beta").await.unwrap();
        let none = mock.embed("zzz yyy xxx www").await.unwrap();
        let sim_half = cosine_similarity(&base, &half);
        let sim_none = cosine_similarity(&base, &none);
        assert!(
            sim_half > sim_none,
            "partial overlap ({sim_half}) should beat no overlap ({sim_none})"
        );
    }
}
