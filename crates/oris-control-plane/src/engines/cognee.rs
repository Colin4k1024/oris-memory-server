//! Cognee Engine Adapter (§3.7).
//!
//! Wraps the Cognee REST API as a pluggable [`MemoryEngine`] implementation.
//! Cognee transforms raw text into a knowledge graph through entity
//! extraction, relationship inference, and multi-hop reasoning. This adapter
//! translates Oris [`EngineQuery`] / [`EngineWriteItem`] calls into Cognee
//! HTTP requests.
//!
//! ## REST API mapping
//!
//! | Trait method | Cognee endpoint |
//! |--------------|-----------------|
//! | `search` | `POST /api/v1/search` |
//! | `write` | `POST /api/v1/datasets/{name}/content` (+ create dataset) |
//! | `delete` | `DELETE /api/v1/datasets/{name}` |
//! | `health` | `GET /api/v1/datasets` |
//!
//! ## Dataset mapping
//!
//! Cognee organizes content into *datasets*. This adapter maps the Oris
//! `tenant_id` to a Cognee dataset name (sanitized). All writes from a
//! tenant go into the same dataset. Deletion targets the dataset named
//! by `memory_id` (or the tenant dataset if the id matches the tenant).
//!
//! ## Authentication
//!
//! Cognee uses `Authorization: Bearer <token>`. The token is configured at
//! construction time via [`CogneeEngine::new`] or the builder
//! [`CogneeEngineBuilder`].

use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::engine::{
    EngineCapabilities, EngineError, EngineHealth, EngineQuery, EngineResult, EngineWriteItem,
    MemoryEngine,
};

// ─────────────────────────────────────────────────────────────────────
// Request / Response types
// ─────────────────────────────────────────────────────────────────────

/// Body for `POST /api/v1/datasets`.
#[derive(Debug, Serialize)]
struct CogneeCreateDatasetRequest<'a> {
    name: &'a str,
}

/// Body for `POST /api/v1/datasets/{name}/content`.
#[derive(Debug, Serialize)]
struct CogneeContentRequest<'a> {
    data: &'a str,
}

/// Body for `POST /api/v1/search`.
#[derive(Debug, Serialize)]
struct CogneeSearchRequest<'a> {
    query: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    datasets: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

/// A single search result returned by Cognee.
#[derive(Debug, Deserialize)]
struct CogneeSearchResultItem {
    #[serde(default, alias = "id")]
    id: Option<String>,
    #[serde(default, alias = "content", alias = "text")]
    content: Option<String>,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

/// Response shape for `POST /api/v1/search`.
#[derive(Debug, Deserialize)]
struct CogneeSearchResponse {
    #[serde(default)]
    results: Vec<serde_json::Value>,
}

// ─────────────────────────────────────────────────────────────────────
// CogneeEngine
// ─────────────────────────────────────────────────────────────────────

/// Cognee memory engine adapter — wraps the Cognee REST API.
///
/// Configure with a base URL, auth token, and optional timeout. Register
/// into an [`EngineRegistry`](crate::engine::EngineRegistry) to use behind
/// the Oris control plane with circuit-breaker isolation.
pub struct CogneeEngine {
    client: Client,
    base_url: String,
    auth_token: String,
    timeout: Duration,
}

impl CogneeEngine {
    /// Create a new Cognee engine with the given base URL and auth token.
    ///
    /// The base URL should be the Cognee server root (e.g.
    /// `http://localhost:8000`).
    pub fn new(base_url: impl Into<String>, auth_token: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("failed to build reqwest client"),
            base_url: base_url.into(),
            auth_token: auth_token.into(),
            timeout: Duration::from_secs(60),
        }
    }

    /// Builder for more control over client configuration.
    pub fn builder() -> CogneeEngineBuilder {
        CogneeEngineBuilder::default()
    }

    fn api_base(&self) -> String {
        format!("{}/api/v1", self.base_url)
    }

    fn build_search_url(&self) -> String {
        format!("{}/search", self.api_base())
    }

    fn build_datasets_url(&self) -> String {
        format!("{}/datasets", self.api_base())
    }

    fn build_dataset_content_url(&self, dataset_name: &str) -> String {
        format!("{}/datasets/{}/content", self.api_base(), dataset_name)
    }

    fn build_dataset_url(&self, dataset_name: &str) -> String {
        format!("{}/datasets/{}", self.api_base(), dataset_name)
    }

    fn build_cognify_url(&self) -> String {
        format!("{}/cognify", self.api_base())
    }

    /// Sanitize a tenant or memory ID into a valid Cognee dataset name.
    /// Cognee dataset names should be alphanumeric with underscores/hyphens.
    fn sanitize_dataset_name(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }

    fn map_status_error(status: StatusCode, body: &str) -> EngineError {
        match status.as_u16() {
            401 | 403 => EngineError::Internal(format!("cognee auth error: {body}")),
            404 => EngineError::Internal(format!("cognee not found: {body}")),
            429 => EngineError::Internal(format!("cognee rate limited: {body}")),
            500..=599 => EngineError::Internal(format!("cognee server error: {body}")),
            _ => EngineError::Internal(format!(
                "cognee http {}: {}",
                status.as_u16(),
                body
            )),
        }
    }

    /// Ensure a dataset exists before writing content. Creates it if not.
    #[instrument(skip(self), fields(engine = "cognee", dataset = %dataset_name))]
    async fn ensure_dataset(&self, dataset_name: &str) -> Result<(), EngineError> {
        let url = self.build_datasets_url();
        let body = CogneeCreateDatasetRequest {
            name: dataset_name,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("cognee create dataset failed: {e}"))
                }
            })?;

        let status = response.status();
        // 200/201 = created, 409 = already exists — both acceptable
        if status.is_success() || status == StatusCode::CONFLICT {
            debug!(status = %status, dataset = %dataset_name, "dataset ready");
            return Ok(());
        }

        if status == StatusCode::NOT_FOUND {
            let body_text = response.text().await.unwrap_or_default();
            return Err(Self::map_status_error(status, &body_text));
        }

        let body_text = response.text().await.unwrap_or_default();
        warn!(status = %status, body = %body_text, "cognee dataset creation error");
        Err(Self::map_status_error(status, &body_text))
    }

    /// Extract search results from the raw JSON values returned by Cognee.
    fn extract_search_results(results: Vec<serde_json::Value>) -> Vec<EngineResult> {
        results
            .into_iter()
            .filter_map(|v| {
                // Cognee can return results in varying shapes — try the
                // typed struct first, fall back to raw JSON extraction.
                if let Ok(item) = serde_json::from_value::<CogneeSearchResultItem>(v.clone()) {
                    let id = item
                        .id
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                    let content = item.content.unwrap_or_else(|| v.to_string());
                    let score = item.score.unwrap_or(0.0);
                    let metadata = item.metadata.unwrap_or(serde_json::Value::Null);
                    return Some(EngineResult {
                        engine_name: "cognee".into(),
                        memory_id: id,
                        content,
                        score,
                        metadata,
                        evidence_refs: vec![],
                    });
                }
                // Fall back: use the raw JSON as content.
                let content = v.to_string();
                Some(EngineResult {
                    engine_name: "cognee".into(),
                    memory_id: uuid::Uuid::new_v4().to_string(),
                    content,
                    score: 0.0,
                    metadata: v,
                    evidence_refs: vec![],
                })
            })
            .collect()
    }
}

#[async_trait]
impl MemoryEngine for CogneeEngine {
    fn name(&self) -> &str {
        "cognee"
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            semantic_search: true,
            keyword_search: false,
            graph_search: true,
            temporal_search: false,
            entity_linking: true,
            ontology_grounding: true,
        }
    }

    #[instrument(skip(self, query), fields(engine = "cognee", q_len = query.text.len()))]
    async fn search(&self, query: &EngineQuery) -> Result<Vec<EngineResult>, EngineError> {
        // Determine which datasets to search. If a specific dataset filter
        // is provided, use it; otherwise use the tenant's default dataset.
        let datasets: Vec<String> = if let Some(ds) = query.filters.get("datasets").and_then(|v| v.as_array()) {
            ds.iter()
                .filter_map(|d| d.as_str().map(String::from))
                .collect()
        } else {
            let tenant_ds = Self::sanitize_dataset_name(&query.tenant_id);
            vec![tenant_ds]
        };

        let body = CogneeSearchRequest {
            query: &query.text,
            datasets,
            limit: Some(query.top_k),
        };

        let url = self.build_search_url();
        debug!(url = %url, "cognee search request");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("cognee search request failed: {e}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "cognee search error");
            return Err(Self::map_status_error(status, &body_text));
        }

        let search_resp: CogneeSearchResponse = response
            .json()
            .await
            .map_err(|e| EngineError::Internal(format!("cognee search decode error: {e}")))?;

        Ok(Self::extract_search_results(search_resp.results))
    }

    #[instrument(skip(self, item), fields(engine = "cognee", mid = %item.memory_id))]
    async fn write(&self, item: &EngineWriteItem) -> Result<(), EngineError> {
        let dataset_name = Self::sanitize_dataset_name(&item.tenant_id);

        // Step 1: ensure the dataset exists (idempotent).
        self.ensure_dataset(&dataset_name).await?;

        // Step 2: add content to the dataset.
        let content_url = self.build_dataset_content_url(&dataset_name);
        let content_body = CogneeContentRequest {
            data: &item.content,
        };

        debug!(url = %content_url, "cognee add content request");
        let response = self
            .client
            .post(&content_url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&content_body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("cognee add content failed: {e}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "cognee add content error");
            return Err(Self::map_status_error(status, &body_text));
        }

        // Step 3: trigger cognify to process the content into the knowledge
        // graph. This is asynchronous in Cognee, but we fire the request
        // and accept success or "already processing" as valid.
        let cognify_url = self.build_cognify_url();
        let response = self
            .client
            .post(&cognify_url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&serde_json::json!({"datasets": [dataset_name]}))
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("cognee cognify failed: {e}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "cognee cognify error");
            return Err(Self::map_status_error(status, &body_text));
        }

        Ok(())
    }

    #[instrument(skip(self), fields(engine = "cognee", id = %id))]
    async fn delete(&self, id: &str) -> Result<(), EngineError> {
        let dataset_name = Self::sanitize_dataset_name(id);
        let url = self.build_dataset_url(&dataset_name);
        debug!(url = %url, "cognee delete dataset request");

        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("cognee delete failed: {e}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() && status != StatusCode::NOT_FOUND {
            let body_text = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "cognee delete error");
            return Err(Self::map_status_error(status, &body_text));
        }

        Ok(())
    }

    async fn health(&self) -> Result<EngineHealth, EngineError> {
        let url = self.build_datasets_url();

        let result = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .timeout(self.timeout)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => Ok(EngineHealth::Healthy),
            Ok(resp) => {
                warn!(status = %resp.status(), "cognee health degraded");
                Ok(EngineHealth::Degraded)
            }
            Err(e) if e.is_timeout() => {
                warn!("cognee health check timed out");
                Ok(EngineHealth::Unreachable)
            }
            Err(e) => {
                warn!(error = %e, "cognee health check failed");
                Ok(EngineHealth::Unreachable)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────

/// Builder for [`CogneeEngine`] with configurable HTTP client and timeout.
#[derive(Default)]
pub struct CogneeEngineBuilder {
    base_url: Option<String>,
    auth_token: Option<String>,
    timeout: Option<Duration>,
    client_builder: Option<reqwest::ClientBuilder>,
}

impl CogneeEngineBuilder {
    /// Set the Cognee base URL (e.g. `http://localhost:8000`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the auth token for authentication (sent as `Authorization: Bearer`).
    pub fn auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Override the per-request timeout (default: 60s for cognify processing).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Provide a custom `reqwest::ClientBuilder` for advanced configuration.
    pub fn client_builder(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.client_builder = Some(builder);
        self
    }

    /// Build the [`CogneeEngine`] instance.
    pub fn build(self) -> Result<CogneeEngine, EngineError> {
        let base_url = self
            .base_url
            .ok_or_else(|| EngineError::Internal("cognee base_url is required".into()))?;
        let auth_token = self
            .auth_token
            .ok_or_else(|| EngineError::Internal("cognee auth_token is required".into()))?;
        let timeout = self.timeout.unwrap_or(Duration::from_secs(60));

        let builder = self.client_builder.unwrap_or_else(|| {
            Client::builder()
                .timeout(timeout)
        });

        let client = builder
            .build()
            .map_err(|e| EngineError::Internal(format!("failed to build reqwest client: {e}")))?;

        Ok(CogneeEngine {
            client,
            base_url,
            auth_token,
            timeout,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use mockito::Server;

    fn make_engine(server: &Server) -> CogneeEngine {
        CogneeEngine::builder()
            .base_url(server.url())
            .auth_token("test-token")
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    fn make_query(text: &str, tenant: &str) -> EngineQuery {
        EngineQuery {
            text: text.into(),
            tenant_id: tenant.into(),
            user_id: None,
            task_id: None,
            top_k: 5,
            filters: HashMap::new(),
        }
    }

    fn make_write_item(content: &str, tenant: &str) -> EngineWriteItem {
        EngineWriteItem {
            memory_id: "mem-456".into(),
            tenant_id: tenant.into(),
            content: content.into(),
            memory_type: "semantic".into(),
            scope: "enterprise".into(),
            metadata: serde_json::json!({}),
            evidence_refs: vec![],
        }
    }

    #[tokio::test]
    async fn search_returns_results() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/api/v1/search")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"results": [{"id": "c1", "content": "entity A relates to B", "score": 0.88}]}"#,
            )
            .create_async()
            .await;

        let engine = make_engine(&server);
        let query = make_query("entity A", "tenant-1");
        let results = engine.search(&query).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].engine_name, "cognee");
        assert_eq!(results[0].memory_id, "c1");
        assert!((results[0].score - 0.88).abs() < 0.01);
        assert_eq!(results[0].content, "entity A relates to B");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn search_empty_results() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/api/v1/search")
            .with_status(200)
            .with_body(r#"{"results": []}"#)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let query = make_query("nothing", "tenant-1");
        let results = engine.search(&query).await.unwrap();
        assert!(results.is_empty());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn search_server_error_returns_internal_error() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/api/v1/search")
            .with_status(500)
            .with_body("internal error")
            .create_async()
            .await;

        let engine = make_engine(&server);
        let query = make_query("test", "tenant-1");
        let err = engine.search(&query).await.unwrap_err();
        assert!(matches!(err, EngineError::Internal(ref m) if m.contains("server error")));
    }

    #[tokio::test]
    async fn write_creates_dataset_adds_content_and_cognifies() {
        let mut server = Server::new_async().await;

        let dataset_mock = server
            .mock("POST", "/api/v1/datasets")
            .with_status(201)
            .create_async()
            .await;

        let content_mock = server
            .mock("POST", "/api/v1/datasets/tenant-1/content")
            .match_header("Authorization", "Bearer test-token")
            .with_status(200)
            .create_async()
            .await;

        let cognify_mock = server
            .mock("POST", "/api/v1/cognify")
            .with_status(200)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let item = make_write_item("some knowledge content", "tenant-1");
        engine.write(&item).await.unwrap();

        dataset_mock.assert_async().await;
        content_mock.assert_async().await;
        cognify_mock.assert_async().await;
    }

    #[tokio::test]
    async fn write_dataset_already_exists_is_ok() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/api/v1/datasets")
            .with_status(409)
            .create_async()
            .await;

        server
            .mock("POST", "/api/v1/datasets/tenant-1/content")
            .with_status(200)
            .create_async()
            .await;

        server
            .mock("POST", "/api/v1/cognify")
            .with_status(200)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let item = make_write_item("content", "tenant-1");
        engine.write(&item).await.unwrap();
    }

    #[tokio::test]
    async fn write_content_error_propagates() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/api/v1/datasets")
            .with_status(201)
            .create_async()
            .await;

        server
            .mock("POST", "/api/v1/datasets/tenant-1/content")
            .with_status(500)
            .with_body("content error")
            .create_async()
            .await;

        let engine = make_engine(&server);
        let item = make_write_item("content", "tenant-1");
        let err = engine.write(&item).await.unwrap_err();
        assert!(matches!(err, EngineError::Internal(ref m) if m.contains("server error")));
    }

    #[tokio::test]
    async fn delete_succeeds() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("DELETE", "/api/v1/datasets/dataset-1")
            .with_status(200)
            .create_async()
            .await;

        let engine = make_engine(&server);
        engine.delete("dataset-1").await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn delete_not_found_is_ok() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("DELETE", "/api/v1/datasets/missing")
            .with_status(404)
            .create_async()
            .await;

        let engine = make_engine(&server);
        engine.delete("missing").await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_healthy() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/api/v1/datasets")
            .with_status(200)
            .with_body(r#"[]"#)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let health = engine.health().await.unwrap();
        assert_eq!(health, EngineHealth::Healthy);
    }

    #[tokio::test]
    async fn health_degraded_on_500() {
        let mut server = Server::new_async().await;
        server
            .mock("GET", "/api/v1/datasets")
            .with_status(500)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let health = engine.health().await.unwrap();
        assert_eq!(health, EngineHealth::Degraded);
    }

    #[test]
    fn capabilities_advertises_graph_and_entity() {
        let caps = CogneeEngine::new("http://localhost", "token").capabilities();
        assert!(caps.semantic_search);
        assert!(caps.graph_search);
        assert!(caps.entity_linking);
        assert!(caps.ontology_grounding);
        assert!(!caps.keyword_search);
    }

    #[test]
    fn sanitize_dataset_name_replaces_special_chars() {
        assert_eq!(CogneeEngine::sanitize_dataset_name("tenant@1"), "tenant_1");
        assert_eq!(CogneeEngine::sanitize_dataset_name("ok-tenant_1"), "ok-tenant_1");
        assert_eq!(CogneeEngine::sanitize_dataset_name("tenant 1"), "tenant_1");
    }

    #[test]
    fn builder_requires_base_url_and_token() {
        let result = CogneeEngineBuilder::default().build();
        assert!(matches!(result, Err(EngineError::Internal(_))));
    }

    #[test]
    fn builder_succeeds_with_all_fields() {
        let engine = CogneeEngine::builder()
            .base_url("http://localhost:8000")
            .auth_token("secret")
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap();
        assert_eq!(engine.name(), "cognee");
    }
}
