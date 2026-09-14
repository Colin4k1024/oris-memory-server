//! Mem0 Engine Adapter (§3.6).
//!
//! Wraps the Mem0 REST API as a pluggable [`MemoryEngine`] implementation.
//! Mem0 provides personal long-term inference memory with semantic search,
//! conflict merge, and delete propagation. This adapter translates Oris
//! [`EngineQuery`] / [`EngineWriteItem`] calls into Mem0 HTTP requests.
//!
//! ## REST API mapping
//!
//! | Trait method | Mem0 endpoint |
//! |--------------|---------------|
//! | `search` | `POST /search` |
//! | `write` | `POST /memories` |
//! | `delete` | `DELETE /memories/{id}` |
//! | `health` | `GET /memories?user_id=__health__&limit=1` |
//!
//! ## Authentication
//!
//! Mem0 supports `X-API-Key` header or `Authorization: Bearer <token>`.
//! The key is configured at construction time via [`Mem0Engine::new`] or
//! the builder [`Mem0EngineBuilder`].

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

/// Body for `POST /memories`.
#[derive(Debug, Serialize)]
struct Mem0WriteRequest<'a> {
    messages: Vec<Mem0Message<'a>>,
    user_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct Mem0Message<'a> {
    role: &'a str,
    content: &'a str,
}

/// Body for `POST /search`.
#[derive(Debug, Serialize)]
struct Mem0SearchRequest<'a> {
    query: &'a str,
    user_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<&'a str>,
    limit: usize,
}

/// A single memory returned by Mem0 search or list.
#[derive(Debug, Deserialize)]
struct Mem0Memory {
    id: String,
    #[serde(default)]
    memory: String,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

/// Response shape for `POST /search` — Mem0 returns a list (or paginated wrapper).
#[derive(Debug, Deserialize)]
struct Mem0SearchResponse {
    #[serde(default)]
    results: Vec<Mem0Memory>,
}

// ─────────────────────────────────────────────────────────────────────
// Mem0Engine
// ─────────────────────────────────────────────────────────────────────

/// Mem0 memory engine adapter — wraps the Mem0 REST API.
///
/// Configure with a base URL, API key, and optional timeout. Register into
/// an [`EngineRegistry`](crate::engine::EngineRegistry) to use behind the
/// Oris control plane with circuit-breaker isolation.
pub struct Mem0Engine {
    client: Client,
    base_url: String,
    api_key: String,
    timeout: Duration,
}

impl Mem0Engine {
    /// Create a new Mem0 engine with the given base URL and API key.
    ///
    /// The base URL should include the API prefix (e.g.
    /// `http://localhost:8080` for self-hosted Mem0).
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to build reqwest client"),
            base_url: base_url.into(),
            api_key: api_key.into(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Builder for more control over client configuration.
    pub fn builder() -> Mem0EngineBuilder {
        Mem0EngineBuilder::default()
    }

    fn build_search_url(&self) -> String {
        format!("{}/search", self.base_url)
    }

    fn build_memories_url(&self) -> String {
        format!("{}/memories", self.base_url)
    }

    fn build_memory_url(&self, id: &str) -> String {
        format!("{}/memories/{}", self.base_url, id)
    }

    fn build_health_url(&self) -> String {
        format!(
            "{}/memories?user_id=__health__&limit=1",
            self.base_url
        )
    }

    fn map_status_error(status: StatusCode, body: &str) -> EngineError {
        match status.as_u16() {
            401 | 403 => EngineError::Internal(format!("mem0 auth error: {body}")),
            404 => EngineError::Internal(format!("mem0 not found: {body}")),
            429 => EngineError::Internal(format!("mem0 rate limited: {body}")),
            500..=599 => EngineError::Internal(format!("mem0 server error: {body}")),
            _ => EngineError::Internal(format!("mem0 http {}: {}", status.as_u16(), body)),
        }
    }

    fn extract_user_id<'a>(&'a self, query: &'a EngineQuery) -> &'a str {
        query
            .user_id
            .as_deref()
            .unwrap_or(&query.tenant_id)
    }
}

#[async_trait]
impl MemoryEngine for Mem0Engine {
    fn name(&self) -> &str {
        "mem0"
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            semantic_search: true,
            keyword_search: true,
            graph_search: false,
            temporal_search: false,
            entity_linking: false,
            ontology_grounding: false,
        }
    }

    #[instrument(skip(self, query), fields(engine = "mem0", q_len = query.text.len()))]
    async fn search(&self, query: &EngineQuery) -> Result<Vec<EngineResult>, EngineError> {
        let user_id = self.extract_user_id(query);
        let agent_id = query.filters.get("agent_id").and_then(|v| v.as_str());

        let body = Mem0SearchRequest {
            query: &query.text,
            user_id,
            agent_id,
            limit: query.top_k,
        };

        let url = self.build_search_url();
        debug!(url = %url, "mem0 search request");

        let response = self
            .client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("mem0 search request failed: {e}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "mem0 search error");
            return Err(Self::map_status_error(status, &body_text));
        }

        let search_resp: Mem0SearchResponse = response
            .json()
            .await
            .map_err(|e| EngineError::Internal(format!("mem0 search decode error: {e}")))?;

        let results = search_resp
            .results
            .into_iter()
            .map(|m| EngineResult {
                engine_name: "mem0".into(),
                memory_id: m.id,
                content: m.memory,
                score: m.score.unwrap_or(0.0),
                metadata: m.metadata.unwrap_or(serde_json::Value::Null),
                evidence_refs: vec![],
            })
            .collect();

        Ok(results)
    }

    #[instrument(skip(self, item), fields(engine = "mem0", mid = %item.memory_id))]
    async fn write(&self, item: &EngineWriteItem) -> Result<(), EngineError> {
        let user_id = item.metadata.get("user_id").and_then(|v| v.as_str());
        let user_id = user_id.unwrap_or(&item.tenant_id);
        let agent_id = item.metadata.get("agent_id").and_then(|v| v.as_str());

        let write_body = Mem0WriteRequest {
            messages: vec![Mem0Message {
                role: "system",
                content: &item.content,
            }],
            user_id,
            agent_id,
            run_id: None,
            metadata: Some(&item.metadata),
        };

        let url = self.build_memories_url();
        debug!(url = %url, "mem0 write request");

        let response = self
            .client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .json(&write_body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("mem0 write request failed: {e}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "mem0 write error");
            return Err(Self::map_status_error(status, &body_text));
        }

        Ok(())
    }

    #[instrument(skip(self), fields(engine = "mem0", id = %id))]
    async fn delete(&self, id: &str) -> Result<(), EngineError> {
        let url = self.build_memory_url(id);
        debug!(url = %url, "mem0 delete request");

        let response = self
            .client
            .delete(&url)
            .header("X-API-Key", &self.api_key)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    EngineError::Timeout(self.timeout)
                } else {
                    EngineError::Internal(format!("mem0 delete request failed: {e}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() && status != StatusCode::NOT_FOUND {
            let body_text = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body_text, "mem0 delete error");
            return Err(Self::map_status_error(status, &body_text));
        }

        Ok(())
    }

    async fn health(&self) -> Result<EngineHealth, EngineError> {
        let url = self.build_health_url();

        let result = self
            .client
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .timeout(self.timeout)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => Ok(EngineHealth::Healthy),
            Ok(resp) => {
                warn!(status = %resp.status(), "mem0 health degraded");
                Ok(EngineHealth::Degraded)
            }
            Err(e) if e.is_timeout() => {
                warn!("mem0 health check timed out");
                Ok(EngineHealth::Unreachable)
            }
            Err(e) => {
                warn!(error = %e, "mem0 health check failed");
                Ok(EngineHealth::Unreachable)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────

/// Builder for [`Mem0Engine`] with configurable HTTP client and timeout.
#[derive(Default)]
pub struct Mem0EngineBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Option<Duration>,
    client_builder: Option<reqwest::ClientBuilder>,
}

impl Mem0EngineBuilder {
    /// Set the Mem0 base URL (e.g. `http://localhost:8080`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the API key for authentication (sent as `X-API-Key` header).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the per-request timeout (default: 30s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Provide a custom `reqwest::ClientBuilder` for advanced configuration
    /// (TLS, proxy, etc.).
    pub fn client_builder(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.client_builder = Some(builder);
        self
    }

    /// Build the [`Mem0Engine`] instance.
    pub fn build(self) -> Result<Mem0Engine, EngineError> {
        let base_url = self
            .base_url
            .ok_or_else(|| EngineError::Internal("mem0 base_url is required".into()))?;
        let api_key = self
            .api_key
            .ok_or_else(|| EngineError::Internal("mem0 api_key is required".into()))?;
        let timeout = self.timeout.unwrap_or(Duration::from_secs(30));

        let builder = self.client_builder.unwrap_or_else(|| {
            Client::builder()
                .timeout(timeout)
        });

        let client = builder
            .build()
            .map_err(|e| EngineError::Internal(format!("failed to build reqwest client: {e}")))?;

        Ok(Mem0Engine {
            client,
            base_url,
            api_key,
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

    fn make_engine(server: &Server) -> Mem0Engine {
        Mem0Engine::builder()
            .base_url(server.url())
            .api_key("test-key")
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    fn make_query(text: &str, user_id: &str) -> EngineQuery {
        EngineQuery {
            text: text.into(),
            tenant_id: "tenant-1".into(),
            user_id: Some(user_id.into()),
            task_id: None,
            top_k: 5,
            filters: HashMap::new(),
        }
    }

    fn make_write_item(content: &str) -> EngineWriteItem {
        let metadata = serde_json::json!({"user_id": "user-1"});
        EngineWriteItem {
            memory_id: "mem-123".into(),
            tenant_id: "tenant-1".into(),
            content: content.into(),
            memory_type: "episodic".into(),
            scope: "personal".into(),
            metadata,
            evidence_refs: vec![],
        }
    }

    #[tokio::test]
    async fn search_returns_results() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/search")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"results": [{"id": "m1", "memory": "user likes rust", "score": 0.95, "metadata": {"source": "chat"}}]}"#,
            )
            .create_async()
            .await;

        let engine = make_engine(&server);
        let query = make_query("rust programming", "user-1");
        let results = engine.search(&query).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].engine_name, "mem0");
        assert_eq!(results[0].memory_id, "m1");
        assert!((results[0].score - 0.95).abs() < 0.01);
        assert_eq!(results[0].content, "user likes rust");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn search_empty_results() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/search")
            .with_status(200)
            .with_body(r#"{"results": []}"#)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let query = make_query("nothing", "user-1");
        let results = engine.search(&query).await.unwrap();
        assert!(results.is_empty());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn search_server_error_returns_internal_error() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/search")
            .with_status(500)
            .with_body("internal error")
            .create_async()
            .await;

        let engine = make_engine(&server);
        let query = make_query("test", "user-1");
        let err = engine.search(&query).await.unwrap_err();
        assert!(matches!(err, EngineError::Internal(ref m) if m.contains("server error")));
    }

    #[tokio::test]
    async fn write_sends_memories_request() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("POST", "/memories")
            .match_header("X-API-Key", "test-key")
            .with_status(201)
            .with_body(r#"{"id": "m2"}"#)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let item = make_write_item("user is a developer");
        engine.write(&item).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn write_auth_error() {
        let mut server = Server::new_async().await;
        server
            .mock("POST", "/memories")
            .with_status(401)
            .with_body("unauthorized")
            .create_async()
            .await;

        let engine = make_engine(&server);
        let item = make_write_item("content");
        let err = engine.write(&item).await.unwrap_err();
        assert!(matches!(err, EngineError::Internal(ref m) if m.contains("auth")));
    }

    #[tokio::test]
    async fn delete_succeeds() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("DELETE", "/memories/mem-123")
            .with_status(200)
            .create_async()
            .await;

        let engine = make_engine(&server);
        engine.delete("mem-123").await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn delete_not_found_is_ok() {
        let mut server = Server::new_async().await;
        let mock = server
            .mock("DELETE", "/memories/missing")
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
            .mock("GET", "/memories")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"{"results": []}"#)
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
            .mock("GET", "/memories")
            .with_status(500)
            .create_async()
            .await;

        let engine = make_engine(&server);
        let health = engine.health().await.unwrap();
        assert_eq!(health, EngineHealth::Degraded);
    }

    #[test]
    fn capabilities_advertises_semantic_and_keyword() {
        let caps = Mem0Engine::new("http://localhost", "key").capabilities();
        assert!(caps.semantic_search);
        assert!(caps.keyword_search);
        assert!(!caps.graph_search);
    }

    #[test]
    fn builder_requires_base_url_and_api_key() {
        let result = Mem0EngineBuilder::default().build();
        assert!(matches!(result, Err(EngineError::Internal(_))));
    }

    #[test]
    fn builder_succeeds_with_all_fields() {
        let engine = Mem0Engine::builder()
            .base_url("http://localhost:8080")
            .api_key("secret")
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        assert_eq!(engine.name(), "mem0");
    }
}
