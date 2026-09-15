//! Graphiti Engine Adapter (§3.8 — optional, Phase 2+).
//!
//! Wraps the Graphiti REST API as a pluggable [`MemoryEngine`] implementation.
//! Graphiti provides temporal (bi-temporal) context graphs with source
//! provenance, fact-validity windows, and hybrid semantic/keyword/graph
//! traversal search.
//!
//! Per the design document (§3.8), Graphiti is a **later-phase optional**
//! component. It should only be enabled when PostgreSQL's time fields +
//! entity_relation table + Cognee (if adopted) cannot meet business
//! precision or query-efficiency requirements for dual-temporal / fact
//! expiration scenarios. This adapter provides the integration surface so
//! that Graphiti can be activated behind the `EngineRegistry` without
//! changes to the control plane.
//!
//! ## REST API mapping
//!
//! | Trait method | Graphiti endpoint |
//! |--------------|-------------------|
//! | `search` | `POST /search` |
//! | `write` | `POST /episodes` |
//! | `delete` | `DELETE /episodes/{id}` |
//! | `health` | `GET /health` |
//!
//! ## Capabilities
//!
//! Graphiti advertises: temporal_search, graph_search, keyword_search, and
//! entity_linking — but NOT semantic_search (left to pgvector/Cognee) or
//! ontology_grounding (left to Cognee).

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

/// Body for `POST /episodes` (ingest content into the temporal graph).
#[derive(Debug, Serialize)]
struct GraphitiWriteRequest<'a> {
    episodes: Vec<GraphitiEpisode<'a>>,
}

#[derive(Debug, Serialize)]
struct GraphitiEpisode<'a> {
    #[serde(rename = "type")]
    episode_type: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a serde_json::Value>,
}

/// Body for `POST /search`.
#[derive(Debug, Serialize)]
struct GraphitiSearchRequest<'a> {
    query: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    group_ids: Option<Vec<&'a str>>,
    num_results: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_time: Option<u64>,
    search_filter: GraphitiSearchFilter<'a>,
}

#[derive(Debug, Serialize)]
struct GraphitiSearchFilter<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    entity_type: Option<&'a str>,
}

/// A single result returned by Graphiti search.
#[derive(Debug, Deserialize)]
struct GraphitiSearchResult {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    fact: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    source_id: Option<String>,
    #[serde(default)]
    valid_at: Option<String>,
    #[serde(default)]
    invalid_at: Option<String>,
}

/// Health response from Graphiti.
#[derive(Debug, Deserialize)]
struct GraphitiHealthResponse {
    #[serde(default)]
    status: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────
// GraphitiEngine
// ─────────────────────────────────────────────────────────────────────

/// Graphiti temporal graph memory engine adapter.
///
/// Construct via [`GraphitiEngineBuilder`] or [`GraphitiEngine::new`].
pub struct GraphitiEngine {
    base_url: String,
    client: Client,
    group_id: String,
    timeout: Duration,
}

impl GraphitiEngine {
    /// Create a new Graphiti engine adapter.
    pub fn new(base_url: impl Into<String>, group_id: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: Client::new(),
            group_id: group_id.into(),
            timeout: Duration::from_secs(10),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

/// Builder for [`GraphitiEngine`] with configurable timeout and HTTP client.
pub struct GraphitiEngineBuilder {
    base_url: String,
    group_id: String,
    timeout: Duration,
    client: Option<Client>,
}

impl GraphitiEngineBuilder {
    pub fn new(base_url: impl Into<String>, group_id: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            group_id: group_id.into(),
            timeout: Duration::from_secs(10),
            client: None,
        }
    }

    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    pub fn client(mut self, c: Client) -> Self {
        self.client = Some(c);
        self
    }

    pub fn build(self) -> GraphitiEngine {
        GraphitiEngine {
            base_url: self.base_url,
            client: self.client.unwrap_or_default(),
            group_id: self.group_id,
            timeout: self.timeout,
        }
    }
}

#[async_trait]
impl MemoryEngine for GraphitiEngine {
    fn name(&self) -> &str {
        "graphiti"
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            semantic_search: false,
            keyword_search: true,
            graph_search: true,
            temporal_search: true,
            entity_linking: true,
            ontology_grounding: false,
        }
    }

    #[instrument(skip(self, query), fields(engine = "graphiti"))]
    async fn search(&self, query: &EngineQuery) -> Result<Vec<EngineResult>, EngineError> {
        let group_ids = if query.tenant_id.is_empty() {
            None
        } else {
            Some(vec![query.tenant_id.as_str()])
        };

        let body = GraphitiSearchRequest {
            query: &query.text,
            group_ids,
            num_results: query.top_k,
            max_time: Some(self.timeout.as_millis() as u64),
            search_filter: GraphitiSearchFilter {
                entity_type: None::<&str>,
            },
        };

        let resp = self
            .client
            .post(self.url("/search"))
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| EngineError::Internal(e.to_string()))?;

        match resp.status() {
            StatusCode::OK => {
                let results: Vec<GraphitiSearchResult> = resp
                    .json()
                    .await
                    .map_err(|e| EngineError::Internal(e.to_string()))?;

                Ok(results
                    .into_iter()
                    .map(|r| EngineResult {
                        engine_name: "graphiti".into(),
                        memory_id: Some(r.id.unwrap_or_default()),
                        content: r
                            .fact
                            .or(r.content)
                            .unwrap_or_default(),
                        score: r.score.unwrap_or(0.5),
                        metadata: serde_json::json!({
                            "source_id": r.source_id,
                            "valid_at": r.valid_at,
                            "invalid_at": r.invalid_at,
                        }),
                        evidence_refs: r.source_id.into_iter().collect(),
                    })
                    .collect())
            }
            s => {
                warn!(status = %s, "graphiti search failed");
                Err(EngineError::Internal(format!("graphiti search HTTP {s}")))
            }
        }
    }

    #[instrument(skip(self, item), fields(engine = "graphiti"))]
    async fn write(&self, item: &EngineWriteItem) -> Result<(), EngineError> {
        let body = GraphitiWriteRequest {
            episodes: vec![GraphitiEpisode {
                episode_type: "text",
                content: &item.content,
                source_id: Some(&item.memory_id),
                metadata: Some(&item.metadata),
            }],
        };

        let resp = self
            .client
            .post(self.url("/episodes"))
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| EngineError::Internal(e.to_string()))?;

        match resp.status() {
            StatusCode::OK | StatusCode::CREATED | StatusCode::ACCEPTED => {
                debug!("graphiti write accepted");
                Ok(())
            }
            StatusCode::CONFLICT => {
                debug!("graphiti episode already exists — treating as success");
                Ok(())
            }
            s => {
                warn!(status = %s, "graphiti write failed");
                Err(EngineError::Internal(format!(
                    "graphiti write HTTP {s}"
                )))
            }
        }
    }

    #[instrument(skip(self), fields(engine = "graphiti"))]
    async fn delete(&self, id: &str) -> Result<(), EngineError> {
        let resp = self
            .client
            .delete(self.url(&format!("/episodes/{id}")))
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| EngineError::Internal(e.to_string()))?;

        match resp.status() {
            StatusCode::OK | StatusCode::NO_CONTENT => Ok(()),
            StatusCode::NOT_FOUND => {
                debug!("graphiti episode not found — treating delete as success");
                Ok(())
            }
            s => Err(EngineError::Internal(format!(
                "graphiti delete HTTP {s}"
            ))),
        }
    }

    async fn health(&self) -> Result<EngineHealth, EngineError> {
        let resp = self
            .client
            .get(self.url("/health"))
            .timeout(self.timeout)
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => Ok(EngineHealth::Healthy),
            Ok(r) if r.status().is_server_error() => Ok(EngineHealth::Degraded),
            Ok(r) => {
                warn!(status = %r.status(), "graphiti health unexpected");
                Ok(EngineHealth::Degraded)
            }
            Err(e) => {
                warn!(error = %e, "graphiti unreachable");
                Ok(EngineHealth::Unreachable)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests (mockito-based)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use std::collections::HashMap;

    fn make_engine(server: &Server) -> GraphitiEngine {
        GraphitiEngine::new(server.url(), "tenant-1")
    }

    fn make_query(text: &str) -> EngineQuery {
        EngineQuery {
            text: text.into(),
            tenant_id: "tenant-1".into(),
            user_id: Some("user-1".into()),
            task_id: None,
            top_k: 5,
            filters: HashMap::new(),
        }
    }

    fn make_write_item(content: &str) -> EngineWriteItem {
        EngineWriteItem {
            memory_id: "mem-1".into(),
            tenant_id: "tenant-1".into(),
            content: content.into(),
            memory_type: "semantic".into(),
            scope: "task".into(),
            metadata: serde_json::json!({}),
            evidence_refs: vec![],
        }
    }

    #[tokio::test]
    async fn search_returns_results() {
        let mut server = Server::new_async().await;
        let body = r#"[
            {"id": "ep-1", "fact": "bearing replaced", "score": 0.92, "source_id": "src-a"},
            {"id": "ep-2", "fact": "lubrication changed", "score": 0.71, "source_id": "src-b"}
        ]"#;
        let m = server
            .mock("POST", "/search")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;

        let eng = make_engine(&server);
        let results = eng.search(&make_query("bearing history")).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].memory_id.as_deref(), Some("ep-1"));
        assert!(results[0].score > 0.9);
        m.assert_async().await;
    }

    #[tokio::test]
    async fn search_empty_results() {
        let mut server = Server::new_async().await;
        let m = server
            .mock("POST", "/search")
            .with_status(200)
            .with_body("[]")
            .create_async()
            .await;

        let eng = make_engine(&server);
        let results = eng.search(&make_query("nothing")).await.unwrap();
        assert!(results.is_empty());
        m.assert_async().await;
    }

    #[tokio::test]
    async fn search_server_error_returns_internal_error() {
        let mut server = Server::new_async().await;
        let m = server.mock("POST", "/search").with_status(500).create_async().await;

        let eng = make_engine(&server);
        let err = eng.search(&make_query("fail")).await.unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
        m.assert_async().await;
    }

    #[tokio::test]
    async fn write_creates_episode() {
        let mut server = Server::new_async().await;
        let m = server.mock("POST", "/episodes").with_status(202).create_async().await;

        let eng = make_engine(&server);
        eng.write(&make_write_item("bearing replaced")).await.unwrap();
        m.assert_async().await;
    }

    #[tokio::test]
    async fn write_conflict_is_ok() {
        let mut server = Server::new_async().await;
        let m = server.mock("POST", "/episodes").with_status(409).create_async().await;

        let eng = make_engine(&server);
        eng.write(&make_write_item("duplicate")).await.unwrap();
        m.assert_async().await;
    }

    #[tokio::test]
    async fn write_server_error_propagates() {
        let mut server = Server::new_async().await;
        let m = server.mock("POST", "/episodes").with_status(500).create_async().await;

        let eng = make_engine(&server);
        assert!(eng.write(&make_write_item("fail")).await.is_err());
        m.assert_async().await;
    }

    #[tokio::test]
    async fn delete_succeeds() {
        let mut server = Server::new_async().await;
        let m = server.mock("DELETE", "/episodes/ep-1").with_status(204).create_async().await;

        let eng = make_engine(&server);
        eng.delete("ep-1").await.unwrap();
        m.assert_async().await;
    }

    #[tokio::test]
    async fn delete_not_found_is_ok() {
        let mut server = Server::new_async().await;
        let m = server.mock("DELETE", "/episodes/ep-404").with_status(404).create_async().await;

        let eng = make_engine(&server);
        eng.delete("ep-404").await.unwrap();
        m.assert_async().await;
    }

    #[tokio::test]
    async fn health_healthy() {
        let mut server = Server::new_async().await;
        let m = server.mock("GET", "/health").with_status(200).with_body(r#"{"status":"ok"}"#).create_async().await;

        let eng = make_engine(&server);
        assert_eq!(eng.health().await.unwrap(), EngineHealth::Healthy);
        m.assert_async().await;
    }

    #[tokio::test]
    async fn health_degraded_on_500() {
        let mut server = Server::new_async().await;
        let m = server.mock("GET", "/health").with_status(500).create_async().await;

        let eng = make_engine(&server);
        assert_eq!(eng.health().await.unwrap(), EngineHealth::Degraded);
        m.assert_async().await;
    }

    #[tokio::test]
    async fn health_unreachable_on_connection_error() {
        let eng = GraphitiEngineBuilder::new("http://127.0.0.1:1", "grp-1")
            .timeout(Duration::from_millis(100))
            .build();
        assert_eq!(eng.health().await.unwrap(), EngineHealth::Unreachable);
    }

    #[tokio::test]
    async fn capabilities_advertises_temporal_and_graph() {
        let eng = GraphitiEngine::new("http://localhost:8080", "grp-1");
        let caps = eng.capabilities();
        assert!(caps.temporal_search);
        assert!(caps.graph_search);
        assert!(caps.keyword_search);
        assert!(!caps.semantic_search);
        assert!(!caps.ontology_grounding);
    }

    #[tokio::test]
    async fn builder_succeeds_with_all_fields() {
        let eng = GraphitiEngineBuilder::new("http://localhost:8080", "grp-1")
            .timeout(Duration::from_secs(5))
            .build();
        assert_eq!(eng.name(), "graphiti");
        assert_eq!(eng.group_id, "grp-1");
    }
}
